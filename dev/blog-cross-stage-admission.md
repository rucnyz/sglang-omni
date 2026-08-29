# 给 sglang-omni 加一层 cross-stage 准入调度：AdaptiveGate + Deadline Shedding

> **TL;DR**
>
> 给 sglang-omni 的 pipeline coordinator 补上了缺失的 **cross-stage 准入决策 seam**：一个 `AdmissionPolicy` 协议 + 一个会自调参的 **AdaptiveGate**（in-flight 门控 + 可选的 deadline shedding）。默认 `NoOp` 与原来的 fire-and-forget 字节级等价，开了才生效。一套参数不调每模型：在会"塌方"的模型（TTS）上自动找到拐点把吞吐救回来，在会 batch 的平台型模型（omni / ASR / Higgs）上门控自动失活、构造性地不回归。

## 背景：框架把 per-stage policy 外部化了，cross-stage 却是 fire-and-forget

sglang-omni 的多模态服务被拆成一条 stage DAG（thinker → talker → code2wav 等），框架把**单个 stage 内**的策略都做成了 import-string hook：`route_fn` / `wait_for_fn` / `merge_fn` / `placement_policy`。但**跨 stage 的全局决策**没有任何接口——coordinator 的 `_submit_request` 就是 fire-and-forget：请求一到就立刻塞进 entry stage，没有准入控制、没有优先级、没有 deadline、没有负载感知。

这在过载时会出问题。最典型的是 Qwen3-TTS：并发超过 ~8 之后吞吐会**塌方**（throughput cliff），开环过载下 achieved rps 掉到 0.6–1.0、ttfa p99 飙到几十秒到几分钟、goodput 接近 0。而框架现有的 per-stage hook 表达不了"在入口处限制总在飞请求数"这种全局策略。

所以这次的贡献就是：**把这层缺失的 cross-stage policy seam 补上**，并且用同一套 import-string 风格接进去。

## 架构设计

只在 coordinator 上开一个口子，一个 server 一个 policy 实例：

```
request ──▶ Coordinator._submit_request
                 │
                 ▼
        AdmissionPolicy.on_submit(ctx)      ← 唯一的 cross-stage 决策点
           ├─ 放行：in-flight < L，进入 entry stage
           ├─ 排队：达到 L 则 FIFO 等待（excess wait）
           └─ shed：预测必然超时 → AdmissionRejected → HTTP 429
                 │
           ┌─────┴──── pipeline 各 stage（thinker → talker → code2wav …）
                 │
                 ▼
        on_complete(request_id)             ← 释放 in-flight 名额
        （成功 / 失败 / abort / 准入后失败 都会调，且幂等）
```

- **协议**：`async on_submit(ctx)`（可 await 来 gate / delay / reorder，return 即放行，raise 即拒绝）+ `on_complete(request_id)`（释放 `on_submit` 占的账，幂等）。`AdmissionContext` 是一个只读快照 `{request_id, request, entry_stage, inflight}`。
- **接入方式**：`PipelineConfig.admission_policy`（dotted path）或环境变量 `SGLANG_OMNI_ADMISSION_POLICY`。
- **默认 `NoOpAdmission`**：立即放行，与原 fire-and-forget **字节级等价**——也就是说"把 seam 接进去"这件事本身不改变任何行为，配了真 policy 才生效。这条性质有单测兜底（`NoOp` 与"无 policy"在 submit/complete/abort 上输出完全一致）。

一个关键的设计取舍：**不搞 policy 动物园**。整个系统只有一个会演进的 policy，静态限流 `FifoGate(N)` 只是它 `adapt=False` 的退化特例：

```
NoOp  ⊂  FifoGate(N)  ⊂  AdaptiveGate  ⊂  AdaptiveGate(slo_s=...)
```

没有"选哪个 policy / fallback 到哪个"的问题。

## 关键实现

### AdaptiveGate：用吞吐爬山自动找拐点

门控维持一个在飞上限 `L`（FIFO，超了排队）。`L` 不是写死的，而是用**吞吐**（completions / 时间窗，在 seam 上免费可得的信号）做爬山：往上探 `L`，只有当吞吐相对低 `L` 处的最优值**塌方**时才退回最优工作点。

为什么是吞吐而不是延迟？因为延迟爬山会找到**延迟拐点**，而平台型模型（batch 高效）的延迟拐点在吞吐拐点之下，会把它误杀。吞吐直接对准 capacity，所以能泛化到形状完全不同的负载曲线：

- **塌方型**（TTS，过 ~8 就掉）：往上探越过拐点吞吐下降 → 退回拐点并把这个点记成 ceiling，不再越界。
- **平台型**（omni batch 到 ~16）：往上探吞吐不掉 → `L` 自由升到 max ≈ 不门控，**构造性地不可能回归**。

一套参数不调每模型。控制器的健壮性靠 7 条不变量（每条都是被实际服务器上的 `[ADMGATE]` trace 暴露出的某个具体翻车 case 逼出来的）：

1. **fail-open**：从 floor 起步、向上探、只在有证据时下调 → 永不掐没拥塞的系统。
2. **固定 wall-clock 吞吐窗**：`raw = n/Δt`，`Δt ≥ window_s`。batch serving 会在一个 decode step 里成簇完成（clump），基于计数的窗会以 `Δt≈ms` 关闭、`raw` 炸到几千、毒化 best_tput。固定时间窗按构造把 `raw` 限住。
3. **outlier-cap**：单窗速率超过平滑估计 2× 当作测量伪迹丢弃。
4. **持久 best（慢衰减 0.999）**：长跑也记得住好工作点，不会漂走。
5. **ceiling memory**：绝不往已知会塌方的并发上探 → 不在 cliff 上来回横跳。
6. **in-flight 饱和门**：只有峰值在飞 ≥ `sat_frac·L` 时才把吞吐下降当塌方——把真 cliff（贴着上限跑、吞吐掉了）和 demand drop（请求变少、门有余量）区分开。
7. **小 up_factor（1.15）**：辨别 cliff vs plateau 必须越过拐点一次，小步长把这个一次性 overshoot 的**深度**限住（8→9.2→10.6，而不是 8→10→13→17）。

### Deadline Shedding：过载时主动丢"注定超时"的请求

门控只排队的话能救回**吞吐**，但救不了 **goodput**——rate ≫ capacity 时排队会无界增长，请求还是会错过 SLO。所以 `AdaptiveGate(slo_s=...)` 加一条：`on_submit` 时按 Little's law 预测新请求的 time-in-system ≈ `(in_system + 1) / drain_rate`，超过 deadline 就直接 `raise AdmissionRejected` → 让出 capacity 给还能达标的请求。

一个容易踩的坑：drain_rate 必须用**服务 capacity（best_tput）**，而不是当前吞吐——低负载时当前速率是 arrival-limited 的，用它会高估等待、过度 shed。**fail-open**：没设 `slo_s` 或还没有 capacity 估计就永不 shed（静态 `FifoGate` 也就永不 shed）。

`AdmissionRejected` 在 serve 层被映射成 **HTTP 429**，覆盖所有入口：chat / speech / transcription（streaming 路径先 peek 第一个 chunk，让 shed 在 200 header 发出之前变成干净的 429），以及 realtime WebSocket（按 pass 发一个 failed `response.done` 关闭生命周期，而不是把客户端挂死）。

## 效果

**TTS（塌方型）——把塌方变成可持续 capacity：**

| offered rate | achieved（无策略 → 门控） | total p99（无策略 → 门控） |
|---|---|---|
| R=3 | 0.48 → **2.18 rps**（+350%） | 70s → **11s**（−84%） |
| R=4 / R=6 | 无策略基本死掉（~0） | 门控稳在 ~2.5/s（= capacity） |

门控自动收敛到 `L ≈ 8`，且在 capacity 以下完全透明（R=2 achieved 1.52 vs 1.53）。

**omni / omni-understand / ASR / Higgs（平台型）：** `[ADMGATE]` trace 显示 `L` 全程顶在 max、门控全程失活 → 在相同 offered 下 achieved 持平，**按构造不回归**。

**Shedding：** 过载下 goodput 分层 `shed > queue-only > no-policy`，并且把 total-latency p99 钉在 SLO 上（TTS ~4.3s@R4、Higgs ~4.3–5.3s@R5，而无策略/纯排队会涨到 28–74s）。capacity 以下透明（shed=0）。

整个评测走的是**开环 paired A/B**（同一块卡上 no-policy / queue / shed 背靠背各跑一遍，per-(rate,rep) 固定种子），并经过三轮对抗式 audit（找 + 修了 submit 槽位泄漏、abort 与门控排队的竞态、realtime shed 挂死等问题），36 个单测覆盖门控时序、控制器收敛、shedding 与 coordinator 集成。

## 走过的弯路（诚实记录）

不是所有方向都成。两个负结果也写下来：

- **ttfa-SLO shedding（放弃）**：想按"首 token 时间"而不是总完成时间来 shed。机制能跑，但有**根本性限制**：门控只能看到流过 coordinator 的信号，对 omni-speech 那是首个**文本** token（thinker），而 SLO 是首个**音频**——音频由下游解耦的 talker→code2wav 产出，门控的 first-text 信号管不到下游模态。要做得有下游 first-token 信号。已回退。
- **per-class in-flight 限额（未做）**：本想给 speech / text 混合负载按类别切 in-flight。但结构性地用不上：唯一会卡门控的模型（TTS cliff）是单一类别、没东西可切；唯一混合类别的模型（omni）门控从不卡（平台型，`L→max`）。两边对不上 → 在现有 workload 上注定 inert，没有能展示价值的场景。已记录分析、未实现。

## 使用方法

### 开启自适应门控（最常用，纯排队、不 shed）

config 里给一个 dotted path 即可：

```yaml
pipeline:
  admission_policy: sglang_omni.pipeline.admission.AdaptiveGate
```

默认参数（`min_limit=8, max_limit=64, window_s=3.0, slo_s=None`）已经是通用值，不需要每模型调。

### 开启 deadline shedding（需要传 `slo_s`）

> **Note:** 当前 config 字段只接受 dotted path、按零参数实例化；要传 `slo_s` / `max_inflight` 这类构造参数，走环境变量（`SGLANG_OMNI_ADMISSION_ARGS` 是一段 JSON）。把构造参数也搬进 config 在 roadmap 上。

```bash
export SGLANG_OMNI_ADMISSION_POLICY=sglang_omni.pipeline.admission.AdaptiveGate
export SGLANG_OMNI_ADMISSION_ARGS='{"slo_s": 4.0}'    # 总完成时间 deadline，单位秒
```

### 静态限流（baseline / 退化特例）

```bash
export SGLANG_OMNI_ADMISSION_POLICY=sglang_omni.pipeline.admission.FifoGate
export SGLANG_OMNI_ADMISSION_ARGS='{"max_inflight": 8}'   # ≡ AdaptiveGate(limit=8, adapt=False)
```

### 观测门控行为

```bash
export SGLANG_OMNI_ADMGATE_LOG=1    # 打开后每个控制步打一行 [ADMGATE] step/limit/ceiling/tput…
```

`L` 全程贴着 max 就是门控失活（平台型，符合预期）；稳定收敛到某个拐点就是发现了 cliff。

## Future Roadmap

- **Config 直接传参**：把构造参数（`slo_s` 等）从"只能走 env"提升为 config 一等公民（对齐仓库里 `StageConfig.factory_args` 的模式）。
- **优先级 / class-aware 准入**：reorder hook + cross-stage HoL 优先级。per-class in-flight 留作其中一块（等出现"会卡门控的混合负载"场景再激活）。
- **下游模态 SLO**：要真正 bound omni-speech 的首音频时间，需要一个下游 first-token 信号回灌到门控——ttfa 那条路绕不过这个前提。
- **动态 placement / fleet-level scheduling**：另外两个 seam（placement = 静态 config、relay backpressure = 写死 credits=2）还是静态的。真正的结构性 GPU 空洞在 fleet 级别（disagg 会让一块卡 ~70% 闲置），动态 placement 是能真正回收 GPU 的方向，但 scope 更大（跨节点）。

## 相关笔记

- [[sglang-omni-scheduling]]：本仓库里 cross-stage 调度问题的全景与可做方向（admission / deadline / placement）。
- [[sm-occupancy-scheduling-interface]]：coloc server 满载下 SM 仅 ~30%、那 70% 空闲算力的 spatial co-exec 接口——与本篇是同一棵"框架把 per-stage 外部化、cross-stage 留空"的树。
- [[admission-testbed]]：开环 paired A/B testbed（多模型回归 + 性能），本篇所有数字的来源。
- [[checkpoint-engine]] / [[online_rl]]：throughput-driven 自调参 + 过载保护在 RL rollout 侧的对应物（权重热更新 / disaggregated rollout）。
