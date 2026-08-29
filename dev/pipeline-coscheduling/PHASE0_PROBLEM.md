# Phase 0 — 浪费在哪、有多少、是否真实

> 跨阶段流水线协同调度（cross-stage co-scheduling）研究的第 0 章。
> **本章不解决问题，只回答：bubble/bottleneck 是否真实存在、发生在哪、绝对浪费多少时间、占比多少、在哪些真实场景下出现。**
> 在这章给出令人信服的数据之前，不写任何调度算法。

---

## 0. 为什么先做这一章

设计一个调度器之前必须先证明被调度的浪费是真的、可观测、可归因、且足够大。否则会陷入"优化了一个不存在的瓶颈"。本章的产出物是一张 **motivation 图 + 一组百分比**，能一句话说服人："在场景 X 下，价值 \$30k 的 thinker GPU 有 N% 的 wall-time 在空转等一个 <2% 利用率的 talker，每请求净浪费 M ms（占端到端延迟 K%）。"

目标读者用三个问题质问这一章，我们逐一回答：

- **Q1（存在 & 定位）** GPU 时间到底去哪了？每个 stage 是在忙、在饿（等上游）、还是在被堵（等下游）？
- **Q2（量级）** 每类浪费的**绝对时间**（ms/请求）和**占比**（占 wall-time %、占端到端延迟 %）是多少？
- **Q3（真实性 / regime）** 随并发和负载形状怎么变？continuous batching 会不会把它填满？哪些真实部署场景最严重？
- **Q4（归因）** 浪费是**机制造成的**（`credits=2`、FIFO、零准入——可被调度修掉）还是**结构性的**（stage 吞吐失衡——需要 placement/co-location）？这决定了是 P3 还是 P1 的问题。

---

## 1. 把"浪费"定义清楚：GPU 时间的四分解

每个 stage 独占（或共享）一块 GPU。在 wall-clock 窗口 `T` 内，该 stage 的 GPU 处于以下之一：

```
T  =  BUSY  +  STARVED  +  BLOCKED  +  IDLE
```

| 状态 | 定义 | 判据（可观测） |
|---|---|---|
| **BUSY** | 在为某请求做有用 forward | 处于一次 step 的 forward begin→end 之间 |
| **STARVED** | 想干活但没输入：上游还没产出 / 流水线在填充 | 非 BUSY，且 `inbox` 为空 **但** 存在已准入、尚未到达本 stage 的在途请求 |
| **BLOCKED** | 干完了推不出去：被下游背压（信用耗尽） | 非 BUSY，且本 stage 输出在等 relay 信用 / 下游 `acquire` |
| **IDLE** | 真没活：系统整体也没有待处理的端到端工作 | 非 BUSY，inbox 空，且管线里无在途请求 |

**关键定义：**

```
bubble  =  STARVED + BLOCKED          # 「明明全局有活、这块 GPU 却空转」的时间
```

`IDLE` 不算浪费（低负载下空闲是合理的）。`STARVED` ↔ §前文 B1 填充/速率失配；`BLOCKED` ↔ B2 背压（`credits=2`）。区分 STARVED vs BLOCKED 就是区分"等上游"还是"卡下游"，是归因的核心。

> 这套四分解直接对应 aginfer DESIGN 的"先讲物理事实再量化"：这里的物理事实是**GPU-second 是不可回收的稀缺资源，bubble 就是被烧掉的 GPU-second**。

---

## 2. 现有 instrumentation：什么免费、什么要补

仓库已有 `sglang_omni/profiler/`：每进程把事件写进 `events_<stage>_<pid>.jsonl`，`views.py` 按 `request_id` 合并出 timeline / stage breakdown / hop breakdown。

### 2.1 已经免费可得（现有事件，`event_recorder.py` / `views.py`）

| 已有事件 | 能算出的量 |
|---|---|
| `request_admission`（coordinator.py:216） | 请求进入系统的 t0 |
| `preprocess_start/end`, `encoder_start/end` | 前处理 / 编码器耗时 |
| `scheduler_queue_enter`（omni_scheduler.py:514） | 进入某 AR stage 等待队列的时刻 |
| `scheduler_prefill_start`（:729） | prefill 首次被调度（→ 排队时延 = 这个减 queue_enter） |
| `scheduler_first_emit`（:656） | thinker 出首 token（TTFT） |
| `stage_first_stream_chunk_sent` / `stage_stream_chunk_sent`（runtime.py） | thinker 流出 hidden state 的每个 chunk 的发送时刻（带 `chunk_id`） |
| `stage_stream_chunk_received` | talker 收到每个 chunk 的时刻 → **hop_breakdown 已能算 thinker→talker 每 chunk 传输时延** |
| `code2wav_first_audio` | TTFA（首音延迟） |
| `stage_complete` / `terminal_response` | 端到端结束 |

→ **端到端 timeline、各 stage 区间耗时、跨 stage hop 时延：现在就能出。** `views.stage_breakdown` / `hop_breakdown` 已给 count/total/p50/p95/max。

### 2.2 必须补的埋点（少量，全是几行）

要算 §1 的四分解和归因，缺以下 5 处。每处都是在已有 recorder 上 `emit()` 一行：

| 缺口 | 补在哪 | 拿到什么 | 用于 |
|---|---|---|---|
| **G1 每 step forward begin/end** | `ModelRunner.execute` 首尾 / `OmniScheduler._run_batch`（omni_scheduler.py:609） | 每个 stage 的 **BUSY 区间**（精确到 step），含 `batch_size`、`is_prefill` | M1/M2 BUSY；M-batch |
| **G2 背压 stall 计时** | `send_stream_chunk` 里 `acquire_async()` 前后（relay credit 等待），或 `relay_io.py:417` 的 `put_async` await | thinker **BLOCKED** 的绝对时长，按 (rid, chunk) 归集 | M3，B2 铁证 |
| **G3 队列深度采样** | 每个 `_event_loop_*` 循环顶（omni_scheduler.py:902） | `len(waiting_queue)`、`len(running_batch.reqs)` 时间序列 | STARVED vs IDLE 判定；M-queue |
| **G4 缓冲占用** | talker 的 `stream_chunks` deque 长度（omni_scheduler.py:1151）+ relay 在用信用数 | 背压缓冲的水位 | 佐证 BLOCKED |
| **G5 GPU 利用率采样** | 旁路 NVML / `nvidia-smi dmon` 或 nsys，按 wall-clock 对齐 | 每 GPU 真实 SM 占用率时间序列 | 校验 BUSY 推断；共置场景争抢 |

> G1+G3 一旦有了，四分解就是纯离线计算：把每个 stage 的 wall 窗口减去 BUSY 区间得到所有 idle 段，再用 G3 的队列深度 + 在途请求集合把每段 idle 标成 STARVED / BLOCKED / IDLE（BLOCKED 段与 G2 的 stall 区间求交确认）。

---

## 3. 要给人看的 metric（直观优先）

### M1. Stage-GPU 占用泳道图（**hero figure**）
横轴 wall-clock，每个 stage/GPU 一条泳道，染色 BUSY(绿)/STARVED(黄)/BLOCKED(红)/IDLE(灰)。一眼看出 bubble 在哪、谁堵谁。单请求和并发两版都画。
> 这是回答 Q1 的"直观 metric"。红色（BLOCKED）出现在 thinker 泳道、同时 talker 泳道在忙 = `credits=2` 把贵 GPU 堵停的视觉铁证。

### M2. Bubble 分解条（每 stage 一根）
`BUSY% / STARVED% / BLOCKED% / IDLE%` 堆叠条。**头条百分比**。
> 例如目标陈述："talker GPU BUSY 仅 4%，thinker BLOCKED 占 22%。"

### M3. 背压 stall（绝对 ms + 占比）
thinker 因等 talker 信用而空转的累计 ms/请求；占 thinker active 时间 %；占端到端延迟 %。
> B2 的绝对代价，最有冲击力的单一数字。

### M4. Stage 吞吐饱和曲线（回答"能不能被并发填满"）
每个 stage 单独压测：req/s（或 tokens/s）随并发上升直到饱和。得到每 stage 的 **max sustainable throughput** 和**饱和并发点**。两 stage 的比值 = 结构性失衡比。
> 若 thinker 在并发 8 饱和、talker 要并发 50 才饱和，则在让 thinker 满载的并发下 talker 永远大半空转——**bubble 不随负载消失，是结构性的（→Q4 指向 P1）**；反之若 stall 随并发降到 0，则是机制性的（→ P3 可修）。

### M5. 端到端关键路径分解
单请求端到端延迟 = Σ(各 stage compute) + Σ(hop 传输) + Σ(排队/等待)。用已有 `stage_breakdown`+`hop_breakdown`+补的等待段。
> 给出"延迟里有百分之几是纯等待/bubble，而非计算"。

### M6. 流式实时性（语音场景专用）
TTFA（已有 `code2wav_first_audio`）+ 帧间间隔分布 + **音频 buffer 下溢率**（帧到达晚于实时播放节奏的比例）。
> 回答"bubble 是否伤害真实语音体验"，不只是吞吐。

---

## 4. 场景矩阵（回答 Q3：是否真实）

bubble 的大小**强依赖负载 regime**——这本身就是要测的核心，而不是预设。

| 场景 | 配置 | 为什么真实 | 预期主导 bubble |
|---|---|---|---|
| **S1 单语音会话** | concurrency=1，短 prompt，~5–10s 音频 | voice agent / `/v1/realtime` 的核心用例，**并发就是 1** | B1 填充 + B2 速率失配；无其他请求可填 |
| **S2 批量 TTS** | concurrency ∈ {1,4,8,16,32,64,128} | 离线 TTS / 高吞吐服务 | 测 batching 能否填满 bubble（regime 转折点） |
| **S3 混合负载** | text-only + speech 请求交织 | 真实 omni 部署同时服务两类 | B4 跨 stage 队头阻塞（共享 thinker） |
| **S4 长多模态 prefill** | 视频/长音频输入 | 多模态理解类请求 | B1：talker+vocoder 在长 prefill 期间整段 STARVED |
| **S5 共置部署** | thinker_gpu==talker_gpu（config.py:337） | 官方为省卡提供的配置 | B5 intra-GPU 争抢 + jitter；对照 GIL-sleep hack |

**两套硬件对照**：分卡（thinker GPU0 / talker GPU1）vs 共置（同卡）。分卡场景里 talker 的低占用率直接就是整卡 bubble（RFC 自述 talker 在 H200 上 <2% → 那是一整张卡 ~98% 的 bubble，这本身就是 motivation 的起点）。

**并发扫描是 Q3 的关键实验**：对每个场景把 concurrency 从 1 扫到饱和，画 M2（bubble%）随并发的曲线。结论无非三种，每种都有意义：
1. bubble 随并发快速降到 ~0 → 只有低并发（=单会话语音，但那恰恰是 realtime 的核心场景）才有问题 → P3 价值窄但仍真实；
2. bubble 降一些但因结构失衡（M4）卡在某地板不再降 → P3/P1 价值大、持续存在；
3. bubble 几乎不随并发降（强结构失衡）→ 主要是 placement 问题（P1）。

---

## 5. 归因实验（回答 Q4：机制 vs 结构）

在不写新调度器的前提下，用**最小 ablation** 把浪费归因，这决定整个研究是 P3 还是 P1：

- **A1 信用窗口扫描**：把 `credits=2`（runtime_config.py:145）改成 {2, 4, 8, 32, ∞}，看 M3（thinker BLOCKED）和端到端吞吐。
  - 若 stall 随 credit 增大而消失、吞吐上升 → **机制性背压**，调度可修（强支持 P3）。
  - 若 credit 拉满后 talker 仍跟不上、bubble 转移到别处 → 结构失衡（指向 P1）。
- **A2 准入并发上限**：在 coordinator 给一个固定 in-flight 上限（粗准入），看队头阻塞（M5 等待段）和尾延迟是否改善 → 量化 B4。
- **A3 batch 上限错配**：调 thinker `max_running_requests` 看 talker 是否被瞬时灌爆（M2 talker BLOCKED 升高）→ 量化 B3。

> A1–A3 都是改配置/几行，不是研究算法。它们的作用是**把每类 bubble 的可修性钉死**，为后续设计选对战场。

---

## 6. Go / No-Go 判据

Phase 0 结束时，按下面任一条成立则问题成立、值得做调度：

- 在**任一真实场景**下，某块贵 GPU 的 `bubble = STARVED+BLOCKED > 25%` wall-time；**或**
- thinker `BLOCKED`（等 talker 信用）`> 10%` 端到端延迟；**或**
- 单语音会话（S1）TTFA / 帧间 jitter 因填充/争抢 bubble 显著退化（M6 下溢率 > 阈值）；**且**
- A1 表明至少部分 bubble 是机制性的（可被调度而非仅靠加卡解决）。

若全部场景在并发≥4 时 bubble 被 batching 填到 <10% 且无机制性 stall → P3 退化为"仅低并发语音有效"，应缩小 scope 或转向 P1。**这个否定结论本身也是有价值的产出**，避免在不存在的瓶颈上做半年。

---

## 7. 本章交付物

1. 一个**离线分析脚本**：吃 `events_*.jsonl`（+ NVML 采样），输出 M1–M6 和四分解（复用 `views.py`，扩展 G1–G5）。
2. **5 处埋点 PR**（G1–G5），全部走现有 `emit()`，profiler 关闭时零开销。
3. **场景矩阵跑批结果**：每个场景 × 并发点的 M2/M3/M5 表 + S1/S5 的 M1 泳道图。
4. 一页 **motivation 结论**：绝对 ms + 百分比 + regime 曲线 + Q4 归因，一句话能讲清楚浪费在哪、多大、对谁真实。

---

## 附：与现有代码的精确锚点

- 四分解 BUSY 来源：`OmniScheduler._run_batch`（omni_scheduler.py:609）/ `ModelRunner.execute`。
- STARVED/IDLE 判定数据：`waiting_queue`（:519）、`running_batch.reqs`、coordinator 在途请求集（coordinator.py `_requests`）。
- BLOCKED / 背压根因：`CreditAllocator`（relay/base.py:121）credits 默认 2（runtime_config.py:145）；`send_stream_chunk`（relay_io.py:370）的 `put_async` await（:417）。
- 共置争抢佐证：GIL-starvation 注释 + `time.sleep(0.001)`（omni_scheduler.py:895–919）。
- 现有可复用 view：`views.stage_breakdown` / `hop_breakdown` / `reconstruct_timelines`（profiler/views.py）。
