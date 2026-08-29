# 上游对齐评估 — cross-stage admission seam 能否迁到当前 main

> 调查日期 **2026-08-29**。基线：本地 `deb0dd6` (feat/cross-stage-admission，基于 2026-06-09 的
> origin/main) vs 上游 `origin/main` = **d5eac262**（2026-08-27，PR #1728）。
> 两者相距 **434 个 commit**。
>
> 本文回答三个问题：(1) 上游后来自己做了什么；(2) 这套东西还迁得过去吗；(3) 迁过去还有同样的提升吗。
> 结论先行：**代码迁得过去且比预期便宜；但 headline 数字（+350%/+618%）迁不过去——不是因为方法
> 失效，而是因为对照组消失了。价值重心必须从「防塌方」移到「自动定位工作点」+「deadline」。**

---

## 1. 上游后来做了什么：独立做出了同一件事的**静态版本**

### 1.1 Issue #1399 §3.1 → PR #1449（已合入）

**Issue #1399**（2026-08-08，"SGLang Omni Production Serving V0.1.2"，来自社区部署反馈）§3.1 标题
即 **"Collapse past the admission ceiling — highest priority in this section"**：

> TTFA 从 **c32 的 257ms → c64 的 9.6s**，一次并发翻倍带来 ~37× 劣化。
> *"I want to be direct that this is the finding in the whole report I had least expected.
> We tune and publish at c16, so the region past the admission ceiling was never characterised."*

这正是本仓库 `dev/testbed/RESULTS.md` §2 在 2026-06 就测到的 Qwen3-TTS throughput cliff
（closed-loop C=8 rps 3.04 → C=16 rps 0.31，延迟 2.6s→35s，且 `ok=144/144` 无任何错误）。
Issue 列的 TODO 与本仓库已完成的工作几乎逐条对应：复现 sustained overshoot、拆分 queue wait vs
engine time、**"define and document an explicit overload policy — load beyond capacity should be
rejected fast rather than admitted and starved"**、把 TTS benchmark sweep 延伸到 ceiling 之外。

**PR #1449**（2026-08-17，已合入）是它的落地：

- 新建 `sglang_omni/admission.py` —— **23 行，只有一个 `QueueFullError`**
- `--max_queued_requests`：有界等待队列，`OmniScheduler` 层 fast-reject
- `Coordinator.__init__(max_in_flight=...)`；`_submit_request` 里
  `len(self._requests) >= max_in_flight → raise QueueFullError()` → **HTTP 503**
- `PipelineConfig.generation_admission_defaults()` 每模型 classmethod；
  `mp_runner.resolve_coordinator_max_in_flight()` = `(running + queued) × num_replicas`
- benchmark 增加 `--sustained-overshoot` / `overshoot_duration_s`

**它改的文件与本地 `deb0dd6` 高度重合**：`pipeline/coordinator.py`、`pipeline/mp_runner.py`、
`config/schema.py`、`serve/openai_api.py`，连模块名都撞
（上游 `sglang_omni/admission.py` vs 本地 `sglang_omni/pipeline/admission.py`）。

### 1.2 其它相关上游工作

| PR | 日期 | 内容 |
|---|---|---|
| **#1014** merged | 07-20 | `[router]` overload protection：router 层全局 in-flight bound + fast 503 + `Retry-After`，`--max-connections` 即 admission bound。实测"无准入时开环 0.75–0.84× capacity 就 FD 耗尽 / 队列发散，goodput 仅为最优的 17–31%" |
| **#1049** merged | 08-06 | 多进程 CP/DP router，**共享内存 admission**（seqlock + generation fencing），跨 data plane 的全局 in-flight bound |
| **#931** merged | — | router in-flight 计数器在 mid-stream 上游失败时泄漏 —— 与本仓库 audit 修过的 submit 槽位泄漏同类 |
| **#1648** open | 08-21 | coordinator entry submit 失败时回滚 `_requests`/`_completion_futures`/`_stream_queues` —— 同样是准入记账泄漏 |
| **#1709** open | 08-25 | `feat(pd): make PD overload visible`。原文几乎复刻本仓库结论：*"admission rate is constant on this path，从健康请求到 41 秒请求都读 100%，所以 overload 表现为无界延迟而不是任何会说话的信号"* |
| **#1718** merged | 08-25 | PD handoff in-flight 上限，把"限制 lease"与"决定 batch 大小"两个耦合旋钮拆开 |

### 1.3 上游**没有**的（全树 grep 零命中）

- 任何 **adaptive / self-tuning** 准入：`adaptive_gate` / `hill.?climb` / `best_tput` 一个都没有
- 任何**可插拔 policy seam**：没有 `AdmissionPolicy` 协议；`max_in_flight` 是构造参数里的一个
  `int`，不是能换实现的 hook
- 任何 **deadline / SLO / goodput** 感知的准入：只有"满了就拒"，没有"预测会超时所以提前拒"
- 覆盖面：**只有 `models/qwen3_tts/config.py` 一个模型实现了 `generation_admission_defaults()`**。
  omni-speech / omni-understand / omni-mixed / Higgs / ASR 全部 `max_in_flight = None`，
  **coordinator 层对它们仍然是完全的 fire-and-forget**。

**一句话**：上游从"零准入"走到了"**静态、每模型手调的上限 + fast-reject**"，走完了本仓库那条路的
第一段；第二段（自调参）和第三段（deadline）仍然是它没有的。

---

## 2. 机械迁移：实测 rebase 结果

在隔离 worktree 里真跑了 `git rebase origin/main`（分支 `tmp/rebase-onto-main`，起点 `deb0dd6`）。
8 个文件里 **4 个冲突、共 17 个 hunk**：

| 文件 | 冲突 hunk | 性质 |
|---|---|---|
| `sglang_omni/pipeline/admission.py`（393 行） | **0** ✅ | **只 import stdlib**（asyncio / collections / logging / os / time / dataclasses / typing），零仓库耦合，原样落地 |
| `tests/unit_test/test_admission.py`（19 测试） | **0** ✅ | 只 import 自己的 gate，原样通过 |
| `sglang_omni/config/schema.py` | 0 ✅ | auto-merge 成功 |
| `tests/unit_test/pipeline/test_coordinator_admission.py`（8 测试） | 0 ✅ | 依赖的 `RecordingCoordinatorControlPlane` fixture 在 main 上仍在；只需对齐 Coordinator 新构造签名 |
| `sglang_omni/pipeline/coordinator.py` | 6 | 4 个纯文本相邻（构造参数列表 / docstring / import），2 个需要语义合并 |
| `sglang_omni/pipeline/mp_runner.py` | 3 | 全部是"两边各算一个东西再传进去"，加法式合并 |
| `sglang_omni/serve/openai_api.py` | 7 | 上游大改过，需重写而非重贴 |
| `sglang_omni/serve/realtime/session.py` | 1 | 上游重写了整个 terminal 生命周期（`claim_terminal` / `emit_terminals_safely` / `ResponseOutput`），必须重新实现 |

**核心资产零风险**：`AdaptiveGate` 那 393 行不碰仓库任何东西——7 条不变量、吞吐爬山控制器、
19 个单测原封不动。真正要动手的是 seam 的接线。

### 2.1 两个需要动脑子的 coordinator 冲突

都落在此前 audit 过的那条路上：

1. **流式 teardown**：上游新增了 abort + finally 清理；本地新增了 `_notify_admission_complete`。
   合并时 **notify 必须落在上游那个 finally 里**，否则就是三轮 audit 修过的 submit 槽位泄漏再来一次。
2. **`abort()`**：上游新增了 `info = self._requests.get(request_id); if info is None: return False`
   早退守卫；本地在同一位置释放槽位。合并后要确认早退分支不会跳过 `on_complete`
   （若请求已不在 `_requests` 里，说明已完成、槽位已释放，应当安全）——**这条必须写测试钉住**。

### 2.2 serve 层反而变便宜了

上游 #1449 建了中心化错误映射：

- `serve/speech_errors.py::speech_generation_error()`（queue-full → 503）+ `_speech_generation_failure_response()`
  → 本地那 116 行手写 try/except 大部分塌缩成往这个 map 里加一条 `AdmissionRejected → 429`
- `_speech_audio_response()` **本来就从第一个 audio chunk 派生 header**
  → 为了"在 200 header 发出前变 429"写的那套 peek 机制，在 speech 路径上已是上游既有行为，**可删除复用**

---

## 3. 三个必须重新决策的语义点（真正的难点）

### 3.1 双重门控的顺序

上游现在在 `_submit_request` 顶部检查 `len(self._requests) >= max_in_flight`；本地 `on_submit` 也在
同一位置。若 gate 先排队，请求压在 `_requests` **外面**，`len(_requests) ≤ L`，**只要
`L ≤ max_in_flight`，上游那个静态检查永远不触发**——等于悄悄接管了它。必须显式决定谁在外层。

### 3.2 ⚠️ 无界队列回归 —— 最需要警惕的一条

上游做 #1449 的全部意义就是"过载时快速拒绝，而不是接收后饿死"。而本地 `AdaptiveGate` 在
`slo_s=None` 时**只排队、永不丢，队列无界**。直接迁过去按默认参数开启，等于把 #1399 §3.1 抱怨的
那个失败模式（无界排队 → 延迟爆炸）用一个更聪明的方式重新引入。

**迁移方案必须给 gate 的等待队列一个上限，或在 main 上把 `slo_s` 从可选改为事实必填。**
这一条要直接写进 PR 描述，否则 reviewer 必问。

### 3.3 429 vs 503

本地 `AdmissionRejected` → **429**；上游 `QueueFullError` → **503 + Retry-After**（router #1014 同）。
两者语义确实不同（"你太快了" vs "我满了"），但要么统一到 503，要么明确文档化区分——
**不能默默引入第二个状态码**。

---

## 4. 提升还会不会一样？分三块

### (a) 防塌方的 headline 数字：**没了**

`+350% achieved / −84% p99` 的分母是一个**已经塌方**的对照组（achieved → 0.008 / 0.000 rps，
p99 → 2.5–3.5 分钟）。在当前 main 上那个对照组**在构造上不可能再出现**：coordinator 把 in-flight
卡在 32，超了直接 503，无界队列发散这条路已被堵死。这组数字在新 baseline 上必然大幅缩水甚至归零。

且上游同期落了一批 TTS 性能工作（#1286/#1336 声称 TTFA 平在 ~180ms 到 75s 音频、c8 吞吐 +45%）、
prefill coalescing 泛化 (#1073)、breakable prefill CUDA graph (#1581)。#1399 自己就猜测那个 cliff
"看起来像一个狭窄的排队/准入缺陷而不是根本吞吐上限"，并点名 `prefill_coalesce_wait_ms` 的 hold-off
累积为嫌疑。**若 cliff 本身已被修掉，`AdaptiveGate` 在 TTS 上就没有 cliff 可发现。**
—— 这一点无法从代码判断，**必须重测**。

### (b) 自动定位工作点：**大概率还在，且有个很有说服力的数字**

上游给 Qwen3-TTS 定的是 `max_running_requests=16` + `max_queued_requests=16`
→ coordinator `max_in_flight = 32`。

而本仓库实测 TTS **拐点 L≈8**，**塌方点恰好就是 C=16**（RESULTS.md §2）。

> **上游的 running 上限正好压在实测的塌方点上，coordinator 上限是拐点的 4 倍。**
> 503 只保证了队列不发散，没有保证服务器跑在高效区。

上游 cookbook（`docs/cookbook/qwen3_tts.md` §Overload / admission policy）自己写着
"Capacity is about `running + queued`"，还给了个手动的 "ceiling-32 experiment" 配方
—— **这个数字今天仍然是人肉试出来的**，正是 AdaptiveGate 要替掉的东西。

叠加 §1.3 的覆盖面事实：另外 5 个配置在 coordinator 层仍是完全无准入，
**它们面对的还是原来那个空 baseline**。

### (c) Deadline shedding：**结构性仍然成立，可纯分析论证**

与 #1449 完全正交，且能算：上游把 in-flight 卡在 32，TTS 实测 capacity ≈ 2.5 rps，
按 Little's law 一个刚被放行的请求最坏等待 ≈ **32 / 2.5 ≈ 12.8 秒**。对 4 秒 SLO 而言，
"有界队列"与"没队列"没有区别——它把**无界**变成了**有界但仍远超 SLO**。

本仓库的 shedding 结果（p99 钉在 SLO 上：TTS ~4.3s、Higgs 4.3–5.3s，而无策略/纯排队为 28–74s）
针对的正是这个残留缺口。上游至今**没有任何 deadline / SLO / goodput 维度**。
所以这块价值不但没被 #1449 吃掉，反而因为 #1449 把问题收敛成一个干净的"有界但超时"而**更好讲了**。

---

## 5. 必须重测的三件事（按性价比排序）

1. **在 5 个未设 `generation_admission_defaults` 的模型上重跑 no-regression**
   —— baseline 未变，`L→max` 的机制论证应原样成立。成本最低、最先能拿到结论。
2. **在当前 main 上重跑 TTS 开环 baseline** —— cliff 还在不在、拐点移到哪。
   **这一条决定 (b) 的成败**，其余都是次要。
3. **新的 A/B 对照组必须换成"上游默认（running 16 + queued 16, 503）"**，
   而不是 "no policy"。旧的 no-policy 臂在 main 上已不是有意义的对照。

好消息：`dev/testbed/` 只驱动公开 serving API（`/v1/chat/completions`、`/v1/audio/speech`、
`/v1/audio/transcriptions`），不 patch engine，**harness 本身可直接复用**；
要改的是 `scenarios.yaml` 里的 A 臂定义和那份已经 stale 的 `baselines/`。

---

## 6. 本分支不包含什么（以及如何重建）

`dev/` 的完整工作目录在本地是 **3.7 GB**，其中 **416 KB / 56 个文件**是源码、文档、结论 JSON 与
baseline，即本分支的全部内容。其余为机器生成、可重建的产物，且**含 GitHub 硬性拒绝的
>100 MB 单文件**，故由 `dev/.gitignore` 排除：

| 排除项 | 体积 | 重建方式 |
|---|---|---|
| `pipeline-coscheduling/results/*.sqlite`、`*.nsys-rep` | 1.6 GB（单文件最大 543 MB / 211 MB） | `campaign_*.sh` + `_nsys*.sh` |
| `pipeline-coscheduling/wheels/*.whl` | 559 MB（单文件 558 MB） | `build_sgl_kernel.sh`（B300 sm_103 重建） |
| `events_*/` 原始事件流（111 个 jsonl） | 399 MB | 置 `SGLANG_OMNI_EVENT_DIR` 后重跑 campaign |
| `.ccache/`、`triton_cache/`、`buildtmp/`、`__pycache__/` | ~700 MB | 编译副产物，无需保留 |
| `testbed/results/`、`logs/` | ~6 MB | `testbed.py` 每次运行重新产出 |

结论所依赖的派生数据（`pipeline-coscheduling/findings/*.json`、`testbed/baselines/*.json`）
**已完整保留**在本分支。
