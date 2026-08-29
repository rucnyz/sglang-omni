# Phase 0 results (corrected after adversarial review)

Measured on **8× B300**, Qwen3-Omni-30B-A3B-Instruct, text→speech & text-only,
streaming, warm steady state. Instrumentation G1 (per-stage forward intervals)
via `analyze_bubbles.py`; **GPU utilization cross-checked against NVML**
(`nvml_sampler.py`). An adversarial review (see "Corrections" below) overturned
the first draft's headline; this version is the honest one.

> Data: `events_coloc/`, `events_disagg/`, `results/gpu_*.csv`,
> `findings/campaign_*_*.json`. Reproduce: `setup_and_build.sh` (B300 sgl-kernel
> sm_103 rebuild) + `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas`, then
> `campaign_coloc.sh` / `campaign_disagg.sh` + `analyze_campaign.py`.

## What is and isn't wasted (the corrected headline)

The waste is **NOT** "the thinker idles so its GPU is wasted in co-location" — on a
shared GPU the thinker's idle time is filled by the talker, and the GPU stays busy.
The real, measured findings are:

1. **Disaggregation wastes a whole GPU.** Thinker alone on GPU0 for speech:
   per-GPU **NVML median util = 0% at every concurrency** (mean 8% / 14% / 15% at
   C=1/4/8; max ~57%); G1 union-busy 12.5→28% (a CPU-inclusive upper bound). The
   thinker's B300 is **idle the majority of wall-time** — a whole ~\$40k device
   mostly wasted — while GPU1 (talker+vocoder) runs NVML mean 49–59% (p50 67–74%).
2. **Co-location keeps the GPU active but is throughput-capped and memory-mismatched.**
   All speech stages on one GPU: **NVML "not idle" 70–78%** (G1 union-busy 87→95%
   is a CPU-inclusive upper bound; true SM occupancy unmeasured). The talker is the
   compute bottleneck (77–88% duty, saturates by C=2) and caps throughput; meanwhile
   the thinker uses only 12–33% of compute-time yet is allotted **77% of VRAM**
   (`total_gpu_memory_fraction: 0.769`, vs talker's 0.123) — a memory/compute mismatch.
3. **Understanding workloads leave the GPU under-used.** Text-only output (only the
   thinker runs): **NVML 42–53%, G1 union-busy 61–65%** → roughly **half the GPU
   genuinely idle**. The lone memory-bound thinker can't fill the device.

→ **No single static config wins.** Disaggregate and you strand a whole GPU on the
compute-light stage; co-locate and you cap throughput at the bottleneck stage +
mismatch VRAM; and the right answer flips by workload (speech → talker-bound, GPU
full; understanding → thinker-bound, GPU 35–58% idle). This is the case for
**dynamic, workload-aware cross-stage resource allocation / co-scheduling**.

## Hardening v2 — CIs + validated four-way (`findings/campaign_hard2.json`)

Concurrency sweep ×3 repeats (mean ± stdev → confidence intervals) on free GPU4
(the box is shared; GPU0–3 were taken by other users — pinned via
`CUDA_VISIBLE_DEVICES`), fixed work, pynvml NVML:

Values are **mean ± sample stdev (n=3)**:

| C | GPU union-busy% | NVML% | thinker duty | talker duty | code2wav duty |
|---|---|---|---|---|---|
| 1 | 79±1 | 58±1 | 15±1 | 61±2 | 19±1 |
| 2 | 78±0 | 66±1 | 21±1 | 70±1 | 24±2 |
| 4 | 79±1 | 65±2 | 23±1 | 69±0 | 40±2 |
| 8 | 85±2 | 65±7 | 28±2 | 72±1 | 57±2 |
| 16 | 89±2 | 72±3 | 33±1 | 72±5 | **78±3** |

Stable across repeats (±1–7). talker saturates ~70% by C=2 (bottleneck); thinker
15→33%; **code2wav becomes a co-bottleneck at C=16 (78%)**. *Caveat:* the spread is
dominated by a one-directional r1 cold-start (e.g. C=8 NVML r1=57 vs r2/r3=67/71),
not i.i.d. noise — treat ± as a warmup band, not a true CI. Short-segment GPU% is a
slight **under**estimate (the since/until window includes a ~1 s post-completion tail
with no GPU work; conservative — never inflates headlines).

**The four-way decomposition is now functional and validated.** A genuine
classifier bug — labeling a whole idle gap STARVED if *any* request was in flight
during it (`positive_in` over the whole interval) — over-reported STARVED and made
IDLE≡0. Fixed: split each gap by the in-flight curve (STARVED = sub-interval with a
request in flight, IDLE = sub-interval with none). Validated against a low-load
(3 s think-time) run: **IDLE = 69%** (GPU genuinely idle in the think gaps) vs a
**saturated C=16 run: IDLE = 8%** (pipeline nearly always full). The metric now
correctly tracks the load regime; union-busy and four-way agree
(low-load union-busy 27% ↔ IDLE 69%+overlap).

## Hardened re-measurement — fixed work, pynvml (`findings/campaign_hard.json`)

Addresses the audit's methodology gaps: **bounded per-request work** (`ignore_eos`,
32 tokens → removes the long tail, `frac_long=0`, vs the earlier bimodal 6-vs-110;
note work is *bounded, not constant* — `ignore_eos` only caps long outputs, so
short-prompt requests still stop early: text_chunks 16–32 CV~28%, audio CV~40%);
**pynvml** NVML (true ~50 ms cadence, not nvidia-smi's ~181 ms); **coordinator
`request_admission`/terminal events captured** (in-flight curve now correct);
N reported. `GPU%` = union-busy over the full segment wall-window.

| segment | C | workload | **GPU%** | NVML | thinker | talker | code2wav | N | rps |
|---|---|---|---|---|---|---|---|---|---|
| hard | 1 | text→speech | 79% | 59% | 15% | 60% | 20% | 8 | 0.78 |
| hard | 4 | text→speech | 87% | 72% | 30% | 77% | 46% | 24 | 1.55 |
| hard | 8 | text→speech | 88% | 61% | 26% | 78% | 54% | 32 | 1.88 |
| hard | 1 | →text | 47% | 35% | 47% | — | — | 8 | 4.0 |
| hard | 8 | →text | 56% | 36% | 56% | — | — | 32 | 12.8 |
| **hard** | 8 | **mixed 50/50** | **86%** | 71% | 29% | 62% | 60% | 32 | 3.0 |

Confirms (now trustworthy) the corrected picture: speech keeps the GPU active
(79–88%, talker-bottlenecked); text-only leaves it ~half idle (GPU 47–56%, NVML
35–36%). **New: mixed text+speech traffic fills the GPU (86%) and activates all
three AR stages** (thinker 29% / talker 62% / code2wav 60%) — text requests load
the thinker while speech requests load talker/vocoder, naturally co-filling. A
deployment *cannot control its incoming mix*, so static placement/VRAM-split tuned
for one workload is wrong for another — directly motivating workload-aware
cross-stage scheduling.

Note on the four-way decomposition: with admission events now captured the
in-flight curve is correct, but at these back-to-back loads a request is essentially
always in flight, so per-stage IDLE≈0 (all non-busy is STARVED) is a **legitimate**
result, not the earlier bug. The GPU-waste metric remains **union-busy / NVML**, not
per-stage STARVED — that framing is unchanged and now properly supported.

## Co-location campaign — pre-hardening (kept for reference; `findings/campaign_coloc_corrected.json`)

`GPU%` = union of all stage-busy intervals = the GPU's real utilization; per-stage
= **duty cycle** (stages time-share the GPU, so they sum to >100%).

| segment | C | workload | **GPU%** | NVML | thinker duty | talker duty | code2wav duty |
|---|---|---|---|---|---|---|---|
| sweep | 1 | text→speech | 87% | 70% | 12% | 77% | 23% |
| sweep | 2 | text→speech | 89% | 75% | 24% | 88% | 37% |
| sweep | 4 | text→speech | 92% | 75% | 29% | 87% | 53% |
| sweep | 8 | text→speech | 94% | 78% | 33% | 87% | 69% |
| sweep | 16 | text→speech | 95% | 71% | 29% | 83% | **84%** |
| text | 1 | →text | **61%** | 53% | 82% | — | — |
| text | 4 | →text | **65%** | 47% | 84% | — | — |
| text | 8 | →text | **65%** | 42% | 85% | — | — |
| audio_short | 4 | speech MT=64 | 86% | 72% | 34% | 85% | 49% |
| audio_long | 4 | speech MT=512 | 91% | 73% | 25% | 87% | 48% |

- talker_ar is the speech compute bottleneck (saturates ~87% by C=2); code2wav
  becomes a 2nd bottleneck at C=16 (84%); thinker compute-light (12–33%).
- audio length drives talker load: MT 64→512 raises talker/thinker imbalance,
  throughput 1.14→0.26 r/s.

## Disaggregation campaign (thinker GPU0 | talker+code2wav GPU1; `findings/campaign_disagg.json`)

GPU0 = thinker only; GPU1 = talker+code2wav. NVML reported per-GPU
(mean / p50). Placement verified from server logs (thinker→GPU0, talker_ar &
code2wav→GPU1) + the analysis `--gpu-map`; corroborated by the `gpu` field in
thinker/talker fwd_begin metadata (code2wav fwd events carry no `gpu` field).

| segment | C | GPU0 union-busy (G1) | GPU1 union-busy (G1) | **NVML gpu0** | NVML gpu1 |
|---|---|---|---|---|---|
| disagg | 1 | 12.5% | 74.5% | **8% / p50 0%** | 59% / p50 74% |
| disagg | 4 | 23.4% | 71.7% | **14% / p50 0%** | 51% / p50 68% |
| disagg | 8 | 27.7% | 77.8% | **15% / p50 0%** | 49% / p50 67% |

**GPU0 (thinker) NVML median is 0% at every concurrency** — the device is idle
more than half the time. Throughput 0.18→0.42→0.57 r/s, **0.86–0.95× of
co-location** (cross-GPU relay overhead) — so disaggregation pays a whole mostly-idle
GPU AND is not faster.

## Corrections vs the first draft (what the review caught)
- **"87% wasted GPU in co-location" was WRONG.** Per-stage STARVED ≠ GPU idle on a
  shared GPU; the GPU is 87–95% busy. Fixed: report union-busy + NVML; the
  whole-GPU-waste claim now comes from the *disaggregated* data (correct).
- **The four-way decomposition is now FIXED and functional** (see "Hardening v2").
  It was broken by TWO bugs, both now resolved: (1) no `request_admission` events
  recorded (coordinator recorder not started) → added env auto-start in the
  coordinator process; (2) the classifier labeled a whole idle gap STARVED if *any*
  request was in flight during it → split each gap by the in-flight curve. Validated:
  IDLE = 69% at low load (think-time) vs 8% at saturation. (Earlier drafts that called
  it "non-functional / legitimately all-STARVED" are superseded.)
- **`union-busy` and NVML are both "any-kernel-active" fractions, not SM occupancy.**
  union-busy brackets `run_batch` (CPU scheduling/sampling/detokenize included) → a
  CPU-inclusive upper bound; that is the union(87%) vs NVML(70%) gap. True compute
  occupancy is lower and unmeasured. State GPU "active / not idle", not "utilized".
- **Backpressure (G2) numbers were an artifact** (shm releases the credit
  immediately; bp_wait ≈ 52 µs/event event-loop latency). Dropped.
- **Instrumentation coverage is workload-limited.** G1 covers only thinker / talker /
  code2wav. Encoders (image/audio, SimpleScheduler) and the CPU `decode` stage are
  uninstrumented; they were idle for these text-input workloads (encoder event files
  empty), so union-busy is valid HERE — but any audio/image-INPUT workload would run
  the uninstrumented encoders and silently undercount.
- **NVML cadence is ~181 ms (not 50 ms)** — `nvidia-smi` fork/exec latency dominates
  `--interval-ms 50`; `text_c1` has only ~25 samples → wide CIs shown as points.
- **"Bottleneck flips" is not apples-to-apples** (text-only literally runs fewer
  stages). Stated as utilization/bottleneck *difference* by workload, not a clean flip.
- **Weak statistics**: N=6–32/segment, output length bimodal (~6 vs ~100+ audio
  chunks); `sweep_c1` is long-request-heavy (frac_long 0.33 vs ~0.25) which understates
  C=1 rps and inflates apparent scaling. The sub-linear trend (16× C → 4.2× rps) is
  robust; specific rps are directional, CIs not computed.

## Verdict
**GO, with a corrected (and sharper) thesis.** The motivation is not "co-location
wastes GPU" — co-location works (GPU ~90% busy). It is: **every static resource
configuration is wrong for some workload** — disaggregation strands a whole GPU on
the compute-light stage (thinker GPU0: 75–92% idle, measured), co-location caps
throughput at the bottleneck stage and mismatches VRAM, and the optimal choice
flips between speech and understanding. That is a real, measured argument for
dynamic cross-stage resource allocation, not removable by batching or per-stage
point optimization.

## Honest next steps (to harden, not re-measure the same thing)
- Fix the workload generator to hold per-request work constant (fixed decode
  length / ignore_eos) and report N + variance → trustworthy throughput-vs-C curves.
- Collect coordinator `request_admission`/terminal events so IDLE-vs-STARVED is
  meaningful at low load (add recorder auto-start to the coordinator process).
- Mixed text+speech traffic on one deployment (the case where the workload-dependent
  bottleneck argument bites hardest).
- Only then: move to the solution side (optimality-gap analysis + a first dynamic
  co-schedule / VRAM-split intervention vs the static baseline).
