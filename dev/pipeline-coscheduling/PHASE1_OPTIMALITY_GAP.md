# Phase 1 — optimality gap (the target a dynamic scheduler must close)

Computed from the hardened measurements (no new runs). `optimality_gap.py`,
`findings/optimality_gap_*gpu.json`.

## Metric
**GPU-resource efficiency** = useful GPU-work / provisioned GPU-time
= (Σ over the G provisioned GPUs of that GPU's **union-busy** fraction) / G.
**Optimality gap = 1 − efficiency.** Work-conserving optimum = 100% (every
provisioned GPU-second on useful stage-work).

> Uses union-busy (actual GPU work), NOT sum-of-stage-duty: stages time-share the
> GPU so their forward wall-spans overlap and sum-of-duty exceeds 100% — it is not
> GPU-work. union-busy is validated + NVML-corroborated.

## ⚠️⚠️ HARD measurement (nsys SM-active) — "the gap" is a STACK of metrics

Measured true GPU hardware utilization with `nsys --gpu-metrics-devices` (1 kHz,
`SMs Active [Throughput %]` = fraction of SMs doing work; metric 3), GPU4, C=8,
fixed work. This is the "hard number" — not union-busy (CPU-inclusive) nor NVML
(any-kernel flag). It **does not give one gap; it shows the utilization stack
diverges by an order of magnitude**:

| C=8 segment | union-busy (timeline) | NVML util | GR-Active (HW engine) | **SM-Active (HW SMs)** |
|---|---|---|---|---|
| speech | 88% | 61% | 30% | **9.5%** |
| text-only | 56% | 36% | 32% | **18.6%** |
| mixed 50/50 | 86% | 71% | 76% | **24.8%** |

**Even the "busy" mixed case runs the SMs at only ~25%.** The GPU's compute is
75–90% idle *even when a kernel is on it* — memory-bound AR decode, small batches,
tiny kernels.

### Corrections after audit #5 (drain artifact + concurrency sweep)
The first HARD table was **wrong**: the 18 s nsys window outlived the workload, so
session-averages were diluted by post-drain idle (speech "9.5%" → steady-state ~26%).
Drain-excluded (GR-active>5%) and with a concurrency sweep, the real numbers:

**SM occupancy is workload-independent at C=8 (~26% for speech/text/mixed alike)** —
the earlier speech-vs-mixed spread was pure drain artifact. **SM-active vs concurrency
(speech, drain-excluded):** C=1 ~17% (noisy) → C=8 27% → C=16 30% → C=32 28%. It
**rises with batch then plateaus at ~28–30% — never approaches saturation even at
C=32** (warps-in-flight ~9%, DRAM ~8%: not compute- or memory-bound — the omni kernels,
small active-MoE + tiny talker/vocoder kernels, can't fill a 148-SM B300).

### The decisive reframe (corrected)
Two **different** gaps at different levels, and BOTH are smaller-for-scheduling than
the first draft implied:

1. **Timeline-packing gap** (is *some* stage's kernel on the GPU): the truest measure
   is **GR-Active busy-fraction**, which AT LOAD is **speech 97–98%, mixed 95%, text
   71%**. So co-location already packs the timeline ~95–98% for speech/mixed → **almost
   no scheduling gap at load**; the real timeline gap is **text/understanding ~29%**
   (thinker decode stalls leave the GPU idle) + low-load fill + disaggregation. (The
   earlier union-busy 60–86% / "14–40% gap" was CPU-inclusive and too loose; GR-active
   is the hardware truth.)
2. **SM-occupancy gap** (~70%, plateaus at ~30% even at C=32): partly concurrency-
   recoverable (17→30%) but mostly a **structural kernel/batch-efficiency ceiling**, NOT
   a cross-stage-scheduling lever.

**Honest implication — the thesis NARROWS:** at load, co-location already achieves
~95–98% timeline packing for speech/mixed, so the cross-stage-scheduling win there is
small. The defensible win is **workload-dependent and modest**: text/understanding
timeline stalls (~29%, packable with complementary work = consolidation), low-load fill
bubbles, and not-stranding-a-GPU (disaggregation). The headline 70% SM-idle is a
separate kernel/batching-efficiency problem (orthogonal to scheduling). Any claim must
be denominated in GR-active (hardware), not union-busy, and must NOT sell the SM-idle
as a scheduling opportunity.

## Secondary view — timeline-level gap (union-busy / NVML range)

The "efficiency" depends on what counts as useful GPU-work, which we did NOT measure
at the SM level. Two proxies bracket it: **union-busy** (G1 forward wall-spans,
*includes CPU* scheduling/sampling/detokenize → **over**states efficiency, **under**states
gap) and **NVML mean util** (any-kernel-active sampling → lower, still not SM occupancy).
Same segments, both proxies (audit-recomputed from raw NVML over the SEG windows):

| config / workload | C | gap (union, optimistic) | gap (NVML-mean) | **honest gap range** |
|---|---|---|---|---|
| **Disaggregation**, speech | 8 | 47% | 68% | **47–68%** |
| Disaggregation, speech | 1 | 56% | 66% | 56–66% |
| **Co-location, text-only** | 8 | 44% | 64% | **44–64%** |
| Co-location, speech | 8 | 12% | 39% | **12–39%** |
| Co-location, speech | 16 | 10–13% | ~30%? | ~13–35% |
| Co-location, MIXED 50/50 | 8 | 14% | ~30%? | 14–~30% |

**True SM occupancy is unmeasured** (would need DCGM SM_ACTIVE / nsys). The absolute
percentages are NOT physical constants; only the **ordering and large gaps** are robust.

## What is ROBUST (survives the metric choice)

1. **Disaggregation structurally strands a GPU.** thinker needs only ~28% of a GPU
   (duty cycle, measured directly, ~identical in disagg and coloc), so a dedicated
   thinker GPU is ~70% wasted under *any* metric (gap 47–68%). Recoverable by *static*
   co-location (which the team ships) → this is "don't disaggregate", **not the
   research target**.

2. **Single-workload underfill on a shared GPU is large and not closeable by any
   static config** (text-only gap 44–64%): one memory-bound stage can't fill a GPU,
   and co-locating the *same* workload doesn't add a complementary stage. The only
   lever is **packing complementary work** — a dynamic/workload-aware concern.

3. **The win is fleet CONSOLIDATION, not per-workload speedup.** The text-only idle is
   memory-bound thinker idle — you cannot run text *faster* on it. The idle is only
   usable by *other* work, i.e. serving a mixed fleet on **fewer GPU-seconds**. State
   it as consolidation, not throughput.

## What is NOT yet supported (needs more data before claiming)

- **"Mixed traffic recovers the underfill" rests on ONE 50/50 point** (mixed_c8). And
  86% there is largely because the *speech* half already nearly fills the GPU (mix
  talker-duty 62% ≈ pure-speech), absorbing the text half into its idle — not symmetric
  complementary packing. A text-heavy mix (e.g. 90/10) would likely fall back toward
  the text-only ~56%. **Needs a mix-ratio sweep** before claiming a recovery curve or a
  "hold 85–90% across the mix space" target.
- **"Achievable optimum ~88%"** is "the best we measured" (and union-inflated), not a
  principled bound. Don't quote it as the ceiling.

## Honest caveats
- **Absolute gaps are metric artifacts** (union vs NVML-mean ≈ 2× apart); SM occupancy
  unmeasured (DCGM/nsys needed). Only ordering + the large disagg/text gaps are robust.
- Disagg numbers are from a **pre-hardening** run (variable work); re-run hardened
  before quoting them — though the *fraction* is structurally robust (thinker ~28%).
- Efficiency = work-conservation, NOT end-to-end latency / real-time audio deadline
  (a separate objective, not modeled here).
- Per-GPU disagg attribution uses the stage→GPU map; code2wav fwd events lack the gpu
  field, so GPU1's split is map-assumed.
- `optimality_gap.py` has no selftest; lowload/underloaded segments produce a
  meaningless "gap" (offered-load, not inefficiency) — excluded here.
