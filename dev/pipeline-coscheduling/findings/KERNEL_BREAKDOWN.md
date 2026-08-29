# Per-stage per-kernel GPU breakdown (talker decode bottleneck)

Method: server under nsys (`--trace=cuda`), **talker CUDA-graph OFF** (so its eager
kernels are visible — torch profiler via /start_profile captured CPU-only because
graph-internal kernels are invisible to CUPTI). C=8 speech, fixed work, ~30s steady
load. Isolated per OS process from the nsys sqlite (`talkern.sqlite`).

| stage | GPU kernel time | launches | dominant kernels |
|---|---|---|---|
| **talker_ar** | **8936 ms** | **2.6M** | elementwise 16.7% (495k), small GEMM nvjet 32x64/8x64/64x8 ~31% (206k+181k+103k), cutlass grouped-GEMM 8.2% (51k = secondary-head codebook), flash-attn 6.6% (103k), RMSNorm/rotary/silu/add tail |
| code2wav (vocoder) | 4391 ms | 698k | implicit_convolve_sgemm 34% (2956 × 511µs), elementwise 21%+11%, cudnn nchwToNhwc 6.6% |
| thinker (MoE) | 925 ms | 136k | fused_moe 41%, softmax, reduce |

## Findings
- **talker is the GPU bottleneck — ~10× the thinker** (8.9s vs 0.9s); vocoder second (4.4s).
- The talker is **"death by a thousand tiny kernels": 2.6M launches**, NO single hotspot
  (largest is elementwise at 16.7%, avg ~3µs). The bulk is small GEMMs (nvjet **32x64 /
  8x64 / 64x8** tiles — tiny matrices = per-step batch×hidden of AR decode) + a flood of
  elementwise/norm/rotary ops + a grouped-GEMM (secondary-head codebook AR).
- Small tiles + tiny elementwise → **low SM occupancy (~30%)** — the kernel-level cause
  of the measured SM-idle.
- NOTE: profiled EAGER (graph off) so launch *counts/overhead* are inflated; in
  production (graph on) launch overhead is hidden by graph replay, but the kernels still
  run at the same low occupancy → the occupancy problem persists, the launch problem does not.

## Optimization implications (honest)
- **No single kernel to optimize** — it's the aggregate of the AR decode's many small ops.
- Levers: (a) **fuse** the elementwise/norm flood into fewer kernels (some FusedAddRMSNorm
  already present; more fusion = fewer launches + marginally better occupancy); (b) the
  small-GEMM low occupancy is **structural** to small per-step matrices (AR decode) — not
  fixable by fusion; bigger batch would enlarge them but throughput is latency-bound
  (batching didn't help, measured) so that lever is closed; (c) the secondary-head codebook
  grouped-GEMM is serial per codebook layer — a model-structure cost.
- Net: a quick kernel win is unlikely; the talker AR decode is fundamentally a
  small-matrix/low-occupancy workload. Throughput is bounded by per-step latency, which is
  the sum of these many small kernels. Reducing per-step kernel COUNT (fusion) is the most
  tractable but bounded lever.
