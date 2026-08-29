# sglang-omni scheduling testbed

A model-agnostic, declarative testbed for **regression** and **performance** testing of
cross-stage scheduling / placement / admission changes to sglang-omni.

Goal: when we modify the framework (e.g. add a coordinator admission-policy seam, a
runtime placement controller, adaptive backpressure), run **one command** that exercises a
broad matrix of *models × pipeline shapes × workload types × load patterns* and tells us:

1. **Regression** — did behavior/latency/throughput change vs a stored baseline (within tolerance)?
2. **Performance** — did the change move the scheduling metrics we care about (throughput, TTFT,
   TTFA, frame jitter, per-stage GPU duty, GR-active / SM-active occupancy)?

This is deliberately framework-respecting: it only drives the public serving API
(`/v1/chat/completions` with `modalities`, `/v1/audio/speech`) and reads the existing
event-recorder + nsys metrics. It does not fork or patch the engine to measure.

## Layout

```
dev/testbed/
  scenarios.yaml     declarative matrix: servers (launch spec) + scenarios (workload × load)
  testbed.py         runner: launch server per config → drive load → collect metrics → teardown
  run_all.sh         one command: run the whole matrix (inside the omni-phase0 container)
  run_baseline.sh    build the FIXED baseline, fast models first, --update-baseline per group
  baselines/         stored baseline metrics for regression compare (git-tracked)
  results/<run_id>/  per-run outputs + report.md
```

## Building the fixed baseline

```bash
docker exec omni-phase0 bash /workspace/sglang-omni/dev/testbed/run_baseline.sh
```
Runs each model group (asr → higgs → qwen3tts → omni) and `--update-baseline`s it as it
finishes, so an interruption during the slow Qwen3-Omni group keeps the fast baselines saved.
Only points that succeeded ≥90% (`MIN_OK_RATIO`) are baked — a contention-degraded run is
never promoted to the fixed reference.

## Evaluating an admission/scheduling policy — do it the VALID way

Two methodology rules (a shared, contended box makes the naive comparison untrustworthy):

1. **Open-loop, not closed-loop.** Closed-loop (`concurrency` workers) self-clocks at the
   worker count, so an admission policy can only ever throttle — the overload regime it's FOR
   is unreachable. Use `mode: openloop` scenarios (Poisson arrivals at a target `rate`,
   decoupled from completion) and read **tail latency (p95/p99), goodput@SLO, shed-rate** —
   not closed-loop rps. (Closed-loop scenarios remain only as a reproducibility/throughput
   sanity check and for cross-commit regression of the engine.)

2. **Same-session paired A/B, not vs a stored baseline.** A baseline baked hours ago on a
   shared box is confounded with whatever else was running then. `--ab-policy` runs each
   server **twice back-to-back this session** (A = no policy, B = policy) on the same GPU and
   reports the within-session paired delta, with a GPU-exclusivity preflight and per-arm
   box/GPU context (load, gpu_util, co-tenant count) recorded so you can see if the box drifted.

```bash
# the valid policy experiment: open-loop scenarios, same-session paired A/B
docker exec omni-phase0 python /workspace/sglang-omni/dev/testbed/testbed.py \
  --filter openloop \
  --ab-policy sglang_omni.pipeline.admission.FifoGate --ab-policy-args '{"max_inflight": 8}'
# -> results/<ts>/paired_report.md : per-rate Δ = (B−A)/A for p95/p99/goodput, both arms' env
```

`--policy` (single-arm, vs stored baseline) also exists but is the WEAKER form — only use it
for a quick smoke, never to conclude a policy is good/bad on this box.

## Run

```bash
# full matrix (auto-skips models whose weights aren't cached)
bash dev/testbed/run_all.sh

# subset / single
docker exec omni-phase0 python /workspace/sglang-omni/dev/testbed/testbed.py --filter omni_speech
docker exec omni-phase0 python /workspace/sglang-omni/dev/testbed/testbed.py --scenario omni_speech_csweep

# perf profiling (adds nsys GR-active/SM-active per scenario; slower)
... --profile

# regression: compare this run to the stored baseline (PASS/FAIL within tolerance)
... --regression
# accept current run as the new baseline
... --update-baseline
```

## Adding a model = one row

A new model is a `servers:` entry (config path + which weights it needs) plus one or more
`scenarios:` rows pointing at it. The runner skips any server whose `requires_weights` is not
in the local HF cache (printed as `SKIPPED: weights not cached`), so the matrix can declare
full coverage while only running what's downloaded.

## Stability

Each (scenario, point) is run `repeats` times (default 3, set in `scenarios.yaml` `defaults`
or per-scenario `load.repeats`) and reported as **mean ± sample stdev** — the shared box is
noisy, so single-shot numbers aren't trustworthy. Regression compares the per-point means
within `regression_tol`.

## Metrics

- **Client-side (always):** `ttft` (first text delta), `ttfa` (first audio chunk),
  `total`, `frame_gap_max` (worst-case audio jitter / underrun proxy), `throughput_rps`
  (CLOSED-LOOP), `ok/total`. Each reported as mean ± stdev over the repeats.
- **Server-side (`--profile`):** GR-active / SM-active (drain-aware, via nsys gpu-metrics);
  per-stage GPU duty + 4-way bubble decomposition (via the event recorder, when enabled).

See `../pipeline-coscheduling/` for the per-stage decomposition tooling this reuses.
