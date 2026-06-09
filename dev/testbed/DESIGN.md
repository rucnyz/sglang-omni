# Cross-Stage Admission & Scheduling Policy for sglang-omni — Design

This document is the first-principles design of the cross-stage admission/scheduling contribution:
*why* the seam is needed, *what* the ideal policy is, *where* each piece lives in the code, and
*what* it measured. Every section follows **Motivation → Observation → Solution (→ code) →
Results (→ where + analysis)**. Measured numbers and reproduce commands live in
[`RESULTS.md`](./RESULTS.md); this file is the reasoning that produced them.

The contribution is **one evolving policy**, not a zoo:

```
NoOpAdmission                              # the framework's current behaviour (fire-and-forget)
  └─ AdaptiveGate(adapt=False, limit=N)    # ≡ FifoGate(N): a static in-flight cap
       └─ AdaptiveGate(adapt=True)         # self-tuning in-flight limit (throughput controller)
            └─ AdaptiveGate(slo_s=S)       # + deadline-aware shedding
```
Each is a strict special case of the next (`slo_s=None`, `adapt=False`, or absent ⇒ the simpler
behaviour), so there is no policy selection or fallback — only one knob-set with degenerate points.

## TL;DR — what wins where (two mechanisms, two scopes)

The policy has **two mechanisms on two independent axes**, and each "wins" in a different scope —
both are no-ops below capacity (admission control only acts under overload):

| mechanism | what it does | wins where | on the tested models |
|--|--|--|--|
| **concurrency gate** (`AdaptiveGate`) | bounds in-flight at the throughput knee | only where the pipeline has a **collapse cliff** (raising concurrency *drops* throughput) | **1 win** (TTS); the plateau models (omni speech/understand/mixed, Higgs, ASR) have no collapse to prevent → gate provably `L→max`, inactive → **no-regression by construction** |
| **deadline shedding** (`slo_s`) | drops requests that can't meet their deadline | wherever there is **overload** (a growing queue) — which is *every* model past capacity | **wins on all 3 tested** (TTS/ASR/Higgs): goodput up, p99 pinned at the SLO vs 28–74 s |

So the honest one-liner: **the gate wins only where a model collapses (TTS among those tested);
shedding wins wherever a model is overloaded (all of them). Below capacity both are transparent
(no-regression).** "Only one place wins" is true *for the gate alone* — shedding broadens the win
to every overloaded model. (One near-capacity wrinkle: at ~capacity, Higgs R=12, shedding trades a
little goodput; the clear shedding win is *above* capacity.)

---

## 0. The gap in the framework (root motivation)

**Motivation.** sglang-omni cleanly splits *mechanism* from *policy*: the generic stage runtime
never branches on scheduler type; a model attaches behaviour through config + import-string hooks
(`route_fn`, `wait_for_fn`, `merge_fn`, `placement_policy`, …).

**Observation (grounded in code).** *Every* existing hook is **per-stage / local**. The
**cross-stage** decisions a scheduler needs have **no seam**:
- **Admission** is fire-and-forget — `Coordinator._submit_request` emits a `request_admission`
  telemetry event and then `submit_to_stage(entry_stage)` *immediately*: zero gating / priority /
  deadline / load-awareness.
- **Placement** is static config + a validate-only `placement_policy` (no runtime seam).
- **Backpressure** is a hardcoded per-relay `credits=2`.

So the framework externalizes per-stage policy but leaves **cross-stage coordination**
fire-and-forget/static. That is the gap.

**Solution.** Add the missing cross-stage policy seam *in the same import-string style*, default to
a no-op (zero behaviour change), and put one real, general policy on it. Honest framing: this is a
**systems / scheduling-layer** contribution, **not** "recover X% GPU" — single-model serving
occupancy is already structural (~28% SM at load; five audits closed the util gaps). The value is
the policy *tier* the framework lacks, which static config + per-stage hooks cannot express.

---

## 1. The seam

**Solution.** A minimal protocol the coordinator calls around a request's lifetime.
- `AdmissionPolicy.on_submit(ctx)` — `await`ed *before* the request enters the entry stage; may
  gate (limit in-flight), delay (pace), reorder (priority), or **shed** (raise to reject).
- `AdmissionPolicy.on_complete(request_id)` — release accounting when the request leaves.
- `AdmissionContext = {request_id, request, entry_stage, inflight}` — read-only snapshot.
- `NoOpAdmission` — the default; **byte-for-byte equivalent to the current fire-and-forget path**.

**Code.**
- `sglang_omni/pipeline/admission.py` — `AdmissionPolicy` (Protocol), `AdmissionContext`,
  `NoOpAdmission`, `resolve_admission_policy(spec)`, `admission_policy_from_env()`.
- `sglang_omni/pipeline/coordinator.py` — `Coordinator.__init__(admission_policy=None)`; the hook
  in `_submit_request` (calls `on_submit` before `submit_to_stage`, default `None` ⇒ skipped);
  `_inflight_count()` (RUNNING requests); `_notify_admission_complete()` in `submit()/stream()`
  `finally` and in `abort()` (so a slot can't leak).
- `sglang_omni/config/schema.py` — `PipelineConfig.admission_policy: str | None` (import string).
- `sglang_omni/pipeline/mp_runner.py` — resolves the policy (config or
  `SGLANG_OMNI_ADMISSION_POLICY` env) and passes it to the `Coordinator`.

**Result.** Wiring the seam in changes nothing until a policy is configured (verified by the NoOp
unit test + "identical offered/equal achieved" omni runs). It's the framework-respecting foundation
the rest builds on.

---

## 2. Why a self-tuning in-flight limit (the controller)

### 2.1 Why gate at all

**Motivation.** Admission policies only matter under **open-loop overload** (arrivals decoupled
from completion — the realistic serving regime), where offered load can exceed service capacity.

**Observation.** Some pipelines **self-collapse** past a concurrency. Qwen3-TTS has a throughput
*cliff* (~C16): under sustained open-loop overload, achieved rps collapses toward 0 and ttfa p99
reaches **tens of seconds to minutes** (the coordinator queue grows unboundedly while the server
thrashes). Capping concurrent in-flight at the *knee* keeps the server in its efficient regime.

**Observation (the catch).** The knee is **model-dependent**. A static `FifoGate(8)` recovers TTS
(2–4× achieved, p99 −69..−94%) but *hurts* Qwen3-Omni −14..−88% — omni batches efficiently (no
cliff; its sweet spot is ~16, not 8), so an 8-cap pins it below its own optimum. The optimal cap
can't be a constant.

**Solution.** Make the limit `L` **self-tune online** from each pipeline's *own* throughput
response, with one fixed parameter set and no per-model number. The static cap becomes the
degenerate `adapt=False` case: `FifoGate(N) ≡ AdaptiveGate(limit=N, adapt=False)`.

**Code.** `sglang_omni/pipeline/admission.py` — `AdaptiveGate`, `FifoGate`.
**Results.** Static-cap baselines (TTS win, omni hurt) in `RESULTS.md` §6 "Policy #1 / #1b".

### 2.2 The signal: throughput, not latency

**Observation.** An early controller drove `L` toward keeping in-pipeline *latency* near its
unloaded minimum. It **failed on omni** (−17..−22% achieved): it finds the *latency* knee, but a
plateauing model's efficient batch point already has high latency, so it throttled omni down to ~8.

**Solution.** Drive `L` from **throughput** (completions per unit time, free at the seam). This
targets *capacity* directly, so one rule generalises: a cliff model (raising `L` past the knee
*drops* throughput) and a plateau model (throughput merely *flattens*, never drops → `L` rises to
max, gate inactive) are handled by the same logic.

### 2.3 The seven invariants — each fixes a *failure caught in the trace*

**Method.** The controller prints its state every control step (`[ADMGATE] step=… limit=…
best_limit=… ceiling=… inflight=… sat=… tput=… best_tput=…`) and exposes `stats()`. Running it on
the real servers and *reading that trace* — not analysis — exposed each failure below. Each
invariant makes a bad state **impossible by construction**.

| invariant | failure it prevents (seen in the trace) |
|--|--|
| **fail-open** — start at floor `min_limit=8`, probe up, reduce only on hard evidence | throttling an uncontended system (the R=1 cold-start regression) |
| **fixed wall-clock throughput window** — `raw = n/Δt`, Δt ≥ `window_s=3s` | a completion *clump* (many sequences finishing in one decode step) closing a window with Δt≈3 ms → `raw=n/Δt` **exploded to ~7700** → `best_tput` poisoned → permanent spurious-collapse oscillation |
| **outlier-cap** — reject a window whose rate > `2× best_tput`-smoothed | a residual clump inflating `best_tput` (real throughput can't jump 2× in one window) |
| **persistent best** — `best_decay=0.999` (~0.1%/step) | forgetting the good operating point over a long run → `L` drifts *up* off the knee (12→16, tput 2.8→1.5) |
| **ceiling memory** — never probe up to/over a known collapse point | oscillating back into the cliff |
| **inflight-level saturation gate** — act on `L` only when peak in-flight ≥ `0.75·L` | a *demand drop* (warmup ended / arrival fell → gate slack, low tput) misread as a capacity collapse → `L` clamped spuriously |
| **small `up_factor=1.15`** | a *deep* cliff-discovery overshoot — bounds the one-time startup transient to 8→9.2→…→12, not 8→…→23 |

**The fundamental-overshoot argument (first principles).** Telling a *cliff* (back off) from a
*plateau* (keep rising to max) requires stepping past the knee **once** and observing whether
throughput holds or drops — **no prior-free controller can avoid touching the cliff once**. The
invariants make that touch *mild* (one small step) and *one-time* (the ceiling stops a re-touch),
which is the principled minimum, not a tuned constant. Zero-overshoot would require per-model prior
knowledge, which contradicts "one general parameter set".

**Code.** `sglang_omni/pipeline/admission.py`:
- `AdaptiveGate.__init__` — params `min_limit, max_limit, window_s, up_factor, collapse_frac,
  ceiling_relax, best_decay, ewma_alpha, sat_frac, outlier_cap`.
- `AdaptiveGate._observe(now)` — the control step (time-windowed throughput, outlier-cap, EWMA,
  persistent best + decay, ceiling relax, **saturation-gated** collapse vs probe-up, the
  `[ADMGATE]` print).
- `AdaptiveGate.on_submit` — the FIFO gate (cancellation-safe + idempotent grant protocol);
  `on_complete` — release + `_observe`.
- `FifoGate(AdaptiveGate)` — the `adapt=False` static-cap subclass.

**Tests.** `tests/unit_test/test_admission.py` — gate FIFO/cancellation/idempotency (the C1/C2
audit bugs), `FifoGate ⊂ AdaptiveGate`, and controller convergence on synthetic curves: sharp
cliff → L≈knee, gradual roll-off → clamp, plateau → L→max, low-load → L→max, **completion-clump
survives** (regression for the n/Δt explosion), **holds-knee-over-long-run** (regression for the
fast-decay drift). 14 tests, all pass.

### 2.4 Results — cross-model generality

**Where.** `RESULTS.md` §6 "AdaptiveGate" + "Cross-model generality sweep"; raw runs under
`dev/testbed/results/<ts>/`.

**What.** All 4 models / 6 open-loop configs, paired A/B (no-policy vs gate), one parameter set:
- **TTS (cliff): WIN** — gate auto-finds L≈8; at R=4/6 no-policy is dead (0.008/0) while the gate
  holds ~2.5/s = capacity. Two measurements (distinct runs — do not conflate): the **val10** run
  (`20260608_113233`, exclusive GPU) R=3 achieved **0.48→2.18 (+350%)**, p99 **70→11 s (−84%)**;
  the **cross-model sweep** (`20260608_164403`, load-matched loadavg **7.0 vs 7.7**) R=3
  **0.33→2.33 (+618%)**. The two differ because they ran under different box conditions; both show
  the same direction.
- **omni-speech / omni-understand / omni-mixed / Higgs / ASR (plateau): NO REGRESSION** — the gate
  drives **L→max and is inactive**, achieved equal at identical offered.

**Analysis (why it's a guarantee, not luck).** The `[ADMGATE]` trace shows the gate **clamps `L`
only on a model with a real throughput cliff (TTS)** — where it wins — and **provably goes to
`L→max` (inactive) on every plateau/batching/mixed model**, so it *cannot* regress them by
construction. The shared box was wildly noisy during the sweep (loadavg swung 7→121); the apparent
A/B deltas on the inactive models track the **loadavg delta**, not the policy — proven per-pair
(when arm-B ran hotter it "lost" −17..−25%; when cooler it "won" +5..+98%; both with `L=max`
traces showing no throttle mechanism). So the raw deltas on a loaded shared box are not trusted —
the gate-activity trace is.

---

## 3. Deadline-aware shedding — the gate's next action

### 3.1 Motivation & observation

**Motivation.** The gate restores **throughput** under overload, but the coordinator **wait-queue
still grows** at rate ≫ capacity. Throughput ≠ goodput.

**Observation (our own data).** TTS gate-queue holds achieved ~2.5/s but p99 climbs **13→39 s** as
the rate rises, while the SLO is 4 s → **goodput collapses (0.36→0.10)**: the server is healthy but
nearly every completion is *too late to be useful*. The gate fixed "server collapse"; it did not
fix "queue blowup".

### 3.2 Solution (first principles)

A request that *cannot* meet its deadline should be **shed** (rejected now), not queued to certainly
miss SLO — freeing capacity for requests that *can*. By **Little's law**, a newly-arriving request's
time-in-system ≈ `(in-flight + queued + 1) / drain-rate`. **The drain rate is the service
*capacity* (`best_tput`), NOT the current throughput** — below capacity the current rate is
arrival-limited (e.g. 1.7/s when offered < 2.5/s capacity) and would *over-estimate* the wait and
over-shed; the queue actually drains at capacity. Shed iff predicted `> slo_s`. **Fail-open**: no
`slo_s`, or no capacity estimate yet ⇒ never shed (so `AdaptiveGate(slo_s=None)` ≡ the queue-only
gate, and a static `FifoGate` — no throughput estimate — never sheds). This is the gate's *next
action*, not a new policy.

> **A real bug caught live, and fixed at the root.** The first implementation predicted with the
> *current* rate (`tput_ewma`). Below capacity it over-sheds: at R=2 it shed **73** requests and
> cut achieved 1.74→1.14. Root cause: arrival-limited rate ≠ drain rate. Fix: predict at
> `best_tput` (capacity). Re-run: R=2 shed dropped to **4** and achieved returned to ~queue level
> (transparent). This is the "make the bad state impossible" principle — the predictor now uses the
> physically-correct rate.

### 3.3 Code

- `sglang_omni/pipeline/admission.py` — `AdmissionRejected` (exception = deliberate drop, *not* an
  error); the `slo_s` param; the shed check at the top of `AdaptiveGate.on_submit`
  (`(inflight+waiters+1)/best_tput > slo_s ⇒ raise`).
- `sglang_omni/pipeline/coordinator.py` — `_submit_request` wraps `on_submit` in
  `try/except AdmissionRejected`: undo the request's tracking (it holds no slot, so `on_complete`
  is a no-op) and propagate (don't mark FAILED — it's a deliberate drop).
- `sglang_omni/serve/openai_api.py` — maps `AdmissionRejected` → **HTTP 429** on every path used:
  chat non-stream, speech-audio, speech non-stream, transcription (all `await`ed), and the
  **chat-stream path restructured to peek the first chunk** so the shed becomes a 429 *before* the
  `StreamingResponse`'s 200 headers are sent.
- `dev/testbed/testbed.py` — client `_err` sets `policy_rejected=True` on status 429 (counted as
  *shed*, not error); the streaming `omni_chat` client checks `status != 200` before iterating.

**Tests.** `tests/unit_test/test_admission.py` — `sheds-when-predicted-misses-SLO`,
`fail-open-no-shed` (no `slo_s` / no capacity estimate ⇒ never sheds).

### 3.4 Results — goodput protection across models

**Where.** `RESULTS.md` §6 "Deadline-aware shedding"; runs `dev/testbed/results/_shed_*` (3-arm:
A=no-policy, B0=gate-queue, B1=gate-shed — except TTS, whose B0 arm failed to launch (`code -9`
under load), so its gate-queue column is from a prior valid run; queue-only `slo_s=None` is
unaffected by the shed-predictor, so it is comparable).

**What — goodput (on-time, in-window req/s) and total-latency p99** (corrected metric: in-window
population / `duration_s`, so goodput ≤ achieved; run `20260609_*`, 3 repeats with per-repeat
arrival seeds so ± is real Poisson variance):

| model | rate | no-policy goodput / p99 | queue goodput / p99 | **shed goodput / p99 / shed** |
|--|--|--|--|--|
| **TTS** slo 4s | R=2 (≤cap) | 0.47 / 56 s ᶜ | 1.56 / 20 s | 1.87 / **4.1 s** / 13 |
| | R=3 | 0.26 / 64 s ᶜ | 0.28 / 29 s | **2.12 / 4.3 s** / 82 |
| | R=4 | 0.11 / 73 s ᶜ | 0.25 / 21 s | **1.85 / 4.6 s** / 180 |
| | R=6 | 0.03 / 74 s ᶜ | 0.22 / 46 s | **1.83 / 4.4 s** / 400 |
| **ASR** slo 3s | R=8/16 (≤cap) | 8.1 / 15.2 | = | = / shed 0 |
| | R=24 | 5.6 (±8.6) | 11.6 | **17.3** / 171 |
| | R=32 | **0.0** | 3.4 | **6.1** / 1260 |
| **Higgs** slo 5s | R=6 (≤cap) | 6.06 | 6.03 | 6.04 / 0 |
| | R=12 (~cap) | 10.35 | 8.69 | 9.58 / 68 |
| | R=18 | 2.31 / 28 s | 2.11 / 33 s | **5.73 / 4.3 s** / 831 |
| | R=24 | 1.83 / 45 s | 1.83 / 46 s | **3.58 / 5.3 s** / 1503 |

ᶜ = backlog-contaminated (the collapsed no-policy arm can't drain in 45 s, so its baseline starts
dirty — now flagged, not hidden).

**Analysis.** The **cleanest, confound-free signal is total-latency p99**: shedding **pins p99 at
the SLO** (TTS ~4.3 s @ slo 4; Higgs ~4.3–5.3 s @ slo 5) while no-policy and queue-only blow up to
**28–74 s**. On goodput, three consistent tiers under overload — **shed > queue > no-policy** —
e.g. TTS R=3 2.12 vs 0.28 vs 0.26; ASR R=32 6.1 vs 3.4 vs 0.0; Higgs R=18 5.7 vs 2.1 vs 2.3. The
multiplier depends on the rate: where no-policy fully collapses (TTS R≥3, ASR R=32) the gain is
unbounded (→0); elsewhere ~1.5–8× over queue-only. **Transparent at/below capacity** (ASR R=8/16
identical, shed 0; Higgs R=6). One honest wrinkle: at ~capacity (Higgs R=12) shedding slightly
*reduces* goodput (9.58 vs 10.35) — it sheds a few requests that would just barely have made it;
the crossover to a clear win is just above capacity (R=18+). A surfaced insight: queue-only is
*inconsistent* — it helps ASR but **hurts** Higgs (clamping a batching model adds latency for no
gain) — whereas shedding wins because it bounds latency **regardless of where the wait sits**.

**Honest scope.** Corrected goodput magnitudes are lower than the earlier (over-counted) metric;
the per-repeat seeds expose real metastable-knee variance (e.g. ASR R=24 ±8.6 — the no-policy arm
is bimodal there). The table above is the *total*-latency shed (TTS/ASR/Higgs); ttfa-SLO shedding
is below.

### Deadline shedding on a time-to-first-token SLO — ATTEMPTED, ABANDONED

A `slo_metric="ttfa"` variant (an `on_first_token` hook + a `queue_wait + ttft_ewma` predictor)
was implemented and unit-tested, but **abandoned and reverted** because it does not work and the
limitation is fundamental, not fixable by tuning:

- **The gate can't bound a *downstream-modality* SLO.** It only observes what flows through the
  coordinator stream — for omni-speech that's the first **text** token (thinker), whereas the SLO
  is on first **audio**, produced later by the *decoupled* downstream talker→code2wav stages.
  Shedding fired but couldn't bound the audio ttfa (shed arm still ~17–21s ttfa-p95 at overload).
- It would be sound for *single-modality* streaming (omni-understand, text-only — first content
  token *is* the SLO token), but that clean validation was also blocked by the omni-coloc server's
  repeated startup crashes / degraded loads on this shared box.

Net: a **negative result** — no demonstrated improvement on any scenario; the code was reverted.
The robust goodput protection is the *total*-latency shed above. Protecting a downstream-modality
SLO would need a downstream first-token signal — not pursued.

---

## 4. Methodology — why the numbers are trustworthy

Five testbed properties make the open-loop paired A/B valid; each fixed a confound found during
validation. **Code:** `dev/testbed/testbed.py`; **detail:** `RESULTS.md` §3 + "Methodology
hardening".

- **Open-loop driver** (`_drive_openloop`) — Poisson arrivals decoupled from completion; metrics
  offered/achieved rps, ttfa/total p50/p95/p99, **goodput@SLO**, shed-rate. (Closed-loop self-clocks
  at `concurrency` and *cannot* exercise an admission policy — it's biased against them.)
- **Seeded arrivals** — paired arms see the *same* arrival sequence ⇒ identical offered load
  (removes the independent-Poisson offered-mismatch confound).
- **Achieved = in-window completions** (not `ok/wall`) — `wall` is inflated by the post-window
  drain of a collapsed arm, which would flatter the policy.
- **Closed-loop warmup** (`_warmup_closedloop`) — bounded in-flight saturates the controller's
  one-time cliff discovery *without* an unbounded queue (an open-loop saturating warmup built a
  499-deep backlog → every measured rate timed out to 0/0).
- **Inter-point drain-to-idle** (`_wait_server_idle`) — each rate measures from a clean queue (else
  one rate's leftover backlog contaminates the next: omni p99 22→8.6 s once fixed).
- **Process-group SIGKILL** (`_kill_servers`) — the omni server's `spawn_main` stage children
  survived a parent-only kill, orphaning 235 GB and OOM-blocking the *next* paired arm (silently
  confounding arm B in earlier runs). Now the whole setsid group is always SIGKILLed.

**Known limitations** (in `RESULTS.md`, clearly separated from the policy — the policy itself is
fail-open, NoOp-default, cancellation-safe, queue-or-shed-only, and has **no crash/hang/throughput
cost**): open-loop arrival generator starves under collapse (offered drops — conservative for the
policy); drain-cap can't clear a deeply-collapsed no-policy queue; shared-box temporal confound;
and an *environment* note (omni coloc cold-Triton-cache startup is fragile — unrelated to any
policy).

---

## 5. Design principles (the through-line)

1. **One evolving policy, never a zoo.** `NoOp ⊂ FifoGate ⊂ AdaptiveGate ⊂ AdaptiveGate+shed`, each
   a degenerate case of the next. No selection, no fallback.
2. **Make the bad state impossible by construction**, never a band-aid that shrinks a symptom.
   Every controller invariant and the shed predictor are root fixes (e.g. fail-OPEN so the worst
   case is "inactive", not "throttling"; capacity-based shed so below-capacity load can't over-shed).
3. **Observability-driven.** The `[ADMGATE]` trace turned invisible steady-state failures (clump
   explosion, knee drift, demand-drop collapse) into things you can *see and fix* — and turned a
   noisy-box "regression" into a provable confound.
4. **Throughput, then goodput.** The gate keeps the *server* off its cliff; shedding keeps *useful
   work* (on-time completions) high under arbitrary overload. Together they are the cross-stage
   admission+deadline scheduling tier the framework left fire-and-forget.

---

## 6. File / result index

| concern | code | results |
|--|--|--|
| seam | `sglang_omni/pipeline/admission.py` (protocol, NoOp, resolve), `pipeline/coordinator.py`, `config/schema.py`, `pipeline/mp_runner.py` | NoOp ≡ fire-and-forget (unit test + equal-achieved omni) |
| static cap | `admission.py` `FifoGate` | `RESULTS.md` §6 Policy #1/#1b |
| adaptive controller | `admission.py` `AdaptiveGate._observe / on_submit / on_complete` | `RESULTS.md` §6; `results/<ts>/`; `[ADMGATE]` traces |
| deadline shedding | `admission.py` (`AdmissionRejected`, `slo_s`, shed check); `coordinator.py`; `serve/openai_api.py` (429 + chat-stream peek); `testbed.py` (`_err`) | `RESULTS.md` §6 "shedding"; `results/_shed_*` |
| testbed / methodology | `dev/testbed/testbed.py`, `scenarios.yaml` | `RESULTS.md` §3 + "Methodology hardening" |
| tests | `tests/unit_test/test_admission.py` (14) | all pass |
