"""Cross-stage admission / scheduling policy seam for the pipeline coordinator.

The coordinator's default behavior is fire-and-forget: every request is submitted to the
entry stage the instant it arrives, with no admission control, priority, deadline, or
load-aware gating (see ``Coordinator._submit_request``). The framework externalizes
*per-stage* policy (route_fn / wait_for_fn / merge_fn / placement_policy) but has no seam
for *cross-stage* admission decisions.

This module adds that seam in the same import-string-policy style. A policy is attached via
``PipelineConfig.admission_policy`` (dotted path to a class or zero-arg callable). The
default is ``NoOpAdmission``, which admits immediately and is byte-for-byte equivalent to the
current fire-and-forget path — so wiring the seam in changes nothing until a real policy is
configured.

Implement a policy by providing ``async on_submit(ctx)`` (may await to gate/delay/reorder
before the request enters the entry stage) and ``on_complete(request_id)`` (release any
in-flight accounting when the request leaves the pipeline).
"""
from __future__ import annotations

import asyncio
import collections
import os
import time
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


class AdmissionRejected(Exception):
    """Raised by a policy's ``on_submit`` to SHED a request (refuse admission) — distinct from
    an error: the request was deliberately dropped (e.g. it could not meet its deadline). The
    coordinator surfaces it as a shed (HTTP 429), not a 5xx failure."""


@dataclass(frozen=True)
class AdmissionContext:
    """Read-only snapshot handed to a policy at submit time."""

    request_id: str
    request: Any
    entry_stage: str
    inflight: int  # requests currently RUNNING in the pipeline (excludes this one)


@runtime_checkable
class AdmissionPolicy(Protocol):
    """Cross-stage admission policy. Attached at the coordinator; one instance per server."""

    async def on_submit(self, ctx: AdmissionContext) -> None:
        """Called before a request enters the entry stage.

        May ``await`` to gate (limit in-flight), delay (rate/deadline pacing), or reorder
        (priority) admission. Returning admits the request. Raising rejects it (the
        coordinator surfaces the error to the caller).
        """
        ...

    def on_complete(self, request_id: str) -> None:
        """Called when a request leaves the pipeline (success, error, or abort).

        Use to release any accounting taken in ``on_submit`` (e.g. an in-flight slot).
        Must be idempotent and tolerate ids it never admitted.
        """
        ...


class NoOpAdmission:
    """Default policy: admit immediately. Preserves the fire-and-forget behavior exactly."""

    async def on_submit(self, ctx: AdmissionContext) -> None:  # noqa: D401
        return

    def on_complete(self, request_id: str) -> None:
        return


class AdaptiveGate:
    """In-flight admission limiter with an optional throughput-driven controller + shedding.

    Maintains a limit ``L`` on concurrently-admitted requests, enforced FIFO (excess wait).
    This is the ONE cross-stage admission policy; the static cap (``FifoGate``) is its
    degenerate ``adapt=False`` case — there is no separate policy to fall back to or select.

    When adapting, ``L`` floats in ``[min_limit, max_limit]`` driven by a **throughput**
    hill-climb: probe ``L`` upward; back off to the best-known operating point only when
    achieved throughput **collapses** relative to the best seen at a lower ``L``. This targets
    *capacity* directly, so it generalizes across pipelines with very different load curves:
      - a pipeline that self-collapses past some concurrency (e.g. Qwen3-TTS past ~8) — raising
        L past the knee drops throughput → snap back to the knee;
      - a pipeline whose throughput merely *plateaus* with concurrency (e.g. Qwen3-Omni, which
        batches efficiently to ~16) — raising L never drops throughput → L rises freely (≈ no
        gating, so the policy never hurts it).
    A latency-based controller would instead find the *latency* knee, which for a plateauing
    model sits below the throughput knee and needlessly throttles it — hence throughput, not
    latency, is the signal. ONE general parameter set, no per-model number. The signal is free
    at the seam: completions per unit time at the current L.

    Config:
      AdaptiveGate()                      -> adaptive, L in [min_limit, max_limit]
      AdaptiveGate(limit=N, adapt=False)  -> static cap N  (== FifoGate(N))
      AdaptiveGate(adapt=False)           -> passthrough (no gating)
    """

    def __init__(
        self,
        limit: int | None = None,
        min_limit: int = 8,
        max_limit: int = 64,
        adapt: bool = True,
        window_s: float = 3.0,
        up_factor: float = 1.15,
        collapse_frac: float = 0.7,
        ceiling_relax: float = 1.002,
        best_decay: float = 0.999,
        ewma_alpha: float = 0.3,
        sat_frac: float = 0.75,
        outlier_cap: float = 2.0,
        slo_s: float | None = None,
    ):
        self._adapt = bool(adapt)
        # min_limit is a sane "minimum useful concurrency" FLOOR: the gate never admits fewer
        # than this, so an uncontended system (low-load in-flight < floor) is never throttled.
        self.min_limit = float(min_limit)
        self.max_limit = float(max_limit)
        self.window_s = float(window_s)            # FIXED wall-clock control window. Throughput is
        #   sampled as completions/elapsed over >= window_s of real time — NOT per N completions.
        #   Batched serving finishes many sequences in one decode step (a "clump"), so a
        #   completion-count window can close with dt~milliseconds and raw=n/dt explodes to
        #   thousands, poisoning best_tput into a permanent spurious-collapse oscillation. A
        #   fixed time window bounds raw = n/elapsed by construction (dt >= window_s): a clump
        #   just contributes its count to the window it lands in.
        self.best_decay = float(best_decay)        # best_tput decays VERY slowly so the good
        #   operating point is remembered for the long run (a fast decay forgets it and lets L
        #   drift up off the knee — the v1 regression). Clump-robustness comes from the fixed
        #   window + outlier_cap below, NOT from fast decay.
        self.ewma_alpha = float(ewma_alpha)        # throughput EWMA weight (smaller = smoother)
        self.sat_frac = float(sat_frac)            # the limit is "binding" (throughput is
        #   informative about it) only when peak in-flight this window reached >= sat_frac*limit.
        #   Using the in-flight LEVEL — not "did anything queue" — correctly distinguishes a real
        #   cliff (running near the limit, throughput dropped) from a mere demand drop (gate slack,
        #   throughput low only because offered load fell): only the former should lower the limit.
        self.outlier_cap = float(outlier_cap)      # reject a window whose rate exceeds this * the
        #   smoothed estimate as a measurement artifact (a completion clump): real throughput can't
        #   jump that much in one window, so capping it keeps best_tput from being poisoned.
        self.slo_s = None if slo_s is None else float(slo_s)  # per-request latency deadline (s).
        #   When set, SHED (reject) a request whose predicted time-in-system already exceeds the
        #   deadline, instead of queueing it to certainly miss SLO — this protects goodput under
        #   overload (the gate alone restores throughput but the wait queue still grows). None =
        #   pure queueing (no shedding), so AdaptiveGate(slo_s=None) == the queue-only gate.
        self.up_factor = float(up_factor)           # multiplicative probe-up. SMALL on purpose:
        #   discovering a cliff online (vs a plateau) is only possible by stepping past it and
        #   observing whether throughput holds (plateau -> keep rising to max) or drops (cliff
        #   -> snap back + remember the ceiling). That one-step overshoot is fundamental — no
        #   prior-free controller can avoid touching the cliff once. A small factor bounds the
        #   overshoot's DEPTH (e.g. 8->9.2->10.6->12 instead of 8->10->13->17->23), so the
        #   one-time startup transient is mild rather than a deep collapse. Generality's price.
        self.collapse_frac = float(collapse_frac)   # tput < frac*best => collapse
        self.ceiling_relax = float(ceiling_relax)   # slowly raise the ceiling so capacity that
        #                                             later IMPROVES can be re-discovered (M2).
        if self._adapt:
            self._passthrough = False
            # ALWAYS start at the floor and probe UP (a static initial `limit` for an adaptive
            # gate would risk recording a collapsed L as the persistent best — M1).
            self._limit = self.min_limit
        else:
            self._passthrough = limit is None or limit <= 0
            self._limit = float(limit) if limit and limit > 0 else 0.0
        self._inflight = 0
        self._admit: dict[str, float] = {}
        # FIFO queue of (request_id, future) for slow-path waiters, plus a request_id->future
        # index so a request that is ABORTED while still queued can be found and removed (else
        # its waiter future leaks and _grant later hands it a slot that is never released).
        self._waiters: collections.deque[tuple[str, asyncio.Future]] = collections.deque()
        self._waiting: dict[str, asyncio.Future] = {}
        self._log = bool(os.environ.get("SGLANG_OMNI_ADMGATE_LOG"))  # opt-in [ADMGATE] trace
        # throughput-control state
        self._win_n = 0
        self._win_t0: float | None = None
        self._best_tput = 0.0          # best observed throughput (decays toward current, below)
        self._best_limit = self._limit
        self._ceiling = self.max_limit  # lowest L observed to collapse (none yet)
        self._tput_ewma: float | None = None  # smoothed throughput (noise-robust collapse test)
        self._steps = 0                         # control-step counter (observability)
        self._win_inflight_peak = 0             # peak in-flight seen this window (vs limit ->
        #                                         "is the limit binding?"; see sat_frac)

    @property
    def limit(self) -> int:
        """Current admission limit (for observability / tests)."""
        return self._cap()

    def _cap(self) -> int:
        return 10**9 if self._passthrough else max(1, int(self._limit))

    async def on_submit(self, ctx: AdmissionContext) -> None:
        if self._passthrough:
            return
        # Deadline-aware shedding (fail-open). If admitting this request would predict a
        # completion past its deadline, refuse it now so capacity goes to requests that CAN
        # still meet SLO. By Little's law the time-in-system of a newly-arriving request is
        # ~ (current in-system + 1) / DRAIN RATE, where in-system = in-flight + queued. The drain
        # rate is the service CAPACITY (best_tput), NOT the current throughput: below capacity the
        # current rate is arrival-limited (e.g. 1.7/s when offered<capacity) and would overestimate
        # the wait and over-shed; the queue actually drains at capacity. Only acts when slo_s is
        # set AND a capacity estimate exists (adaptive) AND the prediction exceeds the deadline —
        # so at low load (short queue) nothing is shed, and a static FifoGate never sheds.
        if self.slo_s is not None and self._best_tput > 0:
            in_system = self._inflight + len(self._waiters)
            predicted = (in_system + 1) / self._best_tput
            if predicted > self.slo_s:
                raise AdmissionRejected(
                    f"predicted {predicted:.1f}s > slo {self.slo_s:.1f}s "
                    f"(in_system={in_system}, cap={self._best_tput:.2f}/s)")
        # Fast path: a slot is free AND nobody is already queued (preserve FIFO fairness).
        if self._inflight < self._cap() and not self._waiters:
            self._inflight += 1
            self._win_inflight_peak = max(self._win_inflight_peak, self._inflight)
            self._admit[ctx.request_id] = time.perf_counter()
            return
        # Slow path: queue FIFO. The waker (_grant) increments _inflight, records _admit, and
        # resolves our future — i.e. it hands us the slot atomically; we never re-check or
        # re-queue (true FIFO). The request_id->future index lets on_complete cancel us if the
        # request is aborted while still queued.
        rid = ctx.request_id
        fut = asyncio.get_running_loop().create_future()
        self._waiters.append((rid, fut))
        self._waiting[rid] = fut
        try:
            await fut
        except BaseException:  # cancelled/timed-out/aborted while waiting OR just after grant
            if self._waiting.pop(rid, None) is not None:   # still queued: leave the FIFO
                try:
                    self._waiters.remove((rid, fut))
                except ValueError:
                    pass
            elif self._admit.pop(rid, None) is not None:   # granted (slot+admit set) but abandoned
                self._inflight = max(0, self._inflight - 1)
                self._grant()
            raise
        # Granted: _grant already set _admit[rid] and incremented _inflight. Nothing to do.

    def _grant(self) -> None:
        """Hand free slots to FIFO waiters (front first); skip waiters that already left."""
        while self._waiters and self._inflight < self._cap():
            rid, fut = self._waiters.popleft()
            self._waiting.pop(rid, None)
            if fut.cancelled() or fut.done():
                continue                     # waiter gone — do NOT consume a slot for it
            self._inflight += 1
            self._win_inflight_peak = max(self._win_inflight_peak, self._inflight)
            self._admit[rid] = time.perf_counter()   # admit atomically with the grant
            fut.set_result(None)

    def on_complete(self, request_id: str) -> None:
        # Admitted (running) request leaving the pipeline: release its slot.
        if self._admit.pop(request_id, None) is not None:
            self._inflight = max(0, self._inflight - 1)
            if self._adapt:
                self._observe(time.perf_counter())
            self._grant()
            return
        # Aborted while still QUEUED (never admitted, holds no slot): cancel its waiter so the
        # blocked on_submit unblocks, and drop it from the FIFO. Without this the waiter future
        # leaks and _grant would later hand it a slot that is never released (gate ratchets shut).
        fut = self._waiting.pop(request_id, None)
        if fut is not None:
            try:
                self._waiters.remove((request_id, fut))
            except ValueError:
                pass
            if not fut.done():
                fut.cancel()
        # else: unknown id -> idempotent no-op.

    def _observe(self, now: float) -> None:
        """Throughput-gradient control, fail-open. One step per ``window_s`` of wall time.

        Default is the ceiling (no gating); the limit is only reduced on *evidence* that a
        lower concurrency yields higher throughput (collapse). The invariants that make the bad
        states impossible by construction (see DESIGN.md §2.3 for the full set; the four enforced
        in this step plus outlier-cap, inflight-saturation gating, and a small probe step):
          * fail-open start + reduce-only-on-evidence  -> never throttles an uncontended system;
          * FIXED-time throughput window               -> a completion clump (many seqs finishing
                                                          in one decode step) can't divide by a
                                                          ~0 dt and poison the estimate;
          * best decays toward the estimate            -> a transiently-inflated reading can't
                                                          pin the limit forever (self-corrects);
          * ceiling memory (never probe above a known  -> can't oscillate back into the cliff.
            collapse point)
        """
        if self._win_t0 is None:
            self._win_t0 = now
        self._win_n += 1
        # The window's PEAK concurrency is captured on every ADMISSION (fast path + _grant), so it
        # reflects the true binding level; this post-completion sample is only a floor and never
        # exceeds it (inflight is already decremented here).
        if self._inflight > self._win_inflight_peak:
            self._win_inflight_peak = self._inflight
        # Close the window only after window_s of WALL TIME (not after a completion count): a
        # clump of completions in one decode step would otherwise close a window with dt~ms and
        # raw=n/dt would explode. With a fixed window, dt >= window_s, so raw is bounded.
        if (now - self._win_t0) < self.window_s:
            return
        dt = now - self._win_t0
        n, self._win_n, self._win_t0 = self._win_n, 0, now
        peak, self._win_inflight_peak = self._win_inflight_peak, 0
        if dt <= 0:
            return
        raw = n / dt
        # Was the limit BINDING this window? Only if peak in-flight got near the limit. If not,
        # the system is demand-limited (running below the limit) and throughput reflects offered
        # load, not the limit — so a drop must NOT be read as collapse. This distinguishes a real
        # cliff (running AT the limit, throughput dropped) from a demand drop (warmup ended /
        # arrival fell, gate slack): only the former is evidence about the limit.
        saturated = peak >= self.sat_frac * self._cap()
        # Reject an implausible spike (a completion clump that slipped through the time window) as
        # a measurement artifact — real throughput can't jump > outlier_cap× in one window — so it
        # can't poison best_tput. (Genuine probe-up gains are ~15%/step, far under the cap.)
        if self._tput_ewma is not None:
            raw = min(raw, self.outlier_cap * self._tput_ewma)
        # EWMA-smooth the throughput so a single noisy window (±10–25% box noise) can't dominate.
        self._tput_ewma = raw if self._tput_ewma is None else (
            (1.0 - self.ewma_alpha) * self._tput_ewma + self.ewma_alpha * raw)
        tput = self._tput_ewma
        # Best operating point: persistent but DECAYS toward the current estimate, so a
        # transiently-inflated reading (a residual clump that slipped through) self-corrects
        # within ~tens of steps instead of pinning the limit forever. This is safe — and not a
        # regression to v1's stuck-collapse — because the ceiling memory below independently
        # prevents re-entering a known cliff even if best is briefly forgotten.
        self._best_tput *= self.best_decay
        if tput > self._best_tput:
            self._best_tput, self._best_limit = tput, self._limit
        # Slowly relax the ceiling so a cliff that later moves up can be re-explored (M2).
        self._ceiling = min(self.max_limit, self._ceiling * self.ceiling_relax)
        # collapse: while the limit was BINDING, throughput fell well below the known-good at a
        # higher L => raising L overshot a cliff; remember it as a ceiling and snap back. The
        # `saturated` guard is essential: without it, a demand drop (which also lowers throughput
        # but leaves the gate slack) would be misread as a cliff and clamp L spuriously.
        if (saturated and self._best_tput > 0 and tput < self._best_tput * self.collapse_frac
                and self._limit > self._best_limit):
            self._ceiling = min(self._ceiling, self._limit)
            self._limit = max(self.min_limit, self._best_limit)
        else:
            # No collapse evidence (or demand-limited): probe UP toward more concurrency, never
            # up to/over a known cliff (ceiling). Probing up while slack is correct — a limit
            # that isn't binding should relax toward max so it won't throttle when load returns.
            nxt = self._limit * self.up_factor
            if nxt < self._ceiling:
                self._limit = min(self.max_limit, nxt)
            # at/over the ceiling (or max): hold — don't re-enter the cliff.
        self._steps += 1
        # Observability: emit the controller's state each step so the limit's trajectory is
        # directly inspectable (is the gate inactive at L=max, or clamped at a discovered knee?).
        # Opt-in (SGLANG_OMNI_ADMGATE_LOG) — off by default so library use doesn't spam stdout.
        if self._log:
            print(f"[ADMGATE] step={self._steps} limit={self._limit:.1f} best_limit={self._best_limit:.1f} "
                  f"ceiling={self._ceiling:.1f} inflight={self._inflight} sat={int(saturated)} "
                  f"tput={tput:.2f} best_tput={self._best_tput:.2f}", flush=True)

    def stats(self) -> dict:
        """Current controller state (observability / tests)."""
        return {
            "limit": round(self._limit, 2), "best_limit": round(self._best_limit, 2),
            "ceiling": round(self._ceiling, 2), "inflight": self._inflight,
            "best_tput": round(self._best_tput, 3), "steps": self._steps,
            "adapt": self._adapt, "passthrough": self._passthrough,
        }


class FifoGate(AdaptiveGate):
    """Static in-flight cap — the degenerate (non-adaptive) case of :class:`AdaptiveGate`.

    ``FifoGate(N) ≡ AdaptiveGate(limit=N, adapt=False)``; ``max_inflight <= 0`` = passthrough.
    Kept as a named policy for the static baseline experiments.
    """

    def __init__(self, max_inflight: int = 0):
        super().__init__(limit=max_inflight, adapt=False)


def resolve_admission_policy(spec: Any, args: dict | None = None) -> AdmissionPolicy | None:
    """Resolve a config value (dotted path, class, instance, or None) to a policy instance.

    ``None`` -> None (coordinator skips the hook entirely, i.e. fire-and-forget).
    A dotted string is imported; if it resolves to a class it is instantiated with ``args``
    (keyword arguments) — e.g. ``FifoGate`` with ``{"max_inflight": 8}``.
    """
    if spec is None:
        return None
    if isinstance(spec, str):
        from sglang_omni.utils.imports import import_string

        obj = import_string(spec)
    else:
        obj = spec
    if isinstance(obj, type):
        obj = obj(**(args or {}))
    return obj


def admission_policy_from_env() -> AdmissionPolicy | None:
    """Resolve a policy from the environment, for A/B experiments without editing config.

    ``SGLANG_OMNI_ADMISSION_POLICY`` = dotted path to a policy class/callable.
    ``SGLANG_OMNI_ADMISSION_ARGS``   = JSON object of keyword args (optional).
    Returns None when unset. Config-declared ``admission_policy`` takes precedence.
    """
    import json
    import os

    spec = os.environ.get("SGLANG_OMNI_ADMISSION_POLICY")
    if not spec:
        return None
    raw = os.environ.get("SGLANG_OMNI_ADMISSION_ARGS")
    args = json.loads(raw) if raw else None
    return resolve_admission_policy(spec, args)
