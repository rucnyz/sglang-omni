# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the cross-stage admission policy (sglang_omni/pipeline/admission.py).

Covers the gate mechanics (FIFO ordering, cancellation-safety, idempotency), the AdaptiveGate
throughput controller's convergence on synthetic load curves, and deadline shedding.
"""
import asyncio
import random

import pytest

from sglang_omni.pipeline.admission import (
    AdaptiveGate,
    AdmissionContext,
    AdmissionRejected,
    FifoGate,
    NoOpAdmission,
    resolve_admission_policy,
)


def _ctx(i):
    return AdmissionContext(request_id=str(i), request=None, entry_stage="e", inflight=0)


# --------------------------------------------------------------------------- #
# Gate mechanics
# --------------------------------------------------------------------------- #
def test_noop_admits_immediately():
    p = NoOpAdmission()
    asyncio.run(p.on_submit(_ctx(1)))
    p.on_complete("1")  # no error


def test_fifo_static_cap_blocks_and_releases():
    async def run():
        g = FifoGate(2)
        assert g.limit == 2 and g._adapt is False
        await g.on_submit(_ctx("a"))
        await g.on_submit(_ctx("b"))
        third = asyncio.create_task(g.on_submit(_ctx("c")))
        await asyncio.sleep(0.02)
        assert not third.done(), "3rd must block at cap 2"
        g.on_complete("a")
        await asyncio.wait_for(third, 1.0)
        assert third.done()
    asyncio.run(run())


def test_gate_fifo_order():
    """Waiters are granted in strict arrival order (a freed slot must not re-queue to the tail)."""
    async def run():
        g = FifoGate(1)
        order = []
        await g.on_submit(_ctx("a"))

        async def w(name):
            await g.on_submit(_ctx(name))
            order.append(name)

        tb = asyncio.create_task(w("b"))
        await asyncio.sleep(0.01)
        tc = asyncio.create_task(w("c"))
        await asyncio.sleep(0.01)
        g.on_complete("a")
        await asyncio.sleep(0.01)
        g.on_complete("b")
        await asyncio.gather(tb, tc)
        assert order == ["b", "c"], order
    asyncio.run(run())


def test_gate_cancellation_does_not_leak_or_wedge():
    """A waiter cancelled while queued must leave the deque and not consume a slot."""
    async def run():
        g = FifoGate(2)
        await g.on_submit(_ctx("a"))
        await g.on_submit(_ctx("b"))
        third = asyncio.create_task(g.on_submit(_ctx("c")))
        await asyncio.sleep(0.02)
        assert len(g._waiters) == 1
        third.cancel()
        with pytest.raises(asyncio.CancelledError):
            await third
        await asyncio.sleep(0.02)
        assert len(g._waiters) == 0, "cancelled waiter leaked in deque"
        # gate must not be wedged: a freed slot admits a new request
        g.on_complete("a")
        fourth = asyncio.create_task(g.on_submit(_ctx("d")))
        await asyncio.wait_for(fourth, 1.0)
        assert fourth.done()
    asyncio.run(run())


def test_grant_skips_cancelled_waiter_without_consuming_slot():
    """A waiter whose future was cancelled while still in the deque must be skipped by _grant
    without consuming the freed slot — the slot goes to the next live waiter."""
    async def run():
        g = FifoGate(1)
        await g.on_submit(_ctx("a"))                       # takes the single slot
        loop = asyncio.get_running_loop()
        dead = loop.create_future(); dead.cancel()          # a cancelled waiter still enqueued
        live = loop.create_future()
        g._waiters.append(("dead", dead)); g._waiting["dead"] = dead
        g._waiters.append(("live", live)); g._waiting["live"] = live
        g.on_complete("a")                                  # frees the slot -> _grant runs
        await asyncio.wait_for(live, 1.0)
        assert live.done() and not live.cancelled(), "live waiter must get the slot"
        assert g._inflight == 1, f"dead waiter consumed a slot: inflight={g._inflight}"
        assert not g._waiting, "waiters index not drained"
    asyncio.run(run())


def test_queued_request_abort_does_not_leak_slot():
    """Aborting a request still QUEUED at the gate (coordinator calls on_complete before it was
    granted) must cancel its waiter and remove it from the FIFO — without leaking a slot. Else
    _grant later hands the stale future a slot that is never released and the gate ratchets shut."""
    async def run():
        g = FifoGate(1)
        await g.on_submit(_ctx("a"))                       # takes the 1 slot
        tb = asyncio.create_task(g.on_submit(_ctx("b")))   # queues (slow path)
        await asyncio.sleep(0.02)
        assert len(g._waiters) == 1 and "b" in g._waiting
        g.on_complete("b")                                  # abort b while queued
        with pytest.raises(asyncio.CancelledError):
            await tb
        assert len(g._waiters) == 0 and "b" not in g._waiting, "queued waiter leaked"
        # gate not wedged: free a's slot, a brand-new request must get it (no leaked inflight)
        g.on_complete("a")
        await asyncio.wait_for(g.on_submit(_ctx("c")), 1.0)
        assert g._inflight == 1, f"slot leaked: inflight={g._inflight}"
    asyncio.run(run())


def test_on_complete_idempotent_and_unknown_id():
    async def run():
        g = FifoGate(4)
        await g.on_submit(_ctx("a"))
        g.on_complete("unknown")   # never admitted -> no-op
        g.on_complete("a")
        g.on_complete("a")          # double -> no-op
        assert g._inflight == 0
    asyncio.run(run())


def test_fifo_is_adaptive_static_special_case():
    assert FifoGate(8).limit == 8
    assert FifoGate(8)._adapt is False
    assert FifoGate(0)._passthrough is True       # <=0 disables gating
    assert isinstance(resolve_admission_policy(
        "sglang_omni.pipeline.admission.AdaptiveGate"), AdaptiveGate)


# --------------------------------------------------------------------------- #
# AdaptiveGate throughput controller — closed-loop convergence on synthetic curves
# --------------------------------------------------------------------------- #
def _converge(cap, *, arrival=None, ctrl_steps=80, seed=0):
    """Drive _observe in a closed loop until `ctrl_steps` control steps occur.

    completion rate = cap(L) (capped by `arrival`). The gate steps on a fixed WALL-CLOCK
    window, so we advance synthetic time and run until enough control steps have fired (rather
    than a fixed number of completions, which would scale with rate). Returns (tail_mean, gate,
    hist) so tests can assert the TRAJECTORY (probed up, recorded a collapse), not just the
    final level — a controller frozen at the floor would otherwise pass a loose lower bound.
    """
    rng = random.Random(seed)
    g = AdaptiveGate(min_limit=8, max_limit=64, window_s=3.0)
    t, hist = 0.0, []
    while g._steps < ctrl_steps:
        L = g.limit
        rate = cap(L)
        # The gate is SATURATED (limit binding) when capacity, not arrival, caps the rate;
        # demand-limited otherwise. The controller treats throughput drops as collapse only when
        # saturated, judged by peak in-flight vs the limit — so the harness sets in-flight to ~L
        # when saturated (running at the limit) and low when demand-limited (production maintains
        # _inflight via on_submit/on_complete; here _observe is driven directly).
        saturated = arrival is None or rate <= arrival
        if arrival is not None:
            rate = min(arrival, rate)
        rate = max(0.05, rate * (0.88 + 0.24 * rng.random()))  # ±12% noise
        t += 1.0 / rate
        g._inflight = int(round(L)) if saturated else 1
        g._observe(t)
        hist.append(L)
    tail = hist[int(len(hist) * 0.6):]
    return sum(tail) / len(tail), g, hist


def test_controller_finds_knee_on_sharp_cliff():
    cap = lambda L: 0.38 * L if L <= 8 else max(0.3, 3.04 * (8.0 / L) ** 3)
    mean, g, hist = _converge(cap)
    assert 6 <= mean <= 12
    # The controller must have ACTIVELY found the knee, not merely sat at the floor: it probed
    # above the floor and recorded a collapse (ceiling pulled below max). Guards against a
    # do-nothing gate passing the loose lower bound.
    assert max(hist) > 8, f"never probed above the floor: max L={max(hist)}"
    assert g.stats()["ceiling"] < 64, f"no collapse recorded: ceiling={g.stats()['ceiling']}"


def test_controller_clamps_on_gradual_rolloff():
    cap = lambda L: 0.38 * L if L <= 8 else max(0.3, 3.04 - (L - 8) * 0.4)
    mean, g, hist = _converge(cap)
    assert 6 <= mean <= 14
    assert max(hist) > 8 and g.stats()["ceiling"] < 64


def test_controller_does_not_throttle_a_plateau_model():
    cap = lambda L: min(2.05, 0.21 * L) if L <= 9 else min(2.05, 1.55 + 0.02 * L)
    mean, g, _ = _converge(cap)
    assert mean >= 20  # rises toward max — no needless throttling
    assert g.stats()["ceiling"] == 64  # never recorded a collapse on a plateau model


def test_controller_does_not_throttle_low_load():
    # arrival far below capacity: limit must rise high so the gate never binds (R=1 regression)
    assert _converge(lambda L: 50.0, arrival=1.0)[0] >= 20
    assert _converge(lambda L: 50.0, arrival=5.0)[0] >= 20


async def _shed_or(g, ctx):
    """Returns 'shed' if on_submit rejects, 'queued' if it blocks, else 'admitted'."""
    try:
        await asyncio.wait_for(g.on_submit(ctx), 0.05)
        return "admitted"
    except AdmissionRejected:
        return "shed"
    except asyncio.TimeoutError:
        return "queued"


def test_deadline_shedding_sheds_when_predicted_misses_slo():
    """slo_s set + predicted time-in-system ((in_system+1)/throughput) > slo => SHED; else admit."""
    async def run():
        g = AdaptiveGate(min_limit=8, max_limit=64, slo_s=2.0)
        g._best_tput = 4.0                       # capacity ~4 req/s established
        assert await _shed_or(g, _ctx("a")) == "admitted"   # (0+1)/4=0.25s < 2 -> admit
        g._inflight = 12                          # (12+1)/4=3.25s > 2 -> shed
        assert await _shed_or(g, _ctx("b")) == "shed"
    asyncio.run(run())


def test_deadline_shedding_admits_at_exact_boundary():
    """The shed test is strict (>): a request predicted to land EXACTLY on the deadline is
    admitted, not shed."""
    async def run():
        g = AdaptiveGate(min_limit=8, max_limit=64, slo_s=2.0)
        g._best_tput = 4.0
        g._inflight = 7                           # (7+1)/4 = 2.0 == slo -> admit (not > slo)
        assert await _shed_or(g, _ctx("a")) == "admitted"
    asyncio.run(run())


def test_no_shedding_failopen():
    """Fail-open: no slo_s, or no throughput estimate yet => never sheds (queues instead)."""
    async def run():
        g = AdaptiveGate(min_limit=8, max_limit=64, slo_s=None)  # no deadline
        g._best_tput, g._inflight = 4.0, 100
        assert await _shed_or(g, _ctx("a")) != "shed"
        g2 = AdaptiveGate(min_limit=8, max_limit=64, slo_s=2.0)  # deadline but no capacity estimate
        g2._inflight = 100                                       # _best_tput is 0
        assert await _shed_or(g2, _ctx("b")) != "shed"
    asyncio.run(run())


def test_deadline_shedding_predicts_from_capacity_not_current_rate():
    """The wait is predicted from service CAPACITY (best_tput), NOT the current arrival-limited
    rate (tput_ewma). A low current rate must not inflate the prediction and over-shed — a
    regression here (using tput_ewma) would shed requests that can comfortably meet SLO."""
    async def run():
        g = AdaptiveGate(min_limit=8, max_limit=64, slo_s=2.0)
        g._best_tput = 8.0     # capacity 8/s established
        g._tput_ewma = 1.0     # but currently only ~1/s served (offered load is low)
        g._inflight = 12       # (12+1)/8 = 1.625s < 2 via capacity -> NOT shed
        #                        (12+1)/1 = 13s   >> 2 if it wrongly used the current rate
        # (the request then queues at the in-flight gate since 12 > the L=8 floor; "not shed"
        # is the invariant under test, whether it ends up admitted or queued).
        assert await _shed_or(g, _ctx("a")) != "shed"
    asyncio.run(run())


def _drive_window(g, t, rate):
    """Advance synthetic time at `rate` completions/s until exactly one control window closes
    (g._steps increments); returns the new time."""
    step, s0 = 1.0 / rate, g._steps
    while g._steps == s0:
        t += step
        g._inflight = int(round(g.limit))
        g._observe(t)
    return t


def test_outlier_cap_clamps_a_spike_window():
    """A window whose raw completion rate exceeds outlier_cap x the smoothed estimate is clamped,
    so a completion spike that slips past the time window cannot poison best_tput. (The fixed
    time window is the primary clump defense; this is the secondary guard.)"""
    g = AdaptiveGate(min_limit=8, max_limit=64, window_s=3.0, outlier_cap=2.0, ewma_alpha=1.0)
    t = 0.0
    for _ in range(3):
        t = _drive_window(g, t, rate=2.0)        # establish a healthy ~2/s estimate
    healthy = g._best_tput
    assert 1.5 <= healthy <= 2.6, healthy
    t = _drive_window(g, t, rate=40.0)           # a 40/s SPIKE window
    assert g._best_tput <= g.outlier_cap * healthy + 0.5, \
        f"spike poisoned best_tput={g._best_tput} (should be clamped to ~{g.outlier_cap*healthy})"


def test_controller_holds_knee_over_long_run():
    """Over a LONG run the controller must keep L at the knee, not let it drift up off it.
    A fast best-decay would forget the good operating point within ~100 steps and accept a
    worse, higher L as 'best'; a near-persistent best (slow decay) holds the knee indefinitely."""
    cap = lambda L: 0.38 * L if L <= 8 else max(0.3, 3.04 * (8.0 / L) ** 3)
    mean, _, hist = _converge(cap, ctrl_steps=300)
    assert 6 <= mean <= 12
    # And the tail must not have drifted up off the knee over the long run.
    assert max(hist[len(hist) // 2:]) <= 14, "L drifted up off the knee in the second half"


def test_controller_survives_completion_clump():
    """A batch of completions finishing in ~0 wall time (one decode step) must NOT poison
    best_tput into thousands and pin the limit. A count-based window would divide n completions
    by a ~ms dt and explode the rate estimate; a fixed wall-clock window bounds raw = n/window_s
    by construction, so a clump just contributes its count to the window it lands in.
    """
    g = AdaptiveGate(min_limit=8, max_limit=64, window_s=3.0)
    t = 0.0

    def plateau(n, rate=2.0):  # constant capacity regardless of L (pure plateau model)
        nonlocal t
        for _ in range(n):
            t += 1.0 / rate
            g._inflight = int(round(g.limit))  # saturated: running at the limit (collapse IS
            g._observe(t)                       # armed — yet a clump must still not trigger it)

    plateau(120)                       # establish a healthy ~2/s operating point
    for _ in range(40):                # CLUMP: 40 completions within ~1 ms (one decode step)
        t += 2.5e-5
        g._observe(t)
    assert g._best_tput < 25.0, f"clump poisoned best_tput={g._best_tput} (n/dt explosion)"
    plateau(400)                       # plateau continues — there is no real collapse
    assert g.limit >= 20, f"controller pinned by the clump (limit={g.limit})"
