#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Phase 0 offline analyzer: per-stage GPU-time four-way decomposition.

Consumes the request-event JSONL files written by
``sglang_omni.profiler.event_recorder`` (``events_<stage>_<pid>.jsonl``) and
computes, for every stage, the bubble decomposition defined in
``PHASE0_PROBLEM.md``:

    wall = BUSY + STARVED + BLOCKED + IDLE
    bubble = STARVED + BLOCKED          # GPU idle *despite pipeline work*

Outputs (stdlib only; matplotlib PNG is best-effort optional):
  - M2: per-stage bubble% bar (text table)
  - M3: backpressure stall (absolute ms + % of stage window)
  - M1: ASCII swimlane over wall-clock (BUSY/STARVED/BLOCKED/IDLE)
  - JSON dump of all of the above

This script depends on FIVE instrumentation events (see INSTRUMENTATION.md):
  G1 BUSY     : "fwd_begin" / "fwd_end"   (request_id="__stage__", stage=S)
  G2 BLOCKED  : "bp_wait_begin"/"bp_wait_end" (request_id=rid, stage=sender S)
  G3 queue    : "q_sample"  (optional; not required for the decomposition)
plus the already-emitted lifecycle events:
  "request_admission" (admit) and "terminal_response"/"stage_complete" (done)
which give the in-flight-request curve used to split idle into STARVED vs IDLE.

Run ``python analyze_bubbles.py --selftest`` to verify the decomposition logic
on synthetic events (no sglang install required).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

# --- event-name contract ---------------------------------------------------
EV_FWD_BEGIN = "fwd_begin"        # G1
EV_FWD_END = "fwd_end"            # G1
EV_BP_BEGIN = "bp_wait_begin"     # G2
EV_BP_END = "bp_wait_end"        # G2
EV_ADMIT = "request_admission"    # existing (coordinator)
EV_TERMINAL = ("terminal_response", "stage_complete")  # existing
STAGE_RID = "__stage__"           # synthetic request_id for stage-scoped events

NS_PER_MS = 1e6


# --- loading ---------------------------------------------------------------
def iter_events(source: str | Path | Iterable[str | Path]):
    paths: list[Path] = []
    srcs = [source] if isinstance(source, (str, Path)) else list(source)
    for raw in srcs:
        p = Path(raw).expanduser()
        if p.is_dir():
            paths.extend(sorted(p.glob("events_*.jsonl")))
        elif p.is_file():
            paths.append(p)
        else:
            print(f"WARN: skipping non-existent path {p}", file=sys.stderr)
    for path in paths:
        with path.open("r", encoding="utf-8") as fp:
            for ln, line in enumerate(fp, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    print(f"WARN: bad line {ln} in {path}", file=sys.stderr)


# --- interval helpers ------------------------------------------------------
def _match_intervals(events: list[dict], begin_name: str, end_name: str,
                     key_fn) -> dict[Any, list[tuple[int, int]]]:
    """Stack-match begin/end events into intervals, grouped by key_fn(ev)."""
    stacks: dict[Any, list[int]] = defaultdict(list)
    out: dict[Any, list[tuple[int, int]]] = defaultdict(list)
    for ev in events:
        name = ev.get("event_name")
        if name == begin_name:
            stacks[key_fn(ev)].append(int(ev["timestamp_ns"]))
        elif name == end_name:
            k = key_fn(ev)
            if stacks[k]:
                b = stacks[k].pop()
                out[k].append((b, int(ev["timestamp_ns"])))
    return out


def _merge(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    s = sorted(intervals)
    merged = [list(s[0])]
    for b, e in s[1:]:
        if b <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([b, e])
    return [(b, e) for b, e in merged]


def _total(intervals: list[tuple[int, int]]) -> int:
    return sum(e - b for b, e in intervals)


def _complement(intervals: list[tuple[int, int]], lo: int, hi: int):
    """Gaps in [lo,hi) not covered by the (merged) intervals."""
    gaps, cur = [], lo
    for b, e in intervals:
        if b > cur:
            gaps.append((cur, min(b, hi)))
        cur = max(cur, e)
        if cur >= hi:
            break
    if cur < hi:
        gaps.append((cur, hi))
    return [(b, e) for b, e in gaps if e > b]


def _intersect(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Intersection of two merged interval lists → list of overlap intervals."""
    out, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        lo = max(a[i][0], b[j][0])
        hi = min(a[i][1], b[j][1])
        if hi > lo:
            out.append((lo, hi))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def _overlap(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> int:
    """Total overlap (ns) between two merged interval lists."""
    i = j = 0
    tot = 0
    while i < len(a) and j < len(b):
        lo = max(a[i][0], b[j][0])
        hi = min(a[i][1], b[j][1])
        if hi > lo:
            tot += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return tot


# --- in-flight curve -------------------------------------------------------
@dataclass
class InFlight:
    """Piecewise-constant count of admitted-but-not-terminal requests."""
    deltas: list[tuple[int, int]]  # sorted (ts, +1/-1)

    def positive_in(self, lo: int, hi: int) -> bool:
        """True if in-flight count > 0 anywhere in [lo, hi)."""
        count = 0
        for ts, d in self.deltas:
            if ts < lo:
                count += d
            elif ts < hi:
                if count > 0:
                    return True
                count += d
                if count > 0:
                    return True
            else:
                break
        return count > 0

    def positive_intervals(self, lo: int, hi: int) -> list[tuple[int, int]]:
        """Merged sub-intervals of [lo,hi) where in-flight count > 0. Used to
        split a stage's idle gap into STARVED (a request is in flight elsewhere)
        vs IDLE (nothing in flight)."""
        out: list[tuple[int, int]] = []
        count = sum(d for ts, d in self.deltas if ts < lo)  # in-flight at lo
        cur = lo
        for ts, d in self.deltas:
            if ts < lo or ts >= hi:
                continue
            if count > 0 and ts > cur:
                out.append((cur, ts))
            count += d
            cur = ts
        if count > 0 and hi > cur:
            out.append((cur, hi))
        return _merge(out)


def build_inflight(events: list[dict]) -> InFlight:
    admit: dict[str, int] = {}
    done: dict[str, int] = {}
    first: dict[str, int] = {}
    last: dict[str, int] = {}
    for ev in events:
        rid = ev.get("request_id")
        # Skip stage-scoped (__stage__) and SYNTHETIC sub-request ids. The relay
        # blob key is "<rid>:stream:<from>:<to>:<chunk>", so bp_wait emits carry
        # colon-bearing ids; counting them tiles the whole window and makes
        # in-flight never reach 0 (collapsing IDLE into STARVED). Only real
        # top-level request ids (no ':') define the in-flight curve.
        if not rid or rid == STAGE_RID or ":" in rid:
            continue
        ts = int(ev["timestamp_ns"])
        first[rid] = min(first.get(rid, ts), ts)
        last[rid] = max(last.get(rid, ts), ts)
        name = ev.get("event_name")
        if name == EV_ADMIT:
            admit[rid] = min(admit.get(rid, ts), ts)
        elif name in EV_TERMINAL:
            done[rid] = max(done.get(rid, ts), ts)
    deltas: list[tuple[int, int]] = []
    for rid in first:
        a = admit.get(rid, first[rid])
        d = done.get(rid, last[rid])
        if d <= a:
            d = a + 1
        deltas.append((a, +1))
        deltas.append((d, -1))
    deltas.sort()
    return InFlight(deltas)


# --- decomposition ---------------------------------------------------------
@dataclass
class StageDecomp:
    stage: str
    window_ns: int
    busy_ns: int = 0
    starved_ns: int = 0
    blocked_ns: int = 0
    idle_ns: int = 0
    bp_stall_ns: int = 0       # G2 total (== blocked attributable to backpressure)
    bp_count: int = 0
    busy_intervals: list[tuple[int, int]] = field(default_factory=list)
    blocked_intervals: list[tuple[int, int]] = field(default_factory=list)
    starved_intervals: list[tuple[int, int]] = field(default_factory=list)

    def pct(self, ns: int) -> float:
        return 100.0 * ns / self.window_ns if self.window_ns else 0.0

    @property
    def bubble_ns(self) -> int:
        return self.starved_ns + self.blocked_ns

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "window_ms": round(self.window_ns / NS_PER_MS, 2),
            "busy_ms": round(self.busy_ns / NS_PER_MS, 2),
            "starved_ms": round(self.starved_ns / NS_PER_MS, 2),
            "blocked_ms": round(self.blocked_ns / NS_PER_MS, 2),
            "idle_ms": round(self.idle_ns / NS_PER_MS, 2),
            "busy_pct": round(self.pct(self.busy_ns), 1),
            "starved_pct": round(self.pct(self.starved_ns), 1),
            "blocked_pct": round(self.pct(self.blocked_ns), 1),
            "idle_pct": round(self.pct(self.idle_ns), 1),
            "bubble_pct": round(self.pct(self.bubble_ns), 1),
            "backpressure_stall_ms": round(self.bp_stall_ns / NS_PER_MS, 2),
            "backpressure_events": self.bp_count,
        }


def decompose(events: list[dict], starve_signal: str = "inflight",
              window_bounds: tuple[int, int] | None = None):
    events = sorted(events, key=lambda e: int(e.get("timestamp_ns", 0)))
    if not events:
        return [], (0, 0)
    # Prefer explicit window bounds (segment wall-time incl. startup/idle gaps)
    # over event-span, so GPU-idle from gaps isn't silently dropped.
    if window_bounds is not None:
        W0, W1 = window_bounds
    else:
        W0 = int(events[0]["timestamp_ns"])
        W1 = int(events[-1]["timestamp_ns"])
    window = max(W1 - W0, 1)

    stages = sorted({ev.get("stage", "unknown") for ev in events
                     if ev.get("event_name") in (EV_FWD_BEGIN, EV_FWD_END,
                                                 EV_BP_BEGIN, EV_BP_END)})
    busy_by_stage = _match_intervals(events, EV_FWD_BEGIN, EV_FWD_END,
                                     key_fn=lambda e: e.get("stage", "unknown"))
    blk_by_stage = _match_intervals(events, EV_BP_BEGIN, EV_BP_END,
                                    key_fn=lambda e: (e.get("stage", "unknown"),
                                                      e.get("request_id")))
    # regroup blocked by stage only
    blk_stage: dict[str, list[tuple[int, int]]] = defaultdict(list)
    bp_count: dict[str, int] = defaultdict(int)
    for (stg, _rid), ivs in blk_by_stage.items():
        blk_stage[stg].extend(ivs)
        bp_count[stg] += len(ivs)

    inflight = build_inflight(events)
    # fallback starvation signal: union of *other* stages' busy intervals
    all_busy = {s: _merge(busy_by_stage.get(s, [])) for s in stages}

    results: list[StageDecomp] = []
    for s in stages:
        busy = _merge(busy_by_stage.get(s, []))
        blocked_all = _merge(blk_stage.get(s, []))
        bp_total = _total(blocked_all)
        d = StageDecomp(stage=s, window_ns=window,
                        busy_intervals=busy, blocked_intervals=blocked_all,
                        bp_stall_ns=bp_total, bp_count=bp_count.get(s, 0))
        d.busy_ns = _total(busy)
        # idle gaps = window minus busy
        gaps = _complement(busy, W0, W1)
        # blocked = overlap(gaps, backpressure) -- bp can't overlap busy anyway
        blocked_in_gaps = _merge([
            (max(g0, b0), min(g1, b1))
            for (g0, g1) in gaps for (b0, b1) in blocked_all
            if min(g1, b1) > max(g0, b0)
        ])
        d.blocked_ns = _total(blocked_in_gaps)
        # remaining gap time (non-blocked) -> STARVED vs IDLE
        non_blocked = _complement(blocked_in_gaps, W0, W1)
        # intersect non_blocked with gaps
        rest = _merge([
            (max(g0, n0), min(g1, n1))
            for (g0, g1) in gaps for (n0, n1) in non_blocked
            if min(g1, n1) > max(g0, n0)
        ])
        # Split each idle gap by the in-flight curve: STARVED = the part of the
        # gap where a request is in flight (some other stage could be working),
        # IDLE = the part where nothing is in flight. (Coarsely labeling a whole
        # gap by "any in-flight" over-reports STARVED — that was the bug.)
        if starve_signal == "inflight":
            active_iv = _merge([iv for (a, b) in rest
                                for iv in inflight.positive_intervals(a, b)])
        else:  # "otherbusy": active where another stage is busy
            other = _merge([iv for o in stages if o != s for iv in all_busy[o]])
            active_iv = _intersect(_merge(rest), other)
        starved_ivs = _intersect(_merge(rest), active_iv)
        d.starved_ns = _total(starved_ivs)
        d.idle_ns = _total(rest) - d.starved_ns
        d.starved_intervals = starved_ivs
        results.append(d)
    results.sort(key=lambda r: -r.bubble_ns)
    return results, (W0, W1)


# --- rendering -------------------------------------------------------------
_GLYPH = {"B": "█", "S": "▒", "X": "▓", ".": "·"}  # busy/starved/blocked/idle


def gpu_union_busy(results: list[StageDecomp]) -> tuple[list[tuple[int, int]], int]:
    """Union of ALL stages' BUSY intervals = wall-time the (shared) GPU is doing
    useful work for SOME stage. On a co-located single GPU this is the honest
    GPU-utilization metric: true_idle = window - union_busy. (Per-stage STARVED
    does NOT mean the GPU is idle — another co-tenant stage may be running.)"""
    allivs: list[tuple[int, int]] = []
    for r in results:
        allivs.extend(r.busy_intervals)
    union = _merge(allivs)
    return union, _total(union)


def load_nvml(paths, since_ns=None, until_ns=None):
    """Load nvml_sampler CSVs → {gpu_index: [util,...]} within the window."""
    import csv as _csv
    out: dict[str, list[float]] = defaultdict(list)
    files: list[Path] = []
    for p in paths:
        pp = Path(p)
        if pp.is_dir():
            files.extend(sorted(pp.glob("gpu_*.csv")))
        elif pp.is_file():
            files.append(pp)
    for f in files:
        try:
            with f.open() as fp:
                for row in _csv.DictReader(fp):
                    ts = int(row["timestamp_ns"])
                    if since_ns and ts < since_ns:
                        continue
                    if until_ns and ts > until_ns:
                        continue
                    out[row["gpu_index"]].append(float(row["util_gpu_pct"]))
        except Exception as e:  # noqa: BLE001
            print(f"WARN nvml {f}: {e}", file=sys.stderr)
    return out


def swimlane(results: list[StageDecomp], W: tuple[int, int], width: int = 100,
             ascii_only: bool = False) -> str:
    W0, W1 = W
    span = max(W1 - W0, 1)
    g = {k: (v if not ascii_only else k) for k, v in _GLYPH.items()}
    name_w = max((len(r.stage) for r in results), default=5)
    lines = []
    for r in results:
        cells = []
        for i in range(width):
            t0 = W0 + span * i // width
            t1 = W0 + span * (i + 1) // width
            # pick dominant state in this cell
            ob = _overlap([(t0, t1)], r.busy_intervals)
            ox = _overlap([(t0, t1)], r.blocked_intervals)
            os_ = _overlap([(t0, t1)], r.starved_intervals)
            cell = max(("B", ob), ("X", ox), ("S", os_), (".", (t1 - t0) - ob - ox - os_),
                       key=lambda kv: kv[1])[0]
            cells.append(g[cell])
        lines.append(f"  {r.stage.rjust(name_w)} |{''.join(cells)}|")
    legend = f"  {'':>{name_w}}  {g['B']}=BUSY {g['S']}=STARVED {g['X']}=BLOCKED {g['.']}=IDLE  ({span/NS_PER_MS:.0f} ms wall)"
    return "\n".join(["  [M1] stage-GPU swimlane", *lines, legend])


def bar(pct: float, width: int = 24) -> str:
    n = int(round(pct / 100 * width))
    return "█" * n + " " * (width - n)


def report(results: list[StageDecomp]) -> str:
    out = ["", "  [M2] bubble decomposition (% of wall window)"]
    out.append(f"  {'stage':>12} | {'BUSY':>5} {'STARV':>5} {'BLOCK':>5} {'IDLE':>5} | "
               f"{'BUBBLE':>6} | bar(bubble)")
    out.append("  " + "-" * 78)
    for r in results:
        out.append(
            f"  {r.stage:>12} | {r.pct(r.busy_ns):4.0f}% {r.pct(r.starved_ns):4.0f}% "
            f"{r.pct(r.blocked_ns):4.0f}% {r.pct(r.idle_ns):4.0f}% | "
            f"{r.pct(r.bubble_ns):5.0f}% | {bar(r.pct(r.bubble_ns))}")
    out += ["", "  [M3] backpressure stall (G2: time blocked on relay credits)"]
    out.append(f"  {'stage':>12} | {'stall_ms':>9} {'events':>7} | {'% of window':>11}")
    out.append("  " + "-" * 50)
    for r in results:
        if r.bp_count == 0 and r.bp_stall_ns == 0:
            continue
        out.append(f"  {r.stage:>12} | {r.bp_stall_ns/NS_PER_MS:9.1f} {r.bp_count:7d} | "
                   f"{r.pct(r.bp_stall_ns):10.1f}%")
    if all(r.bp_count == 0 for r in results):
        out.append("  (no bp_wait events found — G2 instrumentation not applied?)")
    return "\n".join(out)


def maybe_png(results, W, path: str) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    W0, W1 = W
    color = {"busy": "#2ca02c", "starved": "#ff7f0e", "blocked": "#d62728", "idle": "#dddddd"}
    fig, ax = plt.subplots(figsize=(12, 0.6 * len(results) + 1))
    for i, r in enumerate(results):
        y = len(results) - 1 - i
        ax.broken_barh([(W0, W1 - W0)], (y - 0.4, 0.8), facecolors=color["idle"])
        for tag, ivs in (("starved", r.starved_intervals), ("blocked", r.blocked_intervals),
                         ("busy", r.busy_intervals)):
            bars = [((b - W0) / NS_PER_MS, (e - b) / NS_PER_MS) for b, e in ivs]
            ax.broken_barh(bars, (y - 0.4, 0.8), facecolors=color[tag])
    ax.set_yticks(range(len(results)))
    ax.set_yticklabels([r.stage for r in reversed(results)])
    ax.set_xlabel("wall-clock (ms)")
    ax.set_title("M1 stage-GPU swimlane  (green=busy orange=starved red=blocked grey=idle)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    return True


# --- selftest --------------------------------------------------------------
def _selftest() -> int:
    """Synthetic 2-stage scenario: thinker busy 0-100ms then BLOCKED 100-180ms
    (talker can't keep up); talker STARVED 0-100ms then busy 100-200ms. One
    request admitted at 0, done at 200ms."""
    ms = lambda x: int(x * NS_PER_MS)
    evs = [
        {"request_id": "r1", "stage": "coordinator", "event_name": EV_ADMIT, "timestamp_ns": ms(0)},
        # thinker busy 0..100
        {"request_id": STAGE_RID, "stage": "thinker", "event_name": EV_FWD_BEGIN, "timestamp_ns": ms(0)},
        {"request_id": STAGE_RID, "stage": "thinker", "event_name": EV_FWD_END, "timestamp_ns": ms(100)},
        # thinker blocked on credits 100..180
        {"request_id": "r1", "stage": "thinker", "event_name": EV_BP_BEGIN, "timestamp_ns": ms(100)},
        {"request_id": "r1", "stage": "thinker", "event_name": EV_BP_END, "timestamp_ns": ms(180)},
        # talker busy 100..200
        {"request_id": STAGE_RID, "stage": "talker", "event_name": EV_FWD_BEGIN, "timestamp_ns": ms(100)},
        {"request_id": STAGE_RID, "stage": "talker", "event_name": EV_FWD_END, "timestamp_ns": ms(200)},
        {"request_id": "r1", "stage": "code2wav", "event_name": "terminal_response", "timestamp_ns": ms(200)},
    ]
    results, W = decompose(evs)
    by = {r.stage: r for r in results}
    ok = True

    def check(name, got, exp, tol=2.0):
        nonlocal ok
        good = abs(got - exp) <= tol
        ok = ok and good
        print(f"    {'PASS' if good else 'FAIL'} {name}: got {got:.0f}ms exp {exp:.0f}ms")

    # window = 200ms
    check("thinker.busy", by["thinker"].busy_ns / NS_PER_MS, 100)
    check("thinker.blocked", by["thinker"].blocked_ns / NS_PER_MS, 80)
    check("thinker.starved", by["thinker"].starved_ns / NS_PER_MS, 20)  # 180..200 in-flight
    check("thinker.bp_stall", by["thinker"].bp_stall_ns / NS_PER_MS, 80)
    check("talker.busy", by["talker"].busy_ns / NS_PER_MS, 100)
    check("talker.starved", by["talker"].starved_ns / NS_PER_MS, 100)  # 0..100 in-flight, no busy
    check("talker.blocked", by["talker"].blocked_ns / NS_PER_MS, 0)
    # Scenario 2: exercise the gap-split (IDLE vs STARVED). Two requests with an
    # idle gap between them; stage 's' busy 10-30ms. in-flight: r-a [0,80], r-b
    # [150,230] → gap [80,150] has nothing in flight = IDLE. Expect for 's':
    # busy=20, starved=140 (rest ∩ in-flight), idle=70 (the [80,150] gap).
    evs2 = [
        {"request_id": "a", "stage": "coordinator", "event_name": EV_ADMIT, "timestamp_ns": ms(0)},
        {"request_id": STAGE_RID, "stage": "s", "event_name": EV_FWD_BEGIN, "timestamp_ns": ms(10)},
        {"request_id": STAGE_RID, "stage": "s", "event_name": EV_FWD_END, "timestamp_ns": ms(30)},
        {"request_id": "a", "stage": "coordinator", "event_name": "terminal_response", "timestamp_ns": ms(80)},
        {"request_id": "b", "stage": "coordinator", "event_name": EV_ADMIT, "timestamp_ns": ms(150)},
        {"request_id": "b", "stage": "coordinator", "event_name": "terminal_response", "timestamp_ns": ms(230)},
    ]
    r2 = {x.stage: x for x in decompose(evs2)[0]}["s"]
    print()
    check("scn2 s.busy", r2.busy_ns / NS_PER_MS, 20)
    check("scn2 s.starved", r2.starved_ns / NS_PER_MS, 140)
    check("scn2 s.idle (gap-split)", r2.idle_ns / NS_PER_MS, 70)
    print()
    print(report(results))
    print()
    print(swimlane(results, W, width=60, ascii_only=False))
    print()
    print("  SELFTEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# --- main ------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", help="event JSONL file(s) or dir(s)")
    ap.add_argument("--selftest", action="store_true", help="run synthetic self-test")
    ap.add_argument("--width", type=int, default=100, help="swimlane char width")
    ap.add_argument("--ascii", action="store_true", help="ascii-only swimlane glyphs")
    ap.add_argument("--starve-signal", choices=["inflight", "otherbusy"],
                    default="inflight", help="how to classify idle as STARVED")
    ap.add_argument("--since-ns", type=int, default=None,
                    help="only analyze events with timestamp_ns >= this (skip warmup)")
    ap.add_argument("--until-ns", type=int, default=None,
                    help="only analyze events with timestamp_ns <= this")
    ap.add_argument("--json", dest="json_out", help="write JSON report to this path")
    ap.add_argument("--png", help="write M1 swimlane PNG (needs matplotlib)")
    args = ap.parse_args()

    if args.selftest:
        return _selftest()
    if not args.paths:
        ap.error("provide event JSONL path(s)/dir(s) or --selftest")

    events = list(iter_events(args.paths))
    if args.since_ns is not None:
        events = [e for e in events if int(e.get("timestamp_ns", 0)) >= args.since_ns]
    if args.until_ns is not None:
        events = [e for e in events if int(e.get("timestamp_ns", 0)) <= args.until_ns]
    print(f"loaded {len(events)} events from {args.paths}"
          + (f" (windowed since {args.since_ns})" if args.since_ns else ""))
    wb = None
    if args.since_ns is not None and args.until_ns is not None:
        wb = (args.since_ns, args.until_ns)
    results, W = decompose(events, starve_signal=args.starve_signal, window_bounds=wb)
    if not results:
        print("no stage forward (G1) events found — is instrumentation applied?")
        return 1
    window = max(W[1] - W[0], 1)
    union, union_busy = gpu_union_busy(results)
    nvml = load_nvml(args.paths, since_ns=args.since_ns, until_ns=args.until_ns)
    print("\n  [M0] GPU-level utilization (shared GPU — the HONEST waste metric)")
    print(f"  union-busy (some stage computing): {100*union_busy/window:5.1f}%   "
          f"true-idle: {100*(window-union_busy)/window:5.1f}%   "
          f"window: {window/NS_PER_MS:.0f} ms")
    if nvml:
        for g, vals in sorted(nvml.items()):
            vals_s = sorted(vals)
            p50 = vals_s[len(vals_s)//2] if vals_s else 0
            mean = sum(vals)/len(vals) if vals else 0
            print(f"  NVML gpu{g}: mean util {mean:4.0f}%  p50 {p50:4.0f}%  "
                  f"(n={len(vals)})  [independent check of union-busy]")
    has_admit = any(e.get("event_name") == EV_ADMIT for e in events)
    if not has_admit:
        print("\n  *** WARNING: no request_admission events — the STARVED/IDLE/BLOCKED"
              "\n  split is NON-FUNCTIONAL (in-flight never reaches 0; all idle is"
              "\n  mislabeled STARVED). Trust ONLY 'BUSY' (duty cycle) and M0 union-busy.")
    print("\n  NOTE: per-stage rows below are DUTY CYCLE (share of wall a stage was"
          "\n  computing). Stages TIME-SHARE the GPU, so they sum to >100%. Per-stage"
          "\n  'STARVED' = that stage idle, NOT the GPU idle (see M0 for GPU idle).")
    print(report(results))
    print()
    print(swimlane(results, W, width=args.width, ascii_only=args.ascii))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {"window_ms": (W[1] - W[0]) / NS_PER_MS,
             "stages": [r.to_dict() for r in results]}, indent=2))
        print(f"\nwrote JSON -> {args.json_out}")
    if args.png:
        print(("wrote PNG -> " + args.png) if maybe_png(results, W, args.png)
              else "matplotlib unavailable; skipped PNG")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
