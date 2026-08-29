#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Optimality-gap analysis (solution-side Phase 1) — corrected.

GPU-resource efficiency = useful GPU-work / provisioned GPU-time
  = (sum over the G provisioned GPUs of that GPU's union-busy fraction) / G.
Work-conserving optimum = 100% efficiency (every provisioned GPU-second spent on
useful stage-work). Optimality gap = 1 - efficiency.

NOTE on why union-busy (not sum-of-stage-duty): stages run in separate processes
and time-share the GPU, so their forward wall-spans OVERLAP — sum-of-duty can
exceed 100% and is NOT GPU-work. The actual GPU work is the UNION of stage-busy
intervals (validated, NVML-corroborated).

Per request: useful GPU-work = (sum_g union_busy_g * window) / N  [GPU-seconds].
Computed from already-measured data (analyze_campaign JSON, with --gpu-map for
per-GPU union on disaggregated configs). No new runs.

Usage:
  python optimality_gap.py --campaign findings/campaign_hard2.json --log <log> \
      --results results --gpus 1
  python optimality_gap.py --campaign findings/campaign_disagg.json --log <log> \
      --results results --gpus 2   # per_gpu_union_busy must be present
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path

NS = 1e9


def windows(log):
    t = Path(log).read_text()
    s = dict(re.findall(r"SEG=(\S+) SINCE=(\d+)", t))
    u = dict(re.findall(r"SEG=(\S+) UNTIL=(\d+)", t))
    return {k: (int(s[k]), int(u[k])) for k in s if k in u}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--results", default="results")
    ap.add_argument("--gpus", type=int, default=1)
    args = ap.parse_args()

    win = windows(args.log)
    camp = {r["segment"]: r for r in json.load(open(args.campaign))}
    G = args.gpus

    hdr = (f"{'segment':<15} {'win_s':>6} {'N':>3} {'rps':>6} | "
           f"{'GPUwork/req_s':>13} {'ceiling@%dgpu' % G:>13} | {'GPU_eff':>8} {'OPT_GAP':>8}")
    print(hdr); print("-" * len(hdr))
    rows = []
    for seg, (s, u) in sorted(win.items()):
        c = camp.get(seg)
        rp = Path(args.results) / f"{seg}.json"
        if not c or not rp.exists():
            continue
        # Guard: an intentionally underloaded segment's idle is OFFERED-LOAD, not
        # scheduling inefficiency — its "gap" is meaningless. Skip with a note.
        if any(k in seg for k in ("lowload", "think", "warm")):
            print(f"{seg:<15}  (skipped — underloaded; idle is offered-load, not a gap)")
            continue
        window_s = (u - s) / NS
        summ = json.load(open(rp)).get("summary", {})
        N, rps = summ.get("ok", 0), summ.get("throughput_rps", 0.0)
        if not N or not rps:
            continue
        # per-GPU union-busy: use per_gpu map for disagg, else the single union.
        pg = c.get("per_gpu_union_busy") or {}
        if G > 1 and pg:
            per_gpu = [pg[k] / 100.0 for k in sorted(pg)][:G]
        else:
            per_gpu = [c.get("gpu_union_busy_pct", 0.0) / 100.0]
        eff = sum(per_gpu) / G                       # GPU-resource efficiency
        gap = 1.0 - eff
        useful_work_per_req = sum(per_gpu) * window_s / N   # GPU-sec/req (across G)
        ceiling = G / useful_work_per_req if useful_work_per_req else 0.0
        rows.append({"segment": seg, "gpus": G, "window_s": round(window_s, 1), "N": N,
                     "rps": round(rps, 3), "gpu_work_per_req_s": round(useful_work_per_req, 3),
                     "ceiling_rps": round(ceiling, 2), "per_gpu_union": per_gpu,
                     "gpu_efficiency": round(eff, 3), "optimality_gap": round(gap, 3)})
        print(f"{seg:<15} {window_s:6.1f} {N:>3} {rps:6.2f} | {useful_work_per_req:11.3f}s "
              f"{ceiling:9.2f}rps | {eff*100:6.0f}% {gap*100:6.0f}%")
    out = Path(f"findings/optimality_gap_{G}gpu.json")
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
