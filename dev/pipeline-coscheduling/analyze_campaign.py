#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Aggregate a measurement campaign: parse SEG=<name> SINCE/UNTIL markers from a
campaign log, decompose each window, and print a cross-segment comparison table.

Usage:
    python analyze_campaign.py --log logs/campaign_coloc.log --events events_coloc \
        --json findings/campaign_coloc.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import analyze_bubbles as ab

SEG_SINCE = re.compile(r"SEG=(\S+)\s+SINCE=(\d+)(?:\s+C=(\S+)\s+N=(\S+)\s+MODS=(\S+)\s+MT=(\S+))?")
SEG_UNTIL = re.compile(r"SEG=(\S+)\s+UNTIL=(\d+)")
FOCUS = ("thinker", "talker_ar", "code2wav")


def parse_segments(log_path: str):
    since, until, meta = {}, {}, {}
    for line in Path(log_path).read_text(errors="replace").splitlines():
        m = SEG_SINCE.search(line)
        if m:
            since[m.group(1)] = int(m.group(2))
            if m.group(3):
                meta[m.group(1)] = {"C": m.group(3), "N": m.group(4),
                                    "mods": m.group(5), "mt": m.group(6)}
        m = SEG_UNTIL.search(line)
        if m:
            until[m.group(1)] = int(m.group(2))
    segs = []
    for name in since:
        if name in until:
            segs.append((name, since[name], until[name], meta.get(name, {})))
    return segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--events", required=True)
    ap.add_argument("--nvml-dir", default=None, help="dir with gpu_*.csv (NVML)")
    ap.add_argument("--gpu-map", default=None,
                    help="stage:gpu CSV, e.g. 'thinker:0,talker_ar:1,code2wav:1' "
                         "→ report per-GPU union-busy (the wasted-whole-GPU metric)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    all_events = list(ab.iter_events(args.events))
    segs = parse_segments(args.log)
    print(f"{len(all_events)} events; {len(segs)} segments")
    print("GPU%=union-busy (some stage computing, the real shared-GPU util); "
          "NVML=measured; per-stage = DUTY CYCLE (time-share, sum>100%)\n")

    hdr = (f"{'segment':<13} {'C':>2} {'mods':>10} {'mt':>4} | {'GPU%':>5} {'NVML':>5} | "
           + " | ".join(f"{s+' duty':>13}" for s in FOCUS))
    print(hdr); print("-" * len(hdr))

    rows = []
    for name, s, u, meta in segs:
        evs = [e for e in all_events if s <= int(e.get("timestamp_ns", 0)) <= u]
        results, _ = ab.decompose(evs, window_bounds=(s, u))
        window = max(u - s, 1)
        _, union_busy = ab.gpu_union_busy(results)
        gpu_pct = 100.0 * union_busy / window
        # per-GPU union-busy (the wasted-whole-GPU metric for disaggregation)
        per_gpu = {}
        if args.gpu_map:
            gmap = dict(kv.split(":") for kv in args.gpu_map.split(","))
            byname = {r.stage: r for r in results}
            for g in sorted(set(gmap.values())):
                ivs = []
                for st, gg in gmap.items():
                    if gg == g and st in byname:
                        ivs.extend(byname[st].busy_intervals)
                per_gpu[g] = round(100.0 * ab._total(ab._merge(ivs)) / window, 1)
        nvml = ab.load_nvml([args.nvml_dir] if args.nvml_dir else [],
                            since_ns=s, until_ns=u)
        nvml_all = [v for vs in nvml.values() for v in vs]
        nvml_mean = sum(nvml_all) / len(nvml_all) if nvml_all else None
        by = {r.stage: r for r in results}
        cells = []
        rowd = {"segment": name, **meta, "gpu_union_busy_pct": round(gpu_pct, 1),
                "nvml_mean_pct": round(nvml_mean, 1) if nvml_mean is not None else None,
                "stages": {}}
        for st in FOCUS:
            r = by.get(st)
            if r:
                cells.append(f"{r.pct(r.busy_ns):3.0f}% busy")
                rowd["stages"][st] = {"duty_busy": round(r.pct(r.busy_ns), 1),
                                      "stage_idle": round(r.pct(r.bubble_ns + r.idle_ns), 1)}
            else:
                cells.append(f"{'--':>9}")
        nv = f"{nvml_mean:.0f}%" if nvml_mean is not None else "-"
        pg = ("  per-GPU: " + " ".join(f"gpu{g}={p}%" for g, p in per_gpu.items())) if per_gpu else ""
        rowd["per_gpu_union_busy"] = per_gpu
        print(f"{name:<13} {meta.get('C','?'):>2} {meta.get('mods','?'):>10} "
              f"{meta.get('mt','?'):>4} | {gpu_pct:4.0f}% {nv:>5} | "
              + " | ".join(f"{c:>13}" for c in cells) + pg)
        rows.append(rowd)

    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
