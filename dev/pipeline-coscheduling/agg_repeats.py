#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Aggregate repeated segments (sweep2_c<C>_r<rep>) into mean±std per concurrency.

Reads the per-segment JSON written by analyze_campaign.py and groups by the base
name (strips the _r<N> suffix), reporting mean ± stdev for GPU union-busy, NVML,
and per-stage duty — i.e. confidence intervals on the headline numbers.

Usage: python agg_repeats.py findings/campaign_hard2.json
"""
import json
import re
import statistics
import sys
from collections import defaultdict

rows = json.load(open(sys.argv[1]))
groups = defaultdict(list)
for r in rows:
    base = re.sub(r"_r\d+$", "", r["segment"])
    groups[base].append(r)


def ms(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "—"
    if len(vals) == 1:
        return f"{vals[0]:.0f}"
    # sample stdev (Bessel) for small-n repeats, not population stdev
    return f"{statistics.fmean(vals):.0f}±{statistics.stdev(vals):.0f}"


FOCUS = ("thinker", "talker_ar", "code2wav")
hdr = f"{'group':<16} {'n':>2} | {'GPU%':>8} {'NVML':>8} | " + " | ".join(f"{s+' duty':>10}" for s in FOCUS)
print(hdr); print("-" * len(hdr))
for base in sorted(groups):
    g = groups[base]
    gpu = ms([x.get("gpu_union_busy_pct") for x in g])
    nv = ms([x.get("nvml_mean_pct") for x in g])
    cells = []
    for st in FOCUS:
        cells.append(ms([x["stages"].get(st, {}).get("duty_busy") for x in g if st in x.get("stages", {})]))
    print(f"{base:<16} {len(g):>2} | {gpu:>8} {nv:>8} | " + " | ".join(f"{c:>10}" for c in cells))
