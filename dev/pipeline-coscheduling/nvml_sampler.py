#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""G5: GPU util/mem sampler aligned to event_recorder timestamps (time.time_ns).

Uses pynvml for true high-resolution sampling (~the requested interval; no
subprocess fork). Falls back to `nvidia-smi` (effective cadence ~150-200ms due
to fork/exec latency) only if pynvml is unavailable.

CSV columns: timestamp_ns,gpu_index,util_gpu_pct,mem_used_mib
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time


def _run_pynvml(args, gpus):
    import pynvml
    pynvml.nvmlInit()
    handles = {g: pynvml.nvmlDeviceGetHandleByIndex(g) for g in gpus} if gpus else \
        {i: pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())}
    with open(args.out, "w", encoding="utf-8") as fp:
        fp.write("timestamp_ns,gpu_index,util_gpu_pct,mem_used_mib\n")
        print(f"[pynvml] sampling {sorted(handles)} -> {args.out} every "
              f"{args.interval_ms}ms", file=sys.stderr)
        period = args.interval_ms / 1000.0
        try:
            while True:
                ts = time.time_ns()
                for g, h in handles.items():
                    u = pynvml.nvmlDeviceGetUtilizationRates(h)
                    m = pynvml.nvmlDeviceGetMemoryInfo(h)
                    fp.write(f"{ts},{g},{u.gpu},{m.used // (1024*1024)}\n")
                fp.flush()
                time.sleep(period)
        except KeyboardInterrupt:
            print("\nstopped", file=sys.stderr)


def _run_smi(args, gpus):
    query = "index,utilization.gpu,memory.used"
    cmd = ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"]
    if gpus:
        cmd += [f"--id={','.join(str(g) for g in gpus)}"]
    with open(args.out, "w", encoding="utf-8") as fp:
        fp.write("timestamp_ns,gpu_index,util_gpu_pct,mem_used_mib\n")
        print(f"[nvidia-smi fallback] -> {args.out}", file=sys.stderr)
        try:
            while True:
                ts = time.time_ns()
                try:
                    out = subprocess.check_output(cmd, text=True, timeout=5)
                except Exception as e:  # noqa: BLE001
                    print(f"WARN sample failed: {e}", file=sys.stderr)
                    time.sleep(args.interval_ms / 1000)
                    continue
                for line in out.strip().splitlines():
                    idx, util, mem = (c.strip() for c in line.split(","))
                    fp.write(f"{ts},{idx},{util},{mem}\n")
                fp.flush()
                time.sleep(args.interval_ms / 1000)
        except KeyboardInterrupt:
            print("\nstopped", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval-ms", type=int, default=50)
    ap.add_argument("--gpus", default="", help="comma-separated indices; empty=all")
    args = ap.parse_args()
    gpus = [int(g) for g in args.gpus.split(",") if g.strip() != ""]
    try:
        import pynvml  # noqa: F401
        _run_pynvml(args, gpus)
    except Exception as e:  # noqa: BLE001
        print(f"pynvml unavailable ({e}); falling back to nvidia-smi", file=sys.stderr)
        _run_smi(args, gpus)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
