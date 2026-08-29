#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Phase-0 load generator: concurrent streaming speech requests.

Drives ``POST /v1/chat/completions`` with ``modalities=["text","audio"]`` and
``stream=true`` at a chosen concurrency, recording client-side TTFT (first text
delta), TTFA (first audio delta), and total per request. The server-side event
recorder (G1/G2) captures the GPU bubble decomposition; this driver only
generates the load and the client-observed latency/jitter.

Concurrency sweep answers Phase-0 Q3 ("does batching fill the bubble?"): run at
C=1 (single voice session, the realtime regime) up through saturation.

Usage:
    python run_workload.py --base-url http://localhost:8000 \
        --concurrency 1 --num-requests 8 --label s1_coloc --out results/s1.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import httpx

PROMPTS = [
    "Please reply: Hello, how are you today?",
    "Tell me a short story about a robot who learns to paint.",
    "Please respond verbatim: The quick brown fox jumps over the lazy dog "
    "while the sun sets over the quiet hills.",
    "Explain in two sentences why the sky is blue.",
]


async def _one(client, base_url, model, prompt, seed, timeout_s, label, idx,
               modalities=("text", "audio"), max_tokens=256, ignore_eos=False):
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "modalities": list(modalities),
        "stream": True,
        "seed": seed,
        "max_tokens": max_tokens,
        "metadata": {"client_label": f"{label}-{idx}"},
    }
    if ignore_eos:
        # force the thinker to emit exactly max_tokens → holds per-request work
        # roughly constant (kills the bimodal EOS-at-6 vs run-to-100 variance).
        payload["ignore_eos"] = True
    if "audio" in modalities:
        payload["audio"] = {"voice": "alloy", "format": "wav"}
    t0 = time.perf_counter()
    ttft = ttfa = None
    audio_chunks = text_chunks = 0
    audio_arrivals: list[float] = []
    status = 0
    try:
        async with client.stream("POST", url, json=payload, timeout=timeout_s) as r:
            status = r.status_code
            if status >= 400:
                body = await r.aread()
                return {"idx": idx, "status": status, "error": body[:300].decode("utf-8", "replace")}
            async for raw in r.aiter_lines():
                line = raw.strip()
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    continue
                try:
                    evt = json.loads(body)
                except json.JSONDecodeError:
                    continue
                for ch in evt.get("choices", []):
                    delta = ch.get("delta") or {}
                    if delta.get("content"):
                        text_chunks += 1
                        if ttft is None:
                            ttft = time.perf_counter() - t0
                    audio = delta.get("audio")
                    if audio and audio.get("data"):
                        now = time.perf_counter() - t0
                        if ttfa is None:
                            ttfa = now
                        audio_arrivals.append(now)
                        audio_chunks += 1
    except Exception as e:  # noqa: BLE001 - record failures, don't crash the sweep
        return {"idx": idx, "status": status, "error": repr(e)[:300]}
    total = time.perf_counter() - t0
    # inter-audio-frame gaps (jitter signal)
    gaps = [audio_arrivals[i] - audio_arrivals[i - 1] for i in range(1, len(audio_arrivals))]
    return {
        "idx": idx, "status": status, "ttft": ttft, "ttfa": ttfa, "total": total,
        "text_chunks": text_chunks, "audio_chunks": audio_chunks,
        "frame_gap_mean": statistics.fmean(gaps) if gaps else None,
        "frame_gap_max": max(gaps) if gaps else None,
    }


async def _run(args):
    sem = asyncio.Semaphore(args.concurrency)
    results = []
    async with httpx.AsyncClient(http2=False) as client:
        base_mods = tuple(m.strip() for m in args.modalities.split(",") if m.strip())

        async def guarded(i):
            async with sem:
                prompt = PROMPTS[i % len(PROMPTS)]
                # mixed traffic: deterministically interleave text-only
                # (understanding) requests among speech ones.
                if args.speech_frac >= 1.0:
                    mods = base_mods
                else:
                    text_every = max(2, round(1.0 / max(1e-9, 1.0 - args.speech_frac)))
                    mods = ("text",) if (i % text_every == 0) else ("text", "audio")
                res = await _one(client, args.base_url, args.model, prompt,
                                 1000 + i, args.timeout_s, args.label, i,
                                 modalities=mods, max_tokens=args.max_tokens,
                                 ignore_eos=args.ignore_eos)
                res["modalities"] = ",".join(mods)
                # think-time: idle gap after each request → pipeline empties →
                # genuine IDLE (vs STARVED). Validates the four-way decomposition.
                if args.think_time > 0:
                    await asyncio.sleep(args.think_time)
                ok = res.get("ttfa") is not None
                print(f"  req {i:3d} status={res.get('status')} "
                      f"ttft={res.get('ttft')} ttfa={res.get('ttfa')} "
                      f"total={res.get('total')} audio={res.get('audio_chunks')}"
                      f"{'' if ok else '  ERR='+str(res.get('error'))}")
                results.append(res)
        # launch all; semaphore caps in-flight at concurrency
        await asyncio.gather(*(guarded(i) for i in range(args.num_requests)))
    return results


def _summ(results):
    # "ok" = produced audio (speech) OR text (understanding) with status 200
    ok = [r for r in results if r.get("ttfa") is not None
          or (r.get("status") == 200 and (r.get("text_chunks") or 0) > 0)]
    if not ok:
        return {"ok": 0, "total": len(results)}
    f = statistics.fmean
    ttfas = [r["ttfa"] for r in ok if r.get("ttfa") is not None]
    out = {
        "ok": len(ok), "total": len(results),
        "ttft_mean": f([r["ttft"] for r in ok if r.get("ttft") is not None] or [0]),
        "total_mean": f([r["total"] for r in ok]),
    }
    sd = statistics.pstdev
    if ttfas:
        out["ttfa_mean"] = f(ttfas)
        out["ttfa_stdev"] = sd(ttfas) if len(ttfas) > 1 else 0.0
        out["ttfa_p95"] = sorted(ttfas)[int(0.95 * (len(ttfas) - 1))]
        out["frame_gap_max"] = max(
            (r["frame_gap_max"] for r in ok if r.get("frame_gap_max")), default=None)
    # per-request work distribution — surfaces the bimodal-length confound
    ac = sorted(r.get("audio_chunks") or 0 for r in ok)
    if any(ac):
        out["audio_chunks"] = {
            "min": ac[0], "median": ac[len(ac) // 2], "max": ac[-1],
            "mean": round(f(ac), 1), "stdev": round(sd(ac), 1) if len(ac) > 1 else 0.0,
            "frac_long": round(sum(1 for x in ac if x > 40) / len(ac), 2)}
    tc = sorted(r.get("text_chunks") or 0 for r in ok)
    out["text_chunks_median"] = tc[len(tc) // 2] if tc else 0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", default="qwen3-omni")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--num-requests", type=int, default=8)
    ap.add_argument("--timeout-s", type=float, default=600.0)
    ap.add_argument("--label", default="run")
    ap.add_argument("--modalities", default="text,audio",
                    help="comma list, e.g. 'text,audio' (speech) or 'text' (understanding)")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--ignore-eos", action="store_true",
                    help="force exactly max_tokens (constant per-request work)")
    ap.add_argument("--speech-frac", type=float, default=1.0,
                    help="<1.0 → mix in text-only requests (1-frac fraction)")
    ap.add_argument("--think-time", type=float, default=0.0,
                    help="idle seconds after each request (low-load → genuine IDLE)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    print(f"workload: C={args.concurrency} N={args.num_requests} -> {args.base_url}")
    t0 = time.perf_counter()
    results = asyncio.run(_run(args))
    wall = time.perf_counter() - t0
    summ = _summ(results)
    summ["wall_s"] = wall
    summ["throughput_rps"] = summ["ok"] / wall if wall else 0
    print("\nSUMMARY", json.dumps(summ, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"args": vars(args) | {"out": str(args.out)}, "summary": summ, "runs": results},
            indent=2, default=str))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
