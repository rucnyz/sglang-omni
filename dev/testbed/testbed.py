#!/usr/bin/env python3
"""sglang-omni scheduling testbed runner.

Reads scenarios.yaml, groups scenarios by server, launches each server once (with the
cold-start fix: persistent TRITON_CACHE_DIR + generous startup timeout), drives the
declared workload at each concurrency, records client-side perf metrics, tears the
server down, and writes a structured report. Supports regression comparison.

Runs INSIDE the omni-phase0 container (needs to launch the server + curl localhost).

A "server" launches via either `--config <yaml>` or `--model-path <repo>` (+extra_args,
+--colocate). Servers whose `requires_weights` HF-cache dir is absent are skipped.

Workload clients:
  omni_chat       POST /v1/chat/completions, modalities=[text(,audio)], streaming
  tts_speech      POST /v1/audio/speech, stream_format=audio (raw PCM), streaming
  asr_transcribe  POST /v1/audio/transcriptions (multipart audio file -> text)

Usage:
    python testbed.py [--filter TAG] [--scenario ID] [--profile] [--regression]
                      [--update-baseline] [--gpu N] [--list]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

import httpx
import yaml

ROOT = Path("/workspace/sglang-omni")
HERE = Path(__file__).resolve().parent
HF_HUB = Path(os.environ.get("HF_HOME", "/root/.cache/huggingface")) / "hub"
TRITON_CACHE = HERE / "triton_cache"
RESULTS = HERE / "results"
BASELINES = HERE / "baselines"
SAMPLE_AUDIO = ROOT / "tests/data/query_to_cars.wav"  # for ASR transcription input
SEED = 1234
MIN_OK_RATIO = 0.9  # R1: a scenario must succeed >=90% per point to be baked as a baseline
PROMPTS = [
    "Tell me a short story about a robot learning to paint.",
    "Explain how rainbows form in two sentences.",
    "What are three tips for staying focused while studying?",
    "Describe the sound of a thunderstorm at night.",
]

_SERVER_PROC: subprocess.Popen | None = None
_RUN_DIR: Path | None = None
_POLICY_ENV: dict[str, str] = {}  # admission policy injected into every server launch (A/B)


# --------------------------------------------------------------------------- #
# Server lifecycle
# --------------------------------------------------------------------------- #
def _kill_servers() -> None:
    """Tear down the server we launched (whole process group) + any sglang-omni stragglers."""
    global _SERVER_PROC
    if _SERVER_PROC is not None:
        # The server is a setsid leader, so its pgid == its pid; orphaned stage children keep
        # that pgid even after reparenting to init. So ALWAYS SIGKILL the group after the grace
        # period — do NOT gate the SIGKILL on the parent still being alive. The old guard
        # (`if poll() is None`) skipped the kill whenever the parent died from SIGTERM while a
        # spawn_main stage child (mid-CUDA-init, ignoring SIGTERM) survived — orphaning a 235 GB
        # process that then OOM-blocked and confounded the next paired arm.
        pgid = _SERVER_PROC.pid
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(pgid, sig)
            except (ProcessLookupError, PermissionError):
                pass
            time.sleep(3 if sig is signal.SIGTERM else 0)
    # fallbacks: the launcher main keeps "-m sglang_omni.cli serve" in argv; the spawn_main stage
    # children do NOT, so reap them by their (container-scoped) multiprocessing-fork cmdline too.
    subprocess.run(["pkill", "-9", "-f", "sglang_omni.cli serve"], check=False)
    subprocess.run(["pkill", "-9", "-f", "multiprocessing.spawn import spawn_main"], check=False)
    _SERVER_PROC = None
    time.sleep(3)


def _gpu_compute_pids(gpu: int) -> list[str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader", "-i", str(gpu)],
            capture_output=True, text=True, timeout=10).stdout
        return [l.strip() for l in out.splitlines() if l.strip()]
    except Exception:
        return []


def _env_snapshot(gpu: int) -> dict:
    """Box/GPU context recorded with every scenario so a delta can be judged for fairness
    (C1: shared-box temporal confound). gpu_procs counts compute PIDs on the physical GPU."""
    snap = {}
    try:
        snap["loadavg1"] = round(os.getloadavg()[0], 1)
    except Exception:
        pass
    try:
        u = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits",
             "-i", str(gpu)], capture_output=True, text=True, timeout=10).stdout.strip()
        snap["gpu_util_pct"], snap["gpu_mem_mib"] = [x.strip() for x in u.split(",")][:2]
    except Exception:
        pass
    snap["gpu_procs"] = len(_gpu_compute_pids(gpu))
    return snap


def _build_serve_cmd(spec: dict) -> list[str]:
    cmd = [sys.executable, "-m", "sglang_omni.cli", "serve", "--port", "8000", "--host", "0.0.0.0"]
    if spec.get("colocate"):
        cmd.append("--colocate")
    if spec.get("config"):
        cmd += ["--config", str(ROOT / spec["config"])]
    elif spec.get("model_path"):
        cmd += ["--model-path", spec["model_path"]]
    else:
        raise ValueError(f"server spec needs config or model_path: {spec}")
    cmd += list(spec.get("extra_args", []))
    return cmd


def launch_server(spec: dict, gpu: int, timeout_s: float = 2400) -> bool:
    """Launch a server for `spec`; return True once /health is 200, else False."""
    global _SERVER_PROC
    _kill_servers()
    # Wait for the GPU to actually drain — a killed 235 GB stage process takes seconds to release
    # its CUDA context, and launching the next arm before it frees would OOM-block it (and any
    # leftover would confound the measurement). Poll until exclusive or a bounded timeout.
    for _ in range(30):
        pre = _gpu_compute_pids(gpu)
        if not pre:
            break
        time.sleep(2)
    if pre:
        print(f"  WARNING: GPU{gpu} not exclusive after wait — {len(pre)} compute PID(s) still "
              f"present; measurements may be confounded by co-tenants", flush=True)
    TRITON_CACHE.mkdir(parents=True, exist_ok=True)
    log = HERE / "results" / "_server.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "TRITON_CACHE_DIR": str(TRITON_CACHE),
        "SGLANG_OMNI_STARTUP_TIMEOUT": str(int(timeout_s)),
        "TMPDIR": "/tmp",
        "TRITON_PTXAS_PATH": "/usr/local/cuda/bin/ptxas",
        "SGLANG_OMNI_ADMGATE_LOG": "1",  # emit the [ADMGATE] controller trace for analysis
        **spec.get("env", {}),
        **_POLICY_ENV,  # admission-policy A/B overlay (empty unless --policy given)
    }
    cmd = _build_serve_cmd(spec)
    print(f"  launching: {' '.join(cmd[4:])}  (GPU{gpu})", flush=True)
    with open(log, "w") as fh:
        _SERVER_PROC = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                        stdin=subprocess.DEVNULL, env=env,
                                        preexec_fn=os.setsid, cwd=str(ROOT))
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if _SERVER_PROC.poll() is not None:
            print(f"  SERVER EXITED early (code {_SERVER_PROC.returncode}); see {log}", flush=True)
            return False
        try:
            if httpx.get("http://localhost:8000/health", timeout=3).status_code == 200:
                print(f"  ready in {int(time.time() - t0)}s", flush=True)
                return True
        except Exception:
            pass
        time.sleep(10)
    print("  TIMEOUT waiting for server", flush=True)
    return False


# --------------------------------------------------------------------------- #
# Workload clients  ->  per-request result dict
# --------------------------------------------------------------------------- #
async def _omni_chat_one(client, model, prompt, modalities, max_tokens, timeout_s):
    """Greedy (temperature=0)+fixed seed => deterministic output => regression-stable."""
    url = "http://localhost:8000/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "modalities": list(modalities),
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": SEED,
    }
    if "audio" in modalities:
        payload["audio"] = {"voice": "alloy", "format": "wav"}
    t0 = time.perf_counter()
    ttft = ttfa = None
    audio_arrivals, text_chunks, audio_chunks, status = [], 0, 0, 0
    try:
        async with client.stream("POST", url, json=payload, timeout=timeout_s) as r:
            status = r.status_code
            if status != 200:  # e.g. 429 admission shed -> _err marks it policy_rejected
                body = await r.aread()
                return _err(status, RuntimeError(body[:120]), t0, ttft=ttft, ttfa=ttfa)
            async for line in r.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data.strip() == "[DONE]":
                    break
                try:
                    delta = json.loads(data)["choices"][0]["delta"]
                except Exception:
                    continue
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
    except Exception as e:
        return _err(status, e, t0, ttft=ttft, ttfa=ttfa, audio_chunks=audio_chunks)
    return _ok(status, t0, ttft=ttft, ttfa=ttfa, text_chunks=text_chunks,
               audio_chunks=audio_chunks, arrivals=audio_arrivals)


async def _tts_speech_one(client, model, prompt, wl, timeout_s):
    """Raw-PCM streaming (stream_format=audio): ttfa = first PCM bytes; track jitter.

    Voice selection: `references` (voice-clone models, e.g. Qwen3-TTS Base / Fish) take a
    local/HTTP ref clip; `voice` (preset models, e.g. Voxtral) a speaker name; models that
    accept plain text (e.g. Higgs) need neither.
    """
    url = "http://localhost:8000/v1/audio/speech"
    payload = {
        "model": model, "input": prompt, "response_format": "pcm",
        "stream": True, "stream_format": "audio", "seed": SEED,
    }
    refs = wl.get("references")
    if refs:
        payload["references"] = [
            {"audio_path": str(ROOT / r["audio_path"]) if not str(r["audio_path"]).startswith("http")
             else r["audio_path"], **({"text": r["text"]} if r.get("text") else {})}
            for r in refs
        ]
    elif wl.get("voice"):
        payload["voice"] = wl["voice"]
    t0 = time.perf_counter()
    ttfa = None
    arrivals, nbytes, status = [], 0, 0
    try:
        async with client.stream("POST", url, json=payload, timeout=timeout_s) as r:
            status = r.status_code
            if status != 200:
                body = (await r.aread()).decode("utf-8", "replace")[:160]
                return _err(status, RuntimeError(body), t0)
            async for chunk in r.aiter_bytes():
                if not chunk:
                    continue
                now = time.perf_counter() - t0
                if ttfa is None:
                    ttfa = now
                arrivals.append(now)
                nbytes += len(chunk)
    except Exception as e:
        return _err(status, e, t0, ttfa=ttfa)
    res = _ok(status, t0, ttft=ttfa, ttfa=ttfa, audio_chunks=len(arrivals), arrivals=arrivals)
    res["audio_bytes"] = nbytes
    return res


async def _asr_transcribe_one(client, model, prompt, wl, timeout_s):
    """Audio-in -> text-out. POST multipart to /v1/audio/transcriptions."""
    url = "http://localhost:8000/v1/audio/transcriptions"
    audio_path = Path(wl.get("audio_file", str(SAMPLE_AUDIO)))
    t0 = time.perf_counter()
    status = 0
    try:
        files = {"file": (audio_path.name, audio_path.read_bytes(), "audio/wav")}
        data = {"model": model, "response_format": "json"}
        r = await client.post(url, files=files, data=data, timeout=timeout_s)
        status = r.status_code
        if status != 200:
            return _err(status, RuntimeError(r.text[:120]), t0)
        txt = r.json().get("text", "")
    except Exception as e:
        return _err(status, e, t0)
    total = time.perf_counter() - t0
    # non-streaming: ttft == total (whole transcript returned at once)
    return {"status": status, "ttft": total, "ttfa": None, "total": total,
            "text_chunks": 1 if txt else 0, "audio_chunks": 0, "frame_gap_max": None}


def _ok(status, t0, *, ttft=None, ttfa=None, text_chunks=0, audio_chunks=0, arrivals=None):
    total = time.perf_counter() - t0
    arrivals = arrivals or []
    gaps = [arrivals[i] - arrivals[i - 1] for i in range(1, len(arrivals))]
    return {"status": status, "ttft": ttft, "ttfa": ttfa, "total": total,
            "text_chunks": text_chunks, "audio_chunks": audio_chunks,
            "frame_gap_max": max(gaps) if gaps else None}


def _err(status, e, t0, **extra):
    r = {"status": status or -1, "error": str(e)[:140], "total": time.perf_counter() - t0, **extra}
    if status == 429:  # admission SHED: a deliberate policy drop, not a failure -> count as shed
        r["policy_rejected"] = True
    return r


CLIENTS = {"omni_chat": _omni_chat_one, "tts_speech": _tts_speech_one,
           "asr_transcribe": _asr_transcribe_one}


def _modalities_for(i, base_mods, speech_frac):
    """Exact-fraction interleaving: index i is speech iff its cumulative quota ticks."""
    if speech_frac is None:
        return base_mods
    if speech_frac <= 0:
        return ["text"]
    if speech_frac >= 1:
        return ["text", "audio"]
    is_speech = round((i + 1) * speech_frac) - round(i * speech_frac) == 1
    return ["text", "audio"] if is_speech else ["text"]


async def _drive(client_kind, model, wl, concurrency, num_requests, timeout_s,
                 speech_frac=None, think_time=0.0):
    """Closed-loop: num_requests through `concurrency` workers (semaphore-bounded)."""
    sem = asyncio.Semaphore(concurrency)
    results = []
    fn = CLIENTS[client_kind]

    async def one(i, client):
        async with sem:
            prompt = PROMPTS[i % len(PROMPTS)]
            if client_kind == "omni_chat":
                mods = _modalities_for(i, wl["modalities"], speech_frac)
                res = await fn(client, model, prompt, mods, wl.get("max_tokens", 32), timeout_s)
            else:
                res = await fn(client, model, prompt, wl, timeout_s)
            if think_time:
                await asyncio.sleep(think_time)
            results.append(res)

    async with httpx.AsyncClient() as client:  # shared client => connection reuse
        await asyncio.gather(*(one(i, client) for i in range(num_requests)))
    return results


async def _warmup_closedloop(client_kind, model, wl, concurrency, duration_s, timeout_s,
                             speech_frac=None):
    """CLOSED-LOOP warmup (does not count): keep `concurrency` requests in flight for
    `duration_s`, each worker firing the next only after its previous completes.

    Why closed- and not open-loop: an open-loop saturating warmup (rate > capacity, no
    admission control) builds an UNBOUNDED server queue — a no-policy arm reached a 499-deep
    backlog that never drained, so every subsequently-measured rate timed out to 0/0. A
    closed-loop warmup bounds in-flight to `concurrency`, so it drives the server hot (warming
    caches AND letting an adaptive controller complete its one-time cliff discovery before
    measurement) WITHOUT a runaway queue, and it self-drains: the final gather waits for each
    worker's last in-flight request, so the server is left clean for the measured sweep."""
    fn = CLIENTS[client_kind]
    start = time.perf_counter()

    async def worker(w, client):
        i = w
        while time.perf_counter() - start < duration_s:
            prompt = PROMPTS[i % len(PROMPTS)]
            try:
                if client_kind == "omni_chat":
                    mods = _modalities_for(i, wl["modalities"], speech_frac)
                    await fn(client, model, prompt, mods, wl.get("max_tokens", 32), timeout_s)
                else:
                    await fn(client, model, prompt, wl, timeout_s)
            except Exception:
                pass
            i += concurrency

    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=concurrency + 4)) as client:
        await asyncio.gather(*(worker(w, client) for w in range(concurrency)))


async def _wait_server_idle(max_wait_s=45.0, quiet_s=4.0):
    """Wait until the server quiesces (its per-batch log stops growing) so each measured point
    starts from a CLEAN queue. Without this, a no-policy arm's leftover backlog from one rate
    contaminates the next (cross-point confound — and a warmup/earlier overload can leave a
    deep queue that times every later request out to 0/0). Best-effort + capped: if the server
    never drains within the cap, the point is flagged backlog-contaminated rather than hanging."""
    log = HERE / "results" / "_server.log"
    t0 = time.perf_counter()
    last_size, last_change = -1, t0
    while time.perf_counter() - t0 < max_wait_s:
        try:
            sz = log.stat().st_size
        except OSError:
            sz = last_size
        now = time.perf_counter()
        if sz != last_size:
            last_size, last_change = sz, now
        elif now - last_change >= quiet_s:
            return now - t0, True  # log quiet for quiet_s => server idle (drained)
        await asyncio.sleep(0.5)
    print(f"  [drain] server still busy after {max_wait_s:.0f}s — point flagged "
          f"backlog-contaminated", flush=True)
    return max_wait_s, False  # never drained within the cap


async def _drive_openloop(client_kind, model, wl, rate, duration_s, timeout_s,
                          speech_frac=None, arrival="poisson", max_outstanding=4000, rep=0):
    """OPEN-LOOP driver (C2): fire requests at a target arrival `rate` (req/s) DECOUPLED
    from completion, for `duration_s`. This is the regime admission/scheduling policies
    target — sustained offered load can exceed service capacity, so the server queue grows
    and tail latency / goodput become the signal. `max_outstanding` is only a memory safety
    cap (set high so it does NOT throttle the offered load). Returns (results, wall, contaminated)
    where `contaminated` is True if the pre-measurement drain timed out (the server carried a
    backlog into this point — cross-point confound, flagged so the report can mark it)."""
    fn = CLIENTS[client_kind]
    results = []
    sem = asyncio.Semaphore(max_outstanding)
    # Quiesce the server so this point measures from a clean queue (cross-point fairness).
    _, drained = await _wait_server_idle()
    start = time.perf_counter()

    async def fire(idx, arrival_t, client):
        async with sem:
            prompt = PROMPTS[idx % len(PROMPTS)]
            if client_kind == "omni_chat":
                mods = _modalities_for(idx, wl["modalities"], speech_frac)
                res = await fn(client, model, prompt, mods, wl.get("max_tokens", 32), timeout_s)
            else:
                res = await fn(client, model, prompt, wl, timeout_s)
            res["arrival"] = arrival_t
            results.append(res)

    # Seed the arrival process DETERMINISTICALLY per (rate, repeat) — not Python's salted hash(),
    # which differs across processes and is not reproducible (audit M2). Paired arms (A vs B) use
    # the same (rate, rep) so they see the SAME arrival sequence (fair A/B), while DIFFERENT
    # repeats get DIFFERENT sequences so the 3-repeat ±stdev reflects real Poisson-arrival
    # variance, not just server noise at one fixed realization (audit M1).
    rng = random.Random(int(round(rate * 1000)) * 1000 + rep)
    tasks = []
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=max_outstanding)) as client:
        i = 0
        while time.perf_counter() - start < duration_s:
            tasks.append(asyncio.create_task(fire(i, time.perf_counter() - start, client)))
            i += 1
            gap = rng.expovariate(rate) if arrival == "poisson" else 1.0 / rate
            await asyncio.sleep(gap)
        # Bounded drain: don't wait out a collapsed backlog forever (a no-policy collapse can
        # take ~10× the window to drain). Cancel stragglers past the cap — this also exercises
        # the gate's cancellation path. Steady-state metrics use in-window completions anyway.
        if tasks:
            done, pending = await asyncio.wait(tasks, timeout=duration_s)
            for t in pending:
                t.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
    return results, (time.perf_counter() - start), (not drained)


def _pctile(xs, q):
    xs = sorted(x for x in xs if x is not None)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))] if xs else None


def _summarize_openloop(results, wall_s, duration_s, slo_ttfa=None, slo_total=None):
    """Open-loop metrics: offered vs achieved rate, tail latency, goodput@SLO, shed-rate.
    offered = fired / arrival-window (the load we PUSHED). achieved = the STEADY-STATE service
    rate = completions that finished WITHIN the arrival window / window — NOT ok/wall, because
    `wall` is inflated by the post-window backlog drain (a collapsed no-policy arm can drain for
    ~10× the window, which would deflate its achieved and unfairly flatter the policy — audit H2).
    'shed' = requests rejected BY POLICY (distinct from errors/timeouts) — S2; 0 until a
    shedding policy exists (FifoGate only queues)."""
    ok = [r for r in results if r.get("status") == 200 and not r.get("error")]
    shed = [r for r in results if r.get("policy_rejected")]
    errs = [r for r in results if r.get("error") and not r.get("policy_rejected")]
    ttfas = [r.get("ttfa") for r in ok if r.get("ttfa") is not None]
    totals = [r["total"] for r in ok if r.get("total") is not None]
    # completions that landed within the arrival window (steady-state service rate)
    in_window = [r for r in ok if r.get("arrival") is not None and r.get("total") is not None
                 and (r["arrival"] + r["total"]) <= duration_s]
    out = {
        "offered_rps": len(results) / duration_s if duration_s else None,
        "achieved_rps": len(in_window) / duration_s if duration_s else None,
        "achieved_overall_rps": len(ok) / wall_s if wall_s else None,  # incl. drain (secondary)
        "ttfa_p50": _pctile(ttfas, 0.5), "ttfa_p95": _pctile(ttfas, 0.95), "ttfa_p99": _pctile(ttfas, 0.99),
        "total_p50": _pctile(totals, 0.5), "total_p95": _pctile(totals, 0.95), "total_p99": _pctile(totals, 0.99),
        "ok": len(ok), "total": len(results), "shed": len(shed), "errors_n": len(errs),
    }
    # goodput = on-time-AND-in-window completions per second. SAME population (in_window) and
    # denominator (duration_s) as achieved, so goodput <= achieved by construction (audit H2:
    # the old version used the drain-inclusive `ok` population over `wall_s`, which is a different
    # rate base and could exceed achieved). For a ttfa SLO, fall back to ttft for text-only
    # requests (no audio first-token) so the cheap requests that trivially meet SLO aren't dropped
    # from the numerator (audit M3, e.g. omni_mixed).
    def _resp(r):  # responsiveness latency for a ttfa-style SLO
        return r.get("ttfa") if r.get("ttfa") is not None else r.get("ttft")
    if slo_ttfa is not None:
        out["slo_ttfa"] = slo_ttfa
        g = sum(1 for r in in_window if _resp(r) is not None and _resp(r) <= slo_ttfa)
        out["goodput_rps"] = g / duration_s if duration_s else None
    elif slo_total is not None:
        out["slo_total"] = slo_total
        g = sum(1 for r in in_window if r.get("total") is not None and r["total"] <= slo_total)
        out["goodput_rps"] = g / duration_s if duration_s else None
    out["errors"] = [r.get("error") for r in errs][:3]
    return out


_AGG_KEYS = ["ttft_mean", "ttfa_mean", "ttfa_p95", "total_mean", "frame_gap_max", "throughput_rps"]
_AGG_KEYS_OPENLOOP = ["offered_rps", "achieved_rps", "ttfa_p50", "ttfa_p95", "ttfa_p99",
                      "total_p50", "total_p95", "total_p99", "goodput_rps"]


def _agg_repeats(summaries, keys=_AGG_KEYS):
    """Aggregate N repeat summaries -> mean + sample stdev per metric (stability signal)."""
    out = {"repeats": len(summaries)}
    for k in keys:
        vals = [s[k] for s in summaries if s.get(k) is not None]
        out[k] = statistics.fmean(vals) if vals else None
        out[k + "_std"] = statistics.stdev(vals) if len(vals) >= 2 else 0.0
    out["ok"] = sum(s["ok"] for s in summaries)
    out["total"] = sum(s["total"] for s in summaries)
    out["shed"] = sum(s.get("shed", 0) for s in summaries)
    out["errors"] = [e for s in summaries for e in s.get("errors", [])][:3]
    out["contaminated"] = sum(1 for s in summaries if s.get("contaminated"))  # # of repeats that
    #                                                started on an undrained backlog (cross-point)
    return out


def _summarize(results, wall_s):
    def m(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None
    ok = [r for r in results if r.get("status") == 200 and not r.get("error")]
    ttfas = sorted(r["ttfa"] for r in ok if r.get("ttfa") is not None)
    gaps = [r.get("frame_gap_max") for r in ok if r.get("frame_gap_max") is not None]
    return {
        "ok": len(ok), "total": len(results),
        "ttft_mean": m([r.get("ttft") for r in ok]),
        "ttfa_mean": m([r.get("ttfa") for r in ok]),
        "ttfa_p95": ttfas[min(len(ttfas) - 1, int(round(0.95 * (len(ttfas) - 1))))] if ttfas else None,
        "total_mean": m([r.get("total") for r in ok]),
        "frame_gap_max": max(gaps) if gaps else None,        # true worst-case jitter
        "throughput_rps": (len(ok) / wall_s) if wall_s > 0 else None,  # CLOSED-LOOP rps
        "errors": [r.get("error") for r in results if r.get("error")][:3],
    }


# --------------------------------------------------------------------------- #
# Optional nsys GR-active / SM-active under load
# --------------------------------------------------------------------------- #
def profile_gpu(gpu: int, out_prefix: Path, duration: int = 12) -> dict | None:
    rep, sqlite = out_prefix.with_suffix(".nsys-rep"), out_prefix.with_suffix(".sqlite")
    try:
        subprocess.run(
            ["nsys", "profile", f"--gpu-metrics-devices={gpu}", "--gpu-metrics-frequency=10000",
             f"--duration={duration}", "-o", str(out_prefix), "--force-overwrite", "true",
             "sleep", str(duration + 1)],
            check=False, capture_output=True, env={**os.environ, "TMPDIR": "/tmp"}, timeout=duration + 60)
        subprocess.run(["nsys", "export", "--type", "sqlite", "--force-overwrite", "true",
                        "-o", str(sqlite), str(rep)], check=False, capture_output=True)
        r = subprocess.run([sys.executable, str(ROOT / "dev/pipeline-coscheduling/sm_parse.py"), str(sqlite)],
                           capture_output=True, text=True, check=False)
        out = {}
        for ln in r.stdout.splitlines():
            if "SMs Active" in ln:
                out["sm_active"] = _last_pct(ln)
            elif "GR Active" in ln:
                out["gr_active"] = _last_pct(ln)
        return out or None
    except Exception as e:
        return {"profile_error": str(e)[:120]}


def _last_pct(line):
    toks = [t for t in line.replace("%", "").split() if _isfloat(t)]
    return float(toks[-1]) if toks else None


def _isfloat(s):
    try:
        float(s); return True
    except ValueError:
        return False


# --------------------------------------------------------------------------- #
# Scenario execution
# --------------------------------------------------------------------------- #
async def _point(client_kind, model, wl, C, num_requests, timeout_s, repeats,
                 speech_frac=None, think_time=0.0, profile=False, gpu=0, prof_prefix=None):
    """Run one (concurrency[,frac]) point `repeats` times -> aggregated mean±std point."""
    summaries, gpu_m = [], None
    for rep in range(repeats):
        prof_task = None
        if profile and rep == 0 and prof_prefix is not None:  # profile once (first repeat)
            prof_task = asyncio.get_running_loop().run_in_executor(None, profile_gpu, gpu, prof_prefix)
        t0 = time.perf_counter()
        res = await _drive(client_kind, model, wl, C, num_requests, timeout_s,
                           speech_frac=speech_frac, think_time=think_time)
        summaries.append(_summarize(res, time.perf_counter() - t0))
        if prof_task is not None:
            gpu_m = await prof_task
    agg = _agg_repeats(summaries)
    if gpu_m:
        agg["gpu"] = gpu_m
    return agg


async def _point_openloop(client_kind, model, wl, rate, duration_s, timeout_s, repeats,
                          speech_frac=None, arrival="poisson", slo_ttfa=None, slo_total=None):
    """One open-loop point (arrival rate `rate`) repeated -> aggregated mean±std."""
    summaries = []
    for rep in range(repeats):
        res, wall, contaminated = await _drive_openloop(
            client_kind, model, wl, rate, duration_s, timeout_s,
            speech_frac=speech_frac, arrival=arrival, rep=rep)
        s = _summarize_openloop(res, wall, duration_s, slo_ttfa=slo_ttfa, slo_total=slo_total)
        s["contaminated"] = contaminated
        summaries.append(s)
    return _agg_repeats(summaries, keys=_AGG_KEYS_OPENLOOP)


async def _run_openloop(scn, model, defaults, out, repeats, timeout_s):
    """Open-loop rate-sweep: the regime where admission/scheduling policies matter (C2)."""
    wl, load = scn["workload"], scn["load"]
    client_kind = wl["client"]
    dur = float(load.get("duration_s", 30))
    arrival = load.get("arrival", "poisson")
    slo_ttfa, slo_total = load.get("slo_ttfa"), load.get("slo_total")
    fr = wl.get("speech_frac")
    rates = load["rate"]
    # Warmup (does not count). CLOSED-LOOP saturation — bounded in-flight, NOT an open-loop
    # rate (which would build an unbounded queue and contaminate the no-policy arm; see
    # _warmup_closedloop). Saturating long enough lets an adaptive admission controller complete
    # its one-time cliff discovery (overshoot -> snap-back -> converge) BEFORE measurement, so
    # every measured point reflects the converged steady state, not the cold-start transient.
    # Identical for both arms (fair); the no-policy arm just warms caches and self-drains clean.
    warm_dur = min(40.0, max(25.0, dur))
    await _warmup_closedloop(client_kind, model, wl, 32, warm_dur, timeout_s, speech_frac=fr)
    for R in rates:
        agg = await _point_openloop(client_kind, model, wl, R, dur, timeout_s, repeats,
                                    speech_frac=fr, arrival=arrival, slo_ttfa=slo_ttfa, slo_total=slo_total)
        out["points"].append({"rate": R, **agg})
        contam = f" CONTAM={agg['contaminated']}/{agg['repeats']}" if agg.get("contaminated") else ""
        print(f"    R={R}/s: offered={_pm(agg,'offered_rps')} achieved={_pm(agg,'achieved_rps')} "
              f"ttfaP95={_pm(agg,'ttfa_p95')} ttfaP99={_pm(agg,'ttfa_p99')} "
              f"goodput={_pm(agg,'goodput_rps')} shed={agg.get('shed')} ok={agg['ok']}/{agg['total']}{contam}",
              flush=True)
    return out


async def run_scenario(scn, model, gpu, defaults, profile):
    wl, load = scn["workload"], scn["load"]
    timeout_s = defaults.get("timeout_s", 600)
    repeats = int(load.get("repeats", defaults.get("repeats", 3)))
    client_kind = wl["client"]
    mode = load.get("mode", "closed")
    think_time = float(wl.get("think_time", 0.0))
    out = {"id": scn["id"], "server": scn["server"], "tags": scn.get("tags", []),
           "workload": wl, "mode": mode, "repeats": repeats,
           "env": _env_snapshot(gpu),  # C1: record box/GPU context for fairness judgement
           "policy": _POLICY_ENV.get("SGLANG_OMNI_ADMISSION_POLICY"),
           "policy_args": _POLICY_ENV.get("SGLANG_OMNI_ADMISSION_ARGS"), "points": []}
    if mode == "openloop":
        return await _run_openloop(scn, model, defaults, out, repeats, timeout_s)

    nreq = load["num_requests"]
    frac_sweep = wl.get("speech_frac_sweep")

    # warmup (does not count)
    await _drive(client_kind, model, wl, min(4, load["concurrency"][0]),
                 defaults.get("warmup_requests", 8), timeout_s,
                 speech_frac=(frac_sweep[len(frac_sweep) // 2] if frac_sweep else None))

    if frac_sweep:
        C = load["concurrency"][0]
        for fr in frac_sweep:
            agg = await _point(client_kind, model, wl, C, nreq, timeout_s, repeats,
                               speech_frac=fr, think_time=think_time)
            out["points"].append({"speech_frac": fr, "concurrency": C, **agg})
            print(f"    frac={fr} C={C}: rps={_pm(agg,'throughput_rps')} "
                  f"ttfa={_pm(agg,'ttfa_mean')} total={_pm(agg,'total_mean')} (n={repeats})", flush=True)
    else:
        peak = max(load["concurrency"])
        for C in load["concurrency"]:
            pfx = (_RUN_DIR / f"{scn['id']}_C{C}_sm") if (profile and C == peak) else None
            agg = await _point(client_kind, model, wl, C, nreq, timeout_s, repeats,
                               think_time=think_time, profile=profile, gpu=gpu, prof_prefix=pfx)
            point = {"concurrency": C, **agg}
            out["points"].append(point)
            g = point.get("gpu", {})
            print(f"    C={C}: rps={_pm(agg,'throughput_rps')} ttft={_pm(agg,'ttft_mean')} "
                  f"ttfa={_pm(agg,'ttfa_mean')} total={_pm(agg,'total_mean')} "
                  f"gap={_pm(agg,'frame_gap_max')} ok={agg['ok']}/{agg['total']}"
                  + (f" SM={g.get('sm_active')}% GR={g.get('gr_active')}%" if g else ""), flush=True)
    return out


def _pm(agg, k):
    """format mean±std."""
    m, s = agg.get(k), agg.get(k + "_std")
    if not isinstance(m, (int, float)):
        return "—"
    return f"{m:.3f}±{s:.3f}" if isinstance(s, (int, float)) and s else f"{m:.3f}"


def _fmt(x):
    return f"{x:.3f}" if isinstance(x, (int, float)) else "—"


# --------------------------------------------------------------------------- #
# Regression
# --------------------------------------------------------------------------- #
def _pkey(p):
    if "rate" in p:
        return f"R{p['rate']}"
    if "speech_frac" in p:
        return f"frac{p['speech_frac']}"
    return f"C{p.get('concurrency')}"


def regression_compare(run_dir, tol, metrics):
    rows = []
    for f in sorted(run_dir.glob("*.json")):
        cur = json.loads(f.read_text())
        if cur.get("skipped") or not cur.get("points"):
            rows.append((f.stem, "SKIP", cur.get("skipped", "no points")))
            continue
        base_f = BASELINES / f.name
        if not base_f.exists():
            rows.append((f.stem, "NO-BASELINE", ""))
            continue
        base = json.loads(base_f.read_text())
        if not base.get("points"):
            rows.append((f.stem, "NO-BASELINE", "baseline has no points"))
            continue
        verdict, details = "PASS", []
        bpts = {_pkey(p): p for p in base["points"]}
        for p in cur["points"]:
            bp = bpts.get(_pkey(p))
            if not bp:
                continue
            for mk in metrics:
                cv, bv = p.get(mk), bp.get(mk)
                if cv is None or bv is None or bv == 0:
                    continue
                drift = (cv - bv) / bv
                # S3 variance-aware: FAIL only if the drift exceeds tol AND the absolute
                # change exceeds 2× the baseline's sample stdev — so box noise (large stdev)
                # doesn't trip a FAIL, and a clean low-variance shift isn't excused.
                bstd = bp.get(mk + "_std") or 0.0
                if abs(drift) > tol and abs(cv - bv) > 2 * bstd:
                    verdict = "FAIL"
                    details.append(f"{_pkey(p)}:{mk} {bv:.3f}±{bstd:.3f}->{cv:.3f} ({drift:+.0%})")
        rows.append((f.stem, verdict, "; ".join(details[:4])))
    return rows


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter", help="scenarios whose id contains / tags include this")
    ap.add_argument("--scenario", help="single scenario id")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--profile", action="store_true", help="add nsys GR/SM at peak concurrency")
    ap.add_argument("--regression", action="store_true", help="compare to baselines/, PASS/FAIL")
    ap.add_argument("--update-baseline", action="store_true")
    ap.add_argument("--policy", help="dotted path to an admission policy injected into every "
                                     "server (vs a stored baseline; weaker — see --ab-policy)")
    ap.add_argument("--policy-args", help='JSON kwargs for the policy, e.g. {"max_inflight":8}')
    ap.add_argument("--ab-policy", help="SAME-SESSION paired A/B: run each server twice back-to-"
                                        "back (A=no policy, B=this policy) — controls the shared-"
                                        "box temporal confound. Dotted path to the policy.")
    ap.add_argument("--ab-policy-args", help='JSON kwargs for the --ab-policy policy')
    ap.add_argument("--ab-sweep-args", help='JSON LIST of kwargs dicts — sweep the policy over '
                                            'each (1 A arm + one B arm per variant, same session). '
                                            'e.g. \'[{"max_inflight":4},{"max_inflight":8},{"max_inflight":16}]\'')
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.policy:  # inject into every server launch; servers read it at coordinator init
        _POLICY_ENV["SGLANG_OMNI_ADMISSION_POLICY"] = args.policy
        if args.policy_args:
            _POLICY_ENV["SGLANG_OMNI_ADMISSION_ARGS"] = args.policy_args
        print(f"[policy] A/B overlay: {args.policy} {args.policy_args or ''}", flush=True)

    cfg = yaml.safe_load((HERE / "scenarios.yaml").read_text())
    servers, defaults, scns = cfg["servers"], cfg.get("defaults", {}), cfg["scenarios"]
    if args.scenario:
        scns = [s for s in scns if s["id"] == args.scenario]
    elif args.filter:
        scns = [s for s in scns if args.filter in s["id"] or args.filter in s.get("tags", [])]

    if args.list:
        for s in scns:
            sv = servers[s["server"]]
            avail = (HF_HUB / sv["requires_weights"]).exists()
            print(f"  {'[ok]  ' if avail else '[skip]'} {s['id']:24s} server={s['server']:16s} "
                  f"client={s['workload']['client']:14s} tags={s.get('tags')}")
        return

    if not scns:  # C1: a 0-match filter must NOT silently exit green
        print(f"ERROR: no scenarios matched (filter={args.filter!r} scenario={args.scenario!r})",
              flush=True)
        sys.exit(2)

    global _RUN_DIR
    run_id = time.strftime("%Y%m%d_%H%M%S")
    _RUN_DIR = RESULTS / run_id
    _RUN_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=== testbed run {run_id}  ({len(scns)} scenarios) ===", flush=True)

    by_server = {}
    for s in scns:
        by_server.setdefault(s["server"], []).append(s)

    def _exec_group(server_name, server_scns, suffix=""):
        spec = servers[server_name]
        def _skip(reason):
            for s in server_scns:
                (_RUN_DIR / f"{s['id']}{suffix}.json").write_text(json.dumps(
                    {"id": s["id"], "server": server_name, "skipped": reason}, indent=2))
        if not (HF_HUB / spec["requires_weights"]).exists():
            print(f"\n## {server_name}{suffix}: SKIPPED (weights not cached: {spec['requires_weights']})", flush=True)
            _skip("weights_not_cached"); return
        model = _model_name(spec)
        print(f"\n## {server_name}{suffix}  (model={model})", flush=True)
        if not launch_server(spec, args.gpu):
            print(f"   server failed to start; SKIP {len(server_scns)} scenarios", flush=True)
            _skip("server_failed_to_start"); return
        try:
            for s in server_scns:
                print(f"  >> {s['id']}{suffix}", flush=True)
                try:
                    out = asyncio.run(run_scenario(s, model, args.gpu, defaults, args.profile))
                except Exception as e:  # isolate a bad scenario; keep the rest of this server
                    out = {"id": s["id"], "server": server_name, "error": str(e)[:200], "points": []}
                    print(f"     scenario errored: {str(e)[:160]}", flush=True)
                (_RUN_DIR / f"{s['id']}{suffix}.json").write_text(json.dumps(out, indent=2))
        finally:
            _kill_servers()

    if args.ab_policy:
        # Same-session paired A/B (C1): for each server, run arm A (no policy) then one B arm
        # per policy variant BACK-TO-BACK on the same GPU, so the temporal/box-load confound
        # between arms is minutes (not the hours/different-session gap of a stored baseline).
        if args.ab_sweep_args:
            variants = json.loads(args.ab_sweep_args)          # [ {kwargs}, ... ]
        elif args.ab_policy_args:
            variants = [json.loads(args.ab_policy_args)]
        else:
            variants = [{}]
        print(f"[ab] paired A=baseline vs B={args.ab_policy} variants={variants}", flush=True)
        for server_name, server_scns in by_server.items():
            _POLICY_ENV.clear()
            _exec_group(server_name, server_scns, "__A")
            for bi, vargs in enumerate(variants):
                _POLICY_ENV.clear()
                _POLICY_ENV["SGLANG_OMNI_ADMISSION_POLICY"] = args.ab_policy
                _POLICY_ENV["SGLANG_OMNI_ADMISSION_ARGS"] = json.dumps(vargs)
                _exec_group(server_name, server_scns, f"__B{bi}" if len(variants) > 1 else "__B")
            _POLICY_ENV.clear()
    else:
        for server_name, server_scns in by_server.items():
            _exec_group(server_name, server_scns, "")

    _write_report(_RUN_DIR)
    if args.ab_policy:
        _write_paired_report(_RUN_DIR, args.ab_policy)
    if args.update_baseline:
        BASELINES.mkdir(exist_ok=True)
        n = 0
        for f in _RUN_DIR.glob("*.json"):
            d = json.loads(f.read_text())
            pts = d.get("points") or []
            # R1 quality gate: only bake a scenario whose EVERY point met the ok-ratio
            # floor — a degraded run (contention, partial failures) must not become the
            # fixed reference that all future experiments compare against.
            healthy = bool(pts) and all(
                p.get("total") and (p.get("ok", 0) / p["total"]) >= MIN_OK_RATIO for p in pts
            )
            if healthy:
                (BASELINES / f.name).write_text(f.read_text()); n += 1
            elif pts:
                worst = min((p.get("ok", 0) / p["total"]) for p in pts if p.get("total"))
                print(f"  NOT baselined (ok-ratio {worst:.0%} < {MIN_OK_RATIO:.0%}): {f.stem}", flush=True)
        print(f"\nbaselines updated: {n} scenarios from {run_id}", flush=True)
    if args.regression:
        tol = defaults.get("regression_tol", 0.2)
        metrics = defaults.get("regression_metrics", ["throughput_rps"])
        print(f"\n=== REGRESSION (vs baselines/, tol ±{tol:.0%}) ===", flush=True)
        any_fail, compared = False, 0
        for name, verdict, det in regression_compare(_RUN_DIR, tol, metrics):
            print(f"  {verdict:12s} {name}  {det}", flush=True)
            any_fail = any_fail or verdict == "FAIL"
            compared += verdict in ("PASS", "FAIL")
        if compared == 0:  # C1: nothing actually checked -> not a green pass
            print("  WARNING: 0 points compared (no matching baselines) — not a pass", flush=True)
            sys.exit(3)
        sys.exit(1 if any_fail else 0)


def _model_name(spec):
    if spec.get("model_path"):
        return spec["model_path"]
    for ln in (ROOT / spec["config"]).read_text().splitlines():
        if ln.strip().startswith("model_path:"):
            return ln.split("model_path:", 1)[1].strip()
    return "default"


def _write_report(run_dir):
    lines = [f"# testbed report {run_dir.name}", "",
             "_throughput_rps is CLOSED-LOOP (bounded by concurrency/latency); use for "
             "relative regression, not absolute capacity._", ""]
    for f in sorted(run_dir.glob("*.json")):
        d = json.loads(f.read_text())
        if d.get("skipped"):
            lines.append(f"- **{d['id']}**: SKIPPED ({d['skipped']})"); continue
        if d.get("error") and not d.get("points"):
            lines.append(f"- **{d['id']}**: ERROR ({d['error'][:80]})"); continue
        env = d.get("env", {})
        envstr = (f" env[load={env.get('loadavg1')} gpu_util={env.get('gpu_util_pct')}% "
                  f"gpu_procs={env.get('gpu_procs')}]") if env else ""
        pol = f" policy={d['policy']}" if d.get("policy") else ""
        lines.append(f"\n## {d['id']}  (server={d['server']}, mode={d.get('mode','closed')}, "
                     f"n={d.get('repeats', 1)}; mean±stdev){pol}{envstr}")
        if d.get("mode") == "openloop":
            lines.append("| rate | offered | achieved | ttfa_p95 | ttfa_p99 | total_p99 | goodput | shed | ok |")
            lines.append("|--|--|--|--|--|--|--|--|--|")
            for p in d.get("points", []):
                lines.append(f"| {_pkey(p)} | {_pm(p,'offered_rps')} | {_pm(p,'achieved_rps')} "
                             f"| {_pm(p,'ttfa_p95')} | {_pm(p,'ttfa_p99')} | {_pm(p,'total_p99')} "
                             f"| {_pm(p,'goodput_rps')} | {p.get('shed','—')} | {p.get('ok')}/{p.get('total')} |")
        else:
            lines.append("| point | rps | ttft | ttfa | total | gap_max | SM% | GR% | ok |")
            lines.append("|--|--|--|--|--|--|--|--|--|")
            for p in d.get("points", []):
                g = p.get("gpu", {})
                lines.append(f"| {_pkey(p)} | {_pm(p,'throughput_rps')} | {_pm(p,'ttft_mean')} "
                             f"| {_pm(p,'ttfa_mean')} | {_pm(p,'total_mean')} "
                             f"| {_pm(p,'frame_gap_max')} | {g.get('sm_active','—')} "
                             f"| {g.get('gr_active','—')} | {p.get('ok')}/{p.get('total')} |")
    (run_dir / "report.md").write_text("\n".join(lines))
    print(f"\nreport: {run_dir / 'report.md'}", flush=True)


def _write_paired_report(run_dir, policy):
    """Same-session paired A (no policy) vs B (policy) delta report. Compares the __A and
    __B arms point-by-point — both arms ran back-to-back on the same GPU this session, so
    the delta is attributable to the policy, not to cross-session box drift (C1)."""
    # which metrics to delta, by mode
    OPEN = ["achieved_rps", "ttfa_p95", "ttfa_p99", "total_p99", "goodput_rps"]
    CLOSED = ["throughput_rps", "ttfa_mean", "total_mean", "frame_gap_max"]
    lines = [f"# paired A/B report {run_dir.name}", "",
             f"A = baseline (no policy)  ·  B = {policy}", "",
             "Δ = (B − A)/A per point; all arms ran back-to-back same-session/same-GPU.", ""]
    for fa in sorted(run_dir.glob("*__A.json")):
        sid = fa.name[:-len("__A.json")]
        da = json.loads(fa.read_text())
        # one or many B arms (a policy-args sweep => __B0, __B1, ...; single => __B)
        b_files = sorted(run_dir.glob(f"{sid}__B*.json"))
        if da.get("skipped") or not da.get("points") or not b_files:
            lines.append(f"- **{sid}**: SKIPPED/incomplete"); continue
        mode = da.get("mode", "closed")
        mks = OPEN if mode == "openloop" else CLOSED
        ea = da.get("env", {})
        lines.append(f"\n## {sid}  (mode={mode})  envA[load={ea.get('loadavg1')},"
                     f"gpu_procs={ea.get('gpu_procs')}]")
        apts = {_pkey(p): p for p in da["points"]}
        for fb in b_files:
            db = json.loads(fb.read_text())
            if db.get("skipped") or not db.get("points"):
                continue
            eb = db.get("env", {})
            blabel = db.get("policy_args") or "B"
            lines.append(f"\n### B = {blabel}  envB[load={eb.get('loadavg1')},"
                         f"gpu_procs={eb.get('gpu_procs')}]")
            lines.append("| point | metric | A | B | Δ |")
            lines.append("|--|--|--|--|--|")
            for pb in db["points"]:
                pa = apts.get(_pkey(pb))
                if not pa:
                    continue
                for mk in mks:
                    a, b = pa.get(mk), pb.get(mk)
                    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)) or a == 0:
                        continue
                    lines.append(f"| {_pkey(pb)} | {mk} | {a:.3f} | {b:.3f} | {(b-a)/a:+.0%} |")
    (run_dir / "paired_report.md").write_text("\n".join(lines))
    print(f"paired report: {run_dir / 'paired_report.md'}", flush=True)


if __name__ == "__main__":
    main()
