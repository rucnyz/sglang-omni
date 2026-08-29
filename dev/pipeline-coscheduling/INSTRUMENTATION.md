# Phase 0 instrumentation spec (G1–G5)

The offline analyzer (`analyze_bubbles.py`) needs a few events the codebase does
**not** emit yet. All go through the existing `sglang_omni.profiler.event_recorder.emit`
(no-op when the recorder is off → zero overhead in normal serving). The recorder
is turned on per-run via the profiler control plane with an `event_dir`; events
land in `events_<stage>_<pid>.jsonl` and the analyzer merges them.

## Event contract (what the analyzer consumes)

| ID | event_name | request_id | stage | metadata | purpose |
|----|-----------|-----------|-------|----------|---------|
| G1 | `fwd_begin` / `fwd_end` | `"__stage__"` | the AR/compute stage | `{bs, mode, gpu}` | **BUSY** intervals per stage GPU |
| G2 | `bp_wait_begin` / `bp_wait_end` | the chunk's `request_id` | sender stage | `{to, credits}` | **BLOCKED** (relay-credit backpressure) |
| G3 | `q_sample` | `"__stage__"` | stage | `{waiting, running}` | queue depth (diagnostic; not required) |
| G4 | `buf_sample` | `"__stage__"` | consumer stage | `{stream_chunks, credits_in_use}` | buffer water-line (diagnostic) |
| G5 | — (CSV, not an event) | — | — | — | measured GPU util, via `nvml_sampler.py` |

Already-emitted lifecycle events the analyzer also uses (no change needed):
`request_admission` (admit) and `terminal_response` / `stage_complete` (done) →
the in-flight-request curve that splits idle into STARVED vs IDLE.

## Where to add each emit (verified file:line on current main)

### G1 — BUSY (the one that matters most)
Bracket the per-step forward in **each scheduler kind**:

- **AR stages (thinker, talker)** — `sglang_omni/scheduling/omni_scheduler.py:609`
  `_run_batch(self, batch, ...)`. This is the single chokepoint for an AR forward
  step and runs on the scheduler thread (active-stage bound). Wrap the body:
  ```python
  def _run_batch(self, batch, pp_proxy_tensors=None):
      _emit_event(request_id="__stage__", stage=None, event_name="fwd_begin",
                  metadata={"bs": len(getattr(batch, "reqs", []) or []),
                            "mode": "decode" if self._batch_is_decode(batch) else "prefill",
                            "gpu": self.gpu_id})
      try:
          return self._run_batch_inner(batch, pp_proxy_tensors)   # existing body
      finally:
          _emit_event(request_id="__stage__", stage=None, event_name="fwd_end",
                      metadata={"gpu": self.gpu_id})
  ```
  > For a clean first measurement, run with the **normal** event loop
  > (`enable_async_decode=False`, `enable_overlap=False`) so BUSY = one forward
  > per `_run_batch`. The async-decode loop (`:1041`) splits launch/resolve; if
  > used, additionally bracket `execute_launch`→`execute_resolve` instead.

- **Vocoder (code2wav)** — `models/qwen3_omni/components/code2wav_scheduler.py:206`
  `_decode_and_emit`. Bracket the decode call (this is the GPU work).

- **Encoders / preprocessing (SimpleScheduler)** — `scheduling/simple_scheduler.py`
  around `_run_single` / `_fn(payload)`. Optional for the first cut (the thesis is
  thinker/talker/vocoder), add when you want the full swimlane.

### G2 — BLOCKED (backpressure stall)
`sglang_omni/relay/shm.py:139`, in `ShmRelay.put_async`, around the credit wait:
```python
async def put_async(self, tensor, request_id=None, dst_rank=None):
    _emit_event(request_id=request_id or "__stage__", stage=None,
                event_name="bp_wait_begin", metadata={"credits": self._sem._value})
    await self._sem.acquire()
    _emit_event(request_id=request_id or "__stage__", stage=None,
                event_name="bp_wait_end")
    ...
```
Mirror the same two lines in the **other relay backends actually used** for the
cross-GPU edge in the separate-GPU config: `relay/nccl.py`, `relay/nixl.py`,
`relay/mooncake.py` (same credit/semaphore acquire site).

> **Config dependence — important.** Credit backpressure only bites on the
> **relay** path (cross-GPU). In the **co-location** config the thinker→talker
> edge is same-GPU CUDA IPC / LOCAL_OBJECT (no credit wait), so G2 will be ~0
> there — the bubble shows up instead as **interleaved BUSY gaps on the shared
> GPU** (G1 on both stages over one `gpu` id) plus the GIL-yield (`sleep`) in
> `omni_scheduler.py:895`. So:
> - **separate-GPU** (thinker GPU0 / talker GPU1): G2 is the smoking gun (B2).
> - **co-location** (same GPU): G1-interleaving + NVML (G5) is the signal (B5).
> Measure BOTH; they expose different mechanisms.

### G3 / G4 — diagnostics (optional)
- G3: top of `omni_scheduler._event_loop_normal` (`:902`), throttled to ~every
  20 ms: `emit("__stage__", "q_sample", {"waiting": len(self.waiting_queue),
  "running": len(self.running_batch.reqs)})`.
- G4: where talker accumulates chunks (`omni_scheduler.py:1151`
  `_append_stream_chunk_default`): emit deque length + relay credits-in-use.

### G5 — measured GPU util
Run `nvml_sampler.py --out gpu.csv --interval-ms 50` for the run window; align by
`timestamp_ns`. Used to cross-check that inferred BUSY (G1) tracks real SM util,
and to see contention on the shared GPU in the co-location config.

## Turning the recorder on for a run
The recorder is dormant until a `ProfilerStartMessage` with `event_dir` is
broadcast (see `profiler/profiler_control.py`). Easiest paths:
1. the profiler control endpoint the server already exposes, or
2. a one-line `get_recorder().start(run_id, event_dir, stage)` at stage startup
   for a throwaway measurement build.

## Analyzer usage
```bash
# after a run that produced events_*.jsonl in $DIR (+ optional gpu.csv):
python dev/pipeline-coscheduling/analyze_bubbles.py $DIR --json out.json --png swimlane.png
# verify analyzer logic without any sglang install:
python dev/pipeline-coscheduling/analyze_bubbles.py --selftest
```

## What each Phase-0 deliverable maps to
- **M1 swimlane** ← G1 (+G2 for blocked coloring, +G5 overlay)
- **M2 bubble%** ← G1 + G2 + admission/terminal (in-flight)
- **M3 backpressure stall** ← G2
- **M4 saturation curves** ← run the workload at a concurrency sweep, read M2 per point
- **M6 TTFA/jitter** ← existing `code2wav_first_audio` + per-frame emit (add later)
