# SPDX-License-Identifier: Apache-2.0
"""Integration tests for the admission seam at the Coordinator boundary.

The gate mechanics are unit-tested in tests/unit_test/test_admission.py; these tests pin the
*wiring*: that on_submit is consulted before a request enters the pipeline, that a SHED is
cleaned up and propagated (not marked FAILED), and that on_complete / slot release fires on
every exit path (completion, abort, and a failure after admission).
"""
from __future__ import annotations

import asyncio

import pytest

from sglang_omni.pipeline.admission import (
    AdmissionRejected,
    FifoGate,
    NoOpAdmission,
)
from sglang_omni.pipeline.coordinator import Coordinator
from sglang_omni.proto import CompleteMessage
from tests.unit_test.fixtures.pipeline_fakes import RecordingCoordinatorControlPlane


class _RecordingPolicy:
    """Counts on_submit / on_complete calls; admits everything."""

    def __init__(self):
        self.submitted: list[str] = []
        self.completed: list[str] = []

    async def on_submit(self, ctx):
        self.submitted.append(ctx.request_id)

    def on_complete(self, request_id):
        self.completed.append(request_id)


class _SheddingPolicy:
    """Always sheds (refuses admission)."""

    async def on_submit(self, ctx):
        raise AdmissionRejected("test shed")

    def on_complete(self, request_id):
        pass


class _FailingControlPlane(RecordingCoordinatorControlPlane):
    """submit_to_stage fails — simulates the request being admitted (slot taken) but never
    reaching the entry stage."""

    async def submit_to_stage(self, stage, endpoint, msg):
        raise RuntimeError("stage submit failed")


class _BlockingPolicy:
    """on_submit blocks until released — simulates a request queued at the gate."""

    def __init__(self):
        self.release = asyncio.Event()
        self.completed: list[str] = []

    async def on_submit(self, ctx):
        await self.release.wait()

    def on_complete(self, request_id):
        self.completed.append(request_id)


def _coordinator(policy, *, control_plane=None, terminal_stages=("decode",)):
    c = Coordinator(
        "inproc://complete",
        "inproc://abort",
        entry_stage="preprocess",
        terminal_stages=list(terminal_stages),
        admission_policy=policy,
    )
    c.control_plane = control_plane or RecordingCoordinatorControlPlane()
    c.register_stage("preprocess", "inproc://preprocess")
    return c


def test_shed_propagates_and_cleans_up_without_marking_failed():
    async def _run():
        c = _coordinator(_SheddingPolicy())
        with pytest.raises(AdmissionRejected):
            await c.submit("req-1", "hello")
        # Tracking undone, future cleaned up, never submitted to the entry stage.
        assert "req-1" not in c._requests
        assert "req-1" not in c._completion_futures
        assert c.control_plane.submitted == []

    asyncio.run(_run())


def test_stream_shed_propagates_on_first_iteration():
    async def _run():
        c = _coordinator(_SheddingPolicy())
        gen = c.stream("req-1", "hello")
        with pytest.raises(AdmissionRejected):
            await gen.__anext__()
        assert "req-1" not in c._requests

    asyncio.run(_run())


def test_on_complete_fires_once_on_normal_completion():
    async def _run():
        policy = _RecordingPolicy()
        c = _coordinator(policy)
        task = asyncio.create_task(c.submit("req-1", "hello"))
        await asyncio.sleep(0)  # let submit() reach the awaited completion future
        await c._handle_completion(
            CompleteMessage("req-1", "decode", True, result={"text": "hi"})
        )
        assert await task == {"text": "hi"}
        assert policy.submitted == ["req-1"]
        assert policy.completed == ["req-1"]  # exactly once

    asyncio.run(_run())


def test_abort_releases_admission_slot():
    async def _run():
        gate = FifoGate(2)
        c = _coordinator(gate)
        task = asyncio.create_task(c.submit("req-1", "hello"))
        await asyncio.sleep(0)
        assert gate._inflight == 1
        assert await c.abort("req-1") is True
        assert gate._inflight == 0  # slot released by abort
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0)  # submit()'s finally runs on_complete again (idempotent)
        assert gate._inflight == 0

    asyncio.run(_run())


def test_slot_released_when_submit_to_stage_fails_after_admission():
    """Regression: a request admitted by the gate (slot taken) but whose submit_to_stage then
    fails must still release the slot — else the in-flight count leaks and the gate wedges shut."""
    async def _run():
        gate = FifoGate(2)
        c = _coordinator(gate, control_plane=_FailingControlPlane())
        with pytest.raises(RuntimeError):
            await c.submit("req-1", "hello")
        assert gate._inflight == 0, f"slot leaked after submit failure: inflight={gate._inflight}"

    asyncio.run(_run())


def test_abort_while_queued_at_gate_does_not_submit_or_keyerror():
    """Regression: a request aborted while still waiting at the admission gate (it sits in
    _requests as PENDING across the on_submit await) must NOT be submitted to the entry stage
    when on_submit later returns, and must not KeyError updating its (already-popped) state."""
    async def _run():
        policy = _BlockingPolicy()
        c = _coordinator(policy)
        task = asyncio.create_task(c.submit("req-1", "hello"))
        await asyncio.sleep(0)  # task now blocked inside on_submit (queued at the gate)
        assert "req-1" in c._requests
        assert await c.abort("req-1") is True
        assert "req-1" not in c._requests  # abort popped tracking
        policy.release.set()  # on_submit returns "admitted"; _submit_request resumes
        with pytest.raises(asyncio.CancelledError):
            await task
        # The aborted request must never reach the entry stage (no orphaned submit, no KeyError).
        assert c.control_plane.submitted == []

    asyncio.run(_run())


def test_stream_success_releases_slot_once():
    async def _run():
        gate = FifoGate(2)
        c = _coordinator(gate)

        async def _consume():
            return [msg async for msg in c.stream("req-1", "hello")]

        task = asyncio.create_task(_consume())
        await asyncio.sleep(0)
        assert gate._inflight == 1
        await c._handle_completion(
            CompleteMessage("req-1", "decode", True, result={"text": "ok"})
        )
        await task
        assert gate._inflight == 0  # released exactly once via stream()'s finally

    asyncio.run(_run())


def test_noop_policy_matches_no_policy():
    """NoOpAdmission must be behaviorally identical to no policy at all (the safety claim that
    wiring the seam in changes nothing until a real policy is configured)."""
    async def _seq(policy):
        c = _coordinator(policy)
        task = asyncio.create_task(c.submit("req-1", "hello"))
        await asyncio.sleep(0)
        await c._handle_completion(
            CompleteMessage("req-1", "decode", True, result={"text": "ok"})
        )
        result = await task
        stages = [s[0] for s in c.control_plane.submitted]
        return result, stages, "req-1" in c._requests, "req-1" in c._completion_futures

    assert asyncio.run(_seq(None)) == asyncio.run(_seq(NoOpAdmission()))
