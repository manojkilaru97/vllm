# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for running sample_tokens on a helper thread that parks at a host sync."""

import contextvars
import threading

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip("CUDA required for deferred sampling", allow_module_level=True)

from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.gpu import deferred_sample as ds

_var: contextvars.ContextVar[str] = contextvars.ContextVar("deferred_test", default="")


@pytest.fixture
def runner():
    r = ds.DeferredSampleRunner(torch.device("cuda", torch.cuda.current_device()))
    yield r
    r.shutdown()
    assert ds._active_runner is None


def _output(tag: str) -> ModelRunnerOutput:
    return ModelRunnerOutput(req_ids=[tag], req_id_to_index={tag: 0})


def test_parks_at_host_sync_and_preserves_order(runner):
    release = threading.Event()
    events = []

    def body(x):
        on_helper = threading.current_thread() is not threading.main_thread()
        events.append(("body", on_helper))
        y = x * 2
        ds.notify_host_sync()
        ds.notify_host_sync()  # only the first sync parks
        release.wait()
        events.append(("item", int(y.sum().item())))
        return _output("a")

    try:
        out = runner.run(body, torch.ones(4, device="cuda"))
        assert isinstance(out, ds.DeferredSampleOutput)
        events.append(("main",))
    finally:
        release.set()
    ds.join_pending()
    assert events == [("body", True), ("main",), ("item", 8)]
    assert out.get_output().req_ids == ["a"]
    assert out.get_kv_connector_output() is None


def test_completes_inline_without_sync(runner):
    out = runner.run(lambda: _output("b"))
    assert isinstance(out, ModelRunnerOutput) and out.req_ids == ["b"]
    ds.notify_host_sync()  # no job on this thread: a no-op


def _parked(tag: str, gate: threading.Event):
    def body():
        ds.notify_host_sync()
        gate.wait()
        return _output(tag)

    return body


def test_old_output_does_not_wait_for_newer_step(runner):
    gate1, gate2 = threading.Event(), threading.Event()
    try:
        first = runner.run(_parked("1", gate1))
        assert isinstance(first, ds.DeferredSampleOutput)
        gate1.set()
        runner.join()
        second = runner.run(_parked("2", gate2))
        assert first.get_output().req_ids == ["1"]
        assert runner._pending is not None
        gate2.set()
        assert second.get_output().req_ids == ["2"]
        assert runner._pending is None
    finally:
        gate1.set()
        gate2.set()


def test_propagates_stream_and_context(runner):
    stream = torch.cuda.Stream()
    _var.set("main")
    seen = {}

    def body():
        seen["stream"] = torch.cuda.current_stream()
        seen["var"] = _var.get()
        seen["inference"] = torch.is_inference_mode_enabled()
        return _output("c")

    with torch.cuda.stream(stream), torch.inference_mode():
        runner.run(body)
    assert seen["stream"] == stream and seen["var"] == "main"
    assert seen["inference"] is False  # callers decorate the body themselves


def test_error_after_park_raises_at_join_and_output(runner):
    gate = threading.Event()

    def body():
        ds.notify_host_sync()
        gate.wait()
        raise ValueError("boom")

    try:
        out = runner.run(body)
        assert isinstance(out, ds.DeferredSampleOutput)
    finally:
        gate.set()
    with pytest.raises(ValueError, match="boom"):
        runner.join()
    with pytest.raises(ValueError, match="boom"):
        out.get_output()

    def early():
        raise ValueError("early")

    with pytest.raises(ValueError, match="early"):
        runner.run(early)
