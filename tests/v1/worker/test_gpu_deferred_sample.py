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


def test_kv_connector_output_returned_while_parked(runner):
    gate = threading.Event()
    kv = object()

    def body():
        ds.publish_kv_connector_output(kv)
        ds.notify_host_sync()
        gate.wait()
        return _output("kv")

    try:
        out = runner.run(body)
        assert isinstance(out, ds.DeferredSampleOutput)
        assert out.get_kv_connector_output() is kv
        assert runner._pending is not None  # sampling still parked
    finally:
        gate.set()
    assert out.get_output().req_ids == ["kv"]


def test_unpublished_kv_connector_output_waits_for_step(runner):
    gate = threading.Event()

    def body():
        ds.notify_host_sync()
        gate.wait()
        out = _output("late")
        out.kv_connector_output = "late-kv"
        return out

    out = runner.run(body)
    gate.set()
    assert out.get_kv_connector_output() == "late-kv"


class _FakeConnector:
    """Records worker-side connector calls in the order they were made."""

    def __init__(self, log, lock):
        self.log, self.lock = log, lock

    def _rec(self, what):
        with self.lock:
            self.log.append(what)

    def pre_forward(self, step):
        self._rec(f"pre_forward {step}")

    def post_forward(self, step):
        self._rec(f"save {step}")
        self._rec(f"load {step}")
        return f"kv {step}"


def test_step_n_transfers_ordered_before_step_n1_forward(runner):
    # Mirrors model_runner: execute_model joins the helper before pre_forward.
    log, lock = [], threading.Lock()
    conn = _FakeConnector(log, lock)
    gates = {n: threading.Event() for n in range(3)}

    def sample_body(n):
        def body():
            ds.publish_kv_connector_output(conn.post_forward(n))
            ds.notify_host_sync()
            gates[n].wait()
            with lock:
                log.append(f"sampled {n}")
            return _output(str(n))

        return body

    outs = []
    try:
        for n in range(3):
            ds.join_pending()
            conn.pre_forward(n)
            with lock:
                log.append(f"forward {n}")
            out = runner.run(sample_body(n))
            assert out.get_kv_connector_output() == f"kv {n}"
            with lock:
                log.append(f"schedule {n + 1}")
            gates[n].set()
            outs.append(out)
    finally:
        for g in gates.values():
            g.set()
    ds.join_pending()
    for n in range(3):
        assert outs[n].get_output().req_ids == [str(n)]
        i = log.index
        assert (
            i(f"forward {n}") < i(f"save {n}") < i(f"load {n}") < i(f"schedule {n + 1}")
        )
        if n < 2:
            assert i(f"sampled {n}") < i(f"pre_forward {n + 1}") < i(f"forward {n + 1}")


def test_store_launched_on_helper_waits_for_step_forward(runner):
    # A store issued from the helper must see step N's forward output.
    x = torch.zeros(1 << 20, device="cuda")
    dst = torch.empty_like(x)
    store_stream = torch.cuda.Stream()

    def body():
        done = torch.cuda.Event()
        done.record(torch.cuda.current_stream())
        store_stream.wait_event(done)
        with torch.cuda.stream(store_stream):
            dst.copy_(x, non_blocking=True)
        ds.notify_host_sync()
        return _output("s")

    torch.cuda._sleep(50_000_000)  # long "forward" N
    x.fill_(1.0)
    runner.run(body)
    ds.join_pending()
    store_stream.synchronize()
    assert bool((dst == 1.0).all())


@pytest.mark.parametrize(
    "connector,extra,allowed",
    [
        (None, None, True),
        ("SimpleCPUOffloadConnector", {"lazy_offload": True}, True),
        ("SimpleCPUOffloadConnector", {"kv_offload_backend": "cpu"}, True),
        ("SimpleCPUOffloadConnector", {"kv_offload_backend": "disk"}, False),
        ("OffloadingConnector", {}, False),
        ("NixlConnector", None, False),
    ],
)
def test_kv_transfer_allows_deferral(connector, extra, allowed):
    from vllm.config import KVTransferConfig

    cfg = None
    if connector is not None:
        cfg = KVTransferConfig(
            kv_connector=connector,
            kv_role="kv_both",
            kv_connector_extra_config=extra or {},
        )
    assert ds.kv_transfer_allows_deferral(cfg) is allowed


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
