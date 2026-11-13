# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Let the engine-core thread return from sample_tokens at a blocking host sync
while a helper thread finishes the step; GPU submission order is unchanged."""

import contextvars
import queue
import threading
from collections.abc import Callable
from typing import Any

import torch

from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput

_local = threading.local()
_active_runner: "DeferredSampleRunner | None" = None


def notify_host_sync() -> None:
    """Called by a sampler right before a blocking device-to-host read."""
    job = getattr(_local, "job", None)
    if job is not None:
        _local.job = None
        job.handoff.set()


def publish_kv_connector_output(output: Any) -> None:
    """Called once the step's KV connector output is final, before the host sync."""
    job = getattr(_local, "job", None)
    if job is not None:
        job.kv_connector_output = output
        job.kv_published = True


def kv_transfer_allows_deferral(kv_transfer_config: Any) -> bool:
    """SimpleCPUOffload's CPU-backend worker hooks touch only worker state on the
    step's stream, and every later worker entry joins the deferred step first."""
    if kv_transfer_config is None:
        return True
    extra = kv_transfer_config.kv_connector_extra_config or {}
    return (
        kv_transfer_config.kv_connector == "SimpleCPUOffloadConnector"
        and extra.get("kv_offload_backend", "cpu") == "cpu"
    )


def join_pending() -> None:
    """Finish any deferred sampling step before touching the worker."""
    if _active_runner is not None:
        _active_runner.join()


class _Job:
    def __init__(self, fn: Callable[..., Any], args: tuple[Any, ...]) -> None:
        self.fn = fn
        self.args = args
        self.stream = torch.cuda.current_stream()
        self.context = contextvars.copy_context()
        self.handoff = threading.Event()
        self.finished = threading.Event()
        self.value: Any = None
        self.error: BaseException | None = None
        self.kv_connector_output: Any = None
        self.kv_published = False

    def run(self) -> None:
        _local.job = self
        try:
            with torch.cuda.stream(self.stream):
                self.value = self.context.run(self.fn, *self.args)
        except BaseException as e:
            self.error = e
        finally:
            _local.job = None
            self.finished.set()
            self.handoff.set()

    def result(self) -> Any:
        if self.error is not None:
            raise self.error
        return self.value


class DeferredSampleOutput(AsyncModelRunnerOutput):
    def __init__(self, runner: "DeferredSampleRunner", job: _Job) -> None:
        self._runner = runner
        self._job = job

    def _result(self) -> AsyncModelRunnerOutput | ModelRunnerOutput:
        self._runner.join(self._job)
        return self._job.result()

    def get_output(self) -> ModelRunnerOutput:
        output = self._result()
        if isinstance(output, AsyncModelRunnerOutput):
            return output.get_output()
        return output

    def get_kv_connector_output(self):
        # Published before the park, so this does not wait for sampling.
        if self._job.kv_published:
            return self._job.kv_connector_output
        output = self._result()
        if isinstance(output, AsyncModelRunnerOutput):
            return output.get_kv_connector_output()
        return output.kv_connector_output


class DeferredSampleRunner:
    """Runs sample_tokens bodies on one persistent helper thread."""

    def __init__(self, device: torch.device) -> None:
        global _active_runner
        self._device = device
        self._jobs: queue.SimpleQueue[_Job | None] = queue.SimpleQueue()
        self._pending: _Job | None = None
        self._thread = threading.Thread(
            target=self._loop, name="vllm-deferred-sample", daemon=True
        )
        self._thread.start()
        _active_runner = self

    def _loop(self) -> None:
        torch.cuda.set_device(self._device)
        while (job := self._jobs.get()) is not None:
            job.run()

    def run(self, fn: Callable[..., Any], *args: Any) -> Any:
        self.join()
        job = _Job(fn, args)
        self._jobs.put(job)
        job.handoff.wait()
        if job.finished.is_set():
            return job.result()
        self._pending = job
        return DeferredSampleOutput(self, job)

    def join(self, job: _Job | None = None) -> None:
        """Wait for the pending step, or only for ``job`` if it is still pending."""
        pending = self._pending
        if pending is None or (job is not None and job is not pending):
            return
        pending.finished.wait()
        self._pending = None
        pending.result()

    def shutdown(self) -> None:
        global _active_runner
        if self._pending is not None:
            self._pending.finished.wait()
            self._pending = None
        self._jobs.put(None)
        if _active_runner is self:
            _active_runner = None
