# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fail-open hardening for offload transfer / pending-job death paths."""

from unittest.mock import MagicMock

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
    TransferJob,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    OffloadingConnectorScheduler,
    TransferJobStatus,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
    OffloadingConnectorWorker,
)
from vllm.v1.kv_offload.base import OffloadingSpec
from vllm.v1.kv_offload.worker.worker import TransferResult
from vllm.v1.outputs import KVConnectorOutput

pytestmark = pytest.mark.cpu_test


def _make_connector_worker() -> OffloadingConnectorWorker:
    spec = MagicMock(spec=OffloadingSpec)
    spec.kv_cache_config = MagicMock()
    spec.vllm_config = MagicMock()
    spec.get_handlers.return_value = iter([])
    worker = OffloadingConnectorWorker(spec=spec)
    worker.worker = MagicMock()
    return worker


def test_get_finished_fail_open_on_transfer_failure():
    connector = _make_connector_worker()
    connector._load_jobs[9] = "req-a"
    connector.worker.get_finished.return_value = [
        TransferResult(job_id=9, success=False),
    ]

    finished_sending, finished_recving = connector.get_finished(set())

    assert finished_sending == set()
    assert finished_recving == {"req-a"}
    assert 9 not in connector._load_jobs
    meta = connector.build_connector_worker_meta()
    assert meta is not None
    assert meta.completed_jobs == {9: 1}
    assert meta.failed_jobs == {9: 1}


def test_start_kv_transfers_fail_open_on_submit_failure():
    connector = _make_connector_worker()
    connector.worker.transfer_async.return_value = False
    metadata = OffloadingConnectorMetadata(
        load_jobs={
            3: TransferJob(req_id="req-b", transfer_spec=MagicMock()),
        },
        store_jobs={},
    )

    connector.start_kv_transfers(metadata)

    meta = connector.build_connector_worker_meta()
    assert meta is not None
    assert meta.completed_jobs == {3: 1}
    assert meta.failed_jobs == {3: 1}


def test_remove_pending_job_fail_open_on_stale_block():
    scheduler = object.__new__(OffloadingConnectorScheduler)
    scheduler._block_id_to_pending_jobs = {10: {1}}

    # Missing block_id and missing job_id must not raise.
    scheduler._remove_pending_job(1, [10, 99])
    scheduler._remove_pending_job(7, [10])
    assert scheduler._block_id_to_pending_jobs == {}


def test_update_connector_output_fail_open_failed_store():
    scheduler = object.__new__(OffloadingConnectorScheduler)
    scheduler._stale_job_threshold = 0
    scheduler._jobs = {
        5: TransferJobStatus(
            req_id="r1",
            pending_count=1,
            keys=set(),
            is_store=True,
        )
    }
    req_status = MagicMock()
    req_status.req_context = MagicMock()
    req_status.req.is_finished.return_value = False
    req_status.transfer_jobs = {5}
    scheduler._req_status = {"r1": req_status}
    scheduler._block_id_to_pending_jobs = {}
    scheduler._blocks_being_loaded = set()
    scheduler._connector_stats = None
    scheduler.manager = MagicMock()

    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
        OffloadingWorkerMetadata,
    )

    meta = OffloadingWorkerMetadata(completed_jobs={5: 1}, failed_jobs={5: 1})
    scheduler.update_connector_output(
        KVConnectorOutput(kv_connector_worker_meta=meta)
    )

    scheduler.manager.complete_store.assert_called_once_with(
        set(), req_status.req_context, success=False
    )
    assert 5 not in scheduler._jobs
