# SPDX-License-Identifier: Apache-2.0
"""Tests for priority slot-pressure preemption at max_num_seqs and YieldStats tiers."""

import pytest

from vllm.config import CacheConfig, ModelConfig, SchedulerConfig, VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.priority_yield import YieldStats
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager


def _create_scheduler(policy: str = "priority", max_num_seqs: int = 2) -> Scheduler:
    from tests.v1.core.utils import create_scheduler

    return create_scheduler(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=200,
        num_blocks=100,
        scheduling_policy=policy,
    )


def _make_req(req_id: str, priority: int, arrival_time: float, num_tokens: int = 8) -> Request:
    init_none_hash(hash)
    block_hasher = get_request_block_hasher(16, hash)
    return Request(
        request_id=req_id,
        prompt_token_ids=[1] * num_tokens,
        sampling_params=SamplingParams(max_tokens=16),
        pooling_params=None,
        arrival_time=arrival_time,
        priority=priority,
        block_hasher=block_hasher,
    )


def _dummy_output(scheduler_output) -> ModelRunnerOutput:
    req_ids = list(scheduler_output.num_scheduled_tokens.keys())
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
        sampled_token_ids=[[2] for _ in req_ids],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[None for _ in req_ids],
    )


def test_priority_preempts_lower_priority_runner_at_max_num_seqs():
    sched = _create_scheduler(policy="priority", max_num_seqs=2)
    lo0 = _make_req("lo-0", priority=0, arrival_time=1.0)
    lo1 = _make_req("lo-1", priority=0, arrival_time=2.0)
    sched.add_request(lo0)
    sched.add_request(lo1)

    out1 = sched.schedule()
    assert set(out1.num_scheduled_tokens.keys()) == {"lo-0", "lo-1"}
    sched.update_from_output(out1, _dummy_output(out1))

    hi = _make_req("hi-0", priority=-1000, arrival_time=3.0)
    sched.add_request(hi)

    out2 = sched.schedule()
    # Latest-arrived lowest-priority runner (lo-1) is preempted; hi-0 is admitted immediately.
    assert "hi-0" in out2.num_scheduled_tokens
    assert "lo-0" in out2.num_scheduled_tokens
    assert "lo-1" not in out2.num_scheduled_tokens
    assert lo1.status == RequestStatus.PREEMPTED
    assert hi in sched.running
    assert lo1 in sched.waiting


def test_equal_priority_never_preempts_at_max_num_seqs():
    sched = _create_scheduler(policy="priority", max_num_seqs=2)
    r0 = _make_req("r0", priority=0, arrival_time=1.0)
    r1 = _make_req("r1", priority=0, arrival_time=2.0)
    sched.add_request(r0)
    sched.add_request(r1)
    out1 = sched.schedule()
    sched.update_from_output(out1, _dummy_output(out1))

    r2 = _make_req("r2", priority=0, arrival_time=3.0)
    sched.add_request(r2)
    out2 = sched.schedule()
    assert set(out2.num_scheduled_tokens.keys()) == {"r0", "r1"}
    assert r2 in sched.waiting


def test_fcfs_never_preempts_at_max_num_seqs():
    sched = _create_scheduler(policy="fcfs", max_num_seqs=2)
    r0 = _make_req("r0", priority=0, arrival_time=1.0)
    r1 = _make_req("r1", priority=0, arrival_time=2.0)
    sched.add_request(r0)
    sched.add_request(r1)
    out1 = sched.schedule()
    sched.update_from_output(out1, _dummy_output(out1))

    hi = _make_req("hi", priority=-1000, arrival_time=3.0)
    sched.add_request(hi)
    out2 = sched.schedule()
    assert set(out2.num_scheduled_tokens.keys()) == {"r0", "r1"}
    assert hi in sched.waiting


def test_yield_stats_tracks_high_normal_low_tiers():
    stats = YieldStats(now=0.0)
    stats.record_admit(-1000, 0.05)
    stats.record_admit(0, 0.5)
    stats.record_admit(100, 4.0)
    assert stats.admitted == {"high": 1, "normal": 1, "low": 1}
    assert stats.wait_max == {"high": 0.05, "normal": 0.5, "low": 4.0}
    assert stats.maybe_log(now=61.0) is True
    assert stats.admitted == {"high": 0, "normal": 0, "low": 0}
