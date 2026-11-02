# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest

from tests.v1.core.test_scheduler import (
    create_requests_with_priority,
    create_scheduler_with_priority,
)
from vllm.v1.core.sched import priority_yield


def _req(priority, num_tokens, computed=0):
    return SimpleNamespace(
        priority=priority, num_tokens=num_tokens, num_computed_tokens=computed
    )


def test_reserve_frac_env(monkeypatch):
    monkeypatch.delenv(priority_yield.ENV_RESERVE_FRAC, raising=False)
    assert priority_yield.reserve_frac_from_env() == 0.0
    monkeypatch.setenv(priority_yield.ENV_RESERVE_FRAC, "2")
    assert priority_yield.reserve_frac_from_env() == 1.0
    monkeypatch.setenv(priority_yield.ENV_RESERVE_FRAC, "bogus")
    assert priority_yield.reserve_frac_from_env() == 0.0


def test_no_reserve_without_higher_priority_waiting():
    running = [_req(100, 4000, 512)]
    assert (
        priority_yield.build_reserve(running, [_req(100, 900)], 4, 512, 1024, 1.0)
        is None
    )
    assert priority_yield.build_reserve(running, [], 4, 512, 1024, 1.0) is None
    assert (
        priority_yield.build_reserve(running, [_req(0, 900)], 0, 512, 1024, 1.0) is None
    )
    # Equal priorities everywhere: never reserves.
    assert (
        priority_yield.build_reserve([_req(0, 4000)], [_req(0, 9)], 4, 0, 1024, 1.0)
        is None
    )


def test_reserve_counts_only_strictly_higher_priority_and_caps():
    running = [_req(0, 50), _req(100, 4000, 512)]
    waiting = [_req(0, 900), _req(0, 300), _req(100, 4000)]
    reserve = priority_yield.build_reserve(running, waiting, 8, 512, 1024, 1.0)
    assert reserve is not None
    assert reserve.for_priority(100) == 512 + 300
    assert reserve.for_priority(0) == 0
    # Frac caps the reserve; free slots bound the candidates.
    capped = priority_yield.build_reserve(running, waiting, 8, 512, 1024, 0.5)
    assert capped.for_priority(100) == 512
    one_slot = priority_yield.build_reserve(running, waiting, 1, 512, 1024, 1.0)
    assert one_slot.for_priority(100) == 512


def test_stats_window_logs_and_resets():
    stats = priority_yield.YieldStats(now=0.0)
    stats.reserve_steps += 1
    stats.record_cap(512, 0)
    stats.record_cap(512, 100)
    stats.record_admit(0, 0.5)
    stats.record_admit(100, 9.0)
    assert stats.skipped_chunks == 1 and stats.capped_chunks == 1
    assert stats.deferred_tokens == 512 + 412
    assert not stats.maybe_log(now=10.0)
    assert stats.maybe_log(now=61.0)
    assert stats.reserve_steps == 0 and stats.admitted == {"normal": 0, "low": 0}


def _two_low_then_high(monkeypatch, frac):
    if frac is None:
        monkeypatch.delenv(priority_yield.ENV_RESERVE_FRAC, raising=False)
    else:
        monkeypatch.setenv(priority_yield.ENV_RESERVE_FRAC, frac)
    scheduler = create_scheduler_with_priority(
        max_num_batched_tokens=1024,
        long_prefill_token_threshold=512,
        max_model_len=8192,
    )
    low = create_requests_with_priority(
        num_requests=2,
        priorities=[100, 100],
        arrival_times=[1.0, 2.0],
        num_tokens=4000,
        req_ids=["low0", "low1"],
    )
    for request in low:
        scheduler.add_request(request)
    first = scheduler.schedule()
    assert first.num_scheduled_tokens == {"low0": 512, "low1": 512}
    high = create_requests_with_priority(
        num_requests=1,
        priorities=[0],
        arrival_times=[3.0],
        num_tokens=900,
        req_ids=["high"],
        starting_idx=7,
    )[0]
    scheduler.add_request(high)
    return scheduler, scheduler.schedule()


def test_without_yield_low_priority_prefills_starve_new_request(monkeypatch):
    _, output = _two_low_then_high(monkeypatch, None)
    assert "high" not in output.num_scheduled_tokens


def test_low_priority_prefill_yields_to_waiting_high_priority(monkeypatch):
    scheduler, output = _two_low_then_high(monkeypatch, "1.0")
    assert output.num_scheduled_tokens["high"] == 512
    assert output.num_scheduled_tokens.get("low0", 0) == 512
    assert "low1" not in output.num_scheduled_tokens
    # Running list is priority-ordered so the admitted prefill continues first.
    third = scheduler.schedule()
    assert third.num_scheduled_tokens["high"] == 388
    assert sum(third.num_scheduled_tokens.values()) == 1024
    assert scheduler.priority_yield_stats.admitted["normal"] == 1


@pytest.mark.parametrize("frac", [None, "1.0"])
def test_equal_priorities_unchanged(monkeypatch, frac):
    if frac is None:
        monkeypatch.delenv(priority_yield.ENV_RESERVE_FRAC, raising=False)
    else:
        monkeypatch.setenv(priority_yield.ENV_RESERVE_FRAC, frac)
    scheduler = create_scheduler_with_priority(
        max_num_batched_tokens=1024,
        long_prefill_token_threshold=512,
        max_model_len=8192,
    )
    for request in create_requests_with_priority(
        num_requests=3,
        priorities=[0, 0, 0],
        arrival_times=[1.0, 2.0, 3.0],
        num_tokens=4000,
    ):
        scheduler.add_request(request)
    outputs = [scheduler.schedule().num_scheduled_tokens for _ in range(3)]
    assert outputs[0] == {"0": 512, "1": 512}
    assert outputs[1] == {"0": 512, "1": 512}
    assert outputs[2] == {"0": 512, "1": 512}
