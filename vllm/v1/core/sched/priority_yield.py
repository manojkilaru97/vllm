# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Priority policy: low-priority prefill chunks yield step budget to pending
higher-priority prefills. Opt-in via VLLM_PRIORITY_PREFILL_RESERVE_FRAC > 0."""

import os
import time
from collections.abc import Iterable

from vllm.logger import init_logger
from vllm.v1.request import Request

logger = init_logger(__name__)

ENV_RESERVE_FRAC = "VLLM_PRIORITY_PREFILL_RESERVE_FRAC"
LOG_INTERVAL_S = 60.0


def reserve_frac_from_env() -> float:
    raw = os.environ.get(ENV_RESERVE_FRAC, "").strip()
    if not raw:
        return 0.0
    try:
        value = float(raw)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r", ENV_RESERVE_FRAC, raw)
        return 0.0
    return min(max(value, 0.0), 1.0)


class PrefillReserve:
    """Tokens held back from a prefill chunk of a given priority this step."""

    def __init__(self, pending: list[tuple[int, int]], cap: int):
        # (priority, prefill tokens) of admissible waiting requests.
        self._pending = pending
        self._cap = cap
        self._cache: dict[int, int] = {}

    def for_priority(self, priority: int) -> int:
        reserve = self._cache.get(priority)
        if reserve is None:
            total = sum(tokens for p, tokens in self._pending if p < priority)
            reserve = self._cache[priority] = min(total, self._cap)
        return reserve


def build_reserve(
    running: list[Request],
    waiting: Iterable[Request],
    free_slots: int,
    chunk_cap: int,
    token_budget: int,
    frac: float,
) -> PrefillReserve | None:
    """None unless a waiting request outranks a running one. `waiting` must be
    priority-ordered and exclude blocked requests."""
    if free_slots <= 0 or not running:
        return None
    lowest_running = max(r.priority for r in running)
    pending: list[tuple[int, int]] = []
    for request in waiting:
        if len(pending) >= free_slots:
            break
        tokens = max(request.num_tokens - request.num_computed_tokens, 0)
        if chunk_cap > 0:
            tokens = min(tokens, chunk_cap)
        pending.append((request.priority, tokens))
    if not pending or min(p for p, _ in pending) >= lowest_running:
        return None
    return PrefillReserve(pending, int(token_budget * frac))


class YieldStats:
    """Windowed counters, logged at most once per LOG_INTERVAL_S."""

    def __init__(self, now: float | None = None):
        self._window_start = time.monotonic() if now is None else now
        self._reset()

    def _reset(self) -> None:
        self.reserve_steps = 0
        self.capped_chunks = 0
        self.skipped_chunks = 0
        self.deferred_tokens = 0
        # "normal" = priority <= 0, "low" = priority > 0.
        self.admitted = {"normal": 0, "low": 0}
        self.wait_sum = {"normal": 0.0, "low": 0.0}
        self.wait_max = {"normal": 0.0, "low": 0.0}

    def record_cap(self, requested: int, allowed: int) -> None:
        self.deferred_tokens += requested - max(allowed, 0)
        if allowed <= 0:
            self.skipped_chunks += 1
        else:
            self.capped_chunks += 1

    def record_admit(self, priority: int, wait_s: float) -> None:
        tier = "low" if priority > 0 else "normal"
        self.admitted[tier] += 1
        self.wait_sum[tier] += wait_s
        self.wait_max[tier] = max(self.wait_max[tier], wait_s)

    def maybe_log(self, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        if now - self._window_start < LOG_INTERVAL_S:
            return False
        if self.reserve_steps or self.admitted["low"]:

            def mean(tier: str) -> float:
                return self.wait_sum[tier] / max(self.admitted[tier], 1)

            logger.info(
                "Priority prefill yield (%.0fs): reserve_steps=%d capped_chunks=%d "
                "skipped_chunks=%d deferred_tokens=%d | admitted normal=%d "
                "wait_mean=%.2fs wait_max=%.2fs | low=%d wait_mean=%.2fs "
                "wait_max=%.2fs",
                now - self._window_start,
                self.reserve_steps,
                self.capped_chunks,
                self.skipped_chunks,
                self.deferred_tokens,
                self.admitted["normal"],
                mean("normal"),
                self.wait_max["normal"],
                self.admitted["low"],
                mean("low"),
                self.wait_max["low"],
            )
        self._window_start = now
        self._reset()
        return True
