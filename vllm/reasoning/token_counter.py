# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable, Sequence


class ReasoningTokenCounter:
    """Incrementally count generated tokens inside reasoning spans."""

    def __init__(
        self,
        *,
        start_sequences: Iterable[Sequence[int]] = (),
        end_sequences: Iterable[Sequence[int]] = (),
        neutral_sequences: Iterable[Sequence[int]] = (),
        initial_in_reasoning: bool = False,
    ) -> None:
        self._start_sequences = self._normalize_sequences(start_sequences)
        self._end_sequences = self._normalize_sequences(end_sequences)
        self._neutral_sequences = self._normalize_sequences(neutral_sequences)
        self._in_reasoning = initial_in_reasoning
        self._pending: list[int] = []
        self.total = 0

    @staticmethod
    def _normalize_sequences(
        sequences: Iterable[Sequence[int]],
    ) -> tuple[tuple[int, ...], ...]:
        return tuple(
            sorted(
                {tuple(sequence) for sequence in sequences if sequence},
                key=len,
                reverse=True,
            )
        )

    def seed(self, token_ids: Sequence[int]) -> None:
        """Advance delimiter state through prompt tokens without counting them."""
        self._consume(token_ids, finished=True, count_tokens=False)
        self.total = 0

    def update(self, token_ids: Sequence[int], *, finished: bool = False) -> int:
        """Consume one generated-token delta and return the cumulative count."""
        self._consume(token_ids, finished=finished, count_tokens=True)
        return self.total

    def _consume(
        self,
        token_ids: Sequence[int],
        *,
        finished: bool,
        count_tokens: bool,
    ) -> None:
        tokens = self._pending + list(token_ids)
        self._pending = []
        index = 0
        while index < len(tokens):
            start = self._match_at(self._start_sequences, tokens, index)
            if start is not None:
                self._in_reasoning = True
                index += len(start)
                continue
            end = self._match_at(self._end_sequences, tokens, index)
            if end is not None:
                self._in_reasoning = False
                index += len(end)
                continue
            if self._in_reasoning:
                neutral = self._match_at(self._neutral_sequences, tokens, index)
                if neutral is not None:
                    index += len(neutral)
                    continue
            if not finished and self._is_marker_prefix_at(tokens, index):
                # Only retain the small terminal suffix that may become a
                # complete marker when the next delta arrives.
                self._pending = tokens[index:]
                break
            if count_tokens and self._in_reasoning:
                self.total += 1
            index += 1

    @staticmethod
    def _match_at(
        sequences: tuple[tuple[int, ...], ...],
        token_ids: Sequence[int],
        index: int,
    ) -> tuple[int, ...] | None:
        for sequence in sequences:
            if index + len(sequence) <= len(token_ids) and all(
                token_ids[index + offset] == token_id
                for offset, token_id in enumerate(sequence)
            ):
                return sequence
        return None

    def _is_marker_prefix_at(self, token_ids: Sequence[int], index: int) -> bool:
        candidate_length = len(token_ids) - index
        marker_sequences = self._start_sequences + self._end_sequences
        if self._in_reasoning:
            marker_sequences += self._neutral_sequences
        return any(
            candidate_length < len(sequence)
            and all(
                token_ids[index + offset] == sequence[offset]
                for offset in range(candidate_length)
            )
            for sequence in marker_sequences
        )
