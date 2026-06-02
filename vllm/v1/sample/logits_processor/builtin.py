# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
import torch

from vllm import SamplingParams
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.sample.logits_processor.interface import (
    BatchUpdate,
    LogitsProcessor,
    MoveDirectionality,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

T = TypeVar("T")


class MinPLogitsProcessor(LogitsProcessor):
    def __init__(
        self, vllm_config: "VllmConfig", device: torch.device, is_pin_memory: bool
    ):
        max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.min_p_count: int = 0

        self.min_p_cpu_tensor = torch.zeros(
            (max_num_reqs,), dtype=torch.float32, device="cpu", pin_memory=is_pin_memory
        )
        self.min_p_cpu = self.min_p_cpu_tensor.numpy()

        self.use_double_tensor = torch.device(device).type != "cpu"

        if self.use_double_tensor:
            # Pre-allocated device tensor
            self.min_p_device: torch.Tensor = torch.empty(
                (max_num_reqs,), dtype=torch.float32, device=device
            )
        else:
            self.min_p_device = self.min_p_cpu_tensor
        # Current slice of the device tensor
        self.min_p: torch.Tensor = self.min_p_device[:0]

    def is_argmax_invariant(self) -> bool:
        """Min-p never impacts greedy sampling"""
        return True

    def get_min_p_by_index(self, index: int) -> float:
        return float(self.min_p_cpu[index])

    def update_state(self, batch_update: BatchUpdate | None):
        if not batch_update:
            return

        needs_update = False
        # Process added requests.
        for index, params, _, _ in batch_update.added:
            min_p = params.min_p
            min_p_before = self.min_p_cpu[index]
            if min_p_before != min_p:
                needs_update = True
                self.min_p_cpu[index] = min_p
                if min_p and not min_p_before:
                    self.min_p_count += 1
                elif not min_p and min_p_before:
                    self.min_p_count -= 1

        if self.min_p_count:
            # Process removed requests.
            if batch_update.removed:
                needs_update = True
                for index in batch_update.removed:
                    if self.min_p_cpu[index]:
                        self.min_p_cpu[index] = 0
                        self.min_p_count -= 1

            # Process moved requests, unidirectional (a->b) and swap (a<->b).
            for adx, bdx, direct in batch_update.moved:
                min_p_a, min_p_b = self.min_p_cpu[adx], self.min_p_cpu[bdx]
                if min_p_a != min_p_b:
                    needs_update = True
                    self.min_p_cpu[bdx] = min_p_a
                    if direct == MoveDirectionality.SWAP:
                        self.min_p_cpu[adx] = min_p_b
                if direct == MoveDirectionality.UNIDIRECTIONAL:
                    if min_p_a:
                        self.min_p_cpu[adx] = 0
                    if min_p_b:
                        self.min_p_count -= 1

        # Update tensors if needed.
        size = batch_update.batch_size
        if self.min_p_count and (needs_update or self.min_p.shape[0] != size):
            self.min_p = self.min_p_device[:size]
            if self.use_double_tensor:
                self.min_p.copy_(self.min_p_cpu_tensor[:size], non_blocking=True)
            self.min_p.unsqueeze_(1)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.min_p_count:
            return logits

        # Convert logits to probability distribution
        probability_values = torch.nn.functional.softmax(logits, dim=-1)
        # Calculate maximum probabilities per sequence
        max_probabilities = torch.amax(probability_values, dim=-1, keepdim=True)
        # Adjust min_p
        adjusted_min_p = max_probabilities.mul_(self.min_p)
        # Identify valid tokens using threshold comparison
        invalid_token_mask = probability_values < adjusted_min_p
        # Apply mask using boolean indexing
        logits.masked_fill_(invalid_token_mask, -float("inf"))
        return logits


class LogitBiasLogitsProcessor(LogitsProcessor):
    def __init__(self, _, device: torch.device, is_pin_memory: bool):
        self.device = device
        self.biases: dict[int, dict[int, float]] = {}

        self.bias_tensor: torch.Tensor = torch.tensor(())
        self.logits_slice = (
            self._device_tensor([], torch.int32),
            self._device_tensor([], torch.int32),
        )

    def is_argmax_invariant(self) -> bool:
        """Logit bias can rebalance token probabilities and change the
        outcome of argmax in greedy sampling."""
        return False

    def update_state(self, batch_update: BatchUpdate | None):
        needs_update = process_dict_updates(
            self.biases, batch_update, lambda params, _, __: params.logit_bias or None
        )

        # Update tensors if needed.
        if needs_update:
            reqs: list[int] = []
            tok_ids: list[int] = []
            biases: list[float] = []
            for req, lb in self.biases.items():
                reqs.extend([req] * len(lb))
                tok_ids.extend(lb.keys())
                biases.extend(lb.values())

            self.bias_tensor = self._device_tensor(biases, torch.float32)
            self.logits_slice = (
                self._device_tensor(reqs, torch.int32),
                self._device_tensor(tok_ids, torch.int32),
            )

    def _device_tensor(self, data: list, dtype: torch.dtype) -> torch.Tensor:
        return async_tensor_h2d(data, device=self.device, dtype=dtype)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if self.biases:
            logits[self.logits_slice] += self.bias_tensor
        return logits


class MinTokensLogitsProcessor(LogitsProcessor):
    def __init__(
        self, vllm_config: "VllmConfig", device: torch.device, is_pin_memory: bool
    ):
        # index -> (min_toks, output_token_ids, stop_token_ids)
        self.device = device
        self.min_toks: dict[int, tuple[int, Sequence[int], set[int]]] = {}

        # (req_idx_tensor,eos_tok_id_tensor)
        self.logits_slice: tuple[torch.Tensor, torch.Tensor] = (
            self._device_tensor([], torch.int32),
            self._device_tensor([], torch.int32),
        )

        self.neg_inf_tensor = torch.tensor(
            -float("inf"), dtype=torch.float32, device=self.device
        )

    def is_argmax_invariant(self) -> bool:
        """By censoring stop tokens, min-tokens can change the outcome
        of the argmax operation in greedy sampling."""
        return False

    @staticmethod
    def add_request(
        params: SamplingParams, _: list[int] | None, output_tok_ids: list[int]
    ) -> tuple[int, Sequence[int], set[int]] | None:
        min_tokens = params.min_tokens
        if not min_tokens or len(output_tok_ids) >= min_tokens:
            return None
        return min_tokens, output_tok_ids, params.all_stop_token_ids

    def update_state(self, batch_update: BatchUpdate | None):
        needs_update = process_dict_updates(
            self.min_toks, batch_update, self.add_request
        )
        if self.min_toks:
            # Check for any requests that have attained their min tokens.
            to_remove = tuple(
                index
                for index, (min_toks, out_tok_ids, _) in self.min_toks.items()
                if len(out_tok_ids) >= min_toks
            )
            if to_remove:
                needs_update = True
                for index in to_remove:
                    del self.min_toks[index]

        # Update tensors if needed.
        if needs_update:
            reqs: list[int] = []
            tok_ids: list[int] = []
            for req, (_, _, stop_tok_ids) in self.min_toks.items():
                reqs.extend([req] * len(stop_tok_ids))
                tok_ids.extend(stop_tok_ids)

            self.logits_slice = (
                self._device_tensor(reqs, torch.int32),
                self._device_tensor(tok_ids, torch.int32),
            )

    def _device_tensor(self, data: list, dtype: torch.dtype) -> torch.Tensor:
        return async_tensor_h2d(data, device=self.device, dtype=dtype)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if self.min_toks:
            # Inhibit EOS token for requests which have not reached min length
            logits.index_put_(self.logits_slice, self.neg_inf_tensor)
        return logits

    def apply_with_spec_decode(
        self,
        logits: torch.Tensor,
        num_draft_tokens: list[int],
    ) -> torch.Tensor:
        """Spec-decode version of apply().
        Priority: ``min_tokens`` > ``stop_token_ids`` / EOS.
        Example: ``num_draft_tokens = [2, 3, 1]``
          → ``logits`` shape ``[6, V]``, ``cumsum = [0, 2, 5, 6]``
          → request 0 owns rows 0‑1, request 1 rows 2‑4, request 2 row 5.
        """
        if not self.min_toks:
            return logits

        num_draft_arr = np.array(num_draft_tokens, dtype=np.int64)
        cumsum = np.concatenate([[0], np.cumsum(num_draft_arr)])

        entries = [
            (req_idx, min_tok, len(out_tok_ids), list(stop_tok_ids))
            for req_idx, (min_tok, out_tok_ids, stop_tok_ids) in self.min_toks.items()
            if stop_tok_ids
        ]

        if not entries:
            return logits

        all_rows: list[np.ndarray] = []  # row indices to mask
        all_toks: list[np.ndarray] = []  # stop-token ids at those rows

        for req_idx, min_tok, current_len, stop_toks in entries:
            remaining = min_tok - current_len
            # How many leading draft positions still need stop-token masking.
            n_mask = int(min(max(remaining, 0), num_draft_arr[req_idx]))

            if n_mask > 0:
                offset = cumsum[req_idx]
                row_indices = np.arange(offset, offset + n_mask, dtype=np.int64)
                n_stop = len(stop_toks)
                all_rows.append(np.repeat(row_indices, n_stop))
                all_toks.append(np.tile(stop_toks, n_mask))

        if all_rows:
            rows_arr = np.concatenate(all_rows)
            toks_arr = np.concatenate(all_toks)
            # (row_indices, token_indices) for index_put_ to set -inf.
            logits_slice = (
                async_tensor_h2d(rows_arr, device=self.device),
                async_tensor_h2d(toks_arr, device=self.device),
            )
            logits.index_put_(logits_slice, self.neg_inf_tensor)

        return logits


class ReasoningBudgetLogitsProcessor(LogitsProcessor):
    """Force end-of-think tokens once the configured reasoning budget is hit."""

    def __init__(
        self, vllm_config: "VllmConfig", device: torch.device, is_pin_memory: bool
    ):
        self.logit_processor_state: dict[int, dict[str, Any]] = {}

    def is_argmax_invariant(self) -> bool:
        return False

    @staticmethod
    def _suffix_prefix_overlap(a: list[int], b: list[int]) -> int:
        m = min(len(a), len(b))
        for k in range(m, 0, -1):
            if a[-k:] == b[:k]:
                return k
        return 0

    @staticmethod
    def _contains_subsequence(
        tokens: list[int], subsequence: list[int], start: int = 0
    ) -> bool:
        if not subsequence or len(tokens) < len(subsequence):
            return False

        start = max(start, 0)
        last_start = len(tokens) - len(subsequence)
        for i in range(start, last_start + 1):
            if tokens[i : i + len(subsequence)] == subsequence:
                return True
        return False

    def _maybe_mark_natural_end(self, state: dict[str, Any]) -> bool:
        output_tok_ids: list[int] = state["output_tok_ids"]
        end_token_ids: list[int] = state["end_token_ids"]
        if not end_token_ids:
            return False

        checked_len = int(state.get("natural_end_checked_len", 0) or 0)
        # Re-scan the small overlap where an end marker could straddle the
        # previously checked suffix and the newly generated tokens.
        scan_start = max(0, checked_len - len(end_token_ids) + 1)
        if self._contains_subsequence(output_tok_ids, end_token_ids, scan_start):
            state["end_of_end"] = True
            state["is_thinking"] = False
            state["natural_end_seen"] = True
            return True

        state["natural_end_checked_len"] = len(output_tok_ids)
        return False

    def _maybe_end_thinking(
        self, idx: int, logits: torch.Tensor, state: dict[str, Any]
    ) -> torch.Tensor:
        if state.get("end_of_end", False):
            return logits

        output_tok_ids: list[int] = state["output_tok_ids"]
        if self._maybe_mark_natural_end(state):
            return logits

        budget: int = state["thinking_budget"]
        grace: int = state["thinking_budget_grace_period"]

        newline_ids: set[int] = state.get("newline_token_ids", set())
        if not isinstance(newline_ids, set):
            newline_ids = set()
            state["newline_token_ids"] = newline_ids

        if (
            len(output_tok_ids) >= budget + grace
            and not state.get("start_of_end", False)
        ):
            state["start_of_end"] = True

        if (
            len(output_tok_ids) >= budget
            and output_tok_ids
            and output_tok_ids[-1] in newline_ids
            and not state.get("start_of_end", False)
        ):
            state["start_of_end"] = True

        if not state.get("start_of_end", False) or state.get("end_of_end", False):
            return logits

        end_token_ids: list[int] = state["end_token_ids"]
        if not end_token_ids:
            return logits

        last_n_inputs = list(output_tok_ids[-len(end_token_ids) :])
        overlap = self._suffix_prefix_overlap(last_n_inputs, end_token_ids)

        if overlap < len(end_token_ids):
            insert_id = end_token_ids[overlap]
            if not 0 <= insert_id < logits.shape[1]:
                state["end_of_end"] = True
                state["is_thinking"] = False
                return logits

            logits[idx, :] = float("-inf")
            logits[idx, insert_id] = 0.0

            if overlap + 1 == len(end_token_ids):
                state["end_of_end"] = True
                state["is_thinking"] = False
        else:
            state["end_of_end"] = True
            state["is_thinking"] = False

        return logits

    def update_state(self, batch_update: BatchUpdate | None):
        if not batch_update:
            return

        for index in batch_update.removed:
            self.logit_processor_state.pop(index, None)

        for a_index, b_index, direct in batch_update.moved:
            a_entry = self.logit_processor_state.pop(a_index, None)
            b_entry = self.logit_processor_state.pop(b_index, None)
            if a_entry is not None:
                self.logit_processor_state[b_index] = a_entry
            if direct == MoveDirectionality.SWAP and b_entry is not None:
                self.logit_processor_state[a_index] = b_entry

        for index, params, _, output_tok_ids in batch_update.added:
            extra = params.extra_args if isinstance(params.extra_args, dict) else {}
            budget = extra.get("reasoning_budget")
            if budget is None:
                self.logit_processor_state.pop(index, None)
                continue

            try:
                budget_int = int(budget)
            except Exception:
                self.logit_processor_state.pop(index, None)
                continue

            if budget_int == -1:
                self.logit_processor_state.pop(index, None)
                continue

            grace = extra.get("reasoning_budget_grace_period", 0) or 0
            try:
                grace_int = int(grace)
            except Exception:
                grace_int = 0

            end_token_ids = extra.get("end_token_ids")
            if (
                isinstance(end_token_ids, Sequence)
                and not isinstance(end_token_ids, (str, bytes))
            ):
                parsed_end_ids = [int(tid) for tid in end_token_ids]
            else:
                parsed_end_ids = []

            if not parsed_end_ids and extra.get("think_end_token_id") is not None:
                try:
                    parsed_end_ids = [int(extra["think_end_token_id"])]
                except Exception:
                    parsed_end_ids = []

            if not parsed_end_ids:
                self.logit_processor_state.pop(index, None)
                continue

            newline_token_ids = extra.get("newline_token_ids")
            newline_ids_set: set[int] = set()
            if (
                isinstance(newline_token_ids, Sequence)
                and not isinstance(newline_token_ids, (str, bytes))
            ):
                try:
                    newline_ids_set = {int(tid) for tid in newline_token_ids}
                except Exception:
                    newline_ids_set = set()

            enable_thinking = extra.get("enable_thinking")
            is_thinking = enable_thinking is not False

            self.logit_processor_state[index] = {
                "output_tok_ids": output_tok_ids,
                "thinking_budget": budget_int,
                "thinking_budget_grace_period": grace_int,
                "end_token_ids": parsed_end_ids,
                "newline_token_ids": newline_ids_set,
                "is_thinking": is_thinking,
                "start_of_end": False,
                "end_of_end": False,
                "natural_end_checked_len": 0,
            }

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.logit_processor_state:
            return logits

        for idx, state in self.logit_processor_state.items():
            if idx >= logits.shape[0]:
                continue
            if not state.get("is_thinking", False):
                continue
            logits = self._maybe_end_thinking(idx, logits, state)
        return logits

    def apply_with_spec_decode(
        self,
        logits: torch.Tensor,
        num_draft_tokens: list[int],
    ) -> torch.Tensor:
        if not self.logit_processor_state:
            return logits

        row_offset = 0
        for req_idx, num_rows in enumerate(num_draft_tokens):
            if num_rows <= 0:
                continue
            state = self.logit_processor_state.get(req_idx)
            if state is None or not state.get("is_thinking", False):
                row_offset += num_rows
                continue
            for row_idx in range(row_offset, row_offset + num_rows):
                if row_idx >= logits.shape[0]:
                    break
                logits = self._maybe_end_thinking(row_idx, logits, state)
            row_offset += num_rows

        return logits


def process_dict_updates(
    req_entries: dict[int, T],
    batch_update: BatchUpdate | None,
    new_state: Callable[[SamplingParams, list[int] | None, list[int]], T | None],
) -> bool:
    """Utility function to update dict state for sparse LogitsProcessors."""

    if not batch_update:
        # Nothing to do.
        return False

    updated = False
    for index, params, prompt_tok_ids, output_tok_ids in batch_update.added:
        if (state := new_state(params, prompt_tok_ids, output_tok_ids)) is not None:
            req_entries[index] = state
            updated = True
        elif req_entries.pop(index, None) is not None:
            updated = True

    if req_entries:
        # Process removed requests.
        for index in batch_update.removed:
            if req_entries.pop(index, None):
                updated = True

        # Process moved requests, unidirectional (a->b) and
        # swapped (a<->b)
        for a_index, b_index, direct in batch_update.moved:
            a_entry = req_entries.pop(a_index, None)
            b_entry = req_entries.pop(b_index, None)
            if a_entry is not None:
                req_entries[b_index] = a_entry
                updated = True
            if b_entry is not None:
                updated = True
                if direct == MoveDirectionality.SWAP:
                    req_entries[a_index] = b_entry

    return updated
