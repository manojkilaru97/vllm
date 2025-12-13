# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, TypeVar

import torch

from vllm import SamplingParams
from vllm.v1.sample.logits_processor.interface import (
    BatchUpdate,
    LogitsProcessor,
    MoveDirectionality,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

T = TypeVar("T")
logger = None


#
# NOTE: We intentionally no longer keep a giant static set of newline token
# IDs here. Instead, newline IDs are discovered dynamically per tokenizer in
# `OpenAIServingChat._inject_think_end_token_id` and passed via
# `extra_args["newline_token_ids"]`. If none are provided, the reasoning
# budget processor simply falls back to using the budget + grace criteria.


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


class ReasoningBudgetLogitsProcessor(LogitsProcessor):
    """Enforce a reasoning budget by forcing the model to emit the end-of-think
    token sequence once the budget is reached.

    This reimplements the logic from the vLLM docs `ThinkingBudgetLogitsProcessor`
    example, but it is wired to the existing configuration flow:

    - Users configure reasoning via `chat_template_kwargs`, e.g.:
        `{\"enable_thinking\": True, \"reasoning_budget\": 50}`
    - `ChatCompletionRequest.to_sampling_params` copies these into
      `SamplingParams.extra_args` as:
        * `reasoning_budget` (int, required, != -1)
        * `reasoning_budget_grace_period` (int, optional, default 0)
        * `enable_thinking` (bool, optional)
    - `_inject_think_end_token_id` adds:
        * `think_end_token_id` (int, required)
        * `end_token_ids` (list[int], optional; defaults to `[think_end_token_id]`)

    For requests with a valid budget and end token ids, this processor:
    - Waits until either:
        * `len(output_tok_ids) >= reasoning_budget + reasoning_budget_grace_period`, or
        * `len(output_tok_ids) >= reasoning_budget` and the last token is a newline.
    - Then incrementally forces the `end_token_ids` sequence using a
      suffix/prefix overlap heuristic so that naturally generated prefixes
      of the end sequence are respected.
    """

    def __init__(
        self, vllm_config: "VllmConfig", device: torch.device, is_pin_memory: bool
    ):
        # Per-request state, keyed by batch index.
        #
        # Each value is a dict with keys:
        #   - "output_tok_ids": list[int]  (reference to decoder output tokens)
        #   - "thinking_budget": int
        #   - "thinking_budget_grace_period": int
        #   - "end_token_ids": list[int]
        #   - "is_thinking": bool
        #   - "start_of_end": bool
        #   - "end_of_end": bool
        self.logit_processor_state: dict[int, dict[Any, Any]] = {}

    def is_argmax_invariant(self) -> bool:
        # Forcing a specific token sequence changes argmax.
        return False

    @staticmethod
    def _suffix_prefix_overlap(a: list[int], b: list[int]) -> int:
        """Return length of the longest suffix of `a` that is a prefix of `b`."""
        m = min(len(a), len(b))
        for k in range(m, 0, -1):
            if a[-k:] == b[:k]:
                return k
        return 0

    def _maybe_end_thinking(
        self, idx: int, logits: torch.Tensor, state: dict[Any, Any]
    ) -> torch.Tensor:
        # Once we've fully emitted the end sequence, we no longer intervene.
        if state.get("end_of_end", False):
            return logits

        output_tok_ids: list[int] = state["output_tok_ids"]
        budget: int = int(state["thinking_budget"])
        grace: int = int(state.get("thinking_budget_grace_period", 0))

        # Prefer request-specific newline tokens if provided.
        req_newline_ids = state.get("newline_token_ids")
        if isinstance(req_newline_ids, set):
            newline_ids = req_newline_ids
        elif isinstance(req_newline_ids, (list, tuple)):
            newline_ids = set(int(t) for t in req_newline_ids)
            state["newline_token_ids"] = newline_ids
        else:
            newline_ids = set()

        # Decide when to start forcing the end sequence.
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

        if state.get("start_of_end", False) and not state.get("end_of_end", False):
            end_token_ids: list[int] = state["end_token_ids"]
            if not end_token_ids:
                return logits

            last_n_inputs = list(output_tok_ids[-len(end_token_ids) :])
            overlap = self._suffix_prefix_overlap(last_n_inputs, end_token_ids)

            if overlap < len(end_token_ids):
                # We are still in the middle of emitting the end sequence.
                # Mask out all tokens except the next required token.
                logits[idx, :] = float("-inf")
                insert_id = end_token_ids[overlap]
                logits[idx, insert_id] = 0.0

                # If this is the LAST token of the end sequence, mark as done
                # immediately so we don't intervene on the next call.
                if overlap + 1 == len(end_token_ids):
                    state["end_of_end"] = True
                    state["is_thinking"] = False  # Release control completely
            else:
                # Completed the entire end sequence (detected via overlap).
                state["end_of_end"] = True
                state["is_thinking"] = False  # Release control completely

        return logits

    def update_state(self, batch_update: BatchUpdate | None):
        if not batch_update:
            return

        # Remove finished requests.
        for index in batch_update.removed:
            self.logit_processor_state.pop(index, None)

        # Handle moved requests (unidirectional and swap).
        for a, b, direction in batch_update.moved:
            a_val = self.logit_processor_state.pop(a, None)
            b_val = self.logit_processor_state.pop(b, None)
            if a_val is not None:
                self.logit_processor_state[b] = a_val
            if direction == MoveDirectionality.SWAP and b_val is not None:
                self.logit_processor_state[a] = b_val

        # Add / update requests.
        for index, params, prompt_tok_ids, output_tok_ids in batch_update.added:
            extra = params.extra_args if isinstance(params.extra_args, dict) else {}

            budget = extra.get("reasoning_budget")
            if budget is None or budget == -1:
                # No budget configured -> nothing to enforce.
                self.logit_processor_state.pop(index, None)
                continue

            # Optional grace period, default 0.
            grace = extra.get("reasoning_budget_grace_period", 0) or 0

            # End-of-think token sequence:
            #  - Prefer a full sequence if provided (`end_token_ids`)
            #  - Fallback to a single closing token (`think_end_token_id`)
            end_token_ids = extra.get("end_token_ids")
            think_end_id = extra.get("think_end_token_id")
            if end_token_ids is None and think_end_id is not None:
                end_token_ids = [int(think_end_id)]

            if not end_token_ids:
                # Can't enforce without a closing token id.
                self.logit_processor_state.pop(index, None)
                continue

            # Optional, model-specific newline token IDs.
            newline_token_ids = extra.get("newline_token_ids")
            if newline_token_ids is not None:
                try:
                    newline_token_ids = {int(t) for t in newline_token_ids}
                except Exception:
                    newline_token_ids = None

            # Respect explicit enable_thinking=False from chat_template_kwargs.
            enable_thinking = extra.get("enable_thinking")
            is_thinking = enable_thinking is not False

            state: dict[Any, Any] = {
                "output_tok_ids": output_tok_ids,
                "thinking_budget": int(budget),
                "thinking_budget_grace_period": int(grace),
                "end_token_ids": [int(tid) for tid in end_token_ids],
                "newline_token_ids": newline_token_ids,
                "is_thinking": is_thinking,
                # Internal tracking flags for the end sequence.
                "start_of_end": False,
                "end_of_end": False,
            }

            # If we have prompt tokens and explicit thinking is known to be off,
            # keep the state but mark is_thinking=False so apply() skips it.
            # (This keeps behaviour predictable even if extra_args are reused.)
            self.logit_processor_state[index] = state

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


class LogitBiasLogitsProcessor(LogitsProcessor):
    def __init__(self, _, device: torch.device, is_pin_memory: bool):
        self.device = device
        self.pin_memory = is_pin_memory
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
        return torch.tensor(
            data, device="cpu", dtype=dtype, pin_memory=self.pin_memory
        ).to(device=self.device, non_blocking=True)

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
        self.pin_memory = is_pin_memory
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
        return torch.tensor(
            data, device="cpu", dtype=dtype, pin_memory=self.pin_memory
        ).to(device=self.device, non_blocking=True)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if self.min_toks:
            # Inhibit EOS token for requests which have not reached min length
            logits.index_put_(self.logits_slice, self.neg_inf_tensor)
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
