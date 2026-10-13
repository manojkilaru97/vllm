# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model Runner V2 state for the opt-in "no end-of-turn while reasoning" mask.

See `vllm.v1.sample.no_eos_in_reasoning` for the request contract. The mask is
applied to the target logits in `Sampler.apply_sampling_params`, which feeds
both normal sampling and the spec-decode rejection sampler, so a drafted stop
token gets zero target probability and is rejected.
"""

import numpy as np
import torch

from vllm.sampling_params import SamplingParams
from vllm.triton_utils import tl, triton
from vllm.v1.sample.no_eos_in_reasoning import (
    MAX_BLOCKED_TOKEN_IDS,
    MAX_REASONING_END_TOKEN_IDS,
    get_normalized_token_ids,
)
from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor, UvaBackedTensor
from vllm.v1.worker.gpu.states import RequestState

_SCAN_BLOCK = 1024
_VOCAB_BLOCK = 1024


class NoEosInReasoningState:
    def __init__(self, req_states: RequestState):
        self.req_states = req_states
        self.vocab_size = req_states.vocab_size
        max_num_reqs = req_states.max_num_reqs
        device = req_states.device

        self.active = np.zeros(max_num_reqs, dtype=bool)
        # num_end_ids == 0 disables the request on the GPU side.
        self.num_end_ids = UvaBackedTensor(max_num_reqs, dtype=torch.int32)
        self.end_ids = StagedWriteTensor(
            (max_num_reqs, MAX_REASONING_END_TOKEN_IDS),
            dtype=torch.int32,
            device=device,
        )
        self.num_blocked_ids = UvaBackedTensor(max_num_reqs, dtype=torch.int32)
        self.blocked_ids = StagedWriteTensor(
            (max_num_reqs, MAX_BLOCKED_TOKEN_IDS),
            dtype=torch.int32,
            device=device,
        )
        # Committed-token cache: whether a reasoning-end token has been
        # generated, and the next committed position to scan.
        self.closed = StagedWriteTensor(max_num_reqs, dtype=torch.int32, device=device)
        self.scan_pos = StagedWriteTensor(
            max_num_reqs, dtype=torch.int32, device=device
        )
        self._dirty = False

    def add_request(
        self, req_idx: int, prompt_len: int, sampling_params: SamplingParams
    ) -> None:
        token_ids = get_normalized_token_ids(
            sampling_params.extra_args, self.vocab_size
        )
        if token_ids is None and not self.active[req_idx]:
            return
        self._dirty = True
        self.active[req_idx] = token_ids is not None
        if token_ids is None:
            self.num_end_ids.np[req_idx] = 0
            self.num_blocked_ids.np[req_idx] = 0
            return
        end_ids, blocked_ids = token_ids
        self.num_end_ids.np[req_idx] = len(end_ids)
        self.end_ids.stage_write(req_idx, 0, end_ids)
        self.num_blocked_ids.np[req_idx] = len(blocked_ids)
        self.blocked_ids.stage_write(req_idx, 0, blocked_ids)
        self.closed.stage_write_elem(req_idx, 0)
        self.scan_pos.stage_write_elem(req_idx, prompt_len)

    def apply_staged_writes(self) -> None:
        if not self._dirty:
            return
        self.num_end_ids.copy_to_uva()
        self.num_blocked_ids.copy_to_uva()
        self.end_ids.apply_write()
        self.blocked_ids.apply_write()
        self.closed.apply_write()
        self.scan_pos.apply_write()
        self._dirty = False

    def any_active(self, idx_mapping_np: np.ndarray) -> bool:
        return bool(np.any(self.active[idx_mapping_np]))

    def apply(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        input_ids: torch.Tensor,
        expanded_local_pos: torch.Tensor,
    ) -> None:
        if not self.any_active(idx_mapping_np):
            return
        apply_no_eos_in_reasoning(
            logits,
            idx_mapping,
            expanded_idx_mapping,
            expanded_local_pos,
            input_ids,
            self.req_states.all_token_ids.gpu,
            self.req_states.prompt_len.gpu,
            self.req_states.total_len.gpu,
            self.num_end_ids.gpu,
            self.end_ids.gpu,
            self.num_blocked_ids.gpu,
            self.blocked_ids.gpu,
            self.closed.gpu,
            self.scan_pos.gpu,
        )


@triton.jit
def _update_closed_kernel(
    idx_mapping_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    prompt_len_ptr,
    total_len_ptr,
    num_end_ids_ptr,
    end_ids_ptr,
    end_ids_stride,
    closed_ptr,
    scan_pos_ptr,
    MAX_END: tl.constexpr,
    BLOCK: tl.constexpr,
):
    req_state_idx = tl.load(idx_mapping_ptr + tl.program_id(0))
    if req_state_idx < 0:
        return
    num_end = tl.load(num_end_ids_ptr + req_state_idx)
    if num_end == 0:
        return
    if tl.load(closed_ptr + req_state_idx) != 0:
        return

    total_len = tl.load(total_len_ptr + req_state_idx)
    scan_pos = tl.load(scan_pos_ptr + req_state_idx)
    prompt_len = tl.load(prompt_len_ptr + req_state_idx)
    if scan_pos < prompt_len or scan_pos > total_len:
        scan_pos = prompt_len

    end_offs = tl.arange(0, MAX_END)
    end_ids = tl.load(
        end_ids_ptr + req_state_idx * end_ids_stride + end_offs,
        mask=end_offs < num_end,
        other=-1,
    )
    row_ptr = all_token_ids_ptr + req_state_idx.to(tl.int64) * all_token_ids_stride
    found = 0
    start = scan_pos
    while (found == 0) & (start < total_len):
        offs = start + tl.arange(0, BLOCK)
        tokens = tl.load(row_ptr + offs, mask=offs < total_len, other=-2)
        hit = (tokens[:, None] == end_ids[None, :]).to(tl.int32)
        found = tl.max(tl.max(hit, axis=1), axis=0)
        start += BLOCK
    tl.store(closed_ptr + req_state_idx, found)
    tl.store(scan_pos_ptr + req_state_idx, total_len)


@triton.jit
def _mask_stop_tokens_kernel(
    logits_ptr,
    logits_stride,
    vocab_size,
    expanded_idx_mapping_ptr,
    expanded_local_pos_ptr,
    input_ids_ptr,
    num_end_ids_ptr,
    end_ids_ptr,
    end_ids_stride,
    num_blocked_ids_ptr,
    blocked_ids_ptr,
    blocked_ids_stride,
    closed_ptr,
    MAX_END: tl.constexpr,
    MAX_BLOCKED: tl.constexpr,
    VOCAB_BLOCK: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)
    if req_state_idx < 0:
        return
    num_end = tl.load(num_end_ids_ptr + req_state_idx)
    if num_end == 0:
        return
    if tl.load(closed_ptr + req_state_idx) != 0:
        return

    end_offs = tl.arange(0, MAX_END)
    end_ids = tl.load(
        end_ids_ptr + req_state_idx * end_ids_stride + end_offs,
        mask=end_offs < num_end,
        other=-1,
    )
    # Spec decode: row `local_pos` sees the draft tokens at local positions
    # 1..local_pos (position 0 is the last committed token, already scanned).
    local_pos = tl.load(expanded_local_pos_ptr + token_idx)
    first_idx = token_idx - local_pos
    seen = 0
    for k in tl.range(1, local_pos + 1):
        token = tl.load(input_ids_ptr + first_idx + k).to(tl.int32)
        seen = tl.maximum(seen, tl.max((end_ids == token).to(tl.int32), axis=0))

    if seen == 0:
        row_ptr = logits_ptr + token_idx * logits_stride
        blocked_offs = tl.arange(0, MAX_BLOCKED)
        num_blocked = tl.load(num_blocked_ids_ptr + req_state_idx)
        blocked_ids = tl.load(
            blocked_ids_ptr + req_state_idx * blocked_ids_stride + blocked_offs,
            mask=blocked_offs < num_blocked,
            other=-1,
        )
        blocked_valid = (blocked_ids >= 0) & (blocked_ids < vocab_size)
        blocked_logits = tl.load(
            row_ptr + blocked_ids, mask=blocked_valid, other=float("-inf")
        )
        num_finite_blocked = tl.sum((blocked_logits > float("-inf")).to(tl.int32))
        # Nothing to mask if every stop logit is already -inf. Otherwise a
        # structured-output grammar that allows only a stop token wins: never
        # remove the last finite logits. (Static conflicts such as
        # allowed_token_ids of stop tokens only are rejected at admission.)
        # Blocked ids are unique, so a row has a finite non-stop token iff it
        # has more finite logits than finite stop-token logits; usually decided
        # by the first block.
        num_finite = 0
        start = 0
        if num_finite_blocked == 0:
            start = vocab_size
        while (num_finite <= num_finite_blocked) & (start < vocab_size):
            offs = start + tl.arange(0, VOCAB_BLOCK)
            x = tl.load(row_ptr + offs, mask=offs < vocab_size, other=float("-inf"))
            num_finite += tl.sum((x > float("-inf")).to(tl.int32))
            start += VOCAB_BLOCK

        if (num_finite_blocked > 0) & (num_finite > num_finite_blocked):
            tl.store(row_ptr + blocked_ids, float("-inf"), mask=blocked_valid)


def apply_no_eos_in_reasoning(
    logits: torch.Tensor,
    idx_mapping: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    input_ids: torch.Tensor,
    all_token_ids: torch.Tensor,
    prompt_len: torch.Tensor,
    total_len: torch.Tensor,
    num_end_ids: torch.Tensor,
    end_ids: torch.Tensor,
    num_blocked_ids: torch.Tensor,
    blocked_ids: torch.Tensor,
    closed: torch.Tensor,
    scan_pos: torch.Tensor,
) -> None:
    _update_closed_kernel[(idx_mapping.shape[0],)](
        idx_mapping,
        all_token_ids,
        all_token_ids.stride(0),
        prompt_len,
        total_len,
        num_end_ids,
        end_ids,
        end_ids.stride(0),
        closed,
        scan_pos,
        MAX_END=MAX_REASONING_END_TOKEN_IDS,
        BLOCK=_SCAN_BLOCK,
    )
    _mask_stop_tokens_kernel[(logits.shape[0],)](
        logits,
        logits.stride(0),
        logits.shape[1],
        expanded_idx_mapping,
        expanded_local_pos,
        input_ids,
        num_end_ids,
        end_ids,
        end_ids.stride(0),
        num_blocked_ids,
        blocked_ids,
        blocked_ids.stride(0),
        closed,
        MAX_END=MAX_REASONING_END_TOKEN_IDS,
        MAX_BLOCKED=MAX_BLOCKED_TOKEN_IDS,
        VOCAB_BLOCK=_VOCAB_BLOCK,
    )
