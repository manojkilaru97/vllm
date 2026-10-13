# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model Runner V2: opt-in stop-token mask while reasoning is open."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.platforms import current_platform
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.no_eos_in_reasoning import (
    BLOCKED_IDS,
    END_IDS,
    NO_EOS_IN_REASONING_KEY,
)
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler
from vllm.v1.worker.gpu.states import RequestState

pytestmark = pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA")

VOCAB = 20_000
EOS, IM_END, THINK, THINK_END, TOOL_CALL = 2, 11, 12, 13, 14
STOP_IDS = [EOS, IM_END]
END_IDS_ = [THINK_END, TOOL_CALL]
PROMPT = [100, 101, THINK_END, 102, THINK]  # history `</think>` is ignored
NUM_SPEC = 3


def _params(flag: bool, blocked=STOP_IDS, **kwargs) -> SamplingParams:
    # The worker reads the input processor's normalized form.
    extra_args = (
        {NO_EOS_IN_REASONING_KEY: {END_IDS: END_IDS_, BLOCKED_IDS: blocked}}
        if flag
        else None
    )
    return SamplingParams(stop_token_ids=STOP_IDS, extra_args=extra_args, **kwargs)


class _Harness:
    def __init__(self, max_num_reqs: int = 8):
        self.device = torch.device("cuda")
        self.req_states = RequestState(
            max_num_reqs=max_num_reqs,
            max_model_len=16384,
            max_num_batched_tokens=16384,
            num_speculative_steps=NUM_SPEC,
            vocab_size=VOCAB,
            device=self.device,
        )
        self.sampler = Sampler(
            max_num_reqs, VOCAB, self.device, self.req_states, num_speculative_tokens=3
        )
        self.rejection_sampler = RejectionSampler(
            self.sampler,
            SimpleNamespace(
                num_speculative_tokens=NUM_SPEC, rejection_sample_method=""
            ),
            self.device,
        )

    def add(self, req_id: str, outputs: list[int], params: SamplingParams) -> int:
        self.req_states.add_request(
            req_id, len(PROMPT), PROMPT + outputs, len(PROMPT) + len(outputs), 64
        )
        idx = self.req_states.req_id_to_index[req_id]
        self.sampler.add_request(idx, len(PROMPT), params)
        self.req_states.apply_staged_writes()
        self.sampler.apply_staged_writes()
        # The model runner applies staged writes once per step; its rotating
        # UVA staging buffers assume that cadence, so do not queue several.
        torch.cuda.synchronize()
        return idx

    def remove(self, req_id: str) -> None:
        self.req_states.remove_request(req_id)

    def append(self, idx: int, tokens: list[int]) -> None:
        # Mirrors post_update: commit sampled tokens on the GPU.
        total = int(self.req_states.total_len.gpu[idx])
        row = torch.tensor(tokens, dtype=torch.int32, device=self.device)
        self.req_states.all_token_ids.gpu[idx, total : total + len(tokens)] = row
        self.req_states.total_len.gpu[idx] = total + len(tokens)

    def batch(self, idxs: list[int], drafts: list[list[int]] | None = None):
        rows = [1 + len(d) for d in drafts] if drafts else [1] * len(idxs)
        idx_np = np.array(idxs, dtype=np.int32)
        input_ids = []
        for i, idx in enumerate(idxs):
            total = int(self.req_states.total_len.gpu[idx])
            last = int(self.req_states.all_token_ids.gpu[idx, total - 1])
            input_ids += [last] + (drafts[i] if drafts else [])
        dev = self.device
        return dict(
            idx_mapping=torch.from_numpy(idx_np).to(dev),
            idx_mapping_np=idx_np,
            expanded_idx_mapping=torch.from_numpy(np.repeat(idx_np, rows)).to(dev),
            expanded_local_pos=torch.from_numpy(
                np.concatenate([np.arange(n, dtype=np.int32) for n in rows])
            ).to(dev),
            input_ids=torch.tensor(input_ids, dtype=torch.int64, device=dev),
            pos=torch.arange(len(input_ids), dtype=torch.int64, device=dev),
            cu_num_logits=torch.tensor(
                np.concatenate([[0], np.cumsum(rows)]), dtype=torch.int32, device=dev
            ),
            num_rows=sum(rows),
        )

    def processed(self, logits: torch.Tensor, b) -> torch.Tensor:
        return self.sampler.apply_sampling_params(
            logits,
            b["expanded_idx_mapping"],
            b["idx_mapping"],
            b["idx_mapping_np"],
            b["pos"],
            b["input_ids"],
            b["expanded_local_pos"],
        )


def _logits(num_rows: int, hot: dict[int, float] | None = None) -> torch.Tensor:
    logits = torch.randn(num_rows, VOCAB, device="cuda", dtype=torch.bfloat16)
    for token, value in (hot or {EOS: 30.0, IM_END: 30.0}).items():
        logits[:, token] = value
    return logits


def _masked(row: torch.Tensor) -> bool:
    return bool(torch.isneginf(row[STOP_IDS]).all())


def test_masks_stop_tokens_only_while_reasoning_is_open():
    h = _Harness()
    open_req = h.add("open", [200, 201], _params(True))
    closed_think = h.add("think", [200, THINK_END, 201], _params(True))
    closed_tool = h.add("tool", [200, TOOL_CALL, 201], _params(True))
    unflagged = h.add("plain", [200], _params(False))
    b = h.batch([open_req, closed_think, closed_tool, unflagged])
    out = h.processed(_logits(b["num_rows"]), b)

    assert _masked(out[0])
    assert not torch.isneginf(out[1:, STOP_IDS]).any()
    # Non-stop logits of the masked row are untouched.
    assert torch.isfinite(out[0, 100:]).all()


def test_mask_lifts_after_committed_end_token_and_stays_lifted():
    h = _Harness()
    idx = h.add("r", [], _params(True))
    assert _masked(h.processed(_logits(1), h.batch([idx]))[0])
    h.append(idx, [300, 301])
    assert _masked(h.processed(_logits(1), h.batch([idx]))[0])
    h.append(idx, [TOOL_CALL])
    assert not _masked(h.processed(_logits(1), h.batch([idx]))[0])
    h.append(idx, [400, 401, 402])
    assert not _masked(h.processed(_logits(1), h.batch([idx]))[0])


def test_greedy_sampling_never_emits_stop_while_open():
    h = _Harness()
    idx = h.add("r", [], _params(True, temperature=0.0))
    b = h.batch([idx])
    for _ in range(16):
        logits = _logits(1)
        sampled, _ = h.sampler.sample(
            logits,
            b["expanded_idx_mapping"],
            b["idx_mapping"],
            b["idx_mapping_np"],
            b["pos"],
            b["input_ids"],
            b["expanded_local_pos"],
        )
        assert int(sampled[0]) not in STOP_IDS


def test_no_flag_leaves_logits_untouched():
    h = _Harness()
    idx = h.add("r", [], _params(False))
    logits = _logits(1)
    out = h.processed(logits, h.batch([idx]))
    assert out is logits


def test_unflagged_requests_ignore_unnormalized_values():
    h = _Harness()
    params = SamplingParams(
        stop_token_ids=STOP_IDS, extra_args={NO_EOS_IN_REASONING_KEY: True}
    )
    idx = h.add("r", [], params)
    assert not _masked(h.processed(_logits(1), h.batch([idx]))[0])


def test_blocks_every_listed_stop_token():
    h = _Harness()
    idx = h.add("r", [], _params(True, blocked=[EOS, 7, IM_END]))
    out = h.processed(_logits(1, {EOS: 30.0, 7: 30.0, IM_END: 30.0}), h.batch([idx]))
    assert torch.isneginf(out[0, [EOS, 7, IM_END]]).all()


def test_grammar_row_with_only_a_stop_token_is_kept():
    # A structured-output grammar that allows only a stop token wins.
    h = _Harness()
    idx = h.add("r", [], _params(True))
    logits = torch.full((1, VOCAB), float("-inf"), device="cuda")
    logits[0, IM_END] = 0.0
    out = h.processed(logits, h.batch([idx]))
    assert out[0, IM_END] == 0.0


def test_rows_without_finite_stop_logits_are_untouched():
    h = _Harness()
    idx = h.add("r", [], _params(True))
    logits = torch.full((1, VOCAB), float("-inf"), device="cuda")
    logits[0, VOCAB - 1] = 0.0
    out = h.processed(logits, h.batch([idx]))
    assert torch.equal(out, logits)


def test_readd_after_preemption_rescans_generated_tokens():
    h = _Harness()
    outputs = [300] * 3000 + [THINK_END] + [301] * 5000
    idx = h.add("r", outputs, _params(True))
    assert not _masked(h.processed(_logits(1), h.batch([idx]))[0])
    h.remove("r")
    idx = h.add("r", [300] * 5000, _params(True))
    assert _masked(h.processed(_logits(1), h.batch([idx]))[0])


def test_slot_reuse_resets_state():
    h = _Harness(max_num_reqs=1)
    idx = h.add("a", [TOOL_CALL], _params(True))
    assert not _masked(h.processed(_logits(1), h.batch([idx]))[0])
    h.remove("a")
    idx = h.add("b", [], _params(True))
    assert _masked(h.processed(_logits(1), h.batch([idx]))[0])
    h.remove("b")
    idx = h.add("c", [], _params(False, temperature=0.5))
    assert not _masked(h.processed(_logits(1), h.batch([idx]))[0])


def test_spec_decode_rows_see_draft_end_tokens():
    h = _Harness()
    idx = h.add("r", [200], _params(True))
    b = h.batch([idx], drafts=[[300, THINK_END, 301]])
    out = h.processed(_logits(b["num_rows"]), b)
    # Rows 0 and 1 condition on no end token; rows 2 and 3 follow `</think>`.
    assert [_masked(out[i]) for i in range(4)] == [True, True, False, False]


@pytest.mark.parametrize("temperature", [0.0, 1.0])
def test_rejection_sampler_rejects_drafted_stop_token(temperature: float):
    h = _Harness()
    idx = h.add("r", [200], _params(True, temperature=temperature))
    b = h.batch([idx], drafts=[[IM_END, 301, 302]])
    # Target puts nearly all mass on EOS at every row.
    logits = _logits(b["num_rows"], {IM_END: 50.0, EOS: 49.0, 500: 30.0})
    _, sampled, num_sampled = h.rejection_sampler._verify(
        logits,
        None,
        b["input_ids"],
        b["pos"],
        b["cu_num_logits"],
        b["idx_mapping"],
        b["idx_mapping_np"],
        b["expanded_idx_mapping"],
        b["expanded_local_pos"],
    )
    assert int(num_sampled[0]) == 1
    assert int(sampled[0, 0]) == 500


def test_rejection_sampler_accepts_stop_after_drafted_end_token():
    h = _Harness()
    idx = h.add("r", [200], _params(True, temperature=0.0))
    b = h.batch([idx], drafts=[[THINK_END, IM_END, 302]])
    logits = _logits(b["num_rows"], {500: 5.0})
    logits[0, THINK_END] = 50.0
    logits[1:, IM_END] = 50.0
    _, sampled, num_sampled = h.rejection_sampler._verify(
        logits,
        None,
        b["input_ids"],
        b["pos"],
        b["cu_num_logits"],
        b["idx_mapping"],
        b["idx_mapping_np"],
        b["expanded_idx_mapping"],
        b["expanded_local_pos"],
    )
    # `</think>` and EOS are accepted, then the target argmax (EOS) at row 2.
    assert int(num_sampled[0]) == 3
    assert sampled[0, :3].tolist() == [THINK_END, IM_END, IM_END]


def test_mask_leaves_rejection_stats_finite_with_sparse_rows():
    # A grammar-like row whose first vocab block allows only stop tokens.
    h = _Harness()
    idx = h.add("r", [200], _params(True))
    b = h.batch([idx], drafts=[[301, 302, 303]])
    logits = torch.full((b["num_rows"], VOCAB), float("-inf"), device="cuda")
    logits[:, IM_END] = 5.0
    logits[:, VOCAB - 1] = 0.0
    _, sampled, num_sampled = h.rejection_sampler._verify(
        logits,
        None,
        b["input_ids"],
        b["pos"],
        b["cu_num_logits"],
        b["idx_mapping"],
        b["idx_mapping_np"],
        b["expanded_idx_mapping"],
        b["expanded_local_pos"],
    )
    assert int(num_sampled[0]) == 1
    assert int(sampled[0, 0]) == VOCAB - 1
