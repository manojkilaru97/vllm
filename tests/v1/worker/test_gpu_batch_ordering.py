# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch ordering and classification in Model Runner V2 (vllm.v1.worker.gpu).

split_decodes_and_prefills assumes decode -> short_extend -> prefill request
ordering. With spec decode (decode_query_len > 1), a shorter chunked-prefill
tail sorted in front of the uniform decodes misclassifies every decode as a
prefill.

Conversely, a prompt chunk of exactly decode_query_len tokens has a decode
batch's shape, and must not be classified as a uniform decode batch.
"""

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.mamba2_attn import Mamba2AttentionMetadataBuilder
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.worker.gpu.model_runner import GPUModelRunner, sort_batch_req_ids
from vllm.v1.worker.utils import get_uniform_decode_token_count


def _make_common_attn_metadata(query_lens: list[int]) -> CommonAttentionMetadata:
    num_reqs = len(query_lens)
    num_tokens = sum(query_lens)
    query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32)
    torch.cumsum(
        torch.tensor(query_lens, dtype=torch.int32), 0, out=query_start_loc[1:]
    )
    seq_lens = torch.tensor([1000 + q for q in query_lens], dtype=torch.int32)
    return CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc,
        seq_lens=seq_lens,
        seq_lens_cpu_upper_bound=seq_lens,
        max_seq_len=int(seq_lens.max()),
        num_reqs=num_reqs,
        num_actual_tokens=num_tokens,
        max_query_len=max(query_lens),
        block_table_tensor=torch.zeros(num_reqs, 1, dtype=torch.int32),
        slot_mapping=torch.zeros(num_tokens, dtype=torch.int64),
    )


def _make_runner(
    req_states: dict[str, tuple[int, int]], decode_query_len: int = 8
) -> Any:
    """Build a runner stub from {req_id: (num_computed_tokens, prefill_len)}."""
    prefill_lens = np.array([s[1] for s in req_states.values()], dtype=np.int32)
    num_computed = np.array([s[0] for s in req_states.values()], dtype=np.int32)
    runner: Any = GPUModelRunner.__new__(GPUModelRunner)
    runner.decode_query_len = decode_query_len
    runner.req_states = SimpleNamespace(
        req_id_to_index={req_id: i for i, req_id in enumerate(req_states)},
        # The runner keeps this as min(num_computed_tokens, prefill_len).
        num_computed_prefill_tokens=np.minimum(num_computed, prefill_lens),
        prefill_len=SimpleNamespace(np=prefill_lens),
    )
    return runner


def _uniform_token_count(
    req_states: dict[str, tuple[int, int]],
    query_len: int,
    dummy_run: bool = False,
) -> int | None:
    """Classify via the runner's pre-dispatch gather, as execute_model does."""
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={req_id: query_len for req_id in req_states},
        total_num_scheduled_tokens=query_len * len(req_states),
        scheduled_spec_decode_tokens={},
    )
    runner = _make_runner(req_states, decode_query_len=query_len)
    _, uniform_tok_count = runner.gather_batch_req_state(scheduler_output, dummy_run)
    return uniform_tok_count


def test_spec_decode_batch_is_uniform_decode():
    # decode_query_len == 8 (K=7): every request past its prefill.
    decodes = {f"d{i}": (16, 16) for i in range(6)}
    assert _uniform_token_count(decodes, 8) == 8


def test_prompt_chunk_of_decode_query_len_is_not_uniform_decode():
    # Two prompt chunks whose query length coincides with the K+1 spec-decode
    # query length must reject the batch (issue #49918).
    batch = {f"d{i}": (16, 16) for i in range(6)}
    batch.update({f"p{i}": (8, 40) for i in range(2)})
    assert _uniform_token_count(batch, 8) is None

    # Same collision without spec decoding: 1-token prompt vs 1-token decode.
    assert _uniform_token_count({"p0": (0, 1)}, 1) is None

    # A fresh prompt of exactly decode_query_len tokens.
    assert _uniform_token_count({"p0": (0, 8)}, 8) is None


def test_dummy_batches_stay_uniform_decode():
    # Dummy batches have no request state to consult and must stay classified
    # by shape alone; rejecting them would make capture and replay disagree.
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={f"_dummy_req_{i}": 8 for i in range(4)},
        total_num_scheduled_tokens=32,
    )
    state, uniform_tok_count = _make_runner({}).gather_batch_req_state(
        scheduler_output, True
    )
    assert state is None
    assert uniform_tok_count == 8


def test_sort_batch_req_ids_no_spec():
    # decode_query_len == 1: plain ascending order (decodes first).
    num_tokens_per_req = {"p1": 100, "d1": 1, "p2": 7, "d2": 1}
    assert sort_batch_req_ids(num_tokens_per_req, 1) == ["d1", "d2", "p2", "p1"]


def test_sort_batch_req_ids_spec_decode():
    # decode_query_len == 2 (MTP k=1): uniform decodes lead, then the 1-token
    # chunked-prefill tail, then longer prefills.
    num_tokens_per_req = {"tail": 1, "d1": 2, "p1": 100, "d2": 2}
    assert sort_batch_req_ids(num_tokens_per_req, 2) == ["d1", "d2", "tail", "p1"]


def test_spec_decodes_lead_short_prefill_tail():
    # With the fixed ordering, split_decodes_and_prefills classifies the
    # uniform 2-token decodes as decodes even when a 1-token prefill tail is
    # in the batch (indexer-style: require_uniform, threshold=1+k).
    num_tokens_per_req = {"tail": 1, **{f"d{i}": 2 for i in range(8)}}
    req_ids = sort_batch_req_ids(num_tokens_per_req, 2)
    query_lens = [num_tokens_per_req[r] for r in req_ids]
    assert query_lens == [2] * 8 + [1]

    num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
        split_decodes_and_prefills(
            _make_common_attn_metadata(query_lens),
            decode_threshold=2,
            require_uniform=True,
        )
    )
    assert (num_decodes, num_prefills) == (8, 1)
    assert (num_decode_tokens, num_prefill_tokens) == (16, 1)


@pytest.mark.parametrize(
    "order",
    [
        # Final prompt chunk of a running request, scheduled ahead of decodes.
        ("tail", "d0", "d1"),
        # Short cache-hit prefill between running decodes.
        ("d0", "tail", "d1"),
    ],
)
def test_decode_length_prompt_chunk_trails_spec_decodes(order):
    """A prompt chunk of exactly decode_query_len tokens must not lead decodes.

    E.g. a 4-token prompt tail (MTP k=3) ahead of verify rows. Mamba splits on
    is_prefilling, so every decode behind the chunk would run through the
    prefill kernels and corrupt its recurrent state.
    """
    states = {"tail": (2044, 2048), "d0": (64, 64), "d1": (64, 64)}
    runner = _make_runner({r: states[r] for r in order}, decode_query_len=4)
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={r: 4 for r in order},
        total_num_scheduled_tokens=12,
        scheduled_spec_decode_tokens={"d0": [1] * 3, "d1": [1] * 3},
    )
    state, _ = runner.gather_batch_req_state(scheduler_output, False)
    assert state.req_ids == ["d0", "d1", "tail"]

    metadata = _make_common_attn_metadata([4] * 3)
    metadata.is_prefilling = torch.from_numpy(state.is_prefilling_np)
    num_decodes, num_prefills, _, _ = split_decodes_and_prefills(
        metadata, decode_threshold=4, treat_short_extends_as_decodes=False
    )
    assert (num_decodes, num_prefills) == (2, 1)


def test_padded_prompt_tail_sorts_with_spec_decodes():
    # A one-token prompt tail padded with placeholder drafts is a spec-decode
    # row: it must stay ahead of prompt chunks, not trail them.
    runner = _make_runner(
        {"chunk": (2044, 2048), "padded": (2047, 2048), "d0": (64, 64)},
        decode_query_len=4,
    )
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"chunk": 4, "padded": 4, "d0": 4},
        total_num_scheduled_tokens=12,
        scheduled_spec_decode_tokens={"padded": [-1] * 3, "d0": [1] * 3},
    )
    state, _ = runner.gather_batch_req_state(scheduler_output, False)
    assert state.req_ids == ["padded", "d0", "chunk"]


def test_mamba_runs_padded_prompt_tail_as_spec_decode():
    """A padded one-token prompt tail must take Mamba's spec-decode path.

    Its placeholder drafts can only be rolled back by the decode kernels; the
    prefill kernels would fold them into the running state. A real 4-token
    prompt chunk behind it stays a prefill.
    """
    spec = MambaSpec(
        block_size=16,
        shapes=((1,),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
        num_speculative_blocks=3,
    )
    builder: Any = Mamba2AttentionMetadataBuilder.__new__(
        Mamba2AttentionMetadataBuilder
    )
    builder.kv_cache_spec = spec
    builder.reorder_batch_threshold = 4
    builder.num_spec_tokens = 3
    builder.use_spec_decode = True
    builder.use_replayssm = False
    builder.decode_bc_pre_scratch = None
    builder.chunk_size = 8
    builder.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(mamba_cache_mode="align")
    )
    builder.compilation_config = SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE)

    # Rows: verify decode, padded prompt tail (1 real token), 4-token chunk.
    metadata = _make_common_attn_metadata([4, 4, 4])
    metadata.seq_lens = torch.tensor([68, 36, 36], dtype=torch.int32)
    metadata.seq_lens_cpu_upper_bound = metadata.seq_lens
    metadata.block_table_tensor = torch.arange(24, dtype=torch.int32).view(3, 8)
    metadata.is_prefilling = torch.tensor([False, True, True])
    common = builder.build(
        0,
        metadata,
        num_accepted_tokens=torch.ones(3, dtype=torch.int32),
        num_decode_draft_tokens_cpu=torch.tensor([3, 3, -1], dtype=torch.int32),
    )
    assert (common.num_decodes, common.num_prefills) == (2, 1)


def test_uniform_decode_uses_state_index_not_batch_position():
    """The gather must read each request's own state, not its batch slot.

    Batch position and state index diverge as requests come and go; here the
    only prefilling request sits at state index 0, so a gather keyed on batch
    position would read its row and wrongly reject the two decodes.
    """
    runner: Any = GPUModelRunner.__new__(GPUModelRunner)
    runner.decode_query_len = 8
    # State arrays in state-index order: a prefilling request, then two decodes.
    runner.req_states = SimpleNamespace(
        req_id_to_index={"prefilling": 0, "decode_a": 1, "decode_b": 2},
        num_computed_prefill_tokens=np.array([8, 16, 16], dtype=np.int32),
        prefill_len=SimpleNamespace(np=np.array([40, 16, 16], dtype=np.int32)),
    )
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"decode_a": 8, "decode_b": 8},
        total_num_scheduled_tokens=16,
        scheduled_spec_decode_tokens={},
    )

    state, uniform_tok_count = runner.gather_batch_req_state(scheduler_output, False)
    assert not state.has_prefill
    assert uniform_tok_count == 8

    # Scheduling the prefilling request alongside them must reject the batch.
    scheduler_output.num_scheduled_tokens = {
        "decode_a": 8,
        "prefilling": 8,
        "decode_b": 8,
    }
    scheduler_output.total_num_scheduled_tokens = 24
    state, uniform_tok_count = runner.gather_batch_req_state(scheduler_output, False)
    assert state.has_prefill
    assert uniform_tok_count is None


def test_uniform_decode_predicate():
    # Shape and prefill state must both pass.
    assert get_uniform_decode_token_count(2, 16, 8, False) == 8
    assert get_uniform_decode_token_count(2, 16, 8, True) is None
    # 12 tokens over 2 requests is no shared query length.
    assert get_uniform_decode_token_count(2, 12, 8, False) is None


def test_no_speculator_dispatches_on_query_length_alone():
    """No speculator may pick its cudagraph from a shape-only test.

    Written over the whole package because speculators are added by copying an
    existing one: the shape-only call already reappeared verbatim in a
    speculator added after the original call sites were fixed.
    """
    import vllm.v1.worker.gpu.spec_decode as spec_decode

    shape_only = {"get_uniform_token_count", "is_uniform_query_len"}
    root = Path(spec_decode.__file__).parent
    offenders = []
    for path in sorted(root.rglob("speculator.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
            if name in shape_only:
                offenders.append(f"{path.relative_to(root)}:{node.lineno} {name}")

    assert not offenders, (
        "these speculators classify a decode batch by query length alone; use "
        f"get_uniform_decode_token_count instead: {offenders}"
    )
