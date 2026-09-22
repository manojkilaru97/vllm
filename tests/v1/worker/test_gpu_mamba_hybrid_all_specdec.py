# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V2 runner plumbing of ``prev_last_scheduled_idx`` for Mamba "all" prefix
caching + speculative decoding.

The Mamba2 decode kernel reads the previous step's state snapshot from the
block of that step's last *scheduled* token, so the runner must tell the
metadata builder where that was (or -1 to fall back to the last computed
token's block). This mirrors V1 ``postprocess_mamba_all`` /
``preprocess_mamba_all_specdec`` and guards the CUDA-graph capture assert in
``mamba_mixer2`` (``block_idx_last_scheduled_token_prev_step_d is not None``).
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backends.mamba2_attn import Mamba2AttentionMetadataBuilder
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, MambaSpec
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
from vllm.v1.worker.utils import AttentionGroup

BLOCK_SIZE = 4
NUM_SPEC_TOKENS = 3
MAX_NUM_REQS = 4
DEVICE = torch.device("cpu")


class _RecordingMamba2Builder(Mamba2AttentionMetadataBuilder):
    """Records ``build`` kwargs; keeps the real ``build_for_cudagraph_capture``."""

    def __init__(self, mamba_cache_mode: str, num_spec_tokens: int):
        self.build_kwargs: list[dict] = []
        self.num_spec_tokens = num_spec_tokens
        self.use_spec_decode = num_spec_tokens > 0
        self.decode_cudagraph_max_bs = MAX_NUM_REQS
        self.vllm_config = SimpleNamespace(
            cache_config=SimpleNamespace(mamba_cache_mode=mamba_cache_mode)
        )

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False, **kw):
        self.build_kwargs.append(kw)
        return object()


def _make_model_state(
    monkeypatch: pytest.MonkeyPatch, mamba_cache_mode: str, num_spec_tokens: int
) -> MambaHybridModelState:
    def fake_default_init(self, vllm_config, model, encoder_cache, device):
        self.vllm_config = vllm_config
        self.model_config = None
        self.scheduler_config = None
        self.model = model
        self.device = device
        self.max_model_len = 64
        self.max_num_reqs = MAX_NUM_REQS
        self.max_num_tokens = 64
        self.supports_mm_inputs = False
        self.rope_state = None

    monkeypatch.setattr(DefaultModelState, "__init__", fake_default_init)
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(
            mamba_cache_mode=mamba_cache_mode, block_size=BLOCK_SIZE
        ),
        num_speculative_tokens=num_spec_tokens,
    )
    return MambaHybridModelState(vllm_config, None, None, DEVICE)


def _make_attn(mamba_cache_mode: str, num_spec_tokens: int):
    spec = MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=((1,),),
        dtypes=(torch.float32,),
        mamba_cache_mode=mamba_cache_mode,
        num_speculative_blocks=1,
    )
    builder = _RecordingMamba2Builder(mamba_cache_mode, num_spec_tokens)
    group = AttentionGroup(
        backend=object,
        layer_names=["mamba"],
        kv_cache_spec=spec,
        kv_cache_group_id=0,
        metadata_builders=[builder],
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["mamba"], spec)],
    )
    return builder, [[group]], kv_cache_config


def _make_input_batch(
    idx_mapping: list[int],
    num_scheduled: list[int],
    num_computed: list[int],
    num_reqs_padded: int | None = None,
) -> InputBatch:
    num_reqs = len(idx_mapping)
    num_reqs_padded = num_reqs_padded or num_reqs
    num_scheduled_np = np.array(num_scheduled, dtype=np.int32)
    num_computed_np = np.array(num_computed, dtype=np.int32)
    num_tokens = int(num_scheduled_np.sum())
    query_start_loc_np = np.zeros(num_reqs_padded + 1, dtype=np.int32)
    np.cumsum(num_scheduled_np, out=query_start_loc_np[1 : num_reqs + 1])
    query_start_loc_np[num_reqs + 1 :] = num_tokens
    seq_lens_np = np.zeros(num_reqs_padded, dtype=np.int32)
    seq_lens_np[:num_reqs] = num_computed_np + num_scheduled_np
    idx_mapping_np = np.array(idx_mapping, dtype=np.intp)
    idx_mapping_t = torch.from_numpy(idx_mapping_np)
    return InputBatch(
        req_ids=[f"req{i}" for i in idx_mapping],
        num_reqs=num_reqs,
        num_reqs_after_padding=num_reqs_padded,
        idx_mapping=idx_mapping_t,
        idx_mapping_np=idx_mapping_np,
        expanded_idx_mapping=idx_mapping_t,
        expanded_local_pos=torch.zeros(num_reqs, dtype=torch.int32),
        num_scheduled_tokens=num_scheduled_np,
        num_tokens=num_tokens,
        num_tokens_after_padding=num_tokens,
        num_draft_tokens=0,
        num_draft_tokens_per_req=None,
        query_start_loc=torch.from_numpy(query_start_loc_np),
        query_start_loc_np=query_start_loc_np,
        seq_lens=torch.from_numpy(seq_lens_np),
        seq_lens_cpu_upper_bound=torch.from_numpy(seq_lens_np.copy()),
        dcp_local_seq_lens=None,
        num_computed_tokens_np=num_computed_np,
        prefill_len_np=np.zeros(num_reqs, dtype=np.int32),
        num_computed_prefill_tokens_np=np.zeros(num_reqs, dtype=np.int32),
        is_prefilling_np=np.zeros(num_reqs, dtype=np.bool_),
        has_prefill=False,
        max_seq_len_np=None,
        input_ids=torch.zeros(num_tokens, dtype=torch.int32),
        positions=torch.zeros(num_tokens, dtype=torch.int64),
        is_padding=torch.zeros(num_tokens, dtype=torch.bool),
        logits_indices=torch.zeros(num_reqs, dtype=torch.int64),
        cu_num_logits=torch.arange(num_reqs + 1, dtype=torch.int32),
        cu_num_logits_np=np.arange(num_reqs + 1, dtype=np.int32),
        has_structured_output_reqs=False,
        prompt_lens=None,
    )


def _run_step(
    state: MambaHybridModelState,
    builder: _RecordingMamba2Builder,
    attn_groups,
    kv_cache_config,
    input_batch: InputBatch,
    *,
    real_batch: bool = True,
    cudagraph_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    for_capture: bool = False,
) -> dict:
    """Drive the runner's per-step model_state calls in execute_model order."""
    num_reqs_padded = input_batch.num_reqs_after_padding
    block_tables = (torch.zeros(num_reqs_padded, 8, dtype=torch.int32),)
    slot_mappings = torch.zeros(1, input_batch.num_tokens, dtype=torch.int64)
    if real_batch:
        state.preprocess_state(
            input_batch, block_tables, kv_cache_config, torch.zeros(MAX_NUM_REQS)
        )
    state.prepare_attn(
        input_batch,
        cudagraph_mode,
        block_tables,
        slot_mappings,
        attn_groups,
        kv_cache_config,
        for_capture,
    )
    return builder.build_kwargs[-1]


def _new_req(num_computed_tokens: int = 0):
    return SimpleNamespace(num_computed_tokens=num_computed_tokens)


def test_all_specdec_prev_last_scheduled_idx_tracks_previous_full_decode(
    monkeypatch: pytest.MonkeyPatch,
):
    state = _make_model_state(monkeypatch, "all", NUM_SPEC_TOKENS)
    builder, attn_groups, kv_cache_config = _make_attn("all", NUM_SPEC_TOKENS)
    full_decode = 1 + NUM_SPEC_TOKENS
    slot = 2

    # Step 1: prefill (6 tokens). No previous step -> -1.
    state.add_request(slot, _new_req())
    kw = _run_step(
        state,
        builder,
        attn_groups,
        kv_cache_config,
        _make_input_batch([slot], [6], [0]),
    )
    assert kw["prev_last_scheduled_idx"].tolist() == [-1]

    # Step 2: first full spec decode. The previous step was a prefill, so the
    # runner passes -1 and the builder falls back to (6 - 1) // 4 = 1, the
    # block of step 1's last token.
    kw = _run_step(
        state,
        builder,
        attn_groups,
        kv_cache_config,
        _make_input_batch([slot], [full_decode], [6]),
    )
    assert kw["prev_last_scheduled_idx"].tolist() == [-1]

    # Step 3: only 1 of 3 drafts was accepted (num_computed = 6 + 1 + 1), yet
    # the previous snapshot landed at step 2's last *scheduled* token, position
    # 6 + 4 - 1 = 9 -> block 2 (the fallback would wrongly give (8-1)//4 = 1).
    kw = _run_step(
        state,
        builder,
        attn_groups,
        kv_cache_config,
        _make_input_batch([slot], [full_decode], [8]),
    )
    prev = kw["prev_last_scheduled_idx"]
    assert prev.dtype == torch.int32
    assert prev.tolist() == [2]

    # Step 4: a second request lands in another slot mid-prefill while the
    # first keeps decoding; batch order differs from slot order.
    state.add_request(0, _new_req())
    kw = _run_step(
        state,
        builder,
        attn_groups,
        kv_cache_config,
        _make_input_batch([slot, 0], [full_decode, 5], [12, 0]),
    )
    # (8 + 4 - 1) // 4 = 2 from step 3; the new request has none.
    assert kw["prev_last_scheduled_idx"].tolist() == [2, -1]

    # Slot reuse: the first request finishes and a new one takes its slot.
    state.add_request(slot, _new_req(num_computed_tokens=9))
    kw = _run_step(
        state,
        builder,
        attn_groups,
        kv_cache_config,
        _make_input_batch([0, slot], [full_decode, 3], [5, 9]),
    )
    assert kw["prev_last_scheduled_idx"].tolist() == [-1, -1]


def test_all_specdec_prev_last_scheduled_idx_padding_dummy_and_capture(
    monkeypatch: pytest.MonkeyPatch,
):
    state = _make_model_state(monkeypatch, "all", NUM_SPEC_TOKENS)
    builder, attn_groups, kv_cache_config = _make_attn("all", NUM_SPEC_TOKENS)
    full_decode = 1 + NUM_SPEC_TOKENS
    state.add_request(1, _new_req())
    _run_step(
        state,
        builder,
        attn_groups,
        kv_cache_config,
        _make_input_batch([1], [full_decode], [8]),
    )

    # FULL-CG replay: rows are padded to the graph's batch size with -1.
    kw = _run_step(
        state,
        builder,
        attn_groups,
        kv_cache_config,
        _make_input_batch([1], [full_decode], [12], num_reqs_padded=3),
        cudagraph_mode=CUDAGraphMode.FULL,
    )
    assert kw["prev_last_scheduled_idx"].tolist() == [2, -1, -1]

    # Dummy runs (DP / profiling / PIECEWISE capture) skip preprocess_state and
    # must not consume or corrupt real request state: all rows are -1 ...
    kw = _run_step(
        state,
        builder,
        attn_groups,
        kv_cache_config,
        _make_input_batch([0, 1], [full_decode, full_decode], [0, 0]),
        real_batch=False,
    )
    assert kw["prev_last_scheduled_idx"].tolist() == [-1, -1]
    # ... and the real request still sees its recorded index afterwards.
    kw = _run_step(
        state,
        builder,
        attn_groups,
        kv_cache_config,
        _make_input_batch([1], [full_decode], [16]),
    )
    assert kw["prev_last_scheduled_idx"].tolist() == [(12 + full_decode - 1) // 4]

    # FULL-CG capture goes through the builder's build_for_cudagraph_capture,
    # which supplies a zero dummy so the mixer's spec-decode path is captured.
    kw = _run_step(
        state,
        builder,
        attn_groups,
        kv_cache_config,
        _make_input_batch([0, 1], [full_decode, full_decode], [0, 0]),
        real_batch=False,
        for_capture=True,
    )
    assert kw["prev_last_scheduled_idx"].tolist() == [0, 0]


@pytest.mark.parametrize(
    ("mamba_cache_mode", "num_spec_tokens"),
    [("none", NUM_SPEC_TOKENS), ("align", NUM_SPEC_TOKENS), ("all", 0)],
)
def test_prev_last_scheduled_idx_only_for_all_mode_spec_decode(
    monkeypatch: pytest.MonkeyPatch, mamba_cache_mode: str, num_spec_tokens: int
):
    state = _make_model_state(monkeypatch, mamba_cache_mode, num_spec_tokens)
    if mamba_cache_mode == "align":
        # Align mode runs its own fused GPU pre-copy; keep it out of this test.
        monkeypatch.setattr(state, "_align_mode", False)
    builder, attn_groups, kv_cache_config = _make_attn(
        mamba_cache_mode, num_spec_tokens
    )
    state.add_request(0, _new_req())
    kw = _run_step(
        state,
        builder,
        attn_groups,
        kv_cache_config,
        _make_input_batch([0], [1 + num_spec_tokens], [8]),
    )
    assert "prev_last_scheduled_idx" not in kw
