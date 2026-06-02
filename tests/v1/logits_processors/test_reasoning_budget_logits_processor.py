# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm import SamplingParams
from vllm.v1.sample.logits_processor.builtin import ReasoningBudgetLogitsProcessor
from vllm.v1.sample.logits_processor.interface import BatchUpdate


def _processor_for_output(
    output_tok_ids: list[int],
    *,
    budget: int,
    end_token_ids: list[int],
    newline_token_ids: list[int] | None = None,
) -> ReasoningBudgetLogitsProcessor:
    processor = ReasoningBudgetLogitsProcessor(
        object(), torch.device("cpu"), False
    )
    params = SamplingParams(
        extra_args={
            "reasoning_budget": budget,
            "reasoning_budget_grace_period": 100,
            "end_token_ids": end_token_ids,
            "newline_token_ids": newline_token_ids or [],
            "enable_thinking": True,
        }
    )
    processor.update_state(
        BatchUpdate(
            batch_size=1,
            removed=(),
            added=[(0, params, None, output_tok_ids)],
            moved=(),
        )
    )
    return processor


def test_reasoning_budget_respects_natural_end_before_long_content():
    end_token_ids = [7, 8]
    output_tok_ids = [7, 8, 10, 11]
    processor = _processor_for_output(
        output_tok_ids,
        budget=len(output_tok_ids),
        end_token_ids=end_token_ids,
        newline_token_ids=[11],
    )

    logits = torch.zeros((1, 20))
    processed = processor.apply(logits.clone())

    assert torch.isfinite(processed).all()
    state = processor.logit_processor_state[0]
    assert state["is_thinking"] is False
    assert state["end_of_end"] is True
    assert state["natural_end_seen"] is True


def test_reasoning_budget_still_forces_end_when_no_natural_end():
    end_token_ids = [7, 8]
    output_tok_ids = [1, 2, 3, 11]
    processor = _processor_for_output(
        output_tok_ids,
        budget=len(output_tok_ids),
        end_token_ids=end_token_ids,
        newline_token_ids=[11],
    )

    processed = processor.apply(torch.zeros((1, 20)))

    assert processed[0, 7] == 0
    assert torch.isneginf(processed[0, :7]).all()
    assert torch.isneginf(processed[0, 8:]).all()
    state = processor.logit_processor_state[0]
    assert state["is_thinking"] is True
    assert state["start_of_end"] is True
