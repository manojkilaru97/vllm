# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request contract of the opt-in stop-token mask while reasoning is open."""

import pytest

from vllm.exceptions import VLLMValidationError
from vllm.v1.sample.no_eos_in_reasoning import (
    NO_EOS_IN_REASONING_KEY as KEY,
)
from vllm.v1.sample.no_eos_in_reasoning import (
    get_reasoning_end_token_ids,
    normalize_no_eos_in_reasoning,
    resolve_default_reasoning_end_token_ids,
)


def _normalize(extra_args, *, stop_ids=(2, 11), v2=True, defaults=(13, 14)):
    return normalize_no_eos_in_reasoning(
        extra_args,
        default_end_token_ids=defaults,
        vocab_size=1000,
        all_stop_token_ids=set(stop_ids),
        use_v2_model_runner=v2,
    )


def test_absent_flag_is_a_no_op():
    extra_args = {"other": 1}
    assert _normalize(None) is None
    assert _normalize(extra_args) is extra_args
    assert get_reasoning_end_token_ids(extra_args) is None


@pytest.mark.parametrize("value", [False, None])
def test_disabled_flag_is_removed_without_mutating_caller(value):
    extra_args = {KEY: value, "other": 1}
    assert _normalize(extra_args) == {"other": 1}
    assert extra_args == {KEY: value, "other": 1}


def test_true_resolves_default_tokens():
    extra_args = {KEY: True}
    normalized = _normalize(extra_args)
    assert normalized == {KEY: [13, 14]}
    assert extra_args == {KEY: True}
    assert get_reasoning_end_token_ids(normalized) == [13, 14]


def test_explicit_ids_are_sorted_and_deduplicated():
    assert _normalize({KEY: [14, 13, 14]}) == {KEY: [13, 14]}


@pytest.mark.parametrize(
    "value", [[], [True], ["13"], [13.0], 1, "true", {"ids": [13]}, [-1], [1000]]
)
def test_malformed_values_are_rejected(value):
    with pytest.raises(VLLMValidationError):
        _normalize({KEY: value})


def test_end_token_cannot_be_a_stop_token():
    with pytest.raises(VLLMValidationError):
        _normalize({KEY: [11, 13]})


def test_requires_v2_model_runner():
    with pytest.raises(VLLMValidationError, match="V2 model runner"):
        _normalize({KEY: True}, v2=False)


def test_true_without_tokenizer_is_rejected():
    with pytest.raises(VLLMValidationError):
        _normalize({KEY: True}, defaults=None)


def test_worker_ignores_unnormalized_values():
    assert get_reasoning_end_token_ids({KEY: True}) is None
    assert get_reasoning_end_token_ids({KEY: [True]}) is None


class _Tokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def encode(self, text, add_special_tokens=False):
        return self.vocab[text]


def test_default_tokens_must_be_single_tokens():
    tokenizer = _Tokenizer({"</think>": [13], "<tool_call>": [14]})
    assert resolve_default_reasoning_end_token_ids(tokenizer) == [13, 14]
    tokenizer = _Tokenizer({"</think>": [13], "<tool_call>": [60, 61]})
    with pytest.raises(VLLMValidationError):
        resolve_default_reasoning_end_token_ids(tokenizer)
