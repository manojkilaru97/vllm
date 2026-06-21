# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.structured_output.token_bitmask_utils import (
    get_quote_boundary_token_ids,
    make_token_clear_mask,
    token_allowed,
)


class FakeTokenizer:

    vocab = {
        '"': 1,
        '"ok': 2,
        '}': 3,
        '"}': 4,
        'x"': 5,
    }

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        if text == '"':
            return [self.vocab[text]]
        return []

    def get_vocab(self) -> dict[str, int]:
        return self.vocab

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return ''.join(tokens)


def test_get_quote_boundary_token_ids_finds_quote_prefixed_tokens():
    quote_token_id, quote_prefixed_token_ids = get_quote_boundary_token_ids(
        FakeTokenizer()
    )

    assert quote_token_id == 1
    assert set(quote_prefixed_token_ids) == {2, 4}


def test_make_token_clear_mask_clears_only_requested_tokens():
    bitmask = torch.full((2,), -1, dtype=torch.int32)
    clear_mask = make_token_clear_mask(64, (2, 33, -1, 64))

    bitmask.bitwise_and_(clear_mask)

    assert token_allowed(bitmask, 1)
    assert not token_allowed(bitmask, 2)
    assert token_allowed(bitmask, 3)
    assert not token_allowed(bitmask, 33)
    assert token_allowed(bitmask, 63)
    assert not token_allowed(bitmask, 64)
