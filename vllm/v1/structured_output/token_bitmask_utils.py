# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch


def get_quote_boundary_token_ids(tokenizer) -> tuple[int | None, tuple[int, ...]]:
    quote_ids = tokenizer.encode('"', add_special_tokens=False)
    if len(quote_ids) != 1:
        return None, ()
    quote_token_id = quote_ids[0]

    if hasattr(tokenizer, "get_vocab"):
        vocab = tokenizer.get_vocab()
    else:
        vocab = getattr(tokenizer, "vocab", {})

    quote_prefixed_token_ids: list[int] = []
    for token, token_id in vocab.items():
        if token_id == quote_token_id or '"' not in token:
            continue
        try:
            text = tokenizer.convert_tokens_to_string([token])
        except Exception:
            try:
                text = tokenizer.decode([token_id], skip_special_tokens=False)
            except Exception:
                continue
        if text.startswith('"') and text != '"':
            quote_prefixed_token_ids.append(token_id)

    return quote_token_id, tuple(quote_prefixed_token_ids)


def make_token_clear_mask(vocab_size: int, token_ids: tuple[int, ...]) -> torch.Tensor:
    mask = torch.full(((vocab_size + 31) // 32,), -1, dtype=torch.int32)
    for token_id in token_ids:
        if token_id < 0 or token_id >= vocab_size:
            continue
        word_index = token_id // 32
        bit_index = token_id % 32
        value = int(mask[word_index].item()) & 0xFFFFFFFF
        value &= ~(1 << bit_index) & 0xFFFFFFFF
        if value >= (1 << 31):
            value -= 1 << 32
        mask[word_index] = value
    return mask


def token_allowed(bitmask: torch.Tensor, token_id: int) -> bool:
    word_index = token_id // 32
    if word_index >= bitmask.shape[0]:
        return False
    bit_index = token_id % 32
    return bool(int(bitmask[word_index].item()) & (1 << bit_index))
