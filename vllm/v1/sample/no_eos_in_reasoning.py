# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in "no end-of-turn while reasoning is open" request constraint.

A request opts in with ``SamplingParams.extra_args`` (``vllm_xargs`` in the
OpenAI server) key ``nemotron_block_eos_in_reasoning``. While none of the
request's *generated* tokens is a reasoning-end token, every stop token id of
the request (``SamplingParams.all_stop_token_ids``: EOS, generation-config EOS
ids and ``stop_token_ids``) is masked to ``-inf`` in the target distribution.
Once a reasoning-end token has been generated the constraint is lifted for the
rest of the request. ``max_tokens`` still applies.

Accepted values:

- ``true``: the reasoning-end tokens are ``</think>`` and ``<tool_call>``
  (Nemotron: a tool call can follow reasoning without ``</think>``), resolved
  with the tokenizer. Both must be single tokens.
- a non-empty list of token ids.
- ``false`` / ``null``: disabled.

The input processor normalizes the value to a sorted list of token ids; the
Model Runner V2 sampler reads only that normalized form.
"""

from collections.abc import Collection
from typing import Any

from vllm.exceptions import VLLMValidationError
from vllm.tokenizers import TokenizerLike

NO_EOS_IN_REASONING_KEY = "nemotron_block_eos_in_reasoning"
DEFAULT_REASONING_END_TOKENS = ("</think>", "<tool_call>")
MAX_REASONING_END_TOKEN_IDS = 8
MAX_BLOCKED_STOP_TOKEN_IDS = 128


def _invalid(message: str, value: Any = None) -> VLLMValidationError:
    return VLLMValidationError(
        f"{NO_EOS_IN_REASONING_KEY}: {message}",
        parameter=NO_EOS_IN_REASONING_KEY,
        value=value,
    )


def resolve_default_reasoning_end_token_ids(tokenizer: TokenizerLike) -> list[int]:
    ids = []
    for token in DEFAULT_REASONING_END_TOKENS:
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if len(encoded) != 1:
            raise _invalid(
                f"{token!r} is not a single token for this model; pass the "
                "reasoning-end token ids explicitly"
            )
        ids.append(int(encoded[0]))
    return sorted(set(ids))


def normalize_no_eos_in_reasoning(
    extra_args: dict[str, Any] | None,
    *,
    default_end_token_ids: "Collection[int] | None",
    vocab_size: int,
    all_stop_token_ids: Collection[int],
    use_v2_model_runner: bool,
) -> dict[str, Any] | None:
    """Validate the request flag and return extra_args with the normalized value.

    Returns ``extra_args`` unchanged when the key is absent. Returns a new dict
    (the caller's dict is never mutated) with the key removed when disabled, or
    set to a sorted list of reasoning-end token ids when enabled.

    Raises:
        VLLMValidationError: if the value is malformed or the constraint cannot
            be applied by this engine.
    """
    if not extra_args or NO_EOS_IN_REASONING_KEY not in extra_args:
        return extra_args
    value = extra_args[NO_EOS_IN_REASONING_KEY]
    if value is None or value is False:
        return {k: v for k, v in extra_args.items() if k != NO_EOS_IN_REASONING_KEY}

    if value is True:
        if default_end_token_ids is None:
            raise _invalid("a tokenizer is required to resolve the default tokens")
        end_ids = sorted(set(default_end_token_ids))
    elif isinstance(value, (list, tuple)):
        if not value or not all(
            isinstance(i, int) and not isinstance(i, bool) for i in value
        ):
            raise _invalid("expected true or a non-empty list of token ids", value)
        end_ids = sorted(set(value))
    else:
        raise _invalid("expected true or a non-empty list of token ids", value)

    if len(end_ids) > MAX_REASONING_END_TOKEN_IDS:
        raise _invalid(
            f"at most {MAX_REASONING_END_TOKEN_IDS} reasoning-end token ids", value
        )
    if any(i < 0 or i >= vocab_size for i in end_ids):
        raise _invalid(f"token ids must be in [0, {vocab_size})", value)
    if len(all_stop_token_ids) > MAX_BLOCKED_STOP_TOKEN_IDS:
        raise _invalid(
            f"at most {MAX_BLOCKED_STOP_TOKEN_IDS} stop token ids are supported"
        )
    if set(end_ids) & set(all_stop_token_ids):
        raise _invalid("a reasoning-end token cannot also be a stop token", value)
    if not use_v2_model_runner:
        raise _invalid("requires the V2 model runner (VLLM_USE_V2_MODEL_RUNNER=1)")
    return {**extra_args, NO_EOS_IN_REASONING_KEY: end_ids}


def get_reasoning_end_token_ids(extra_args: dict[str, Any] | None) -> list[int] | None:
    """Return the normalized reasoning-end token ids, or None if not enabled."""
    if not extra_args:
        return None
    value = extra_args.get(NO_EOS_IN_REASONING_KEY)
    if (
        not isinstance(value, list)
        or not value
        or len(value) > MAX_REASONING_END_TOKEN_IDS
        or not all(isinstance(i, int) and not isinstance(i, bool) for i in value)
    ):
        return None
    return value
