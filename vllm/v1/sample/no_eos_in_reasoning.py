# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in "no end-of-turn while reasoning is open" request constraint.

A request opts in with ``SamplingParams.extra_args`` (``vllm_xargs`` in the
OpenAI server) key ``nemotron_block_eos_in_reasoning``. While none of the
request's *generated* tokens is a reasoning-end token, every stop token of the
request (EOS, generation-config EOS ids, ``stop_token_ids``, and for ``true``
also ``<|im_end|>``; a stop token that is itself a reasoning-end token stays
allowed) is masked to ``-inf`` in the target distribution. Once a
reasoning-end token has been generated the constraint is lifted for the rest of
the request. ``max_tokens`` still applies.

Accepted values:

- ``true``: the reasoning-end tokens are ``</think>`` and ``<tool_call>``
  (Nemotron: a tool call can follow reasoning without ``</think>``), resolved
  with the tokenizer. Both must be single tokens.
- a non-empty list of reasoning-end token ids.
- ``false`` / ``null``: disabled (``vllm_xargs`` sends ``1``/``0`` for
  ``true``/``false``; both spellings are accepted).

The engine normalizes the value into ``{"reasoning_end_token_ids": [...],
"blocked_token_ids": [...]}`` (also accepted, and re-validated, as input);
the Model Runner V2 sampler reads only that form.

The flag is ignored (with a warning) for requests with ``allowed_token_ids`` or
bad words, whose token constraints take precedence, and by engines without the
V2 model runner. If a structured-output grammar permits only a stop token while
reasoning is open, the grammar wins and the stop token is kept.
"""

from collections.abc import Callable
from typing import Any

from vllm.exceptions import VLLMValidationError
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.tokenizers import TokenizerLike

logger = init_logger(__name__)

NO_EOS_IN_REASONING_KEY = "nemotron_block_eos_in_reasoning"
END_IDS = "reasoning_end_token_ids"
BLOCKED_IDS = "blocked_token_ids"
DEFAULT_REASONING_END_TOKENS = ("</think>", "<tool_call>")
DEFAULT_BLOCKED_TOKENS = ("<|im_end|>",)
MAX_REASONING_END_TOKEN_IDS = 8
MAX_BLOCKED_TOKEN_IDS = 128


def _invalid(message: str, value: Any = None) -> VLLMValidationError:
    return VLLMValidationError(
        f"{NO_EOS_IN_REASONING_KEY}: {message}",
        parameter=NO_EOS_IN_REASONING_KEY,
        value=value,
    )


def _single_token(tokenizer: TokenizerLike, token: str) -> int | None:
    encoded = tokenizer.encode(token, add_special_tokens=False)
    return int(encoded[0]) if len(encoded) == 1 else None


def resolve_default_token_ids(tokenizer: TokenizerLike) -> tuple[list[int], list[int]]:
    """Return (reasoning-end ids, extra blocked ids) for the ``true`` form."""
    end_ids = []
    for token in DEFAULT_REASONING_END_TOKENS:
        token_id = _single_token(tokenizer, token)
        if token_id is None:
            raise _invalid(
                f"{token!r} is not a single token for this model; pass the "
                "reasoning-end token ids explicitly"
            )
        end_ids.append(token_id)
    blocked = [
        token_id
        for token in DEFAULT_BLOCKED_TOKENS
        if (token_id := _single_token(tokenizer, token)) is not None
    ]
    return sorted(set(end_ids)), sorted(set(blocked))


def _token_ids(value: Any, what: str) -> list[int]:
    if (
        not isinstance(value, (list, tuple))
        or not value
        or not all(isinstance(i, int) and not isinstance(i, bool) for i in value)
    ):
        raise _invalid(f"expected a non-empty list of {what}", value)
    return sorted(set(value))


def request_stop_token_ids(params: SamplingParams) -> set[int]:
    """Every token id that ends this request when sampled."""
    stop_ids = set(params.all_stop_token_ids)
    stop_ids.update(params.stop_token_ids or ())
    if params.eos_token_id is not None:
        stop_ids.add(params.eos_token_id)
    return stop_ids


def normalize_no_eos_in_reasoning(
    params: SamplingParams,
    *,
    resolve_default_token_ids: Callable[[], tuple[list[int], list[int]]] | None,
    vocab_size: int,
    use_v2_model_runner: bool,
) -> None:
    """Validate the request flag and store its normalized form in extra_args.

    The caller's extra_args dict is never mutated; a new dict is assigned.

    Raises:
        VLLMValidationError: if the value is malformed.
    """
    extra_args = params.extra_args
    if not extra_args or NO_EOS_IN_REASONING_KEY not in extra_args:
        return
    value = extra_args[NO_EOS_IN_REASONING_KEY]
    rest = {k: v for k, v in extra_args.items() if k != NO_EOS_IN_REASONING_KEY}
    if type(value) is int and value in (0, 1):
        # vllm_xargs coerces JSON true/false to 1/0.
        value = bool(value)
    if value is None or value is False:
        params.extra_args = rest
        return
    if not use_v2_model_runner:
        logger.warning_once(
            "%s is ignored: it requires the V2 model runner "
            "(VLLM_USE_V2_MODEL_RUNNER=1).",
            NO_EOS_IN_REASONING_KEY,
        )
        params.extra_args = rest
        return
    if params.allowed_token_ids is not None or params.bad_words_token_ids:
        # Client token constraints take precedence: combined with them the mask
        # could leave only stop tokens.
        logger.warning_once(
            "%s is ignored for requests with allowed_token_ids or bad words.",
            NO_EOS_IN_REASONING_KEY,
        )
        params.extra_args = rest
        return

    extra_blocked: list[int] = []
    if value is True:
        if resolve_default_token_ids is None:
            raise _invalid("a tokenizer is required to resolve the default tokens")
        end_ids, extra_blocked = resolve_default_token_ids()
    elif isinstance(value, dict):
        if set(value) - {END_IDS, BLOCKED_IDS}:
            raise _invalid(f"unexpected keys {sorted(set(value))}", value)
        end_ids = _token_ids(value.get(END_IDS), "reasoning-end token ids")
        if BLOCKED_IDS in value:
            extra_blocked = _token_ids(value[BLOCKED_IDS], "blocked token ids")
    else:
        end_ids = _token_ids(value, "reasoning-end token ids")

    # A caller that stops on a reasoning-end token keeps that stop.
    blocked = sorted(request_stop_token_ids(params).union(extra_blocked) - set(end_ids))
    if not blocked:
        params.extra_args = rest
        return
    if len(end_ids) > MAX_REASONING_END_TOKEN_IDS:
        raise _invalid(
            f"at most {MAX_REASONING_END_TOKEN_IDS} reasoning-end token ids", value
        )
    if len(blocked) > MAX_BLOCKED_TOKEN_IDS:
        raise _invalid(f"at most {MAX_BLOCKED_TOKEN_IDS} stop token ids")
    if any(i < 0 or i >= vocab_size for i in (*end_ids, *blocked)):
        raise _invalid(f"token ids must be in [0, {vocab_size})", value)
    params.extra_args = {
        **rest,
        NO_EOS_IN_REASONING_KEY: {END_IDS: end_ids, BLOCKED_IDS: blocked},
    }


def get_normalized_token_ids(
    extra_args: dict[str, Any] | None, vocab_size: int
) -> tuple[list[int], list[int]] | None:
    """Worker side: (end ids, blocked ids) of an enabled request, else None.

    Only the normalized form is honored; anything else (a request that bypassed
    the input processor) is ignored with a warning.
    """
    if not extra_args or NO_EOS_IN_REASONING_KEY not in extra_args:
        return None
    value = extra_args[NO_EOS_IN_REASONING_KEY]
    if value is None or value is False:
        return None
    end_ids = value.get(END_IDS) if isinstance(value, dict) else None
    blocked = value.get(BLOCKED_IDS) if isinstance(value, dict) else None
    ok = all(
        isinstance(ids, list)
        and 0 < len(ids) <= limit
        and all(
            isinstance(i, int) and not isinstance(i, bool) and 0 <= i < vocab_size
            for i in ids
        )
        for ids, limit in (
            (end_ids, MAX_REASONING_END_TOKEN_IDS),
            (blocked, MAX_BLOCKED_TOKEN_IDS),
        )
    )
    if not ok:
        logger.warning_once(
            "Ignoring %s: the value was not normalized by the input processor.",
            NO_EOS_IN_REASONING_KEY,
        )
        return None
    return end_ids, blocked
