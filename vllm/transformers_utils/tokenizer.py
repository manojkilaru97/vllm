# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import warnings

from vllm.tokenizers.mistral import MistralTokenizer as MistralTokenizer


def __getattr__(name: str):
    # Keep until lm-eval is updated
    if name == "get_tokenizer":
        from vllm.tokenizers import get_tokenizer

        warnings.warn(
            "`vllm.transformers_utils.tokenizer.get_tokenizer` "
            "has been moved to `vllm.tokenizers.get_tokenizer`. "
            "The old name will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )

        return get_tokenizer

    # Keep until lm-format-enforcer updates its vLLM integration import.
    if name == "MistralTokenizer":
        from vllm.tokenizers.mistral import MistralTokenizer

        warnings.warn(
            "`vllm.transformers_utils.tokenizer.MistralTokenizer` "
            "has moved to `vllm.tokenizers.mistral.MistralTokenizer`. "
            "The old name will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )

        return MistralTokenizer

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
