# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable, Sequence

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.entrypoints.openai.responses.protocol import (
    ResponsesRequest,
)
from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser


class NemotronV3ReasoningParser(DeepSeekR1ReasoningParser):
    """
    Reasoning parser for Nemotron V3 models.
    """

    def _end_token_prefix_suffix_len(self, text: str) -> int:
        max_len = min(len(self.end_token) - 1, len(text))
        for size in range(max_len, 0, -1):
            if text.endswith(self.end_token[:size]):
                return size
        return 0

    def _decode_token_ids(self, token_ids: Sequence[int]) -> str:
        try:
            return self.model_tokenizer.decode(
                list(token_ids), skip_special_tokens=False
            )
        except TypeError:
            return self.model_tokenizer.decode(list(token_ids))
        except Exception:
            try:
                tokens = self.model_tokenizer.convert_ids_to_tokens(list(token_ids))
                return self.model_tokenizer.convert_tokens_to_string(tokens)
            except Exception:
                return ""

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Iterable[int]
    ) -> bool:
        delta_ids = list(delta_ids)
        if super().is_reasoning_end_streaming(input_ids, delta_ids):
            return True
        if not delta_ids:
            return False

        # Structured-output gating only has token IDs. Mirror the decoded-text
        # fallback used by streaming extraction so grammar constraints start
        # when Ultra emits </think> as ordinary token pieces.
        delta_start = max(len(input_ids) - len(delta_ids), 0)
        tail_start = max(delta_start - 32, 0)
        previous_tail = self._decode_token_ids(input_ids[tail_start:delta_start])
        current_tail = self._decode_token_ids(input_ids[tail_start:])
        return self.end_token in current_tail and self.end_token not in previous_tail

    def extract_reasoning_streaming(
        self,
        previous_text,
        current_text,
        delta_text,
        previous_token_ids,
        current_token_ids,
        delta_token_ids,
    ) -> DeltaMessage | None:
        # Ultra/Nemotron can expose </think> in decoded text without the marker
        # token ID in the same streaming delta. Fall back to text markers so the
        # final answer is not swallowed as reasoning.
        if self.end_token in previous_text:
            if self.end_token in delta_text:
                _, _, content = delta_text.rpartition(self.end_token)
                return DeltaMessage(content=content or None)
            return DeltaMessage(content=delta_text) if delta_text else None
        if self.end_token in current_text:
            reasoning, _, content = current_text.rpartition(self.end_token)
            if self.start_token in reasoning:
                _, _, reasoning = reasoning.partition(self.start_token)
            previous_reasoning_len = len(previous_text) - (
                self._end_token_prefix_suffix_len(previous_text)
            )
            reasoning = reasoning[previous_reasoning_len:]
            return DeltaMessage(
                reasoning=reasoning or None,
                content=content or None,
            )

        previous_suffix_len = self._end_token_prefix_suffix_len(previous_text)
        current_suffix_len = self._end_token_prefix_suffix_len(current_text)
        if previous_suffix_len or current_suffix_len:
            reasoning = (
                (previous_text[-previous_suffix_len:] if previous_suffix_len else "")
                + delta_text
            )
            if current_suffix_len:
                reasoning = reasoning[:-current_suffix_len]
            return DeltaMessage(reasoning=reasoning or None)

        return super().extract_reasoning_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
        )

    def extract_reasoning(
        self, model_output: str, request: ChatCompletionRequest | ResponsesRequest
    ) -> tuple[str | None, str | None]:
        reasoning, final_content = super().extract_reasoning(model_output, request)
        chat_template_kwargs = getattr(request, "chat_template_kwargs", None)
        thinking_disabled = bool(
            chat_template_kwargs
            and chat_template_kwargs.get("enable_thinking") is False
        )

        if (final_content is None or not final_content.strip()) and reasoning:
            if thinking_disabled:
                reasoning, final_content = None, reasoning
            else:
                # Ultra can stop before emitting </think>. Return the text in
                # both fields so clients that only render content do not show
                # an empty assistant message, while reasoning-aware clients can
                # still read the same text from the reasoning field.
                final_content = reasoning

        if final_content and self.end_token in final_content:
            _, _, final_content = final_content.rpartition(self.end_token)
            final_content = final_content or None

        return reasoning, final_content
