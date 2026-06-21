# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import re
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

    def _decode_token_ids(
        self, token_ids: Sequence[int], *, skip_special_tokens: bool = False
    ) -> str:
        try:
            return self.model_tokenizer.decode(
                list(token_ids), skip_special_tokens=skip_special_tokens
            )
        except TypeError:
            return self.model_tokenizer.decode(list(token_ids))
        except Exception:
            try:
                tokens = self.model_tokenizer.convert_ids_to_tokens(list(token_ids))
                if skip_special_tokens:
                    tokens = [
                        token
                        for token in tokens
                        if token not in (self.start_token, self.end_token)
                    ]
                return self.model_tokenizer.convert_tokens_to_string(tokens)
            except Exception:
                return ""

    def _split_stripped_end_token_delta(
        self, delta_text: str, delta_token_ids: Sequence[int]
    ) -> DeltaMessage | None:
        delta_token_ids = list(delta_token_ids)
        if self.end_token_id not in delta_token_ids:
            return None

        end_index = delta_token_ids.index(self.end_token_id)
        reasoning_ids = delta_token_ids[:end_index]
        content_ids = delta_token_ids[end_index + 1 :]

        reasoning = self._decode_token_ids(
            reasoning_ids, skip_special_tokens=True
        )
        content = self._decode_token_ids(content_ids, skip_special_tokens=True)

        if not reasoning and not content and delta_text:
            if end_index == 0:
                content = delta_text
            elif end_index == len(delta_token_ids) - 1:
                reasoning = delta_text

        content = self.strip_reasoning_boundary_content("", content)
        if not reasoning and not content:
            return None
        return DeltaMessage(reasoning=reasoning or None, content=content or None)

    def strip_reasoning_boundary_content(
        self, previous_content: str, content_delta: str
    ) -> str:
        if self.end_token in content_delta:
            _, _, content_delta = content_delta.rpartition(self.end_token)
        if not previous_content or not previous_content.lstrip("\n"):
            content_delta = content_delta.lstrip("\n")
        return content_delta

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
        if self.end_token_id in delta_token_ids and self.end_token not in delta_text:
            return self._split_stripped_end_token_delta(delta_text, delta_token_ids)

        if self.end_token in previous_text:
            previous_content = previous_text.rpartition(self.end_token)[2]
            if self.end_token in delta_text:
                content = self.strip_reasoning_boundary_content(
                    previous_content, delta_text
                )
                return DeltaMessage(content=content or None)
            content = self.strip_reasoning_boundary_content(
                previous_content, delta_text
            )
            return DeltaMessage(content=content) if content else None
        if self.end_token in current_text:
            reasoning, _, content = current_text.rpartition(self.end_token)
            if self.start_token in reasoning:
                _, _, reasoning = reasoning.partition(self.start_token)
            previous_reasoning_len = len(previous_text) - (
                self._end_token_prefix_suffix_len(previous_text)
            )
            reasoning = reasoning[previous_reasoning_len:]
            content = self.strip_reasoning_boundary_content("", content)
            if not reasoning and not content:
                return None
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

        ret = super().extract_reasoning_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
        )
        if ret and ret.content:
            previous_content = ""
            if self.end_token in previous_text:
                previous_content = previous_text.rpartition(self.end_token)[2]
            ret.content = self.strip_reasoning_boundary_content(
                previous_content, ret.content
            ) or None
        return ret

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
        if final_content and self.end_token in model_output:
            final_content = (
                self.strip_reasoning_boundary_content("", final_content) or None
            )

        final_content = self._repair_boundary_brace_leak(final_content)
        return reasoning, final_content

    @staticmethod
    def _repair_boundary_brace_leak(content: str | None) -> str | None:
        """Repair a duplicate opening brace leaked at the reasoning->answer
        boundary under speculative decoding (e.g. '{\\n{"a":1}'). Valid JSON
        never starts with '{' immediately followed by '{' (keys are strings),
        so this is airtight: only fires when the original is invalid JSON and
        dropping the stray brace makes it valid. Non-stream only (streamed bytes
        cannot be retracted)."""
        if not content:
            return content
        stripped = content.lstrip()
        match = re.match(r"\{\s*(\{)", stripped)
        if not match:
            return content
        try:
            json.loads(stripped)
            return content  # already valid; do not touch
        except Exception:
            pass
        repaired = stripped[match.start(1):]
        try:
            json.loads(repaired)
        except Exception:
            return content  # repair did not yield valid JSON; leave unchanged
        return repaired
