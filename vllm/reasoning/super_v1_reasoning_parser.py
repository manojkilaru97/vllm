# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable, Sequence

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.reasoning.abs_reasoning_parsers import ReasoningParser


class SuperV1ReasoningParser(ReasoningParser):
    """Reasoning parser for Llama-3.3-Nemotron-Super-49B-v1.

    This model emits textual ``<think>...</think>`` markers, but those markers
    are not guaranteed to be single tokenizer entries. The parser therefore
    tracks reasoning boundaries from accumulated text and only uses encoded
    marker *sequences* for token-id based checks needed by structured output.
    """

    start_token = "<think>"
    end_token = "</think>"

    def __init__(self, tokenizer, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        self.start_token_ids = self._encode_marker(self.start_token)
        self.end_token_ids = self._encode_marker(self.end_token)
        self._stream_buffer = ""
        self._stream_in_reasoning = False
        self._stream_waiting_for_start = False
        self._stream_seen_reasoning_start = False
        self._stream_seen_reasoning_end = False

    @property
    def supports_prompt_reasoning_end_check(self) -> bool:
        # Wait for the generated </think> before enabling constrained decoding.
        return False

    def _encode_marker(self, marker: str) -> list[int]:
        if hasattr(self.model_tokenizer, "encode"):
            return list(self.model_tokenizer.encode(marker, add_special_tokens=False))
        tokenized = self.model_tokenizer(marker, add_special_tokens=False)
        return list(tokenized.input_ids)

    def _decode_ids(self, input_ids: Sequence[int]) -> str:
        return self.model_tokenizer.decode(
            input_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

    def _find_subsequence(self, haystack: Sequence[int], needle: Sequence[int]) -> int:
        if not needle or len(needle) > len(haystack):
            return -1
        last_start = len(haystack) - len(needle)
        for i in range(last_start + 1):
            if list(haystack[i : i + len(needle)]) == list(needle):
                return i
        return -1

    def _is_partial_start_prefix(self, input_ids: Sequence[int]) -> bool:
        if not input_ids:
            return False
        stripped = self._decode_ids(input_ids).lstrip()
        return bool(stripped) and self.start_token.startswith(stripped)

    def _strip_start_token(self, text: str) -> str:
        stripped = text
        if self.start_token in stripped:
            _, _, stripped = stripped.partition(self.start_token)
        return stripped

    def _suffix_prefix_overlap_len(self, text: str, marker: str) -> int:
        max_overlap = min(len(text), len(marker) - 1)
        for size in range(max_overlap, 0, -1):
            if text.endswith(marker[:size]):
                return size
        return 0

    def _sanitize_post_think_content(self, text: str | None) -> str | None:
        if text is None:
            return None
        sanitized = text
        while True:
            updated = sanitized.lstrip()
            if updated.startswith(self.end_token):
                sanitized = updated[len(self.end_token) :]
                continue
            break
        sanitized = sanitized.replace(self.end_token, "")
        sanitized = sanitized.lstrip("\n")
        return sanitized or None

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        if self._stream_seen_reasoning_start and not self._stream_seen_reasoning_end:
            return False
        if self._stream_waiting_for_start:
            return False
        decoded = self._decode_ids(input_ids)
        if self.end_token in decoded:
            return True
        if self.start_token in decoded:
            return False
        if self._is_partial_start_prefix(input_ids):
            return False
        return True

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Iterable[int]
    ) -> bool:
        return self.is_reasoning_end(input_ids)

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        decoded = self._decode_ids(input_ids)
        if self.end_token not in decoded:
            if self.start_token in decoded or self._is_partial_start_prefix(input_ids):
                return []
            return input_ids

        for idx in range(len(input_ids) + 1):
            if self.end_token in self._decode_ids(input_ids[:idx]):
                return input_ids[idx:]
        return input_ids

    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str | None, str | None]:
        has_start = self.start_token in model_output
        has_end = self.end_token in model_output
        if not has_start and not has_end:
            return None, model_output or None

        model_output = self._strip_start_token(model_output)
        if self.end_token not in model_output:
            return model_output or None, None
        reasoning, _, content = model_output.partition(self.end_token)
        return reasoning or None, self._sanitize_post_think_content(content)

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        self._stream_buffer += delta_text

        reasoning_part: str | None = None
        content_part: str | None = None

        if not self._stream_in_reasoning:
            start_idx = self._stream_buffer.find(self.start_token)
            if start_idx >= 0:
                prefix = self._stream_buffer[:start_idx]
                if prefix:
                    content_part = prefix
                self._stream_buffer = self._stream_buffer[
                    start_idx + len(self.start_token) :
                ]
                self._stream_in_reasoning = True
                self._stream_waiting_for_start = False
                self._stream_seen_reasoning_start = True
            else:
                overlap = self._suffix_prefix_overlap_len(
                    self._stream_buffer, self.start_token
                )
                safe_len = len(self._stream_buffer) - overlap
                self._stream_waiting_for_start = overlap > 0
                if safe_len > 0:
                    content_part = self._stream_buffer[:safe_len]
                    self._stream_buffer = self._stream_buffer[safe_len:]
                    self._stream_waiting_for_start = False
                return DeltaMessage(content=content_part) if content_part else None

        if self._stream_in_reasoning:
            end_idx = self._stream_buffer.find(self.end_token)
            if end_idx >= 0:
                reasoning_text = self._stream_buffer[:end_idx]
                remainder = self._stream_buffer[end_idx + len(self.end_token) :]
                self._stream_buffer = ""
                self._stream_in_reasoning = False
                self._stream_seen_reasoning_end = True
                reasoning_part = reasoning_text or None
                content_part = (content_part or "") + (remainder or "")
                content_part = self._sanitize_post_think_content(content_part)
                return DeltaMessage(
                    reasoning=reasoning_part,
                    content=content_part,
                )

            overlap = self._suffix_prefix_overlap_len(self._stream_buffer, self.end_token)
            safe_len = len(self._stream_buffer) - overlap
            if safe_len > 0:
                reasoning_part = self._stream_buffer[:safe_len]
                self._stream_buffer = self._stream_buffer[safe_len:]

        if content_part or reasoning_part:
            return DeltaMessage(
                reasoning=reasoning_part,
                content=content_part,
            )
        return None
