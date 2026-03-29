# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable, Sequence
from typing import Any
import re

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.entrypoints.openai.responses.protocol import (
    ResponsesRequest,
)
from vllm.reasoning.abs_reasoning_parsers import ReasoningParser
from vllm.tokenizers import TokenizerLike


class NemotronReasoningParser(ReasoningParser):
    """Reasoning parser for Nemotron Nano v2 style <think>...</think> output.

    The checkpoint chat template controls reasoning with `/think` and
    `/no_think`, but the generated answer itself still uses literal
    `<think>...</think>` tags. Those tags may span multiple tokens, so the
    streaming path uses text-buffer parsing instead of relying on single-token
    markers.
    """

    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        self.start_token = "<think>"
        self.end_token = "</think>"
        # Fast tokenizer mutation is not thread-safe during concurrent request
        # setup. Nemotron emits literal reasoning tags, so text-space parsing is
        # sufficient and avoids "Already borrowed" failures under load.
        self.start_token_ids: list[int] = []
        self.end_token_ids: list[int] = []
        self._reasoning_ended_stream = False

    @staticmethod
    def _contains_subsequence(haystack: Sequence[int], needle: Sequence[int]) -> bool:
        if not needle or len(haystack) < len(needle):
            return False
        last = len(haystack) - len(needle) + 1
        for idx in range(last):
            if list(haystack[idx : idx + len(needle)]) == list(needle):
                return True
        return False

    @staticmethod
    def _find_subsequence(haystack: Sequence[int], needle: Sequence[int]) -> int:
        if not needle or len(haystack) < len(needle):
            return -1
        last = len(haystack) - len(needle) + 1
        for idx in range(last):
            if list(haystack[idx : idx + len(needle)]) == list(needle):
                return idx
        return -1

    def _decode_ids(self, input_ids: Sequence[int]) -> str:
        if not input_ids:
            return ""
        try:
            return self.model_tokenizer.decode(
                list(input_ids), skip_special_tokens=False
            )
        except Exception:
            return ""

    def _is_reasoning_end_from_text(self, text: str) -> bool:
        if not text:
            return False
        last_start = text.rfind(self.start_token)
        last_end = text.rfind(self.end_token)
        last_partial_end = text.rfind("</think")
        last_tool_call = text.rfind("<TOOLCALL>")
        if last_end < 0:
            if last_tool_call >= 0:
                if last_partial_end >= 0 and last_partial_end < last_tool_call:
                    return True
                if last_start < 0 or last_tool_call > last_start:
                    return True
            return False
        if last_start < 0:
            return True
        return last_end > last_start

    @staticmethod
    def _message_field(message: Any, field: str, default: Any = None) -> Any:
        if isinstance(message, dict):
            return message.get(field, default)
        return getattr(message, field, default)

    @staticmethod
    def _extract_message_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(parts)
        return ""

    def _thinking_enabled(
        self, request: ChatCompletionRequest | ResponsesRequest | None
    ) -> bool:
        if request is None or not hasattr(request, "messages"):
            return True

        enabled = True
        for message in getattr(request, "messages", []) or []:
            role = self._message_field(message, "role", None)
            if role not in ("system", "user"):
                continue
            text = self._extract_message_text(
                self._message_field(message, "content", "")
            )
            last_marker = None
            for match in re.finditer(r"/(?:no_)?think\b", text):
                last_marker = match.group(0)
            if last_marker == "/no_think":
                enabled = False
            elif last_marker == "/think":
                enabled = True
        return enabled

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        last_marker = None
        pos = 0
        while pos < len(input_ids):
            start_pos = self._find_subsequence(input_ids[pos:], self.start_token_ids)
            end_pos = self._find_subsequence(input_ids[pos:], self.end_token_ids)
            next_start = None if start_pos < 0 else pos + start_pos
            next_end = None if end_pos < 0 else pos + end_pos
            if next_start is None and next_end is None:
                break
            if next_end is not None and (
                next_start is None or next_end <= next_start
            ):
                last_marker = "end"
                pos = next_end + len(self.end_token_ids)
            else:
                last_marker = "start"
                pos = next_start + len(self.start_token_ids)
        if last_marker is not None:
            return last_marker == "end"
        return self._is_reasoning_end_from_text(self._decode_ids(input_ids))

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Iterable[int]
    ) -> bool:
        delta_ids = list(delta_ids)
        if self._contains_subsequence(delta_ids, self.end_token_ids):
            return True
        if len(self.end_token_ids) > 1:
            window = list(input_ids)[-len(self.end_token_ids) :]
            if window == self.end_token_ids:
                return True
        if self._is_reasoning_end_from_text(self._decode_ids(input_ids)):
            return True
        return self._is_reasoning_end_from_text(self._decode_ids(delta_ids))

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        end_idx = self._find_subsequence(input_ids, self.end_token_ids)
        if end_idx < 0:
            return []
        return input_ids[end_idx + len(self.end_token_ids) :]

    def extract_reasoning(
        self, model_output: str, request: ChatCompletionRequest | ResponsesRequest
    ) -> tuple[str | None, str | None]:
        text = model_output
        if text.startswith(self.start_token):
            text = text[len(self.start_token) :]

        if self.end_token not in text:
            if self._thinking_enabled(request):
                return text, None
            return None, text

        reasoning, _, content = text.partition(self.end_token)
        return reasoning or None, content or None

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        del previous_token_ids, current_token_ids, delta_token_ids

        if self._reasoning_ended_stream or self.end_token in previous_text:
            self._reasoning_ended_stream = True
            return DeltaMessage(content=delta_text) if delta_text else None

        prev_text = previous_text
        curr_text = current_text
        if prev_text.startswith(self.start_token):
            prev_text = prev_text[len(self.start_token) :]
        if curr_text.startswith(self.start_token):
            curr_text = curr_text[len(self.start_token) :]

        if self.end_token not in curr_text:
            reasoning_delta = curr_text[len(prev_text) :]
            return DeltaMessage(reasoning=reasoning_delta) if reasoning_delta else None

        self._reasoning_ended_stream = True
        reasoning_part, _, content_part = curr_text.partition(self.end_token)
        prev_reasoning_len = len(prev_text)

        reasoning_delta = None
        if len(reasoning_part) > prev_reasoning_len:
            reasoning_delta = reasoning_part[prev_reasoning_len:]

        content_delta = None
        prev_content = ""
        if self.end_token in prev_text:
            prev_content = prev_text.partition(self.end_token)[2]
        if len(content_part) > len(prev_content):
            content_delta = content_part[len(prev_content) :]
        if not prev_content and content_delta is not None and not content_delta.strip():
            content_delta = None

        if reasoning_delta or content_delta:
            return DeltaMessage(
                reasoning=reasoning_delta or None,
                content=content_delta or None,
            )
        return None
