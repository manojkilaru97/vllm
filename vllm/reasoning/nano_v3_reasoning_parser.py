# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.entrypoints.openai.responses.protocol import (
    ResponsesRequest,
)
from vllm.reasoning.abs_reasoning_parsers import ReasoningParserManager
from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser


@ReasoningParserManager.register_module("nano_v3")
class NanoV3ReasoningParser(DeepSeekR1ReasoningParser):
    def __init__(self, tokenizer, *args, **kwargs):
        self.chat_template_kwargs = dict(kwargs.get("chat_template_kwargs") or {})
        self._trim_initial_post_think_newlines = True
        super().__init__(tokenizer, *args, **kwargs)

    def _sanitize_post_think_content(
        self,
        text: str | None,
        *,
        trim_initial_newlines: bool = True,
    ) -> str | None:
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
        if trim_initial_newlines:
            removed = 0
            while removed < 2 and sanitized.startswith("\n"):
                sanitized = sanitized[1:]
                removed += 1
        return sanitized or None

    @staticmethod
    def _request_chat_template_kwargs(
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> dict:
        return getattr(request, "chat_template_kwargs", None) or {}

    def _thinking_enabled_from_request(
        self,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> bool:
        kwargs = self._request_chat_template_kwargs(request)
        return kwargs.get("enable_thinking", True) is not False

    def _force_nonempty_content_from_request(
        self,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> bool:
        kwargs = self._request_chat_template_kwargs(request)
        return kwargs.get("force_nonempty_content", False) is True

    def _thinking_enabled_streaming(self) -> bool:
        return self.chat_template_kwargs.get("enable_thinking", True) is not False

    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str | None, str | None]:
        reasoning_content, final_content = super().extract_reasoning(
            model_output, request
        )
        if (
            (
                not self._thinking_enabled_from_request(request)
                or self._force_nonempty_content_from_request(request)
            )
            and final_content is None
        ):
            reasoning_content, final_content = final_content, reasoning_content
        elif self._thinking_enabled_from_request(request):
            final_content = self._sanitize_post_think_content(
                final_content,
                trim_initial_newlines=True,
            )

        return reasoning_content, final_content

    def extract_reasoning_streaming(
        self,
        previous_text,
        current_text,
        delta_text,
        previous_token_ids,
        current_token_ids,
        delta_token_ids,
    ):
        if not self._thinking_enabled_streaming():
            return DeltaMessage(content=delta_text or None)

        ret = super().extract_reasoning_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
        )
        if ret and ret.content is not None:
            ret.content = self._sanitize_post_think_content(
                ret.content,
                trim_initial_newlines=self._trim_initial_post_think_newlines,
            )
            if ret.content is not None and self._trim_initial_post_think_newlines:
                self._trim_initial_post_think_newlines = False
        return ret
