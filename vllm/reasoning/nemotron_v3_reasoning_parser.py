# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
        if self.end_token in delta_text:
            reasoning, _, content = delta_text.rpartition(self.end_token)
            if self.start_token in reasoning:
                _, _, reasoning = reasoning.partition(self.start_token)
            return DeltaMessage(
                reasoning=reasoning or None,
                content=content or None,
            )

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

        if final_content is None or not final_content.strip():
            if (
                chat_template_kwargs
                and chat_template_kwargs.get("enable_thinking") is False
            ):
                reasoning, final_content = None, reasoning
            elif reasoning:
                final_content = reasoning

        if final_content and self.end_token in final_content:
            _, _, final_content = final_content.rpartition(self.end_token)
            final_content = final_content or None

        return reasoning, final_content
