# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json
import logging
import os
import time
from collections.abc import AsyncGenerator, AsyncIterator
from collections.abc import Sequence
from collections.abc import Sequence as GenericSequence
from http import HTTPStatus
from typing import TYPE_CHECKING, Any, Final

import partial_json_parser
import regex as re
from fastapi import Request
from partial_json_parser.core.options import Allow

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import (
    ChatTemplateContentFormatOption,
    ConversationMessage,
    get_history_tool_calls_cnt,
    make_tool_call_id,
)
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionLogProb,
    ChatCompletionLogProbs,
    ChatCompletionLogProbsContent,
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
)
from vllm.entrypoints.openai.chat_completion.stream_harmony import (
    TokenState,
    extract_harmony_streaming_delta,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ErrorResponse,
    FunctionCall,
    PromptTokenUsageInfo,
    RequestResponseMetadata,
    ToolCall,
    UsageInfo,
)
from vllm.entrypoints.openai.engine.serving import (
    GenerationError,
    OpenAIServing,
    clamp_prompt_logprobs,
)
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.openai.parser.harmony_utils import (
    get_stop_tokens_for_assistant_actions,
    get_streamable_parser_for_assistant,
    parse_chat_output,
)
from vllm.entrypoints.openai.utils import maybe_filter_parallel_tool_calls
from vllm.entrypoints.utils import get_max_tokens, should_include_usage
from vllm.inputs.data import ProcessorInputs
from vllm.logger import init_logger
from vllm.logprobs import Logprob
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.parser import ParserManager
from vllm.reasoning import ReasoningParser
from vllm.renderers import ChatParams
from vllm.sampling_params import BeamSearchParams, SamplingParams
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers import ToolParser
from vllm.tool_parsers.mistral_tool_parser import MistralToolCall
from vllm.tool_parsers.utils import partial_json_loads
from vllm.utils.collection_utils import as_list
from vllm.utils.mistral import is_mistral_tokenizer

if TYPE_CHECKING:
    from vllm.entrypoints.serve.render.serving import OpenAIServingRender

from vllm.entrypoints.openai.request_metrics import classify_chat_request

logger = init_logger(__name__)
payload_logger = logging.getLogger("vllm.payload")


class OpenAIServingChat(OpenAIServing):
    def __init__(
        self,
        engine_client: EngineClient,
        models: OpenAIServingModels,
        response_role: str,
        *,
        openai_serving_render: "OpenAIServingRender",
        request_logger: RequestLogger | None,
        chat_template: str | None,
        chat_template_content_format: ChatTemplateContentFormatOption,
        trust_request_chat_template: bool = False,
        return_tokens_as_token_ids: bool = False,
        reasoning_parser: str = "",
        enable_auto_tools: bool = False,
        exclude_tools_when_tool_choice_none: bool = False,
        tool_parser: str | None = None,
        enable_prompt_tokens_details: bool = False,
        enable_force_include_usage: bool = False,
        enable_log_outputs: bool = False,
        enable_log_deltas: bool = False,
        log_error_stack: bool = False,
        default_chat_template_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            engine_client=engine_client,
            models=models,
            request_logger=request_logger,
            return_tokens_as_token_ids=return_tokens_as_token_ids,
        )

        self.openai_serving_render = openai_serving_render
        self.response_role = response_role
        self.chat_template = chat_template
        self.chat_template_content_format: Final = chat_template_content_format
        self.trust_request_chat_template = trust_request_chat_template
        self.default_chat_template_kwargs = default_chat_template_kwargs or {}
        self.enable_log_outputs = enable_log_outputs
        self.enable_log_deltas = enable_log_deltas

        # set up reasoning parser
        self.reasoning_parser_cls = ParserManager.get_reasoning_parser(
            reasoning_parser_name=reasoning_parser
        )
        # set up tool use
        self.enable_auto_tools: bool = enable_auto_tools
        self.tool_parser = ParserManager.get_tool_parser(
            tool_parser_name=tool_parser,
            enable_auto_tools=enable_auto_tools,
            model_name=self.model_config.model,
        )
        self.exclude_tools_when_tool_choice_none = exclude_tools_when_tool_choice_none

        self.enable_prompt_tokens_details = enable_prompt_tokens_details
        self.enable_force_include_usage = enable_force_include_usage
        self.default_sampling_params = self.model_config.get_diff_sampling_param()
        mc = self.model_config
        self.override_max_tokens = (
            self.default_sampling_params.get("max_tokens")
            if mc.generation_config not in ("auto", "vllm")
            else getattr(mc, "override_generation_config", {}).get("max_new_tokens")
        )
        self.use_harmony = self.model_config.hf_config.model_type == "gpt_oss"
        if self.use_harmony:
            if "stop_token_ids" not in self.default_sampling_params:
                self.default_sampling_params["stop_token_ids"] = []
            self.default_sampling_params["stop_token_ids"].extend(
                get_stop_tokens_for_assistant_actions()
            )

        # Handle tool call ID type for Kimi K2 (supporting test mocking via overrides)
        hf_overrides = getattr(self.model_config, "hf_overrides", None)
        if self.model_config.hf_text_config.model_type == "kimi_k2" or (
            isinstance(hf_overrides, dict)
            and hf_overrides.get("model_type") == "kimi_k2"
        ):
            self.tool_call_id_type = "kimi_k2"
        else:
            self.tool_call_id_type = "random"

        # NOTE(woosuk): While OpenAI's chat completion API supports browsing
        # for some models, currently vLLM doesn't support it. Please use the
        # Responses API instead.
        self.supports_browsing = False
        self.browser_tool = None
        # NOTE(woosuk): Chat completion API does not support code interpreter.
        # Please use the Responses API instead.
        self.supports_code_interpreter = False
        self.python_tool = None

    def warmup(self) -> None:
        self.renderer.warmup(
            ChatParams(
                chat_template=self.chat_template,
                chat_template_content_format=self.chat_template_content_format,
                chat_template_kwargs=self.default_chat_template_kwargs,
            )
        )

    def _compute_newline_token_ids(
        self,
        tokenizer: TokenizerLike,
        strings: list[str] | None = None,
    ) -> list[int]:
        cached = getattr(tokenizer, "_vllm_newline_token_ids", None)
        if isinstance(cached, list):
            return cached

        if strings is None:
            strings = ["\n", "\r\n", "\n\n"]

        newline_ids: set[int] = set()

        for s in strings:
            try:
                encoded = tokenizer.encode(s, add_special_tokens=False)
            except TypeError:
                encoded = tokenizer.encode(s)  # type: ignore[call-arg]
            except Exception:
                continue
            for token_id in encoded:
                try:
                    newline_ids.add(int(token_id))
                except Exception:
                    continue

        vocab_size = getattr(tokenizer, "vocab_size", None)
        if isinstance(vocab_size, int) and 0 < vocab_size <= 300000:
            for token_id in range(vocab_size):
                try:
                    token_text = tokenizer.decode([token_id])
                except Exception:
                    continue
                if any(token_text.endswith(s) for s in strings):
                    newline_ids.add(token_id)

        ids_list = sorted(newline_ids)
        setattr(tokenizer, "_vllm_newline_token_ids", ids_list)
        return ids_list

    def _get_reasoning_parser_request_view(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionRequest:
        resolved_chat_template_kwargs = request.get_resolved_chat_template_kwargs()
        if resolved_chat_template_kwargs == (request.chat_template_kwargs or {}):
            return request
        return request.model_copy(
            update={"chat_template_kwargs": resolved_chat_template_kwargs}
        )

    def _inject_think_end_token_id(
        self,
        sampling_params: SamplingParams,
        request: ChatCompletionRequest,
        tokenizer: TokenizerLike,
        reasoning_parser: ReasoningParser | None,
    ) -> None:
        if reasoning_parser is None:
            return

        chat_template_kwargs = self._prepare_extra_chat_template_kwargs(
            request.get_resolved_chat_template_kwargs(),
            self.default_chat_template_kwargs,
        )
        parser_chat_template_kwargs = dict(chat_template_kwargs)

        # Reasoning budget is a request-level knob. Keep backwards
        # compatibility with older clients that sent it via chat_template_kwargs.
        request_reasoning_budget = getattr(request, "reasoning_budget", None)
        reasoning_budget = request_reasoning_budget
        if reasoning_budget is None:
            reasoning_budget = chat_template_kwargs.get("reasoning_budget")
        if (
            reasoning_budget is None
            and request.structured_outputs is not None
            and parser_chat_template_kwargs.get("enable_thinking", True) is not False
        ):
            max_tokens = getattr(sampling_params, "max_tokens", None) or 0
            if max_tokens > 0:
                reserve = max(64, min(256, max_tokens // 2))
                derived_budget = max(32, max_tokens - reserve)
                if parser_chat_template_kwargs.get("low_effort"):
                    derived_budget = min(derived_budget, 128)
                reasoning_budget = derived_budget

        if request_reasoning_budget is not None:
            parser_chat_template_kwargs["reasoning_budget"] = request_reasoning_budget
        if reasoning_budget is None:
            return

        try:
            budget_int = int(reasoning_budget)
        except Exception:
            logger.warning("Invalid reasoning_budget=%r; skipping", reasoning_budget)
            return

        if budget_int == -1:
            return

        request_grace = getattr(request, "reasoning_budget_grace_period", None)
        grace = request_grace
        if grace is None:
            grace = chat_template_kwargs.get("reasoning_budget_grace_period", 0)

        if request_grace is not None:
            parser_chat_template_kwargs["reasoning_budget_grace_period"] = request_grace
        if sampling_params.extra_args is None:
            sampling_params.extra_args = {}
        extra = sampling_params.extra_args

        extra.setdefault("reasoning_budget", budget_int)

        try:
            grace_int = int(grace)
        except Exception:
            grace_int = 0
        extra.setdefault("reasoning_budget_grace_period", grace_int)

        if "enable_thinking" in parser_chat_template_kwargs:
            extra.setdefault(
                "enable_thinking", parser_chat_template_kwargs["enable_thinking"]
            )

        end_token_ids = getattr(reasoning_parser, "end_token_ids", None)
        parsed_end_token_ids: list[int] = []
        if isinstance(end_token_ids, list):
            try:
                parsed_end_token_ids = [int(tid) for tid in end_token_ids]
            except Exception:
                parsed_end_token_ids = []

        if not parsed_end_token_ids:
            end_token_id = getattr(reasoning_parser, "end_token_id", None)
            if end_token_id is not None:
                try:
                    parsed_end_token_ids = [int(end_token_id)]
                except Exception:
                    parsed_end_token_ids = []

        if not parsed_end_token_ids:
            end_token = getattr(reasoning_parser, "end_token", None)
            if isinstance(end_token, str) and end_token:
                try:
                    parsed_end_token_ids = [
                        int(tid)
                        for tid in tokenizer.encode(end_token, add_special_tokens=False)
                    ]
                except TypeError:
                    try:
                        parsed_end_token_ids = [
                            int(tid) for tid in tokenizer.encode(end_token)
                        ]
                    except Exception:
                        parsed_end_token_ids = []
                except Exception:
                    parsed_end_token_ids = []
                if not parsed_end_token_ids:
                    vocab = getattr(tokenizer, "get_vocab", lambda: {})()
                    token_id = vocab.get(end_token)
                    if token_id is not None:
                        try:
                            parsed_end_token_ids = [int(token_id)]
                        except Exception:
                            parsed_end_token_ids = []

        if not parsed_end_token_ids:
            logger.warning(
                "Could not determine end-of-think token ids for reasoning budget"
            )
            return

        extra.setdefault("think_end_token_id", parsed_end_token_ids[0])
        extra.setdefault("end_token_ids", parsed_end_token_ids)

        if "newline_token_ids" not in extra:
            try:
                newline_ids = self._compute_newline_token_ids(tokenizer)
            except Exception:
                newline_ids = []
            if newline_ids:
                extra["newline_token_ids"] = newline_ids

    async def render_chat_request(
        self,
        request: ChatCompletionRequest,
        raw_request: Request | None = None,
    ) -> tuple[list[ConversationMessage], list[ProcessorInputs]] | ErrorResponse:
        """
        Validate the model and preprocess a chat completion request.

        Delegates preprocessing logic to OpenAIServingRender, adding the
        engine-aware checks (LoRA model validation, engine health).

        Returns:
            A tuple of (conversation, engine_prompts) on success,
            or an ErrorResponse on failure.
        """
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            logger.error("Error with model %s", error_check_ret)
            return error_check_ret

        # If the engine is dead, raise the engine's DEAD_ERROR.
        # This is required for the streaming case, where we return a
        # success status before we actually start generating text :).
        if self.engine_client.errored:
            raise self.engine_client.dead_error

        if os.getenv("VLLM_LOG_PAYLOADS", "1") == "1":
            headers_obj = None
            try:
                if raw_request is not None:
                    headers_obj = {k: v for k, v in raw_request.headers.items()}
            except Exception:
                headers_obj = None
            try:
                req_dump = request.model_dump()
            except Exception:
                req_dump = None
            rid_hint = self._base_request_id(
                raw_request, getattr(request, "request_id", None)
            )
            try:
                payload_logger.info(
                    "openai.request",
                    extra={
                        "rid": rid_hint or "",
                        "endpoint": self.__class__.__name__,
                        "payload": req_dump,
                        "headers": headers_obj,
                    },
                )
            except Exception:
                pass

        # For gpt-oss (harmony) models, special tokens are part of the
        # protocol framing. By default, OpenAI-compatible requests set
        # `skip_special_tokens=True`, which can strip these markers from
        # the streamed text. If the caller didn't explicitly set this
        # field, default to keeping special tokens for harmony models.
        if self.use_harmony and "skip_special_tokens" not in request.model_fields_set:
            request.skip_special_tokens = False

        return await self.openai_serving_render.render_chat(request)

    async def create_chat_completion(
        self,
        request: ChatCompletionRequest,
        raw_request: Request | None = None,
    ) -> AsyncGenerator[str, None] | ChatCompletionResponse | ErrorResponse:
        """
        Chat Completion API similar to OpenAI's API.

        See https://platform.openai.com/docs/api-reference/chat/create
        for the API specification. This API mimics the OpenAI
        Chat Completion API.
        """
        # Streaming response
        tokenizer = self.renderer.tokenizer
        assert tokenizer is not None
        reasoning_parser: ReasoningParser | None = None
        try:
            if self.reasoning_parser_cls:
                # Pass the same chat template kwargs as used in tokenization
                chat_template_kwargs = self._prepare_extra_chat_template_kwargs(
                    request.get_resolved_chat_template_kwargs(),
                    self.default_chat_template_kwargs,
                )
                reasoning_parser = self.reasoning_parser_cls(
                    tokenizer,
                    chat_template_kwargs=chat_template_kwargs,  # type: ignore[call-arg]
                )
        except RuntimeError as e:
            logger.exception("Error in reasoning parser creation.")
            return self.create_error_response(str(e))
        result = await self.render_chat_request(request, raw_request=raw_request)
        if isinstance(result, ErrorResponse):
            return result

        classify_chat_request(request)
        conversation, engine_prompts = result

        request_id = (
            f"chatcmpl-{self._base_request_id(raw_request, request.request_id)}"
        )

        request_metadata = RequestResponseMetadata(request_id=request_id)
        if raw_request:
            raw_request.state.request_metadata = request_metadata

        lora_request = self._maybe_get_adapters(request, supports_default_mm_loras=True)

        model_name = self.models.model_name(lora_request)

        # Extract data_parallel_rank from header (router can inject it)
        data_parallel_rank = self._get_data_parallel_rank(raw_request)
        tokenizer = self.renderer.tokenizer

        # Schedule the request and get the result generator.
        max_model_len = self.model_config.max_model_len
        generators: list[AsyncGenerator[RequestOutput, None]] = []
        for i, engine_prompt in enumerate(engine_prompts):
            prompt_token_ids = self._extract_prompt_components(engine_prompt).token_ids

            # If we are creating sub requests for multiple prompts, ensure that they
            # have unique request ids.
            sub_request_id = (
                request_id if len(engine_prompts) == 1 else f"{request_id}_{i}"
            )

            max_tokens = get_max_tokens(
                max_model_len,
                request.max_completion_tokens
                if request.max_completion_tokens is not None
                else request.max_tokens,
                self._extract_prompt_len(engine_prompt),
                self.default_sampling_params,
                self.override_max_tokens,
            )

            sampling_params: SamplingParams | BeamSearchParams
            if request.use_beam_search:
                sampling_params = request.to_beam_search_params(
                    max_tokens, self.default_sampling_params
                )
            else:
                sampling_params = request.to_sampling_params(
                    max_tokens,
                    self.default_sampling_params,
                )

            if not request.use_beam_search:
                self._inject_think_end_token_id(
                    sampling_params=sampling_params,
                    request=request,
                    tokenizer=tokenizer,
                    reasoning_parser=reasoning_parser,
                )

            self._log_inputs(
                sub_request_id,
                engine_prompt,
                params=sampling_params,
                lora_request=lora_request,
            )

            trace_headers = (
                None
                if raw_request is None
                else await self._get_trace_headers(raw_request.headers)
            )

            if isinstance(sampling_params, BeamSearchParams):
                generator = self.beam_search(
                    prompt=engine_prompt,
                    request_id=sub_request_id,
                    params=sampling_params,
                    lora_request=lora_request,
                    trace_headers=trace_headers,
                )
            else:
                reasoning_ended = (
                    reasoning_parser.is_reasoning_end(prompt_token_ids or [])
                    if reasoning_parser
                    else None
                )

                generator = self.engine_client.generate(
                    engine_prompt,
                    sampling_params,
                    sub_request_id,
                    lora_request=lora_request,
                    trace_headers=trace_headers,
                    priority=request.priority,
                    data_parallel_rank=data_parallel_rank,
                    reasoning_ended=reasoning_ended,
                )

            generators.append(generator)

        assert len(generators) == 1
        (result_generator,) = generators

        if request.stream and (
            request.tool_choice == "required"
            or isinstance(request.tool_choice, ChatCompletionNamedToolChoiceParam)
        ):
            try:
                full_response = await self.chat_completion_full_generator(
                    request,
                    result_generator,
                    request_id,
                    model_name,
                    conversation,
                    tokenizer,
                    request_metadata,
                    reasoning_parser,
                )
            except GenerationError as e:
                return self._convert_generation_error_to_response(e)
            except ValueError as e:
                return self.create_error_response(e)

            if isinstance(full_response, ErrorResponse):
                return full_response

            return self.chat_completion_full_response_to_stream(full_response)

        if request.stream:
            return self.chat_completion_stream_generator(
                request,
                result_generator,
                request_id,
                model_name,
                conversation,
                tokenizer,
                request_metadata,
                reasoning_parser,
            )

        return await self.chat_completion_full_generator(
            request,
            result_generator,
            request_id,
            model_name,
            conversation,
            tokenizer,
            request_metadata,
            reasoning_parser,
        )

    def get_chat_request_role(self, request: ChatCompletionRequest) -> str:
        if request.add_generation_prompt:
            return self.response_role
        return request.messages[-1]["role"]

    async def chat_completion_full_response_to_stream(
        self,
        response: ChatCompletionResponse,
    ) -> AsyncGenerator[str, None]:
        first_choice = response.choices[0]
        first_delta = ChatCompletionStreamResponse(
            id=response.id,
            object="chat.completion.chunk",
            created=response.created,
            model=response.model,
            choices=[
                ChatCompletionResponseStreamChoice(
                    index=first_choice.index,
                    delta=DeltaMessage(role="assistant", content=""),
                    logprobs=None,
                    finish_reason=None,
                )
            ],
        )
        yield f"data: {first_delta.model_dump_json(exclude_unset=True)}\n\n"

        message = first_choice.message
        if message.reasoning is not None or message.reasoning_content is not None:
            reasoning_delta = ChatCompletionStreamResponse(
                id=response.id,
                object="chat.completion.chunk",
                created=response.created,
                model=response.model,
                choices=[
                    ChatCompletionResponseStreamChoice(
                        index=first_choice.index,
                        delta=DeltaMessage(
                            reasoning=message.reasoning,
                            reasoning_content=message.reasoning_content,
                            content=None,
                        ),
                        logprobs=None,
                        finish_reason=None,
                    )
                ],
            )
            yield f"data: {reasoning_delta.model_dump_json(exclude_unset=True)}\n\n"

        if message.content:
            content_delta = ChatCompletionStreamResponse(
                id=response.id,
                object="chat.completion.chunk",
                created=response.created,
                model=response.model,
                choices=[
                    ChatCompletionResponseStreamChoice(
                        index=first_choice.index,
                        delta=DeltaMessage(content=message.content),
                        logprobs=None,
                        finish_reason=None,
                    )
                ],
            )
            yield f"data: {content_delta.model_dump_json(exclude_unset=True)}\n\n"

        if message.tool_calls:
            delta_tool_calls = [
                DeltaToolCall(
                    index=idx,
                    id=tool_call.id,
                    type=tool_call.type,
                    function=DeltaFunctionCall(
                        name=tool_call.function.name,
                        arguments=tool_call.function.arguments,
                    ),
                )
                for idx, tool_call in enumerate(message.tool_calls)
            ]
            tools_delta = ChatCompletionStreamResponse(
                id=response.id,
                object="chat.completion.chunk",
                created=response.created,
                model=response.model,
                choices=[
                    ChatCompletionResponseStreamChoice(
                        index=first_choice.index,
                        delta=DeltaMessage(tool_calls=delta_tool_calls),
                        logprobs=None,
                        finish_reason=None,
                    )
                ],
            )
            yield f"data: {tools_delta.model_dump_json(exclude_unset=True)}\n\n"

        finish_delta = ChatCompletionStreamResponse(
            id=response.id,
            object="chat.completion.chunk",
            created=response.created,
            model=response.model,
            choices=[
                ChatCompletionResponseStreamChoice(
                    index=first_choice.index,
                    delta=DeltaMessage(),
                    logprobs=None,
                    finish_reason=first_choice.finish_reason,
                    stop_reason=first_choice.stop_reason,
                    token_ids=first_choice.token_ids,
                )
            ],
        )
        yield f"data: {finish_delta.model_dump_json(exclude_unset=True)}\n\n"

        usage_payload = {
            "id": response.id,
            "object": "chat.completion.chunk",
            "created": response.created,
            "model": response.model,
            "choices": [],
            "usage": response.usage.model_dump(exclude_none=True),
        }
        yield f"data: {json.dumps(usage_payload, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    @staticmethod
    def _bracket_level(s: str, opening="{", closing="}") -> int:
        """
        Calculate the current level of nested brackets in a given string.
        """
        level = 0
        for char in s:
            if char == opening:
                level += 1
            elif char == closing:
                level -= 1
        return level

    @staticmethod
    def _filter_delta_text(delta_text: str, previous_text: str) -> tuple[str, bool]:
        # remove last '},' of the tool definition stemming from the
        # "name"/"parameters" outer object or closing ']' of the tool list
        # count occurrences of opening and closing curly braces and
        # once level 0 is reached stop outputting text
        # if 0 is reached while parsing the delta_text we know the current
        # tool will finish in this current iteration
        bracket_level = OpenAIServingChat._bracket_level(previous_text)
        updated_delta, passed_zero = "", False
        for c in delta_text:
            if c == "{":
                bracket_level += 1
                passed_zero = bracket_level == 0
            elif c == "}":
                bracket_level -= 1
                passed_zero = bracket_level == 0

            if bracket_level != 0:
                updated_delta += c
            else:
                # if a comma is reached at level 0 we can stop
                if c == ",":
                    break
        return updated_delta, passed_zero

    @staticmethod
    def _extract_nth_parameters_obj(text: str, n: int) -> str | None:
        """Best-effort extraction of the n-th `"parameters": {...}` object."""
        if n < 0:
            return None

        needle = "\"parameters\""
        i = 0
        in_str = False
        esc = False
        found = -1
        key_pos = None
        while i < len(text):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == "\"":
                    in_str = False
                i += 1
                continue

            if c == "\"":
                if text.startswith(needle, i):
                    j = i + len(needle)
                    while j < len(text) and text[j].isspace():
                        j += 1
                    if j < len(text) and text[j] == ":":
                        found += 1
                        if found == n:
                            key_pos = j + 1
                            break
                    i = j
                else:
                    in_str = True
                i += 1
                continue

            i += 1

        if key_pos is None:
            return None

        j = key_pos
        while j < len(text) and text[j].isspace():
            j += 1
        while j < len(text) and text[j] != "{":
            if not text[j].isspace():
                return None
            j += 1
        if j >= len(text) or text[j] != "{":
            return None

        out: list[str] = []
        depth = 0
        in_str = False
        esc = False
        k = j
        while k < len(text):
            c = text[k]
            out.append(c)
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == "\"":
                    in_str = False
            else:
                if c == "\"":
                    in_str = True
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        return "".join(out)
            k += 1

        return "".join(out)

    @staticmethod
    def _is_complete_json_object(text: str | None) -> bool:
        if not text:
            return False
        try:
            parsed = json.loads(text)
        except Exception:
            return False
        return isinstance(parsed, dict)

    @staticmethod
    def _tool_allows_empty_arguments(
        request: ChatCompletionRequest,
        tool_name: str | None,
    ) -> bool:
        if not tool_name or not request.tools:
            return False
        for tool in request.tools:
            function = getattr(tool, "function", None)
            if function is None or getattr(function, "name", None) != tool_name:
                continue
            parameters = getattr(function, "parameters", None)
            if not isinstance(parameters, dict):
                return False
            properties = parameters.get("properties")
            required = parameters.get("required")
            return not properties and not required
        return False

    @staticmethod
    def _force_nonempty_content_enabled(request: ChatCompletionRequest) -> bool:
        kwargs = request.get_resolved_chat_template_kwargs()
        return kwargs.get("force_nonempty_content", False) is True

    def _build_force_nonempty_stream_finish_delta(
        self,
        *,
        request: ChatCompletionRequest,
        tokenizer: TokenizerLike | None,
        request_id: str,
        history_tool_call_cnt: int,
        current_text: str,
        current_content: str,
        reasoning_parser: ReasoningParser | None,
        tool_choice_auto: bool,
        previous_tool_calls: list[dict[str, Any]],
    ) -> tuple[DeltaMessage | None, str | None, list[dict[str, Any]] | None]:
        if not self._force_nonempty_content_enabled(request):
            return None, None, None
        if current_content:
            return None, None, None
        if not current_text:
            return None, None, None

        if request.tools and tool_choice_auto and self.enable_auto_tools and self.tool_parser:
            parsed_calls, parsed_content = self._parse_tool_calls_from_content(
                request=request,
                tokenizer=tokenizer,
                content=current_text,
                enable_auto_tools=self.enable_auto_tools,
                tool_parser_cls=self.tool_parser,
            )
            if parsed_calls:
                tool_calls: list[DeltaToolCall] = []
                updated_tool_states = list(previous_tool_calls)
                for idx, parsed_call in enumerate(parsed_calls):
                    tool_call_id = make_tool_call_id(
                        id_type=self.tool_call_id_type,
                        request_id=request_id,
                        idx=history_tool_call_cnt + idx,
                    )
                    tool_calls.append(
                        DeltaToolCall(
                            index=idx,
                            id=tool_call_id,
                            type="function",
                            function=DeltaFunctionCall(
                                name=parsed_call.name,
                                arguments=parsed_call.arguments,
                            ),
                        )
                    )
                    updated_tool_states.append(
                        {
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "name": parsed_call.name,
                                "arguments": parsed_call.arguments,
                            },
                        }
                    )
                return (
                    DeltaMessage(
                        content=parsed_content if parsed_content else None,
                        tool_calls=tool_calls,
                    ),
                    "tool_calls",
                    updated_tool_states,
                )

        return DeltaMessage(content=current_text), "stop", None

    def extract_tool_call_required_streaming(
        self,
        previous_text: str,
        current_text: str | None,
        delta_text: str,
        function_name_returned: bool,
        tool_call_idx: int | None = None,
    ) -> tuple[DeltaMessage | None, bool]:
        if current_text is None or current_text == "":
            # if the current text is empty, we cannot parse it
            return None, function_name_returned
        try:
            flags = Allow.ALL
            obj, _ = partial_json_loads(current_text, flags)
        except (
            partial_json_parser.core.exceptions.MalformedJSON,
            json.JSONDecodeError,
        ):
            logger.debug("not enough tokens to parse into JSON yet")
            obj = None

        prev_obj = None
        if previous_text:
            try:
                flags = Allow.ALL
                prev_obj, _ = partial_json_loads(previous_text, flags)
            except (
                partial_json_parser.core.exceptions.MalformedJSON,
                json.JSONDecodeError,
            ):
                prev_obj = None

        if obj is None or not isinstance(obj, list) or not len(obj) > 0:
            return None, False

        previous_len = len(prev_obj) if isinstance(prev_obj, list) else 0
        delta_tool_calls: list[DeltaToolCall] = []

        for idx, current_tool_call in enumerate(obj):
            if not isinstance(current_tool_call, dict):
                continue

            tool_name = current_tool_call.get("name")
            if not isinstance(tool_name, str) or not tool_name:
                continue

            curr_args = OpenAIServingChat._extract_nth_parameters_obj(current_text, idx)
            if not OpenAIServingChat._is_complete_json_object(curr_args):
                continue

            prev_args = OpenAIServingChat._extract_nth_parameters_obj(previous_text, idx)
            prev_complete = OpenAIServingChat._is_complete_json_object(prev_args)

            if idx >= previous_len or not prev_complete:
                tool_call_id = make_tool_call_id(
                    id_type=self.tool_call_id_type,
                    func_name=tool_name,
                    idx=(
                        tool_call_idx + len(delta_tool_calls)
                        if tool_call_idx is not None
                        else None
                    ),
                )
                delta_tool_calls.append(
                    DeltaToolCall(
                        id=tool_call_id,
                        function=DeltaFunctionCall(
                            name=tool_name,
                            arguments=curr_args,
                        ),
                        index=idx,
                        type="function",
                    )
                )
                continue

            if prev_args and curr_args.startswith(prev_args):
                arguments_delta = curr_args[len(prev_args) :]
            elif prev_args == curr_args:
                arguments_delta = ""
            else:
                arguments_delta = curr_args

            if arguments_delta:
                delta_tool_calls.append(
                    DeltaToolCall(
                        function=DeltaFunctionCall(
                            name=None,
                            arguments=arguments_delta,
                        ),
                        index=idx,
                    )
                )

        if not delta_tool_calls:
            return None, False

        return DeltaMessage(tool_calls=delta_tool_calls), True

    @staticmethod
    def _looks_like_tool_parser_markup(*texts: str | None) -> bool:
        for text in texts:
            if text and ("<tool_call>" in text or "<function=" in text):
                return True
        return False

    def _extract_tool_delta_via_parser(
        self,
        request: ChatCompletionRequest,
        tool_parser: ToolParser | None,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        *,
        required_name: str | None = None,
    ) -> DeltaMessage | None:
        if tool_parser is None or not request.tools:
            return None
        delta_message = tool_parser.extract_tool_calls_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
            request,
        )
        if delta_message is None or not delta_message.tool_calls:
            return None

        filtered_tool_calls: list[DeltaToolCall] = []
        for tool_call in delta_message.tool_calls:
            fn = tool_call.function
            if required_name is not None and fn and fn.name not in (None, required_name):
                continue
            filtered_tool_calls.append(tool_call)

        if not filtered_tool_calls:
            return None

        return DeltaMessage(tool_calls=filtered_tool_calls)

    async def chat_completion_stream_generator(
        self,
        request: ChatCompletionRequest,
        result_generator: AsyncIterator[RequestOutput],
        request_id: str,
        model_name: str,
        conversation: list[ConversationMessage],
        tokenizer: TokenizerLike,
        request_metadata: RequestResponseMetadata,
        reasoning_parser: ReasoningParser | None = None,
    ) -> AsyncGenerator[str, None]:
        created_time = int(time.time())
        chunk_object_type: Final = "chat.completion.chunk"
        first_iteration = True

        # Send response for each token for each request.n (index)
        num_choices = 1 if request.n is None else request.n
        previous_num_tokens = [0] * num_choices
        finish_reason_sent = [False] * num_choices
        num_prompt_tokens = 0
        num_cached_tokens = None
        if self.use_harmony:
            harmony_parsers = [
                get_streamable_parser_for_assistant() for _ in range(num_choices)
            ]
            harmony_tools_streamed = [False] * num_choices
        tools_streamed = [False] * num_choices

        if isinstance(request.tool_choice, ChatCompletionNamedToolChoiceParam):
            tool_choice_function_name = request.tool_choice.function.name
        else:
            tool_choice_function_name = None

        # Determine whether tools are in use with "auto" tool choice
        tool_choice_auto = (
            not tool_choice_function_name
            and self._should_stream_with_auto_tool_parsing(request)
        )

        all_previous_token_ids: list[list[int]] | None
        function_name_returned = [False] * num_choices
        named_tool_previous_args = [""] * num_choices
        if self.tool_call_id_type == "kimi_k2":
            history_tool_call_cnt = get_history_tool_calls_cnt(conversation)
        else:
            history_tool_call_cnt = 0

        # Always track previous text and structured deltas for output logging.
        previous_texts = [""] * num_choices
        previous_reasoning_texts = [""] * num_choices
        previous_content_texts = [""] * num_choices
        previous_tool_calls: list[list[dict[str, Any]]] = [
            [] for _ in range(num_choices)
        ]

        # Only one of these will be used, thus previous_texts and
        # all_previous_token_ids will not be used twice in the same iteration.
        if tool_choice_auto or reasoning_parser:
            # These are only required in "auto" tool choice case
            all_previous_token_ids = [[]] * num_choices
            # For reasoning parser and tool call all enabled
            added_content_delta_arr = [False] * num_choices
            reasoning_end_arr = [False] * num_choices
            prompt_is_reasoning_end_arr: list[bool | None] = [None] * num_choices
        else:
            all_previous_token_ids = None

        # Prepare the tool parser only for auto tool choice.
        # Named/required must stay on grammar-constrained decoding paths.
        try:
            if request.tools and self.tool_parser and tool_choice_auto:
                if tokenizer is None:
                    raise ValueError(
                        "Tokenizer not available when `skip_tokenizer_init=True`"
                    )

                tool_parsers: list[ToolParser | None] = [
                    self.tool_parser(tokenizer)
                ] * num_choices
            else:
                tool_parsers = [None] * num_choices
        except Exception as e:
            logger.exception("Error in tool parser creation.")
            data = self.create_streaming_error_response(e)
            yield f"data: {data}\n\n"
            yield "data: [DONE]\n\n"
            return

        stream_options = request.stream_options
        include_usage, include_continuous_usage = should_include_usage(
            stream_options, self.enable_force_include_usage
        )

        try:
            async for res in result_generator:
                if res.prompt_token_ids is not None:
                    num_prompt_tokens = len(res.prompt_token_ids)
                    if res.encoder_prompt_token_ids is not None:
                        num_prompt_tokens += len(res.encoder_prompt_token_ids)

                # We need to do it here, because if there are exceptions in
                # the result_generator, it needs to be sent as the FIRST
                # response (by the try...catch).
                if first_iteration:
                    num_cached_tokens = res.num_cached_tokens
                    # Send first response for each request.n (index) with
                    # the role
                    role = self.get_chat_request_role(request)

                    # NOTE num_choices defaults to 1 so this usually executes
                    # once per request
                    for i in range(num_choices):
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=DeltaMessage(
                                role=role,
                                content="",
                            ),
                            logprobs=None,
                            finish_reason=None,
                        )

                        # return prompt_token_ids at the first chunk ever
                        chunk = ChatCompletionStreamResponse(
                            id=request_id,
                            object=chunk_object_type,
                            created=created_time,
                            choices=[choice_data],
                            model=model_name,
                            prompt_token_ids=(
                                res.prompt_token_ids
                                if request.return_token_ids
                                else None
                            ),
                        )

                        # if continuous usage stats are requested, add it
                        if include_continuous_usage:
                            chunk.usage = UsageInfo(
                                prompt_tokens=num_prompt_tokens,
                                completion_tokens=0,
                                total_tokens=num_prompt_tokens,
                            )

                        data = chunk.model_dump_json(exclude_unset=True)
                        yield f"data: {data}\n\n"

                    # Send response to echo the input portion of the
                    # last message
                    if request.echo:
                        last_msg_content: str | list[dict[str, str]] = ""
                        if (
                            conversation
                            and "content" in conversation[-1]
                            and conversation[-1].get("role") == role
                        ):
                            last_msg_content = conversation[-1]["content"] or ""

                        if last_msg_content:
                            for i in range(num_choices):
                                choice_data = ChatCompletionResponseStreamChoice(
                                    index=i,
                                    delta=DeltaMessage(content=last_msg_content),
                                    logprobs=None,
                                    finish_reason=None,
                                )
                                chunk = ChatCompletionStreamResponse(
                                    id=request_id,
                                    object=chunk_object_type,
                                    created=created_time,
                                    choices=[choice_data],
                                    model=model_name,
                                )
                                if include_continuous_usage:
                                    chunk.usage = UsageInfo(
                                        prompt_tokens=num_prompt_tokens,
                                        completion_tokens=0,
                                        total_tokens=num_prompt_tokens,
                                    )

                                data = chunk.model_dump_json(exclude_unset=True)
                                yield f"data: {data}\n\n"
                    first_iteration = False

                for output in res.outputs:
                    i = output.index
                    tool_parser = tool_parsers[i]

                    if (
                        reasoning_parser
                        and res.prompt_token_ids
                        and prompt_is_reasoning_end_arr[i] is None
                    ):
                        # only check once per choice, because prompt_token_ids
                        # are the same for all deltas in that choice
                        prompt_is_reasoning_end_arr[i] = (
                            reasoning_parser.is_reasoning_end(res.prompt_token_ids)
                        )
                    if finish_reason_sent[i]:
                        continue

                    if request.logprobs and request.top_logprobs is not None:
                        assert output.logprobs is not None, "Did not output logprobs"
                        logprobs = self._create_chat_logprobs(
                            token_ids=output.token_ids,
                            top_logprobs=output.logprobs,
                            tokenizer=tokenizer,
                            num_output_top_logprobs=request.top_logprobs,
                            return_as_token_id=request.return_tokens_as_token_ids,
                        )
                    else:
                        logprobs = None

                    if self.use_harmony:
                        harmony_parser = harmony_parsers[i]
                        prev_recipient = harmony_parser.current_recipient

                        # Track accumulated content per token with their state
                        token_states: list[TokenState] = []
                        for token_id in output.token_ids:
                            harmony_parser.process(token_id)
                            token_delta = harmony_parser.last_content_delta or ""
                            token_states.append(
                                TokenState(
                                    harmony_parser.current_channel,
                                    harmony_parser.current_recipient,
                                    token_delta,
                                )
                            )
                        delta_text = "".join(delta for _, _, delta in token_states)
                        cur_channel = harmony_parser.current_channel

                        # handle the case where several tokens where generated at once
                        # including the final token, leading to a delta in the text
                        # but the current channel to be empty (start state)
                        if not cur_channel and delta_text:
                            cur_channel = "final"
                    else:
                        delta_text = output.text

                    if (
                        not delta_text
                        and not output.token_ids
                        and not previous_num_tokens[i]
                    ):
                        # Chunked prefill case, don't return empty chunks
                        continue

                    delta_message: DeltaMessage | None

                    # just update previous_texts and previous_token_ids
                    if tool_choice_auto or reasoning_parser:
                        assert previous_texts is not None
                        assert all_previous_token_ids is not None
                        previous_text = previous_texts[i]
                        previous_token_ids = all_previous_token_ids[i]
                        current_text = previous_text + delta_text
                        # avoid the None + list error.
                        if previous_token_ids:
                            current_token_ids = previous_token_ids + as_list(
                                output.token_ids
                            )
                        else:
                            current_token_ids = as_list(output.token_ids)

                    if self.use_harmony:
                        delta_message, tools_streamed_flag = (
                            extract_harmony_streaming_delta(
                                harmony_parser=harmony_parser,
                                token_states=token_states,
                                prev_recipient=prev_recipient,
                                include_reasoning=request.include_reasoning,
                            )
                        )
                        harmony_tools_streamed[i] |= tools_streamed_flag
                    # handle streaming deltas for tools with named tool_choice
                    elif tool_choice_function_name:
                        # When encountering think end id in prompt_token_ids
                        # i.e {"enable_thinking": False},
                        # check BEFORE calling the parser to avoid a spurious
                        # reasoning delta on the first chunk.
                        if (
                            reasoning_parser
                            and not reasoning_end_arr[i]
                            and prompt_is_reasoning_end_arr[i]
                        ):
                            reasoning_end_arr[i] = True

                        if (
                            reasoning_parser
                            and not reasoning_end_arr[i]
                            and not reasoning_parser.is_reasoning_end(
                                previous_token_ids
                            )
                        ):
                            assert reasoning_parser is not None
                            delta_message = (
                                reasoning_parser.extract_reasoning_streaming(
                                    previous_text,
                                    current_text,
                                    delta_text,
                                    previous_token_ids,
                                    current_token_ids,
                                    output.token_ids,
                                )
                            )
                            # When encountering think end id in delta_token_ids,
                            # set reasoning status to end.
                            # Only keep 'content', remove 'reasoning'.
                            if reasoning_parser.is_reasoning_end(
                                as_list(output.token_ids)
                            ):
                                reasoning_end_arr[i] = True
                                if delta_message and delta_message.content:
                                    # This need to be added to next `delta_text`
                                    current_text = delta_message.content
                                    delta_message.content = None
                                elif (
                                    delta_message
                                    and delta_message.reasoning
                                    and delta_message.reasoning.lstrip().startswith(
                                        ("{", "[")
                                    )
                                ):
                                    # Some models emit raw JSON in reasoning channel
                                    # (without explicit </think>). Preserve it for
                                    # tool-argument parsing instead of dropping it.
                                    current_text = delta_message.reasoning
                                    delta_message.reasoning = None
                                    delta_message.reasoning_content = None
                                else:
                                    current_text = ""
                            elif output.finish_reason is not None:
                                # Fallback: thinking was enabled but model never emitted
                                # an explicit reasoning-end marker. Parse final accumulated
                                # content and stream only the tool-argument delta.
                                accumulated_text = previous_text + delta_text
                                current_text = accumulated_text
                                parsed_content = accumulated_text
                                try:
                                    _, extracted_content = reasoning_parser.extract_reasoning(
                                        accumulated_text,
                                        request=self._get_reasoning_parser_request_view(
                                            request
                                        ),
                                    )
                                    if extracted_content is not None:
                                        parsed_content = extracted_content
                                except Exception:
                                    parsed_content = accumulated_text
                                try:
                                    parsed_calls, _ = self._parse_tool_calls_from_content(
                                        request=request,
                                        tokenizer=tokenizer,
                                        content=parsed_content,
                                        enable_auto_tools=self.enable_auto_tools,
                                        tool_parser_cls=self.tool_parser,
                                    )
                                except Exception:
                                    parsed_calls = None

                                matched_call = None
                                if parsed_calls:
                                    matched_call = next(
                                        (
                                            fc
                                            for fc in parsed_calls
                                            if fc.name == tool_choice_function_name
                                        ),
                                        parsed_calls[0],
                                    )
                                extracted_args = (
                                    matched_call.arguments
                                    if matched_call
                                    and matched_call.arguments is not None
                                    else accumulated_text.strip()
                                )
                                if extracted_args:
                                    try:
                                        json.loads(extracted_args)
                                    except json.JSONDecodeError:
                                        extracted_args = ""

                                previous_args = named_tool_previous_args[i]
                                if extracted_args.startswith(previous_args):
                                    arguments_delta = extracted_args[len(previous_args) :]
                                else:
                                    arguments_delta = extracted_args
                                named_tool_previous_args[i] = extracted_args

                                if arguments_delta or not function_name_returned[i]:
                                    if function_name_returned[i]:
                                        delta_tool_call = DeltaToolCall(
                                            function=DeltaFunctionCall(
                                                arguments=arguments_delta
                                            ),
                                            index=i,
                                        )
                                    else:
                                        delta_tool_call = DeltaToolCall(
                                            id=make_tool_call_id(),
                                            type="function",
                                            function=DeltaFunctionCall(
                                                name=tool_choice_function_name,
                                                arguments=arguments_delta,
                                            ),
                                            index=i,
                                        )
                                        function_name_returned[i] = True
                                    delta_message = DeltaMessage(
                                        tool_calls=[delta_tool_call]
                                    )
                                    tools_streamed[i] = True
                                else:
                                    delta_message = None
                                reasoning_end_arr[i] = True
                        else:
                            if output.finish_reason is None:
                                delta_message = None
                            else:
                                parsed_calls, _ = self._parse_tool_calls_from_content(
                                    request=request,
                                    tokenizer=tokenizer,
                                    content=current_text,
                                    enable_auto_tools=self.enable_auto_tools,
                                    tool_parser_cls=self.tool_parser,
                                )

                                matched_call = None
                                if parsed_calls:
                                    matched_call = next(
                                        (
                                            fc
                                            for fc in parsed_calls
                                            if fc.name == tool_choice_function_name
                                        ),
                                        parsed_calls[0],
                                    )

                                if matched_call is None:
                                    delta_message = None
                                else:
                                    tool_call_id = (
                                        MistralToolCall.generate_random_id()
                                        if is_mistral_tokenizer(tokenizer)
                                        else make_tool_call_id(
                                            id_type=self.tool_call_id_type,
                                            func_name=tool_choice_function_name,
                                            idx=history_tool_call_cnt,
                                        )
                                    )
                                    delta_message = DeltaMessage(
                                        tool_calls=[
                                            DeltaToolCall(
                                                id=tool_call_id,
                                                type="function",
                                                function=DeltaFunctionCall(
                                                    name=tool_choice_function_name,
                                                    arguments=matched_call.arguments,
                                                ),
                                                index=i,
                                            )
                                        ]
                                    )
                                    function_name_returned[i] = True
                                    tools_streamed[i] = True
                                    history_tool_call_cnt += 1

                    elif request.tool_choice == "required":
                        assert previous_texts is not None
                        previous_text = previous_texts[i]
                        current_text = previous_text + delta_text
                        fn_name_returned = function_name_returned[i]
                        output_token_ids = as_list(output.token_ids)

                        if (
                            reasoning_parser is not None
                            and not reasoning_end_arr[i]
                            and prompt_is_reasoning_end_arr[i]
                        ):
                            reasoning_end_arr[i] = True

                        if reasoning_parser and not reasoning_end_arr[i]:
                            delta_message = (
                                reasoning_parser.extract_reasoning_streaming(
                                    previous_text,
                                    current_text,
                                    delta_text,
                                    previous_token_ids,
                                    current_token_ids,
                                    output_token_ids,
                                )
                            )
                            if reasoning_parser.is_reasoning_end(output_token_ids):
                                reasoning_end_arr[i] = True
                                if delta_message and delta_message.content:
                                    current_text = delta_message.content
                                    delta_message.content = None
                                else:
                                    # reasoning ended
                                    current_text = ""
                            elif output.finish_reason is not None:
                                accumulated_text = previous_text + delta_text
                                current_text = accumulated_text
                                parsed_content = accumulated_text
                                try:
                                    _, extracted_content = reasoning_parser.extract_reasoning(
                                        accumulated_text,
                                        request=self._get_reasoning_parser_request_view(
                                            request
                                        ),
                                    )
                                    if extracted_content is not None:
                                        parsed_content = extracted_content
                                except Exception:
                                    parsed_content = accumulated_text
                                try:
                                    parsed_calls, _ = self._parse_tool_calls_from_content(
                                        request=request,
                                        tokenizer=tokenizer,
                                        content=parsed_content,
                                        enable_auto_tools=self.enable_auto_tools,
                                        tool_parser_cls=self.tool_parser,
                                    )
                                except Exception:
                                    parsed_calls = None

                                if parsed_calls:
                                    delta_tool_calls: list[DeltaToolCall] = []
                                    for idx, parsed_call in enumerate(parsed_calls):
                                        delta_tool_calls.append(
                                            DeltaToolCall(
                                                id=make_tool_call_id(
                                                    id_type=self.tool_call_id_type,
                                                    func_name=parsed_call.name,
                                                    idx=history_tool_call_cnt + idx,
                                                ),
                                                type="function",
                                                function=DeltaFunctionCall(
                                                    name=parsed_call.name,
                                                    arguments=parsed_call.arguments,
                                                ),
                                                index=idx,
                                            )
                                        )
                                    delta_message = DeltaMessage(
                                        tool_calls=delta_tool_calls
                                    )
                                    function_name_returned[i] = True
                                    reasoning_end_arr[i] = True
                                else:
                                    delta_message = None

                        else:
                            # either finished reasoning or no reasoning at all
                            content = current_text
                            if output.finish_reason is None:
                                delta_message = None
                            else:
                                parsed_calls, _ = self._parse_tool_calls_from_content(
                                    request=request,
                                    tokenizer=tokenizer,
                                    content=content,
                                    enable_auto_tools=self.enable_auto_tools,
                                    tool_parser_cls=self.tool_parser,
                                )
                                if parsed_calls:
                                    delta_tool_calls: list[DeltaToolCall] = []
                                    for idx, parsed_call in enumerate(parsed_calls):
                                        delta_tool_calls.append(
                                            DeltaToolCall(
                                                id=make_tool_call_id(
                                                    id_type=self.tool_call_id_type,
                                                    func_name=parsed_call.name,
                                                    idx=history_tool_call_cnt + idx,
                                                ),
                                                type="function",
                                                function=DeltaFunctionCall(
                                                    name=parsed_call.name,
                                                    arguments=parsed_call.arguments,
                                                ),
                                                index=idx,
                                            )
                                        )
                                    delta_message = DeltaMessage(
                                        tool_calls=delta_tool_calls
                                    )
                                    function_name_returned[i] = True
                                    tools_streamed[i] = True
                                    history_tool_call_cnt += len(parsed_calls)
                                else:
                                    delta_message = None

                    # handle streaming deltas for tools with "auto" tool choice
                    # and reasoning parser
                    elif tool_choice_auto and reasoning_parser:
                        assert tool_parser is not None
                        assert added_content_delta_arr is not None
                        assert reasoning_end_arr is not None
                        output_token_ids = as_list(output.token_ids)
                        reasoning_delta_message: DeltaMessage | None = None
                        if not reasoning_end_arr[i]:
                            # When encountering think end id in prompt_token_ids
                            # i.e {"enable_thinking": False},
                            # set reasoning status to end.
                            if prompt_is_reasoning_end_arr[i]:
                                reasoning_end_arr[i] = True
                                current_token_ids = output_token_ids
                                # Don't update current_text, keep it as is from delta
                            else:
                                reasoning_delta_message = (
                                    reasoning_parser.extract_reasoning_streaming(
                                        previous_text,
                                        current_text,
                                        delta_text,
                                        previous_token_ids,
                                        current_token_ids,
                                        output_token_ids,
                                    )
                                )

                                # When encountering think end id in delta_token_ids,
                                # set reasoning status to end.
                                # Remove the text and token ids related
                                # to 'reasoning'.
                                if reasoning_parser.is_reasoning_end(output_token_ids):
                                    reasoning_end_arr[i] = True
                                    current_token_ids = (
                                        reasoning_parser.extract_content_ids(
                                            output_token_ids
                                        )
                                    )
                                    if (
                                        reasoning_delta_message
                                        and reasoning_delta_message.content
                                    ):
                                        current_text = reasoning_delta_message.content
                                        reasoning_delta_message.content = None
                                    else:
                                        current_text = ""
                                delta_message = reasoning_delta_message

                        # handle tool calls only after reasoning is done,
                        if reasoning_end_arr[i]:
                            delta_token_ids = output_token_ids
                            # First time to tool call,
                            # add the remaining text and token ids
                            # to delta from previous
                            if not added_content_delta_arr[i]:
                                added_content_delta_arr[i] = True
                                previous_text = ""
                                previous_token_ids = []
                                delta_text = current_text
                                delta_token_ids = current_token_ids

                            delta_message = tool_parser.extract_tool_calls_streaming(
                                previous_text=previous_text,
                                current_text=current_text,
                                delta_text=delta_text,
                                previous_token_ids=previous_token_ids,
                                current_token_ids=current_token_ids,
                                delta_token_ids=delta_token_ids,
                                request=request,
                            )
                            delta_message = self._merge_delta_messages(
                                reasoning_delta_message,
                                delta_message,
                            )
                            if delta_message and delta_message.tool_calls:
                                tools_streamed[i] = True
                    # when only tool calls
                    elif tool_choice_auto:
                        assert tool_parser is not None
                        delta_message = tool_parser.extract_tool_calls_streaming(
                            previous_text=previous_text,
                            current_text=current_text,
                            delta_text=delta_text,
                            previous_token_ids=previous_token_ids,
                            current_token_ids=current_token_ids,
                            delta_token_ids=output.token_ids,
                            request=request,
                        )
                        if delta_message and delta_message.tool_calls:
                            tools_streamed[i] = True

                    # when only reasoning
                    elif reasoning_parser:
                        # When encountering think end id in prompt_token_ids
                        # i.e {"enable_thinking": False},
                        # set reasoning status to end.
                        # Route all generated tokens as content directly.
                        if prompt_is_reasoning_end_arr[i]:
                            delta_message = DeltaMessage(content=delta_text)
                        else:
                            delta_message = (
                                reasoning_parser.extract_reasoning_streaming(
                                    previous_text,
                                    current_text,
                                    delta_text,
                                    previous_token_ids,
                                    current_token_ids,
                                    output.token_ids,
                                )
                            )
                    # handle streaming just a content delta
                    else:
                        delta_message = DeltaMessage(content=delta_text)

                    # update the previous values for the next iteration
                    if (tool_choice_auto or reasoning_parser) and not self.use_harmony:
                        assert previous_texts is not None
                        assert all_previous_token_ids is not None
                        previous_texts[i] = current_text
                        all_previous_token_ids[i] = current_token_ids
                    else:
                        # Update for comprehensive logging even in simple case
                        assert previous_texts is not None
                        previous_texts[i] += delta_text

                    if tool_choice_auto and delta_message and delta_message.tool_calls:
                        kept_tool_calls: list[DeltaToolCall] = []
                        for delta_tc in delta_message.tool_calls:
                            tc_idx = delta_tc.index if delta_tc.index is not None else 0
                            fn_name: str | None = None
                            fn_args: str | None = None
                            if isinstance(delta_tc.function, dict):
                                fn_name = delta_tc.function.get("name")
                                fn_args = delta_tc.function.get("arguments")
                            elif delta_tc.function is not None:
                                fn_name = delta_tc.function.name
                                fn_args = delta_tc.function.arguments

                            prior_args = ""
                            if len(previous_tool_calls[i]) > tc_idx:
                                prior_args = str(
                                    (
                                        previous_tool_calls[i][tc_idx].get("function")
                                        or {}
                                    ).get("arguments")
                                    or ""
                                )

                            is_empty_boundary = (
                                (fn_name is None or fn_name == "")
                                and (fn_args is None or fn_args == "")
                            )
                            is_empty_arg_header = (
                                fn_name not in (None, "")
                                and fn_args in (None, "", "{", "{}")
                                and not prior_args
                                and not self._tool_allows_empty_arguments(
                                    request, fn_name
                                )
                            )
                            if is_empty_boundary or is_empty_arg_header:
                                continue
                            kept_tool_calls.append(delta_tc)

                        if kept_tool_calls:
                            delta_message = DeltaMessage(
                                content=delta_message.content,
                                reasoning=delta_message.reasoning,
                                reasoning_content=delta_message.reasoning_content,
                                tool_calls=kept_tool_calls,
                            )
                        elif delta_message.content is None and delta_message.reasoning is None:
                            delta_message = None
                        else:
                            delta_message = DeltaMessage(
                                content=delta_message.content,
                                reasoning=delta_message.reasoning,
                                reasoning_content=delta_message.reasoning_content,
                            )

                    # set the previous values for the next iteration
                    previous_num_tokens[i] += len(output.token_ids)

                    # if the message delta is None (e.g. because it was a
                    # "control token" for tool calls or the parser otherwise
                    # wasn't ready to send a token, then
                    #   get the next token without streaming a chunk
                    if delta_message is None:
                        # NOTE: If return_token_ids is enabled, we still need to
                        # send a chunk with token_ids even if delta_message is None
                        # to ensure all tokens are included in the response
                        if (
                            output.finish_reason is None
                            and not request.return_token_ids
                        ):
                            continue
                        delta_message = DeltaMessage()

                    # Accumulate complete output for a single end-of-stream log.
                    if delta_message is not None:
                        reasoning_delta = (
                            delta_message.reasoning_content
                            if delta_message.reasoning_content is not None
                            else delta_message.reasoning
                        )
                        if reasoning_delta:
                            previous_reasoning_texts[i] += reasoning_delta
                        if delta_message.content:
                            previous_content_texts[i] += delta_message.content
                        if delta_message.tool_calls:
                            for delta_tc in delta_message.tool_calls:
                                tc_idx = (
                                    delta_tc.index
                                    if delta_tc.index is not None
                                    else 0
                                )
                                while len(previous_tool_calls[i]) <= tc_idx:
                                    previous_tool_calls[i].append(
                                        {
                                            "id": None,
                                            "type": "function",
                                            "function": {
                                                "name": None,
                                                "arguments": "",
                                            },
                                        }
                                    )

                                tool_state = previous_tool_calls[i][tc_idx]
                                if delta_tc.id:
                                    tool_state["id"] = delta_tc.id
                                if delta_tc.type:
                                    tool_state["type"] = delta_tc.type

                                fn_name: str | None = None
                                fn_args: str | None = None
                                if isinstance(delta_tc.function, dict):
                                    fn_name = delta_tc.function.get("name")
                                    fn_args = delta_tc.function.get("arguments")
                                elif delta_tc.function is not None:
                                    fn_name = delta_tc.function.name
                                    fn_args = delta_tc.function.arguments

                                function_state = tool_state["function"]
                                if fn_name:
                                    function_state["name"] = fn_name
                                if fn_args:
                                    function_state["arguments"] += fn_args

                    # Log streaming delta if output logging is enabled
                    if self.enable_log_outputs and self.request_logger:
                        delta_content_parts = []
                        if delta_message.content:
                            delta_content_parts.append(delta_message.content)
                        if delta_message.reasoning:
                            reasoning = delta_message.reasoning
                            delta_content_parts.append(f"[reasoning: {reasoning}]")
                        if delta_message.tool_calls:
                            tool_args = "".join(
                                tc.function.arguments
                                for tc in delta_message.tool_calls
                                if tc.function and tc.function.arguments
                            )
                            if tool_args:
                                delta_content_parts.append(f"[tool_calls: {tool_args}]")

                        if delta_content_parts and self.enable_log_deltas:
                            delta_content = " ".join(delta_content_parts)
                            self.request_logger.log_outputs(
                                request_id=request_id,
                                outputs=delta_content,
                                output_token_ids=as_list(output.token_ids),
                                finish_reason=output.finish_reason,
                                is_streaming=True,
                                delta=True,
                            )

                    if output.finish_reason is None:
                        # Send token-by-token response for each request.n
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=delta_message,
                            logprobs=logprobs,
                            finish_reason=None,
                            token_ids=(
                                as_list(output.token_ids)
                                if request.return_token_ids
                                else None
                            ),
                        )

                    # if the model is finished generating
                    else:
                        # check for error finish reason and abort streaming
                        # finish_reason='error' indicates a retryable error
                        self._raise_if_error(output.finish_reason, request_id)

                        # check to make sure we haven't "forgotten" to stream
                        #   any tokens that were generated but previously
                        #   matched by partial json parsing
                        # only happens if we are NOT using structured outputs
                        auto_tools_called = False
                        if tool_parser:
                            auto_tools_called = len(tool_parser.prev_tool_call_arr) > 0
                            index = (
                                len(tool_parser.prev_tool_call_arr) - 1
                                if auto_tools_called
                                else 0
                            )
                        else:
                            index = 0
                        should_check = self._should_check_for_unstreamed_tool_arg_tokens(
                            delta_message, output
                        )
                        reparsed_calls = None
                        if (
                            output.finish_reason is not None
                            and self.enable_auto_tools
                            and self.tool_parser
                            and tool_parser
                            and (auto_tools_called or tools_streamed[i] or should_check)
                        ):
                            final_tool_text = current_text
                            if final_tool_text:
                                try:
                                    reparsed_calls, _ = (
                                        self._parse_tool_calls_from_content(
                                            request=request,
                                            tokenizer=tokenizer,
                                            content=final_tool_text,
                                            enable_auto_tools=self.enable_auto_tools,
                                            tool_parser_cls=self.tool_parser,
                                        )
                                    )
                                except Exception:
                                    reparsed_calls = None

                            if (
                                reparsed_calls
                                and len(reparsed_calls) > len(previous_tool_calls[i])
                            ):
                                missing_tool_calls: list[DeltaToolCall] = []
                                for missing_idx in range(
                                    len(previous_tool_calls[i]), len(reparsed_calls)
                                ):
                                    missing_call = reparsed_calls[missing_idx]
                                    missing_args = missing_call.arguments
                                    if not isinstance(missing_args, str):
                                        continue
                                    if (
                                        missing_args in ("", "{}")
                                        and not self._tool_allows_empty_arguments(
                                            request, missing_call.name
                                        )
                                    ):
                                        continue

                                    tool_call_id = make_tool_call_id(
                                        id_type=self.tool_call_id_type,
                                        request_id=request_id,
                                        idx=history_tool_call_cnt + missing_idx,
                                    )
                                    missing_tool_calls.append(
                                        DeltaToolCall(
                                            index=missing_idx,
                                            id=tool_call_id,
                                            type="function",
                                            function=DeltaFunctionCall(
                                                name=missing_call.name,
                                                arguments=missing_args,
                                            ),
                                        )
                                    )
                                    previous_tool_calls[i].append(
                                        {
                                            "id": tool_call_id,
                                            "type": "function",
                                            "function": {
                                                "name": missing_call.name,
                                                "arguments": missing_args,
                                            },
                                        }
                                    )

                                if missing_tool_calls:
                                    existing_tool_calls = (
                                        list(delta_message.tool_calls)
                                        if (
                                            delta_message is not None
                                            and delta_message.tool_calls
                                        )
                                        else []
                                    )
                                    delta_message = DeltaMessage(
                                        content=(
                                            delta_message.content
                                            if delta_message is not None
                                            else None
                                        ),
                                        reasoning=(
                                            delta_message.reasoning
                                            if delta_message is not None
                                            else None
                                        ),
                                        reasoning_content=(
                                            delta_message.reasoning_content
                                            if delta_message is not None
                                            else None
                                        ),
                                        tool_calls=existing_tool_calls
                                        + missing_tool_calls,
                                    )

                            latest_delta_len = 0
                            if (
                                delta_message
                                and delta_message.tool_calls
                                and len(delta_message.tool_calls) > index
                            ):
                                current_delta_tc = next(
                                    (
                                        tc
                                        for tc in delta_message.tool_calls
                                        if tc.index == index
                                    ),
                                    None,
                                )
                                if (
                                    current_delta_tc
                                    and isinstance(
                                        current_delta_tc.function,
                                        DeltaFunctionCall,
                                    )
                                    and isinstance(
                                        current_delta_tc.function.arguments, str
                                    )
                                ):
                                    latest_delta_len = len(
                                        current_delta_tc.function.arguments
                                    )

                            expected_call: str | None = None

                            # Prefer reparsing the final accumulated text over
                            # parser-side streaming state. Some models can emit
                            # the full parameter payload in the same decode step
                            # as the closing function/tool tags, which means the
                            # streaming state may still only contain "{}" while
                            # the final text is already correct.
                            # Reparse the full accumulated text for the
                            # current decode step, not the previous-step
                            # snapshot. When the last tool lands entirely in
                            # the finish window, reparsing previous_texts[i]
                            # misses it and leaves a header-only trailing tool.
                            if reparsed_calls:
                                reparsed_idx = min(index, len(reparsed_calls) - 1)
                                reparsed_args = reparsed_calls[
                                    reparsed_idx
                                ].arguments
                                if isinstance(reparsed_args, str):
                                    expected_call = reparsed_args

                            # Fall back to parser streaming state if reparsing
                            # did not produce a concrete argument string.
                            if expected_call is None:
                                # Tool parsers (e.g. Qwen3Coder) store
                                # arguments as a JSON string in
                                # prev_tool_call_arr. Calling json.dumps()
                                # on an already-serialized string would
                                # double-serialize it (e.g. '{"k":1}' becomes
                                # '"{\\"k\\":1}"'), which then causes the
                                # replace() below to fail and append the
                                # entire double-serialized string as a
                                # spurious final delta.
                                args = tool_parser.prev_tool_call_arr[index].get(
                                    "arguments", {}
                                )
                                if isinstance(args, str):
                                    expected_call = args
                                else:
                                    expected_call = json.dumps(
                                        args, ensure_ascii=False
                                    )

                            # get what we've streamed so far for arguments
                            # for the current tool
                            actual_call = ""
                            if len(previous_tool_calls[i]) > index:
                                actual_call = str(
                                    (
                                        previous_tool_calls[i][index].get("function")
                                        or {}
                                    ).get("arguments")
                                    or ""
                                )
                                if latest_delta_len > 0:
                                    actual_call = actual_call[:-latest_delta_len]
                            elif len(tool_parser.streamed_args_for_tool) > index:
                                actual_call = tool_parser.streamed_args_for_tool[index]
                                if latest_delta_len > 0:
                                    actual_call = actual_call[:-latest_delta_len]

                            # check to see if there's anything left to stream
                            if expected_call.startswith(actual_call):
                                remaining_call = expected_call[len(actual_call) :]
                            elif actual_call:
                                remaining_call = expected_call.replace(
                                    actual_call, "", 1
                                )
                            else:
                                remaining_call = expected_call

                            fallback_tool_state = (
                                previous_tool_calls[i][index]
                                if len(previous_tool_calls[i]) > index
                                else None
                            )

                            current_finish_delta = next(
                                (
                                    tc
                                    for tc in (delta_message.tool_calls or [])
                                    if tc.index == index
                                ),
                                None,
                            ) if (delta_message and delta_message.tool_calls) else None
                            current_finish_args = (
                                current_finish_delta.function.arguments
                                if (
                                    current_finish_delta
                                    and current_finish_delta.function is not None
                                    and isinstance(
                                        current_finish_delta.function.arguments, str
                                    )
                                )
                                else ""
                            )

                            preserve_finish_delta = (
                                expected_call is not None
                                and current_finish_args
                                and current_finish_args == expected_call
                            )

                            # Do not manufacture a header-only trailing tool at
                            # finish time. If there is no argument payload left
                            # to stream and the current finish delta also carries
                            # no argument fragment, suppress the tool delta.
                            if preserve_finish_delta:
                                pass
                            elif remaining_call or current_finish_args:
                                delta_message = self._create_remaining_args_delta(
                                    delta_message,
                                    remaining_call,
                                    index,
                                    fallback_tool_state=fallback_tool_state,
                                )
                                if expected_call is not None:
                                    while len(previous_tool_calls[i]) <= index:
                                        previous_tool_calls[i].append(
                                            {
                                                "id": None,
                                                "type": "function",
                                                "function": {
                                                    "name": None,
                                                    "arguments": "",
                                                },
                                            }
                                        )
                                    tool_state = previous_tool_calls[i][index]
                                    fn_state = tool_state.setdefault("function", {})
                                    original_tc = next(
                                        (
                                            tc
                                            for tc in delta_message.tool_calls
                                            if tc.index == index
                                        ),
                                        None,
                                    )
                                    if original_tc is not None:
                                        if original_tc.id:
                                            tool_state["id"] = original_tc.id
                                        if original_tc.type:
                                            tool_state["type"] = original_tc.type
                                        if (
                                            original_tc.function is not None
                                            and original_tc.function.name
                                        ):
                                            fn_state["name"] = original_tc.function.name
                                    fn_state["arguments"] = expected_call
                            else:
                                delta_message = None

                        if tool_choice_auto and delta_message and delta_message.tool_calls:
                            kept_tool_calls: list[DeltaToolCall] = []
                            for delta_tc in delta_message.tool_calls:
                                tc_idx = delta_tc.index if delta_tc.index is not None else 0
                                fn_name: str | None = None
                                fn_args: str | None = None
                                if isinstance(delta_tc.function, dict):
                                    fn_name = delta_tc.function.get("name")
                                    fn_args = delta_tc.function.get("arguments")
                                elif delta_tc.function is not None:
                                    fn_name = delta_tc.function.name
                                    fn_args = delta_tc.function.arguments

                                is_empty_arg_header = (
                                    fn_name not in (None, "")
                                    and fn_args in (None, "", "{", "{}")
                                    and not self._tool_allows_empty_arguments(
                                        request, fn_name
                                    )
                                )
                                if is_empty_arg_header:
                                    continue
                                kept_tool_calls.append(delta_tc)

                            if kept_tool_calls:
                                delta_message = DeltaMessage(
                                    content=delta_message.content,
                                    reasoning=delta_message.reasoning,
                                    reasoning_content=delta_message.reasoning_content,
                                    tool_calls=kept_tool_calls,
                                )
                            elif delta_message.content is None and delta_message.reasoning is None:
                                delta_message = None
                            else:
                                delta_message = DeltaMessage(
                                    content=delta_message.content,
                                    reasoning=delta_message.reasoning,
                                    reasoning_content=delta_message.reasoning_content,
                                )

                        force_nonempty_finish_reason: str | None = None
                        forced_tool_states: list[dict[str, Any]] | None = None
                        forced_delta, force_nonempty_finish_reason, forced_tool_states = (
                            self._build_force_nonempty_stream_finish_delta(
                                request=request,
                                tokenizer=tokenizer,
                                request_id=request_id,
                                history_tool_call_cnt=history_tool_call_cnt,
                                current_text=current_text,
                                current_content=previous_content_texts[i],
                                reasoning_parser=reasoning_parser,
                                tool_choice_auto=tool_choice_auto,
                                previous_tool_calls=previous_tool_calls[i],
                            )
                        )
                        if forced_delta is not None:
                            delta_message = forced_delta
                            if delta_message.content:
                                previous_content_texts[i] = delta_message.content
                            if forced_tool_states is not None:
                                previous_tool_calls[i] = forced_tool_states
                                tools_streamed[i] = True
                            if force_nonempty_finish_reason == "tool_calls":
                                auto_tools_called = True

                        # Send the finish response for each request.n only once
                        # In OpenAI's API, when a tool is called, the
                        # finish_reason is:
                        # "tool_calls" for "auto" or "required" tool calls,
                        # and "stop" for named tool calls.
                        if (
                            auto_tools_called
                            or (tools_streamed[i] and not tool_choice_function_name)
                            or (self.use_harmony and harmony_tools_streamed[i])
                        ):
                            finish_reason_ = "tool_calls"
                        elif force_nonempty_finish_reason is not None:
                            finish_reason_ = force_nonempty_finish_reason
                        else:
                            finish_reason_ = (
                                output.finish_reason if output.finish_reason else "stop"
                            )
                        if delta_message is None:
                            delta_message = DeltaMessage()
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=delta_message,
                            logprobs=logprobs,
                            finish_reason=finish_reason_,
                            stop_reason=output.stop_reason,
                            token_ids=(
                                as_list(output.token_ids)
                                if request.return_token_ids
                                else None
                            ),
                        )

                        finish_reason_sent[i] = True

                    choice_data = maybe_filter_parallel_tool_calls(choice_data, request)
                    chunk = ChatCompletionStreamResponse(
                        id=request_id,
                        object=chunk_object_type,
                        created=created_time,
                        choices=[choice_data],
                        model=model_name,
                    )

                    # handle usage stats if requested & if continuous
                    if include_continuous_usage:
                        completion_tokens = previous_num_tokens[i]
                        chunk.usage = UsageInfo(
                            prompt_tokens=num_prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=num_prompt_tokens + completion_tokens,
                        )

                    data = chunk.model_dump_json(exclude_unset=True)
                    yield f"data: {data}\n\n"

            # once the final token is handled, if stream_options.include_usage
            # is sent, send the usage
            if include_usage:
                completion_tokens = sum(previous_num_tokens)
                final_usage = UsageInfo(
                    prompt_tokens=num_prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=num_prompt_tokens + completion_tokens,
                )
                if self.enable_prompt_tokens_details and num_cached_tokens:
                    final_usage.prompt_tokens_details = PromptTokenUsageInfo(
                        cached_tokens=num_cached_tokens
                    )

                final_usage_chunk = ChatCompletionStreamResponse(
                    id=request_id,
                    object=chunk_object_type,
                    created=created_time,
                    choices=[],
                    model=model_name,
                    usage=final_usage,
                )
                final_usage_data = final_usage_chunk.model_dump_json(
                    exclude_unset=True, exclude_none=True
                )
                yield f"data: {final_usage_data}\n\n"

            # report to FastAPI middleware aggregate usage across all choices
            num_completion_tokens = sum(previous_num_tokens)
            request_metadata.final_usage_info = UsageInfo(
                prompt_tokens=num_prompt_tokens,
                completion_tokens=num_completion_tokens,
                total_tokens=num_prompt_tokens + num_completion_tokens,
            )

            if os.getenv("VLLM_LOG_PAYLOADS", "1") == "1":
                try:
                    usage_dict = (
                        request_metadata.final_usage_info.model_dump()
                        if request_metadata.final_usage_info
                        else None
                    )
                except Exception:
                    usage_dict = None

                choices_list = []
                for i in range(num_choices):
                    choice_data: dict[str, Any] = {
                        "index": i,
                        "message": {
                            "role": "assistant",
                            "content": previous_content_texts[i]
                            if previous_content_texts[i]
                            else None,
                        },
                        "finish_reason": "stop",
                    }
                    if previous_reasoning_texts[i]:
                        choice_data["message"]["reasoning_content"] = (
                            previous_reasoning_texts[i]
                        )

                    if previous_tool_calls[i]:
                        filtered_tool_calls = []
                        for tool_state in previous_tool_calls[i]:
                            fn = tool_state.get("function") or {}
                            if (
                                tool_state.get("id")
                                or fn.get("name")
                                or fn.get("arguments")
                            ):
                                filtered_tool_calls.append(tool_state)
                        if filtered_tool_calls:
                            choice_data["message"]["tool_calls"] = filtered_tool_calls
                            choice_data["finish_reason"] = "tool_calls"
                    choices_list.append(choice_data)

                resp_summary = {
                    "id": request_id,
                    "object": "chat.completion",
                    "created": created_time,
                    "model": model_name,
                    "choices": choices_list,
                    "usage": usage_dict,
                    "stream": True,
                }
                rid_hint = (
                    request_id[len("chatcmpl-") :]
                    if request_id.startswith("chatcmpl-")
                    else request_id
                )
                try:
                    payload_logger.info(
                        "openai.response",
                        extra={
                            "rid": rid_hint,
                            "endpoint": self.__class__.__name__,
                            "payload": resp_summary,
                        },
                    )
                except Exception:
                    pass

            # Log complete streaming response if output logging is enabled
            if self.enable_log_outputs and self.request_logger:
                for i in range(num_choices):
                    summary_payload: dict[str, Any] = {"choice_index": i}
                    if previous_reasoning_texts[i]:
                        summary_payload["reasoning"] = previous_reasoning_texts[i]
                    if previous_content_texts[i]:
                        summary_payload["content"] = previous_content_texts[i]

                    if previous_tool_calls[i]:
                        filtered_tool_calls = []
                        for tool_state in previous_tool_calls[i]:
                            fn = tool_state.get("function") or {}
                            if (
                                tool_state.get("id")
                                or fn.get("name")
                                or fn.get("arguments")
                            ):
                                filtered_tool_calls.append(tool_state)
                        if filtered_tool_calls:
                            summary_payload["tool_calls"] = filtered_tool_calls

                    if len(summary_payload) == 1:
                        summary_payload["content"] = (
                            previous_texts[i]
                            if previous_texts and i < len(previous_texts)
                            else f"<streaming_complete: {previous_num_tokens[i]} tokens>"
                        )

                    self.request_logger.log_outputs(
                        request_id=request_id,
                        outputs=json.dumps(summary_payload, ensure_ascii=False),
                        output_token_ids=None,
                        finish_reason="streaming_complete",
                        is_streaming=True,
                        delta=False,
                    )

        except GenerationError as e:
            yield f"data: {self._convert_generation_error_to_streaming_response(e)}\n\n"
        except Exception as e:
            logger.exception("Error in chat completion stream generator.")
            data = self.create_streaming_error_response(e)
            yield f"data: {data}\n\n"
        # Send the final done message after all response.n are finished
        yield "data: [DONE]\n\n"

    async def chat_completion_full_generator(
        self,
        request: ChatCompletionRequest,
        result_generator: AsyncIterator[RequestOutput],
        request_id: str,
        model_name: str,
        conversation: list[ConversationMessage],
        tokenizer: TokenizerLike,
        request_metadata: RequestResponseMetadata,
        reasoning_parser: ReasoningParser | None = None,
    ) -> ErrorResponse | ChatCompletionResponse:
        from vllm.tokenizers.mistral import MistralTokenizer

        created_time = int(time.time())
        final_res: RequestOutput | None = None

        try:
            async for res in result_generator:
                final_res = res
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")

        if final_res is None:
            return self.create_error_response(
                "No output received from the engine.",
                err_type="InternalServerError",
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

        choices: list[ChatCompletionResponseChoice] = []
        if self.tool_call_id_type == "kimi_k2":
            history_tool_call_cnt = get_history_tool_calls_cnt(conversation)
        else:
            history_tool_call_cnt = 0

        role = self.get_chat_request_role(request)
        for output in final_res.outputs:
            # check for error finish reason and raise GenerationError
            # finish_reason='error' indicates a retryable request-level internal error
            self._raise_if_error(output.finish_reason, request_id)
            token_ids = output.token_ids
            out_logprobs = output.logprobs
            tool_call_info = None

            if request.logprobs and request.top_logprobs is not None:
                assert out_logprobs is not None, "Did not output logprobs"
                logprobs = self._create_chat_logprobs(
                    token_ids=token_ids,
                    top_logprobs=out_logprobs,
                    num_output_top_logprobs=request.top_logprobs,
                    tokenizer=tokenizer,
                    return_as_token_id=request.return_tokens_as_token_ids,
                )
            else:
                logprobs = None

            if self.use_harmony:
                reasoning, content, _ = parse_chat_output(token_ids)
                if not request.include_reasoning:
                    reasoning = None

                if self.tool_parser is not None:
                    if tokenizer is None:
                        raise ValueError(
                            "Tokenizer not available when `skip_tokenizer_init=True`"
                        )

                    tool_parser = self.tool_parser(tokenizer)
                    # NOTE: We use token_ids for openai tool parser
                    tool_call_info = tool_parser.extract_tool_calls(
                        "",
                        request=request,
                        token_ids=token_ids,  # type: ignore
                    )
                    content = tool_call_info.content
                    message = ChatMessage(
                        role=role,
                        reasoning=reasoning,
                        content=content,
                        tool_calls=tool_call_info.tool_calls,
                    )
                else:
                    message = ChatMessage(
                        role=role,
                        reasoning=reasoning,
                        content=content,
                    )

                choice_data = ChatCompletionResponseChoice(
                    index=output.index,
                    message=message,
                    logprobs=logprobs,
                    finish_reason=(
                        "tool_calls"
                        if (tool_call_info is not None and tool_call_info.tools_called)
                        else output.finish_reason
                        if output.finish_reason
                        else "stop"
                    ),
                    stop_reason=output.stop_reason,
                    token_ids=(
                        as_list(output.token_ids) if request.return_token_ids else None
                    ),
                )
                choices.append(choice_data)
                continue

            if reasoning_parser:
                # If the reasoning parser is enabled,
                # tool calls are extracted exclusively from the content.
                reasoning, content = reasoning_parser.extract_reasoning(
                    output.text,
                    request=self._get_reasoning_parser_request_view(request),
                )
                if not request.include_reasoning:
                    reasoning = None
            else:
                reasoning = None
                content = output.text

            auto_tools_called = False
            # if auto tools are not enabled, and a named tool choice using
            #   outlines is not being used
            tool_calls, content = self._parse_tool_calls_from_content(
                request=request,
                tokenizer=tokenizer,
                content=content,
                enable_auto_tools=self.enable_auto_tools,
                tool_parser_cls=self.tool_parser,
            )
            tool_call_class = (
                MistralToolCall if is_mistral_tokenizer(tokenizer) else ToolCall
            )
            if (not self.enable_auto_tools or not self.tool_parser) and (
                not isinstance(request.tool_choice, ChatCompletionNamedToolChoiceParam)
                and request.tool_choice != "required"
            ):
                message = ChatMessage(role=role, reasoning=reasoning, content=content)

            elif (
                request.tool_choice
                and type(request.tool_choice) is ChatCompletionNamedToolChoiceParam
            ):
                assert tool_calls is not None and len(tool_calls) > 0
                tool_call_class_items = []
                for idx, tc in enumerate(tool_calls):
                    # Use native ID if available (e.g., Kimi K2),
                    # otherwise generate ID with correct id_type
                    if tc.id:
                        tool_call_class_items.append(
                            tool_call_class(id=tc.id, function=tc)
                        )
                    else:
                        # Generate ID using the correct format (kimi_k2 or random),
                        # but leave it to the class if it's Mistral to preserve
                        # 9-char IDs
                        if isinstance(tokenizer, MistralTokenizer):
                            tool_call_class_items.append(tool_call_class(function=tc))
                        else:
                            generated_id = make_tool_call_id(
                                id_type=self.tool_call_id_type,
                                func_name=tc.name,
                                idx=history_tool_call_cnt,
                            )
                            tool_call_class_items.append(
                                tool_call_class(id=generated_id, function=tc)
                            )
                    history_tool_call_cnt += 1
                message = ChatMessage(
                    role=role,
                    reasoning=reasoning,
                    content="",
                    tool_calls=tool_call_class_items,
                )

            elif request.tool_choice and request.tool_choice == "required":
                tool_call_class_items = []
                tool_calls = tool_calls or []
                for idx, tool_call in enumerate(tool_calls):
                    # Use native ID if available,
                    # otherwise generate ID with correct id_type
                    if tool_call.id:
                        tool_call_class_items.append(
                            tool_call_class(id=tool_call.id, function=tool_call)
                        )
                    else:
                        # Generate ID using the correct format (kimi_k2 or random),
                        # but leave it to the class if it's Mistral to preserve
                        # 9-char IDs
                        if isinstance(tokenizer, MistralTokenizer):
                            tool_call_class_items.append(
                                tool_call_class(function=tool_call)
                            )
                        else:
                            generated_id = make_tool_call_id(
                                id_type=self.tool_call_id_type,
                                func_name=tool_call.name,
                                idx=history_tool_call_cnt,
                            )
                            tool_call_class_items.append(
                                tool_call_class(id=generated_id, function=tool_call)
                            )
                    history_tool_call_cnt += 1
                message = ChatMessage(
                    role=role,
                    content="",
                    tool_calls=tool_call_class_items,
                    reasoning=reasoning,
                )

            # if the request doesn't use tool choice
            # OR specifies to not use a tool
            elif not request.tool_choice or request.tool_choice == "none":
                message = ChatMessage(role=role, reasoning=reasoning, content=content)

            # handle when there are tools and tool choice is auto
            elif (
                request.tools
                and (request.tool_choice == "auto" or request.tool_choice is None)
                and self.enable_auto_tools
                and self.tool_parser
            ):
                # In the OpenAI API the finish_reason is "tools_called"
                # if the tool choice is auto and the model produced a tool
                # call. The same is not true for named function calls
                auto_tools_called = tool_calls is not None and len(tool_calls) > 0
                if tool_calls:
                    tool_call_items = []
                    for idx, tc in enumerate(tool_calls):
                        # Use native ID if available (e.g., Kimi K2),
                        # otherwise generate ID with correct id_type
                        if tc.id:
                            tool_call_items.append(
                                tool_call_class(id=tc.id, function=tc)
                            )
                        else:
                            # Generate ID using the correct format (kimi_k2 or random),
                            # but leave it to the class if it's Mistral to preserve
                            # 9-char IDs
                            if isinstance(tokenizer, MistralTokenizer):
                                tool_call_items.append(tool_call_class(function=tc))
                            else:
                                generated_id = make_tool_call_id(
                                    id_type=self.tool_call_id_type,
                                    func_name=tc.name,
                                    idx=history_tool_call_cnt,
                                )
                                tool_call_items.append(
                                    tool_call_class(id=generated_id, function=tc)
                                )
                        history_tool_call_cnt += 1
                    message = ChatMessage(
                        role=role,
                        reasoning=reasoning,
                        content=content,
                        tool_calls=tool_call_items,
                    )

                else:
                    # FOR NOW make it a chat message; we will have to detect
                    # the type to make it later.
                    ret_content = content

                    # try to use content return from tool parser first,
                    # tool parser may do some modify for the content.
                    if content and len(content) > 0:
                        ret_content = content
                    message = ChatMessage(
                        role=role,
                        reasoning=reasoning,
                        content=ret_content,
                    )

            # undetermined case that is still important to handle
            else:
                logger.error(
                    "Error in chat_completion_full_generator - cannot determine"
                    " if tools should be extracted. Returning a standard chat "
                    "completion."
                )
                message = ChatMessage(role=role, reasoning=reasoning, content=content)
            # In OpenAI's API, when a tool is called, the finish_reason is:
            # "tool_calls" for "auto" or "required" tool calls,
            # and "stop" for named tool calls.
            is_finish_reason_tool_calls = auto_tools_called or (
                request.tool_choice
                and request.tool_choice == "required"
                and output.finish_reason == "stop"
            )

            choice_data = ChatCompletionResponseChoice(
                index=output.index,
                message=message,
                logprobs=logprobs,
                finish_reason="tool_calls"
                if is_finish_reason_tool_calls
                else output.finish_reason
                if output.finish_reason
                else "stop",
                stop_reason=output.stop_reason,
                token_ids=(
                    as_list(output.token_ids) if request.return_token_ids else None
                ),
            )
            choice_data = maybe_filter_parallel_tool_calls(choice_data, request)

            choices.append(choice_data)

        if request.echo:
            last_msg_content: str | list[dict[str, str]] = ""
            if (
                conversation
                and "content" in conversation[-1]
                and conversation[-1].get("role") == role
            ):
                last_msg_content = conversation[-1]["content"] or ""
            if isinstance(last_msg_content, list):
                last_msg_content = "\n".join(msg["text"] for msg in last_msg_content)

            for choice in choices:
                full_message = last_msg_content + (choice.message.content or "")
                choice.message.content = full_message

        assert final_res.prompt_token_ids is not None
        num_prompt_tokens = len(final_res.prompt_token_ids)
        if final_res.encoder_prompt_token_ids is not None:
            num_prompt_tokens += len(final_res.encoder_prompt_token_ids)
        num_generated_tokens = sum(
            len(output.token_ids) for output in final_res.outputs
        )
        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
        )
        if self.enable_prompt_tokens_details and final_res.num_cached_tokens:
            usage.prompt_tokens_details = PromptTokenUsageInfo(
                cached_tokens=final_res.num_cached_tokens
            )

        request_metadata.final_usage_info = usage

        response = ChatCompletionResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=choices,
            usage=usage,
            prompt_logprobs=clamp_prompt_logprobs(final_res.prompt_logprobs),
            prompt_token_ids=(
                final_res.prompt_token_ids if request.return_token_ids else None
            ),
            kv_transfer_params=final_res.kv_transfer_params,
        )

        if os.getenv("VLLM_LOG_PAYLOADS", "1") == "1":
            rid_hint = (
                request_id[len("chatcmpl-") :]
                if request_id.startswith("chatcmpl-")
                else request_id
            )
            try:
                payload_logger.info(
                    "openai.response",
                    extra={
                        "rid": rid_hint,
                        "endpoint": self.__class__.__name__,
                        "payload": response.model_dump(),
                    },
                )
            except Exception:
                pass

        # Log complete response if output logging is enabled
        if self.enable_log_outputs and self.request_logger:
            for choice in choices:
                output_text = ""
                payload: dict[str, Any] = {"choice_index": choice.index}
                reasoning_text = (
                    choice.message.reasoning_content or choice.message.reasoning
                )
                if reasoning_text:
                    payload["reasoning"] = reasoning_text
                if choice.message.content:
                    payload["content"] = choice.message.content
                if choice.message.tool_calls:
                    tool_calls_payload = []
                    for tc in choice.message.tool_calls:
                        fn_name = None
                        fn_args = None
                        if hasattr(tc.function, "name"):
                            fn_name = tc.function.name
                        if hasattr(tc.function, "arguments"):
                            fn_args = tc.function.arguments
                        tool_calls_payload.append(
                            {
                                "id": tc.id,
                                "type": tc.type,
                                "function": {
                                    "name": fn_name,
                                    "arguments": fn_args or "",
                                },
                            }
                        )
                    if tool_calls_payload:
                        payload["tool_calls"] = tool_calls_payload

                if len(payload) > 1:
                    output_text = json.dumps(payload, ensure_ascii=False)
                    # Get the corresponding output token IDs
                    output_token_ids = None
                    if choice.index < len(final_res.outputs):
                        output_token_ids = final_res.outputs[choice.index].token_ids

                    self.request_logger.log_outputs(
                        request_id=request_id,
                        outputs=output_text,
                        output_token_ids=output_token_ids,
                        finish_reason=choice.finish_reason,
                        is_streaming=False,
                        delta=False,
                    )

        return response

    def _get_top_logprobs(
        self,
        logprobs: dict[int, Logprob],
        top_logprobs: int | None,
        tokenizer: TokenizerLike | None,
        should_return_as_token_id: bool,
    ) -> list[ChatCompletionLogProb]:
        return [
            ChatCompletionLogProb(
                token=(
                    token := self._get_decoded_token(
                        p[1],
                        p[0],
                        tokenizer,
                        return_as_token_id=should_return_as_token_id,
                    )
                ),
                logprob=max(p[1].logprob, -9999.0),
                bytes=list(token.encode("utf-8", errors="replace")),
            )
            for i, p in enumerate(logprobs.items())
            if (top_logprobs and i < top_logprobs or top_logprobs == -1)
        ]

    def _create_chat_logprobs(
        self,
        token_ids: GenericSequence[int],
        top_logprobs: GenericSequence[dict[int, Logprob] | None],
        tokenizer: TokenizerLike | None,
        num_output_top_logprobs: int | None = None,
        return_as_token_id: bool | None = None,
    ) -> ChatCompletionLogProbs:
        """Create OpenAI-style logprobs."""
        logprobs_content: list[ChatCompletionLogProbsContent] = []

        should_return_as_token_id = (
            return_as_token_id
            if return_as_token_id is not None
            else self.return_tokens_as_token_ids
        )
        for i, token_id in enumerate(token_ids):
            step_top_logprobs = top_logprobs[i]
            if step_top_logprobs is None or step_top_logprobs.get(token_id) is None:
                if should_return_as_token_id:
                    token = f"token_id:{token_id}"
                else:
                    if tokenizer is None:
                        raise ValueError(
                            "Unable to get tokenizer because `skip_tokenizer_init=True`"
                        )

                    token = tokenizer.decode(token_id)

                logprobs_content.append(
                    ChatCompletionLogProbsContent(
                        token=token,
                        bytes=list(token.encode("utf-8", errors="replace")),
                    )
                )
            else:
                step_token = step_top_logprobs[token_id]
                step_decoded = step_token.decoded_token

                logprobs_content.append(
                    ChatCompletionLogProbsContent(
                        token=self._get_decoded_token(
                            step_token,
                            token_id,
                            tokenizer,
                            should_return_as_token_id,
                        ),
                        logprob=max(step_token.logprob, -9999.0),
                        bytes=(
                            None
                            if step_decoded is None
                            else list(step_decoded.encode("utf-8", errors="replace"))
                        ),
                        top_logprobs=self._get_top_logprobs(
                            step_top_logprobs,
                            num_output_top_logprobs,
                            tokenizer,
                            should_return_as_token_id,
                        ),
                    )
                )

        return ChatCompletionLogProbs(content=logprobs_content)

    def _should_stream_with_auto_tool_parsing(self, request: ChatCompletionRequest):
        """
        Utility function to check if streamed tokens should go through the tool
        call parser that was configured.

        We only want to do this IF user-provided tools are set, a tool parser
        is configured, "auto" tool choice is enabled, and the request's tool
        choice field indicates that "auto" tool choice should be used.
        """
        return (
            request.tools
            and self.tool_parser
            and self.enable_auto_tools
            and request.tool_choice in ["auto", None]
        )

    def _should_check_for_unstreamed_tool_arg_tokens(
        self,
        delta_message: DeltaMessage | None,
        output: CompletionOutput,
    ) -> bool:
        """
        Check to see if we should check for unstreamed tool arguments tokens.
        This is only applicable when auto tool parsing is enabled, the delta
        is a tool call with arguments.
        """

        return bool(
            # if there is a delta message that includes tool calls which
            # include a function that has arguments
            output.finish_reason is not None
            and self.enable_auto_tools
            and self.tool_parser
            and delta_message
            and delta_message.tool_calls
            and delta_message.tool_calls[0]
            and delta_message.tool_calls[0].function
            and delta_message.tool_calls[0].function.arguments is not None
        )

    @staticmethod
    def _create_remaining_args_delta(
        delta_message: DeltaMessage,
        remaining_call: str,
        index: int,
        fallback_tool_state: dict[str, Any] | None = None,
    ) -> DeltaMessage:
        """
        Create a delta message for remaining tool arguments, preserving
        id/type/name from the original delta when present, otherwise from the
        already-streamed tool state.
        """
        tool_calls = delta_message.tool_calls or []
        original_tc = next((tc for tc in tool_calls if tc.index == index), None)
        original_fn = original_tc.function if original_tc else None
        fallback_fn = (
            (fallback_tool_state or {}).get("function", {})
            if fallback_tool_state
            else {}
        )
        return DeltaMessage(
            tool_calls=[
                DeltaToolCall(
                    index=index,
                    id=(
                        original_tc.id
                        if original_tc and original_tc.id is not None
                        else (fallback_tool_state or {}).get("id")
                    ),
                    type=(
                        original_tc.type
                        if original_tc and original_tc.type is not None
                        else (fallback_tool_state or {}).get("type")
                    ),
                    function=DeltaFunctionCall(
                        name=(
                            original_fn.name
                            if original_fn and original_fn.name is not None
                            else fallback_fn.get("name")
                        ),
                        arguments=remaining_call,
                    ),
                )
            ]
        )

    @staticmethod
    def _merge_delta_messages(
        first: DeltaMessage | None,
        second: DeltaMessage | None,
    ) -> DeltaMessage | None:
        if first is None:
            return second
        if second is None:
            return first
        return DeltaMessage(
            role=first.role or second.role,
            content=(first.content or "") + (second.content or "") or None,
            reasoning=(
                (first.reasoning_content or first.reasoning or "")
                + (second.reasoning_content or second.reasoning or "")
            )
            or None,
            tool_calls=[*(first.tool_calls or []), *(second.tool_calls or [])],
        )

    def _make_request_with_harmony(
        self,
        request: ChatCompletionRequest,
        should_include_tools: bool = True,
    ):
        messages: list[OpenAIMessage] = []

        # because of issues with pydantic we need to potentially
        # re-serialize the tool_calls field of the request
        # for more info: see comment in `maybe_serialize_tool_calls`
        _mt.maybe_serialize_tool_calls(request)  # type: ignore[arg-type]

        # Add system message.
        # NOTE: In Chat Completion API, browsing is enabled by default
        # if the model supports it. TODO: Support browsing.
        assert not self.supports_browsing
        assert not self.supports_code_interpreter
        sys_msg = get_system_message(
            reasoning_effort=request.reasoning_effort,
            browser_description=None,
            python_description=None,
            with_custom_tools=should_include_tools,
        )
        messages.append(sys_msg)

        # Add developer message.
        if request.tools:
            dev_msg = get_developer_message(
                tools=request.tools if should_include_tools else None  # type: ignore[arg-type]
            )
            messages.append(dev_msg)

        # Add user message.
        messages.extend(parse_chat_inputs_to_harmony_messages(request.messages))

        # Render prompt token ids.
        prompt_token_ids = render_for_completion(messages)
        engine_prompt = TokensPrompt(prompt_token_ids=prompt_token_ids)

        # Add cache_salt if provided in the request
        if request.cache_salt is not None:
            engine_prompt["cache_salt"] = request.cache_salt

        return messages, [engine_prompt]
