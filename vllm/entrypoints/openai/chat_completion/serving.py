# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import io
import json
import logging
import os
import time
from collections.abc import AsyncGenerator, AsyncIterator
from collections.abc import Sequence as GenericSequence
from http import HTTPStatus
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pybase64 as base64
from fastapi import Request

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import (
    ChatTemplateContentFormatOption,
    ConversationMessage,
    get_history_tool_calls_cnt,
    get_tool_call_id_type,
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
from vllm.entrypoints.openai.request_metrics import (
    classify_chat_request,
    record_aborted_request,
    summarize_request_payload,
)
from vllm.entrypoints.openai.utils import maybe_filter_parallel_tool_calls
from vllm.entrypoints.utils import get_max_tokens, should_include_usage
from vllm.inputs import EngineInput
from vllm.logger import init_logger
from vllm.logprobs import Logprob
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.parser import ParserManager
from vllm.parser.abstract_parser import Parser
from vllm.payload_sanitization import prepare_request_payload_for_logging
from vllm.reasoning import ReasoningParser
from vllm.renderers import ChatParams
from vllm.sampling_params import BeamSearchParams, SamplingParams
from vllm.tokenizers import TokenizerLike
from vllm.utils.collection_utils import as_list
from vllm.utils.mistral import is_mistral_tokenizer, is_mistral_tool_parser

if TYPE_CHECKING:
    from vllm.entrypoints.serve.render.serving import OpenAIServingRender

logger = init_logger(__name__)
payload_logger = logging.getLogger("vllm.payload")


def _log_raw_chat_generation_debug(
    request_id: str,
    phase: str,
    choice_index: int,
    tokenizer: TokenizerLike,
    token_ids: list[int],
    text: str,
) -> None:
    if not logger.isEnabledFor(logging.DEBUG):
        return

    token_pieces = [
        {
            "id": token_id,
            "piece": tokenizer.decode([token_id], skip_special_tokens=False),
        }
        for token_id in token_ids
    ]
    logger.debug(
        "raw_chat_generation request_id=%s phase=%s choice=%s text=%r "
        "token_ids=%s token_pieces=%r",
        request_id,
        phase,
        choice_index,
        text,
        token_ids,
        token_pieces,
    )


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
        enable_log_deltas: bool = True,
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
        self.parser_cls = ParserManager.get_parser(
            tool_parser_name=tool_parser,
            reasoning_parser_name=reasoning_parser,
            enable_auto_tools=enable_auto_tools,
            model_name=self.model_config.model,
        )
        if (
            is_mistral_tool_parser(self.tool_parser)
            and self.reasoning_parser_cls is not None
        ):
            from vllm.tool_parsers.mistral_tool_parser import MistralToolParser

            MistralToolParser.model_can_reason = True

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

        self.tool_call_id_type = get_tool_call_id_type(self.model_config)

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
                reasoning_budget = max(32, max_tokens - reserve)

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

    def _effective_chat_template_kwargs(
        self, request: ChatCompletionRequest
    ) -> dict[str, Any]:
        return (
            request.build_chat_params(
                self.chat_template,
                self.chat_template_content_format,
            )
            .with_defaults(self.default_chat_template_kwargs)
            .chat_template_kwargs
        )

    async def _log_chat_request_payload(
        self,
        request: ChatCompletionRequest,
        raw_request: Request | None,
        rid: str,
    ) -> None:
        if os.getenv("VLLM_LOG_PAYLOADS", "1") != "1":
            return
        headers_obj = None
        if raw_request is not None:
            try:
                headers_obj = {k: v for k, v in raw_request.headers.items()}
            except Exception:
                headers_obj = None
        payload: dict[str, Any] | None = None
        if raw_request is not None:
            try:
                body = await raw_request.body()
                payload = json.loads(body) if body else None
            except Exception:
                payload = None
        if payload is None:
            try:
                payload = request.model_dump(mode="json")
            except Exception:
                payload = None
        try:
            allowed_local_media_path = (
                self.openai_serving_render.model_config.allowed_local_media_path
            )
        except Exception:
            allowed_local_media_path = ""
        try:
            summary = summarize_request_payload(payload)
            payload_logger.info(
                "openai.request",
                extra={
                    "rid": rid,
                    "endpoint": self.__class__.__name__,
                    "input_image_count": summary.image_count,
                    "input_video_count": summary.video_count,
                    "input_audio_count": summary.audio_count,
                    "input_tool_count": summary.tool_count,
                    "has_images": summary.has_images,
                    "has_videos": summary.has_videos,
                    "has_audios": summary.has_audios,
                    "has_tools": summary.has_tools,
                    "has_tool_calls_enabled": summary.has_tool_calls_enabled,
                    "has_structured_output": summary.has_structured_output,
                    **(
                        {"tool_choice": summary.tool_choice}
                        if summary.tool_choice is not None
                        else {}
                    ),
                    **(
                        {"structured_output_kind": summary.structured_output_kind}
                        if summary.structured_output_kind is not None
                        else {}
                    ),
                    "payload": prepare_request_payload_for_logging(
                        payload,
                        headers=headers_obj,
                        allowed_local_media_path=allowed_local_media_path,
                    ),
                    "headers": headers_obj,
                },
            )
        except Exception:
            logger.exception("Failed to log openai.request rid=%s", rid)

    def _log_chat_response_payload(self, rid: str, payload: dict[str, Any]) -> None:
        if os.getenv("VLLM_LOG_PAYLOADS", "1") != "1":
            return
        try:
            payload_logger.info(
                "openai.response",
                extra={
                    "rid": rid,
                    "request_id": rid,
                    "endpoint": self.__class__.__name__,
                    "payload": payload,
                },
            )
        except Exception:
            logger.exception("Failed to log openai.response rid=%s", rid)

    async def render_chat_request(
        self,
        request: ChatCompletionRequest,
    ) -> tuple[list[ConversationMessage], list[EngineInput]] | ErrorResponse:
        """
        Validate the model and preprocess a chat completion request.

        Delegates preprocessing logic to OpenAIServingRender, adding the
        engine-aware checks (LoRA model validation, engine health).

        Returns:
            A tuple of (conversation, engine_inputs) on success,
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
        return await self._with_kv_transfer_rejection_cleanup(
            self._create_chat_completion(request, raw_request), request, raw_request
        )

    async def _create_chat_completion(
        self,
        request: ChatCompletionRequest,
        raw_request: Request | None = None,
    ) -> AsyncGenerator[str, None] | ChatCompletionResponse | ErrorResponse:
        # Streaming response
        tokenizer = self.renderer.tokenizer
        assert tokenizer is not None
        chat_template_kwargs = self._effective_chat_template_kwargs(request)
        reasoning_parser: ReasoningParser | None = None
        if self.reasoning_parser_cls:
            reasoning_parser = self.reasoning_parser_cls(
                tokenizer,
                chat_template_kwargs=chat_template_kwargs,  # type: ignore[call-arg]
            )
        rid_hint = self._base_request_id(raw_request, request.request_id)
        await self._log_chat_request_payload(request, raw_request, rid_hint)
        result = await self.render_chat_request(request)
        if isinstance(result, ErrorResponse):
            return result

        classify_chat_request(request)
        conversation, engine_inputs = result

        request_id = f"chatcmpl-{rid_hint}"

        request_metadata = RequestResponseMetadata(request_id=request_id)
        if raw_request:
            raw_request.state.request_metadata = request_metadata

        lora_request = self._maybe_get_adapters(request, supports_default_mm_loras=True)

        model_name = self.models.model_name(lora_request)

        # Extract data_parallel_rank from header (router can inject it)
        data_parallel_rank = self._get_data_parallel_rank(raw_request)

        # Schedule the request and get the result generator.
        max_model_len = self.model_config.max_model_len
        generators: list[AsyncGenerator[RequestOutput, None]] = []
        for i, engine_input in enumerate(engine_inputs):
            prompt_token_ids = self._extract_prompt_components(engine_input).token_ids

            # If we are creating sub requests for multiple prompts, ensure that they
            # have unique request ids.
            sub_request_id = (
                request_id if len(engine_inputs) == 1 else f"{request_id}_{i}"
            )

            max_tokens = get_max_tokens(
                max_model_len,
                request.max_completion_tokens
                if request.max_completion_tokens is not None
                else request.max_tokens,
                self._extract_prompt_len(engine_input),
                self.default_sampling_params,
                self.override_max_tokens,
                truncate_prompt_tokens=request.truncate_prompt_tokens,
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
                self._inject_think_end_token_id(
                    sampling_params,
                    request,
                    tokenizer,
                    reasoning_parser,
                )

            self._log_inputs(
                sub_request_id,
                engine_input,
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
                    prompt=engine_input,
                    request_id=sub_request_id,
                    params=sampling_params,
                    lora_request=lora_request,
                    trace_headers=trace_headers,
                )
            else:
                if (
                    request.structured_outputs is not None
                    and reasoning_parser
                    and chat_template_kwargs.get("enable_thinking", True) is not False
                ):
                    reasoning_ended = False
                elif not request.include_reasoning:
                    reasoning_ended = True
                elif request._grammar_from_tool_parser:
                    # The Mistral grammar already includes an optional
                    # `think?` rule that handles both reasoning and
                    # non-reasoning outputs.
                    reasoning_ended = True
                elif reasoning_parser:
                    reasoning_ended = reasoning_parser.is_reasoning_end(
                        prompt_token_ids or []
                    )
                else:
                    reasoning_ended = None

                generator = self.engine_client.generate(
                    engine_input,
                    sampling_params,
                    sub_request_id,
                    lora_request=lora_request,
                    trace_headers=trace_headers,
                    priority=request.priority,
                    data_parallel_rank=data_parallel_rank,
                    reasoning_ended=reasoning_ended,
                    reasoning_parser_kwargs={
                        "chat_template_kwargs": chat_template_kwargs,
                    }
                    if reasoning_parser
                    else None,
                )

            generators.append(generator)

        assert len(generators) == 1
        (result_generator,) = generators

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
                chat_template_kwargs=chat_template_kwargs,
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
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> AsyncGenerator[str, None]:
        created_time = int(time.time())
        chunk_object_type: Final = "chat.completion.chunk"
        first_iteration = True

        # Send response for each token for each request.n (index)
        num_choices = 1 if request.n is None else request.n
        previous_num_tokens = [0] * num_choices
        finish_reason_sent = [False] * num_choices
        final_finish_reasons: list[str | None] = [None] * num_choices
        num_prompt_tokens = 0
        num_cached_tokens = None
        if self.use_harmony:
            harmony_parsers = [
                get_streamable_parser_for_assistant() for _ in range(num_choices)
            ]
            harmony_tools_streamed = [False] * num_choices
        tools_streamed = [False] * num_choices

        is_mistral_grammar_path = request._grammar_from_tool_parser

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
        if self.tool_call_id_type == "kimi_k2":
            history_tool_call_cnt = get_history_tool_calls_cnt(conversation)
        else:
            history_tool_call_cnt = 0

        # Always track previous_texts for comprehensive output logging
        previous_texts = [""] * num_choices
        streamed_content_texts = [""] * num_choices
        streamed_reasoning_texts = [""] * num_choices
        pending_tool_whitespace_content = [""] * num_choices
        streamed_tool_calls: list[dict[int, dict[str, Any]]] = [
            {} for _ in range(num_choices)
        ]

        # Only one of these will be used, thus previous_texts and
        # all_previous_token_ids will not be used twice in the same iteration.
        if (
            is_mistral_grammar_path
            or tool_choice_auto
            or tool_choice_function_name
            or request.tool_choice == "required"
            or reasoning_parser
        ):
            all_previous_token_ids = [[] for _ in range(num_choices)]
            reasoning_end_arr = [False] * num_choices
            prompt_is_reasoning_end_arr: list[bool | None] = [None] * num_choices
        else:
            all_previous_token_ids = None

        try:
            if self.parser_cls is not None:
                if tokenizer is None:
                    raise ValueError(
                        "Tokenizer not available when `skip_tokenizer_init=True`"
                    )
                parsers: list[Parser | None] = [
                    self.parser_cls(
                        tokenizer,
                        request.tools,
                        chat_template_kwargs=chat_template_kwargs,
                    )
                    for _ in range(num_choices)
                ]
                for p in parsers:
                    if p is not None:
                        p._stream_state.tool_call_id_type = self.tool_call_id_type
                        p._stream_state.history_tool_call_cnt = history_tool_call_cnt
            else:
                parsers = [None] * num_choices
        except Exception as e:
            logger.exception("Error in parser creation.")
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

                    # ``res.prompt`` is the rendered chat-templated prompt
                    prompt_text = res.prompt if request.return_prompt_text else None

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
                            prompt_text=prompt_text,
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
                    parser = parsers[i]
                    tool_parser = parser.tool_parser if parser is not None else None

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
                        if not delta_text and output.token_ids:
                            delta_text = tokenizer.decode(
                                as_list(output.token_ids),
                                skip_special_tokens=True,
                            )
                    _log_raw_chat_generation_debug(
                        request_id,
                        "stream_delta",
                        i,
                        tokenizer,
                        as_list(output.token_ids),
                        delta_text,
                    )

                    if (
                        not delta_text
                        and not output.token_ids
                        and not previous_num_tokens[i]
                    ):
                        # Chunked prefill case, don't return empty chunks
                        continue

                    delta_message: DeltaMessage | None

                    # just update previous_texts and previous_token_ids
                    if (
                        is_mistral_grammar_path
                        or tool_choice_auto
                        or tool_choice_function_name
                        or request.tool_choice == "required"
                        or reasoning_parser
                    ):
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

                    thinking_disabled = (
                        bool(chat_template_kwargs)
                        and (
                            chat_template_kwargs.get("enable_thinking") is False
                            or chat_template_kwargs.get("thinking") is False
                        )
                    ) or request.reasoning_effort == "none"

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
                    # Mistral grammar path: combined reasoning + tool streaming
                    elif is_mistral_grammar_path:
                        from vllm.tool_parsers.mistral_tool_parser import (
                            MistralToolParser,
                        )

                        assert tool_parser is not None
                        assert isinstance(tool_parser, MistralToolParser)
                        assert reasoning_end_arr is not None
                        output_token_ids = as_list(output.token_ids)
                        result = tool_parser.extract_maybe_reasoning_and_tool_streaming(
                            reasoning_parser=reasoning_parser,
                            previous_text=previous_text,
                            current_text=current_text,
                            delta_text=delta_text,
                            previous_token_ids=previous_token_ids,
                            current_token_ids=current_token_ids,
                            output_token_ids=output_token_ids,
                            reasoning_ended=reasoning_end_arr[i],
                            prompt_is_reasoning_end=(prompt_is_reasoning_end_arr[i]),
                            request=request,
                        )
                        delta_message = result.delta_message
                        reasoning_end_arr[i] = result.reasoning_ended
                        current_text = result.current_text
                        current_token_ids = result.current_token_ids
                        if result.tools_called:
                            tools_streamed[i] = True

                    elif parser is not None:
                        delta_message = parser.parse_delta(
                            delta_text=delta_text,
                            delta_token_ids=as_list(output.token_ids),
                            request=request,
                            prompt_token_ids=res.prompt_token_ids,
                        )
                        if (
                            reasoning_parser
                            and not request.tools
                            and (delta_message is None or not delta_message.content)
                            and not (
                                delta_message is not None
                                and delta_message.tool_calls
                            )
                        ):
                            reasoning_end = getattr(
                                reasoning_parser, "reasoning_end_str", None
                            )
                            if reasoning_end and reasoning_end in current_text:
                                previous_content = (
                                    previous_text.rsplit(reasoning_end, 1)[1]
                                    if reasoning_end in previous_text
                                    else ""
                                )
                                current_content = current_text.rsplit(
                                    reasoning_end, 1
                                )[1]
                                if current_content.startswith(previous_content):
                                    content_delta = current_content[
                                        len(previous_content) :
                                    ]
                                    if content_delta:
                                        if delta_message is None:
                                            delta_message = DeltaMessage()
                                        delta_message.content = content_delta
                        if delta_message and delta_message.tool_calls:
                            tools_streamed[i] = True
                    # handle streaming just a content delta (no parsers)
                    else:
                        delta_message = DeltaMessage(content=delta_text)

                    # update the previous values for the next iteration
                    if (
                        is_mistral_grammar_path
                        or tool_choice_auto
                        or tool_choice_function_name
                        or request.tool_choice == "required"
                        or reasoning_parser
                    ) and not self.use_harmony:
                        assert previous_texts is not None
                        assert all_previous_token_ids is not None
                        previous_texts[i] = current_text
                        all_previous_token_ids[i] = current_token_ids
                    else:
                        # Update for comprehensive logging even in simple case
                        assert previous_texts is not None
                        previous_texts[i] += delta_text

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
                    if (
                        thinking_disabled
                        and not request.tools
                        and delta_message.reasoning
                        and not delta_message.content
                        and not delta_message.tool_calls
                    ):
                        delta_message.content = delta_message.reasoning
                        delta_message.reasoning = None
                    if (
                        thinking_disabled
                        and not request.tools
                        and not delta_message.content
                        and not delta_message.reasoning
                        and not delta_message.tool_calls
                        and delta_text
                    ):
                        delta_message.content = delta_text

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
                        index = 0
                        auto_tools_called = False
                        if tool_parser:
                            auto_tools_called = len(tool_parser.prev_tool_call_arr) > 0
                            index = (
                                len(tool_parser.prev_tool_call_arr) - 1
                                if auto_tools_called
                                else 0
                            )
                        should_check = (
                            self._should_check_for_unstreamed_tool_arg_tokens(
                                delta_message, output
                            )
                            or (
                                output.finish_reason is not None
                                and self.enable_auto_tools
                                and self.tool_parser
                                and auto_tools_called
                            )
                        )
                        # only check if there are any tool calls
                        # detected by partial parsing
                        if should_check and tool_parser and auto_tools_called:
                            full_tool_info = tool_parser.extract_tool_calls(
                                current_text, request
                            )
                            delta_args_by_index = {}
                            delta_by_index = {}
                            for tc in delta_message.tool_calls:
                                if tc.index is not None:
                                    delta_by_index[tc.index] = tc
                                if (
                                    tc.index is not None
                                    and tc.function is not None
                                    and isinstance(tc.function.arguments, str)
                                ):
                                    delta_args_by_index[tc.index] = (
                                        tc.function.arguments
                                    )

                            repair_tool_calls = []
                            for repair_index, full_tool_call in enumerate(
                                full_tool_info.tool_calls
                            ):
                                expected_call = full_tool_call.function.arguments
                                if not isinstance(expected_call, str):
                                    expected_call = json.dumps(
                                        expected_call, ensure_ascii=False
                                    )
                                actual_call = (
                                    tool_parser.streamed_args_for_tool[repair_index]
                                    if repair_index
                                    < len(tool_parser.streamed_args_for_tool)
                                    else ""
                                )
                                latest_delta = delta_args_by_index.get(
                                    repair_index, ""
                                )
                                if latest_delta and actual_call.endswith(
                                    latest_delta
                                ):
                                    actual_call = actual_call[: -len(latest_delta)]

                                # Only send a final repair delta when the
                                # complete parsed arguments extend what was
                                # actually streamed. Checking every tool index
                                # matters for XML tool streams where an earlier
                                # call can emit "{" and then never receive its
                                # remaining argument tail.
                                remaining_call = (
                                    expected_call[len(actual_call) :]
                                    if expected_call.startswith(actual_call)
                                    else ""
                                )
                                if remaining_call:
                                    original_delta = delta_by_index.get(repair_index)
                                    original_fn = (
                                        original_delta.function
                                        if original_delta
                                        and original_delta.function is not None
                                        else None
                                    )
                                    repair_tool_calls.append(
                                        DeltaToolCall(
                                            index=repair_index,
                                            id=(
                                                original_delta.id
                                                if original_delta
                                                and original_delta.id is not None
                                                else make_tool_call_id()
                                            ),
                                            type=(
                                                original_delta.type
                                                if original_delta
                                                and original_delta.type is not None
                                                else "function"
                                            ),
                                            function=DeltaFunctionCall(
                                                name=(
                                                    original_fn.name
                                                    if original_fn
                                                    and original_fn.name is not None
                                                    else full_tool_call.function.name
                                                ),
                                                arguments=remaining_call
                                            ),
                                        )
                                    )

                            if repair_tool_calls:
                                delta_message = DeltaMessage(
                                    tool_calls=repair_tool_calls
                                )

                        if (
                            tool_parser
                            and auto_tools_called
                            and delta_message
                            and delta_message.tool_calls
                        ):
                            full_tool_info = tool_parser.extract_tool_calls(
                                current_text, request
                            )
                            full_tool_calls = full_tool_info.tool_calls
                            for tool_delta in delta_message.tool_calls:
                                delta_index = tool_delta.index
                                if delta_index is None or delta_index >= len(
                                    full_tool_calls
                                ):
                                    continue
                                if tool_delta.function is None:
                                    continue
                                expected_args = full_tool_calls[
                                    delta_index
                                ].function.arguments
                                if not isinstance(expected_args, str):
                                    expected_args = json.dumps(
                                        expected_args, ensure_ascii=False
                                    )
                                current_args = tool_delta.function.arguments or ""
                                if (
                                    expected_args
                                    and expected_args != "{}"
                                    and current_args in ("", "{}")
                                ):
                                    tool_delta.function.arguments = expected_args

                        if (
                            thinking_disabled
                            and not request.tools
                            and not streamed_content_texts[i]
                            and current_text
                        ):
                            delta_message.content = current_text
                            delta_message.reasoning = None

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
                        else:
                            finish_reason_ = (
                                output.finish_reason if output.finish_reason else "stop"
                            )
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

                    if choice_data.delta.tool_calls:
                        for tool_delta in choice_data.delta.tool_calls:
                            tool_index = tool_delta.index or 0
                            state = streamed_tool_calls[i].setdefault(
                                tool_index,
                                {
                                    "id": tool_delta.id,
                                    "type": tool_delta.type or "function",
                                    "function": {"name": None, "arguments": ""},
                                },
                            )
                            if tool_delta.id:
                                state["id"] = tool_delta.id
                            if tool_delta.type:
                                state["type"] = tool_delta.type
                            if tool_delta.function is not None:
                                if tool_delta.function.name:
                                    state["function"]["name"] = (
                                        tool_delta.function.name
                                    )
                                if tool_delta.function.arguments:
                                    state["function"]["arguments"] += (
                                        tool_delta.function.arguments
                                    )
                    if choice_data.finish_reason is not None:
                        final_finish_reasons[i] = choice_data.finish_reason
                    choice_data = maybe_filter_parallel_tool_calls(choice_data, request)
                    if request.tools:
                        if choice_data.delta.tool_calls:
                            pending_tool_whitespace_content[i] = ""
                            if (
                                choice_data.delta.content
                                and not choice_data.delta.content.strip()
                            ):
                                choice_data.delta.content = None
                        elif choice_data.delta.content:
                            if not choice_data.delta.content.strip():
                                pending_tool_whitespace_content[i] += (
                                    choice_data.delta.content
                                )
                                choice_data.delta.content = None
                            elif pending_tool_whitespace_content[i]:
                                choice_data.delta.content = (
                                    pending_tool_whitespace_content[i]
                                    + choice_data.delta.content
                                )
                                pending_tool_whitespace_content[i] = ""
                        elif (
                            choice_data.finish_reason is not None
                            and pending_tool_whitespace_content[i]
                            and not tools_streamed[i]
                        ):
                            choice_data.delta.content = (
                                pending_tool_whitespace_content[i]
                            )
                            pending_tool_whitespace_content[i] = ""
                    if choice_data.delta.content:
                        streamed_content_texts[i] += choice_data.delta.content
                    if (
                        choice_data.delta.reasoning
                        and request.include_reasoning
                    ):
                        streamed_reasoning_texts[i] += choice_data.delta.reasoning
                    chunk = ChatCompletionStreamResponse(
                        id=request_id,
                        object=chunk_object_type,
                        created=created_time,
                        choices=[choice_data],
                        model=model_name,
                    )
                    # Stamp the fingerprint on terminal chunks only (those with
                    # finish_reason set). When ``include_usage`` is on, the
                    # trailing usage chunk below overrides this as the true
                    # final message.
                    if (
                        not include_usage
                        and self.system_fingerprint is not None
                        and choice_data.finish_reason is not None
                    ):
                        chunk.system_fingerprint = self.system_fingerprint

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
                    system_fingerprint=self.system_fingerprint,
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

            choices_list = []
            for i in range(num_choices):
                tool_calls = [
                    streamed_tool_calls[i][idx]
                    for idx in sorted(streamed_tool_calls[i])
                    if streamed_tool_calls[i][idx].get("function", {}).get("name")
                    or streamed_tool_calls[i][idx].get("function", {}).get(
                        "arguments"
                    )
                ]
                message: dict[str, Any] = {
                    "role": self.get_chat_request_role(request),
                }
                if streamed_content_texts[i] or not tool_calls:
                    message["content"] = streamed_content_texts[i] or ""
                if streamed_reasoning_texts[i]:
                    message["reasoning_content"] = streamed_reasoning_texts[i]
                if tool_calls:
                    message["tool_calls"] = tool_calls
                choices_list.append(
                    {
                        "index": i,
                        "message": message,
                        "finish_reason": final_finish_reasons[i] or "stop",
                    }
                )
            self._log_chat_response_payload(
                request_id[len("chatcmpl-") :]
                if request_id.startswith("chatcmpl-")
                else request_id,
                {
                    "id": request_id,
                    "object": "chat.completion",
                    "created": created_time,
                    "model": model_name,
                    "choices": choices_list,
                    "usage": request_metadata.final_usage_info.model_dump(),
                    "stream": True,
                },
            )

            # Log complete streaming response if output logging is enabled
            if self.enable_log_outputs and self.request_logger:
                # Log the complete response for each choice
                for i in range(num_choices):
                    full_text = (
                        previous_texts[i]
                        if previous_texts and i < len(previous_texts)
                        else f"<streaming_complete: {previous_num_tokens[i]} tokens>"
                    )
                    self.request_logger.log_outputs(
                        request_id=request_id,
                        outputs=full_text,
                        output_token_ids=None,  # Consider also logging all token IDs
                        finish_reason="streaming_complete",
                        is_streaming=True,
                        delta=False,
                    )

        except (asyncio.CancelledError, GeneratorExit):
            record_aborted_request()
            raise
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
            _log_raw_chat_generation_debug(
                request_id,
                "full_output",
                output.index,
                tokenizer,
                as_list(token_ids),
                output.text,
            )
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

                    tool_parser = self.tool_parser(tokenizer, request.tools)
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

                # Encode routed_experts for transport. JSON can't carry raw
                # bytes, so we write the ndarray as a ``.npy`` byte stream
                # and base64-encode it. ``pybase64`` is ~3x faster than the
                # stdlib ``base64`` on large payloads thanks to SIMD.
                routed_experts_b64 = None
                if output.routed_experts is not None:
                    buf = io.BytesIO()
                    np.save(buf, output.routed_experts)
                    routed_experts_b64 = base64.b64encode(buf.getvalue()).decode(
                        "ascii"
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
                    routed_experts=routed_experts_b64,
                )
                choices.append(choice_data)
                continue

            if reasoning_parser:
                # If the reasoning parser is enabled,
                # tool calls are extracted exclusively from the content.
                if request.reasoning_effort == "none":
                    reasoning = None
                    content = output.text
                else:
                    reasoning, content = reasoning_parser.extract_reasoning(
                        output.text, request=request
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
            if is_mistral_tokenizer(tokenizer):
                from vllm.tool_parsers.mistral_tool_parser import MistralToolCall

                tool_call_class: type[ToolCall] = MistralToolCall
            else:
                tool_call_class = ToolCall

            use_mistral_tool_parser = request._grammar_from_tool_parser
            if use_mistral_tool_parser:
                from vllm.tool_parsers.mistral_tool_parser import MistralToolParser

                tool_call_items = MistralToolParser.build_non_streaming_tool_calls(
                    tool_calls
                )
                if tool_call_items:
                    auto_tools_called = (
                        request.tool_choice is None or request.tool_choice == "auto"
                    )
                message = ChatMessage(
                    role=role,
                    reasoning=reasoning,
                    content=content,
                    tool_calls=tool_call_items,
                )

            elif (not self.enable_auto_tools or not self.tool_parser) and (
                not isinstance(request.tool_choice, ChatCompletionNamedToolChoiceParam)
                and request.tool_choice != "required"
            ):
                message = ChatMessage(role=role, reasoning=reasoning, content=content)

            elif (
                request.tool_choice
                and type(request.tool_choice) is ChatCompletionNamedToolChoiceParam
            ):
                tool_call_class_items = []
                tool_calls = tool_calls or []
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
                        if is_mistral_tokenizer(tokenizer):
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
                        if is_mistral_tokenizer(tokenizer):
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
                            if is_mistral_tokenizer(tokenizer):
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

            # Encode routed_experts for transport. JSON can't carry raw
            # bytes, so we write the ndarray as a ``.npy`` byte stream
            # and base64-encode it. ``pybase64`` is ~3x faster than the
            # stdlib ``base64`` on large payloads thanks to SIMD.
            routed_experts_b64 = None
            if output.routed_experts is not None:
                buf = io.BytesIO()
                np.save(buf, output.routed_experts)
                routed_experts_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

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
                routed_experts=routed_experts_b64,
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

        # ``final_res.prompt`` is the rendered chat-templated prompt text
        prompt_text = final_res.prompt if request.return_prompt_text else None

        response = ChatCompletionResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=choices,
            usage=usage,
            system_fingerprint=self.system_fingerprint,
            prompt_logprobs=clamp_prompt_logprobs(final_res.prompt_logprobs),
            prompt_token_ids=(
                final_res.prompt_token_ids if request.return_token_ids else None
            ),
            prompt_text=prompt_text,
            kv_transfer_params=final_res.kv_transfer_params,
        )
        self._log_chat_response_payload(
            request_id[len("chatcmpl-") :]
            if request_id.startswith("chatcmpl-")
            else request_id,
            response.model_dump(),
        )

        # Log complete response if output logging is enabled
        if self.enable_log_outputs and self.request_logger:
            for choice in choices:
                output_text = ""
                if choice.message.content:
                    output_text = choice.message.content
                elif choice.message.tool_calls:
                    # For tool calls, log the function name and arguments
                    tool_call_descriptions = []
                    for tc in choice.message.tool_calls:  # type: ignore
                        function_call: FunctionCall = tc.function  # type: ignore
                        tool_call_descriptions.append(
                            f"{function_call.name}({function_call.arguments})"
                        )
                    tool_calls_str = ", ".join(tool_call_descriptions)
                    output_text = f"[tool_calls: {tool_calls_str}]"

                if output_text:
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
    ) -> DeltaMessage:
        """
        Create a delta message for remaining tool arguments, preserving
        id/type/name from the original delta.
        """
        original_tc = next(
            (tc for tc in delta_message.tool_calls if tc.index == index),
            None,
        )
        original_fn = original_tc.function if original_tc else None
        return DeltaMessage(
            tool_calls=[
                DeltaToolCall(
                    index=index,
                    id=original_tc.id if original_tc else None,
                    type=original_tc.type if original_tc else None,
                    function=DeltaFunctionCall(
                        name=original_fn.name if original_fn else None,
                        arguments=remaining_call,
                    ),
                )
            ]
        )
