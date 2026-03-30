# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json
import logging
import os
import time
from collections.abc import AsyncGenerator, AsyncIterator
from collections.abc import Sequence as GenericSequence
from http import HTTPStatus
from typing import TYPE_CHECKING, Any, Final

import partial_json_parser
import regex as re
import base64
from pathlib import Path
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
from vllm.payload_sanitization import prepare_request_payload_for_logging
from vllm.parser import ParserManager
from vllm.reasoning import ReasoningParser
from vllm.renderers import ChatParams
from vllm.sampling_params import BeamSearchParams, SamplingParams
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers import ToolParser
from vllm.tool_parsers.mistral_tool_parser import MistralToolCall
from vllm.tool_parsers.utils import partial_json_loads
from vllm.request_context import reset_request_id, set_request_id
from vllm.utils.collection_utils import as_list
from vllm.utils.mistral import is_mistral_tokenizer

if TYPE_CHECKING:
    from vllm.entrypoints.serve.render.serving import OpenAIServingRender

from vllm.entrypoints.openai.request_metrics import (
    classify_chat_request,
    record_aborted_request,
)

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
            request.chat_template_kwargs,
            self.default_chat_template_kwargs,
        )
        reasoning_budget = getattr(request, "reasoning_budget", None)
        if reasoning_budget is None:
            reasoning_budget = chat_template_kwargs.get("reasoning_budget")
        if reasoning_budget is None:
            return

        try:
            budget_int = int(reasoning_budget)
        except Exception:
            logger.warning("Invalid reasoning_budget=%r; skipping", reasoning_budget)
            return

        if budget_int == -1:
            return

        if sampling_params.extra_args is None:
            sampling_params.extra_args = {}
        extra = sampling_params.extra_args

        extra.setdefault("reasoning_budget", budget_int)

        grace = getattr(request, "reasoning_budget_grace_period", None)
        if grace is None:
            grace = chat_template_kwargs.get("reasoning_budget_grace_period", 0)
        try:
            grace_int = int(grace)
        except Exception:
            grace_int = 0
        extra.setdefault("reasoning_budget_grace_period", grace_int)

        if "enable_thinking" in chat_template_kwargs:
            extra.setdefault("enable_thinking", chat_template_kwargs["enable_thinking"])

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

        rid_hint = self._base_request_id(raw_request, getattr(request, "request_id", None))
        ctx_token = set_request_id(rid_hint)
        try:
            # Log request payload BEFORE any chat template is applied
            if os.getenv("VLLM_LOG_PAYLOADS", "1") == "1":
                headers_obj = None
                try:
                    if raw_request is not None:
                        headers_obj = {k: v for k, v in raw_request.headers.items()}
                except Exception:
                    headers_obj = None
                try:
                    req_dump = request.model_dump(mode="json")
                except Exception:
                    req_dump = None
                try:
                    allowed_local_media_path = (
                        self.openai_serving_render.model_config.allowed_local_media_path
                    )
                except Exception:
                    allowed_local_media_path = ""
                try:
                    payload_logger.info(
                        "openai.request",
                        extra={
                            "rid": rid_hint or "",
                            "endpoint": self.__class__.__name__,
                            # Pass dict directly for proper OTEL structured logging
                            "payload": prepare_request_payload_for_logging(
                                req_dump,
                                headers=headers_obj,
                                allowed_local_media_path=allowed_local_media_path,
                            ),
                            "headers": headers_obj,
                        },
                    )
                except Exception:
                    logger.exception("Failed to log openai.request rid=%s", rid_hint)

            # Resolve NVCF assets in-place before preprocessing.
            # This happens after request payload logging so we preserve what
            # the caller actually sent (asset_id refs, base64, etc.).
            if raw_request is not None:
                try:
                    request.messages = self._resolve_nvcf_image_assets(
                        request.messages, raw_request
                    )
                except Exception as e:
                    logger.exception("Error while resolving NVCF assets")
                    return self.create_error_response(str(e))

            # Strip raw multimodal special tokens from plain text to avoid
            # backend crashes when no corresponding media is provided.
            try:
                request.messages = self._strip_mm_special_tokens_in_messages(
                    request.messages
                )
            except Exception:
                logger.exception("Error while stripping MM special tokens")

            tokenizer = self.renderer.tokenizer
            tool_parser = self.tool_parser

            # For gpt-oss (harmony) models, special tokens are part of the
            # protocol framing. By default, OpenAI-compatible requests set
            # `skip_special_tokens=True`, which can strip these markers from
            # the streamed text. If the caller didn't explicitly set this
            # field, default to keeping special tokens for harmony models.
            if (
                self.use_harmony
                and "skip_special_tokens" not in request.model_fields_set
            ):
                request.skip_special_tokens = False
            return await self.openai_serving_render.render_chat(request)
        except RuntimeError as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(e)
        finally:
            try:
                reset_request_id(ctx_token)
            except Exception:
                pass

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
        if self.reasoning_parser_cls:
            try:
                # Pass the same chat template kwargs as used in tokenization
                chat_template_kwargs = self._prepare_extra_chat_template_kwargs(
                    request.chat_template_kwargs,
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
    def _parse_partial_required_tool_list(text: str | None) -> list[dict[str, Any]]:
        if not text:
            return []
        try:
            flags = Allow.ALL
            obj, _ = partial_json_loads(text, flags)
        except (
            partial_json_parser.core.exceptions.MalformedJSON,
            json.JSONDecodeError,
        ):
            return []
        if not isinstance(obj, list):
            return []
        return [tool for tool in obj if isinstance(tool, dict)]

    @staticmethod
    def _required_tool_arguments_json(tool_call: dict[str, Any] | None) -> str:
        if not isinstance(tool_call, dict):
            return ""
        params = tool_call.get("parameters")
        if params is None:
            return ""
        try:
            return json.dumps(params, ensure_ascii=False)
        except TypeError:
            return ""

    @staticmethod
    def _required_emitted_tool_state(
        emitted_tool_calls: list[dict[str, Any]] | None,
        index: int,
    ) -> tuple[str | None, str, str | None, str | None]:
        if emitted_tool_calls is None or index >= len(emitted_tool_calls):
            return None, "", None, None
        tool_call = emitted_tool_calls[index]
        if not isinstance(tool_call, dict):
            return None, "", None, None
        function = tool_call.get("function", {})
        if not isinstance(function, dict):
            function = {}
        name = function.get("name")
        arguments = function.get("arguments", "")
        if not isinstance(arguments, str):
            arguments = ""
        tool_id = tool_call.get("id")
        tool_type = tool_call.get("type")
        return (
            name if isinstance(name, str) else None,
            arguments,
            tool_id if isinstance(tool_id, str) else None,
            tool_type if isinstance(tool_type, str) else None,
        )

    @staticmethod
    def _accumulate_stream_delta_state(
        *,
        delta_message: DeltaMessage | None,
        previous_reasoning_texts: list[str],
        previous_content_texts: list[str],
        previous_tool_calls: list[list[dict[str, Any]]],
        index: int,
    ) -> None:
        if delta_message is None:
            return

        reasoning_delta = (
            delta_message.reasoning_content
            if delta_message.reasoning_content is not None
            else delta_message.reasoning
        )
        if reasoning_delta:
            previous_reasoning_texts[index] += reasoning_delta
        if delta_message.content:
            previous_content_texts[index] += delta_message.content
        if not delta_message.tool_calls:
            return

        for delta_tc in delta_message.tool_calls:
            tc_idx = delta_tc.index if delta_tc.index is not None else 0
            while len(previous_tool_calls[index]) <= tc_idx:
                previous_tool_calls[index].append(
                    {
                        "id": None,
                        "type": "function",
                        "function": {
                            "name": None,
                            "arguments": "",
                        },
                    }
                )

            tool_state = previous_tool_calls[index][tc_idx]
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

    def _force_nonempty_content_enabled(self, request: ChatCompletionRequest) -> bool:
        chat_template_kwargs = self._prepare_extra_chat_template_kwargs(
            request.chat_template_kwargs,
            self.default_chat_template_kwargs,
        )
        return chat_template_kwargs.get("force_nonempty_content") is True

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
            # If tool calls were already streamed, the regular finish path has
            # already flushed any missing argument suffixes. Do not synthesize
            # them a second time here or we will duplicate JSON arguments.
            if previous_tool_calls:
                return None, None, None
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
                        func_name=parsed_call.name,
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
        emitted_tool_calls: list[dict[str, Any]] | None = None,
    ) -> tuple[DeltaMessage | None, bool]:
        if current_text is None or current_text == "":
            # if the current text is empty, we cannot parse it
            return None, function_name_returned

        current_tools = OpenAIServingChat._parse_partial_required_tool_list(
            current_text
        )
        if not current_tools:
            return None, False

        tool_deltas: list[DeltaToolCall] = []
        next_tool_call_idx = tool_call_idx or 0

        for idx, current_tool_call in enumerate(current_tools):
            current_name = current_tool_call.get("name")
            current_name = current_name if isinstance(current_name, str) else None
            current_arguments = OpenAIServingChat._required_tool_arguments_json(
                current_tool_call
            )
            emitted_name, emitted_arguments, _, _ = (
                OpenAIServingChat._required_emitted_tool_state(
                    emitted_tool_calls, idx
                )
            )
            is_last_tool = idx == len(current_tools) - 1
            list_complete = current_text.rstrip().endswith("]")

            # Emit the header as soon as a new tool object appears.
            if emitted_name is None:
                if not current_name:
                    continue
                if is_last_tool and not current_arguments and not list_complete:
                    continue
                tool_deltas.append(
                    DeltaToolCall(
                        id=make_tool_call_id(
                            id_type=self.tool_call_id_type,
                            func_name=current_name,
                            idx=next_tool_call_idx,
                        ),
                        function=DeltaFunctionCall(
                            name=current_name, arguments=""
                        ),
                        index=idx,
                        type="function",
                    )
                )
                next_tool_call_idx += 1
            # Defer arguments for the current last tool until either a later
            # tool appears or the finish path flushes the final state. This
            # avoids emitting synthetic closed JSON for still-growing objects.
            if is_last_tool:
                continue

            if current_arguments == emitted_arguments:
                continue

            if (
                emitted_arguments
                and current_arguments.startswith(emitted_arguments)
            ):
                delta_arguments = current_arguments[len(emitted_arguments) :]
            else:
                delta_arguments = current_arguments

            if not delta_arguments:
                continue

            tool_deltas.append(
                DeltaToolCall(
                    function=DeltaFunctionCall(
                        # OpenAI API returns None instead of the name on
                        # argument continuation chunks.
                        name=None,
                        arguments=delta_arguments,
                    ),
                    index=idx,
                )
            )

        if not tool_deltas:
            return None, bool(current_tools)

        return DeltaMessage(tool_calls=tool_deltas), bool(current_tools)

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
        if self.tool_call_id_type == "kimi_k2":
            history_tool_call_cnt = get_history_tool_calls_cnt(conversation)
        else:
            history_tool_call_cnt = 0

        # Track full streamed state so we can emit one final structured
        # response log without re-enabling token-by-token logging.
        previous_texts = [""] * num_choices
        previous_reasoning_texts = [""] * num_choices
        previous_content_texts = [""] * num_choices
        previous_tool_calls: list[list[dict[str, Any]]] = [
            [] for _ in range(num_choices)
        ]
        final_finish_reasons: list[str | None] = [None] * num_choices

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

        # Prepare the tool parser if it's needed
        try:
            if tool_choice_auto and self.tool_parser:
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
        chat_template_kwargs = self._prepare_extra_chat_template_kwargs(
            request.chat_template_kwargs,
            self.default_chat_template_kwargs,
        )
        # force_nonempty_content still needs reasoning parsing in streaming so
        # we can strip markers and split the </think> boundary correctly.
        bypass_reasoning_in_stream = bool(
            chat_template_kwargs.get("enable_thinking") is False
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
                            True
                            if bypass_reasoning_in_stream
                            else reasoning_parser.is_reasoning_end(
                                res.prompt_token_ids
                            )
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
                                else:
                                    current_text = ""
                        else:
                            # Just to add remaining `content`
                            if reasoning_parser:
                                delta_text = previous_text + delta_text
                                current_text = ""

                            if function_name_returned[i]:
                                delta_tool_call = DeltaToolCall(
                                    function=DeltaFunctionCall(arguments=delta_text),
                                    index=i,
                                )
                            else:
                                # Generate ID based on tokenizer type
                                if is_mistral_tokenizer(tokenizer):
                                    tool_call_id = MistralToolCall.generate_random_id()
                                else:
                                    tool_call_id = make_tool_call_id(
                                        id_type=self.tool_call_id_type,
                                        func_name=tool_choice_function_name,
                                        idx=history_tool_call_cnt,
                                    )
                                delta_tool_call = DeltaToolCall(
                                    id=tool_call_id,
                                    type="function",
                                    function=DeltaFunctionCall(
                                        name=tool_choice_function_name,
                                        arguments=delta_text,
                                    ),
                                    index=i,
                                )
                                function_name_returned[i] = True
                                history_tool_call_cnt += 1

                            delta_message = DeltaMessage(
                                tool_calls=[
                                    delta_tool_call,
                                ]
                            )
                            tools_streamed[i] = True

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

                        else:
                            # either finished reasoning or no reasoning at all
                            content = current_text

                            delta_message, function_name_returned[i] = (
                                self.extract_tool_call_required_streaming(
                                    previous_text=previous_text,
                                    current_text=content,
                                    delta_text=delta_text,
                                    function_name_returned=fn_name_returned,
                                    tool_call_idx=history_tool_call_cnt,
                                    emitted_tool_calls=previous_tool_calls[i],
                                )
                            )
                            if delta_message and delta_message.tool_calls:
                                history_tool_call_cnt += sum(
                                    1
                                    for tc in delta_message.tool_calls
                                    if tc.id is not None
                                )
                                tools_streamed[i] = True

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
                        reasoning_parser is not None
                        and delta_message.content is not None
                        and not previous_content_texts[i]
                    ):
                        trimmed_content = delta_message.content
                        removed = 0
                        while removed < 2 and trimmed_content.startswith("\n"):
                            trimmed_content = trimmed_content[1:]
                            removed += 1
                        delta_message.content = trimmed_content or None

                    # Do not surface a standalone whitespace-only content delta
                    # on the first tool-call chunk. Non-streaming responses do
                    # not expose this boundary artifact, and dropping it does
                    # not change any semantic content returned to clients.
                    if (
                        delta_message.tool_calls
                        and delta_message.content is not None
                        and delta_message.content.strip() == ""
                        and not previous_content_texts[i]
                    ):
                        delta_message.content = None

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

                        if tool_parser and auto_tools_called:
                            if (
                                delta_message is None
                                or not delta_message.tool_calls
                            ):
                                # Some tool parsers finish with an empty/content
                                # delta after having already streamed most of the
                                # tool arguments. In that case we still need to
                                # compare the parser's final expected arguments
                                # against what was streamed so far and emit the
                                # missing suffix (commonly the final "}").
                                delta_message = DeltaMessage(
                                    tool_calls=[
                                        DeltaToolCall(
                                            index=index,
                                            function=DeltaFunctionCall(arguments=""),
                                        )
                                    ]
                                )

                            if self._should_check_for_unstreamed_tool_arg_tokens(
                                delta_message, output
                            ):
                                latest_delta_len = 0
                                if (
                                    isinstance(
                                        delta_message.tool_calls[0].function,
                                        DeltaFunctionCall,
                                    )
                                ) and isinstance(
                                    delta_message.tool_calls[0].function.arguments,
                                    str,
                                ):
                                    latest_delta_len = len(
                                        delta_message.tool_calls[0].function.arguments
                                    )

                            # get the expected call based on partial JSON
                            # parsing which "autocompletes" the JSON.
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
                                actual_call_full = (
                                    tool_parser.streamed_args_for_tool[index]
                                )
                                # Some tool parsers initialize their internal
                                # state with a placeholder "{}" and only replace it
                                # once they fully parse the closing tag. If that
                                # replacement never happens but we already streamed
                                # concrete argument text, prefer the streamed text
                                # over the placeholder to avoid appending a bogus
                                # trailing "{}" on the finish chunk.
                                if (
                                    expected_call == "{}"
                                    and actual_call_full
                                    and actual_call_full != "{}"
                                ):
                                    expected_call = actual_call_full

                                actual_call = actual_call_full
                                if latest_delta_len > 0:
                                    actual_call = actual_call[:-latest_delta_len]

                            # Only synthesize a final arguments delta when the
                            # expected full arguments are a clean extension of
                            # what we had already streamed before this chunk.
                            # If they diverge, keep the original delta rather
                            # than appending junk such as a placeholder "{}".
                                if expected_call.startswith(actual_call):
                                    remaining_call = expected_call[
                                        len(actual_call) :
                                    ]
                                    delta_message = self._create_remaining_args_delta(
                                        delta_message, remaining_call, index
                                    )
                        elif request.tool_choice == "required":
                            final_required_tools = (
                                self._parse_partial_required_tool_list(current_text)
                            )
                            required_tool_deltas: list[DeltaToolCall] = []
                            for tc_idx, final_tool_call in enumerate(final_required_tools):
                                expected_arguments = (
                                    self._required_tool_arguments_json(final_tool_call)
                                )
                                (
                                    emitted_name,
                                    emitted_arguments,
                                    emitted_id,
                                    emitted_type,
                                ) = self._required_emitted_tool_state(
                                    previous_tool_calls[i], tc_idx
                                )
                                if not expected_arguments:
                                    continue
                                if expected_arguments == emitted_arguments:
                                    continue
                                if (
                                    emitted_arguments
                                    and expected_arguments.startswith(emitted_arguments)
                                ):
                                    remaining_call = expected_arguments[
                                        len(emitted_arguments) :
                                    ]
                                else:
                                    remaining_call = expected_arguments
                                if not remaining_call:
                                    continue
                                final_name = final_tool_call.get("name")
                                final_name = (
                                    final_name if isinstance(final_name, str) else None
                                )
                                required_tool_deltas.append(
                                    DeltaToolCall(
                                        index=tc_idx,
                                        id=(
                                            emitted_id
                                            or make_tool_call_id(
                                                id_type=self.tool_call_id_type,
                                                func_name=final_name,
                                                idx=tc_idx,
                                            )
                                        ),
                                        type=emitted_type or "function",
                                        function=DeltaFunctionCall(
                                            name=(
                                                None
                                                if emitted_name is not None
                                                else final_name
                                            ),
                                            arguments=remaining_call,
                                        ),
                                    )
                                )
                            if required_tool_deltas:
                                delta_message = DeltaMessage(
                                    tool_calls=required_tool_deltas
                                )

                        force_nonempty_finish_reason: str | None = None
                        forced_tool_states: list[dict[str, Any]] | None = None
                        (
                            forced_delta,
                            force_nonempty_finish_reason,
                            forced_tool_states,
                        ) = self._build_force_nonempty_stream_finish_delta(
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
                        final_finish_reasons[i] = finish_reason_

                    # Accumulate the final streamed response only after any
                    # finish-time parser/tool-call fixups have run.
                    self._accumulate_stream_delta_state(
                        delta_message=delta_message,
                        previous_reasoning_texts=previous_reasoning_texts,
                        previous_content_texts=previous_content_texts,
                        previous_tool_calls=previous_tool_calls,
                        index=i,
                    )

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
                            else "",
                        },
                        "finish_reason": final_finish_reasons[i] or "stop",
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
                    choices_list.append(choice_data)

                rid_hint = (
                    request_id[len("chatcmpl-") :]
                    if request_id.startswith("chatcmpl-")
                    else request_id
                )
                resp_summary = {
                    "id": request_id,
                    "object": "chat.completion",
                    "created": created_time,
                    "model": model_name,
                    "choices": choices_list,
                    "usage": usage_dict,
                    "stream": True,
                }
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
            await self.engine_client.abort(request_id)
            record_aborted_request()
            logger.info("Streaming request %s cancelled by client disconnect", request_id)
            return
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

    def _strip_mm_special_tokens_in_messages(self, messages: list[dict]) -> list[dict]:
        """
        Remove raw multimodal special tokens typed inside text content.

        This prevents crashes when users include tokens like '<|image|>' or
        '<image>' without supplying corresponding multimodal data. Structured
        content parts (e.g., {'type': 'image_url', ...}) are preserved.
        """
        token_pattern = (
            r"(?i)"  # case-insensitive
            r"(<\|begin_of_image\|>|<\|image_start\|>|<\|image_end\|>|"
            r"<\|image\|>|<\|audio\|>|<\|video\|>|<image>|</image>|"
            r"<audio>|</audio>|<video>|</video>|<\|patch\|>)"
        )
        mm_token_re = re.compile(token_pattern)

        def clean_text(text: str) -> str:
            cleaned = mm_token_re.sub(" ", text)
            return re.sub(r"\s+", " ", cleaned).strip()

        def clean_message(msg: dict) -> dict:
            if not isinstance(msg, dict):
                return msg
            new_msg = dict(msg)
            content = new_msg.get("content")
            if isinstance(content, str):
                new_msg["content"] = clean_text(content)
            elif isinstance(content, list):
                new_parts = []
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text" and isinstance(
                                part.get("text"), str):
                            new_part = dict(part)
                            new_part["text"] = clean_text(part["text"])
                            new_parts.append(new_part)
                        else:
                            new_parts.append(part)
                    else:
                        new_parts.append(part)
                new_msg["content"] = new_parts
            return new_msg

        return [clean_message(m) for m in messages]

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

    def _resolve_nvcf_image_assets(
        self, messages: list[dict], raw_request: Request
    ) -> list[dict]:
        """
        Convert any NVCF image/video asset references to base64 data URLs and,
        when present inside plain text content with <img>/<video> tags,
        transform the message into structured parts to ensure media loading by
        the multimodal parser.

        Supported inputs:
        - Structured image: {"type":"image_url", "image_url":{"url":"data:image/...;asset_id,<id>"}}
        - Structured video: {"type":"video_url", "video_url":{"url":"data:video/...;asset_id,'<id>'"}}
        - Text with HTML: "... <img src=\"data:image/...;asset_id,<id>\"/> ..."
        - Text with HTML: "... <video src=\"data:video/mp4;asset_id,'<id>'\"/> ..."
        Headers used:
        - NVCF-ASSET-DIR: absolute directory containing assets
        - NVCF-FUNCTION-ASSET-IDS: comma-separated allowed asset ids
        """
        headers = raw_request.headers
        asset_dir = headers.get("NVCF-ASSET-DIR")
        allowed_ids_hdr = headers.get("NVCF-FUNCTION-ASSET-IDS")

        if not asset_dir or not allowed_ids_hdr:
            # Nothing to resolve
            return messages

        asset_root = Path(asset_dir)
        if not asset_root.exists() or not asset_root.is_dir():
            raise ValueError(f"Invalid NVCF-ASSET-DIR: {asset_dir}")

        def normalize_asset_id(val: str) -> str:
            v = (val or "").strip().strip(",").strip()
            # NVCF sometimes wraps asset ids in quotes (e.g., asset_id,'abc').
            while len(v) >= 2 and v[0] in ("'", '"') and v[-1] == v[0]:
                v = v[1:-1].strip()
            return v

        allowed_ids = {normalize_asset_id(s) for s in allowed_ids_hdr.split(',') if s.strip()}

        def to_base64_data_url(data_url: str) -> str:
            # data:<mime>;asset_id,<id> (images/videos)
            m = re.match(
                r"^data:(?P<mime>(?:image|video)/[^;]+);asset_id,(?P<asset_id>.+)$",
                data_url,
            )
            if not m:
                return data_url
            mime = m.group("mime")
            asset_id = normalize_asset_id(m.group("asset_id"))
            if asset_id not in allowed_ids:
                raise ValueError(f"Asset id '{asset_id}' not permitted by NVCF-FUNCTION-ASSET-IDS")
            file_path = (asset_root / asset_id).resolve()
            # prevent traversal
            if asset_root not in file_path.parents and file_path != asset_root:
                raise ValueError("Asset path escapes NVCF-ASSET-DIR")
            with open(file_path, 'rb') as f:
                raw = f.read()

            # Emit a durable mapping for short-lived NVCF assets (async).
            try:
                from vllm.request_context import get_request_id
                from vllm.otel_instrumentation import enqueue_media_mirror

                rid = get_request_id() or ""
                kind = "video" if mime.startswith("video/") else "image"
                enqueue_media_mirror(
                    rid=rid,
                    kind=kind,
                    original=f"asset_id:{asset_id}",
                    data=raw,
                    mime=mime,
                    source="nvcf_asset",
                )
            except Exception:
                pass

            data_b64 = base64.b64encode(raw).decode('ascii')
            return f"data:{mime};base64,{data_b64}"

        def transform_message(msg: dict) -> dict:
            content = msg.get("content")
            # Case 1: structured content list
            if isinstance(content, list):
                new_parts = []
                for part in content:
                    if not isinstance(part, dict):
                        new_parts.append(part)
                        continue

                    if "image_url" in part:
                        url_obj = part["image_url"]
                        if isinstance(url_obj, dict):
                            url = url_obj.get("url")
                            if isinstance(url, str) and ";asset_id," in url:
                                url_obj["url"] = to_base64_data_url(url)
                        elif isinstance(url_obj, str) and ";asset_id," in url_obj:
                            part["image_url"] = {"url": to_base64_data_url(url_obj)}
                        new_parts.append(part)
                        continue

                    if "video_url" in part:
                        url_obj = part["video_url"]
                        if isinstance(url_obj, dict):
                            url = url_obj.get("url")
                            if isinstance(url, str) and ";asset_id," in url:
                                url_obj["url"] = to_base64_data_url(url)
                        elif isinstance(url_obj, str) and ";asset_id," in url_obj:
                            part["video_url"] = {"url": to_base64_data_url(url_obj)}
                        new_parts.append(part)
                        continue

                    new_parts.append(part)
                msg["content"] = new_parts
                return msg

            # Case 2: plain text possibly containing <img>/<video> tags
            if isinstance(content, str):
                pattern = re.compile(
                    r"<(?P<tag>img|video)\s+[^>]*src=\"(?P<src>[^\"]+)\"[^>]*\/?>",
                    re.IGNORECASE,
                )
                idx = 0
                parts = []
                for m in pattern.finditer(content):
                    start, end = m.span()
                    tag = (m.group("tag") or "").lower()
                    url = m.group("src")
                    if start > idx:
                        text_chunk = content[idx:start]
                        if text_chunk:
                            parts.append({"type": "text", "text": text_chunk})
                    if isinstance(url, str) and ";asset_id," in url:
                        if tag == "img" and url.startswith("data:image/"):
                            b64_url = to_base64_data_url(url)
                            parts.append({"type": "image_url", "image_url": {"url": b64_url}})
                        elif tag == "video" and url.startswith("data:video/"):
                            b64_url = to_base64_data_url(url)
                            parts.append({"type": "video_url", "video_url": {"url": b64_url}})
                        else:
                            parts.append({"type": "text", "text": m.group(0)})
                    else:
                        # keep as text if not asset_id pattern
                        parts.append({"type": "text", "text": m.group(0)})
                    idx = end
                if parts:
                    # trailing text
                    if idx < len(content):
                        tail = content[idx:]
                        if tail:
                            parts.append({"type": "text", "text": tail})
                    msg["content"] = parts
                return msg

            return msg

        return [transform_message(dict(m)) for m in messages]

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
