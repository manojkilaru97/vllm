# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json
import time
from collections.abc import AsyncGenerator, AsyncIterator
from collections.abc import Sequence as GenericSequence
from typing import Any, Final
import os
import logging

import jinja2
import partial_json_parser
import regex as re
import base64
from pathlib import Path
from fastapi import Request
from openai_harmony import Message as OpenAIMessage
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
    get_developer_message,
    get_stop_tokens_for_assistant_actions,
    get_streamable_parser_for_assistant,
    get_system_message,
    parse_chat_inputs_to_harmony_messages,
    parse_chat_output,
    render_for_completion,
)
from vllm.entrypoints.openai.utils import maybe_filter_parallel_tool_calls
from vllm.entrypoints.utils import get_max_tokens, should_include_usage
from vllm.inputs.data import ProcessorInputs, TokensPrompt
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
from vllm.request_context import reset_request_id, set_request_id
from vllm.utils.async_utils import tokenizer_lock
from vllm.utils.collection_utils import as_list
from vllm.utils.mistral import is_mistral_tokenizer
from vllm.utils.mistral import mt as _mt

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
            log_error_stack=log_error_stack,
        )

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

        # Handle tool call ID type for Kimi K2.
        # Some Kimi K2 checkpoints may expose model_type as deepseek_v3, so
        # also detect via parser names and hf_overrides for test mocking.
        hf_overrides = getattr(self.model_config, "hf_overrides", None)
        if (
            self.model_config.hf_text_config.model_type == "kimi_k2"
            or self.model_config.hf_config.model_type == "kimi_k2"
            or tool_parser == "kimi_k2"
            or reasoning_parser == "kimi_k2"
            or (
                isinstance(hf_overrides, dict)
                and hf_overrides.get("model_type") == "kimi_k2"
            )
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

    async def render_chat_request(
        self,
        request: ChatCompletionRequest,
        raw_request: Request | None = None,
    ) -> tuple[list[ConversationMessage], list[ProcessorInputs]] | ErrorResponse:
        """
        render chat request by validating and preprocessing inputs.

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
                # Collect all incoming headers unfiltered
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
                try:
                    payload_logger.info(
                        "openai.request",
                        extra={
                            "rid": rid_hint or "",
                            "endpoint": self.__class__.__name__,
                            # Pass dict directly for proper OTEL structured logging
                            "payload": req_dump,
                            "headers": headers_obj,
                        },
                    )
                except Exception:
                    pass

            # Resolve NVCF assets in-place before preprocessing.
            # This happens after request payload logging so we preserve what
            # the caller actually sent (asset_id refs, base64, etc.).
            if raw_request is not None:
                try:
                    request.messages = self._resolve_nvcf_image_assets(
                        request.messages, raw_request)
                except Exception as e:
                    logger.exception("Error while resolving NVCF assets")
                    return self.create_error_response(str(e))

            # Strip raw multimodal special tokens from plain text to avoid
            # backend crashes when no corresponding media is provided.
            try:
                request.messages = self._strip_mm_special_tokens_in_messages(
                    request.messages)
            except Exception:
                logger.exception("Error while stripping MM special tokens")

            renderer = self.engine_client.renderer
            tokenizer = renderer.tokenizer

            # For gpt-oss (harmony) models, special tokens are part of the
            # protocol framing. By default, OpenAI-compatible requests set
            # `skip_special_tokens=True`, which can strip these markers from
            # the streamed text. If the caller didn't explicitly set this
            # field, default to keeping special tokens for harmony models.
            if self.use_harmony and "skip_special_tokens" not in request.model_fields_set:
                request.skip_special_tokens = False

            tool_parser = self.tool_parser

            if is_mistral_tokenizer(tokenizer):
                # because of issues with pydantic we need to potentially
                # re-serialize the tool_calls field of the request
                # for more info: see comment in `maybe_serialize_tool_calls`
                _mt.maybe_serialize_tool_calls(request)  # type: ignore[arg-type]
                _mt.truncate_tool_call_ids(request)  # type: ignore[arg-type]
                _mt.validate_request_params(request)

            # Check if tool parsing is unavailable (common condition)
            tool_parsing_unavailable = (
                tool_parser is None
                and not is_mistral_tokenizer(tokenizer)
                and not self.use_harmony
            )

            # Validate tool_choice when tool parsing is required but unavailable
            if tool_parsing_unavailable and request.tool_choice not in (
                None,
                "none",
            ):
                if request.tool_choice == "auto" and not self.enable_auto_tools:
                    # for hf tokenizers, "auto" tools requires
                    # --enable-auto-tool-choice and --tool-call-parser
                    return self.create_error_response(
                        '"auto" tool choice requires '
                        "--enable-auto-tool-choice and --tool-call-parser to be set"
                    )
                elif request.tool_choice != "auto":
                    # "required" or named tool requires tool parser
                    return self.create_error_response(
                        f'tool_choice="{request.tool_choice}" requires '
                        "--tool-call-parser to be set"
                    )

            if request.tools is None or (
                request.tool_choice == "none"
                and self.exclude_tools_when_tool_choice_none
            ):
                tool_dicts = None
            else:
                tool_dicts = [tool.model_dump() for tool in request.tools]

            if not self.use_harmony:
                # Common case.
                error_check_ret = self._validate_chat_template(
                    request_chat_template=request.chat_template,
                    chat_template_kwargs=request.chat_template_kwargs,
                    trust_request_chat_template=self.trust_request_chat_template,
                )
                if error_check_ret is not None:
                    return error_check_ret

                conversation, engine_prompts = await self._preprocess_chat(
                    request,
                    request.messages,
                    default_template=self.chat_template,
                    default_template_content_format=self.chat_template_content_format,
                    default_template_kwargs=self.default_chat_template_kwargs,
                    tool_dicts=tool_dicts,
                    tool_parser=tool_parser,
                )
            else:
                # For GPT-OSS.
                should_include_tools = tool_dicts is not None
                conversation, engine_prompts = self._make_request_with_harmony(
                    request, should_include_tools
                )
        except (ValueError, TypeError, RuntimeError, jinja2.TemplateError) as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(e)
        finally:
            try:
                reset_request_id(ctx_token)
            except Exception:
                pass

        return conversation, engine_prompts

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

        result = await self.render_chat_request(request, raw_request)
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

        try:
            lora_request = self._maybe_get_adapters(
                request, supports_default_mm_loras=True
            )

            model_name = self.models.model_name(lora_request)
        except (ValueError, TypeError, RuntimeError) as e:
            logger.exception("Error preparing request components")
            return self.create_error_response(e)

        # Extract data_parallel_rank from header (router can inject it)
        data_parallel_rank = self._get_data_parallel_rank(raw_request)

        # Schedule the request and get the result generator.
        max_model_len = self.model_config.max_model_len
        generators: list[AsyncGenerator[RequestOutput, None]] = []
        try:
            for i, engine_prompt in enumerate(engine_prompts):
                prompt_token_ids = self._extract_prompt_components(
                    engine_prompt
                ).token_ids

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
        except ValueError as e:
            return self.create_error_response(e)

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

        try:
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
        except GenerationError as e:
            return self._convert_generation_error_to_response(e)
        except ValueError as e:
            return self.create_error_response(e)

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
    def _has_kimi_k2_markers(text: str) -> bool:
        """Check if text contains Kimi K2 tool call markers."""
        markers = [
            "<|tool_calls_section_begin|>",
            "<|tool_call_begin|>",
            "<|tool_call_argument_begin|>",
        ]
        return any(marker in text for marker in markers)

    @staticmethod
    def _extract_kimi_k2_arguments(text: str, partial_ok: bool = False) -> str | None:
        """Extract arguments from Kimi K2 tool call format.

        Args:
            text: The text containing Kimi K2 markers
            partial_ok: If True, return partial args even without end marker (for named streaming).
                       If False, only return when end marker is found (for required streaming).
        """
        arg_begin = "<|tool_call_argument_begin|>"
        arg_end = "<|tool_call_end|>"
        if arg_begin in text:
            start = text.find(arg_begin) + len(arg_begin)
            end_pos = text.find(arg_end, start)
            if end_pos > start:
                # Complete arguments found
                return text[start:end_pos].strip()
            elif partial_ok:
                # End marker not found yet, but partial extraction is allowed
                return text[start:].strip()
            else:
                # End marker not found and partial extraction not allowed
                return None
        return None

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
        """Best-effort extraction of the n-th `"parameters": {...}` object.

        This is intentionally tolerant to partial/incomplete JSON and is used to
        avoid leaking tokens from a *next* tool call into the current tool's
        streamed `arguments` when `tool_choice=required`.
        """
        if n < 0:
            return None

        # 1) Find the n-th occurrence of the `"parameters"` key *outside* of
        # JSON strings.
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
                # Potential start of a JSON string / key.
                if text.startswith(needle, i):
                    # Ensure this looks like a key: `"parameters"\s*:`
                    j = i + len(needle)
                    while j < len(text) and text[j].isspace():
                        j += 1
                    if j < len(text) and text[j] == ":":
                        found += 1
                        if found == n:
                            key_pos = j + 1  # position after ':'
                            break
                    i = j
                else:
                    in_str = True
                i += 1
                continue

            i += 1

        if key_pos is None:
            return None

        # 2) From after the colon, find the opening '{' for the object.
        j = key_pos
        while j < len(text) and text[j].isspace():
            j += 1
        while j < len(text) and text[j] != "{":
            # For required tool calling, parameters should be an object.
            # If we don't see it yet, bail (partial generation).
            if not text[j].isspace():
                return None
            j += 1
        if j >= len(text) or text[j] != "{":
            return None

        # 3) Capture a JSON object starting at '{', respecting strings/escapes.
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

        # Partial (unterminated) object.
        return "".join(out)

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

        # For robust multi-tool streaming, also parse previous_text so we can
        # detect when a new tool object is appended to the JSON array. Relying
        # on `tool_call_idx` (which may include history) can cause the server to
        # reuse/shift indices and corrupt arguments by concatenating multiple
        # tools into a single streamed `arguments` buffer.
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

        # check if the current text is a valid array
        # containing a partial tool calling object
        # if not repeat
        if obj is None or not isinstance(obj, list) or not len(obj) > 0:
            function_name_returned = False
            delta_message = None
        else:
            _, finishes_previous_tool = OpenAIServingChat._filter_delta_text(
                delta_text, previous_text
            )
            # take the last tool call from the generated list
            current_tool_call = obj[-1]

            # once parameters have been generated the name is complete as well
            if not finishes_previous_tool and (
                "name" not in current_tool_call or "parameters" not in current_tool_call
            ):
                function_name_returned = False
                delta_message = None
            else:
                if not function_name_returned:
                    # get partly generated arguments from the latest tool call
                    param_match = re.search(
                        r'.*"parameters":\s*(.*)', current_text, re.DOTALL
                    )
                    arguments = param_match.group(1) if param_match else ""
                    arguments, _ = OpenAIServingChat._filter_delta_text(
                        arguments, previous_text
                    )

                    # if this iteration finishes a previous tool call but a
                    # new incomplete tool is already generated, take the
                    # previous from the list
                    if finishes_previous_tool and "parameters" not in current_tool_call:
                        current_tool_call = obj[-2]

                    function_name_returned = True
                    tool_call_id = make_tool_call_id(
                        id_type=self.tool_call_id_type,
                        func_name=current_tool_call["name"],
                        idx=tool_call_idx,
                    )
                    delta_message = DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                id=tool_call_id,
                                function=DeltaFunctionCall(
                                    name=current_tool_call["name"], arguments=arguments
                                ),
                                index=len(obj) - 1,
                                type="function",
                            )
                        ]
                    )

                else:
                    # Check if a NEW tool call has appeared in the array
                    current_tool_idx = len(obj) - 1
                    prev_len = len(prev_obj) if isinstance(prev_obj, list) else 0
                    new_tool_appended = prev_len != 0 and len(obj) > prev_len
                    is_new_tool = (
                        new_tool_appended
                        and "name" in current_tool_call
                        and "parameters" in current_tool_call
                    )

                    if is_new_tool:
                        # This is a NEW tool call appended to the array. Extract
                        # its parameters object directly to avoid mixing with
                        # adjacent tool objects.
                        arguments = OpenAIServingChat._extract_nth_parameters_obj(
                            current_text, current_tool_idx
                        ) or ""

                        tool_call_id = make_tool_call_id(
                            id_type=self.tool_call_id_type,
                            func_name=current_tool_call["name"],
                            idx=tool_call_idx,
                        )
                        delta_message = DeltaMessage(
                            tool_calls=[
                                DeltaToolCall(
                                    id=tool_call_id,
                                    function=DeltaFunctionCall(
                                        name=current_tool_call["name"],
                                        arguments=arguments,
                                    ),
                                    index=current_tool_idx,
                                    type="function",
                                )
                            ]
                        )
                    else:
                        # Continue streaming arguments for the current (last)
                        # tool in the array. We stream a safe suffix delta of
                        # that tool's parameters object to avoid accidentally
                        # appending tokens from an adjacent tool call.
                        current_streaming_idx = len(obj) - 1

                        prev_args = OpenAIServingChat._extract_nth_parameters_obj(
                            previous_text, current_streaming_idx
                        )
                        curr_args = OpenAIServingChat._extract_nth_parameters_obj(
                            current_text, current_streaming_idx
                        )

                        if not curr_args:
                            delta_message = None
                        else:
                            if prev_args and curr_args.startswith(prev_args):
                                arguments_delta = curr_args[len(prev_args) :]
                            elif prev_args is None:
                                # First time we can extract params for this tool.
                                arguments_delta = curr_args
                            else:
                                # If we can't compute a safe delta, don't stream
                                # anything rather than corrupting JSON.
                                arguments_delta = ""

                            if arguments_delta:
                                delta_message = DeltaMessage(
                                    tool_calls=[
                                        DeltaToolCall(
                                            function=DeltaFunctionCall(
                                                name=None,
                                                arguments=arguments_delta,
                                            ),
                                            index=current_streaming_idx,
                                        )
                                    ]
                                )
                            else:
                                delta_message = None

        return delta_message, function_name_returned

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
        rid_hint = request_id.split("-", 1)[1] if request_id.startswith("chatcmpl-") else request_id

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
        # Track previously sent arguments for named tool_choice streaming (delta computation)
        named_tool_previous_args = [""] * num_choices
        if self.tool_call_id_type == "kimi_k2":
            history_tool_call_cnt = get_history_tool_calls_cnt(conversation)
        else:
            history_tool_call_cnt = 0

        # Always track previous_texts for comprehensive output logging
        previous_texts = [""] * num_choices
        
        # Track reasoning, content and tool calls separately for proper logging
        previous_reasoning_texts = [""] * num_choices
        previous_content_texts = [""] * num_choices
        previous_tool_calls: list[list[DeltaToolCall]] = [[] for _ in range(num_choices)]
        # Some tool parsers can produce sparse/raw tool indices (e.g. 0,2,...)
        # due to internal boundary bookkeeping. Normalize to dense indices before
        # emitting streaming chunks.
        tool_index_dense_maps: list[dict[int, int]] = [dict() for _ in range(num_choices)]
        tool_index_dense_next: list[int] = [0] * num_choices

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
                                    and delta_message.reasoning.lstrip().startswith(("{", "["))
                                    and tool_choice_function_name
                                ):
                                    # CRITICAL FIX: Model output raw JSON without <thinking> tags.
                                    # The "reasoning" is actually the start of JSON arguments.
                                    # Preserve it so it can be combined with subsequent chunks.
                                    current_text = delta_message.reasoning
                                    # Don't send this as reasoning to the client
                                    delta_message.reasoning = None
                                    delta_message.reasoning_content = None
                                else:
                                    current_text = ""
                            elif output.finish_reason is not None:
                                # Fallback for cases where thinking mode is enabled
                                # but the model never emits </think>. Parse the final
                                # accumulated text as named tool arguments.
                                accumulated_text = previous_text + delta_text
                                current_text = accumulated_text
                                try:
                                    parsed_calls, _ = self._parse_tool_calls_from_content(
                                        request=request,
                                        tokenizer=tokenizer,
                                        content=accumulated_text,
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
                                    if matched_call and matched_call.arguments is not None
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
                                        emitted_tool_call_id = make_tool_call_id()
                                        delta_tool_call = DeltaToolCall(
                                            id=emitted_tool_call_id,
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
                            # Named tool_choice streaming - handle Kimi K2 marker format
                            # Accumulate full text to extract clean JSON arguments
                            # Detection of <think> content handles stripping
                            accumulated_text = previous_text + delta_text
                            current_text = accumulated_text  # Persist for next iteration

                            # Check if the model outputs Kimi K2 marker format
                            if OpenAIServingChat._has_kimi_k2_markers(accumulated_text):
                                # Extract clean arguments from between markers
                                # Allow partial extraction for streaming deltas (partial_ok=True)
                                extracted_args = OpenAIServingChat._extract_kimi_k2_arguments(accumulated_text, partial_ok=True)

                                if extracted_args is not None:
                                    # We have extracted arguments - compute delta from what was sent before
                                    previous_args = named_tool_previous_args[i]

                                    # Compute delta: only send the new portion
                                    if extracted_args.startswith(previous_args):
                                        arguments_delta = extracted_args[len(previous_args):]
                                    else:
                                        # Fallback: send full extracted args if prefix doesn't match
                                        arguments_delta = extracted_args

                                    # Update tracking
                                    named_tool_previous_args[i] = extracted_args

                                    # Send delta if there's new content
                                    if arguments_delta or not function_name_returned[i]:
                                        if function_name_returned[i]:
                                            delta_tool_call = DeltaToolCall(
                                                function=DeltaFunctionCall(arguments=arguments_delta),
                                                index=i,
                                            )
                                        else:
                                            # First delta includes function name
                                            if is_mistral_tokenizer(tokenizer):
                                                tool_call_id = MistralToolCall.generate_random_id()
                                            else:
                                                tool_call_id = make_tool_call_id(
                                                    id_type=self.tool_call_id_type,
                                                    func_name=tool_choice_function_name,
                                                    idx=i,
                                                )
                                            delta_tool_call = DeltaToolCall(
                                                id=tool_call_id,
                                                type="function",
                                                function=DeltaFunctionCall(
                                                    name=tool_choice_function_name,
                                                    arguments=arguments_delta,
                                                ),
                                                index=i,
                                            )
                                            function_name_returned[i] = True

                                        delta_message = DeltaMessage(
                                            tool_calls=[
                                                delta_tool_call,
                                            ]
                                        )
                                        tools_streamed[i] = True
                                    else:
                                        # No new content, don't send a delta
                                        delta_message = None
                                else:
                                    # Markers detected but arguments not yet complete, wait
                                    delta_message = None
                            else:
                                # No markers - either reasoning text before tool call or standard format
                                # For Kimi K2, suppress reasoning text (don't send as arguments)
                                # For non-Kimi models, fall back to original behavior
                                if self.tool_call_id_type == "kimi_k2":
                                    # At finish time without markers, use accumulated text as args
                                    # (similar to non-streaming behavior in engine/serving.py)
                                    if (
                                        output.finish_reason is not None
                                        and not function_name_returned[i]
                                        and accumulated_text.strip()
                                    ):
                                        if is_mistral_tokenizer(tokenizer):
                                            tool_call_id = MistralToolCall.generate_random_id()
                                        else:
                                            tool_call_id = make_tool_call_id(
                                                id_type=self.tool_call_id_type,
                                                func_name=tool_choice_function_name,
                                                idx=i,
                                            )
                                        delta_tool_call = DeltaToolCall(
                                            id=tool_call_id,
                                            type="function",
                                            function=DeltaFunctionCall(
                                                name=tool_choice_function_name,
                                                arguments=accumulated_text.strip(),
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
                                else:
                                    # For named tool_choice, parse once at finish for all
                                    # non-Kimi outputs. Incremental deltas are fragile and can
                                    # yield malformed JSON arguments under larger stream intervals.
                                    if output.finish_reason is None:
                                        delta_message = None
                                    else:
                                        try:
                                            parsed_calls, _ = self._parse_tool_calls_from_content(
                                                request=request,
                                                tokenizer=tokenizer,
                                                content=accumulated_text,
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
                                            if matched_call and matched_call.arguments is not None
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
                                                if is_mistral_tokenizer(tokenizer):
                                                    emitted_tool_call_id = (
                                                        MistralToolCall.generate_random_id()
                                                    )
                                                else:
                                                    emitted_tool_call_id = make_tool_call_id(
                                                        id_type=self.tool_call_id_type,
                                                        func_name=tool_choice_function_name,
                                                        idx=i,
                                                    )
                                                delta_tool_call = DeltaToolCall(
                                                    id=emitted_tool_call_id,
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
                                # Fallback for cases where thinking mode is enabled
                                # but the model never emits </think>. Parse at
                                # finish from the full accumulated text.
                                reasoning_end_arr[i] = True
                                content = current_text
                                try:
                                    parsed_calls, _ = self._parse_tool_calls_from_content(
                                        request=request,
                                        tokenizer=tokenizer,
                                        content=content,
                                        enable_auto_tools=self.enable_auto_tools,
                                        tool_parser_cls=self.tool_parser,
                                    )
                                except Exception:
                                    parsed_calls = None

                                delta_tool_calls: list[DeltaToolCall] = []
                                for j, call in enumerate(parsed_calls or []):
                                    if not call.name:
                                        continue
                                    args = (
                                        call.arguments
                                        if call.arguments is not None
                                        else "{}"
                                    )
                                    tool_call_id = call.id
                                    if tool_call_id is None:
                                        tool_call_id = make_tool_call_id(
                                            id_type=self.tool_call_id_type,
                                            func_name=call.name,
                                            idx=history_tool_call_cnt + j,
                                        )
                                    delta_tool_calls.append(
                                        DeltaToolCall(
                                            id=tool_call_id,
                                            type="function",
                                            function=DeltaFunctionCall(
                                                name=call.name,
                                                arguments=args,
                                            ),
                                            index=j,
                                        )
                                    )

                                if delta_tool_calls:
                                    delta_message = DeltaMessage(tool_calls=delta_tool_calls)
                                    function_name_returned[i] = True
                                    history_tool_call_cnt += len(delta_tool_calls)
                                    tools_streamed[i] = True
                                else:
                                    delta_message = None

                        else:
                            # either finished reasoning or no reasoning at all
                            content = current_text

                            # For `tool_choice=required`, we only emit tool_call
                            # deltas once we have the *full* tool-call JSON.
                            # Incremental parsing here is fragile and can
                            # mis-index tool calls (merging arguments across
                            # tools), especially when the JSON array grows
                            # between token boundaries.
                            if output.finish_reason is None:
                                delta_message = None
                            else:
                                try:
                                    parsed_calls, _ = self._parse_tool_calls_from_content(
                                        request=request,
                                        tokenizer=tokenizer,
                                        content=content,
                                        enable_auto_tools=self.enable_auto_tools,
                                        tool_parser_cls=self.tool_parser,
                                    )
                                except Exception:
                                    parsed_calls = None

                                delta_tool_calls: list[DeltaToolCall] = []
                                for j, call in enumerate(parsed_calls or []):
                                    if not call.name:
                                        continue
                                    args = call.arguments if call.arguments is not None else "{}"
                                    tool_call_id = call.id
                                    if tool_call_id is None:
                                        tool_call_id = make_tool_call_id(
                                            id_type=self.tool_call_id_type,
                                            func_name=call.name,
                                            idx=history_tool_call_cnt + j,
                                        )
                                    delta_tool_calls.append(
                                        DeltaToolCall(
                                            id=tool_call_id,
                                            type="function",
                                            function=DeltaFunctionCall(
                                                name=call.name,
                                                arguments=args,
                                            ),
                                            index=j,
                                        )
                                    )

                                if delta_tool_calls:
                                    delta_message = DeltaMessage(tool_calls=delta_tool_calls)
                                    function_name_returned[i] = True
                                else:
                                    delta_message = None
                            if (
                                delta_message
                                and delta_message.tool_calls
                                and delta_message.tool_calls[0].id is not None
                            ):
                                history_tool_call_cnt += len(delta_message.tool_calls)
                                tools_streamed[i] = True

                    # handle streaming deltas for tools with "auto" tool choice
                    # and reasoning parser
                    elif tool_choice_auto and reasoning_parser:
                        assert tool_parser is not None
                        assert added_content_delta_arr is not None
                        assert reasoning_end_arr is not None
                        output_token_ids = as_list(output.token_ids)
                        if not reasoning_end_arr[i]:
                            # When encountering think end id in prompt_token_ids
                            # i.e {"enable_thinking": False},
                            # set reasoning status to end.
                            if prompt_is_reasoning_end_arr[i]:
                                reasoning_end_arr[i] = True
                                current_token_ids = output_token_ids
                                # Don't update current_text, keep it as is from delta
                            else:
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
                                    if delta_message and delta_message.content:
                                        current_text = delta_message.content
                                        delta_message.content = None
                                    else:
                                        current_text = ""

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
                    # Some tool parsers may emit boundary/control deltas with no
                    # function name and empty arguments. These should not be
                    # surfaced to clients as standalone tool calls.
                    if tool_choice_auto and delta_message and delta_message.tool_calls:
                        kept_tool_calls: list[DeltaToolCall] = []
                        dropped_empty = 0
                        for delta_tc in delta_message.tool_calls:
                            fn = delta_tc.function
                            if isinstance(fn, dict):
                                fn_name = fn.get("name")
                                fn_args = fn.get("arguments")
                            else:
                                fn_name = fn.name if fn else None
                                fn_args = fn.arguments if fn else None

                            is_empty_boundary = (
                                (fn_name is None or fn_name == "")
                                and (fn_args is None or fn_args == "")
                            )
                            if is_empty_boundary:
                                dropped_empty += 1
                                continue

                            if delta_tc.index is not None:
                                raw_idx = delta_tc.index
                                dense_map = tool_index_dense_maps[i]
                                if raw_idx not in dense_map:
                                    dense_map[raw_idx] = tool_index_dense_next[i]
                                    tool_index_dense_next[i] += 1
                                dense_idx = dense_map[raw_idx]
                                delta_tc.index = dense_idx
                            kept_tool_calls.append(delta_tc)

                        if dropped_empty:
                            delta_kwargs: dict[str, object] = {}
                            if delta_message.content is not None:
                                delta_kwargs["content"] = delta_message.content
                            if delta_message.reasoning is not None:
                                delta_kwargs["reasoning"] = delta_message.reasoning
                            if kept_tool_calls:
                                delta_kwargs["tool_calls"] = kept_tool_calls
                            delta_message = (
                                DeltaMessage(**delta_kwargs) if delta_kwargs else None
                            )

                    if (
                        tool_choice_auto or reasoning_parser or tool_choice_function_name
                    ) and not self.use_harmony:
                        assert previous_texts is not None
                        assert all_previous_token_ids is not None
                        previous_texts[i] = current_text
                        all_previous_token_ids[i] = current_token_ids
                        
                        # Track reasoning, content, and tool calls separately for logging
                        if delta_message:
                            if delta_message.reasoning:
                                previous_reasoning_texts[i] += delta_message.reasoning
                            if delta_message.content:
                                previous_content_texts[i] += delta_message.content
                            if delta_message.tool_calls:
                                # Track tool calls - merge or add new ones
                                for delta_tc in delta_message.tool_calls:
                                    if delta_tc.index is not None:
                                        # Ensure we have enough slots
                                        while len(previous_tool_calls[i]) <= delta_tc.index:
                                            previous_tool_calls[i].append(DeltaToolCall(index=len(previous_tool_calls[i])))
                                        # Merge with existing tool call at this index
                                        existing_tc = previous_tool_calls[i][delta_tc.index]
                                        if delta_tc.id is not None:
                                            existing_tc.id = delta_tc.id
                                        if delta_tc.type is not None:
                                            existing_tc.type = delta_tc.type
                                        if delta_tc.function is not None:
                                            if existing_tc.function is None:
                                                existing_tc.function = DeltaFunctionCall()
                                            if delta_tc.function.name is not None:
                                                existing_tc.function.name = delta_tc.function.name
                                            if delta_tc.function.arguments is not None:
                                                if existing_tc.function.arguments is None:
                                                    existing_tc.function.arguments = ""
                                                existing_tc.function.arguments += delta_tc.function.arguments
                    else:
                        # Update for comprehensive logging even in simple case
                        assert previous_texts is not None
                        previous_texts[i] += delta_text
                        
                        # Track reasoning and content separately for logging
                        if delta_message:
                            if delta_message.reasoning:
                                previous_reasoning_texts[i] += delta_message.reasoning
                            if delta_message.content:
                                previous_content_texts[i] += delta_message.content
                            if delta_message.tool_calls:
                                # Track tool calls - merge or add new ones
                                for delta_tc in delta_message.tool_calls:
                                    if delta_tc.index is not None:
                                        # Ensure we have enough slots
                                        while len(previous_tool_calls[i]) <= delta_tc.index:
                                            previous_tool_calls[i].append(DeltaToolCall(index=len(previous_tool_calls[i])))
                                        # Merge with existing tool call at this index
                                        existing_tc = previous_tool_calls[i][delta_tc.index]
                                        if delta_tc.id is not None:
                                            existing_tc.id = delta_tc.id
                                        if delta_tc.type is not None:
                                            existing_tc.type = delta_tc.type
                                        if delta_tc.function is not None:
                                            if existing_tc.function is None:
                                                existing_tc.function = DeltaFunctionCall()
                                            if delta_tc.function.name is not None:
                                                existing_tc.function.name = delta_tc.function.name
                                            if delta_tc.function.arguments is not None:
                                                if existing_tc.function.arguments is None:
                                                    existing_tc.function.arguments = ""
                                                existing_tc.function.arguments += delta_tc.function.arguments

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

                        # CRITICAL FIX: If the parser returned a delta_message with tool_calls
                        # in this final iteration, we need to yield the NAME/ID for any tools
                        # that haven't had their name sent yet. We must NOT include arguments
                        # as they will be handled by the remaining args logic below.
                        if delta_message and delta_message.tool_calls and tool_parser:
                            for tc in delta_message.tool_calls:
                                func = tc.function
                                if isinstance(func, dict):
                                    fn_name = func.get("name")
                                else:
                                    fn_name = func.name if func else None

                                # Only yield if this tool has a name that needs sending
                                if fn_name and tc.index is not None:
                                    # Create a delta with ONLY name/id/type (no args to avoid duplicates)
                                    name_only_delta = DeltaMessage(
                                        tool_calls=[
                                            DeltaToolCall(
                                                index=tc.index,
                                                id=tc.id,
                                                type=tc.type or "function",
                                                function=DeltaFunctionCall(
                                                    name=fn_name
                                                ).model_dump(exclude_none=True),
                                            )
                                        ]
                                    )
                                    finish_delta_choice = ChatCompletionResponseStreamChoice(
                                        index=i,
                                        delta=name_only_delta,
                                        logprobs=None,
                                        finish_reason=None,
                                    )
                                    finish_delta_chunk = ChatCompletionStreamResponse(
                                        id=request_id,
                                        created=created_time,
                                        model=model_name,
                                        choices=[finish_delta_choice],
                                    )
                                    yield f"data: {finish_delta_chunk.model_dump_json()}\n\n"

                        # NAMED TOOL_CHOICE FINISH HANDLING: If we have a named tool_choice
                        # but haven't sent the tool_call yet, create it now using accumulated text.
                        # This handles cases where:
                        # 1. Model outputs without Kimi K2 markers (raw JSON)
                        # 2. Reasoning and finish happen in the same iteration
                        if (
                            tool_choice_function_name
                            and not function_name_returned[i]
                            and self.tool_call_id_type == "kimi_k2"
                        ):
                            # Get accumulated text from previous_texts (set during streaming)
                            assert previous_texts is not None
                            finish_accumulated_text = previous_texts[i]
                            if finish_accumulated_text.strip():
                                finish_tool_call_id = make_tool_call_id()
                                finish_tool_call = DeltaToolCall(
                                    id=finish_tool_call_id,
                                    type="function",
                                    function=DeltaFunctionCall(
                                        name=tool_choice_function_name,
                                        arguments=finish_accumulated_text.strip(),
                                    ),
                                    index=i,
                                )
                                delta_message = DeltaMessage(tool_calls=[finish_tool_call])
                                function_name_returned[i] = True
                                tools_streamed[i] = True

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

                        # Check for unstreamed tool arguments - either from current delta
                        # OR from parser state when delta is empty but tools were called
                        should_check_unstreamed = (
                            self._should_check_for_unstreamed_tool_arg_tokens(
                                delta_message, output
                            )
                            and tool_parser
                        )
                        # CRITICAL FIX: Also check when delta_message is empty but
                        # tool_parser has tool calls that were being streamed.
                        # This handles the case where finish_reason arrives in a
                        # separate event with no delta.tool_calls.
                        if (
                            not should_check_unstreamed
                            and tool_parser
                            and auto_tools_called
                            and hasattr(tool_parser, 'tool_calls_emitted')
                            and tool_parser.tool_calls_emitted
                            and output.finish_reason is not None
                        ):
                            should_check_unstreamed = True
                        # Qwen3 XML parser already streams its own argument
                        # closure chunks; forcing "remaining" emission here can
                        # inject duplicate trailing braces and id/name-less deltas.
                        if (
                            should_check_unstreamed
                            and tool_parser
                            and tool_parser.__class__.__name__ == "Qwen3XMLToolParser"
                        ):
                            should_check_unstreamed = False

                        if should_check_unstreamed:
                            latest_delta_len = 0
                            if (
                                delta_message
                                and delta_message.tool_calls
                                and delta_message.tool_calls[0]
                                and isinstance(
                                    delta_message.tool_calls[0].function,
                                    DeltaFunctionCall,
                                )
                            ) and isinstance(
                                delta_message.tool_calls[0].function.arguments, str
                            ):
                                latest_delta_len = len(
                                    delta_message.tool_calls[0].function.arguments
                                )

                            # get the expected call based on partial JSON
                            # parsing which "autocompletes" the JSON
                            raw_args = tool_parser.prev_tool_call_arr[index].get(
                                "arguments", {}
                            )
                            # Some parsers (e.g., kimi) store arguments as a string,
                            # others store as a dict. Only json.dumps if it's a dict.
                            if isinstance(raw_args, str):
                                expected_call = raw_args
                            else:
                                expected_call = json.dumps(raw_args, ensure_ascii=False)

                            # get what we've streamed so far for arguments
                            # for the current tool
                            if index < len(tool_parser.streamed_args_for_tool):
                                actual_call = tool_parser.streamed_args_for_tool[index]
                            else:
                                # Parser and serving can momentarily diverge at
                                # stream boundaries; avoid crashing the stream.
                                actual_call = ""
                            if latest_delta_len > 0:
                                actual_call = actual_call[:-latest_delta_len]

                            # check to see if there's anything left to stream.
                            # Only stream a suffix if what we've already streamed
                            # is a strict prefix of the expected arguments.
                            if expected_call.startswith(actual_call):
                                remaining_call = expected_call[len(actual_call) :]
                                prefix_ok = True
                            else:
                                remaining_call = ""
                                prefix_ok = False

                            # CRITICAL: Check if name was sent for this tool. If not, include it!
                            # This handles the race condition in parallel tool calls where
                            # tool 1's name wasn't streamed before finish.
                            tool_name = None
                            tool_id = None
                            tool_type = None
                            if hasattr(tool_parser, 'tool_name_sent_arr'):
                                name_sent = (
                                    index < len(tool_parser.tool_name_sent_arr)
                                    and tool_parser.tool_name_sent_arr[index]
                                )
                                if not name_sent and index < len(tool_parser.prev_tool_call_arr):
                                    tool_info = tool_parser.prev_tool_call_arr[index]
                                    tool_name = tool_info.get("name")
                                    tool_id = tool_info.get("id")
                                    tool_type = "function"

                            should_emit_remaining = bool(
                                remaining_call or tool_name or tool_id or tool_type
                            )

                            if should_emit_remaining:
                                # set that as a delta message
                                delta_message = DeltaMessage(
                                    tool_calls=[
                                        DeltaToolCall(
                                            index=index,
                                            id=tool_id,
                                            type=tool_type,
                                            function=DeltaFunctionCall(
                                                name=tool_name,
                                                arguments=remaining_call
                                            ).model_dump(exclude_none=True),
                                        )
                                    ]
                                )

                        # CRITICAL: Before sending finish, check if any parallel tools
                        # didn't have their name sent during streaming. If so, send them now.
                        if (
                            tool_parser
                            and hasattr(tool_parser, 'tool_name_sent_arr')
                            and hasattr(tool_parser, 'prev_tool_call_arr')
                        ):
                            for tidx, tool_info in enumerate(tool_parser.prev_tool_call_arr):
                                name_was_sent = (
                                    tidx < len(tool_parser.tool_name_sent_arr)
                                    and tool_parser.tool_name_sent_arr[tidx]
                                )
                                if not name_was_sent and tool_info.get("name"):
                                    # This tool's name was never sent - send it now!
                                    missed_tool_delta = DeltaMessage(
                                        tool_calls=[
                                            DeltaToolCall(
                                                index=tidx,
                                                id=tool_info.get("id"),
                                                type="function",
                                                function=DeltaFunctionCall(
                                                    name=tool_info.get("name"),
                                                    arguments=tool_info.get("arguments", "")
                                                ).model_dump(exclude_none=True),
                                            )
                                        ]
                                    )
                                    missed_choice = ChatCompletionResponseStreamChoice(
                                        index=i,
                                        delta=missed_tool_delta,
                                        logprobs=None,
                                        finish_reason=None,
                                    )
                                    missed_chunk = ChatCompletionStreamResponse(
                                        id=request_id,
                                        created=created_time,
                                        model=model_name,
                                        choices=[missed_choice],
                                    )
                                    yield f"data: {missed_chunk.model_dump_json()}\n\n"
                                    # Mark as sent
                                    tool_parser.tool_name_sent_arr[tidx] = True

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

            # Emit a single streaming summary payload log
            if os.getenv("VLLM_LOG_PAYLOADS", "1") == "1":
                try:
                    usage_dict = (
                        request_metadata.final_usage_info.model_dump()
                        if request_metadata.final_usage_info
                        else None
                    )
                except Exception:
                    usage_dict = None
                # Build choices with actual content, reasoning, and tool_calls
                choices_list = []
                for i in range(num_choices):
                    choice_data = {
                        "index": i,
                        "message": {
                            "role": "assistant",
                        },
                        "finish_reason": "stop",
                    }
                    # Add reasoning if present
                    if previous_reasoning_texts[i]:
                        choice_data["message"]["reasoning_content"] = previous_reasoning_texts[i]
                    # Add content if present
                    if previous_content_texts[i]:
                        choice_data["message"]["content"] = previous_content_texts[i]
                    else:
                        choice_data["message"]["content"] = None
                    # Add tool_calls if present
                    if previous_tool_calls[i]:
                        tool_calls_list = []
                        for tc in previous_tool_calls[i]:
                            if tc.function and tc.function.name:
                                tool_calls_list.append({
                                    "id": tc.id,
                                    "type": tc.type or "function",
                                    "function": {
                                        "name": tc.function.name,
                                        "arguments": tc.function.arguments or "",
                                    },
                                })
                        if tool_calls_list:
                            choice_data["message"]["tool_calls"] = tool_calls_list
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
                try:
                    payload_logger.info(
                        "openai.response",
                        extra={
                            "rid": rid_hint,
                            "endpoint": self.__class__.__name__,
                            # Pass dict directly for proper OTEL structured logging
                            "payload": resp_summary,
                        },
                    )
                except Exception:
                    pass

            # Log complete streaming response if output logging is enabled
            if self.enable_log_outputs and self.request_logger:
                # Log the complete response for each choice
                for i in range(num_choices):
                    reasoning_text = previous_reasoning_texts[i]
                    content_text = previous_content_texts[i]
                    tool_calls_list = previous_tool_calls[i]

                    if reasoning_text:
                        self.request_logger.log_outputs(
                            request_id=request_id,
                            outputs=f"[reasoning] {reasoning_text}",
                            output_token_ids=None,
                            finish_reason=None,
                            is_streaming=True,
                            delta=False,
                        )

                    if content_text:
                        self.request_logger.log_outputs(
                            request_id=request_id,
                            outputs=content_text,
                            output_token_ids=None,
                            finish_reason="streaming_complete",
                            is_streaming=True,
                            delta=False,
                        )

                    # Log tool calls if present (similar to non-streaming mode)
                    if tool_calls_list:
                        tool_call_descriptions = []
                        for tc in tool_calls_list:
                            if tc.function and tc.function.name and tc.function.arguments:
                                tool_call_descriptions.append(
                                    f"{tc.function.name}({tc.function.arguments})"
                                )
                        if tool_call_descriptions:
                            tool_calls_str = ", ".join(tool_call_descriptions)
                            tool_calls_output = f"[tool_calls: {tool_calls_str}]"
                            self.request_logger.log_outputs(
                                request_id=request_id,
                                outputs=tool_calls_output,
                                output_token_ids=None,
                                finish_reason="streaming_complete",
                                is_streaming=True,
                                delta=False,
                            )

                    # If neither reasoning nor content nor tool calls, log a fallback message
                    if not reasoning_text and not content_text and not tool_calls_list:
                        full_text = (
                            previous_texts[i]
                            if previous_texts and i < len(previous_texts)
                            else f"<streaming_complete: {previous_num_tokens[i]} tokens>"
                        )
                        self.request_logger.log_outputs(
                            request_id=request_id,
                            outputs=full_text,
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
        rid_hint = request_id.split("-", 1)[1] if request_id.startswith("chatcmpl-") else request_id
        final_res: RequestOutput | None = None

        try:
            async for res in result_generator:
                final_res = res
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")
        except ValueError as e:
            return self.create_error_response(e)

        assert final_res is not None

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
                assert tool_calls is not None and len(tool_calls) > 0
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

        # Emit a single non-streaming summary payload log
        if os.getenv("VLLM_LOG_PAYLOADS", "1") == "1":
            try:
                usage_dict = usage.model_dump() if usage else None
            except Exception:
                usage_dict = None
            # Build choices with actual content, reasoning, and tool_calls
            choices_list = []
            for choice in choices:
                choice_data = {
                    "index": choice.index,
                    "message": {
                        "role": choice.message.role if hasattr(choice.message, 'role') else "assistant",
                    },
                    "finish_reason": choice.finish_reason,
                }
                # Add reasoning if present
                if hasattr(choice.message, 'reasoning') and choice.message.reasoning:
                    choice_data["message"]["reasoning_content"] = choice.message.reasoning
                # Add content
                choice_data["message"]["content"] = choice.message.content
                # Add tool_calls if present
                if hasattr(choice.message, 'tool_calls') and choice.message.tool_calls:
                    tool_calls_list = []
                    for tc in choice.message.tool_calls:
                        tc_dict = {
                            "id": tc.id if hasattr(tc, 'id') else None,
                            "type": tc.type if hasattr(tc, 'type') else "function",
                            "function": {
                                "name": tc.function.name if hasattr(tc.function, 'name') else None,
                                "arguments": tc.function.arguments if hasattr(tc.function, 'arguments') else "",
                            },
                        }
                        tool_calls_list.append(tc_dict)
                    if tool_calls_list:
                        choice_data["message"]["tool_calls"] = tool_calls_list
                choices_list.append(choice_data)
            resp_summary = {
                "id": request_id,
                "object": "chat.completion",
                "created": created_time,
                "model": model_name,
                "choices": choices_list,
                "usage": usage_dict,
                "stream": False,
            }
            try:
                payload_logger.info(
                    "openai.response",
                    extra={
                        "rid": rid_hint,
                        "endpoint": self.__class__.__name__,
                        # Pass dict directly for proper OTEL structured logging
                        "payload": resp_summary,
                    },
                )
            except Exception:
                pass

        # Log complete response if output logging is enabled
        if self.enable_log_outputs and self.request_logger:
            for choice in choices:
                # Log reasoning 
                if hasattr(choice.message, 'reasoning') and choice.message.reasoning:
                    self.request_logger.log_outputs(
                        request_id=request_id,
                        outputs=f"[reasoning] {choice.message.reasoning}",
                        output_token_ids=None,
                        finish_reason=None,  
                        is_streaming=False,
                        delta=False,
                    )

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
                    # Don't log output_token_ids to reduce log volume
                    self.request_logger.log_outputs(
                        request_id=request_id,
                        outputs=output_text,
                        output_token_ids=None,
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

                    with tokenizer_lock(tokenizer):
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
