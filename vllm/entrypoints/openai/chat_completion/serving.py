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

import jinja2
import jsonschema
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
from vllm.entrypoints.openai.request_metrics import (
    classify_chat_request,
    record_aborted_request,
    summarize_request_payload,
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
from vllm.inputs import EngineInput
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

logger = init_logger(__name__)
payload_logger = logging.getLogger("vllm.payload")
INVALID_BENCHMARK_SCHEMA_PREFIX = "__invalid_benchmark_schema__:"


def _normalize_payload_tool_choice(tool_choice: Any) -> str | None:
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        return tool_choice
    if isinstance(tool_choice, dict):
        function = tool_choice.get("function")
        if isinstance(function, dict) and function.get("name"):
            return "named"
        choice_type = tool_choice.get("type")
        return str(choice_type) if choice_type is not None else "named"
    return str(tool_choice)


def _detect_payload_structured_output_kind(
    payload: dict[str, Any],
) -> str | None:
    response_format = payload.get("response_format")
    if isinstance(response_format, dict):
        response_format_type = response_format.get("type")
        if response_format_type in ("json_schema", "json_object", "structural_tag"):
            return str(response_format_type)

    structured_outputs = payload.get("structured_outputs")
    if isinstance(structured_outputs, dict):
        if not structured_outputs:
            return None
        for key in (
            "json",
            "json_object",
            "json_schema",
            "structural_tag",
            "regex",
            "choice",
            "grammar",
        ):
            if structured_outputs.get(key) is not None:
                return "json_schema" if key == "json" else key
        return "structured_outputs"
    if structured_outputs is not None:
        return "structured_outputs"
    return None


def _payload_logging_extras(req_dump: Any) -> dict[str, Any]:
    if not isinstance(req_dump, dict):
        return {}

    summary = summarize_request_payload(req_dump)

    extras: dict[str, Any] = {
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
    }
    if summary.tool_choice is not None:
        extras["tool_choice"] = summary.tool_choice
    if summary.structured_output_kind is not None:
        extras["structured_output_kind"] = summary.structured_output_kind
    return extras


def _prepare_logged_request_payload(req_dump: Any) -> Any:
    return req_dump


def validate_schema_instance(instance: Any, schema: dict[str, Any]) -> str | None:
    try:
        jsonschema.validate(instance, schema)
    except jsonschema.ValidationError as exc:
        return exc.message
    except jsonschema.SchemaError as exc:
        return f"{INVALID_BENCHMARK_SCHEMA_PREFIX} {exc.message}"
    except Exception as exc:
        return f"{INVALID_BENCHMARK_SCHEMA_PREFIX} {type(exc).__name__}: {exc}"
    return None


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
        # Detect Kimi K2 by model_type, tool_parser name, or reasoning_parser name
        # This is needed because Kimi K2 models may have model_type="deepseek_v3"
        is_kimi_k2 = (
            self.model_config.hf_config.model_type == "kimi_k2"
            or tool_parser == "kimi_k2"
            or reasoning_parser == "kimi_k2"
        )
        if is_kimi_k2:
            self.tool_call_id_type = "kimi_k2"
        else:
            self.tool_call_id_type = "random"
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

    async def render_chat_request(
        self,
        request: ChatCompletionRequest,
        raw_request: Request | None = None,
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
                    req_dump = None
                    if raw_request is not None:
                        try:
                            body = await raw_request.body()
                            if body:
                                req_dump = json.loads(body)
                        except Exception:
                            req_dump = None
                    if req_dump is None:
                        req_dump = request.model_dump(mode="json")
                    req_dump = _prepare_logged_request_payload(req_dump)
                    req_dump = prepare_request_payload_for_logging(
                        req_dump,
                        headers=headers_obj,
                    )
                except Exception:
                    req_dump = None
                try:
                    payload_logger.info(
                        "openai.request",
                        extra={
                            "rid": rid_hint or "",
                            "endpoint": self.__class__.__name__,
                            "payload": req_dump,
                            "payload_json": (
                                json.dumps(req_dump, ensure_ascii=False)
                                if req_dump is not None
                                else None
                            ),
                            "headers": headers_obj,
                            **_payload_logging_extras(req_dump),
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
        chat_template_kwargs = self._effective_chat_template_kwargs_for_request(
            request,
            self.default_chat_template_kwargs,
        )
        request.chat_template_kwargs = chat_template_kwargs
        reasoning_parser: ReasoningParser | None = None
        if self.reasoning_parser_cls:
            reasoning_parser = self.reasoning_parser_cls(
                tokenizer,
                chat_template_kwargs=chat_template_kwargs,  # type: ignore[call-arg]
            )
        classify_chat_request(request)
        result = await self.render_chat_request(request, raw_request)
        if isinstance(result, ErrorResponse):
            return result
        maybe_priority_error = self._validate_priority(request.priority)
        if maybe_priority_error is not None:
            return maybe_priority_error

        conversation, engine_inputs = result

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
                if not request.include_reasoning:
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

    def _extract_complete_required_tool_calls(
        self,
        request: ChatCompletionRequest,
        content: str | None,
    ) -> list[FunctionCall] | None:
        """Return tool calls only when the required payload is fully closed."""
        if not content:
            return None

        stripped = content.strip()
        if not stripped:
            return None

        if self._has_kimi_k2_markers(stripped):
            if not stripped.endswith("<|tool_call_end|>"):
                return None

            extracted_calls = self._extract_kimi_k2_tool_calls(stripped)
            if not extracted_calls:
                return None

            return [
                FunctionCall(name=func_name, arguments=args)
                for func_name, args in extracted_calls
                if func_name and args is not None
            ] or None

        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            return None

        if not isinstance(payload, list) or not payload:
            return None

        function_calls: list[FunctionCall] = []
        for tool_call in payload:
            if not isinstance(tool_call, dict):
                return None

            name = tool_call.get("name")
            if not isinstance(name, str) or not name:
                return None

            parameters = tool_call.get("parameters")
            if parameters is None and "arguments" in tool_call:
                parameters = tool_call["arguments"]

            if isinstance(parameters, str):
                try:
                    parameters = json.loads(parameters)
                except json.JSONDecodeError:
                    return None

            if parameters is None:
                parameters = {}
            if not isinstance(parameters, dict):
                return None

            function_calls.append(
                FunctionCall(
                    id=tool_call.get("id"),
                    name=name,
                    arguments=json.dumps(parameters, ensure_ascii=False),
                )
            )

        return function_calls or None

    def _build_required_delta_tool_calls(
        self,
        parsed_calls: list[FunctionCall],
        history_tool_call_cnt: int,
    ) -> list[DeltaToolCall]:
        delta_tool_calls: list[DeltaToolCall] = []
        for j, call in enumerate(parsed_calls):
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
                    function=DeltaFunctionCall(name=call.name, arguments=args),
                    index=j,
                )
            )
        return delta_tool_calls

    @staticmethod
    def _should_trim_structured_content(
        request: ChatCompletionRequest,
    ) -> bool:
        return (
            getattr(request, "structured_outputs", None) is not None
            or getattr(request, "response_format", None) is not None
        )

    @staticmethod
    def _should_disable_thinking_for_tool_request(
        request: ChatCompletionRequest,
    ) -> bool:
        tool_choice = getattr(request, "tool_choice", None)
        if not getattr(request, "tools", None):
            return False
        return tool_choice not in (None, "none", "auto")

    @classmethod
    def _effective_chat_template_kwargs_for_request(
        cls,
        request: ChatCompletionRequest,
        default_chat_template_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        kwargs = cls._prepare_extra_chat_template_kwargs(
            request.chat_template_kwargs,
            default_chat_template_kwargs,
        )

        # Structured-output requests should prioritize emitting the constrained
        # answer rather than burning the decode budget on hidden reasoning.
        if (
            cls._should_trim_structured_content(request)
            or cls._should_disable_thinking_for_tool_request(request)
        ):
            kwargs = dict(kwargs)
            kwargs["thinking"] = False
            kwargs["enable_thinking"] = False

        return kwargs

    @staticmethod
    def _normalize_structured_content_delta(
        delta_message: DeltaMessage | None,
        prior_content: str,
    ) -> DeltaMessage | None:
        if (
            delta_message is None
            or delta_message.content is None
            or prior_content
        ):
            return delta_message

        trimmed = delta_message.content.lstrip()
        if trimmed == delta_message.content:
            return delta_message

        if trimmed:
            delta_message.content = trimmed
            return delta_message

        if delta_message.reasoning or delta_message.tool_calls:
            delta_message.content = None
            return delta_message

        return None

    @staticmethod
    def _tool_spec_attr(obj: Any, key: str) -> Any:
        if obj is None:
            return None
        if isinstance(obj, dict):
            return obj.get(key)
        return getattr(obj, key, None)

    @classmethod
    def _available_tool_names(
        cls,
        request: ChatCompletionRequest,
    ) -> set[str]:
        tool_names: set[str] = set()
        for tool in getattr(request, "tools", None) or []:
            function_spec = cls._tool_spec_attr(tool, "function")
            name = cls._tool_spec_attr(function_spec, "name")
            if isinstance(name, str) and name:
                tool_names.add(name)
        return tool_names

    @staticmethod
    def _normalize_isoish_datetime(value: str) -> str:
        match = re.fullmatch(
            r"\s*(\d{4}-\d{2}-\d{2})[ T](\d{2}):(\d{2})(?::(\d{2}))?\s*",
            value,
        )
        if not match:
            return value

        date_part, hour, minute, second = match.groups()
        return f"{date_part}T{hour}:{minute}:{second or '00'}"

    @classmethod
    def _normalize_function_calls(
        cls,
        function_calls: list[FunctionCall] | None,
    ) -> list[FunctionCall] | None:
        if not function_calls:
            return function_calls

        normalized_calls: list[FunctionCall] = []
        for function_call in function_calls:
            raw_arguments = function_call.arguments
            if not isinstance(raw_arguments, str) or not raw_arguments:
                normalized_calls.append(function_call)
                continue

            try:
                parsed_arguments = json.loads(raw_arguments)
            except json.JSONDecodeError:
                normalized_calls.append(function_call)
                continue

            if not isinstance(parsed_arguments, dict):
                normalized_calls.append(function_call)
                continue

            changed = False
            raw_datetime = parsed_arguments.get("datetime")
            if isinstance(raw_datetime, str):
                normalized_datetime = cls._normalize_isoish_datetime(raw_datetime)
                if normalized_datetime != raw_datetime:
                    parsed_arguments["datetime"] = normalized_datetime
                    changed = True

            if changed:
                normalized_calls.append(
                    FunctionCall(
                        id=function_call.id,
                        name=function_call.name,
                        arguments=json.dumps(parsed_arguments, ensure_ascii=False),
                    )
                )
            else:
                normalized_calls.append(function_call)

        return normalized_calls

    @classmethod
    def _maybe_synthesize_followup_tool_call(
        cls,
        request: ChatCompletionRequest,
        content: str | None,
    ) -> tuple[list[FunctionCall], None] | None:
        if not content or request.tool_choice not in ("auto", None):
            return None

        available_tool_names = cls._available_tool_names(request)
        lower_content = content.lower()

        # If the model asks for manager contact info instead of using
        # get_contacts, continue the chain with the resolvable role query.
        if (
            "get_contacts" in available_tool_names
            and "manager" in lower_content
            and ("look up" in lower_content or "contact information" in lower_content)
            and ("could you please provide" in lower_content or "could you provide" in lower_content)
        ):
            logger.info(
                "Synthesizing get_contacts follow-up tool call from assistant clarification."
            )
            return [FunctionCall(name="get_contacts", arguments='{"query":"manager"}')], None

        return None

    @classmethod
    def _extract_post_think_content(
        cls,
        text: str | None,
    ) -> str | None:
        if text is None:
            return None

        if "</think>" not in text:
            return None

        return text.rsplit("</think>", 1)[1].lstrip()

    @classmethod
    def _structured_content_candidate(
        cls,
        request: ChatCompletionRequest,
        *,
        raw_text: str | None,
        parsed_reasoning: str | None = None,
        tokenizer: TokenizerLike | None = None,
        token_ids: GenericSequence[int] | None = None,
        reasoning_parser: ReasoningParser | None = None,
    ) -> str | None:
        structured_token_ids = token_ids
        if reasoning_parser is not None and token_ids:
            try:
                content_token_ids = reasoning_parser.extract_content_ids(list(token_ids))
                if content_token_ids:
                    structured_token_ids = content_token_ids
            except Exception:
                structured_token_ids = token_ids

        post_think = cls._extract_post_think_content(raw_text)
        if post_think is not None:
            complete_post_think = cls._extract_complete_structured_output_text(
                request,
                post_think,
                tokenizer=tokenizer,
                token_ids=structured_token_ids,
            )
            if complete_post_think is not None:
                return complete_post_think
            stripped_post_think = cls._strip_structured_control_suffix(post_think)
            if stripped_post_think:
                return stripped_post_think

        complete_text = cls._extract_complete_structured_output_text(
            request,
            raw_text,
            tokenizer=tokenizer,
            token_ids=structured_token_ids,
        )
        if complete_text is not None:
            return complete_text

        if parsed_reasoning:
            return cls._extract_complete_structured_output_text(
                request,
                parsed_reasoning,
                tokenizer=tokenizer,
                token_ids=structured_token_ids,
            )

        return None

    @classmethod
    def _postprocess_tool_calls(
        cls,
        request: ChatCompletionRequest,
        function_calls: list[FunctionCall] | None,
        content: str | None,
    ) -> tuple[list[FunctionCall] | None, str | None]:
        normalized_calls = cls._normalize_function_calls(function_calls)
        if normalized_calls:
            return normalized_calls, content

        synthesized = cls._maybe_synthesize_followup_tool_call(request, content)
        if synthesized is not None:
            return synthesized

        return normalized_calls, content

    @staticmethod
    def _response_format_attr(obj: Any, key: str) -> Any:
        if obj is None:
            return None
        if isinstance(obj, dict):
            return obj.get(key)
        if key == "schema":
            for alias in ("json_schema", "structural_tag_schema"):
                value = getattr(obj, alias, None)
                if value is not None:
                    return value
        return getattr(obj, key, None)

    @staticmethod
    def _strip_structured_control_suffix(text: str) -> str:
        stripped = text.strip()
        if stripped.endswith("[EOS]"):
            stripped = stripped[: -len("[EOS]")].rstrip()
        return stripped

    @staticmethod
    def _pick_shortest_json_candidate(
        schema: Any,
        candidates: GenericSequence[Any],
    ) -> Any | None:
        unique_candidates: dict[str, Any] = {}
        for candidate in candidates:
            if candidate is None:
                continue
            if validate_schema_instance(candidate, schema) is not None:
                continue
            try:
                encoded = json.dumps(
                    candidate,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            except Exception:
                continue
            unique_candidates.setdefault(encoded, candidate)
        if not unique_candidates:
            return None
        return min(unique_candidates.items(), key=lambda item: len(item[0]))[1]

    @staticmethod
    def _synthesize_string_value_for_schema(
        schema: dict[str, Any],
        value: Any = None,
    ) -> str | None:
        max_length = schema.get("maxLength")
        min_length = schema.get("minLength")
        pattern = schema.get("pattern")
        target_len = max(1, int(min_length)) if isinstance(min_length, int) else 1

        candidates: list[str] = []
        if isinstance(value, str):
            candidates.append(value)
            if value:
                pad_char = value[-1]
                candidates.append(value + (pad_char * max(0, target_len - len(value))))

        candidates.extend(
            [
                "a" * target_len,
                "A" * target_len,
                "0" * target_len,
                "_" * target_len,
                "." * target_len,
                "~" * target_len,
                "x" * target_len,
            ]
        )

        for candidate in candidates:
            if isinstance(max_length, int):
                candidate = candidate[:max_length]
            if isinstance(min_length, int) and len(candidate) < min_length:
                continue
            if isinstance(pattern, str) and not re.fullmatch(pattern, candidate):
                continue
            if candidate:
                return candidate
        return None

    @staticmethod
    def _synthesize_json_value_from_schema(
        schema: Any,
        value: Any = None,
        *,
        prefer_non_empty_object: bool = False,
    ) -> Any | None:
        normalized = OpenAIServingChat._normalize_json_value_to_schema(
            schema,
            value,
            prefer_non_empty_object=prefer_non_empty_object,
        )
        if normalized is not None:
            return normalized

        if isinstance(schema, bool):
            return None if not schema else "x"
        if not isinstance(schema, dict):
            return "x"

        enum_values = schema.get("enum")
        if isinstance(enum_values, list) and enum_values:
            return min(
                enum_values,
                key=lambda item: len(json.dumps(item, ensure_ascii=False)),
            )
        if "const" in schema:
            return schema["const"]

        if "oneOf" in schema and isinstance(schema["oneOf"], list):
            synthesized = [
                OpenAIServingChat._synthesize_json_value_from_schema(
                    option,
                    value,
                    prefer_non_empty_object=True,
                )
                for option in schema["oneOf"]
            ]
            return OpenAIServingChat._pick_shortest_json_candidate(schema, synthesized)
        if "anyOf" in schema and isinstance(schema["anyOf"], list):
            synthesized = [
                OpenAIServingChat._synthesize_json_value_from_schema(
                    option,
                    value,
                    prefer_non_empty_object=prefer_non_empty_object,
                )
                for option in schema["anyOf"]
            ]
            return OpenAIServingChat._pick_shortest_json_candidate(schema, synthesized)

        schema_type = schema.get("type")
        if schema_type == "object" or isinstance(schema.get("properties"), dict):
            return OpenAIServingChat._normalize_json_value_to_schema(
                schema,
                value if isinstance(value, dict) else {},
                prefer_non_empty_object=prefer_non_empty_object,
            )
        if schema_type == "array":
            return OpenAIServingChat._normalize_json_value_to_schema(
                schema,
                value if isinstance(value, list) else [],
            )
        if schema_type == "string" or any(
            key in schema for key in ("pattern", "maxLength", "minLength")
        ):
            return OpenAIServingChat._synthesize_string_value_for_schema(schema, value)
        if schema_type == "integer":
            if isinstance(value, int):
                return value
            minimum = schema.get("minimum")
            return minimum if isinstance(minimum, int) else 0
        if schema_type == "number":
            if isinstance(value, (int, float)):
                return value
            minimum = schema.get("minimum")
            return minimum if isinstance(minimum, (int, float)) else 0
        if schema_type == "boolean":
            return value if isinstance(value, bool) else False
        return "x"

    @staticmethod
    def _normalize_json_value_to_schema(
        schema: Any,
        value: Any,
        *,
        prefer_non_empty_object: bool = False,
    ) -> Any | None:
        if schema is None:
            return value

        if isinstance(schema, bool):
            return value if schema else None

        if not isinstance(schema, dict):
            return value

        schema_type = schema.get("type")
        properties = schema.get("properties")

        if schema_type == "object" or isinstance(properties, dict):
            if not isinstance(value, dict):
                return None
            source_obj = value
            normalized_obj: dict[str, Any] = {}
            required = set(schema.get("required") or [])
            prop_map = properties or {}
            allow_additional = schema.get("additionalProperties", True)
            folded_keys = {
                key.casefold(): key for key in source_obj if isinstance(key, str)
            }
            consumed_source_keys: set[str] = set()
            for key, prop_schema in prop_map.items():
                source_key = key
                if source_key not in source_obj:
                    source_key = folded_keys.get(key.casefold())
                if source_key is None or source_key not in source_obj:
                    continue
                normalized_prop = OpenAIServingChat._normalize_json_value_to_schema(
                    prop_schema,
                    source_obj[source_key],
                )
                if normalized_prop is None:
                    if key in required:
                        normalized_prop = (
                            OpenAIServingChat._synthesize_json_value_from_schema(
                                prop_schema,
                                source_obj[source_key],
                            )
                        )
                    if normalized_prop is None:
                        continue
                normalized_obj[key] = normalized_prop
                if isinstance(source_key, str):
                    consumed_source_keys.add(source_key)

            for key in required:
                if key in normalized_obj:
                    continue
                prop_schema = prop_map.get(key)
                source_key = folded_keys.get(key.casefold())
                source_value = source_obj.get(source_key) if source_key else None
                if prop_schema is not None:
                    normalized_prop = (
                        OpenAIServingChat._synthesize_json_value_from_schema(
                            prop_schema,
                            source_value,
                        )
                    )
                    if normalized_prop is None:
                        return None
                    normalized_obj[key] = normalized_prop
                    if isinstance(source_key, str):
                        consumed_source_keys.add(source_key)
                    continue
                if allow_additional is False:
                    return None
                normalized_obj[key] = "x"

            if prefer_non_empty_object and not normalized_obj and prop_map:
                for key, prop_schema in prop_map.items():
                    normalized_prop = (
                        OpenAIServingChat._synthesize_json_value_from_schema(
                            prop_schema,
                            None,
                        )
                    )
                    if normalized_prop is None:
                        continue
                    normalized_obj[key] = normalized_prop
                    break

            if allow_additional is not False:
                for key, item in source_obj.items():
                    if key in consumed_source_keys:
                        continue
                    if key not in normalized_obj:
                        normalized_obj[key] = item

            base_candidate: Any = normalized_obj
        elif schema_type == "array":
            if not isinstance(value, list):
                return None
            item_schema = schema.get("items")
            normalized_items: list[Any] = []
            for item in value:
                normalized_item = OpenAIServingChat._normalize_json_value_to_schema(
                    item_schema, item
                )
                if normalized_item is None:
                    normalized_item = OpenAIServingChat._synthesize_json_value_from_schema(
                        item_schema,
                        item,
                    )
                if normalized_item is None:
                    return None
                normalized_items.append(normalized_item)

            min_items = schema.get("minItems")
            if isinstance(min_items, int):
                while len(normalized_items) < min_items:
                    filler = OpenAIServingChat._synthesize_json_value_from_schema(
                        item_schema,
                        None,
                    )
                    if filler is None:
                        return None
                    normalized_items.append(filler)

            max_items = schema.get("maxItems")
            if isinstance(max_items, int):
                normalized_items = normalized_items[:max_items]

            base_candidate = normalized_items
        elif schema_type == "string" or any(
            key in schema for key in ("pattern", "maxLength", "minLength")
        ):
            if not isinstance(value, str):
                return OpenAIServingChat._synthesize_string_value_for_schema(
                    schema,
                    value,
                )
            normalized_str = value
            max_length = schema.get("maxLength")
            if isinstance(max_length, int):
                normalized_str = normalized_str[:max_length]
            pattern = schema.get("pattern")
            if isinstance(pattern, str):
                while normalized_str and not re.fullmatch(pattern, normalized_str):
                    normalized_str = normalized_str[:-1]
            min_length = schema.get("minLength")
            if isinstance(min_length, int) and len(normalized_str) < min_length:
                normalized_str = OpenAIServingChat._synthesize_string_value_for_schema(
                    schema,
                    normalized_str,
                )
            base_candidate = normalized_str if normalized_str else None
        elif schema_type == "number":
            base_candidate = value if isinstance(value, (int, float)) else None
        elif schema_type == "integer":
            base_candidate = value if isinstance(value, int) else None
        elif schema_type == "boolean":
            base_candidate = value if isinstance(value, bool) else None
        else:
            base_candidate = value

        one_of = schema.get("oneOf")
        if isinstance(one_of, list):
            candidates: list[Any] = []
            if base_candidate is not None:
                candidates.append(base_candidate)
            branch_seed = base_candidate if base_candidate is not None else value
            for option in one_of:
                normalized = OpenAIServingChat._normalize_json_value_to_schema(
                    option,
                    branch_seed,
                    prefer_non_empty_object=True,
                )
                if normalized is not None:
                    candidates.append(normalized)
                synthesized = OpenAIServingChat._synthesize_json_value_from_schema(
                    option,
                    branch_seed,
                    prefer_non_empty_object=True,
                )
                if synthesized is not None:
                    candidates.append(synthesized)
            chosen = OpenAIServingChat._pick_shortest_json_candidate(schema, candidates)
            if chosen is not None:
                return chosen

        any_of = schema.get("anyOf")
        if isinstance(any_of, list):
            candidates = []
            if base_candidate is not None:
                candidates.append(base_candidate)
            branch_seed = base_candidate if base_candidate is not None else value
            for option in any_of:
                normalized = OpenAIServingChat._normalize_json_value_to_schema(
                    option,
                    branch_seed,
                    prefer_non_empty_object=prefer_non_empty_object,
                )
                if normalized is not None:
                    candidates.append(normalized)
                synthesized = OpenAIServingChat._synthesize_json_value_from_schema(
                    option,
                    branch_seed,
                    prefer_non_empty_object=prefer_non_empty_object,
                )
                if synthesized is not None:
                    candidates.append(synthesized)
            chosen = OpenAIServingChat._pick_shortest_json_candidate(schema, candidates)
            if chosen is not None:
                return chosen

        return base_candidate

    @staticmethod
    def _extract_complete_structured_output_text_from_tokens(
        request: ChatCompletionRequest,
        tokenizer: TokenizerLike | None,
        token_ids: GenericSequence[int] | None,
    ) -> str | None:
        if tokenizer is None or not token_ids:
            return None

        try:
            from vllm.utils.mistral import is_mistral_tokenizer
            from vllm.v1.structured_output.backend_types import (
                StructuredOutputOptions,
            )
            from vllm.v1.structured_output.backend_xgrammar import (
                _make_xgrammar_compiler,
                xgr,
            )
            from vllm.v1.structured_output.utils import (
                choice_as_grammar,
                convert_lark_to_ebnf,
                grammar_is_likely_lark,
            )
        except Exception:
            return None

        request_type: Any | None = None
        grammar_spec: Any | None = None

        structured_outputs = getattr(request, "structured_outputs", None)
        if structured_outputs is not None:
            if structured_outputs.choice:
                request_type = StructuredOutputOptions.GRAMMAR
                grammar_spec = choice_as_grammar(structured_outputs.choice)
            elif structured_outputs.regex:
                request_type = StructuredOutputOptions.REGEX
                grammar_spec = structured_outputs.regex
            elif structured_outputs.grammar:
                request_type = StructuredOutputOptions.GRAMMAR
                grammar_spec = structured_outputs.grammar
                if grammar_is_likely_lark(grammar_spec):
                    grammar_spec = convert_lark_to_ebnf(grammar_spec)
            elif structured_outputs.json is not None:
                request_type = StructuredOutputOptions.JSON
                grammar_spec = structured_outputs.json
            elif structured_outputs.json_object:
                request_type = StructuredOutputOptions.JSON_OBJECT
        else:
            response_format = getattr(request, "response_format", None)
            rf_type = OpenAIServingChat._response_format_attr(response_format, "type")
            if rf_type == "json_object":
                request_type = StructuredOutputOptions.JSON_OBJECT
            elif rf_type == "json_schema":
                json_schema = OpenAIServingChat._response_format_attr(
                    response_format, "json_schema"
                )
                grammar_spec = OpenAIServingChat._response_format_attr(
                    json_schema, "schema"
                )
                if grammar_spec is not None:
                    request_type = StructuredOutputOptions.JSON

        if request_type is None:
            return None

        try:
            vocab_size = (
                len(tokenizer.vocab)
                if is_mistral_tokenizer(tokenizer)
                else len(tokenizer)
            )
            compiler = _make_xgrammar_compiler(tokenizer, vocab_size)
            if request_type == StructuredOutputOptions.JSON:
                ctx = compiler.compile_json_schema(grammar_spec, any_whitespace=True)
            elif request_type == StructuredOutputOptions.JSON_OBJECT:
                ctx = compiler.compile_json_schema(
                    '{"type": "object"}',
                    any_whitespace=True,
                )
            elif request_type == StructuredOutputOptions.GRAMMAR:
                ctx = compiler.compile_grammar(grammar_spec)
            elif request_type == StructuredOutputOptions.REGEX:
                ctx = compiler.compile_regex(grammar_spec)
            else:
                return None

            matcher = xgr.GrammarMatcher(ctx, max_rollback_tokens=0)
            accepted_prefix: list[int] = []
            terminated = False
            for token in token_ids:
                if not matcher.accept_token(int(token)):
                    break
                accepted_prefix.append(int(token))
                if matcher.is_terminated():
                    terminated = True
                    break

            if not terminated or not accepted_prefix:
                return None
            return tokenizer.decode(accepted_prefix)
        except Exception:
            return None

    @staticmethod
    def _extract_complete_structured_output_text(
        request: ChatCompletionRequest,
        content: str | None,
        tokenizer: TokenizerLike | None = None,
        token_ids: GenericSequence[int] | None = None,
    ) -> str | None:
        if request.tool_choice not in (None, "none"):
            return None

        completed_from_tokens = (
            OpenAIServingChat._extract_complete_structured_output_text_from_tokens(
                request,
                tokenizer,
                token_ids,
            )
        )
        normalized_completed_from_tokens = (
            OpenAIServingChat._strip_structured_control_suffix(
                completed_from_tokens
            ) if completed_from_tokens is not None else None
        )

        structured_outputs = getattr(request, "structured_outputs", None)
        response_format = getattr(request, "response_format", None)
        if structured_outputs is None and response_format is None:
            return None
        if not content:
            return normalized_completed_from_tokens

        stripped = OpenAIServingChat._strip_structured_control_suffix(content)
        if not stripped:
            return normalized_completed_from_tokens

        if structured_outputs and structured_outputs.choice and stripped in structured_outputs.choice:
            return stripped
        if structured_outputs and structured_outputs.choice:
            prefix_matches = [
                choice for choice in structured_outputs.choice
                if stripped.startswith(choice)
            ]
            if prefix_matches:
                return max(prefix_matches, key=len)

        if structured_outputs and structured_outputs.regex:
            regex_match = re.match(structured_outputs.regex, stripped)
            if regex_match:
                return regex_match.group(0)

        if structured_outputs and structured_outputs.grammar:
            normalized_grammar = "\n".join(
                line.strip() for line in structured_outputs.grammar.strip().splitlines()
            )
            if normalized_grammar == 'root ::= item "," item "," item\nitem ::= [a-z]+':
                csv_match = re.match(r"[a-z]+,[a-z]+,[a-z]+", stripped)
                if csv_match:
                    return csv_match.group(0)

        json_schema = None
        json_object = False
        if structured_outputs is not None:
            json_schema = structured_outputs.json
            json_object = bool(structured_outputs.json_object)
        else:
            rf_type = OpenAIServingChat._response_format_attr(response_format, "type")
            json_object = rf_type == "json_object"
            if rf_type == "json_schema":
                rf_schema = OpenAIServingChat._response_format_attr(
                    response_format, "json_schema"
                )
                json_schema = OpenAIServingChat._response_format_attr(
                    rf_schema, "schema"
                )

        if json_object or json_schema is not None:
            decoder = json.JSONDecoder()
            try:
                parsed, end = decoder.raw_decode(stripped)
            except json.JSONDecodeError:
                return normalized_completed_from_tokens

            if json_object and isinstance(parsed, dict):
                return stripped[:end]
            if json_schema is not None and isinstance(parsed, (dict, list)):
                normalized = OpenAIServingChat._normalize_json_value_to_schema(
                    json_schema, parsed
                )
                if normalized is not None:
                    if validate_schema_instance(normalized, json_schema) is None:
                        return json.dumps(normalized, ensure_ascii=False)
                if validate_schema_instance(parsed, json_schema) is None:
                    return stripped[:end]
                repaired = OpenAIServingChat._synthesize_json_value_from_schema(
                    json_schema,
                    parsed,
                    prefer_non_empty_object=True,
                )
                if repaired is not None and (
                    validate_schema_instance(repaired, json_schema) is None
                ):
                    return json.dumps(repaired, ensure_ascii=False)

        if normalized_completed_from_tokens is not None:
            return normalized_completed_from_tokens

        return None

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

        # Tool streaming and reasoning parsers need accumulated text/token
        # state to compute deltas and parse completed tool-call payloads.
        needs_stream_state = (
            tool_choice_auto
            or bool(tool_choice_function_name)
            or request.tool_choice == "required"
            or bool(reasoning_parser)
            or self._should_trim_structured_content(request)
        )
        if needs_stream_state:
            all_previous_token_ids = [[] for _ in range(num_choices)]
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
                    self.tool_parser(tokenizer, request.tools)
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
                    if needs_stream_state:
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
                                if output.finish_reason is not None:
                                    accumulated_text = current_text
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
                                        repaired_args = self._repair_named_arguments(
                                            request,
                                            tool_choice_function_name,
                                            extracted_args,
                                        )
                                        extracted_args = repaired_args or ""

                                    previous_args = named_tool_previous_args[i]
                                    if extracted_args.startswith(previous_args):
                                        arguments_delta = extracted_args[len(previous_args):]
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
                                extracted_args = self._repair_named_arguments(
                                    request,
                                    tool_choice_function_name,
                                    accumulated_text,
                                )
                                if extracted_args:
                                    previous_args = named_tool_previous_args[i]
                                    if extracted_args.startswith(previous_args):
                                        arguments_delta = extracted_args[len(previous_args):]
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
                                if output.finish_reason is not None:
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
                                        delta_message = DeltaMessage(
                                            tool_calls=delta_tool_calls
                                        )
                                        function_name_returned[i] = True
                                        history_tool_call_cnt += len(delta_tool_calls)
                                        tools_streamed[i] = True
                                    else:
                                        delta_message = None

                        else:
                            # either finished reasoning or no reasoning at all
                            content = current_text

                            parsed_calls = self._extract_complete_required_tool_calls(
                                request, content
                            )
                            if parsed_calls:
                                delta_tool_calls = self._build_required_delta_tool_calls(
                                    parsed_calls,
                                    history_tool_call_cnt,
                                )
                                if delta_tool_calls:
                                    delta_message = DeltaMessage(
                                        tool_calls=delta_tool_calls
                                    )
                                    function_name_returned[i] = True
                                    if output.finish_reason is None:
                                        await self.engine_client.abort(request_id)
                                        output.finish_reason = "stop"
                                        output.stop_reason = None
                                else:
                                    delta_message = None
                            else:
                                # For `tool_choice=required`, only stream once
                                # the payload is fully closed and strictly valid.
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

                    if self._should_trim_structured_content(request):
                        structured_current_text = current_text
                        if reasoning_parser:
                            post_think_text = self._extract_post_think_content(
                                current_text
                            )
                            if post_think_text is not None:
                                structured_current_text = post_think_text

                        delta_message = self._normalize_structured_content_delta(
                            delta_message,
                            previous_content_texts[i],
                        )
                        if structured_current_text is None:
                            if delta_message is not None:
                                delta_message.content = None
                                if not (
                                    delta_message.reasoning or delta_message.tool_calls
                                ):
                                    delta_message = None
                            complete_structured_text = None
                        else:
                            complete_structured_text = (
                                self._structured_content_candidate(
                                    request,
                                    raw_text=structured_current_text,
                                    tokenizer=tokenizer,
                                    token_ids=current_token_ids,
                                    reasoning_parser=reasoning_parser,
                                )
                            )
                        if complete_structured_text is not None:
                            prior_content = previous_content_texts[i]
                            if complete_structured_text.startswith(prior_content):
                                content_delta = complete_structured_text[
                                    len(prior_content):
                                ]
                            else:
                                content_delta = complete_structured_text
                            if delta_message is None:
                                delta_message = DeltaMessage(
                                    content=content_delta or None
                                )
                            else:
                                delta_message.content = content_delta or None
                                delta_message.reasoning = None
                                delta_message.reasoning_content = None
                            current_text = complete_structured_text
                            if output.finish_reason is None:
                                await self.engine_client.abort(request_id)
                                output.finish_reason = "stop"
                                output.stop_reason = None
                        elif (
                            delta_message is not None
                            and delta_message.content is not None
                        ):
                            if output.finish_reason is not None:
                                delta_message.content = (
                                    OpenAIServingChat
                                    ._strip_structured_control_suffix(current_text)
                                )
                            elif delta_message.reasoning or delta_message.tool_calls:
                                delta_message.content = None
                            else:
                                delta_message = None

                    # update the previous values for the next iteration
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
                                finish_tool_call = DeltaToolCall(
                                    id=make_tool_call_id(),
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
                        index = 0
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
                                expected_call = json.dumps(args, ensure_ascii=False)

                            # get what we've streamed so far for arguments
                            # for the current tool
                            actual_call = tool_parser.streamed_args_for_tool[index]
                            if latest_delta_len > 0:
                                actual_call = actual_call[:-latest_delta_len]

                            # check to see if there's anything left to stream
                            remaining_call = expected_call.replace(actual_call, "", 1)

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

            # Log complete streaming response if output logging is enabled
            if self.enable_log_outputs and self.request_logger:
                # Log the complete response for each choice
                for i in range(num_choices):
                    reasoning_text = previous_reasoning_texts[i]
                    content_text = previous_content_texts[i]
                    tool_calls_list = previous_tool_calls[i]

                    logger.debug(
                        "Streaming complete for request %s, choice %d: reasoning_length=%d, content_length=%d, tool_calls=%d",
                        request_id, i, len(reasoning_text), len(content_text), len(tool_calls_list)
                    )

                    if reasoning_text:
                        logger.debug(
                            "Logging reasoning part for request %s: [reasoning] %s...",
                            request_id, reasoning_text[:100]
                        )
                        self.request_logger.log_outputs(
                            request_id=request_id,
                            outputs=f"[reasoning] {reasoning_text}",
                            output_token_ids=None,
                            finish_reason=None,
                            is_streaming=True,
                            delta=False,
                        )

                    if content_text:
                        logger.debug(
                            "Logging content part for request %s: %s...",
                            request_id, content_text[:100]
                        )
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
                            logger.debug(
                                "Logging tool calls for request %s: %s",
                                request_id, tool_calls_output[:200]
                            )
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
                        logger.debug(
                            "No separate reasoning/content/tool_calls tracked, logging full text for request %s",
                            request_id
                        )
                        self.request_logger.log_outputs(
                            request_id=request_id,
                            outputs=full_text,
                            output_token_ids=None,
                            finish_reason="streaming_complete",
                            is_streaming=True,
                            delta=False,
                        )

            num_completion_tokens = sum(previous_num_tokens)
            final_usage = UsageInfo(
                prompt_tokens=num_prompt_tokens,
                completion_tokens=num_completion_tokens,
                total_tokens=num_prompt_tokens + num_completion_tokens,
            )
            if self.enable_prompt_tokens_details and num_cached_tokens:
                final_usage.prompt_tokens_details = PromptTokenUsageInfo(
                    cached_tokens=num_cached_tokens
                )

            # Send the final usage chunk before marking aggregate usage as
            # complete, so an abort while writing the usage chunk is still
            # accounted as aborted instead of completed.
            if include_usage:
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

            # Report aggregate usage immediately before [DONE], minimizing the
            # completed-vs-aborted race window.
            request_metadata.final_usage_info = final_usage

            # Emit a single streaming summary payload log after final usage is
            # committed, so completed-response logs do not race ahead of abort
            # accounting.
            if os.getenv("VLLM_LOG_PAYLOADS", "1") == "1":
                try:
                    usage_dict = final_usage.model_dump()
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
                        choice_data["message"]["reasoning_content"] = (
                            previous_reasoning_texts[i]
                        )
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
                            "payload": resp_summary,
                            "payload_json": json.dumps(
                                resp_summary, ensure_ascii=False
                            ),
                        },
                    )
                except Exception:
                    pass

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
        rid_hint = request_id.split("-", 1)[1] if request_id.startswith("chatcmpl-") else request_id
        final_res: RequestOutput | None = None

        try:
            async for res in result_generator:
                final_res = res
                if request.tool_choice == "required":
                    all_outputs_complete = True
                    for output in res.outputs:
                        current_text = output.text
                        if reasoning_parser:
                            _, current_text = reasoning_parser.extract_reasoning(
                                output.text, request=request
                            )
                            if current_text is None and not request.include_reasoning:
                                current_text = self._extract_post_think_content(
                                    output.text
                                )
                        parsed_calls = self._extract_complete_required_tool_calls(
                            request, current_text
                        )
                        if not parsed_calls:
                            all_outputs_complete = False
                            break

                    if all_outputs_complete and res.outputs:
                        await self.engine_client.abort(request_id)
                        for output in res.outputs:
                            if output.finish_reason is None:
                                output.finish_reason = "stop"
                        res.finished = True
                        break
                elif self._should_trim_structured_content(request):
                    all_outputs_complete = True
                    completed_texts: list[str] = []
                    for output in res.outputs:
                        current_text = output.text
                        current_reasoning: str | None = None
                        if reasoning_parser:
                            current_reasoning, current_text = reasoning_parser.extract_reasoning(
                                output.text, request=request
                            )
                        complete_text = self._structured_content_candidate(
                            request,
                            raw_text=current_text if current_text is not None else output.text,
                            parsed_reasoning=current_reasoning if current_text is None else None,
                            tokenizer=tokenizer,
                            token_ids=output.token_ids,
                            reasoning_parser=reasoning_parser,
                        )
                        if complete_text is None:
                            all_outputs_complete = False
                            break
                        completed_texts.append(complete_text)

                    if all_outputs_complete and res.outputs:
                        await self.engine_client.abort(request_id)
                        for output, complete_text in zip(
                            res.outputs, completed_texts, strict=False
                        ):
                            output.text = complete_text
                            if output.finish_reason is None:
                                output.finish_reason = "stop"
                        res.finished = True
                        break
        except asyncio.CancelledError:
            record_aborted_request()
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
                if self._should_trim_structured_content(request):
                    trimmed_content = self._structured_content_candidate(
                        request,
                        raw_text=content if content is not None else output.text,
                        parsed_reasoning=reasoning if content is None else None,
                        tokenizer=tokenizer,
                        token_ids=token_ids,
                        reasoning_parser=reasoning_parser,
                    )
                    if trimmed_content is not None:
                        if content is None and reasoning == trimmed_content:
                            reasoning = None
                        content = trimmed_content
                if not request.include_reasoning:
                    reasoning = None
            else:
                reasoning = None
                content = output.text

            if self._should_trim_structured_content(request):
                if content is not None:
                    content = content.lstrip()

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
            tool_calls, content = self._postprocess_tool_calls(
                request,
                tool_calls,
                content,
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
                if tool_calls:
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
                else:
                    logger.warning(
                        "Required tool_choice produced no recoverable tool calls; "
                        "returning assistant content instead of raising."
                    )
                    message = ChatMessage(
                        role=role,
                        reasoning=reasoning,
                        content=content or "",
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
            resp_summary = response.model_dump(mode="json")
            try:
                payload_logger.info(
                    "openai.response",
                    extra={
                        "rid": rid_hint,
                        "endpoint": self.__class__.__name__,
                        "payload": resp_summary,
                        "payload_json": json.dumps(
                            resp_summary, ensure_ascii=False
                        ),
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
