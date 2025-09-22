# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json
import time
from collections.abc import AsyncGenerator, AsyncIterator
from collections.abc import Sequence as GenericSequence
from typing import Callable, Final, Optional, Union

import jinja2
import partial_json_parser
import regex as re
from fastapi import Request
from openai_harmony import Message as OpenAIMessage
from pydantic import TypeAdapter
from partial_json_parser.core.options import Allow

from vllm.config import ModelConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import (ChatTemplateContentFormatOption,
                                         ConversationMessage,
                                         get_history_tool_calls_cnt,
                                         make_tool_call_id)
from vllm.entrypoints.harmony_utils import (
    get_developer_message, get_stop_tokens_for_assistant_actions,
    get_streamable_parser_for_assistant, get_system_message, parse_chat_input,
    parse_chat_output, render_for_completion)
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.protocol import (
    ChatCompletionLogProb, ChatCompletionLogProbs,
    ChatCompletionLogProbsContent, ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest, ChatCompletionResponse,
    ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse, ChatMessage, DeltaFunctionCall, DeltaMessage,
    DeltaToolCall, ErrorResponse, FunctionCall, FunctionDefinition,
    PromptTokenUsageInfo, RequestResponseMetadata, ToolCall, UsageInfo)
from vllm.entrypoints.openai.serving_engine import (OpenAIServing,
                                                    clamp_prompt_logprobs)
from vllm.entrypoints.openai.serving_models import OpenAIServingModels
from vllm.entrypoints.openai.tool_parsers import ToolParser, ToolParserManager
from vllm.entrypoints.openai.tool_parsers.mistral_tool_parser import (
    MistralToolCall)
from vllm.entrypoints.utils import get_max_tokens
from vllm.inputs.data import TokensPrompt as EngineTokensPrompt
from vllm.logger import init_logger
from vllm.logprobs import Logprob
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.reasoning import ReasoningParser, ReasoningParserManager
from vllm.sampling_params import BeamSearchParams, SamplingParams
from vllm.transformers_utils.tokenizer import AnyTokenizer, MistralTokenizer
from vllm.transformers_utils.tokenizers import (maybe_serialize_tool_calls,
                                                truncate_tool_call_ids,
                                                validate_request_params)
from vllm.utils import as_list

logger = init_logger(__name__)


class OpenAIServingChat(OpenAIServing):

    def __init__(
        self,
        engine_client: EngineClient,
        model_config: ModelConfig,
        models: OpenAIServingModels,
        response_role: str,
        *,
        request_logger: Optional[RequestLogger],
        chat_template: Optional[str],
        chat_template_content_format: ChatTemplateContentFormatOption,
        return_tokens_as_token_ids: bool = False,
        reasoning_parser: str = "",
        enable_auto_tools: bool = False,
        exclude_tools_when_tool_choice_none: bool = False,
        tool_parser: Optional[str] = None,
        enable_prompt_tokens_details: bool = False,
        enable_force_include_usage: bool = False,
        enable_log_outputs: bool = False,
        log_error_stack: bool = False,
    ) -> None:
        super().__init__(engine_client=engine_client,
                         model_config=model_config,
                         models=models,
                         request_logger=request_logger,
                         return_tokens_as_token_ids=return_tokens_as_token_ids,
                         enable_force_include_usage=enable_force_include_usage,
                         log_error_stack=log_error_stack)

        self.response_role = response_role
        self.chat_template = chat_template
        self.chat_template_content_format: Final = chat_template_content_format
        self.enable_log_outputs = enable_log_outputs

        # set up tool use
        self.enable_auto_tools: bool = enable_auto_tools
        if self.enable_auto_tools:
            logger.info(
                "\"auto\" tool choice has been enabled please note that while"
                " the parallel_tool_calls client option is preset for "
                "compatibility reasons, it will be ignored.")

        self.reasoning_parser: Optional[Callable[[AnyTokenizer],
                                                 ReasoningParser]] = None
        if reasoning_parser:
            try:
                self.reasoning_parser = (
                    ReasoningParserManager.get_reasoning_parser(
                        reasoning_parser))
                assert self.reasoning_parser is not None
            except Exception as e:
                raise TypeError(
                    f"{reasoning_parser=} has not been registered") from e
        self.tool_parser: Optional[Callable[[AnyTokenizer], ToolParser]] = None
        if self.enable_auto_tools:
            try:
                if (tool_parser == "pythonic" and
                        model_config.model.startswith("meta-llama/Llama-3.2")):
                    logger.warning(
                        "Llama3.2 models may struggle to emit valid pythonic"
                        " tool calls")
                self.tool_parser = ToolParserManager.get_tool_parser(
                    tool_parser)
            except Exception as e:
                raise TypeError("Error: --enable-auto-tool-choice requires "
                                f"tool_parser:'{tool_parser}' which has not "
                                "been registered") from e
        self.exclude_tools_when_tool_choice_none = (
            exclude_tools_when_tool_choice_none)

        self.enable_prompt_tokens_details = enable_prompt_tokens_details
        self.enable_force_include_usage = enable_force_include_usage
        self.default_sampling_params = (
            self.model_config.get_diff_sampling_param())
        if self.default_sampling_params:
            source = self.model_config.generation_config
            source = "model" if source == "auto" else source
            logger.info("Using default chat sampling params from %s: %s",
                        source, self.default_sampling_params)
        if self.model_config.hf_config.model_type == 'kimi_k2':
            self.tool_call_id_type = 'kimi_k2'
        else:
            self.tool_call_id_type = 'random'

        self.use_harmony = model_config.hf_config.model_type == "gpt_oss"
        if self.use_harmony:
            if "stop_token_ids" not in self.default_sampling_params:
                self.default_sampling_params["stop_token_ids"] = []
            self.default_sampling_params["stop_token_ids"].extend(
                get_stop_tokens_for_assistant_actions())

        # NOTE(woosuk): While OpenAI's chat completion API supports browsing
        # for some models, currently vLLM doesn't support it. Please use the
        # Responses API instead.
        self.supports_browsing = False
        self.browser_tool = None
        # NOTE(woosuk): Chat completion API does not support code interpreter.
        # Please use the Responses API instead.
        self.supports_code_interpreter = False
        self.python_tool = None

    async def create_chat_completion(
        self,
        request: ChatCompletionRequest,
        raw_request: Optional[Request] = None,
    ) -> Union[AsyncGenerator[str, None], ChatCompletionResponse,
               ErrorResponse]:
        """
        Chat Completion API similar to OpenAI's API.

        See https://platform.openai.com/docs/api-reference/chat/create
        for the API specification. This API mimics the OpenAI
        Chat Completion API.
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

        try:
            lora_request = self._maybe_get_adapters(
                request, supports_default_mm_loras=True)

            model_name = self.models.model_name(lora_request)

            tokenizer = await self.engine_client.get_tokenizer(lora_request)

            tool_parser = self.tool_parser

            if isinstance(tokenizer, MistralTokenizer):
                # because of issues with pydantic we need to potentially
                # re-serialize the tool_calls field of the request
                # for more info: see comment in `maybe_serialize_tool_calls`
                maybe_serialize_tool_calls(request)
                truncate_tool_call_ids(request)
                validate_request_params(request)

            if (request.tool_choice == "auto" and
                    not (self.enable_auto_tools and tool_parser is not None)
                    and not isinstance(tokenizer, MistralTokenizer)
                    and not self.use_harmony):
                # for hf tokenizers, "auto" tools requires
                # --enable-auto-tool-choice and --tool-call-parser
                return self.create_error_response(
                    "\"auto\" tool choice requires "
                    "--enable-auto-tool-choice and --tool-call-parser to be set"
                )

            if (request.tools is None
                    or (request.tool_choice == "none"
                        and self.exclude_tools_when_tool_choice_none)):
                tool_dicts = None
            else:
                tool_dicts = [tool.model_dump() for tool in request.tools]

            if not self.use_harmony:
                # Common case.
                (
                    conversation,
                    request_prompts,
                    engine_prompts,
                ) = await self._preprocess_chat(
                    request,
                    tokenizer,
                    request.messages,
                    chat_template=request.chat_template or self.chat_template,
                    chat_template_content_format=self.
                    chat_template_content_format,
                    add_generation_prompt=request.add_generation_prompt,
                    continue_final_message=request.continue_final_message,
                    tool_dicts=tool_dicts,
                    documents=request.documents,
                    chat_template_kwargs=request.chat_template_kwargs,
                    tool_parser=tool_parser,
                    add_special_tokens=request.add_special_tokens,
                )
            else:
                # For GPT-OSS.
                (
                    conversation,
                    request_prompts,
                    engine_prompts,
                ) = self._make_request_with_harmony(request)
        except (ValueError, TypeError, RuntimeError,
                jinja2.TemplateError) as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(f"{e} {e.__cause__}")

        request_id = "chatcmpl-" \
                     f"{self._base_request_id(raw_request, request.request_id)}"

        request_metadata = RequestResponseMetadata(request_id=request_id)
        if raw_request:
            raw_request.state.request_metadata = request_metadata

        # Schedule the request and get the result generator.
        generators: list[AsyncGenerator[RequestOutput, None]] = []
        try:
            for i, engine_prompt in enumerate(engine_prompts):
                sampling_params: Union[SamplingParams, BeamSearchParams]

                if self.default_sampling_params is None:
                    self.default_sampling_params = {}

                max_tokens = get_max_tokens(
                    max_model_len=self.max_model_len,
                    request=request,
                    input_length=len(engine_prompt["prompt_token_ids"]),
                    default_sampling_params=self.default_sampling_params)

                if request.use_beam_search:
                    sampling_params = request.to_beam_search_params(
                        max_tokens, self.default_sampling_params)
                else:
                    sampling_params = request.to_sampling_params(
                        max_tokens, self.model_config.logits_processor_pattern,
                        self.default_sampling_params)

                self._log_inputs(request_id,
                                 request_prompts[i],
                                 params=sampling_params,
                                 lora_request=lora_request)

                trace_headers = (None if raw_request is None else await
                                 self._get_trace_headers(raw_request.headers))

                if isinstance(sampling_params, BeamSearchParams):
                    generator = self.engine_client.beam_search(
                        prompt=engine_prompt,
                        request_id=request_id,
                        params=sampling_params,
                        lora_request=lora_request,
                    )
                else:
                    generator = self.engine_client.generate(
                        engine_prompt,
                        sampling_params,
                        request_id,
                        lora_request=lora_request,
                        trace_headers=trace_headers,
                        priority=request.priority,
                    )

                generators.append(generator)
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

        assert len(generators) == 1
        result_generator, = generators

        # Streaming response
        if request.stream:
            return self.chat_completion_stream_generator(
                request,
                result_generator,
                request_id,
                model_name,
                conversation,
                tokenizer,
                request_metadata,
                enable_force_include_usage=self.enable_force_include_usage)

        try:
            return await self.chat_completion_full_generator(
                request, result_generator, request_id, model_name,
                conversation, tokenizer, request_metadata)
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

    def get_chat_request_role(self, request: ChatCompletionRequest) -> str:
        if request.add_generation_prompt:
            return self.response_role
        return request.messages[-1]["role"]
    
    def _is_no_think_mode(self, request: ChatCompletionRequest) -> bool:
        """Detect if request is in /no_think mode by checking message content."""
        # If tools are present, Nemotron chat template forces detailed thinking off.
        # Treat any request with tools as no-think to avoid engaging the reasoning parser.
        try:
            if getattr(request, 'tools', None):
                return True
        except Exception:
            pass

        if not hasattr(request, 'messages') or not request.messages:
            return False
        
        for message in request.messages:
            content = None
            if isinstance(message, dict):
                content = message.get('content')
            elif hasattr(message, 'content'):
                content = message.content
            
            if content:
                text = str(content)
                if '/no_think' in text or 'detailed thinking off' in text.lower():
                    return True
        
        return False

    @staticmethod
    def _bracket_level(s: str, opening='{', closing='}') -> int:
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
    def _filter_delta_text(delta_text: str,
                           previous_text: str) -> tuple[str, bool]:
        # remove last '},' of the tool definition stemming from the
        # "name"/"parameters" outer object or closing ']' of the tool list
        # count occurrences of opening and closing curly braces and
        # once level 0 is reached stop outputting text
        # if 0 is reached while parsing the delta_text we know the current
        # tool will finish in this current iteration
        bracket_level = OpenAIServingChat._bracket_level(previous_text)
        updated_delta, passed_zero = "", False
        for c in delta_text:
            if c == '{':
                bracket_level += 1
                passed_zero = bracket_level == 0
            elif c == '}':
                bracket_level -= 1
                passed_zero = bracket_level == 0

            if bracket_level != 0:
                updated_delta += c
            else:
                # if a comma is reached at level 0 we can stop
                if c == ',':
                    break
        return updated_delta, passed_zero

    @staticmethod
    def _sanitize_nemotron_wrappers(text: Optional[str]) -> str:
        """
        Remove Nemotron <TOOLCALL> wrappers from a text block.
        Safe no-op if text is None or does not contain wrappers.
        Also removes partial tag fragments like '<TOOLCALL' or '</TOOLCALL'.
        """
        if not text:
            return ""
        # Remove full or partial start/end tags
        try:
            return re.sub(r"</?TOOLCALL>?", "", text)
        except Exception:
            return text.replace("<TOOLCALL>", "").replace("</TOOLCALL>", "")

    @staticmethod
    def _extract_top_level_json_array(text: str) -> Optional[str]:
        """
        Best-effort extraction of the first top-level JSON array substring
        from the given text. Returns None if no array found.
        """
        if not text:
            return None
        start = text.find('[')
        if start == -1:
            return None
        # Scan to find the matching closing bracket at top level
        i = start
        depth = 0
        in_string = False
        escape = False
        n = len(text)
        while i < n:
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                else:
                    if ch == '\\':
                        escape = True
                    elif ch == '"':
                        in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == '[':
                    depth += 1
                elif ch == ']':
                    depth -= 1
                    if depth == 0:
                        return text[start:i + 1]
            i += 1
        # If not closed, return the partial tail
        return text[start:]

    @staticmethod
    def _load_tool_array_forgiving(text: str) -> Optional[list[dict]]:
        """
        Try to parse a JSON array of tool calls from text. Handles Nemotron
        wrappers and trailing/leading noise. Returns None on failure.
        """
        sanitized = OpenAIServingChat._sanitize_nemotron_wrappers(text).strip()
        candidate = OpenAIServingChat._extract_top_level_json_array(sanitized)
        if candidate is None:
            candidate = sanitized
        try:
            arr = json.loads(candidate)
            if isinstance(arr, list):
                return arr
        except Exception:
            return None
        return None

    @staticmethod
    def _suffix_after_lcp(current_text: str, streamed_prefix: str) -> str:
        if not streamed_prefix:
            return current_text
        i = 0
        max_i = min(len(current_text), len(streamed_prefix))
        while i < max_i and current_text[i] == streamed_prefix[i]:
            i += 1
        return current_text[i:]

    def extract_tool_call_required_streaming(
        self,
        previous_text: str,
        current_text: Optional[str],
        delta_text: str,
        function_name_returned: bool,
        tool_call_idx: Optional[int] = None
    ) -> tuple[Optional[DeltaMessage], bool]:
        if current_text is None or current_text == "":
            # if the current text is empty, we cannot parse it
            return None, function_name_returned
        try:
            # Sanitize Nemotron wrappers and focus on array content
            sanitized_current = self._sanitize_nemotron_wrappers(current_text)
            array_slice = self._extract_top_level_json_array(sanitized_current)
            if array_slice:
                # Use partial JSON parser in permissive mode to support streaming
                try:
                    obj = partial_json_parser.loads(array_slice, Allow.ALL)
                except Exception:
                    obj = None
            else:
                obj = None
        except Exception:
            logger.debug('not enough tokens or invalid partial JSON for required tool parsing')
            obj = None

        # check if the current text is a valid array
        # containing a partial tool calling object
        # if not repeat
        if obj is None or not isinstance(obj, list):
            function_name_returned = False
            delta_message = None
        elif not len(obj) > 0:
            # It's an empty list, probably just saw a '['
            return None, function_name_returned
        else:
            # Use sanitized delta/previous for bracket-level filtering
            sanitized_delta = self._sanitize_nemotron_wrappers(delta_text)
            sanitized_prev = self._sanitize_nemotron_wrappers(previous_text)
            _, finishes_previous_tool = OpenAIServingChat._filter_delta_text(
                sanitized_delta, sanitized_prev)
            # take the last tool call from the generated list
            current_tool_call = obj[-1]

            # once parameters have been generated the name is complete as well
            has_params_key = ("parameters" in current_tool_call
                              or "arguments" in current_tool_call)
            if not finishes_previous_tool and ("name" not in current_tool_call
                                               or not has_params_key):
                function_name_returned = False
                delta_message = None
            else:
                if not function_name_returned:
                    # get partly generated arguments from the latest tool call
                    # Prefer "parameters", fallback to "arguments"
                    search_space = self._sanitize_nemotron_wrappers(
                        current_text)
                    param_match = re.search(r'.*"parameters":\s*(.*)',
                                            search_space)
                    if not param_match:
                        param_match = re.search(r'.*"arguments":\s*(.*)',
                                                search_space)
                    arguments = param_match.group(1) if param_match else ""
                    arguments, _ = OpenAIServingChat._filter_delta_text(
                        arguments, sanitized_prev)

                    # if this iteration finishes a previous tool call but a
                    # new incomplete tool is already generated, take the
                    # previous from the list
                    if (finishes_previous_tool and not has_params_key):
                        current_tool_call = obj[-2]

                    function_name_returned = True
                    tool_call_id = make_tool_call_id(
                        id_type=self.tool_call_id_type,
                        func_name=current_tool_call["name"],
                        idx=tool_call_idx)
                    delta_message = DeltaMessage(tool_calls=[
                        DeltaToolCall(id=tool_call_id,
                                      function=DeltaFunctionCall(
                                          name=current_tool_call["name"],
                                          arguments=arguments),
                                      index=len(obj) - 1,
                                      type="function")
                    ])

                else:
                    delta_text, _ = OpenAIServingChat._filter_delta_text(
                        sanitized_delta, sanitized_prev)

                    if delta_text != "":
                        delta_message = DeltaMessage(tool_calls=[
                            DeltaToolCall(
                                function=DeltaFunctionCall(
                                    # OpenAI API returns None
                                    # instead of name every time
                                    name=None,
                                    arguments=delta_text),
                                index=len(obj) - 1)
                        ])
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
        tokenizer: AnyTokenizer,
        request_metadata: RequestResponseMetadata,
        enable_force_include_usage: bool,
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
                get_streamable_parser_for_assistant()
                for _ in range(num_choices)
            ]
            harmony_tools_streamed = [False] * num_choices
        tools_streamed = [False] * num_choices

        if isinstance(request.tool_choice, ChatCompletionNamedToolChoiceParam):
            tool_choice_function_name = request.tool_choice.function.name
        else:
            tool_choice_function_name = None

        # Incremental tag stripping state for named tool choice streaming
        stream_inside_tag = [False] * num_choices if tool_choice_function_name else None
        # Aggregate sanitized text and track streamed arguments for targeted
        if tool_choice_function_name:
            targeted_sanitized_total = [""] * num_choices
            targeted_streamed_args = [""] * num_choices

        # Determine whether tools are in use with "auto" tool choice
        tool_choice_auto = (
            not tool_choice_function_name
            and self._should_stream_with_auto_tool_parsing(request))

        all_previous_token_ids: Optional[list[list[int]]]
        function_name_returned = [False] * num_choices
        if self.tool_call_id_type == 'kimi_k2':
            history_tool_call_cnt = get_history_tool_calls_cnt(conversation)
        else:
            history_tool_call_cnt = 0

        # Always track previous_texts for comprehensive output logging
        previous_texts = [""] * num_choices

        # In /no_think mode, still need arrays if tools are enabled
        if tool_choice_auto or (self.reasoning_parser and not self._is_no_think_mode(request)):
            # These are only required in "auto" tool choice case
            all_previous_token_ids = [[]] * num_choices
            # For reasoning parser and tool call all enabled
            added_content_delta_arr = [False] * num_choices
            reasoning_end_arr = [False] * num_choices
        elif request.tool_choice == "required":
            all_previous_token_ids = None
        else:
            all_previous_token_ids = None

        # State for forced tool choice buffering
        is_forced_tool_choice = request.tool_choice == "required" or isinstance(
            request.tool_choice, ChatCompletionNamedToolChoiceParam)
        if is_forced_tool_choice:
            forced_tool_choice_buffer = [""] * num_choices

        try:
            if self.reasoning_parser:
                reasoning_parser = self.reasoning_parser(tokenizer)
        except RuntimeError as e:
            logger.exception("Error in reasoning parser creation.")
            data = self.create_streaming_error_response(str(e))
            yield f"data: {data}\n\n"
            yield "data: [DONE]\n\n"
            return
        # Prepare the tool parser if it's needed
        try:
            if tool_choice_auto and self.tool_parser:
                tool_parsers: list[Optional[ToolParser]] = [
                    self.tool_parser(tokenizer)
                ] * num_choices
            else:
                tool_parsers = [None] * num_choices
        except Exception as e:
            logger.exception("Error in tool parser creation.")
            data = self.create_streaming_error_response(str(e))
            yield f"data: {data}\n\n"
            yield "data: [DONE]\n\n"
            return

        stream_options = request.stream_options
        if stream_options:
            include_usage = stream_options.include_usage \
                            or enable_force_include_usage
            include_continuous_usage = include_usage and \
                                       stream_options.continuous_usage_stats
        else:
            include_usage, include_continuous_usage = False, False

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
                            finish_reason=None)

                        # return prompt_token_ids at the first chunk ever
                        chunk = ChatCompletionStreamResponse(
                            id=request_id,
                            object=chunk_object_type,
                            created=created_time,
                            choices=[choice_data],
                            model=model_name,
                            prompt_token_ids=(res.prompt_token_ids
                                              if request.return_token_ids else
                                              None))

                        # if continuous usage stats are requested, add it
                        if include_continuous_usage:
                            chunk.usage = UsageInfo(
                                prompt_tokens=num_prompt_tokens,
                                completion_tokens=0,
                                total_tokens=num_prompt_tokens)

                        data = chunk.model_dump_json(exclude_unset=True)
                        yield f"data: {data}\n\n"

                    # Send response to echo the input portion of the
                    # last message
                    if request.echo:
                        last_msg_content: Union[str, list[dict[str, str]]] = ""
                        if conversation and "content" in conversation[
                                -1] and conversation[-1].get("role") == role:
                            last_msg_content = conversation[-1]["content"] or ""

                        if last_msg_content:
                            for i in range(num_choices):
                                choice_data = (
                                    ChatCompletionResponseStreamChoice(
                                        index=i,
                                        delta=DeltaMessage(
                                            content=last_msg_content),
                                        logprobs=None,
                                        finish_reason=None))
                                chunk = ChatCompletionStreamResponse(
                                    id=request_id,
                                    object=chunk_object_type,
                                    created=created_time,
                                    choices=[choice_data],
                                    model=model_name)
                                if include_continuous_usage:
                                    chunk.usage = UsageInfo(
                                        prompt_tokens=num_prompt_tokens,
                                        completion_tokens=0,
                                        total_tokens=num_prompt_tokens)

                                data = chunk.model_dump_json(
                                    exclude_unset=True)
                                yield f"data: {data}\n\n"
                    first_iteration = False

                for output in res.outputs:
                    i = output.index
                    tool_parser = tool_parsers[i]

                    if finish_reason_sent[i]:
                        continue

                    if request.logprobs and request.top_logprobs is not None:
                        assert output.logprobs is not None, (
                            "Did not output logprobs")
                        logprobs = self._create_chat_logprobs(
                            token_ids=output.token_ids,
                            top_logprobs=output.logprobs,
                            tokenizer=tokenizer,
                            num_output_top_logprobs=request.top_logprobs,
                            return_as_token_id=request.
                            return_tokens_as_token_ids,
                        )
                    else:
                        logprobs = None

                    if self.use_harmony:
                        harmony_parser = harmony_parsers[i]
                        prev_recipient = harmony_parser.current_recipient
                        for token_id in output.token_ids:
                            harmony_parser.process(token_id)
                        cur_channel = harmony_parser.current_channel
                        cur_recipient = harmony_parser.current_recipient
                        delta_text = harmony_parser.last_content_delta or ""
                    else:
                        delta_text = output.text

                    if not delta_text and not output.token_ids and \
                        not previous_num_tokens[i]:
                        # Chunked prefill case, don't return empty chunks
                        continue

                    delta_message: Optional[DeltaMessage]

                    # just update previous_texts and previous_token_ids
                    # In /no_think mode, still need text accumulation if tools are enabled
                    if tool_choice_auto or (self.reasoning_parser and not self._is_no_think_mode(request)):
                        assert previous_texts is not None
                        assert all_previous_token_ids is not None
                        previous_text = previous_texts[i]
                        previous_token_ids = all_previous_token_ids[i]
                        current_text = previous_text + delta_text
                        # avoid the None + list error.
                        if previous_token_ids:
                            current_token_ids = previous_token_ids + as_list(
                                output.token_ids)
                        else:
                            current_token_ids = as_list(output.token_ids)

                    if self.use_harmony:
                        if cur_channel == "final":
                            delta_message = DeltaMessage(content=delta_text)
                        elif cur_channel == "analysis":
                            if request.include_reasoning:
                                delta_message = DeltaMessage(
                                    reasoning_content=delta_text)
                            else:
                                delta_message = None
                        elif (cur_channel == "commentary" and cur_recipient
                              and cur_recipient.startswith("functions.")):
                            # Count completed tool calls to determine index
                            base_index = 0
                            for msg in harmony_parser.messages:
                                if (msg.channel == "commentary"
                                        and msg.recipient
                                        and msg.recipient.startswith(
                                            "functions.")):
                                    base_index += 1

                            if prev_recipient != cur_recipient:
                                tool_name = cur_recipient.split(
                                    "functions.", 1)[1]
                                delta_message = DeltaMessage(tool_calls=[
                                    DeltaToolCall(
                                        id=make_tool_call_id(),
                                        type="function",
                                        function=DeltaFunctionCall(
                                            name=tool_name,
                                            arguments="",
                                        ),
                                        index=base_index,
                                    )
                                ])
                            elif delta_text:
                                delta_message = DeltaMessage(tool_calls=[
                                    DeltaToolCall(
                                        index=base_index,
                                        function=DeltaFunctionCall(
                                            arguments=delta_text),
                                    )
                                ])
                            else:
                                delta_message = None

                            if delta_message is not None:
                                harmony_tools_streamed[i] = True
                        else:
                            delta_message = None
                    # handle streaming deltas for tools with named tool_choice
                    elif tool_choice_function_name:
                        delta_tool_call = None
                        if (self.reasoning_parser
                                and not self._is_no_think_mode(request)
                                and not reasoning_end_arr[i]
                                and not reasoning_parser.is_reasoning_end(
                                    previous_token_ids)):
                            assert reasoning_parser is not None
                            delta_message = (
                                reasoning_parser.
                                extract_reasoning_content_streaming(
                                    previous_text,
                                    current_text,
                                    delta_text,
                                    previous_token_ids,
                                    current_token_ids,
                                    output.token_ids,
                                ))
                            # When encountering think end id in delta_token_ids
                            # or think end id in prompt_token_ids
                            # i.e {"enable_thinking": False},
                            # set reasoning status to end.
                            # Only keep 'content', remove 'reasoning_content'.
                            if reasoning_parser.is_reasoning_end(
                                    as_list(output.token_ids)) or (
                                        res.prompt_token_ids
                                        and reasoning_parser.is_reasoning_end(
                                            res.prompt_token_ids)):
                                reasoning_end_arr[i] = True
                                if delta_message and delta_message.content:
                                    # This need to be added to next `delta_text`
                                    current_text = delta_message.content
                                    delta_message.content = None
                                else:
                                    current_text = ""
                        else:
                            # Just to add remaining `content`
                            if self.reasoning_parser and not self._is_no_think_mode(request):
                                delta_text = previous_text + delta_text
                                current_text = ""

                            if function_name_returned[i]:
                                # Strip any tag-like content incrementally
                                inside = stream_inside_tag[i] if stream_inside_tag is not None else False
                                out_chars = []
                                for ch in delta_text:
                                    if inside:
                                        if ch == '>':
                                            inside = False
                                        continue
                                    if ch == '<':
                                        inside = True
                                        continue
                                    out_chars.append(ch)
                                if stream_inside_tag is not None:
                                    stream_inside_tag[i] = inside
                                sanitized_text = ''.join(out_chars)
                                # Append and compute only the incremental arguments suffix
                                args_delta = ""
                                targeted_sanitized_total[i] += sanitized_text
                                acc = targeted_sanitized_total[i]
                                arg_idx = acc.find('"arguments"')
                                if arg_idx != -1:
                                    brace_start = acc.find('{', arg_idx)
                                    if brace_start != -1:
                                        depth = 0
                                        in_str = False
                                        esc = False
                                        end_pos = None
                                        for j in range(brace_start, len(acc)):
                                            c = acc[j]
                                            if in_str:
                                                if esc:
                                                    esc = False
                                                else:
                                                    if c == '\\':
                                                        esc = True
                                                    elif c == '"':
                                                        in_str = False
                                            else:
                                                if c == '"':
                                                    in_str = True
                                                elif c == '{':
                                                    depth += 1
                                                elif c == '}':
                                                    depth -= 1
                                                    if depth == 0:
                                                        end_pos = j + 1
                                                        break
                                        if end_pos is not None:
                                            args_full = acc[brace_start:end_pos]
                                            prev = targeted_streamed_args[i]
                                            if prev and args_full.startswith(prev):
                                                args_delta = args_full[len(prev):]
                                                targeted_streamed_args[i] = args_full
                                            else:
                                                add = OpenAIServingChat._suffix_after_lcp(args_full, prev)
                                                if not add and (len(args_full) < len(prev) or not prev):
                                                    add = args_full
                                                args_delta = add
                                                targeted_streamed_args[i] = args_full if not prev else (prev + add if args_full.startswith(prev) else args_full)
                                if args_delta:
                                    delta_tool_call = DeltaToolCall(
                                        function=DeltaFunctionCall(arguments=args_delta),
                                        index=i)
                            else:
                                inside = stream_inside_tag[i] if stream_inside_tag is not None else False
                                out_chars = []
                                for ch in delta_text:
                                    if inside:
                                        if ch == '>':
                                            inside = False
                                        continue
                                    if ch == '<':
                                        inside = True
                                        continue
                                    out_chars.append(ch)
                                if stream_inside_tag is not None:
                                    stream_inside_tag[i] = inside
                                sanitized_text = ''.join(out_chars)
                                targeted_sanitized_total[i] += sanitized_text
                                acc = targeted_sanitized_total[i]
                                args_delta = ""
                                arg_idx = acc.find('"arguments"')
                                if arg_idx != -1:
                                    brace_start = acc.find('{', arg_idx)
                                    if brace_start != -1:
                                        depth = 0
                                        in_str = False
                                        esc = False
                                        end_pos = None
                                        for j in range(brace_start, len(acc)):
                                            c = acc[j]
                                            if in_str:
                                                if esc:
                                                    esc = False
                                                else:
                                                    if c == '\\':
                                                        esc = True
                                                    elif c == '"':
                                                        in_str = False
                                            else:
                                                if c == '"':
                                                    in_str = True
                                                elif c == '{':
                                                    depth += 1
                                                elif c == '}':
                                                    depth -= 1
                                                    if depth == 0:
                                                        end_pos = j + 1
                                                        break
                                        if end_pos is not None:
                                            args_full = acc[brace_start:end_pos]
                                            prev = targeted_streamed_args[i]
                                            if prev and args_full.startswith(prev):
                                                args_delta = args_full[len(prev):]
                                                targeted_streamed_args[i] = args_full
                                            else:
                                                add = OpenAIServingChat._suffix_after_lcp(args_full, prev)
                                                if not add and (len(args_full) < len(prev) or not prev):
                                                    add = args_full
                                                args_delta = add
                                                targeted_streamed_args[i] = args_full if not prev else (prev + add if args_full.startswith(prev) else args_full)
                                # Emit first delta with name and any initial arguments delta (may be empty)
                                delta_tool_call = DeltaToolCall(
                                    id=make_tool_call_id(),
                                    type="function",
                                    function=DeltaFunctionCall(
                                        name=tool_choice_function_name,
                                        arguments=args_delta),
                                    index=i)
                                function_name_returned[i] = True

                            if delta_tool_call is not None and (delta_tool_call.function is not None and (delta_tool_call.function.name is not None or delta_tool_call.function.arguments)):
                                delta_message = DeltaMessage(tool_calls=[delta_tool_call])
                                tools_streamed[i] = True
                            else:
                                delta_message = None

                    elif request.tool_choice == "required":
                        assert previous_texts is not None
                        previous_text = previous_texts[i]
                        current_text = previous_text + delta_text
                        fn_name_returned = function_name_returned[i]

                        if self.reasoning_parser and not self._is_no_think_mode(request):
                            _, content = \
                                reasoning_parser.extract_reasoning_content(
                                    current_text,
                                    request
                                )
                        else:
                            content = current_text
                        delta_message, function_name_returned[i] = (
                            self.extract_tool_call_required_streaming(
                                previous_text=previous_text,
                                current_text=content,
                                delta_text=delta_text,
                                function_name_returned=fn_name_returned,
                                tool_call_idx=history_tool_call_cnt))
                        if (delta_message and delta_message.tool_calls and
                                delta_message.tool_calls[0].id is not None):
                            history_tool_call_cnt += 1
                            tools_streamed[i] = True

                        # update the previous values for the next iteration
                        previous_texts[i] = current_text

                    # handle streaming deltas for tools with "auto" tool choice
                    # and reasoning parser - but skip reasoning parser in /no_think mode
                    elif tool_choice_auto and self.reasoning_parser and not self._is_no_think_mode(request):
                        assert tool_parser is not None
                        assert reasoning_parser is not None
                        assert added_content_delta_arr is not None
                        assert reasoning_end_arr is not None
                        output_token_ids = as_list(output.token_ids)
                        if not reasoning_end_arr[i]:
                            delta_message = (
                                reasoning_parser.
                                extract_reasoning_content_streaming(
                                    previous_text,
                                    current_text,
                                    delta_text,
                                    previous_token_ids,
                                    current_token_ids,
                                    output_token_ids,
                                ))
                            # When encountering think end id in prompt_token_ids
                            # i.e {"enable_thinking": False},
                            # set reasoning status to end.
                            # Remove the text and token ids related
                            # to 'reasoning_content'.
                            if res.prompt_token_ids and \
                                reasoning_parser.is_reasoning_end(
                                    res.prompt_token_ids):
                                reasoning_end_arr[i] = True
                                current_token_ids = output_token_ids
                                if delta_message and delta_message.content:
                                    current_text = delta_message.content
                                    delta_message.content = None
                                else:
                                    current_text = ""
                            # When encountering think end id in delta_token_ids,
                            # set reasoning status to end.
                            # Remove the text and token ids related
                            # to 'reasoning_content'.
                            if reasoning_parser.is_reasoning_end(
                                    output_token_ids):
                                reasoning_end_arr[i] = True
                                current_token_ids =  \
                                    reasoning_parser.extract_content_ids(
                                        output_token_ids)
                                if delta_message and delta_message.content:
                                    current_text = delta_message.content
                                    delta_message.content = None
                                else:
                                    current_text = ""

                        # handle tool calls only after reasoning is done,
                        else:
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

                            delta_message = (
                                tool_parser.extract_tool_calls_streaming(
                                    previous_text=previous_text,
                                    current_text=current_text,
                                    delta_text=delta_text,
                                    previous_token_ids=previous_token_ids,
                                    current_token_ids=current_token_ids,
                                    delta_token_ids=delta_token_ids,
                                    request=request))
                            if delta_message and delta_message.tool_calls:
                                tools_streamed[i] = True
                    # when only tool calls
                    elif tool_choice_auto:
                        assert tool_parser is not None
                        delta_message = (
                            tool_parser.extract_tool_calls_streaming(
                                previous_text=previous_text,
                                current_text=current_text,
                                delta_text=delta_text,
                                previous_token_ids=previous_token_ids,
                                current_token_ids=current_token_ids,
                                delta_token_ids=output.token_ids,
                                request=request))
                        if delta_message and delta_message.tool_calls:
                            tools_streamed[i] = True

                    # when only reasoning (skip reasoning parser in /no_think mode)
                    elif self.reasoning_parser and not self._is_no_think_mode(request):
                        delta_message = (reasoning_parser.
                                         extract_reasoning_content_streaming(
                                             previous_text,
                                             current_text,
                                             delta_text,
                                             previous_token_ids,
                                             current_token_ids,
                                             output.token_ids,
                                         ))
                    # handle streaming just a content delta
                    else:
                        delta_message = DeltaMessage(content=delta_text)

                    # update the previous values for the next iteration
                    # In /no_think mode, still need state updates if tools are enabled
                    if tool_choice_auto or (self.reasoning_parser and not self._is_no_think_mode(request)):
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
                        if output.finish_reason is None:
                            continue
                        else:
                            delta_message = DeltaMessage()

                    # Log streaming delta if output logging is enabled
                    if self.enable_log_outputs and self.request_logger:
                        delta_content = ""
                        if delta_message.content:
                            delta_content = delta_message.content
                        elif delta_message.tool_calls:
                            delta_content = "".join(
                                tc.function.arguments
                                for tc in delta_message.tool_calls
                                if tc.function and tc.function.arguments)

                        if delta_content:
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
                            token_ids=(as_list(output.token_ids)
                                       if request.return_token_ids else None))

                    # if the model is finished generating
                    else:
                        # check to make sure we haven't "forgotten" to stream
                        #   any tokens that were generated but previously
                        #   matched by partial json parsing
                        # only happens if we are NOT using guided decoding
                        auto_tools_called = False
                        if tool_parser:
                            auto_tools_called = len(
                                tool_parser.prev_tool_call_arr) > 0
                            index = len(tool_parser.prev_tool_call_arr
                                        ) - 1 if auto_tools_called else 0
                        else:
                            index = 0

                        # Skip the unstreamed token check if the tool parser has already handled end-of-call flush
                        # This prevents duplicate argument streaming when both tool parser and serving layer try to flush
                        should_check_unstreamed = self._should_check_for_unstreamed_tool_arg_tokens(delta_message, output)
                        
                        # Some tool parsers implement their own comprehensive end-of-call flush logic.
                        # Skip the serving layer's unstreamed token check entirely for those parsers
                        if (should_check_unstreamed and tool_parser and 
                            hasattr(tool_parser, '__class__') and (
                                'MistralToolParser' in tool_parser.__class__.__name__ or
                                'NemotronToolParser' in tool_parser.__class__.__name__
                            )):
                            # Parser handles all end-of-call flushing internally, don't duplicate
                            should_check_unstreamed = False

                        if should_check_unstreamed and tool_parser:
                            latest_delta_len = 0
                            if ((isinstance(
                                    delta_message.tool_calls[0].function,
                                    DeltaFunctionCall)) and isinstance(
                                        delta_message.tool_calls[0].function.
                                        arguments, str)):
                                latest_delta_len = len(
                                    delta_message.tool_calls[0].function.
                                    arguments)

                            # get the expected call based on partial JSON
                            # parsing which "autocompletes" the JSON
                            expected_call = json.dumps(
                                tool_parser.prev_tool_call_arr[index].get(
                                    "arguments", {}),
                                ensure_ascii=False)

                            # get what we've streamed so far for arguments
                            # for the current tool
                            actual_call = tool_parser.streamed_args_for_tool[
                                index]
                            if (latest_delta_len > 0):
                                actual_call = actual_call[:-latest_delta_len]

                            # check to see if there's anything left to stream
                            # Use startswith check for more robust remaining content computation
                            if expected_call.startswith(actual_call):
                                remaining_call = expected_call[len(actual_call):]
                            else:
                                # Fallback: use extract_intermediate_diff for better handling
                                from vllm.entrypoints.openai.tool_parsers.utils import extract_intermediate_diff
                                remaining_call = extract_intermediate_diff(expected_call, actual_call)
                            
                            # Only send remaining call if it's non-empty and meaningful
                            if remaining_call and remaining_call.strip():
                                # set that as a delta message
                                delta_message = DeltaMessage(tool_calls=[
                                    DeltaToolCall(index=index,
                                                  function=DeltaFunctionCall(
                                                      arguments=remaining_call).
                                                  model_dump(exclude_none=True))
                                ])
                            else:
                                # No meaningful remaining content, don't send anything
                                delta_message = None

                        # Send the finish response for each request.n only once
                        if auto_tools_called or tools_streamed[i] or (
                                self.use_harmony
                                and harmony_tools_streamed[i]):
                            finish_reason_ = "tool_calls"
                        else:
                            finish_reason_ = output.finish_reason \
                                if output.finish_reason else "stop"
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=delta_message,
                            logprobs=logprobs,
                            finish_reason=finish_reason_,
                            stop_reason=output.stop_reason,
                            token_ids=(as_list(output.token_ids)
                                       if request.return_token_ids else None))

                        finish_reason_sent[i] = True

                    chunk = ChatCompletionStreamResponse(
                        id=request_id,
                        object=chunk_object_type,
                        created=created_time,
                        choices=[choice_data],
                        model=model_name)

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
                final_usage = UsageInfo(prompt_tokens=num_prompt_tokens,
                                        completion_tokens=completion_tokens,
                                        total_tokens=num_prompt_tokens +
                                        completion_tokens)
                if self.enable_prompt_tokens_details and num_cached_tokens:
                    final_usage.prompt_tokens_details = PromptTokenUsageInfo(
                        cached_tokens=num_cached_tokens)

                final_usage_chunk = ChatCompletionStreamResponse(
                    id=request_id,
                    object=chunk_object_type,
                    created=created_time,
                    choices=[],
                    model=model_name,
                    usage=final_usage)
                final_usage_data = (final_usage_chunk.model_dump_json(
                    exclude_unset=True, exclude_none=True))
                yield f"data: {final_usage_data}\n\n"

            # report to FastAPI middleware aggregate usage across all choices
            num_completion_tokens = sum(previous_num_tokens)
            request_metadata.final_usage_info = UsageInfo(
                prompt_tokens=num_prompt_tokens,
                completion_tokens=num_completion_tokens,
                total_tokens=num_prompt_tokens + num_completion_tokens,
            )

            # Log complete streaming response if output logging is enabled
            if self.enable_log_outputs and self.request_logger:
                # Log the complete response for each choice
                for i in range(num_choices):
                    full_text = (
                        previous_texts[i]
                        if previous_texts and i < len(previous_texts) else
                        f"<streaming_complete: {previous_num_tokens[i]} tokens>"
                    )
                    self.request_logger.log_outputs(
                        request_id=request_id,
                        outputs=full_text,
                        output_token_ids=
                        None,  # Consider also logging all token IDs
                        finish_reason="streaming_complete",
                        is_streaming=True,
                        delta=False,
                    )

        except Exception as e:
            # TODO: Use a vllm-specific Validation Error
            logger.exception("Error in chat completion stream generator.")
            data = self.create_streaming_error_response(str(e))
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
        tokenizer: AnyTokenizer,
        request_metadata: RequestResponseMetadata,
    ) -> Union[ErrorResponse, ChatCompletionResponse]:

        created_time = int(time.time())
        final_res: Optional[RequestOutput] = None

        try:
            async for res in result_generator:
                final_res = res
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

        assert final_res is not None

        choices: list[ChatCompletionResponseChoice] = []
        role = self.get_chat_request_role(request)
        for output in final_res.outputs:
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

            message = self._parse_chat_message_from_text(role,
                                                         output.text,
                                                         request,
                                                         tokenizer,
                                                         conversation)

            auto_tools_called = False
            if message.tool_calls:
                # Heuristic to guess if 'auto' was used and resulted in tools
                if request.tool_choice is None or request.tool_choice == "auto":
                    auto_tools_called = True

            choice_data = ChatCompletionResponseChoice(
                index=output.index,
                message=message,
                logprobs=logprobs,
                finish_reason="tool_calls" if auto_tools_called else
                output.finish_reason if output.finish_reason else "stop",
                stop_reason=output.stop_reason,
                token_ids=(as_list(output.token_ids)
                           if request.return_token_ids else None),
            )

            choices.append(choice_data)

        if request.echo:
            last_msg_content: Union[str, list[dict[str, str]]] = ""
            if conversation and "content" in conversation[
                    -1] and conversation[-1].get("role") == role:
                last_msg_content = conversation[-1]["content"] or ""
            if isinstance(last_msg_content, list):
                # TODO: support multimodal echo
                last_msg_content = ""

            if last_msg_content:
                for choice in choices:
                    # Prepend echo to the beginning of the content
                    prepended_content = last_msg_content
                    if choice.message.content:
                        prepended_content += choice.message.content
                    choice.message.content = prepended_content

        return self.create_chat_completion_response(
            request=request,
            choices=choices,
            request_id=request_id,
            request_metadata=request_metadata,
            model_name=model_name,
            created_time=created_time,
            final_res=final_res,
        )

    def _get_top_logprobs(
            self, logprobs: dict[int, Logprob], top_logprobs: Optional[int],
            tokenizer: AnyTokenizer,
            should_return_as_token_id: bool) -> list[ChatCompletionLogProb]:
        return [
            ChatCompletionLogProb(
                token=(token := self._get_decoded_token(
                    p[1],
                    p[0],
                    tokenizer,
                    return_as_token_id=should_return_as_token_id,
                )),
                logprob=max(p[1].logprob, -9999.0),
                bytes=list(token.encode("utf-8", errors="replace")),
            ) for i, p in enumerate(logprobs.items())
            if top_logprobs and i < top_logprobs
        ]

    def _create_chat_logprobs(
        self,
        token_ids: GenericSequence[int],
        top_logprobs: GenericSequence[Optional[dict[int, Logprob]]],
        tokenizer: AnyTokenizer,
        num_output_top_logprobs: Optional[int] = None,
        return_as_token_id: Optional[bool] = None,
    ) -> ChatCompletionLogProbs:
        """Create OpenAI-style logprobs."""
        logprobs_content: list[ChatCompletionLogProbsContent] = []

        should_return_as_token_id = return_as_token_id if \
            return_as_token_id is not None else self.return_tokens_as_token_ids
        for i, token_id in enumerate(token_ids):
            step_top_logprobs = top_logprobs[i]
            if step_top_logprobs is None or step_top_logprobs.get(
                    token_id) is None:
                if should_return_as_token_id:
                    token = f"token_id:{token_id}"
                else:
                    token = tokenizer.decode(token_id)

                logprobs_content.append(
                    ChatCompletionLogProbsContent(
                        token=token,
                        bytes=list(token.encode("utf-8", errors="replace")),
                    ))
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
                        bytes=None if step_decoded is None else list(
                            step_decoded.encode("utf-8", errors="replace")),
                        top_logprobs=self._get_top_logprobs(
                            step_top_logprobs, num_output_top_logprobs,
                            tokenizer, should_return_as_token_id),
                    ))

        return ChatCompletionLogProbs(content=logprobs_content)

    def _parse_chat_message_from_text(
        self,
        role: str,
        content: Optional[str],
        request: ChatCompletionRequest,
        tokenizer: AnyTokenizer,
        conversation: list[ConversationMessage],
    ) -> ChatMessage:
        if self.tool_call_id_type == 'kimi_k2':
            history_tool_call_cnt = get_history_tool_calls_cnt(conversation)
        else:
            history_tool_call_cnt = 0

        if self.reasoning_parser and not self._is_no_think_mode(request):
            try:
                reasoning_parser = self.reasoning_parser(tokenizer)
            except RuntimeError as e:
                logger.exception("Error in reasoning parser creation.")
                raise e
            # If the reasoning parser is enabled,
            # tool calls are extracted exclusively from the content.
            reasoning_content, content = (
                reasoning_parser.extract_reasoning_content(
                    content if content is not None else "", request=request))
            if not request.include_reasoning:
                reasoning_content = None
        else:
            reasoning_content = None

        auto_tools_called = False
        # if auto tools are not enabled, and a named tool choice using
        #   outlines is not being used
        if (not self.enable_auto_tools or not self.tool_parser) and \
            (not isinstance(request.tool_choice,
                            ChatCompletionNamedToolChoiceParam
                            ) and request.tool_choice != "required"):
            message = ChatMessage(role=role,
                                  reasoning_content=reasoning_content,
                                  content=content)

        # if the request uses tools and specified a tool choice
        elif request.tool_choice and type(
                request.tool_choice) is ChatCompletionNamedToolChoiceParam:

            tool_call_class = MistralToolCall if isinstance(
                tokenizer, MistralTokenizer) else ToolCall
            # Attempt to extract only the arguments for the named tool
            selected_name = request.tool_choice.function.name
            arguments_str = ""
            if content:
                # Try to parse Nemotron-style wrapper first
                arr = self._load_tool_array_forgiving(content)
                if arr and isinstance(arr, list):
                    chosen = None
                    for item in arr:
                        try:
                            if isinstance(item, dict) and item.get("name") == selected_name:
                                chosen = item
                                break
                        except Exception:
                            continue
                    if not chosen and arr and isinstance(arr[0], dict):
                        chosen = arr[0]
                    if isinstance(chosen, dict):
                        params = chosen.get("parameters")
                        if params is None:
                            params = chosen.get("arguments")
                        if isinstance(params, str):
                            try:
                                params = json.loads(params)
                            except Exception:
                                # keep as string if not valid JSON
                                pass
                        try:
                            arguments_str = json.dumps(params, ensure_ascii=False)
                        except Exception:
                            arguments_str = str(params)
                if not arguments_str:
                    # Fallback: if content itself is a JSON object, use it; else sanitize wrappers
                    sanitized = self._sanitize_nemotron_wrappers(content).strip()
                    try:
                        obj = json.loads(sanitized)
                        # If it's an object, treat as parameters; if array, pick first matching
                        if isinstance(obj, dict):
                            arguments_str = json.dumps(obj, ensure_ascii=False)
                        elif isinstance(obj, list) and obj:
                            first = obj[0]
                            if isinstance(first, dict):
                                params = first.get("parameters") or first.get("arguments") or first
                                arguments_str = json.dumps(params, ensure_ascii=False)
                    except Exception:
                        # last resort: use sanitized text
                        arguments_str = sanitized
            message = ChatMessage(
                role=role,
                reasoning_content=reasoning_content,
                content="",
                tool_calls=[
                    tool_call_class(function=FunctionCall(
                        name=selected_name,
                        arguments=arguments_str,
                    ))
                ],
            )

        elif request.tool_choice and request.tool_choice == "required":
            tool_call_class = MistralToolCall if isinstance(
                tokenizer, MistralTokenizer) else ToolCall

            # Try strict OpenAI-style array of {name, parameters}
            tool_calls: list[FunctionDefinition]
            parsed_success = False
            if content is None:
                parsed_success = False
                tool_calls = []  # type: ignore
            else:
                sanitized = self._sanitize_nemotron_wrappers(content)
                try:
                    tool_calls = TypeAdapter(
                        list[FunctionDefinition]).validate_json(sanitized)
                    parsed_success = True
                except Exception:
                    parsed_success = False
                    tool_calls = []  # type: ignore

            tool_call_messages: list[ToolCall] = []
            if parsed_success and tool_calls:
                raw_arr = None
                try:
                    raw_arr = json.loads(sanitized) if sanitized else None
                except Exception:
                    raw_arr = None
                for func_def in tool_calls:
                    tc_id = make_tool_call_id(
                        id_type=self.tool_call_id_type,
                        func_name=func_def.name,
                        idx=history_tool_call_cnt)
                    history_tool_call_cnt += 1
                    # Prefer parameters from the structured object; fallback to 'arguments' in raw array
                    if func_def.parameters is not None:
                        args_str = json.dumps(func_def.parameters,
                                                ensure_ascii=False)
                    else:
                        fallback_args = None
                        if isinstance(raw_arr, list):
                            for item in raw_arr:
                                if isinstance(item, dict) and item.get("name") == func_def.name:
                                    fallback_args = item.get("arguments")
                                    if fallback_args is None:
                                        fallback_args = item.get("parameters")
                                    break
                        if isinstance(fallback_args, str):
                            try:
                                fallback_args = json.loads(fallback_args)
                            except Exception:
                                pass
                        try:
                            args_str = json.dumps(fallback_args, ensure_ascii=False)
                        except Exception:
                            args_str = "null" if fallback_args is None else str(fallback_args)
                    tool_call_messages.append(
                        tool_call_class(
                            id=tc_id,
                            function=FunctionCall(
                                name=func_def.name,
                                arguments=args_str,
                            ),
                        ))
            else:
                # Fallback: parse Nemotron-style array and map to required format
                arr = self._load_tool_array_forgiving(content or "")
                if arr and isinstance(arr, list):
                    for item in arr:
                        if not isinstance(item, dict) or "name" not in item:
                            continue
                        name = item.get("name")
                        params = item.get("parameters")
                        if params is None:
                            params = item.get("arguments")
                        if isinstance(params, str):
                            try:
                                params = json.loads(params)
                            except Exception:
                                pass
                        tc_id = make_tool_call_id(
                            id_type=self.tool_call_id_type,
                            func_name=name,
                            idx=history_tool_call_cnt)
                        history_tool_call_cnt += 1
                        try:
                            args_str = json.dumps(params, ensure_ascii=False)
                        except Exception:
                            args_str = str(params)
                        tool_call_messages.append(
                            tool_call_class(
                                id=tc_id,
                                function=FunctionCall(
                                    name=name,
                                    arguments=args_str,
                                ),
                            ))
                # Last fallback: if auto tool parser is available, try it
                elif self.tool_parser is not None:
                    try:
                        tool_parser = self.tool_parser(tokenizer)
                        tci = tool_parser.extract_tool_calls(
                            content if content is not None else "",
                            request=request)
                        if tci.tools_called and tci.tool_calls:
                            tool_call_messages = tci.tool_calls
                    except Exception:
                        logger.debug("Nemotron required fallback parse failed", exc_info=True)

            if not tool_call_messages:
                # If still nothing parsed, avoid raising 400; return as plain content
                message = ChatMessage(role=role,
                                        reasoning_content=reasoning_content,
                                        content=content)
            else:
                message = ChatMessage(
                    role=role,
                    content="",
                    tool_calls=tool_call_messages,
                    reasoning_content=reasoning_content)

        # if the request doesn't use tool choice
        # OR specifies to not use a tool
        elif not request.tool_choice or request.tool_choice == "none":

            message = ChatMessage(role=role,
                                    reasoning_content=reasoning_content,
                                    content=content)

        # handle when there are tools and tool choice is auto
        elif request.tools and (
                request.tool_choice == "auto"
                or request.tool_choice is None) and self.enable_auto_tools \
                and self.tool_parser:

            try:
                tool_parser = self.tool_parser(tokenizer)
            except RuntimeError as e:
                logger.exception("Error in tool parser creation.")
                raise e

            tool_call_info = tool_parser.extract_tool_calls(
                content if content is not None else "", request=request)
            # In the OpenAI API the finish_reason is "tools_called"
            # if the tool choice is auto and the model produced a tool
            # call. The same is not true for named function calls
            auto_tools_called = tool_call_info.tools_called
            if tool_call_info.tools_called:
                message = ChatMessage(role=role,
                                      reasoning_content=reasoning_content,
                                      content=tool_call_info.content,
                                      tool_calls=tool_call_info.tool_calls)

            else:
                # FOR NOW make it a chat message; we will have to detect
                # the type to make it later.
                ret_content = content

                # try to use content return from tool parser first,
                # tool parser may do some modify for the content.
                if (tool_call_info.content
                        and len(tool_call_info.content) > 0):
                    ret_content = tool_call_info.content
                message = ChatMessage(role=role,
                                      reasoning_content=reasoning_content,
                                      content=ret_content)

        # undetermined case that is still important to handle
        else:
            logger.error(
                "Error in chat_completion_full_generator - cannot determine"
                " if tools should be extracted. Returning a standard chat "
                "completion.")
            message = ChatMessage(role=role,
                                  reasoning_content=reasoning_content,
                                  content=content)

        return message
    
    def _should_stream_with_auto_tool_parsing(self,
                                              request: ChatCompletionRequest):
        """
        Utility function to check if streamed tokens should go through the tool
        call parser that was configured.

        We only want to do this IF user-provided tools are set, a tool parser
        is configured, "auto" tool choice is enabled, and the request's tool
        choice field indicates that "auto" tool choice should be used.
        """
        return (request.tools and self.tool_parser and self.enable_auto_tools
                and request.tool_choice in ['auto', None])

    def _should_check_for_unstreamed_tool_arg_tokens(
        self,
        delta_message: Optional[DeltaMessage],
        output: CompletionOutput,
    ) -> bool:
        """
        Check to see if we should check for unstreamed tool arguments tokens.
        This is only applicable when auto tool parsing is enabled, the delta
        is a tool call with arguments.
        """

        # yapf: disable
        return bool(
            # if there is a delta message that includes tool calls which
            # include a function that has arguments
            output.finish_reason is not None
            and self.enable_auto_tools and self.tool_parser and delta_message
            and delta_message.tool_calls and delta_message.tool_calls[0]
            and delta_message.tool_calls[0].function
            and delta_message.tool_calls[0].function.arguments is not None
        )

    def _make_request_with_harmony(
        self,
        request: ChatCompletionRequest,
    ):
        messages: list[OpenAIMessage] = []

        # Add system message.
        # NOTE: In Chat Completion API, browsing is enabled by default
        # if the model supports it. TODO: Support browsing.
        assert not self.supports_browsing
        assert not self.supports_code_interpreter
        sys_msg = get_system_message(
            reasoning_effort=request.reasoning_effort,
            browser_description=None,
            python_description=None)
        messages.append(sys_msg)

        # Add developer message.
        dev_msg = get_developer_message(tools=request.tools)
        messages.append(dev_msg)

        # Add user message.
        for chat_msg in request.messages:
            messages.extend(parse_chat_input(chat_msg))

        # Render prompt token ids.
        prompt_token_ids = render_for_completion(messages)
        engine_prompt = EngineTokensPrompt(prompt_token_ids=prompt_token_ids)

        # Add cache_salt if provided in the request
        if request.cache_salt is not None:
            engine_prompt["cache_salt"] = request.cache_salt

        return messages, [prompt_token_ids], [engine_prompt]

    def create_chat_completion_response(
        self,
        request: ChatCompletionRequest,
        choices: list[ChatCompletionResponseChoice],
        request_id: str,
        request_metadata: RequestResponseMetadata,
        model_name: Optional[str] = None,
        created_time: Optional[int] = None,
        final_res: Optional[RequestOutput] = None,
    ) -> ChatCompletionResponse:
        # Resolve model name
        resolved_model_name = model_name or request.model or self.models.model_name(None)

        # Compute usage
        usage = request_metadata.final_usage_info
        if usage is None:
            prompt_tokens = 0
            completion_tokens = 0
            if final_res is not None:
                if final_res.prompt_token_ids is not None:
                    prompt_tokens += len(final_res.prompt_token_ids)
                if getattr(final_res, "encoder_prompt_token_ids", None):
                    prompt_tokens += len(getattr(final_res, "encoder_prompt_token_ids"))
                try:
                    completion_tokens = sum(
                        len(o.token_ids) for o in final_res.outputs if o.token_ids is not None
                    )
                except Exception:
                    completion_tokens = 0
            usage = UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )

        # Build response
        response = ChatCompletionResponse(
            id=request_id,
            model=resolved_model_name,
            choices=choices,
            usage=usage,
        )

        # Optionally include prompt_token_ids if requested
        if getattr(request, "return_token_ids", False) and final_res is not None:
            try:
                response.prompt_token_ids = final_res.prompt_token_ids
            except Exception:
                pass

        return response
