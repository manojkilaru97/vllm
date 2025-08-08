# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json
import time
from collections.abc import AsyncGenerator, AsyncIterator
from collections.abc import Sequence as GenericSequence
from typing import Any, Callable, Dict, Final, List, Optional, Tuple, Union

import jinja2
import partial_json_parser
import regex as re
from fastapi import Request
from openai_harmony import Message as OpenAIMessage
from pydantic import TypeAdapter

from vllm.config import ModelConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import (ChatTemplateContentFormatOption,
                                         ConversationMessage,
                                         random_tool_call_id)
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
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.reasoning import ReasoningParser, ReasoningParserManager
from vllm.sampling_params import BeamSearchParams, SamplingParams
from vllm.sequence import Logprob
from vllm.transformers_utils.tokenizer import AnyTokenizer, MistralTokenizer
from vllm.transformers_utils.tokenizers import (maybe_serialize_tool_calls,
                                                truncate_tool_call_ids,
                                                validate_request_params)

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
    ) -> None:
        super().__init__(engine_client=engine_client,
                         model_config=model_config,
                         models=models,
                         request_logger=request_logger,
                         return_tokens_as_token_ids=return_tokens_as_token_ids,
                         enable_force_include_usage=enable_force_include_usage)

        self.response_role = response_role
        self.chat_template = chat_template
        self.chat_template_content_format: Final = chat_template_content_format

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
        
        # Video safety guardrail components (initialized lazily)
        self._video_safety_guardrail: Optional["VideoSafetyGuardrail"] = None

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

        # Register NVCF assets (videos/images) if provided via request headers.
        if raw_request is not None:
            headers = raw_request.headers
            if "NVCF-ASSET-DIR" in headers and "NVCF-FUNCTION-ASSET-IDS" in headers:
                from pathlib import Path
                from vllm.multimodal.utils import MediaConnector
                nvcf_asset_dir = Path(headers["NVCF-ASSET-DIR"])
                if not (nvcf_asset_dir.exists() and nvcf_asset_dir.is_dir()):
                    raise RuntimeError(
                        f"NVCF asset directory {nvcf_asset_dir} does not exist or is not a directory.")
                asset_ids = [aid.strip() for aid in headers["NVCF-FUNCTION-ASSET-IDS"].split(",") if aid.strip()]
                MediaConnector.set_default_nvcf_assets({aid: nvcf_asset_dir / aid for aid in asset_ids})
            else:
                # Clear the default assets if not provided in the current request.
                from vllm.multimodal.utils import MediaConnector as _MC
                _MC.set_default_nvcf_assets(None)

        try:
            # Convert inline <video src="..."/> tags in string messages into
            # structured content parts so MediaConnector can load the videos.
            import re
            inline_video_pattern = re.compile(r"<video\s+src=\"([^\"]+)\"\s*/?>", re.IGNORECASE)
            if isinstance(request.messages, list):
                new_messages = []
                for msg in request.messages:
                    msg_dict = msg if isinstance(msg, dict) else msg.model_dump()
                    content = msg_dict.get("content")
                    if isinstance(content, str) and "<video" in content:
                        urls = inline_video_pattern.findall(content)
                        # Remove the inline tags from the text content
                        cleaned_text = inline_video_pattern.sub("", content)
                        structured = []
                        if cleaned_text.strip():
                            structured.append({"type": "text", "text": cleaned_text})
                        for url in urls:
                            structured.append({"type": "video_url", "video_url": {"url": url}})
                        msg_dict["content"] = structured
                    new_messages.append(msg_dict)
                request.messages = new_messages

            # Guardrail pre-check: if any video content is present, run Cosmos Video Content Safety Filter
            try:
                has_video = False
                video_urls: list[str] = []
                if isinstance(request.messages, list):
                    for msg in request.messages:
                        content = msg.get("content") if isinstance(msg, dict) else None
                        if isinstance(content, list):
                            for part in content:
                                if isinstance(part, dict) and part.get("type") == "video_url":
                                    url = ((part.get("video_url") or {}).get("url")
                                           if isinstance(part.get("video_url"), dict) else None)
                                    if isinstance(url, str) and url:
                                        has_video = True
                                        video_urls.append(url)
                logger.debug("Video guardrail: Found video URL: %s", url)
                
                logger.debug("Video guardrail: has_video=%s, video_count=%d", has_video, len(video_urls))
                if has_video:
                    # Apply video content safety guardrail if enabled
                    app = raw_request.app if raw_request is not None else None
                    args = getattr(app.state, "serve_args", None)
                    enable_guard = bool(getattr(args, "enable_video_guardrail", False)) if args else False
                    
                    logger.debug("Video guardrail: enable_guard=%s", enable_guard)
                    if enable_guard:
                        try:
                            logger.debug("Video guardrail: Starting safety check for %d videos", len(video_urls))
                            guardrail_response = await self._check_video_safety(video_urls, args)
                            if guardrail_response is not None:
                                logger.warning("Video guardrail: BLOCKED unsafe content")
                                return guardrail_response
                            logger.debug("Video guardrail: Content passed safety check")
                        except Exception as e:
                            logger.warning("Video guardrail check failed: %s", e)
                            # Continue processing - fail open for non-critical guardrail errors
            except Exception as e:
                # Fail closed? Manager requested blocking unsafe only; if guard fails, continue but log
                logger.warning("Video guardrail check failed: %s", e)

            lora_request = self._maybe_get_adapters(
                request, supports_default_mm_loras=True)

            model_name = self._get_model_name(request.model, lora_request)

            tokenizer = await self.engine_client.get_tokenizer(lora_request)

            tool_parser = self.tool_parser

            # Per-request MM processor overrides (fps, max_pixels, add_timestamps)
            if (hasattr(request, 'fps') and request.fps is not None) or (
                hasattr(request, 'max_pixels') and request.max_pixels is not None) or (
                getattr(request, 'add_timestamps', None)):
                # Update mm_processor_kwargs (these are consumed by HF processors)
                mm_kwargs = dict(request.mm_processor_kwargs or {})
                if getattr(request, 'fps', None) is not None:
                    mm_kwargs['fps'] = request.fps
                if getattr(request, 'max_pixels', None) is not None:
                    mm_kwargs['max_pixels'] = request.max_pixels
                if getattr(request, 'add_timestamps', None):
                    mm_kwargs['add_timestamps'] = True
                request.mm_processor_kwargs = mm_kwargs

                # CRITICAL: Update media_io_kwargs BEFORE _preprocess_chat is called
                # so that VideoMediaIO gets the add_timestamps flag
                original_media_io_kwargs = self.model_config.media_io_kwargs
                media_io_kwargs = dict(original_media_io_kwargs or {})
                video_io_kwargs = dict(media_io_kwargs.get('video', {}))
                
                if getattr(request, 'fps', None) is not None:
                    video_io_kwargs['fps'] = request.fps
                if getattr(request, 'add_timestamps', None):
                    video_io_kwargs['add_timestamps'] = True
                    
                if video_io_kwargs:
                    media_io_kwargs['video'] = video_io_kwargs
                    # Temporarily modify model_config for this request
                    self.model_config.media_io_kwargs = media_io_kwargs

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
                    truncate_prompt_tokens=request.truncate_prompt_tokens,
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

    def extract_tool_call_required_streaming(
        self,
        previous_text: str,
        current_text: Optional[str],
        delta_text: str,
        function_name_returned: bool,
    ) -> tuple[Optional[DeltaMessage], bool]:
        if current_text is None or current_text == "":
            # if the current text is empty, we cannot parse it
            return None, function_name_returned
        try:
            obj = partial_json_parser.loads(current_text)
        except partial_json_parser.core.exceptions.MalformedJSON:
            logger.debug('not enough tokens to parse into JSON yet')
            obj = None

        # check if the current text is a valid array
        # containing a partial tool calling object
        # if not repeat
        if obj is None or not isinstance(obj, list) or not len(obj) > 0:
            function_name_returned = False
            delta_message = None
        else:
            _, finishes_previous_tool = OpenAIServingChat._filter_delta_text(
                delta_text, previous_text)
            # take the last tool call from the generated list
            current_tool_call = obj[-1]

            # once parameters have been generated the name is complete as well
            if not finishes_previous_tool and ("name" not in current_tool_call
                                               or "parameters"
                                               not in current_tool_call):
                function_name_returned = False
                delta_message = None
            else:
                if not function_name_returned:
                    # get partly generated arguments from the latest tool call
                    param_match = re.search(r'.*"parameters":\s*(.*)',
                                            current_text)
                    arguments = param_match.group(1) if param_match else ""
                    arguments, _ = OpenAIServingChat._filter_delta_text(
                        arguments, previous_text)

                    # if this iteration finishes a previous tool call but a
                    # new incomplete tool is already generated, take the
                    # previous from the list
                    if (finishes_previous_tool
                            and "parameters" not in current_tool_call):
                        current_tool_call = obj[-2]

                    function_name_returned = True
                    delta_message = DeltaMessage(tool_calls=[
                        DeltaToolCall(id=random_tool_call_id(),
                                      function=DeltaFunctionCall(
                                          name=current_tool_call["name"],
                                          arguments=arguments),
                                      index=len(obj) - 1,
                                      type="function")
                    ])

                else:
                    delta_text, _ = OpenAIServingChat._filter_delta_text(
                        delta_text, previous_text)

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

        if isinstance(request.tool_choice, ChatCompletionNamedToolChoiceParam):
            tool_choice_function_name = request.tool_choice.function.name
        else:
            tool_choice_function_name = None

        # Determine whether tools are in use with "auto" tool choice
        tool_choice_auto = (
            not tool_choice_function_name
            and self._should_stream_with_auto_tool_parsing(request))

        all_previous_token_ids: Optional[list[list[int]]]
        function_name_returned = [False] * num_choices

        # Only one of these will be used, thus previous_texts and
        # all_previous_token_ids will not be used twice in the same iteration.
        if tool_choice_auto or self.reasoning_parser:
            # These are only required in "auto" tool choice case
            previous_texts = [""] * num_choices
            all_previous_token_ids = [[]] * num_choices
            # For reasoning parser and tool call all enabled
            added_content_delta_arr = [False] * num_choices
            reasoning_end_arr = [False] * num_choices
        elif request.tool_choice == "required":
            previous_texts = [""] * num_choices
            all_previous_token_ids = None
        else:
            previous_texts, all_previous_token_ids = None, None

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
                        chunk = ChatCompletionStreamResponse(
                            id=request_id,
                            object=chunk_object_type,
                            created=created_time,
                            choices=[choice_data],
                            model=model_name)

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
                        for token_id in output.token_ids:
                            harmony_parser.process(token_id)
                        # FIXME(woosuk): Support function calling
                        is_final = harmony_parser.current_channel == "final"
                        if not (request.include_reasoning or is_final):
                            # Skip the reasoning content.
                            continue
                        delta_text = harmony_parser.last_content_delta or ""
                    else:
                        delta_text = output.text

                    if not delta_text and not output.token_ids and \
                        not previous_num_tokens[i]:
                        # Chunked prefill case, don't return empty chunks
                        continue

                    delta_message: Optional[DeltaMessage]

                    # just update previous_texts and previous_token_ids
                    if ((tool_choice_auto or self.reasoning_parser)
                            and not self.use_harmony):
                        assert previous_texts is not None
                        assert all_previous_token_ids is not None
                        previous_text = previous_texts[i]
                        previous_token_ids = all_previous_token_ids[i]
                        current_text = previous_text + delta_text

                        # avoid the None + list error.
                        if previous_token_ids:
                            current_token_ids = previous_token_ids + list(
                                output.token_ids)
                        else:
                            current_token_ids = list(output.token_ids)

                    if self.use_harmony:
                        if is_final:
                            delta_message = DeltaMessage(content=delta_text)
                        else:
                            delta_message = DeltaMessage(
                                reasoning_content=delta_text)
                    # handle streaming deltas for tools with named tool_choice
                    elif tool_choice_function_name:
                        if (self.reasoning_parser and not reasoning_end_arr[i]
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
                                    list(output.token_ids)) or \
                                    (res.prompt_token_ids and
                                        reasoning_parser.is_reasoning_end(
                                            list(res.prompt_token_ids)
                                        )):
                                reasoning_end_arr[i] = True
                                if delta_message and delta_message.content:
                                    # This need to be added to next `delta_text`
                                    current_text = delta_message.content
                                    delta_message.content = None
                                else:
                                    current_text = ""
                        else:
                            # Just to add remaining `content`
                            if self.reasoning_parser:
                                delta_text = previous_text + delta_text
                                current_text = ""

                            if function_name_returned[i]:
                                delta_tool_call = DeltaToolCall(
                                    function=DeltaFunctionCall(
                                        arguments=delta_text),
                                    index=i)
                            else:
                                delta_tool_call = DeltaToolCall(
                                    id=random_tool_call_id(),
                                    type="function",
                                    function=DeltaFunctionCall(
                                        name=tool_choice_function_name,
                                        arguments=delta_text),
                                    index=i)
                                function_name_returned[i] = True

                            delta_message = DeltaMessage(tool_calls=[
                                delta_tool_call,
                            ])

                    elif request.tool_choice == "required":
                        assert previous_texts is not None
                        previous_text = previous_texts[i]
                        current_text = previous_text + delta_text
                        fn_name_returned = function_name_returned[i]

                        if self.reasoning_parser:
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
                                function_name_returned=fn_name_returned))

                        # update the previous values for the next iteration
                        previous_texts[i] = current_text

                    # handle streaming deltas for tools with "auto" tool choice
                    # and reasoning parser
                    elif tool_choice_auto and self.reasoning_parser:
                        assert tool_parser is not None
                        assert reasoning_parser is not None
                        assert added_content_delta_arr is not None
                        assert reasoning_end_arr is not None
                        if not reasoning_end_arr[i]:
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
                            # When encountering think end id in prompt_token_ids
                            # i.e {"enable_thinking": False},
                            # set reasoning status to end.
                            # Remove the text and token ids related
                            # to 'reasoning_content'.
                            if res.prompt_token_ids and \
                                reasoning_parser.is_reasoning_end(
                                    list(res.prompt_token_ids)):
                                reasoning_end_arr[i] = True
                                current_token_ids = list(output.token_ids)
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
                                    list(output.token_ids)):
                                reasoning_end_arr[i] = True
                                current_token_ids =  \
                                    reasoning_parser.extract_content_ids(
                                        list(output.token_ids))
                                if delta_message and delta_message.content:
                                    current_text = delta_message.content
                                    delta_message.content = None
                                else:
                                    current_text = ""

                        # handle tool calls only after reasoning is done,
                        else:
                            delta_token_ids = list(output.token_ids)
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
                    # when only reasoning
                    elif self.reasoning_parser:
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
                    if tool_choice_auto or self.reasoning_parser:
                        assert previous_texts is not None
                        assert all_previous_token_ids is not None
                        previous_texts[i] = current_text
                        all_previous_token_ids[i] = current_token_ids

                    # set the previous values for the next iteration
                    previous_num_tokens[i] += len(output.token_ids)

                    # if the message delta is None (e.g. because it was a
                    # "control token" for tool calls or the parser otherwise
                    # wasn't ready to send a token, then
                    #   get the next token without streaming a chunk
                    if delta_message is None:
                        continue

                    if output.finish_reason is None:
                        # Send token-by-token response for each request.n
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=delta_message,
                            logprobs=logprobs,
                            finish_reason=None)

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

                        if self._should_check_for_unstreamed_tool_arg_tokens(
                                delta_message, output) and tool_parser:
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
                            remaining_call = expected_call.replace(
                                actual_call, "", 1)
                            # set that as a delta message
                            delta_message = DeltaMessage(tool_calls=[
                                DeltaToolCall(index=index,
                                              function=DeltaFunctionCall(
                                                  arguments=remaining_call).
                                              model_dump(exclude_none=True))
                            ])

                        # Send the finish response for each request.n only once
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=delta_message,
                            logprobs=logprobs,
                            finish_reason=output.finish_reason
                            if not auto_tools_called else "tool_calls",
                            stop_reason=output.stop_reason)

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
                total_tokens=num_prompt_tokens + num_completion_tokens)

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
                reasoning_content, final_content, is_tool_call = (
                    parse_chat_output(token_ids))
                if not request.include_reasoning:
                    reasoning_content = None

                if is_tool_call:
                    # TODO(woosuk): Implement tool call for gpt-oss.
                    # For now, only Responses API supports tool call for
                    # gpt-oss.
                    raise NotImplementedError(
                        "Tool call in Chat Completion API is not supported "
                        "for gpt-oss yet. Please use Responses API instead.")
                else:
                    # Normal message
                    message = ChatMessage(
                        role=role,
                        reasoning_content=reasoning_content,
                        content=final_content,
                    )

                choice_data = ChatCompletionResponseChoice(
                    index=output.index,
                    message=message,
                    logprobs=logprobs,
                    finish_reason="tool_calls" if is_tool_call else
                    output.finish_reason if output.finish_reason else "stop",
                    stop_reason=output.stop_reason,
                )
                choices.append(choice_data)
                continue

            if self.reasoning_parser:
                try:
                    reasoning_parser = self.reasoning_parser(tokenizer)
                except RuntimeError as e:
                    logger.exception("Error in reasoning parser creation.")
                    return self.create_error_response(str(e))
                # If the reasoning parser is enabled,
                # tool calls are extracted exclusively from the content.
                reasoning_content, content = (
                    reasoning_parser.extract_reasoning_content(
                        output.text, request=request))
                if not request.include_reasoning:
                    reasoning_content = None
            else:
                reasoning_content = None
                content = output.text

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
                message = ChatMessage(
                    role=role,
                    reasoning_content=reasoning_content,
                    content="",
                    tool_calls=[
                        tool_call_class(function=FunctionCall(
                            name=request.tool_choice.function.name,
                            arguments=content))
                    ])

            elif request.tool_choice and request.tool_choice == "required":
                tool_call_class = MistralToolCall if isinstance(
                    tokenizer, MistralTokenizer) else ToolCall

                # the fields of FunctionDefinition are a superset of the
                # tool call outputs and can be used for parsing
                assert content is not None
                tool_calls = TypeAdapter(
                    list[FunctionDefinition]).validate_json(content)
                message = ChatMessage(
                    role=role,
                    content="",
                    reasoning_content=reasoning_content,
                    tool_calls=[
                        tool_call_class(function=FunctionCall(
                            name=tool_call.name,
                            arguments=json.dumps(tool_call.parameters,
                                                 ensure_ascii=False)))
                        for tool_call in tool_calls
                    ])

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
                    return self.create_error_response(str(e))

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

            choice_data = ChatCompletionResponseChoice(
                index=output.index,
                message=message,
                logprobs=logprobs,
                finish_reason="tool_calls" if auto_tools_called else
                output.finish_reason if output.finish_reason else "stop",
                stop_reason=output.stop_reason)
            choices.append(choice_data)

        if request.echo:
            last_msg_content: Union[str, list[dict[str, str]]] = ""
            if conversation and "content" in conversation[-1] and conversation[
                    -1].get("role") == role:
                last_msg_content = conversation[-1]["content"] or ""
            if isinstance(last_msg_content, list):
                last_msg_content = "\n".join(msg['text']
                                             for msg in last_msg_content)

            for choice in choices:
                full_message = last_msg_content + (choice.message.content
                                                   or "")
                choice.message.content = full_message

        assert final_res.prompt_token_ids is not None
        num_prompt_tokens = len(final_res.prompt_token_ids)
        if final_res.encoder_prompt_token_ids is not None:
            num_prompt_tokens += len(final_res.encoder_prompt_token_ids)
        num_generated_tokens = sum(
            len(output.token_ids) for output in final_res.outputs)
        usage = UsageInfo(prompt_tokens=num_prompt_tokens,
                          completion_tokens=num_generated_tokens,
                          total_tokens=num_prompt_tokens +
                          num_generated_tokens)
        if self.enable_prompt_tokens_details and final_res.num_cached_tokens:
            usage.prompt_tokens_details = PromptTokenUsageInfo(
                cached_tokens=final_res.num_cached_tokens)

        request_metadata.final_usage_info = usage

        response = ChatCompletionResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=choices,
            usage=usage,
            prompt_logprobs=clamp_prompt_logprobs(final_res.prompt_logprobs),
            kv_transfer_params=final_res.kv_transfer_params,
        )

        return response

    def _get_top_logprobs(
            self, logprobs: dict[int, Logprob], top_logprobs: Optional[int],
            tokenizer: AnyTokenizer,
            should_return_as_token_id: bool) -> list[ChatCompletionLogProb]:
        return [
            ChatCompletionLogProb(token=(token := self._get_decoded_token(
                p[1],
                p[0],
                tokenizer,
                return_as_token_id=should_return_as_token_id)),
                                  logprob=max(p[1].logprob, -9999.0),
                                  bytes=list(
                                      token.encode("utf-8", errors="replace")))
            for i, p in enumerate(logprobs.items())
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
                token = tokenizer.decode(token_id)
                if should_return_as_token_id:
                    token = f"token_id:{token_id}"

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
        dev_msg = get_developer_message()
        messages.append(dev_msg)

        # Add user message.
        for chat_msg in request.messages:
            messages.append(parse_chat_input(chat_msg))

        # Render prompt token ids.
        prompt_token_ids = render_for_completion(messages)
        engine_prompt = EngineTokensPrompt(prompt_token_ids=prompt_token_ids)
        return messages, [prompt_token_ids], [engine_prompt]

    async def _check_video_safety(
        self,
        video_urls: List[str],
        args: Any
    ) -> Optional[ErrorResponse]:
        """
        Check video content safety using the Cosmos video content safety filter.
        
        Args:
            video_urls: List of video URLs to check
            args: Server arguments containing guardrail configuration
            
        Returns:
            ErrorResponse if unsafe content is detected, None if safe
        """
        try:
            # Initialize guardrail lazily
            if self._video_safety_guardrail is None:
                self._video_safety_guardrail = VideoSafetyGuardrail(args)
            
            # Check each video
            for video_url in video_urls:
                is_safe, reason = await self._video_safety_guardrail.check_video(
                    video_url, self.model_config
                )
                if not is_safe:
                    from http import HTTPStatus
                    return self.create_error_response(
                        message=f"Unsafe video content detected: {reason}",
                        err_type="ContentSafetyError",
                        status_code=HTTPStatus.UNPROCESSABLE_ENTITY
                    )
            
            return None  # All videos are safe
            
        except Exception as e:
            logger.error("Video safety check failed: %s", e)
            raise


class VideoSafetyGuardrail:
    """
    Production-ready video content safety guardrail using Cosmos models.
    
    This class implements the Video Content Safety Filter from the
    nvidia/Cosmos-Guardrail1 model, providing real-time video content
    safety classification with configurable thresholds.
    """
    
    def __init__(self, args: Any):
        """Initialize the video safety guardrail."""
        self.args = args
        self.threshold = getattr(args, "video_guardrail_threshold", 0.5)
        self.checkpoint_root = getattr(args, "video_guardrail_checkpoint_dir", None)
        
        # Validate configuration
        if not self.checkpoint_root:
            raise ValueError("Video guardrail requires --video-guardrail-checkpoint-dir")
        
        # Initialize models
        self._encoder: Optional["SigLIPEncoder"] = None
        self._safety_model: Optional["VideoSafetyNet"] = None
        self._device = "cuda" if self._is_cuda_available() else "cpu"
        self._dtype = self._get_torch_dtype()
        
        logger.info(
            "Initialized VideoSafetyGuardrail with threshold=%.3f, device=%s",
            self.threshold, self._device
        )

    def _is_cuda_available(self) -> bool:
        """Check if CUDA is available with proper error handling."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            logger.warning("PyTorch not available, falling back to CPU")
            return False

    def _get_torch_dtype(self):
        """Get the appropriate torch dtype."""
        try:
            import torch
            return torch.float32
        except ImportError:
            return None

    def _load_models(self):
        """Load the SigLIP encoder and video safety classifier."""
        if self._encoder is not None and self._safety_model is not None:
            return  # Already loaded
            
        try:
            import os
            import torch
            import torch.nn as nn
            from transformers import SiglipModel, SiglipProcessor
            
            # Resolve checkpoint directory
            base_paths = [
                os.path.join(self.checkpoint_root, "nvidia", "Cosmos-Guardrail1", "video_content_safety_filter"),
                os.path.join(self.checkpoint_root, "Cosmos-Guardrail1", "video_content_safety_filter")
            ]
            
            base_dir = None
            for path in base_paths:
                if os.path.isdir(path):
                    base_dir = path
                    break
                    
            if not base_dir:
                raise FileNotFoundError(
                    f"Video guardrail checkpoints not found under {self.checkpoint_root}. "
                    f"Tried paths: {base_paths}"
                )
            
            logger.info("Loading video safety models from: %s", base_dir)
            
            # Initialize SigLIP encoder
            self._encoder = SigLIPEncoder(base_dir, self._device, self._dtype)
            
            # Initialize and load safety classifier
            self._safety_model = VideoSafetyNet()
            ckpt_path = os.path.join(base_dir, "safety_filter.pt")
            
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Safety filter checkpoint not found: {ckpt_path}")
                
            state = torch.load(ckpt_path, map_location="cpu")
            model_state = state.get("model", state)
            self._safety_model.load_state_dict(model_state, strict=False)
            self._safety_model.to(self._device, dtype=self._dtype).eval()
            
            logger.info("Successfully loaded video safety models")
            
        except Exception as e:
            logger.error("Failed to load video safety models: %s", e)
            raise RuntimeError(f"Video guardrail initialization failed: {e}") from e

    async def check_video(
        self,
        video_url: str,
        model_config: Any
    ) -> Tuple[bool, str]:
        """
        Check if a video contains safe content.
        
        Args:
            video_url: URL of the video to check
            model_config: Model configuration for media processing
            
        Returns:
            Tuple of (is_safe, reason) where is_safe is True if content is safe,
            and reason provides details about the classification
        """
        try:
            # Load models if not already loaded
            self._load_models()
            
            # Fetch video frames
            frames, metadata = await self._fetch_video_frames(video_url, model_config)

            # Normalize frames into a Python list of frame arrays to avoid
            # ambiguous truth-value checks on numpy arrays
            try:
                import numpy as _np  # local import to avoid top-level dependency
                if isinstance(frames, _np.ndarray):
                    # Expect shape [num_frames, H, W, C] or similar
                    if frames.ndim >= 4:
                        frames = [frames[i] for i in range(frames.shape[0])]
                    elif frames.ndim == 3:
                        # Single frame array
                        frames = [frames]
            except Exception:
                # If numpy is unavailable, proceed assuming frames is already a list
                pass

            if frames is None or len(frames) == 0:
                return False, "No valid frames found in video"
            
            # Sample frames at ~2 FPS, but cap the total number of sampled frames
            step = self._calculate_sampling_step(frames, metadata)
            max_samples = int(getattr(self.args, "video_guardrail_max_sampled_frames", 60) or 60)
            
            # Check each sampled frame
            unsafe_frames = 0
            total_checked = 0
            min_safe_prob = 1.0
            
            sampled_indices = list(range(0, len(frames), step))
            if len(sampled_indices) > max_samples:
                # Evenly subsample to max_samples
                import numpy as _np
                sampled_indices = list(_np.linspace(0, len(frames) - 1, max_samples, dtype=int))

            for idx in sampled_indices:
                try:
                    safe_prob = self._get_frame_safety_probability(frames[idx])
                    min_safe_prob = min(min_safe_prob, safe_prob)
                    total_checked += 1
                    
                    if safe_prob < self.threshold:
                        unsafe_frames += 1
                        logger.debug(
                            "Unsafe frame detected at index %d: safe_prob=%.3f < threshold=%.3f",
                            idx, safe_prob, self.threshold
                        )
                        # Fail fast on first unsafe frame
                        return False, f"safe_prob={safe_prob:.3f} < {self.threshold}"
                        
                except Exception as e:
                    logger.debug("Failed to process frame %d: %s", idx, e)
                    continue
            
            if total_checked == 0:
                return False, "No frames could be processed"
                
            logger.debug(
                "Video safety check completed: %d/%d frames checked, min_safe_prob=%.3f",
                total_checked, len(frames), min_safe_prob
            )
            
            return True, f"safe (min_prob={min_safe_prob:.3f})"
            
        except Exception as e:
            logger.error("Video safety check failed for %s: %s", video_url, e)
            raise

    async def _fetch_video_frames(
        self,
        video_url: str,
        model_config: Any
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """Fetch frames from video URL."""
        try:
            from vllm.multimodal.utils import MediaConnector
            
            connector = MediaConnector(
                media_io_kwargs=model_config.media_io_kwargs,
                allowed_local_media_path=getattr(model_config, "allowed_local_media_path", ""),
            )
            
            frames, metadata = connector.fetch_video(video_url)
            return frames, metadata or {}
            
        except Exception as e:
            logger.error("Failed to fetch video frames from %s: %s", video_url, e)
            raise

    def _calculate_sampling_step(
        self,
        frames: List[Any],
        metadata: Dict[str, Any]
    ) -> int:
        """Calculate the frame sampling step to achieve ~2 FPS."""
        try:
            fps = int(metadata.get("fps", 0)) if metadata else 0
            if fps > 0:
                # Sample at 2 FPS
                step = max(1, fps // 2)
            else:
                # Fallback: sample ~30 frames across video length
                step = max(1, len(frames) // 30)
                
            logger.debug(
                "Calculated sampling step=%d for %d frames (fps=%s)",
                step, len(frames), fps or "unknown"
            )
            return step
            
        except Exception:
            # Conservative fallback
            return max(1, len(frames) // 30)

    def _get_frame_safety_probability(self, frame_array) -> float:
        """Get safety probability for a single frame."""
        try:
            import torch
            from PIL import Image
            
            # Convert frame to PIL Image
            pil_image = Image.fromarray(frame_array)
            
            # Encode image using SigLIP
            with torch.inference_mode():
                features = self._encoder.encode_image(pil_image)
                
                # Classify using safety model
                logits = self._safety_model(features)
                probs = torch.softmax(logits, dim=-1)
                
                # Handle different tensor shapes
                if probs.dim() == 1:
                    # Single sample, shape: [num_classes]
                    safe_prob = float(probs[0].item())
                elif probs.dim() == 2:
                    # Batch dimension, shape: [batch_size, num_classes]
                    safe_prob = float(probs[0, 0].item())
                else:
                    raise ValueError(f"Unexpected probs tensor shape: {probs.shape}")
                
                logger.debug("Frame safety probability: %.3f", safe_prob)
                return safe_prob
                
        except Exception as e:
            logger.error("Failed to get frame safety probability: %s", e)
            raise


class SigLIPEncoder:
    """SigLIP image encoder for video content safety classification."""
    
    def __init__(self, base_dir: str, device: str, dtype):
        """Initialize the SigLIP encoder."""
        self.device = device
        self.dtype = dtype
        
        try:
            from transformers import SiglipModel, SiglipProcessor
            
            model_name = "google/siglip-so400m-patch14-384"
            self.model = SiglipModel.from_pretrained(
                model_name,
                cache_dir=base_dir,
                local_files_only=True
            )
            self.processor = SiglipProcessor.from_pretrained(
                model_name,
                cache_dir=base_dir,
                local_files_only=True
            )
            
            self.model.to(self.device, dtype=self.dtype).eval()
            logger.info("Initialized SigLIP encoder on device: %s", self.device)
            
        except Exception as e:
            logger.error("Failed to initialize SigLIP encoder: %s", e)
            raise

    def encode_image(self, pil_image):
        """Encode a PIL image to normalized features."""
        import torch
        
        inputs = self.processor(
            images=pil_image,
            return_tensors="pt"
        ).to(self.device, dtype=self.dtype)
        
        features = self.model.get_image_features(**inputs)
        # L2 normalize features
        normalized_features = features / features.norm(dim=-1, keepdim=True)
        return normalized_features


import torch.nn as _nn  # local alias to avoid top import churn


class VideoSafetyNet(_nn.Module):
    """Video safety classification network (matches checkpoint key layout)."""

    def __init__(self, input_size: int = 1152, num_classes: int = 7) -> None:
        super().__init__()
        # Match the checkpoint structure exactly: network.layers.*
        self.network = _nn.Module()
        self.network.layers = _nn.Sequential(
            _nn.Linear(input_size, 512),
            _nn.BatchNorm1d(512),
            _nn.ReLU(),
            _nn.Linear(512, 256),
            _nn.BatchNorm1d(256),
            _nn.ReLU(),
            _nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.network.layers(x)

