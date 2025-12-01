# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Sequence
from random import choices
from string import ascii_letters, digits

import partial_json_parser
import regex as re
from partial_json_parser.core.options import Allow
from pydantic import Field

from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import (
    ToolParser,
)
from vllm.entrypoints.openai.tool_parsers.utils import extract_intermediate_diff
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import AnyTokenizer, MistralTokenizer

logger = init_logger(__name__)

ALPHANUMERIC = ascii_letters + digits


class MistralToolCall(ToolCall):
    id: str = Field(default_factory=lambda: MistralToolCall.generate_random_id())

    @staticmethod
    def generate_random_id():
        # Mistral Tool Call Ids must be alphanumeric with a length of 9.
        # https://github.com/mistralai/mistral-common/blob/21ee9f6cee3441e9bb1e6ed2d10173f90bd9b94b/src/mistral_common/protocol/instruct/validator.py#L299
        return "".join(choices(ALPHANUMERIC, k=9))

    @staticmethod
    def is_valid_id(id: str) -> bool:
        return id.isalnum() and len(id) == 9


def _is_fn_name_regex_support(model_tokenizer: AnyTokenizer) -> bool:
    return (
        isinstance(model_tokenizer, MistralTokenizer) and model_tokenizer.version >= 11
    )


class MistralToolParser(ToolParser):
    """
    Tool call parser for Mistral 7B Instruct v0.3, intended for use with
    - [`mistral_common`](https://github.com/mistralai/mistral-common/)
    - the examples/tool_chat_template_mistral.jinja template.

    Used when --enable-auto-tool-choice --tool-call-parser mistral are all set
    """

    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)

        if not isinstance(self.model_tokenizer, MistralTokenizer):
            logger.info("Non-Mistral tokenizer detected when using a Mistral model...")

        # initialize properties used for state when parsing tool calls in
        # streaming mode
        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id: int = -1
        self.current_tool_name_sent: bool = False
        self.streamed_args_for_tool: list[
            str
        ] = []  # map what has been streamed for each tool so far to a list
        self.bot_token = "[TOOL_CALLS]"
        self.bot_token_id = self.vocab.get(self.bot_token)
        self.tool_call_regex = re.compile(r"\[{.*}\]", re.DOTALL)
        if _is_fn_name_regex_support(self.model_tokenizer):
            self.fn_name_regex = re.compile(
                r"([a-zA-Z0-9_-]+)(\{[\s\S]*?\})(?=\s*$|,|\s)", re.DOTALL
            )
        else:
            self.fn_name_regex = None

        if self.bot_token_id is None:
            raise RuntimeError(
                "Mistral Tool Parser could not locate the tool call token in "
                "the tokenizer!"
            )

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        # NOTE: We intentionally do NOT call super().adjust_request(request) here.
        # The base class sets up structured output (JSON schema grammar) for tool
        # calling when tool_choice is "required" or a named function. However,
        # Mistral models use their own [TOOL_CALLS] token format for tool calling
        # and do NOT use structured output/grammar-guided generation.
        #
        # Calling super() would set request.structured_outputs.json which triggers
        # xgrammar/structured output initialization. This causes an engine crash
        # (AssertionError: struct_output_request.grammar is not None) because:
        # 1. The grammar is created for JSON schema output format
        # 2. But Mistral outputs in [TOOL_CALLS]fn_name{args} format
        # 3. The grammar never properly initializes/matches, leading to None grammar
        #
        # By skipping the base class, we let Mistral handle tool calls natively
        # through its tokenizer and the extract_tool_calls/streaming methods.
        
        if (
            not isinstance(self.model_tokenizer, MistralTokenizer)
            and request.tools
            and request.tool_choice != "none"
        ):
            # Do not skip special tokens when using chat template
            # with Mistral parser as TOOL_CALL token is needed
            # for tool detection.
            # Note: we don't want skip_special_tokens=False
            # with MistralTokenizer as it is incompatible
            request.skip_special_tokens = False
        return request

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """
        Extract the tool calls from a complete model response. Requires
        find-and-replacing single quotes with double quotes for JSON parsing,
        make sure your tool call arguments don't ever include quotes!
        """

        # case -- if a tool call token is not present, return a text response
        if self.bot_token not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        # first remove the BOT token
        tool_content = model_output.replace(self.bot_token, "").strip()

        try:
            # we first try to directly load the json as parsing very nested
            # jsons is difficult
            try:
                if self.fn_name_regex:
                    matches = self.fn_name_regex.findall(tool_content)

                    function_call_arr = []
                    for match in matches:
                        fn_name = match[0]
                        args = match[1]

                        # fn_name is encoded outside serialized json dump
                        # only arguments are serialized
                        function_call_arr.append(
                            {"name": fn_name, "arguments": json.loads(args)}
                        )
                else:
                    function_call_arr = json.loads(tool_content)
            except json.JSONDecodeError:
                # use a regex to find the part corresponding to the tool call.
                # NOTE: This use case should not happen if the model is trained
                # correctly. It's an easy possible fix so it's included, but
                # can be brittle for very complex / highly nested tool calls
                raw_tool_call = self.tool_call_regex.findall(tool_content)[0]
                function_call_arr = json.loads(raw_tool_call)

            # Tool Call
            tool_calls: list[MistralToolCall] = [
                MistralToolCall(
                    type="function",
                    function=FunctionCall(
                        name=raw_function_call["name"],
                        # function call args are JSON but as a string
                        arguments=json.dumps(
                            raw_function_call["arguments"], ensure_ascii=False
                        ),
                    ),
                )
                for raw_function_call in function_call_arr
            ]

            # get any content before  the tool call
            content = model_output.split(self.bot_token)[0]
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=content if len(content) > 0 else None,
            )

        except Exception:
            logger.exception("Error in extracting tool call from response.")
            # return information to just treat the tool call as regular JSON
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=tool_content
            )

    def _parse_fn_name_format_streaming(
        self, parsable_arr: str
    ) -> list[dict] | None:
        """
        Parse tool calls in the fn_name{args} format used by newer Mistral
        models (version >= 11). Returns a list of dicts with 'name' and
        'arguments' keys, or None if no valid tool calls found yet.

        Example input: 'get_weather{"location": "Paris", "unit": "c"}'
        Example output: [{"name": "get_weather", "arguments": {"location": "Paris", "unit": "c"}}]
        """
        if not self.fn_name_regex:
            return None

        # For streaming, we need a more lenient regex that can match
        # partial JSON. First try the strict regex for complete matches.
        matches = self.fn_name_regex.findall(parsable_arr)
        if matches:
            result = []
            for match in matches:
                fn_name = match[0]
                args_str = match[1]
                try:
                    args = json.loads(args_str)
                    result.append({"name": fn_name, "arguments": args})
                except json.JSONDecodeError:
                    # Args not complete yet, try partial parsing
                    try:
                        args = partial_json_parser.loads(args_str, Allow.ALL)
                        result.append({"name": fn_name, "arguments": args})
                    except Exception:
                        # If we have a name but can't parse args yet,
                        # return with empty arguments
                        result.append({"name": fn_name, "arguments": {}})
            return result if result else None

        # Try to match just the function name with partial/incomplete JSON
        # Pattern: function_name followed by { and possibly incomplete JSON
        partial_fn_regex = re.compile(r"([a-zA-Z0-9_-]+)(\{.*)", re.DOTALL)
        partial_match = partial_fn_regex.match(parsable_arr.strip())
        if partial_match:
            fn_name = partial_match.group(1)
            args_str = partial_match.group(2)
            try:
                # Try to parse with partial JSON parser
                args = partial_json_parser.loads(args_str, Allow.ALL)
                return [{"name": fn_name, "arguments": args}]
            except Exception:
                # We have the function name but args aren't parseable yet
                # Return with just the name so we can stream it
                return [{"name": fn_name, "arguments": {}}]

        # Check if we just have a function name starting (no { yet)
        name_only_regex = re.compile(r"^([a-zA-Z0-9_-]+)$")
        name_match = name_only_regex.match(parsable_arr.strip())
        if name_match:
            # We might be in the middle of receiving the function name
            # Don't return anything yet until we see the {
            return None

        return None

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        # if the tool call token is not in the tokens generated so far, append
        # output to contents since it's not a tool
        if self.bot_token not in current_text:
            return DeltaMessage(content=delta_text)

        # if the tool call token ID IS in the tokens generated so far, that
        # means we're parsing as tool calls now

        # handle if we detected the BOT token which means the start of tool
        # calling
        if self.bot_token_id in delta_token_ids and len(delta_token_ids) == 1:
            # if it's the only token, return None, so we don't send a chat
            # completion any don't send a control token
            return None

        # bit mask flags for partial JSON parsing. If the name hasn't been
        # sent yet, don't allow sending
        # an incomplete string since OpenAI only ever (as far as I have
        # seen) allows sending the entire tool/ function name at once.
        flags = Allow.ALL if self.current_tool_name_sent else Allow.ALL & ~Allow.STR
        try:
            # replace BOT token with empty string, and convert single quotes
            # to double to allow parsing as JSON since mistral uses single
            # quotes instead of double for tool calls
            parsable_arr = current_text.split(self.bot_token)[-1]

            # If there is nothing after the BOT token yet, we don't have
            # any JSON to parse. Calling the partial JSON parser on an empty
            # string can raise a JSONDecodeError, so just wait for more
            # tokens instead.
            if not parsable_arr.strip():
                logger.debug("no tool-call payload after BOT token yet")
                return None

            # First, try to parse using fn_name{args} format for newer models
            tool_call_arr: list[dict] | None = None
            if self.fn_name_regex:
                tool_call_arr = self._parse_fn_name_format_streaming(parsable_arr)

            # If fn_name format didn't work, try the legacy JSON array format
            if tool_call_arr is None:
                # tool calls are generated in an array, so do partial JSON
                # parsing on the entire array
                try:
                    tool_call_arr = partial_json_parser.loads(parsable_arr, flags)
                except partial_json_parser.core.exceptions.MalformedJSON:
                    logger.debug("not enough tokens to parse into JSON yet")
                    return None
                except json.JSONDecodeError:
                    # partial_json_parser may delegate to the stdlib JSON parser;
                    # if it fails here we also just wait for more tokens.
                    logger.debug("JSON decode error while parsing partial tool call")
                    return None

            # Ensure tool_call_arr is a list
            if not isinstance(tool_call_arr, list):
                logger.debug("tool_call_arr is not a list, waiting for more tokens")
                return None

            # select as the current tool call the one we're on the state at

            current_tool_call: dict = (
                tool_call_arr[self.current_tool_id] if len(tool_call_arr) > 0 else {}
            )

            # case -- if no tokens have been streamed for the tool, e.g.
            #   only the array brackets, stream nothing
            if len(tool_call_arr) == 0:
                return None

            # case: we are starting a new tool in the array
            #   -> array has > 0 length AND length has moved past cursor
            elif (
                len(tool_call_arr) > 0 and len(tool_call_arr) > self.current_tool_id + 1
            ):
                # if we're moving on to a new call, first make sure we
                # haven't missed anything in the previous one that was
                # auto-generated due to JSON completions, but wasn't
                # streamed to the client yet.
                if self.current_tool_id >= 0:
                    diff: str | None = current_tool_call.get("arguments")

                    if diff:
                        diff = json.dumps(diff, ensure_ascii=False).replace(
                            self.streamed_args_for_tool[self.current_tool_id], ""
                        )
                        delta = DeltaMessage(
                            tool_calls=[
                                DeltaToolCall(
                                    index=self.current_tool_id,
                                    function=DeltaFunctionCall(
                                        arguments=diff
                                    ).model_dump(exclude_none=True),
                                )
                            ]
                        )
                        self.streamed_args_for_tool[self.current_tool_id] += diff
                    else:
                        delta = None
                else:
                    delta = None
                # re-set stuff pertaining to progress in the current tool
                self.current_tool_id = len(tool_call_arr) - 1
                self.current_tool_name_sent = False
                self.streamed_args_for_tool.append("")
                logger.debug("starting on new tool %d", self.current_tool_id)
                return delta

            # case: update an existing tool - this is handled below

            # if the current tool name hasn't been sent, send if available
            # - otherwise send nothing
            if not self.current_tool_name_sent:
                function_name = current_tool_call.get("name")
                if function_name:
                    delta = DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=self.current_tool_id,
                                type="function",
                                id=MistralToolCall.generate_random_id(),
                                function=DeltaFunctionCall(
                                    name=function_name
                                ).model_dump(exclude_none=True),
                            )
                        ]
                    )
                    self.current_tool_name_sent = True
                else:
                    delta = None

            # now we know we're on the same tool call and we're streaming
            # arguments
            else:
                prev_arguments = self.prev_tool_call_arr[self.current_tool_id].get(
                    "arguments"
                )
                cur_arguments = current_tool_call.get("arguments")

                new_text = delta_text.replace("'", '"')
                if '"}' in new_text:
                    new_text = new_text[: new_text.rindex('"}')]

                if not cur_arguments and not prev_arguments:
                    delta = None
                elif not cur_arguments and prev_arguments:
                    logger.error(
                        "INVARIANT - impossible to have arguments reset mid-arguments"
                    )
                    delta = None
                elif cur_arguments and not prev_arguments:
                    cur_arguments_json = json.dumps(cur_arguments, ensure_ascii=False)
                    # Handle case where arguments is empty dict {}
                    if cur_arguments_json == "{}":
                        delta = None
                    else:
                        logger.debug("finding %s in %s", new_text, cur_arguments_json)

                        if new_text and new_text not in cur_arguments_json:
                            # Just send what we have so far
                            arguments_delta = cur_arguments_json
                        else:
                            arguments_delta = cur_arguments_json[
                                : cur_arguments_json.rindex(new_text) + len(new_text)
                            ] if new_text else cur_arguments_json
                        logger.debug(
                            "First tokens in arguments received: %s", arguments_delta
                        )
                        # Only send a delta if there's actually new content
                        if arguments_delta:
                            delta = DeltaMessage(
                                tool_calls=[
                                    DeltaToolCall(
                                        index=self.current_tool_id,
                                        function=DeltaFunctionCall(
                                            arguments=arguments_delta
                                        ).model_dump(exclude_none=True),
                                    )
                                ]
                            )
                            self.streamed_args_for_tool[self.current_tool_id] += arguments_delta
                        else:
                            delta = None

                elif cur_arguments and prev_arguments:
                    cur_args_json = json.dumps(cur_arguments, ensure_ascii=False)
                    prev_args_json = json.dumps(prev_arguments, ensure_ascii=False)
                    logger.debug(
                        "Searching for diff between \n%s\n%s",
                        cur_args_json,
                        prev_args_json,
                    )

                    argument_diff = extract_intermediate_diff(
                        cur_args_json, prev_args_json
                    )
                    logger.debug("got arguments diff: %s", argument_diff)
                    # Only send a delta if there's actually new content
                    if argument_diff:
                        delta = DeltaMessage(
                            tool_calls=[
                                DeltaToolCall(
                                    index=self.current_tool_id,
                                    function=DeltaFunctionCall(
                                        arguments=argument_diff
                                    ).model_dump(exclude_none=True),
                                )
                            ]
                        )
                        self.streamed_args_for_tool[self.current_tool_id] += argument_diff
                    else:
                        delta = None
                else:
                    # try parsing it with regular JSON - if it works we're
                    # at the end, and we need to send the difference between
                    # tokens streamed so far and the valid JSON
                    delta = None

            # check to see if the name is defined and has been sent. if so,
            # stream the name - otherwise keep waiting
            # finish by setting old and returning None as base case
            self.prev_tool_call_arr = tool_call_arr
            return delta

        except Exception:
            logger.exception("Error trying to handle streaming tool call.")
            logger.debug(
                "Skipping chunk as a result of tool streaming extraction error"
            )
            return None
