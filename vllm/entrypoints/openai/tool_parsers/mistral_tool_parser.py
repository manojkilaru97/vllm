# SPDX-License-Identifier: Apache-2.0

import json
import re
from collections.abc import Sequence
from random import choices
from string import ascii_letters, digits
from typing import Union
from json import JSONDecodeError

import partial_json_parser
from partial_json_parser.core.options import Allow
from pydantic import Field

from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              DeltaFunctionCall, DeltaMessage,
                                              DeltaToolCall,
                                              ExtractedToolCallInformation,
                                              FunctionCall, ToolCall)
from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import (
    ToolParser, ToolParserManager)
from vllm.entrypoints.openai.tool_parsers.utils import (
    extract_intermediate_diff)
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import AnyTokenizer, MistralTokenizer

logger = init_logger(__name__)

ALPHANUMERIC = ascii_letters + digits


class MistralToolCall(ToolCall):
    id: str = Field(
        default_factory=lambda: MistralToolCall.generate_random_id())

    @staticmethod
    def generate_random_id():
        # Mistral Tool Call Ids must be alphanumeric with a length of 9.
        # https://github.com/mistralai/mistral-common/blob/21ee9f6cee3441e9bb1e6ed2d10173f90bd9b94b/src/mistral_common/protocol/instruct/validator.py#L299
        return "".join(choices(ALPHANUMERIC, k=9))

    @staticmethod
    def is_valid_id(id: str) -> bool:
        return id.isalnum() and len(id) == 9


@ToolParserManager.register_module("mistral")
class MistralToolParser(ToolParser):
    """
    Tool call parser for Mistral 7B Instruct v0.3, intended for use with the
    examples/tool_chat_template_mistral.jinja template.

    Used when --enable-auto-tool-choice --tool-call-parser mistral are all set
    """

    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)

        if not isinstance(self.model_tokenizer, MistralTokenizer):
            logger.info("Non-Mistral tokenizer detected when using a Mistral "
                        "model...")

        # initialize properties used for state when parsing tool calls in
        # streaming mode
        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id: int = -1
        self.current_tool_name_sent: bool = False
        self.streamed_args_for_tool: list[str] = [
        ]  # map what has been streamed for each tool so far to a list
        self.bot_token = "[TOOL_CALLS]"
        self.bot_token_id = self.vocab.get(self.bot_token)
        self.tool_call_regex = re.compile(r"\[{.*}\]", re.DOTALL)
        if isinstance(self.model_tokenizer, MistralTokenizer) and self.model_tokenizer.version >= 11:
            self.fn_name_regex = re.compile(r'([a-zA-Z0-9_-]+)(\{.*?\})', re.DOTALL)
        else:
            self.fn_name_regex = None

        if self.bot_token_id is None:
            raise RuntimeError(
                "Mistral Tool Parser could not locate the tool call token in "
                "the tokenizer!")

        self.streaming_tool_call_active = False
        self.streaming_tool_call_buffer = ""

    def adjust_request(
            self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        if not isinstance(
                self.model_tokenizer, MistralTokenizer
        ) and request.tools and request.tool_choice != 'none':
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
            return ExtractedToolCallInformation(tools_called=False,
                                                tool_calls=[],
                                                content=model_output)

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

                        function_call_arr.append({"name": fn_name, "arguments": json.loads(args)})
                else:
                    function_call_arr = json.loads(tool_content)
            except json.JSONDecodeError:
                # use a regex to find the part corresponding to the tool call.
                # NOTE: This use case should not happen if the model is trained
                # correctly. It's a easy possible fix so it's included, but
                # can be brittle for very complex / highly nested tool calls
                logger.debug("Tool content failed to parse; attempting regex fallback. Raw tool_content: %s", tool_content)
                raw_tool_call = self.tool_call_regex.findall(tool_content)[0]
                function_call_arr = json.loads(raw_tool_call)

            # Tool Call
            tool_calls: list[MistralToolCall] = [
                MistralToolCall(
                    type="function",
                    function=FunctionCall(
                        name=raw_function_call["name"],
                        # function call args are JSON but as a string
                        arguments=json.dumps(raw_function_call["arguments"],
                                             ensure_ascii=False)))
                for raw_function_call in function_call_arr
            ]

            # get any content before  the tool call
            content = model_output.split(self.bot_token)[0]
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=content if len(content) > 0 else None)

        except Exception:
            logger.exception("Error in extracting tool call from response.")
            # return information to just treat the tool call as regular JSON
            return ExtractedToolCallInformation(tools_called=False,
                                                tool_calls=[],
                                                content=tool_content)

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> Union[DeltaMessage, None]:

        # Phase 1: Determine if we are in tool call mode and update buffer.
        if not self.streaming_tool_call_active:
            if self.bot_token_id in delta_token_ids:
                # Transitioning to tool call mode
                self.streaming_tool_call_active = True
                self.prev_tool_call_arr = []
                self.current_tool_id = -1
                self.current_tool_name_sent = False
                self.streamed_args_for_tool = []
                
                # Initialize the buffer.
                # current_text includes the current delta.
                if self.bot_token in current_text:
                    # Get text after the LAST occurrence of bot_token.
                    buffer_candidate = current_text.rsplit(self.bot_token, 1)[-1]
                else:
                    # bot_token_id was in delta_token_ids, but the string [TOOL_CALLS]
                    # isn't in current_text. delta_text is the best candidate.
                    buffer_candidate = delta_text
                
                # If this initial buffer_candidate starts with bot_token, remove it.
                if buffer_candidate.startswith(self.bot_token):
                    self.streaming_tool_call_buffer = buffer_candidate[len(self.bot_token):]
                else:
                    self.streaming_tool_call_buffer = buffer_candidate
                
                logger.debug(f"Tool mode activated. Initialized buffer: '{self.streaming_tool_call_buffer}'")

            else:
                # Not in tool call mode and not transitioning.
                return DeltaMessage(content=delta_text)
        else: # self.streaming_tool_call_active is True
            # Already in tool call mode, so append delta_text.
            self.streaming_tool_call_buffer += delta_text
            logger.debug(f"Appending to buffer: '{delta_text}'. Current buffer: '{self.streaming_tool_call_buffer}'")

        # Phase 2: Process the buffer (if in tool call mode)
        # The check for active streaming mode is implicitly handled by reaching here.
        # If not active, the first block would have returned DeltaMessage.

        # Add this check:
        if not self.streaming_tool_call_buffer.strip():
            logger.debug(
                "Streaming buffer is empty or all whitespace, waiting for "
                "more tokens.")
            return None

        # Ensure the buffer looks like JSON before attempting to parse.
        stripped_buf = self.streaming_tool_call_buffer.lstrip()
        if self.fn_name_regex is None and stripped_buf and stripped_buf[0] not in "[{":
            # It might be in the `name{` format or still incomplete. Wait for more tokens.
            logger.debug(
                "Buffer does not yet start with '[' or '{' (and fn_name_regex is None). Current head: '%s'. Waiting for more tokens.",
                stripped_buf[:20])
            return None

        # The rest of the method will use self.streaming_tool_call_buffer
        # instead of current_text.split(self.bot_token)[-1]

        # bit mask flags for partial JSON parsing. If the name hasn't been
        # sent yet, don't allow sending
        # an incomplete string since OpenAI only ever (as far as I have
        # seen) allows sending the entire tool/ function name at once.
        flags = Allow.ALL if self.current_tool_name_sent \
            else Allow.ALL & ~Allow.STR
        try:
            parsable_text_for_json = self.streaming_tool_call_buffer
            # Optional: Handle single quotes if Mistral uses them and partial_json_parser needs double
            # parsable_text_for_json = self.streaming_tool_call_buffer.replace("'", "\"")

            try:
                tool_call_arr: list[dict] = partial_json_parser.loads(
                    parsable_text_for_json, flags)
            except (partial_json_parser.core.exceptions.MalformedJSON, JSONDecodeError) as e:
                logger.debug(f"partial_json_parser.loads failed: {e}. Attempting regex fallback if enabled.")
                # Attempt fallback parsing using the `fn_name_regex` pattern if available.
                if self.fn_name_regex is not None:
                    matches = self.fn_name_regex.findall(parsable_text_for_json)
                    function_call_arr: list[dict] = []
                    for match in matches:
                        fn_name, args_str = match
                        try:
                            args_json = json.loads(args_str)
                        except JSONDecodeError:
                            # Arguments JSON is incomplete; wait for more tokens.
                            logger.debug("Arguments for %s not yet complete ('%s'), waiting for more tokens.", fn_name, args_str)
                            return None
                        function_call_arr.append({"name": fn_name, "arguments": args_json})
                    if function_call_arr:
                        tool_call_arr = function_call_arr
                    else:
                        # No complete function calls parsed yet.
                        logger.debug("Regex fallback: No complete function calls parsed yet. Returning None.")
                        return None
                else:
                    logger.debug("fn_name_regex is None. Cannot use fallback. Returning None.")
                    return None

            # select as the current tool call the one we're on the state at
            # This definition of current_tool_call is for the general state after parsing.
            # Specific blocks might need to re-evaluate based on current_tool_id.

            # Storing the result of the latest parse. This will be copied to self.prev_tool_call_arr before returning or at the end.
            latest_parsed_tool_arr = list(tool_call_arr)

            # Block 1: Handling start of a new tool in the array, or transition between tools
            if len(latest_parsed_tool_arr) > 0 and len(latest_parsed_tool_arr) > self.current_tool_id + 1:
                delta_for_leaving_previous_tool = None
                if self.current_tool_id >= 0: # If there was a previous tool active
                    if self.current_tool_id < len(self.prev_tool_call_arr) and self.current_tool_id < len(latest_parsed_tool_arr):
                        # Check if the previous tool's arguments were already fully sent by a combined delta.
                        # If self.streamed_args_for_tool[self.current_tool_id] is non-empty, it implies we've sent args.
                        # We only need to flush if latest_parsed_tool_arr has *more* args for it.
                        prev_tool_complete_args_json = json.dumps(self.prev_tool_call_arr[self.current_tool_id].get("arguments", {}), ensure_ascii=False)
                        current_args_for_prev_tool_json = json.dumps(latest_parsed_tool_arr[self.current_tool_id].get("arguments", {}), ensure_ascii=False)
                        
                        # We want to find what's new in current_args_for_prev_tool_json compared to what was *last known for this tool in prev_tool_call_arr*.
                        # The extract_intermediate_diff is designed for prev_args_json to be a SUBSET.
                        # If we sent a combined call, self.streamed_args_for_tool holds the full args.
                        # Let's use extract_intermediate_diff based on what was last parsed for that tool vs current.
                        
                        argument_diff_for_flush = extract_intermediate_diff(current_args_for_prev_tool_json, prev_tool_complete_args_json)

                        if argument_diff_for_flush:
                            # Only flush if the diff is not what we already claim to have streamed.
                            # This avoids re-sending if the combined delta already sent everything.
                            # However, this check is tricky. The most straightforward is to send any diff if the structures differ.
                            logger.debug("Flushing arguments for previous tool %d: %s", self.current_tool_id, argument_diff_for_flush)
                            delta_for_leaving_previous_tool = DeltaMessage(tool_calls=[
                                DeltaToolCall(index=self.current_tool_id,
                                              # ID is not resent for just arg updates typically
                                              function=DeltaFunctionCall(arguments=argument_diff_for_flush).model_dump(exclude_none=True))
                            ])
                            self.streamed_args_for_tool[self.current_tool_id] += argument_diff_for_flush # Append delta
                        else:
                            logger.debug(f"  No new argument_diff to flush for previous tool {self.current_tool_id}.")
                    else:
                        logger.debug(f"  Skipping flush for tool {self.current_tool_id}: index out of bounds for prev_tool_call_arr (len {len(self.prev_tool_call_arr)}) or latest_parsed_tool_arr (len {len(latest_parsed_tool_arr)}).")

                # Update state for the NEW tool
                self.current_tool_id = len(latest_parsed_tool_arr) - 1
                self.current_tool_name_sent = False # Reset for the new tool
                while len(self.streamed_args_for_tool) <= self.current_tool_id:
                    self.streamed_args_for_tool.append("")
                self.streamed_args_for_tool[self.current_tool_id] = "" # Reset for new tool

                logger.debug(f"Transitioned to new tool index {self.current_tool_id}. Name sent: {self.current_tool_name_sent}. Streamed args for new tool reset.")

                if delta_for_leaving_previous_tool:
                    logger.debug(f"Returning delta for leaving/flushing previous tool: {delta_for_leaving_previous_tool}")
                    self.prev_tool_call_arr = list(latest_parsed_tool_arr) # Reflect the parse that led to this flush
                    return delta_for_leaving_previous_tool
                logger.debug("No delta for previous tool, falling through to process new tool's name/args.")

            # Block 2: Mark name as seen (no delta sending here)
            if self.current_tool_id != -1 and not self.current_tool_name_sent:
                if self.current_tool_id < len(latest_parsed_tool_arr):
                    current_tool_obj_for_name_check = latest_parsed_tool_arr[self.current_tool_id]
                    function_name_found = current_tool_obj_for_name_check.get("name")
                    if function_name_found:
                        self.current_tool_name_sent = True
                        logger.debug(f"  Name '{function_name_found}' found for tool {self.current_tool_id}. Marking current_tool_name_sent=True. No delta sent here.")
                        # DO NOT return here, fall through to Block 3 to attempt combined send.
                        # DO NOT update self.prev_tool_call_arr here yet.
                    else:
                        logger.debug(f"  Function name is None for tool {self.current_tool_id} in current parse. Waiting for more tokens for name.")
                else:
                    logger.debug(f"  current_tool_id {self.current_tool_id} is out of bounds for latest_parsed_tool_arr (len {len(latest_parsed_tool_arr)}) for name check. Waiting for more tokens.")

            # Block 3: Sending combined arguments and name for the current tool_id
            if self.current_tool_id != -1 and self.current_tool_name_sent:
                # Check if we have already sent this tool's combined information
                if not self.streamed_args_for_tool[self.current_tool_id]: # If empty, means we haven't sent args/combined for this tool yet
                    if self.current_tool_id < len(latest_parsed_tool_arr):
                        current_tool_obj = latest_parsed_tool_arr[self.current_tool_id]
                        
                        function_name = current_tool_obj.get("name")
                        arguments_obj = current_tool_obj.get("arguments")

                        # We need both name and arguments to be considered "complete" by the parser for a combined send.
                        # The fn_name_regex path ensures 'arguments' is complete JSON.
                        # partial_json_parser with Allow.STR for args might give partial strings.
                        # For this combined send, we rely on `arguments_obj` being the fully intended structure.
                        if function_name and arguments_obj is not None: # arguments_obj can be {} for no args.
                            
                            tool_id_for_delta = MistralToolCall.generate_random_id()
                            arguments_json_str = json.dumps(arguments_obj, ensure_ascii=False)
                            
                            logger.debug(f"Sending COMBINED delta for tool {self.current_tool_id}: Name='{function_name}', Args='{arguments_json_str}', ID='{tool_id_for_delta}'")
                            combined_delta = DeltaMessage(tool_calls=[
                                DeltaToolCall(
                                    index=self.current_tool_id,
                                    id=tool_id_for_delta,
                                    type="function",
                                    function=DeltaFunctionCall(name=function_name, arguments=arguments_json_str).model_dump(exclude_none=True)
                                )
                            ])
                            
                            self.streamed_args_for_tool[self.current_tool_id] = arguments_json_str # Mark full args as streamed
                            self.prev_tool_call_arr = list(latest_parsed_tool_arr)
                            return combined_delta
                        else:
                            logger.debug(f"  Name or arguments not sufficiently complete/present for tool {self.current_tool_id} to send combined delta this cycle. Name: '{function_name}', Args: '{arguments_obj}'. Waiting for more tokens.")
                    else:
                        logger.debug(f"  current_tool_id {self.current_tool_id} is out of bounds for latest_parsed_tool_arr (len {len(latest_parsed_tool_arr)}) for combined delta. Waiting for more tokens.")
                else:
                    logger.debug(f"  Arguments for tool {self.current_tool_id} already streamed ('{self.streamed_args_for_tool[self.current_tool_id]}'). Not sending combined delta again. Waiting for more tokens or new tool.")
            
            # If no specific delta was generated and returned by the blocks above
            self.prev_tool_call_arr = list(latest_parsed_tool_arr) # Update prev_tool_call_arr to current parse if nothing was sent
            logger.debug("No new tool call delta to send this cycle (after all blocks). prev_tool_call_arr updated. Returning None.")
            return None

        except Exception:
            logger.exception("Error trying to handle streaming tool call.")
            logger.debug(
                "Skipping chunk as a result of tool streaming extraction "
                "error. Buffer was: '%s'", self.streaming_tool_call_buffer) # Log buffer on error
            # Optionally, reset state if error is unrecoverable for current stream.
            # self.streaming_tool_call_active = False
            # self.streaming_tool_call_buffer = ""
            return None
