# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# code modified from deepseekv3_tool_parser.py

from collections.abc import Sequence

import regex as re

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import (
    Tool,
    ToolParser,
)

logger = init_logger(__name__)


class KimiK2ToolParser(ToolParser):
    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)
        self.current_tool_name_sent: bool = False  # Legacy, kept for compatibility
        self.tool_name_sent_arr: list[bool] = []  # Track name-sent status PER tool
        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id: int = -1
        self.streamed_args_for_tool: list[
            str
        ] = []  # map what has been streamed for each tool so far to a list

        # Section-level state management to prevent token leakage
        self.in_tool_section: bool = False
        self.token_buffer: str = ""
        # Keep only a short rolling suffix for split-marker detection.
        # Tool arguments can be arbitrarily long; buffering their raw bytes here
        # risks truncating still-open JSON and corrupting section-state tracking.
        self.buffer_max_size: int = 128
        self.section_char_count: int = 0  # Track characters processed in tool section
        # Only used as a recovery valve before a real tool call has started.
        self.max_section_chars: int = 65536
        # Track if tool calls were emitted in this request - used to suppress
        # leaked markers in subsequent content deltas after section ends
        self.tool_calls_emitted: bool = False

        # Support both singular and plural variants
        self.tool_calls_start_token: str = "<|tool_calls_section_begin|>"
        self.tool_calls_end_token: str = "<|tool_calls_section_end|>"
        self.tool_calls_start_token_variants: list[str] = [
            "<|tool_calls_section_begin|>",
            "<|tool_call_section_begin|>",  # singular variant
        ]
        self.tool_calls_end_token_variants: list[str] = [
            "<|tool_calls_section_end|>",
            "<|tool_call_section_end|>",  # singular variant
        ]

        self.tool_call_start_token: str = "<|tool_call_begin|>"
        self.tool_call_end_token: str = "<|tool_call_end|>"
        self.buffer_max_size = max(
            self.buffer_max_size,
            2 * max(
                len(marker)
                for marker in (
                    self.tool_calls_start_token_variants
                    + self.tool_calls_end_token_variants
                    + [self.tool_call_start_token, self.tool_call_end_token]
                )
            ),
        )

        self.tool_call_regex = re.compile(
            r"<\|tool_call_begin\|>\s*(?P<tool_call_id>[^<]+:\d+)\s*<\|tool_call_argument_begin\|>\s*(?P<function_arguments>(?:(?!<\|tool_call_begin\|>).)*?)\s*<\|tool_call_end\|>",
            re.DOTALL,
        )

        self.stream_tool_call_portion_regex = re.compile(
            r"(?P<tool_call_id>.+:\d+)\s*<\|tool_call_argument_begin\|>\s*(?P<function_arguments>.*)"
        )

        self.stream_tool_call_name_regex = re.compile(r"(?P<tool_call_id>.+:\d+)\s*")

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ToolParser "
                "constructor during construction."
            )
        self.tool_calls_start_token_id = self.vocab.get(self.tool_calls_start_token)
        self.tool_calls_end_token_id = self.vocab.get(self.tool_calls_end_token)

        # Get token IDs for all variants
        self.tool_calls_start_token_ids: list[int] = [
            tid
            for variant in self.tool_calls_start_token_variants
            if (tid := self.vocab.get(variant)) is not None
        ]
        self.tool_calls_end_token_ids: list[int] = [
            tid
            for variant in self.tool_calls_end_token_variants
            if (tid := self.vocab.get(variant)) is not None
        ]

        self.tool_call_start_token_id = self.vocab.get(self.tool_call_start_token)
        self.tool_call_end_token_id = self.vocab.get(self.tool_call_end_token)

        if (
            self.tool_calls_start_token_id is None
            or self.tool_calls_end_token_id is None
        ):
            raise RuntimeError(
                "Kimi-K2 Tool parser could not locate tool call start/end "
                "tokens in the tokenizer!"
            )

    def _check_and_strip_markers(self, text: str) -> tuple[str, bool, bool]:
        """
        Check for section begin/end markers in text and strip them.
        Returns: (cleaned_text, found_section_begin, found_section_end)
        """
        found_begin = False
        found_end = False
        cleaned = text

        # Check for section begin markers (any variant)
        for variant in self.tool_calls_start_token_variants:
            if variant in cleaned:
                cleaned = cleaned.replace(variant, "")
                found_begin = True

        # Check for section end markers (any variant)
        for variant in self.tool_calls_end_token_variants:
            if variant in cleaned:
                cleaned = cleaned.replace(variant, "")
                found_end = True

        return cleaned, found_begin, found_end

    def _strip_all_tool_markers(self, text: str) -> str:
        """
        Strip ALL tool-related markers from text, including:
        - Section markers: <|tool_calls_section_begin|>, <|tool_calls_section_end|>
        - Individual tool markers: <|tool_call_begin|>, <|tool_call_argument_begin|>, <|tool_call_end|>

        This prevents leaking raw markers into content when streaming ends.
        """
        cleaned = text

        # Strip section markers (all variants)
        for variant in self.tool_calls_start_token_variants:
            cleaned = cleaned.replace(variant, "")
        for variant in self.tool_calls_end_token_variants:
            cleaned = cleaned.replace(variant, "")

        # Strip individual tool call markers
        cleaned = cleaned.replace(self.tool_call_start_token, "")  # <|tool_call_begin|>
        cleaned = cleaned.replace(self.tool_call_end_token, "")    # <|tool_call_end|>
        cleaned = cleaned.replace("<|tool_call_argument_begin|>", "")

        return cleaned

    def _reset_section_state(self) -> None:
        """Reset state when exiting tool section."""
        self.in_tool_section = False
        self.token_buffer = ""
        self.section_char_count = 0
        # Note: We intentionally do NOT reset tool_calls_emitted here.
        # It needs to persist after section ends to suppress leaked markers
        # in subsequent deltas.

    def reset_streaming_state(self) -> None:
        """
        Reset all streaming state. Call this between requests to prevent
        state leakage when parser instance is reused.
        """
        # Reset section state
        self._reset_section_state()

        # Reset parent class state
        self.current_tool_name_sent = False  # Legacy
        self.tool_name_sent_arr = []  # Per-tool name-sent tracking
        self.prev_tool_call_arr = []
        self.current_tool_id = -1
        self.streamed_args_for_tool = []

        # Reset tool calls tracking (this is only reset between requests,
        # not when section ends, to catch leaked markers after section close)
        self.tool_calls_emitted = False

        logger.debug("Streaming state reset")

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        if getattr(request, "tool_choice", None) == "none":
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        # sanity check; avoid unnecessary processing
        if self.tool_calls_start_token not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        else:
            try:
                # there are two possible captures - between tags, or between a
                # tag and end-of-string so the result of
                # findall is an array of tuples where one is a function call and
                # the other is None
                function_call_tuples = self.tool_call_regex.findall(model_output)

                logger.debug("function_call_tuples: %s", function_call_tuples)

                tool_calls = []
                for match in function_call_tuples:
                    function_id, function_args = match
                    # function_id: functions.get_weather:0 or get_weather:0
                    function_name = function_id.split(":")[0].split(".")[-1]
                    tool_calls.append(
                        ToolCall(
                            id=function_id,
                            type="function",
                            function=FunctionCall(
                                name=function_name, arguments=function_args
                            ),
                        )
                    )

                content = model_output[: model_output.find(self.tool_calls_start_token)]
                return ExtractedToolCallInformation(
                    tools_called=True,
                    tool_calls=tool_calls,
                    content=content if content else None,
                )

            except Exception:
                logger.exception("Error in extracting tool call from response.")
                return ExtractedToolCallInformation(
                    tools_called=False, tool_calls=[], content=model_output
                )

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
        logger.debug("delta_text: %s", delta_text)
        logger.debug("delta_token_ids: %s", delta_token_ids)

        if getattr(request, "tool_choice", None) == "none":
            return DeltaMessage(content=delta_text) if delta_text else None

        # Flag to defer section exit until after tool parsing completes
        deferred_section_exit = False

        # Add delta to a short rolling buffer for split-marker detection.
        # We only need enough context to bridge markers across chunk boundaries.
        self.token_buffer = (self.token_buffer + delta_text)[-self.buffer_max_size :]

        # Check buffer for section markers (handles split tokens)
        buffered_text, found_section_begin, found_section_end = (
            self._check_and_strip_markers(self.token_buffer)
        )

        # Track section state transitions
        # Also enter tool section if tool_call_begin is detected directly
        # (model may output tool calls without section wrapper)
        has_direct_tool_call = self.tool_call_start_token_id in current_token_ids
        if (found_section_begin or has_direct_tool_call) and not self.in_tool_section:
            logger.debug("Entering tool section (section_begin=%s, direct_tool=%s)",
                        found_section_begin, has_direct_tool_call)
            self.in_tool_section = True
            self.token_buffer = buffered_text  # Use cleaned buffer
            self.section_char_count = 0  # Reset counter for new section
        if found_section_end and self.in_tool_section:
            logger.debug("Detected section end marker")
            # CRITICAL: Don't exit early if tool_call_end is in this chunk.
            # Tool parser must emit final arguments/close first to avoid dropping
            # the final tool update and leaking tokens into reasoning channel.
            has_tool_end = self.tool_call_end_token_id in delta_token_ids
            if has_tool_end:
                # Defer exit until after tool parsing completes
                deferred_section_exit = True
                logger.debug("Deferring section exit: tool_call_end in same chunk")
                self.token_buffer = buffered_text
            else:
                # No tool call ending, safe to exit immediately
                logger.debug("Exiting tool section")
                remaining = buffered_text
                self._reset_section_state()
                # Return remaining text as reasoning content if non-empty
                # CRITICAL: Strip ALL tool markers (not just section markers) to
                # prevent leaking raw tokens like <|tool_call_begin|> into content
                if remaining.strip():
                    cleaned_remaining = self._strip_all_tool_markers(remaining)
                    # Only return if there's actual content after stripping
                    if cleaned_remaining.strip():
                        return DeltaMessage(content=cleaned_remaining)
                # Return empty delta to maintain function contract
                # (always returns DeltaMessage)
                return DeltaMessage(content="")
        else:
            self.token_buffer = buffered_text

        # Check if any variant of section start token is in current_token_ids
        # Also check for tool_call_begin directly (model may output tool calls
        # without section wrapper)
        has_section_token = any(
            tid in current_token_ids for tid in self.tool_calls_start_token_ids
        )
        has_tool_call_token = self.tool_call_start_token_id in current_token_ids

        # Early return: if no section/tool token detected yet, return as reasoning content
        if not has_section_token and not has_tool_call_token and not self.in_tool_section:
            logger.debug("No tool call tokens found!")
            # Don't clear buffer - it needs to accumulate partial markers across deltas
            # Buffer overflow is already protected by lines 215-224

            # CRITICAL FIX: If tool calls were already emitted in this request,
            # any subsequent content containing tool markers is a leak from the
            # accumulated buffer after section ended. Strip all markers to prevent
            # raw tokens like <|tool_call_begin|> from appearing in content.
            if self.tool_calls_emitted:
                cleaned_content = self._strip_all_tool_markers(delta_text)
                # If content is now empty or just whitespace, suppress entirely
                if not cleaned_content.strip():
                    logger.debug(
                        "Suppressing leaked content after tool calls: %s",
                        delta_text[:100],
                    )
                    return DeltaMessage(content="")
                logger.debug(
                    "Stripped tool markers from leaked content: %s -> %s",
                    delta_text[:50],
                    cleaned_content[:50],
                )
                return DeltaMessage(content=cleaned_content)

            return DeltaMessage(content=delta_text)

        # Strip section markers from delta_text for subsequent processing
        # NOTE: This preprocessing happens BEFORE the regex-based tool call
        # parsing (from PR #24847) to ensure markers are removed cleanly
        # before pattern matching. No double-stripping occurs because
        # section markers and tool call markers are distinct.
        delta_text, _, _ = self._check_and_strip_markers(delta_text)

        # Error recovery: only force-exit oversized sections before a real tool
        # call has started. Once a tool call is active, large argument payloads
        # are legitimate and must not be cut off.
        if self.in_tool_section:
            self.section_char_count += len(delta_text)
            tool_call_active = (
                self.current_tool_id >= 0
                or self.tool_call_start_token_id in current_token_ids
                or self.tool_call_start_token in current_text
            )
            if self.section_char_count > self.max_section_chars and not tool_call_active:
                logger.warning(
                    "Tool section exceeded max length (%d chars), forcing exit. "
                    "This may indicate malformed model output.",
                    self.max_section_chars,
                )
                self._reset_section_state()
                # Deferred exit already handled by forced exit above
                # Return remaining content as reasoning (or empty delta if no content)
                # Strip all tool markers to prevent leaking raw tokens
                cleaned = self._strip_all_tool_markers(delta_text)
                return DeltaMessage(content=cleaned if cleaned.strip() else "")

        try:
            # figure out where we are in the parsing by counting tool call
            # start & end tags
            prev_tool_start_count = previous_token_ids.count(
                self.tool_call_start_token_id
            )
            prev_tool_end_count = previous_token_ids.count(self.tool_call_end_token_id)
            cur_tool_start_count = current_token_ids.count(
                self.tool_call_start_token_id
            )
            cur_tool_end_count = current_token_ids.count(self.tool_call_end_token_id)
            tool_call_portion = None
            text_portion = None

            # case: if we're generating text, OR rounding out a tool call
            # BUT only if all tools have been processed (don't skip if unprocessed tools remain)
            tools_processed_so_far = self.current_tool_id + 1
            if (
                cur_tool_start_count == cur_tool_end_count
                and prev_tool_end_count == cur_tool_end_count
                and self.tool_call_end_token not in delta_text
                and cur_tool_start_count <= tools_processed_so_far  # All tools processed
            ):
                # CRITICAL FIX: Suppress content if in tool section but
                # no tool calls started
                if self.in_tool_section and cur_tool_start_count == 0:
                    logger.debug(
                        "In tool section but no tool calls started yet. "
                        "Suppressing: %s",
                        delta_text,
                    )
                    # Return empty delta to maintain iterator contract
                    return DeltaMessage(content="")
                logger.debug("Generating text content! skipping tool parsing.")
                # DEFENSIVE: If tool calls were emitted, strip any leaked markers
                if self.tool_calls_emitted:
                    delta_text = self._strip_all_tool_markers(delta_text)
                    if not delta_text.strip():
                        return DeltaMessage(content="")
                return DeltaMessage(content=delta_text)

            if self.tool_call_end_token in delta_text:
                logger.debug("tool_call_end_token in delta_text")
                full_text = current_text
                tool_call_portion = (
                    full_text.split(self.tool_call_start_token)[-1]
                    .split(self.tool_call_end_token)[0]
                    .rstrip()
                )
                delta_text = delta_text.split(self.tool_call_end_token)[0].rstrip()
                text_portion = delta_text.split(self.tool_call_end_token)[-1].lstrip()

            # case -- we're starting a new tool call
            # We have a new tool call when there are more tool_call_begin tokens
            # than we've processed. This handles:
            # 1. Normal case: more starts than ends with more starts than before
            # 2. Single chunk case: complete tool call in one chunk
            # 3. Multiple tools in same chunk: process all unprocessed tools
            tools_processed = self.current_tool_id + 1  # Number of tools started
            is_new_tool_call = cur_tool_start_count > tools_processed
            if is_new_tool_call:
                # CRITICAL: Before starting a new tool, complete the previous tool's NAME and arguments
                # This handles the case where tool N wasn't fully streamed before tool N+1 started
                if self.current_tool_id >= 0:
                    prev_tool_idx = self.current_tool_id
                    # Check if previous tool's NAME was sent
                    prev_name_sent = (
                        prev_tool_idx < len(self.tool_name_sent_arr)
                        and self.tool_name_sent_arr[prev_tool_idx]
                    )

                    if not prev_name_sent:
                        # Need to send the previous tool's name first!
                        parts = current_text.split(self.tool_call_start_token)
                        part_idx = prev_tool_idx + 1  # +1 because parts[0] is prefix
                        if part_idx < len(parts):
                            prev_tool_portion = parts[part_idx]
                            if self.tool_call_end_token in prev_tool_portion:
                                prev_tool_portion = prev_tool_portion.split(self.tool_call_end_token)[0]

                            # Try to extract name and args
                            match = self.stream_tool_call_portion_regex.match(prev_tool_portion.strip())
                            if match:
                                tool_id_str, tool_args = match.groups()
                                tool_name = tool_id_str.split(":")[0].split(".")[-1]
                                tool_args = tool_args.strip() if tool_args else ""

                                logger.debug("CRITICAL: Sending missed name for tool %s: %s",
                                           prev_tool_idx, tool_name)

                                # Mark name as sent
                                while len(self.tool_name_sent_arr) <= prev_tool_idx:
                                    self.tool_name_sent_arr.append(False)
                                self.tool_name_sent_arr[prev_tool_idx] = True

                                # Update prev_tool_call_arr
                                while len(self.prev_tool_call_arr) <= prev_tool_idx:
                                    self.prev_tool_call_arr.append({})
                                self.prev_tool_call_arr[prev_tool_idx] = {
                                    "id": tool_id_str,
                                    "name": tool_name,
                                    "arguments": tool_args,
                                }

                                # Track streamed args
                                while len(self.streamed_args_for_tool) <= prev_tool_idx:
                                    self.streamed_args_for_tool.append("")
                                self.streamed_args_for_tool[prev_tool_idx] = tool_args

                                self.tool_calls_emitted = True

                                # Send the name (and args if available) for the PREVIOUS tool
                                func_delta = DeltaFunctionCall(name=tool_name)
                                if tool_args:
                                    func_delta = DeltaFunctionCall(name=tool_name, arguments=tool_args)

                                return DeltaMessage(
                                    tool_calls=[
                                        DeltaToolCall(
                                            index=prev_tool_idx,
                                            type="function",
                                            id=tool_id_str,
                                            function=func_delta.model_dump(exclude_none=True),
                                        )
                                    ]
                                )

                    # Previous tool's name was sent, now check if args need completing
                    elif prev_tool_idx < len(self.prev_tool_call_arr):
                        parts = current_text.split(self.tool_call_start_token)
                        part_idx = prev_tool_idx + 1
                        if part_idx < len(parts):
                            prev_tool_portion = parts[part_idx]
                            if self.tool_call_end_token in prev_tool_portion:
                                prev_tool_portion = prev_tool_portion.split(self.tool_call_end_token)[0]
                            match = self.stream_tool_call_portion_regex.match(prev_tool_portion.strip())
                            if match:
                                _, full_args = match.groups()
                                full_args = full_args.strip()
                                streamed_so_far = self.streamed_args_for_tool[prev_tool_idx] if prev_tool_idx < len(self.streamed_args_for_tool) else ""
                                if full_args and len(full_args) > len(streamed_so_far):
                                    if full_args.startswith(streamed_so_far):
                                        remaining_args = full_args[len(streamed_so_far):]
                                        if remaining_args:
                                            # Emit remaining args, DON'T start new tool yet
                                            logger.debug("Completing previous tool %s args before starting new tool: %s",
                                                       prev_tool_idx, remaining_args)
                                            self.streamed_args_for_tool[prev_tool_idx] = full_args
                                            self.prev_tool_call_arr[prev_tool_idx]["arguments"] = full_args
                                            self.tool_calls_emitted = True
                                            return DeltaMessage(
                                                tool_calls=[
                                                    DeltaToolCall(
                                                        index=prev_tool_idx,
                                                        function=DeltaFunctionCall(
                                                            arguments=remaining_args
                                                        ).model_dump(exclude_none=True),
                                                    )
                                                ]
                                            )

                # Now actually start the new tool
                self.current_tool_id += 1

                if len(delta_token_ids) > 1 or self.tool_call_start_token_id in current_token_ids:
                    # Split by tool_call_begin and get the portion for current tool
                    # Index is current_tool_id + 1 because split gives empty string at index 0
                    parts = current_text.split(self.tool_call_start_token)
                    tool_idx = self.current_tool_id + 1  # +1 because parts[0] is empty/prefix
                    if tool_idx < len(parts):
                        tool_call_portion = parts[tool_idx]
                    else:
                        tool_call_portion = parts[-1]  # Fallback to last
                    # Strip tool_call_end marker if present (for this tool call only)
                    if self.tool_call_end_token in tool_call_portion:
                        tool_call_portion = tool_call_portion.split(self.tool_call_end_token)[0]
                else:
                    tool_call_portion = None
                    delta = None

                text_portion = None
                self.current_tool_name_sent = False  # Legacy
                self.tool_name_sent_arr.append(False)  # Per-tool tracking
                self.streamed_args_for_tool.append("")
                # CRITICAL: Also add placeholder to prev_tool_call_arr to keep arrays in sync
                # This prevents IndexError when closing tool before name is extracted
                if len(self.prev_tool_call_arr) <= self.current_tool_id:
                    self.prev_tool_call_arr.append({})
                logger.debug("Starting on a new tool %s", self.current_tool_id)

            # case -- we're updating an existing tool call
            elif (
                cur_tool_start_count > cur_tool_end_count
                and cur_tool_start_count == prev_tool_start_count
            ):
                # get the portion of the text for the CURRENT tool call (not always the last)
                parts = current_text.split(self.tool_call_start_token)
                tool_idx = self.current_tool_id + 1  # +1 because parts[0] is empty/prefix
                if tool_idx < len(parts):
                    tool_call_portion = parts[tool_idx]
                else:
                    tool_call_portion = parts[-1]  # Fallback to last
                # Strip tool_call_end marker if present
                if self.tool_call_end_token in tool_call_portion:
                    tool_call_portion = tool_call_portion.split(self.tool_call_end_token)[0]
                text_portion = None

            # case -- the current tool call is being closed.
            # Only trigger if we actually have tool calls in progress
            elif (
                cur_tool_start_count == cur_tool_end_count
                and cur_tool_end_count >= prev_tool_end_count
                and len(self.prev_tool_call_arr) > 0  # Must have started a tool call
            ):
                # Check if current_tool_id is a valid index
                if (
                    self.prev_tool_call_arr is None
                    or len(self.prev_tool_call_arr) == 0
                    or self.current_tool_id >= len(self.prev_tool_call_arr)
                ):
                    logger.debug(
                        "attempting to close tool call, but invalid index: "
                        "current_tool_id=%d, arr_len=%d",
                        self.current_tool_id,
                        len(self.prev_tool_call_arr) if self.prev_tool_call_arr else 0,
                    )
                    # Handle deferred section exit before returning
                    if deferred_section_exit and self.in_tool_section:
                        self._reset_section_state()
                    return None

                # CRITICAL: Check if current tool's NAME was sent. If not, send it now!
                # This handles the case where the last tool closes before its name was streamed.
                tool_idx = self.current_tool_id
                name_sent_for_this_tool = (
                    tool_idx < len(self.tool_name_sent_arr)
                    and self.tool_name_sent_arr[tool_idx]
                )
                if not name_sent_for_this_tool:
                    # Extract name and args from current_text for this tool
                    parts = current_text.split(self.tool_call_start_token)
                    part_idx = tool_idx + 1  # +1 because parts[0] is prefix
                    if part_idx < len(parts):
                        tool_portion = parts[part_idx]
                        if self.tool_call_end_token in tool_portion:
                            tool_portion = tool_portion.split(self.tool_call_end_token)[0]

                        match = self.stream_tool_call_portion_regex.match(tool_portion.strip())
                        if match:
                            tool_id_str, tool_args = match.groups()
                            tool_name = tool_id_str.split(":")[0].split(".")[-1]
                            tool_args = tool_args.strip() if tool_args else ""

                            logger.debug("CRITICAL: Sending missed name for closing tool %s: %s",
                                       tool_idx, tool_name)

                            # Mark name as sent
                            while len(self.tool_name_sent_arr) <= tool_idx:
                                self.tool_name_sent_arr.append(False)
                            self.tool_name_sent_arr[tool_idx] = True

                            # Update prev_tool_call_arr
                            self.prev_tool_call_arr[tool_idx] = {
                                "id": tool_id_str,
                                "name": tool_name,
                                "arguments": tool_args,
                            }

                            # Track streamed args
                            while len(self.streamed_args_for_tool) <= tool_idx:
                                self.streamed_args_for_tool.append("")
                            self.streamed_args_for_tool[tool_idx] = tool_args

                            self.tool_calls_emitted = True

                            # Send the name (and args) for this tool
                            func_delta = DeltaFunctionCall(name=tool_name)
                            if tool_args:
                                func_delta = DeltaFunctionCall(name=tool_name, arguments=tool_args)

                            return DeltaMessage(
                                tool_calls=[
                                    DeltaToolCall(
                                        index=tool_idx,
                                        type="function",
                                        id=tool_id_str,
                                        function=func_delta.model_dump(exclude_none=True),
                                    )
                                ]
                            )

                # CRITICAL FIX: When tool closes, extract COMPLETE arguments from current_text
                # regardless of what's in delta_text. This ensures prev_tool_call_arr has
                # the full arguments for the serving layer's remaining args check.
                full_text = current_text
                tool_idx = self.current_tool_id

                # Extract complete arguments for this tool from full_text
                complete_args = None
                parts = full_text.split(self.tool_call_start_token)
                part_idx = tool_idx + 1  # +1 because parts[0] is prefix
                if part_idx < len(parts):
                    tool_portion = parts[part_idx]
                    if self.tool_call_end_token in tool_portion:
                        tool_portion = tool_portion.split(self.tool_call_end_token)[0]
                    # Extract arguments after the argument_begin marker
                    arg_marker = "<|tool_call_argument_begin|>"
                    if arg_marker in tool_portion:
                        arg_start = tool_portion.find(arg_marker)
                        arg_start += len(arg_marker)
                        complete_args = tool_portion[arg_start:].strip()

                if complete_args:
                    # Update prev_tool_call_arr with complete arguments
                    self.prev_tool_call_arr[self.current_tool_id]["arguments"] = complete_args

                    # Calculate what we haven't streamed yet
                    streamed_so_far = self.streamed_args_for_tool[self.current_tool_id] if self.current_tool_id < len(self.streamed_args_for_tool) else ""
                    if complete_args.startswith(streamed_so_far):
                        remaining_args = complete_args[len(streamed_so_far):]
                    else:
                        remaining_args = complete_args  # Fallback

                    # Update streamed args tracking
                    if self.current_tool_id < len(self.streamed_args_for_tool):
                        self.streamed_args_for_tool[self.current_tool_id] = complete_args
                    else:
                        while len(self.streamed_args_for_tool) <= self.current_tool_id:
                            self.streamed_args_for_tool.append("")
                        self.streamed_args_for_tool[self.current_tool_id] = complete_args

                    self.tool_calls_emitted = True

                    logger.debug("Tool %s closing with complete_args=%r, streamed=%r, remaining=%r",
                               tool_idx, complete_args, streamed_so_far, remaining_args)

                    # Handle deferred section exit before returning
                    if deferred_section_exit and self.in_tool_section:
                        logger.debug("Completing deferred section exit")
                        self._reset_section_state()

                    # Only return delta if there's something new to send
                    if remaining_args:
                        return DeltaMessage(
                            tool_calls=[
                                DeltaToolCall(
                                    index=self.current_tool_id,
                                    function=DeltaFunctionCall(arguments=remaining_args).model_dump(
                                        exclude_none=True
                                    ),
                                )
                            ]
                        )
                    else:
                        return None
                else:
                    # No complete args found, handle deferred exit
                    if deferred_section_exit and self.in_tool_section:
                        self._reset_section_state()
                    return None

            # case -- otherwise we're just generating text
            else:
                # Check if we're in tool section - if so, suppress
                if self.in_tool_section:
                    logger.debug("In tool section, suppressing text generation")
                    # Handle deferred section exit before returning
                    if deferred_section_exit:
                        self._reset_section_state()
                    return DeltaMessage(content="")
                # Strip ALL tool markers comprehensively
                text = self._strip_all_tool_markers(delta_text)
                delta = DeltaMessage(tool_calls=[], content=text if text.strip() else "")
                # Handle deferred section exit before returning
                if deferred_section_exit and self.in_tool_section:
                    self._reset_section_state()
                return delta

            current_tool_call = dict()
            if tool_call_portion:
                current_tool_call_matches = self.stream_tool_call_portion_regex.match(
                    tool_call_portion
                )
                if current_tool_call_matches:
                    tool_id, tool_args = current_tool_call_matches.groups()
                    tool_name = tool_id.split(":")[0].split(".")[-1]
                    current_tool_call["id"] = tool_id.strip()
                    current_tool_call["name"] = tool_name
                    current_tool_call["arguments"] = tool_args
                else:
                    current_tool_call_name_matches = (
                        self.stream_tool_call_name_regex.match(tool_call_portion)
                    )
                    if current_tool_call_name_matches:
                        (tool_id_str,) = current_tool_call_name_matches.groups()
                        tool_name = tool_id_str.split(":")[0].split(".")[-1]
                        current_tool_call["id"] = tool_id_str.strip()
                        current_tool_call["name"] = tool_name
                        current_tool_call["arguments"] = ""
                    else:
                        logger.debug("Not enough token")
                        return None
            else:
                pass  # tool_call_portion is None/empty

            # case - we haven't sent the tool name yet. If it's available, send
            #   it. otherwise, wait until it's available.
            # Use per-tool tracking to correctly handle parallel tool calls
            tool_idx = self.current_tool_id
            name_sent_for_this_tool = (
                tool_idx < len(self.tool_name_sent_arr) and self.tool_name_sent_arr[tool_idx]
            )
            if not name_sent_for_this_tool:
                if current_tool_call is None:
                    return None
                function_name: str | None = current_tool_call.get("name")
                tool_id = current_tool_call.get("id")
                tool_args = current_tool_call.get("arguments")
                if function_name:
                    self.current_tool_name_sent = True  # Legacy
                    # Update per-tool tracking
                    while len(self.tool_name_sent_arr) <= tool_idx:
                        self.tool_name_sent_arr.append(False)
                    self.tool_name_sent_arr[tool_idx] = True
                    self.tool_calls_emitted = True  # Mark that tool calls have been emitted

                    # CRITICAL: Save current_tool_call to prev_tool_call_arr before returning.
                    # Without this, if we return early here, prev_tool_call_arr won't have
                    # the tool call info needed for argument streaming in subsequent deltas.
                    if len(self.prev_tool_call_arr) <= self.current_tool_id:
                        self.prev_tool_call_arr.append(current_tool_call)
                    else:
                        self.prev_tool_call_arr[self.current_tool_id] = current_tool_call

                    # If arguments are already available (entire tool call in one chunk),
                    # include them with the name to avoid losing them
                    func_delta: DeltaFunctionCall
                    if tool_args:
                        func_delta = DeltaFunctionCall(name=function_name, arguments=tool_args)
                        # Track that we've streamed these arguments
                        if len(self.streamed_args_for_tool) > self.current_tool_id:
                            self.streamed_args_for_tool[self.current_tool_id] = tool_args
                    else:
                        func_delta = DeltaFunctionCall(name=function_name)

                    return DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=self.current_tool_id,
                                type="function",
                                id=tool_id,
                                function=func_delta.model_dump(exclude_none=True),
                            )
                        ]
                    )
                else:
                    return None
            else:
                pass  # name already sent for this tool

            # case -- otherwise, send the tool call delta

            # if the tool call portion is None, send the delta as text
            if tool_call_portion is None:
                # if there's text but not tool calls, send that -
                # otherwise None to skip chunk
                if text_portion is not None:
                    # Strip all tool markers to prevent leakage
                    cleaned = self._strip_all_tool_markers(delta_text)
                    delta = DeltaMessage(content=cleaned) if cleaned.strip() else None
                else:
                    delta = None
                return delta

            # now, the nitty-gritty of tool calls
            # now we have the portion to parse as tool call.

            logger.debug(
                "Trying to parse current tool call with ID %s", self.current_tool_id
            )

            # if we're starting a new tool call, push an empty object in as
            #   a placeholder for the arguments
            if len(self.prev_tool_call_arr) <= self.current_tool_id:
                self.prev_tool_call_arr.append({})

            # main logic for tool parsing here - compare prev. partially-parsed
            #   JSON to the current partially-parsed JSON
            prev_arguments = self.prev_tool_call_arr[self.current_tool_id].get(
                "arguments"
            )
            cur_arguments = current_tool_call.get("arguments")

            logger.debug("diffing old arguments: %s", prev_arguments)
            logger.debug("against new ones: %s", cur_arguments)

            # case -- no arguments have been created yet. skip sending a delta.
            if not cur_arguments and not prev_arguments:
                logger.debug("Skipping text %s - no arguments", delta_text)
                delta = None

            # case -- prev arguments are defined, but non are now.
            #   probably impossible, but not a fatal error - just keep going
            elif not cur_arguments and prev_arguments:
                logger.error(
                    "should be impossible to have arguments reset "
                    "mid-call. skipping streaming anything."
                )
                delta = None

            # case -- we now have the first info about arguments available from
            #   autocompleting the JSON
            elif cur_arguments and not prev_arguments:
                self.tool_calls_emitted = True  # Mark that tool calls have been emitted
                # CRITICAL: If we haven't sent the name yet, include it with the first args
                # This handles the case where name extraction succeeded but we somehow
                # reached arguments before sending the name delta
                function_name = current_tool_call.get("name")
                tool_id = current_tool_call.get("id")
                # Use per-tool tracking
                tool_idx = self.current_tool_id
                name_sent_for_tool = (
                    tool_idx < len(self.tool_name_sent_arr) and self.tool_name_sent_arr[tool_idx]
                )
                if not name_sent_for_tool and function_name:
                    # Include name with first arguments
                    self.current_tool_name_sent = True  # Legacy
                    while len(self.tool_name_sent_arr) <= tool_idx:
                        self.tool_name_sent_arr.append(False)
                    self.tool_name_sent_arr[tool_idx] = True
                    delta = DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=self.current_tool_id,
                                type="function",
                                id=tool_id,
                                function=DeltaFunctionCall(
                                    name=function_name,
                                    arguments=cur_arguments
                                ).model_dump(exclude_none=True),
                            )
                        ]
                    )
                else:
                    # Name already sent OR function_name is None
                    delta = DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=self.current_tool_id,
                                function=DeltaFunctionCall(
                                    arguments=cur_arguments
                                ).model_dump(exclude_none=True),
                            )
                        ]
                    )
                self.streamed_args_for_tool[self.current_tool_id] = cur_arguments

            # last case -- we have an update to existing arguments.
            elif cur_arguments and prev_arguments:
                if (
                    isinstance(delta_text, str)
                    and cur_arguments != prev_arguments
                    and len(cur_arguments) > len(prev_arguments)
                    and cur_arguments.startswith(prev_arguments)
                ):
                    delta_arguments = cur_arguments[len(prev_arguments) :]
                    logger.debug("got diff %s", delta_text)

                    self.tool_calls_emitted = True  # Mark that tool calls have been emitted

                    # CRITICAL: Check if name was sent for this tool. If not, include it!
                    # This handles the race condition where args start streaming before
                    # the name delta was sent (common in parallel tool calls).
                    tool_idx = self.current_tool_id
                    name_sent_for_tool = (
                        tool_idx < len(self.tool_name_sent_arr) and self.tool_name_sent_arr[tool_idx]
                    )
                    function_name = current_tool_call.get("name")
                    tool_id = current_tool_call.get("id")

                    if not name_sent_for_tool and function_name:
                        # Include name and id with this delta
                        logger.debug("CRITICAL: Including missed name %s with arg update for tool %s",
                                   function_name, tool_idx)
                        while len(self.tool_name_sent_arr) <= tool_idx:
                            self.tool_name_sent_arr.append(False)
                        self.tool_name_sent_arr[tool_idx] = True
                        self.current_tool_name_sent = True  # Legacy
                        delta = DeltaMessage(
                            tool_calls=[
                                DeltaToolCall(
                                    index=self.current_tool_id,
                                    type="function",
                                    id=tool_id,
                                    function=DeltaFunctionCall(
                                        name=function_name,
                                        arguments=delta_arguments
                                    ).model_dump(exclude_none=True),
                                )
                            ]
                        )
                    else:
                        delta = DeltaMessage(
                            tool_calls=[
                                DeltaToolCall(
                                    index=self.current_tool_id,
                                    function=DeltaFunctionCall(
                                        arguments=delta_arguments
                                    ).model_dump(exclude_none=True),
                                )
                            ]
                        )
                    self.streamed_args_for_tool[self.current_tool_id] = cur_arguments
                else:
                    delta = None

            # handle saving the state for the current tool into
            # the "prev" list for use in diffing for the next iteration
            if self.current_tool_id == len(self.prev_tool_call_arr) - 1:
                self.prev_tool_call_arr[self.current_tool_id] = current_tool_call
            else:
                self.prev_tool_call_arr.append(current_tool_call)

            # Handle deferred section exit after tool parsing completes
            if deferred_section_exit and self.in_tool_section:
                logger.debug("Completing deferred section exit")
                self._reset_section_state()

            return delta

        except Exception:
            logger.exception("Error trying to handle streaming tool call.")
            return None  # do not stream a delta. skip this token ID.
