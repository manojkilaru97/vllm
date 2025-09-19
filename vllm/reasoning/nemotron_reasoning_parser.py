# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional, Union

from transformers import PreTrainedTokenizerBase

from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              DeltaMessage)
from vllm.logger import init_logger
from vllm.reasoning import ReasoningParser, ReasoningParserManager

logger = init_logger(__name__)


@ReasoningParserManager.register_module("nemotron")
class NemotronReasoningParser(ReasoningParser):
    """
    Reasoning parser for NVIDIA Nemotron models that use <think>...</think> tags.
    
    The Nemotron model chat template appends '<think>\n' at the end of prompts
    and the model generates responses in the format:
    {reasoning_trace}</think>\n\n{final_response}
    
    This parser extracts the reasoning content from between the <think> and </think>
    tags and puts it into reasoning_content, while the final response goes to content.
    """

    start_token: str = "<think>"
    end_token: str = "</think>"

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        super().__init__(tokenizer)
        self._reasoning_ended = False  # Track if we've seen </think> in streaming
        # Buffer to hold potential tool-call prefix immediately after </think>
        # e.g., a solitary '<' that might precede a tool-call token
        self._pending_post_think_prefix: str = ""
        self._handed_off_to_tool_parser = False  # Track if we've handed off to tool parser
        # Buffer to accumulate potential </think> tag
        self._end_tag_buffer: str = ""

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ReasoningParser "
                "constructor during construction.")

    def reset_state(self):
        """Reset the reasoning state for a new request."""
        self._reasoning_ended = False
        self._pending_post_think_prefix = ""
        self._handed_off_to_tool_parser = False
        self._end_tag_buffer = ""

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        """Check if the reasoning content ends in the input_ids by looking for </think>."""
        if not input_ids:
            return self._reasoning_ended
            
        try:
            text = self.model_tokenizer.decode(input_ids, skip_special_tokens=False)
            has_end_token = self.end_token in text
            
            # Also check our internal state in case </think> was detected in streaming
            if self._reasoning_ended:
                has_end_token = True
                
            logger.debug(f"is_reasoning_end: input_ids={input_ids}, text='{text}', has_end_token={has_end_token}, _reasoning_ended={self._reasoning_ended}")
            return has_end_token
        except Exception as e:
            logger.debug(f"is_reasoning_end: decode failed with {e}")
            return self._reasoning_ended

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        """
        Extract content token ids after the </think> token.
        For Nemotron, we use text-based splitting as fallback.
        """
        return input_ids

    def extract_reasoning_content_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: Optional[Union[ChatCompletionRequest, object]] = None,
    ) -> Union[DeltaMessage, None]:
        """
        Extract reasoning content from streaming delta.
        
        Handles the Nemotron format where:
        - Model starts with reasoning content immediately (no leading <think>)
        - Reasoning ends with </think>
        - Final response follows after </think>
        """
        # DEBUG: Log every single token that enters this method
        logger.debug(f"NEMOTRON_DEBUG: *** METHOD ENTRY *** delta_text='{delta_text}' (repr: {repr(delta_text)}) previous_text='{previous_text}' current_text='{current_text}'")
        
        # FIRST: Detect no_think mode early to ensure all tokens are processed correctly
        is_no_think = False
        if request and hasattr(request, 'messages') and request.messages:
            for message in request.messages:
                content = None
                if isinstance(message, dict):
                    content = message.get('content')
                elif hasattr(message, 'content'):
                    content = message.content
                if content:
                    text = str(content)
                    if '/no_think' in text or 'detailed thinking off' in text.lower():
                        is_no_think = True
                        break
        # If we're in no_think mode, handle it immediately before any other processing
        if is_no_think:
            # Mark reasoning as ended to enable post-reasoning/tool detection path
            self._reasoning_ended = True
            
            # Initialize buffer if it doesn't exist, but don't reset it during the request
            if not hasattr(self, '_no_think_accumulated_text'):
                logger.debug(f"NEMOTRON_DEBUG: /no_think - Initializing buffer for the first time")
                self._no_think_accumulated_text = ""
            # Only reset if we're truly at the start and buffer wasn't just used
            elif not previous_text and not current_text.replace(delta_text, '') and not self._no_think_accumulated_text:
                logger.debug(f"NEMOTRON_DEBUG: /no_think - Resetting empty buffer at start of request")
                self._no_think_accumulated_text = ""

            # Add current token to our accumulated buffer
            self._no_think_accumulated_text += delta_text
            
            logger.debug(f"NEMOTRON_DEBUG: /no_think - POST-ADD accumulated_text='{self._no_think_accumulated_text}' (repr: {repr(self._no_think_accumulated_text)})")
            
            # Check if our accumulated buffer now contains the complete <TOOLCALL> pattern
            has_toolcall = "<TOOLCALL>" in self._no_think_accumulated_text
            logger.debug(f"NEMOTRON_DEBUG: /no_think - Pattern check: has_toolcall={has_toolcall}, handed_off={self._handed_off_to_tool_parser}")
            logger.debug(f"NEMOTRON_DEBUG: /no_think - Checking for '<TOOLCALL>' in '{self._no_think_accumulated_text}'")
            
            if has_toolcall and not self._handed_off_to_tool_parser:
                logger.debug(f"NEMOTRON_DEBUG: /no_think - *** <TOOLCALL> DETECTED *** in accumulated text, handing off to tool parser immediately")
                logger.debug(f"NEMOTRON_DEBUG: /no_think - Setting _handed_off_to_tool_parser=True")
                self._handed_off_to_tool_parser = True
                # Return empty DeltaMessage to signal immediate handoff to tool parser
                result = DeltaMessage()
                logger.debug(f"NEMOTRON_DEBUG: /no_think - RETURNING (handoff): {result} - serving layer should call tool parser for THIS token")
                return result
            elif self._handed_off_to_tool_parser:
                logger.debug(f"NEMOTRON_DEBUG: /no_think - Already handed off, letting tool parser handle: '{delta_text}'")
                # Hand off to tool parser for all subsequent tokens
                result = DeltaMessage()
                logger.debug(f"NEMOTRON_DEBUG: /no_think - RETURNING (handed off): {result}")
                return result
            else:
                logger.debug(f"NEMOTRON_DEBUG: /no_think - Normal content: '{delta_text}' - returning as content")
                # Normal content before tool call
                result = DeltaMessage(content=delta_text)
                logger.debug(f"NEMOTRON_DEBUG: /no_think - RETURNING (normal): {result}")
                return result
        
        # Reset state at the beginning of a new request
        # But don't reset if reasoning has already ended (to preserve buffering state)
        if not previous_text and not self._reasoning_ended:
            self.reset_state()
            
        # If no content, nothing to stream
        if not delta_text:
            return None

        # If we've already handed off to tool parser, return empty DeltaMessage to signal tool parser should handle this
        if self._handed_off_to_tool_parser:
            logger.debug(f"NEMOTRON_DEBUG: Already handed off to tool parser, signaling tool parser should handle: '{delta_text}'")
            # Return empty DeltaMessage to signal tool parser should be called
            return DeltaMessage()

        # Check if we've already passed the </think> boundary
        prev_has_end = self.end_token in previous_text
        curr_end_index = current_text.find(self.end_token)

        # Case 1: Already past </think> → everything is final content
        # Use _reasoning_ended state since previous_text might be reset
        if prev_has_end or self._reasoning_ended:
            logger.debug(f"NEMOTRON_DEBUG: Case 1 - Already past </think>")
            self._reasoning_ended = True  # Mark that reasoning has ended

            # Handle potential tool-call prefix buffering, specifically for
            # a '<' immediately after </think> that may introduce a tool token
            combined = f"{self._pending_post_think_prefix}{delta_text}"
            trimmed = combined.lstrip()
            logger.debug(f"NEMOTRON_DEBUG: Case 1 - combined='{combined}' (repr: {repr(combined)}), trimmed='{trimmed}' (repr: {repr(trimmed)})")

            # If we only have a single '<' (or a run of whitespace + '<'), keep holding
            if trimmed == "<":
                logger.debug(f"NEMOTRON_DEBUG: Case 1 - Holding single '<', setting _pending_post_think_prefix='{combined}'")
                self._pending_post_think_prefix = combined
                return None


            # Handle potential special tokens more carefully
            stripped_delta = delta_text.strip()
            logger.debug(f"NEMOTRON_DEBUG: Case 1 - stripped_delta='{stripped_delta}' (repr: {repr(stripped_delta)})")
            
            # If we see '<', start buffering (could be </think> or <TOOLCALL>)
            if stripped_delta == '<':
                logger.debug(f"NEMOTRON_DEBUG: Case 1 - Detected single '<', buffering")
                self._pending_post_think_prefix += delta_text
                return None
            
            # If we have a buffer starting with '<', keep accumulating
            elif self._pending_post_think_prefix.strip().startswith('<'):
                logger.debug(f"NEMOTRON_DEBUG: Case 1 - Buffer starts with '<', accumulating")
                self._pending_post_think_prefix += delta_text
                
                # Check what we've accumulated so far
                accumulated = self._pending_post_think_prefix.strip()
                logger.debug(f"NEMOTRON_DEBUG: Case 1 - Accumulated: '{accumulated}'")
                
                # Check if it's a complete </think> tag
                if accumulated == '</think>':
                    logger.debug(f"NEMOTRON_DEBUG: Case 1 - Found complete </think>, should not happen in Case 1")
                    # Ignore and reset buffer
                    self._pending_post_think_prefix = ''
                    return None
                
                # If it's a start of a TOOLCALL, return empty DeltaMessage to allow tool parser
                if accumulated.startswith('<TOOLCALL>'):
                    logger.debug(f"NEMOTRON_DEBUG: Case 1 - Detected <TOOLCALL> start, handoff to tool parser next")
                    self._handed_off_to_tool_parser = True
                    return DeltaMessage()
                
            # Otherwise, this is normal content after reasoning
            return DeltaMessage(content=delta_text)

        # Case 2: Still within reasoning → stream reasoning deltas
        if curr_end_index == -1:
            logger.debug(f"NEMOTRON_DEBUG: Case 2 - Still in reasoning, streaming reasoning delta")
            return DeltaMessage(reasoning_content=delta_text)
        
        # Case 3: Boundary appears within this delta → split and emit
        logger.debug(f"NEMOTRON_DEBUG: Case 3 - Found </think> boundary within delta")
        self._reasoning_ended = True
        end_index = curr_end_index + len(self.end_token)
        before = current_text[:end_index]
        after = current_text[end_index:]

        # The delta_text corresponds to the tail part of current_text; compute the
        # portion that is post-</think> to emit as content
        if delta_text:
            # best-effort split: if delta contains the end token, take the post-part
            if self.end_token in delta_text:
                post = delta_text.split(self.end_token, 1)[1]
            else:
                # otherwise, if boundary was earlier in the stream, don't emit
                post = ''
        else:
            post = ''

        return DeltaMessage(reasoning_content=None if not before else None,
                            content=post if post else None)

    def extract_reasoning_content(
        self,
        model_output: str,
        request: Union[ChatCompletionRequest, object],
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Extract reasoning content from complete model output.
        
        For Nemotron format: {reasoning_trace}</think>\n\n{final_response}
        Special case: When /no_think is used, model generates direct content after <think></think>
        Returns: (reasoning_content, final_content)
        """
        # Check if this is a no-thinking scenario by looking at the request
        is_no_think = False
        if hasattr(request, 'messages') and request.messages:
            for message in request.messages:
                # Handle different message formats (dict-like or object-like)
                content = None
                if isinstance(message, dict):
                    content = message.get('content')
                elif hasattr(message, 'content'):
                    content = message.content
                
                if content and '/no_think' in str(content):
                    is_no_think = True
                    break
        
        # Remove leading <think> if present (from prompt)
        if model_output.startswith(self.start_token):
            model_output = model_output[len(self.start_token):]

        # Split at the first </think>
        if self.end_token in model_output:
            reasoning_content, _, content = model_output.partition(self.end_token)
            
            # Clean up the final content (remove leading whitespace/newlines)
            if content:
                content = content.strip()
            
            # Clean up reasoning content (remove any trailing/leading whitespace)
            if reasoning_content:
                reasoning_content = reasoning_content.strip()
                
            return (reasoning_content or None), (content or None)

        # No closing tag found: check if this is no-think mode
        output_stripped = model_output.strip()
        
        # If output is empty, return None for both
        if not output_stripped:
            return None, None
        
        # Check if this is no-think mode by looking at the request
        is_no_think = False
        if hasattr(request, 'messages') and request.messages:
            for message in request.messages:
                content = None
                if isinstance(message, dict):
                    content = message.get('content')
                elif hasattr(message, 'content'):
                    content = message.content
                
                if content and '/no_think' in str(content):
                    is_no_think = True
                    break
        
        if is_no_think:
            # In no-think mode, treat everything as direct content
            return None, output_stripped
        else:
            # Default: treat as reasoning content (no </think> found, so incomplete reasoning)
            return output_stripped, None
