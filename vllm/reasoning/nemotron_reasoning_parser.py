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
                if content and '/no_think' in str(content):
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
                    # This shouldn't happen in Case 1, but if it does, treat as content
                    content = self._pending_post_think_prefix
                    self._pending_post_think_prefix = ""
                    return DeltaMessage(content=content)
                
                # Check if it's building toward </think>
                elif '</think>'.startswith(accumulated):
                    logger.debug(f"NEMOTRON_DEBUG: Case 1 - Building toward </think>, keep buffering")
                    return None
                
                # Check if it's a complete <TOOLCALL> (hand off to tool parser)
                elif accumulated.startswith('<TOOLCALL'):
                    logger.debug(f"NEMOTRON_DEBUG: Case 1 - Found TOOLCALL, handing off to tool parser")
                    self._pending_post_think_prefix = ""
                    self._handed_off_to_tool_parser = True
                    return DeltaMessage()
                
                # Check if it's building toward <TOOLCALL>
                elif '<TOOLCALL>'.startswith(accumulated):
                    logger.debug(f"NEMOTRON_DEBUG: Case 1 - Building toward TOOLCALL, keep buffering")
                    return None
                
                # If it's clearly not a special token (e.g., <html>, <div>), flush it
                else:
                    logger.debug(f"NEMOTRON_DEBUG: Case 1 - Not a special token, flushing: '{accumulated[:20]}...'")
                    content = self._pending_post_think_prefix
                    self._pending_post_think_prefix = ""
                    return DeltaMessage(content=content)
            
            # If we have other buffered content, flush it with current delta
            elif self._pending_post_think_prefix:
                logger.debug(f"NEMOTRON_DEBUG: Case 1 - Flushing buffered content with current delta")
                content = f"{self._pending_post_think_prefix}{delta_text}"
                self._pending_post_think_prefix = ""
                return DeltaMessage(content=content)

            # If there is pending buffered text and we reached here, flush it with current
            if self._pending_post_think_prefix:
                out = f"{self._pending_post_think_prefix}{delta_text}"
                logger.debug(f"NEMOTRON_DEBUG: Case 1 - Flushing pending text: '{out}' (repr: {repr(out)})")
                self._pending_post_think_prefix = ""
                return DeltaMessage(content=out)

            logger.debug(f"NEMOTRON_DEBUG: Case 1 - Returning as content: '{delta_text}' (repr: {repr(delta_text)})")
            return DeltaMessage(content=delta_text)

        # Case 2: </think> appears in this chunk
        if curr_end_index != -1:
            logger.debug(f"NEMOTRON_DEBUG: Case 2 - </think> appears in current chunk at index {curr_end_index}")
            # Get reasoning content up to and including the end tag
            reasoning_upto_and_including_end = current_text[:curr_end_index + len(self.end_token)]

            # Calculate the delta for reasoning by removing previous text
            if reasoning_upto_and_including_end.startswith(previous_text):
                reasoning_delta = reasoning_upto_and_including_end[len(previous_text):]
            else:
                # Fallback: find end tag in delta_text and include it
                local_end_idx = delta_text.find(self.end_token)
                if local_end_idx != -1:
                    reasoning_delta = delta_text[:local_end_idx + len(self.end_token)]
                else:
                    reasoning_delta = delta_text

            # Content after </think>
            content_after_end = current_text[curr_end_index + len(self.end_token):]
            logger.debug(f"NEMOTRON_DEBUG: Case 2 - content_after_end='{content_after_end}' (repr: {repr(content_after_end)})")
            if content_after_end:
                content_after_end = content_after_end.strip()
                logger.debug(f"NEMOTRON_DEBUG: Case 2 - content_after_end (stripped)='{content_after_end}' (repr: {repr(content_after_end)})")

            # Clean up any literal start tag strings in reasoning
            if self.start_token in reasoning_delta:
                reasoning_delta = reasoning_delta.replace(self.start_token, "")
            
            # Filter out partial end tag tokens (like '</', 'think', '>')
            end_tag_parts = ['</', 'think', '>', '</think>']
            for part in end_tag_parts:
                if part in reasoning_delta:
                    reasoning_delta = reasoning_delta.replace(part, "")

            # Mark that reasoning has ended since we found </think>
            self._reasoning_ended = True

            # If content after end appears to begin a tool call with '<', hold it
            if content_after_end and content_after_end.startswith('<'):
                # Determine if the sequence after '<' looks like a tool call
                lookahead = content_after_end
                # Build the immediate word after '<'
                idx = 1
                while (idx < len(lookahead) and lookahead[idx].isalpha()) or (
                    idx < len(lookahead) and lookahead[idx] in ['_', '-']
                ):
                    idx += 1
                following_word = lookahead[1:idx].upper()

                if following_word.startswith('TOOL'):
                    # hold the '<' (and any already-emitted text will be handled
                    # by the tool parser); do not stream content here
                    self._pending_post_think_prefix = '<'
                    return DeltaMessage(
                        reasoning_content=reasoning_delta or None,
                        content=None,
                    )
                else:
                    # Not a tool call; treat the entire sequence as final content
                    self._pending_post_think_prefix = ''
                    return DeltaMessage(
                        reasoning_content=reasoning_delta or None,
                        content=content_after_end or None,
                    )

            return DeltaMessage(
                reasoning_content=reasoning_delta or None,
                content=content_after_end or None,
            )

        # Case 3: Still inside reasoning (no </think> found yet)
        # But we need to check if reasoning has already ended from previous chunks
        if self._reasoning_ended:
            logger.debug(f"NEMOTRON_DEBUG: Case 3 - Reasoning already ended, treating as content")
            # We're already past the reasoning phase, treat as content
            # Handle potential tool-call prefix buffering
            combined = f"{self._pending_post_think_prefix}{delta_text}"
            trimmed = combined.lstrip()
            logger.debug(f"NEMOTRON_DEBUG: Case 3 - combined='{combined}' (repr: {repr(combined)}), trimmed='{trimmed}' (repr: {repr(trimmed)})")

            # If we only have a single '<' (or a run of whitespace + '<'), keep holding
            if trimmed == "<":
                logger.debug(f"NEMOTRON_DEBUG: Case 3 - Holding single '<', setting _pending_post_think_prefix='{combined}'")
                self._pending_post_think_prefix = combined
                return None


            # Handle potential special tokens more carefully
            # Only buffer single '<' or specific patterns that could be special tokens
            stripped_delta = delta_text.strip()
            
            # Only buffer if it's exactly '<' or could be start of </think> tag
            if stripped_delta == '<':
                logger.debug(f"NEMOTRON_DEBUG: Case 3 - Detected single '<', buffering")
                self._pending_post_think_prefix += delta_text
                return None
            elif stripped_delta.startswith('</'):
                # Could be start of </think> - keep buffering for now
                logger.debug(f"NEMOTRON_DEBUG: Case 3 - Detected '</' pattern, buffering")
                self._pending_post_think_prefix += delta_text
                return None
            elif stripped_delta.startswith('<') and len(stripped_delta) > 1:
                # In Case 3 (still reasoning), we shouldn't see tool calls
                # Any '<' content here is part of reasoning content
                logger.debug(f"NEMOTRON_DEBUG: Case 3 - Content starting with '<' during reasoning, flushing as reasoning_content")
                combined_content = f"{self._pending_post_think_prefix}{delta_text}"
                self._pending_post_think_prefix = ""
                return DeltaMessage(reasoning_content=combined_content)
            elif self._pending_post_think_prefix:
                # In Case 3 (still reasoning), we only care about </think> detection
                # No tool calls should appear during reasoning phase
                buffered = self._pending_post_think_prefix.strip()
                combined_trimmed = (buffered + delta_text.strip()).strip()
                
                # Only check for </think> patterns during reasoning
                could_be_think_close = (
                    combined_trimmed.startswith('</') and len(combined_trimmed) <= 8 and
                    ('think'.startswith(combined_trimmed[2:].lower()) if len(combined_trimmed) > 2 else True)
                )
                
                # If it could be </think>, keep buffering
                if could_be_think_close:
                    logger.debug(f"NEMOTRON_DEBUG: Case 3 - Potential </think> pattern, keep buffering")
                    return None
                
                # Otherwise, it's reasoning content - flush it
                else:
                    logger.debug(f"NEMOTRON_DEBUG: Case 3 - Reasoning content, flushing: '{combined_trimmed[:30]}...'")
                    combined_content = f"{self._pending_post_think_prefix}{delta_text}"
                    self._pending_post_think_prefix = ""
                    return DeltaMessage(reasoning_content=combined_content)
            elif stripped_delta in ['>']:
                # These could be part of structured content, but not special tokens
                self._pending_post_think_prefix += delta_text
                return None

            # If there is pending buffered text and we reached here, flush it with current
            if self._pending_post_think_prefix:
                out = f"{self._pending_post_think_prefix}{delta_text}"
                logger.debug(f"NEMOTRON_DEBUG: Case 3 - Flushing pending text: '{out}' (repr: {repr(out)})")
                self._pending_post_think_prefix = ""
                return DeltaMessage(content=out)

            logger.debug(f"NEMOTRON_DEBUG: Case 3 - Returning as reasoning_content: '{delta_text}' (repr: {repr(delta_text)})")
            return DeltaMessage(reasoning_content=delta_text)
        
        # Handle end tag accumulation
        combined_buffer = self._end_tag_buffer + delta_text
        
        # Check if we're starting to accumulate the end tag
        if not self._end_tag_buffer and delta_text.startswith("</"):
            # Check if this could be the start of </think>
            # If delta_text is exactly "</" or starts with "</t" or "</th" etc., buffer it
            if (delta_text == "</" or 
                (len(delta_text) > 2 and "think".startswith(delta_text[2:].lower()))):
                logger.debug(f"NEMOTRON_DEBUG: Starting end tag accumulation with: '{delta_text}'")
                self._end_tag_buffer = delta_text
                return None  # Don't stream yet, accumulate more
            else:
                # This is not a </think> tag (e.g., </html>, </div>), stream it immediately
                logger.debug(f"NEMOTRON_DEBUG: Not a </think> tag, streaming immediately: '{delta_text}'")
                return DeltaMessage(reasoning_content=delta_text)
        
        # If we're already accumulating, continue
        if self._end_tag_buffer:
            self._end_tag_buffer += delta_text
            logger.debug(f"NEMOTRON_DEBUG: Accumulating end tag: '{self._end_tag_buffer}'")
            
            # Check if we've accumulated the complete </think> tag
            if self.end_token in self._end_tag_buffer:
                logger.debug(f"NEMOTRON_DEBUG: Complete </think> detected, marking reasoning as ended")
                self._reasoning_ended = True
                
                # Extract any content after </think>
                end_tag_index = self._end_tag_buffer.find(self.end_token)
                content_after_tag = self._end_tag_buffer[end_tag_index + len(self.end_token):]
                self._end_tag_buffer = ""  # Clear the buffer
                
                if content_after_tag:
                    logger.debug(f"NEMOTRON_DEBUG: Content after </think>: '{content_after_tag}'")
                    # This might be whitespace or start of final content
                    if content_after_tag.strip():
                        return DeltaMessage(content=content_after_tag.strip())
                return None  # Don't stream the </think> tag itself
            
            # Check if the accumulated content can't possibly be </think>
            elif not self.end_token.startswith(self._end_tag_buffer) and len(self._end_tag_buffer) > 1:
                logger.debug(f"NEMOTRON_DEBUG: Buffer doesn't match </think>, flushing: '{self._end_tag_buffer}'")
                # This isn't the end tag, flush the buffer as reasoning content
                content_to_stream = self._end_tag_buffer
                self._end_tag_buffer = ""
                return DeltaMessage(reasoning_content=content_to_stream)
            
            # Still accumulating, wait for more tokens
            return None
        
        # Normal processing - not accumulating end tag
        cleaned_delta = delta_text.replace(self.start_token, "")
        
        # Check if we're approaching or in the middle of the </think> tag
        # by examining if the combined text would contain it
        combined_text = current_text + delta_text
        if self.end_token in combined_text:
            # We're at or past the end tag boundary, but this is Case 3 (no </think> found yet in current_text)
            # This means the delta_text contains part of </think> - we should filter it out
            
            # Find where </think> starts in the combined text
            end_tag_start = combined_text.find(self.end_token)
            
            # Calculate how much of the end tag is in previous_text vs delta_text
            if end_tag_start >= len(current_text):
                # The entire </think> is in delta_text - remove it all
                cleaned_delta = ""
            else:
                # Part of </think> is in previous text, part in delta
                # Remove the portion that's in delta_text
                end_tag_end = end_tag_start + len(self.end_token)
                if end_tag_end > len(current_text):
                    # Some of </think> extends into delta_text
                    delta_start_in_combined = len(current_text)
                    portion_in_delta_start = max(0, end_tag_start - delta_start_in_combined)
                    portion_in_delta_end = min(len(delta_text), end_tag_end - delta_start_in_combined)
                    
                    if portion_in_delta_end > portion_in_delta_start:
                        # Remove the portion of </think> that's in delta_text
                        before_tag = delta_text[:portion_in_delta_start]
                        after_tag = delta_text[portion_in_delta_end:]
                        cleaned_delta = before_tag + after_tag
                        # Also remove start/end tokens from the cleaned result
                        cleaned_delta = cleaned_delta.replace(self.start_token, "").replace(self.end_token, "")
        
        if not cleaned_delta:
            return None
        
        # Check if this is no-think mode by looking at the request
        is_no_think = False
        if request and hasattr(request, 'messages') and request.messages:
            for message in request.messages:
                content = None
                if isinstance(message, dict):
                    content = message.get('content')
                elif hasattr(message, 'content'):
                    content = message.content
                
                if content and '/no_think' in str(content):
                    is_no_think = True
                    break
        
        # If this is no-think mode, treat everything as content
        if is_no_think:
            logger.debug(f"NEMOTRON_DEBUG: No-think mode - Returning as content: '{delta_text}' (repr: {repr(delta_text)})")
            return DeltaMessage(content=delta_text)
        
        # Default: treat as reasoning content
        logger.debug(f"NEMOTRON_DEBUG: Default case - Returning as reasoning_content: '{cleaned_delta}' (repr: {repr(cleaned_delta)})")
        return DeltaMessage(reasoning_content=cleaned_delta)

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
