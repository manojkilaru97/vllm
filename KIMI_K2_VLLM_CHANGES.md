# Kimi K2 vLLM Tool Calling Changes

This document describes all modifications made to vLLM to support Kimi K2.5 model tool calling. Use this as a reference when porting these changes to a different vLLM version.

## Overview

The Kimi K2 model uses a unique marker-based format for tool calls that differs from the standard JSON format expected by OpenAI-compatible APIs. These changes add proper parsing and streaming support for:

1. **Single tool calls** (streaming and non-streaming)
2. **Parallel tool calls** (multiple tools in one response)
3. **tool_choice variants**: `auto`, `required`, `none`, and named (`{"type": "function", "function": {"name": "..."}}`)

## Files Modified

Only **3 files** need to be modified:

| File | Purpose |
|------|---------|
| `vllm/tool_parsers/kimi_k2_tool_parser.py` | Core tool call parsing logic |
| `vllm/entrypoints/openai/chat_completion/serving.py` | Streaming response handling |
| `vllm/entrypoints/openai/engine/serving.py` | Non-streaming response handling |

> **Note for v0.15.0**: In earlier vLLM versions, the files were named `serving_chat.py` and `serving_engine.py` directly in `vllm/entrypoints/openai/`. Starting with v0.15.0, they are reorganized into subdirectories: `chat_completion/serving.py` and `engine/serving.py`.

## Kimi K2 Tool Call Format

The model outputs tool calls in this marker-based format:

```
<|tool_calls_section_begin|>
<|tool_call_begin|>functions.get_weather:0
<|tool_call_argument_begin|>{"location": "Paris"}<|tool_call_end|>
<|tool_call_begin|>functions.calculate:1
<|tool_call_argument_begin|>{"expression": "5+5"}<|tool_call_end|>
<|tool_calls_section_end|>
```

Key markers:
- `<|tool_calls_section_begin|>` / `<|tool_calls_section_end|>` - Section boundaries
- `<|tool_call_begin|>` - Start of individual tool call
- `<|tool_call_argument_begin|>` - Start of JSON arguments
- `<|tool_call_end|>` - End of individual tool call

---

## Change 1: kimi_k2_tool_parser.py

### 1.1 Add Per-Tool State Tracking (Critical for Parallel Calls)

**Location**: `__init__` method

**Original:**
```python
class KimiK2ToolParser(ToolParser):
    def __init__(self, tokenizer: TokenizerLike):
        super().__init__(tokenizer)
        self.current_tool_name_sent: bool = False
        self.prev_tool_call_arr: list[dict] = []
```

**Modified:**
```python
class KimiK2ToolParser(ToolParser):
    def __init__(self, tokenizer: TokenizerLike):
        super().__init__(tokenizer)
        self.current_tool_name_sent: bool = False  # Legacy, kept for compatibility
        self.tool_name_sent_arr: list[bool] = []  # Track name-sent status PER tool
        self.prev_tool_call_arr: list[dict] = []
```

**Why**: The original code used a single `current_tool_name_sent` flag, which broke when multiple tools were called in parallel. Now we track per-tool state.

### 1.2 Reset Per-Tool Tracking

**Location**: Reset method (around line 170)

**Add this line after resetting `current_tool_name_sent`:**
```python
self.current_tool_name_sent = False  # Legacy
self.tool_name_sent_arr = []  # Per-tool name-sent tracking
```

### 1.3 Handle Direct Tool Calls (Without Section Wrapper)

**Location**: Section state transition check (around line 268)

**Original:**
```python
if found_section_begin and not self.in_tool_section:
    logger.debug("Entering tool section")
    self.in_tool_section = True
```

**Modified:**
```python
# Also enter tool section if tool_call_begin is detected directly
# (model may output tool calls without section wrapper)
has_direct_tool_call = self.tool_call_start_token_id in current_token_ids
if (found_section_begin or has_direct_tool_call) and not self.in_tool_section:
    logger.debug("Entering tool section (section_begin=%s, direct_tool=%s)",
                found_section_begin, has_direct_tool_call)
    self.in_tool_section = True
```

### 1.4 Strip All Markers From Remaining Content

**Location**: After section end is found (around line 294)

**Original:**
```python
if remaining.strip():
    return DeltaMessage(content=remaining)
```

**Modified:**
```python
# CRITICAL: Strip ALL tool markers (not just section markers) to
# prevent leaking raw tokens like <|tool_call_begin|> into content
if remaining.strip():
    cleaned_remaining = self._strip_all_tool_markers(remaining)
    # Only return if there's actual content after stripping
    if cleaned_remaining.strip():
        return DeltaMessage(content=cleaned_remaining)
```

### 1.5 Add Helper Method to Strip All Markers

**Location**: Add as a new method in the class

```python
def _strip_all_tool_markers(self, text: str) -> str:
    """Remove all Kimi K2 tool markers from text to prevent leakage."""
    markers = [
        "<|tool_calls_section_begin|>",
        "<|tool_calls_section_end|>",
        "<|tool_call_begin|>",
        "<|tool_call_end|>",
        "<|tool_call_argument_begin|>",
    ]
    for marker in markers:
        text = text.replace(marker, "")
    return text
```

### 1.6 Fix "Generating Text Content" Check

**Location**: Around line 381

**Original:**
```python
if (
    cur_tool_start_count == cur_tool_end_count
    and prev_tool_end_count == cur_tool_end_count
    and self.tool_call_end_token not in delta_text
):
```

**Modified:**
```python
# BUT only if all tools have been processed (don't skip if unprocessed tools remain)
tools_processed_so_far = self.current_tool_id + 1
if (
    cur_tool_start_count == cur_tool_end_count
    and prev_tool_end_count == cur_tool_end_count
    and self.tool_call_end_token not in delta_text
    and cur_tool_start_count <= tools_processed_so_far  # All tools processed
):
```

### 1.7 Handle New Tool Starting Before Previous Name Sent (CRITICAL)

**Location**: When starting a new tool call (around line 419)

This is the most critical change for parallel tool calls. When a new tool starts (`<|tool_call_begin|>` detected) but the previous tool's name hasn't been sent yet, we must send it immediately.

**Replace the new tool call detection block with:**

```python
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

    # Now actually start the new tool
    self.current_tool_id += 1

    # ... rest of new tool handling
    self.current_tool_name_sent = False  # Legacy
    self.tool_name_sent_arr.append(False)  # Per-tool tracking
    self.streamed_args_for_tool.append("")
    # CRITICAL: Also add placeholder to prev_tool_call_arr to keep arrays in sync
    if len(self.prev_tool_call_arr) <= self.current_tool_id:
        self.prev_tool_call_arr.append({})
```

### 1.8 Handle Tool Closing Without Name Sent

**Location**: When a tool closes (around line 561)

When a tool ends (`<|tool_call_end|>` detected) but its name hasn't been sent yet, send it:

```python
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
```

### 1.9 Use Per-Tool Name Tracking When Sending Names

**Location**: Where `current_tool_name_sent` is checked (around line 738)

**Original:**
```python
if not self.current_tool_name_sent:
```

**Modified:**
```python
# Use per-tool tracking to correctly handle parallel tool calls
tool_idx = self.current_tool_id
name_sent_for_this_tool = (
    tool_idx < len(self.tool_name_sent_arr) and self.tool_name_sent_arr[tool_idx]
)
if not name_sent_for_this_tool:
```

And when marking name as sent:

**Original:**
```python
self.current_tool_name_sent = True
```

**Modified:**
```python
self.current_tool_name_sent = True  # Legacy
# Update per-tool tracking
while len(self.tool_name_sent_arr) <= tool_idx:
    self.tool_name_sent_arr.append(False)
self.tool_name_sent_arr[tool_idx] = True
```

### 1.10 Include Arguments With Name If Available

When sending the name delta, also include arguments if they're already available (handles complete tool calls in one chunk):

```python
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
```

---

## Change 2: chat_completion/serving.py (was serving_chat.py)

### 2.1 Add Helper Methods for Kimi K2 Marker Detection

**Location**: Add as static methods in `OpenAIServingChat` class (around line 541)

```python
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
```

### 2.2 Add Tracking for Named Tool Choice Streaming

**Location**: After initializing `previous_tool_calls` (around line 720)

```python
# Track previously sent arguments for named tool_choice streaming (delta computation)
named_tool_previous_args = [""] * num_choices
```

### 2.3 Handle Named Tool Choice Streaming With Markers

**Location**: In the named tool_choice streaming section (around line 987)

Replace the simple streaming logic with marker-aware logic:

```python
# Named tool_choice streaming - handle Kimi K2 marker format
# Accumulate full text to extract clean JSON arguments
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

            delta_message = DeltaMessage(tool_calls=[delta_tool_call])
            tools_streamed[i] = True
        else:
            delta_message = None
    else:
        # Markers detected but arguments not yet complete, wait
        delta_message = None
else:
    # No markers yet - this is reasoning text before the tool call
    # Suppress it (don't send reasoning as arguments)
    delta_message = None
```

### 2.4 Handle tool_choice=required Streaming With Markers

**Location**: In the required tool_choice streaming section (around line 1088)

```python
# Check if the model outputs Kimi K2 marker format
# instead of the expected JSON format
if OpenAIServingChat._has_kimi_k2_markers(content):
    # Handle Kimi K2 marker format for tool_choice=required
    # Wait for complete arguments (partial_ok=False)
    extracted_args = OpenAIServingChat._extract_kimi_k2_arguments(content, partial_ok=False)
    if extracted_args is not None and extracted_args.strip():
        # Try to parse the function name from markers
        func_name = None
        import re as re_std
        # Pattern: <|tool_call_begin|>functions.name:0 or name:0
        name_match = re_std.search(
            r"<\|tool_call_begin\|>\s*(?:functions\.)?(\w+):\d+",
            content
        )
        if name_match:
            func_name = name_match.group(1)

        if not fn_name_returned and func_name:
            delta_message = DeltaMessage(
                tool_calls=[
                    DeltaToolCall(
                        id=make_tool_call_id(),
                        type="function",
                        function=DeltaFunctionCall(
                            name=func_name,
                            arguments=extracted_args,
                        ),
                        index=i,
                    )
                ]
            )
            function_name_returned[i] = True
        else:
            delta_message = None
    else:
        delta_message = None
else:
    # Standard JSON format - use existing method
    delta_message, function_name_returned[i] = (
        self.extract_tool_call_required_streaming(...)
    )
```

### 2.5 Yield Tool Name at Finish Time (CRITICAL FIX)

**Location**: In the finish path, right after `self._raise_if_error` (around line 1374)

When `output.finish_reason` is set in the same iteration where the parser returns a tool name, the normal streaming path is skipped. We must yield the name before it's lost:

```python
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
```

**Why this is critical**: Without this fix, tool 1's name would be lost because:
1. Parser returns delta with tool 1's name, sets `tool_name_sent_arr[1]=True`
2. But `finish_reason` is set, so code goes to finish path
3. The delta gets overwritten by remaining args logic
4. `tool_name_sent_arr[1]=True` prevents "missed tools" check from sending it

### 2.6 Handle String Arguments in Remaining Args Logic

**Location**: When calculating `expected_call` (around line 1451)

**Original:**
```python
expected_call = json.dumps(
    tool_parser.prev_tool_call_arr[index].get("arguments", {}),
    ensure_ascii=False,
)
```

**Modified:**
```python
raw_args = tool_parser.prev_tool_call_arr[index].get("arguments", {})
# Some parsers (e.g., kimi) store arguments as a string,
# others store as a dict. Only json.dumps if it's a dict.
if isinstance(raw_args, str):
    expected_call = raw_args
else:
    expected_call = json.dumps(raw_args, ensure_ascii=False)
```

### 2.7 Include Name With Remaining Args If Not Sent

**Location**: When creating delta for remaining args (around line 1469)

```python
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
```

### 2.8 Fallback: Check for Missed Tool Names at Finish

**Location**: After sending remaining args, before sending finish (around line 1520)

```python
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
```

---

## Change 3: engine/serving.py (was serving_engine.py)

### 3.1 Add Helper Methods for Kimi K2 Marker Handling

**Location**: Add as static methods in `OpenAIServing` class (around line 1507)

```python
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
def _extract_kimi_k2_tool_calls(content: str) -> list[tuple[str, str]]:
    """
    Extract tool calls from Kimi K2 marker format.
    Returns: List of (function_name, arguments_json) tuples.
    """
    import re as re_std
    tool_calls = []

    # Pattern to match each tool call block
    pattern = r"<\|tool_call_begin\|>\s*(?:functions\.)?(\w+):\d+\s*<\|tool_call_argument_begin\|>\s*(.*?)\s*<\|tool_call_end\|>"
    matches = re_std.findall(pattern, content, re_std.DOTALL)

    for func_name, args in matches:
        tool_calls.append((func_name, args.strip()))

    return tool_calls

@staticmethod
def _extract_kimi_k2_single_arguments(text: str) -> str | None:
    """Extract arguments from a single Kimi K2 tool call format."""
    arg_begin = "<|tool_call_argument_begin|>"
    arg_end = "<|tool_call_end|>"

    if arg_begin in text:
        start = text.find(arg_begin) + len(arg_begin)
        end_pos = text.find(arg_end, start)
        if end_pos > start:
            return text[start:end_pos].strip()
        else:
            return text[start:].strip()
    return None
```

### 3.2 Handle Named Tool Choice (Non-Streaming)

**Location**: In `_parse_tool_calls_from_content` method for `ToolChoiceFunction` (around line 1574)

**Original:**
```python
function_calls.append(
    FunctionCall(name=request.tool_choice.name, arguments=content)
)
```

**Modified:**
```python
# Forced Function Call - handle Kimi K2 marker format
if OpenAIServing._has_kimi_k2_markers(content):
    arguments = OpenAIServing._extract_kimi_k2_single_arguments(content)
    if arguments is None:
        arguments = ""
else:
    arguments = content
function_calls.append(
    FunctionCall(name=request.tool_choice.name, arguments=arguments)
)
```

### 3.3 Handle tool_choice=required (Non-Streaming)

**Location**: In `_parse_tool_calls_from_content` for `tool_choice == "required"` (around line 1590)

**Original:**
```python
tool_calls = TypeAdapter(list[FunctionDefinition]).validate_json(content)
function_calls.extend([...])
```

**Modified:**
```python
# Handle Kimi K2 marker format for tool_choice=required
if OpenAIServing._has_kimi_k2_markers(content):
    extracted_calls = OpenAIServing._extract_kimi_k2_tool_calls(content)
    for func_name, args in extracted_calls:
        function_calls.append(
            FunctionCall(name=func_name, arguments=args)
        )
else:
    # Standard JSON format
    tool_calls = TypeAdapter(list[FunctionDefinition]).validate_json(content)
    function_calls.extend([...])
```

---

## Testing

### Test Scripts

1. **Parallel tool calls**: Test that multiple tools get proper names
   ```python
   prompt = "Calculate 7+8*2 and weather in Paris"
   # Should return: calculate + get_weather, both with names
   ```

2. **Concurrent requests**: Test isolation between requests
   ```python
   # Mix of single tools, parallel tools, and no-tools requests
   # Verify no cross-request state leakage
   ```

3. **tool_choice variants**:
   - `auto`: Model decides whether to use tools
   - `required`: Model must use tools
   - `none`: Model must not use tools
   - Named: Model must use specific tool

### Key Success Criteria

1. **Parallel tool calls**: Both/all tools have `name` field populated
2. **No marker leakage**: Response content doesn't contain `<|tool_call...|>` markers
3. **No duplicate args**: Arguments aren't sent twice
4. **Proper finish_reason**: `tool_calls` for auto/required, `stop` for named
5. **Same tool multiple times**: Each occurrence gets correct, separate arguments
6. **Valid JSON**: All tool arguments are valid JSON (no `<think>` content, no corruption)

### Test Results (v0.15.0 port)

| Test Suite | Result |
|------------|--------|
| Parallel tool calls (15 runs) | **15/15 ✅** |
| Concurrent streaming (10 requests) | **10/10 ✅** |
| Full tool calling suite (121 checks) | **121/121 ✅** |
| Edge case tests (79 tests) | **~75/79 (95%)** |

Note: Edge case failures are model behavior (not calling enough tools), not streaming bugs.

---

## Common Issues and Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Tool 1 missing name | Name delta lost at finish time | Yield name_only_delta in finish path (Section 2.5) |
| Duplicate arguments | Args sent by parser AND finish handling | Only send name (no args) in finish path delta |
| Marker leakage | Markers not stripped from content | Use `_strip_all_tool_markers()` |
| Garbage in args | Stale `prev_tool_call_arr` | Reset/update arr when new tool starts |
| Name not sent for closing tool | Tool closes before name extracted | Handle in tool close section (Section 1.8) |

---

## Change 4: Same Tool Multiple Times (tool_choice=required)

### 4.1 Fix Regex Match Index for Repeated Tool Names

**Location**: `extract_tool_call_required_streaming` method (around line 710)

When the same tool appears multiple times (e.g., "calculate 1+1 and calculate 2+2"), the original code used `re.search` which always finds the FIRST occurrence. This caused the second tool's arguments to be extracted from the wrong position.

**Original:**
```python
tool_start_pattern = rf'"name":\s*"{re.escape(current_tool_call["name"])}"[^{{]*"parameters":\s*'
param_match = re.search(tool_start_pattern, current_text, re.DOTALL)
if param_match:
    param_start = param_match.end()
    raw_params = current_text[param_start:]
```

**Modified:**
```python
tool_start_pattern = rf'"name":\s*"{re.escape(current_tool_call["name"])}"[^{{]*"parameters":\s*'
matches = list(re.finditer(tool_start_pattern, current_text))

# Count how many times this tool name has appeared before
# in the array (to find the correct match)
same_name_count = sum(
    1 for i, t in enumerate(obj)
    if i < current_tool_idx and t.get("name") == current_tool_call["name"]
)
match_idx = same_name_count  # 0-indexed: first=0, second=1, etc.

if matches and match_idx < len(matches):
    match = matches[match_idx]
    param_start = match.end()
    raw_params = current_text[param_start:]
    arguments, _ = OpenAIServingChat._filter_delta_text(raw_params, "")
else:
    arguments = ""
```

**Why**: For the second occurrence of `calculate`, we need to find the SECOND match in the text, not the first. The `same_name_count` tracks how many tools with the same name appeared before the current one.

---

## Change 5: Named Tool Choice Text Accumulation

### 5.1 Fix previous_texts[i] Update Condition

**Location**: End of streaming loop iteration (around line 1484)

For named tool_choice streaming, text must accumulate across iterations to build complete JSON arguments. The original condition didn't update `previous_texts[i]` for named tool_choice.

**Original:**
```python
if (
    tool_choice_auto or self.reasoning_parser
) and not self.use_harmony:
    assert previous_texts is not None
    previous_texts[i] = current_text
```

**Modified:**
```python
if (
    tool_choice_auto or self.reasoning_parser or tool_choice_function_name
) and not self.use_harmony:
    assert previous_texts is not None
    previous_texts[i] = current_text
```

**Why**: Without this fix, named tool_choice streaming loses accumulated text between iterations, causing incomplete JSON arguments.

---

## Change 6: Handle Raw JSON Output From Reasoning Parser

### 6.1 Preserve JSON When Reasoning Parser Returns It

**Location**: In reasoning parser block for named tool_choice (around line 1099)

When the model outputs raw JSON without `<thinking>` tags, the reasoning parser may return it as "reasoning" content. We need to preserve this for tool argument extraction.

**Added:**
```python
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
```

---

## Commit History

| Commit | Description |
|--------|-------------|
| `TBD` | Fix same-tool-multiple-times streaming and named tool_choice accumulation |
| `a5cdb0145` | Clean up debug logs and add porting documentation |
| `bc2a9b31e` | Fix parallel tool calls streaming - yield name at finish time |
| `b7caabed4` | Fix tool_choice=required streaming (minimal fix) |
| `a781e0b45` | Revert broken named tool_choice streaming changes |
| `d445ace6b` | Fix tool_choice=required and named tool_choice marker leaks |
| `a6691e781` | Fix stale prev_tool_call_arr causing garbage arguments |
| `a618d78d0` | Fix double-encoding of tool call arguments in streaming |
| `da385abb7` | Fix streaming tool call arguments being dropped |
| `2afb33a3b` | NVCF production features + tool call streaming fix |

---

## Version Compatibility

These changes were developed against vLLM with the following key dependencies:
- Python 3.10+
- Pydantic for model_dump()
- Standard vLLM tool parser interface

When porting to a new vLLM version:
1. Check if `ToolParser` base class interface changed
2. Check if `DeltaMessage`, `DeltaToolCall`, `DeltaFunctionCall` types changed
3. Check if `ChatCompletionResponseStreamChoice` structure changed
4. Review streaming generator logic in `chat_completion/serving.py`
