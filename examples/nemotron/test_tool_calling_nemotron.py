#!/usr/bin/env python3
"""
Nemotron variant of the tool-calling test runner that exercises both
"thinking" modes by injecting a system prompt:
  - "detailed thinking on"
  - "detailed thinking off"

It mirrors the original test runner's functionality and adds:
  --thinking-mode {on,off,both} (default: both)

Each request prepends a system message with the chosen thinking directive.
"""

import json
import argparse
import os
import time
import requests
from typing import List, Dict, Any, Optional, Tuple

# Test result tracking
class TestResult:
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.errors = []
        self.warnings = []
        self.details = {}
    
    def add_error(self, msg: str):
        self.errors.append(msg)
    
    def add_warning(self, msg: str):
        self.warnings.append(msg)
    
    def set_passed(self, passed: bool):
        self.passed = passed
    
    def summary(self) -> str:
        status = "✅ PASS" if self.passed else "❌ FAIL"
        return f"{status} {self.name}"

test_results = []

BASE_URL = os.environ.get("SGLANG_BASE_URL", "http://localhost:8003")
CHAT_URL = f"{BASE_URL}/v1/chat/completions"
MODELS_URL = f"{BASE_URL}/v1/models"
HEALTH_URL = f"{BASE_URL}/health"
DEFAULT_MODEL = os.environ.get("SGLANG_MODEL")
API_KEY = os.environ.get("SGLANG_API_KEY")
HEADERS = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else None
MAX_TOKENS = 4096

# Thinking mode: "on" or "off". Use helper to get precise system prompt string
THINKING_MODE = "on"


def get_system_prompt() -> str:
    base = "detailed thinking on" if THINKING_MODE == "on" else "detailed thinking off"
    # For Nemotron tool-calling, in OFF mode encourage strict <TOOLCALL> emission.
    # if THINKING_MODE == "off":
        # base += (
        #     "\nWhen you decide to use tools, output exactly one tool call in this exact format and nothing else:\n"
        #     "<TOOLCALL>[{\"name\":\"FUNCTION_NAME\",\"arguments\":{...}}]</TOOLCALL>\n"
        #     "No prose, no code fences, no extra text."
        # )
    return base


# Sample tools for testing (same as original)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather information for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform mathematical calculations",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "execute_sql",
            "description": "Execute a SQL query against a database (no actual execution in this test)",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "SQL statement to execute"
                    },
                    "dialect": {
                        "type": "string",
                        "description": "SQL dialect",
                        "enum": ["sqlite", "postgres", "mysql"]
                    },
                    "database": {
                        "type": "string",
                        "description": "Database name or connection key (optional)"
                    }
                },
                "required": ["query", "dialect"]
            }
        }
    }
]


def detect_default_model() -> str:
    """Detect the default model by querying /v1/models; return first model id if available."""
    try:
        resp = requests.get(MODELS_URL, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        models = data.get("data") or data.get("models") or []
        if models and isinstance(models, list):
            first = models[0]
            if isinstance(first, dict) and first.get("id"):
                return first["id"]
            if isinstance(first, str):
                return first
    except Exception:
        pass
    return ""


def validate_tool_response(expected_tools: List[str], actual_tool_calls: List[Dict], 
                          content: str, thinking_mode: str, result: TestResult) -> None:
    """Validate tool calling response based on expected behavior."""
    
    # Safety check - ensure result is actually a TestResult
    if not isinstance(result, TestResult):
        print(f"ERROR: validate_tool_response received {type(result)} instead of TestResult")
        return
    
    # Basic validation
    if expected_tools and not actual_tool_calls:
        result.add_error(f"Expected tool calls for {expected_tools} but got none")
        return
    
    if not expected_tools and actual_tool_calls:
        result.add_warning(f"Got unexpected tool calls: {[tc.get('function', {}).get('name') for tc in actual_tool_calls]}")
    
    # Validate specific tool calls
    if expected_tools and actual_tool_calls:
        actual_names = [tc.get('function', {}).get('name') for tc in actual_tool_calls]
        
        for expected in expected_tools:
            if expected not in actual_names:
                result.add_error(f"Expected tool '{expected}' not found in actual calls: {actual_names}")
        
        # Validate tool call structure
        for tc in actual_tool_calls:
            func = tc.get('function', {})
            if not func.get('name'):
                result.add_error("Tool call missing function name")
            if not func.get('arguments'):
                result.add_warning("Tool call has empty arguments")
            else:
                try:
                    json.loads(func['arguments'])
                except json.JSONDecodeError:
                    result.add_error(f"Tool call arguments are not valid JSON: {func['arguments']}")
    
    # Content validation based on thinking mode
    if thinking_mode == "off" and content and actual_tool_calls:
        # In thinking off mode, we might expect minimal content when tools are called
        pass  # This is acceptable
    
    # Set overall pass/fail
    result.set_passed(len(result.errors) == 0)

def make_request(messages: List[Dict], tools: List[Dict] = None,
                stream: bool = False, model: str = None, tool_choice: Any = None,
                system_override: Optional[str] = None, test_name: str = None,
                expected_tools: List[str] = None) -> Tuple[Dict[str, Any], Optional[TestResult]]:
    """Make a request to the chat completions endpoint with Nemotron system prompt injected."""

    # Create test result if test_name provided
    result = TestResult(test_name) if test_name else None
    
    # Prepend system prompt to toggle thinking behavior
    system_message = {"role": "system", "content": system_override if system_override is not None else get_system_prompt()}
    request_messages = [system_message] + messages.copy()
    
    current_thinking_mode = "on" if "thinking on" in system_message["content"] else "off"

    use_model = model or DEFAULT_MODEL or detect_default_model()

    payload = {
        "model": use_model,
        "messages": request_messages,
        "stream": stream,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.1
    }

    if tools:
        payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        else:
            payload["tool_choice"] = "auto"

    if test_name:
        print(f"\n🔄 {test_name}")
    print(f"\n{'='*80}")
    print(f"REQUEST: stream={stream}, tools={'Yes' if tools else 'No'}")
    auth_present = bool(HEADERS and HEADERS.get('Authorization'))
    tool_names = [t.get('function', {}).get('name') for t in (tools or [])]
    print(
        "Meta: "
        f"base_url={BASE_URL}, model={use_model}, max_tokens={MAX_TOKENS}, temperature=0.1, "
        f"tool_choice={json.dumps(payload.get('tool_choice')) if 'tool_choice' in payload else None}, "
        f"auth_header={'present' if auth_present else 'absent'}, thinking_mode={THINKING_MODE}"
    )
    print(f"System prompt: {get_system_prompt()}")
    if tools:
        print(f"Tools ({len(tool_names)}): {tool_names}")
    print(f"Messages: {json.dumps(request_messages, indent=2)}")
    print(f"{'='*80}")

    try:
        if stream:
            response = requests.post(CHAT_URL, json=payload, headers=HEADERS, stream=True, timeout=60)
            response.raise_for_status()

            print("STREAMING RESPONSE:")
            full_content = ""
            sse_events: List[Dict[str, Any]] = []
            assembled_tool_calls: List[Dict[str, Any]] = []

            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data_str = line[6:]
                        if data_str.strip() == '[DONE]':
                            break
                        try:
                            data = json.loads(data_str)
                            if 'choices' in data and len(data['choices']) > 0:
                                delta = data['choices'][0].get('delta', {})
                                sse_events.append(delta)
                                if 'content' in delta and delta['content']:
                                    print(delta['content'], end='', flush=True)
                                    full_content += delta['content']
                                if 'tool_calls' in delta and delta['tool_calls']:
                                    for idx, tc_delta in enumerate(delta['tool_calls']):
                                        target_index = tc_delta.get('index', idx)
                                        while len(assembled_tool_calls) <= target_index:
                                            assembled_tool_calls.append({
                                                'id': None,
                                                'index': target_index,
                                                'type': 'function',
                                                'function': {'name': None, 'arguments': ''}
                                            })
                                        current = assembled_tool_calls[target_index]
                                        if 'id' in tc_delta and tc_delta['id']:
                                            current['id'] = tc_delta['id']
                                        func_delta = tc_delta.get('function', {})
                                        if func_delta:
                                            if 'name' in func_delta and func_delta['name']:
                                                current['function']['name'] = func_delta['name']
                                            if 'arguments' in func_delta and func_delta['arguments']:
                                                current['function']['arguments'] += func_delta['arguments']
                        except json.JSONDecodeError:
                            continue

            print(f"\n\nFULL CONTENT: {repr(full_content)}")
            if assembled_tool_calls:
                print("ASSEMBLED TOOL CALLS:")
                print(json.dumps(assembled_tool_calls, indent=2))
                print("\nTOOL CALL SUMMARY:")
                for tc in assembled_tool_calls:
                    name = tc.get('function', {}).get('name')
                    args = tc.get('function', {}).get('arguments')
                    print(f"- {name}({args})")

            print(f"\nSSE EVENT COUNT: {len(sse_events)}")

            response_data = {"content": full_content, "tool_calls": assembled_tool_calls, "sse_events": sse_events}
            
            # Validate if this is a test
            if result:
                validate_tool_response(expected_tools or [], assembled_tool_calls, full_content, current_thinking_mode, result)
            
            return response_data, result

        else:
            response = requests.post(CHAT_URL, json=payload, headers=HEADERS, timeout=60)
            response.raise_for_status()
            response_json = response.json()

            print("NON-STREAMING RESPONSE:")
            print(json.dumps(response_json, indent=2))

            if 'choices' in response_json and len(response_json['choices']) > 0:
                message = response_json['choices'][0]['message']
                content = message.get('content', '')
                tool_calls = message.get('tool_calls', [])
                if tool_calls:
                    print("\nNON-STREAM TOOL CALL SUMMARY:")
                    for tc in tool_calls:
                        fn = (tc.get('function') or {}).get('name')
                        args = (tc.get('function') or {}).get('arguments')
                        print(f"- {fn}({args})")

                response_data = {"content": content, "tool_calls": tool_calls}
            else:
                response_data = {"content": "", "tool_calls": []}
            
            # Validate if this is a test
            if result:
                validate_tool_response(expected_tools or [], response_data.get("tool_calls", []), 
                                     response_data.get("content", ""), current_thinking_mode, result)
            
            return response_data, result

    except Exception as e:
        print(f"ERROR: {e}")
        if result:
            result.add_error(f"Request failed: {e}")
        return {"error": str(e)}, result


def test_normal_queries():
    """Test normal queries without tools."""
    print(f"\n{'#'*80}")
    print("TESTING NORMAL QUERIES (NO TOOLS)")
    print(f"{'#'*80}")

    messages = [{"role": "user", "content": "What is the capital of France? Explain briefly."}]

    for stream in [False, True]:
        stream_type = "streaming" if stream else "non-streaming"
        response_data, test_result = make_request(
            messages, tools=None, stream=stream, 
            test_name=f"Normal Query ({stream_type})",
            expected_tools=[]
        )
        if test_result:
            test_results.append(test_result)
        time.sleep(1)


def test_single_tool_calling():
    """Test single tool calling."""
    print(f"\n{'#'*80}")
    print("TESTING SINGLE TOOL CALLING")
    print(f"{'#'*80}")

    messages = [{"role": "user", "content": "What's the weather like in San Francisco?"}]

    for stream in [False, True]:
        stream_type = "streaming" if stream else "non-streaming"
        response_data, test_result = make_request(
            messages, tools=TOOLS, stream=stream, tool_choice="auto",
            test_name=f"Single Tool Call ({stream_type})",
            expected_tools=["get_weather"]
        )
        if test_result:
            test_results.append(test_result)
        time.sleep(1)


def test_parallel_tool_calling():
    """Test parallel tool calling."""
    print(f"\n{'#'*80}")
    print("TESTING PARALLEL TOOL CALLING")
    print(f"{'#'*80}")

    messages = [{"role": "user", "content": "I need the weather in both New York and Los Angeles, and also calculate 15 * 23."}]

    for stream in [False, True]:
        stream_type = "streaming" if stream else "non-streaming"
        response_data, test_result = make_request(
            messages, tools=TOOLS, stream=stream, tool_choice="auto",
            test_name=f"Parallel Tool Calls ({stream_type})",
            expected_tools=["get_weather", "calculate"]  # May get multiple weather calls
        )
        if test_result:
            test_results.append(test_result)
        time.sleep(1)


def test_complex_tool_scenario():
    """Test a complex multi-turn tool calling scenario."""
    print(f"\n{'#'*80}")
    print("TESTING COMPLEX MULTI-TURN TOOL SCENARIO")
    print(f"{'#'*80}")

    messages = [{"role": "user", "content": "What's the weather in Tokyo? Use Celsius."}]

    response_data, test_result = make_request(
        messages, tools=TOOLS, stream=False, tool_choice="auto",
        test_name="Complex Multi-turn (initial)",
        expected_tools=["get_weather"]
    )
    if test_result:
        test_results.append(test_result)

    if response_data.get('tool_calls'):
        messages.append({
            "role": "assistant",
            "content": response_data.get('content', ''),
            "tool_calls": response_data['tool_calls']
        })

        for tool_call in response_data['tool_calls']:
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.get('id', 'call_123'),
                "content": json.dumps({
                    "location": "Tokyo, Japan",
                    "temperature": 22,
                    "unit": "celsius",
                    "description": "Sunny with light clouds"
                })
            })

        messages.append({
            "role": "user",
            "content": "Great! Now can you also search for recent news about Tokyo weather patterns?"
        })

        response_data, test_result2 = make_request(
            messages, tools=TOOLS, stream=True, tool_choice="auto",
            test_name="Complex Multi-turn (follow-up)",
            expected_tools=["search_web"]
        )
        if test_result2:
            test_results.append(test_result2)


def test_edge_cases():
    """Test edge cases and error scenarios."""
    print(f"\n{'#'*80}")
    print("TESTING EDGE CASES")
    print(f"{'#'*80}")

    # Test with no tools available
    messages = [{"role": "user", "content": "What's the weather?"}]
    response_data, test_result = make_request(
        messages, tools=[], stream=False,
        test_name="Edge Case: No tools available",
        expected_tools=[]
    )
    if test_result:
        test_results.append(test_result)

    # Test with very long message
    long_message = "Please " + "really " * 100 + "help me with the weather."
    messages = [{"role": "user", "content": long_message}]
    response_data, test_result2 = make_request(
        messages, tools=TOOLS, stream=False, tool_choice="auto",
        test_name="Edge Case: Long message",
        expected_tools=["get_weather"]
    )
    if test_result2:
        test_results.append(test_result2)


def test_sql_tool_choice_variants():
    """Test SQL tool calls with tool_choice = auto, required, and targeted function name."""
    print(f"\n{'#'*80}")
    print("TESTING SQL TOOL_CHOICE VARIANTS")
    print(f"{'#'*80}")

    messages_auto = [{
        "role": "user",
        "content": "I have a SQLite database. Run this SQL: SELECT 42 AS answer;"
    }]
    print("\n-- SQL tool_choice=auto --")
    for stream in [False, True]:
        stream_type = "streaming" if stream else "non-streaming"
        response_data, test_result = make_request(
            messages_auto, tools=TOOLS, stream=stream, tool_choice="auto",
            test_name=f"SQL auto ({stream_type})",
            expected_tools=["execute_sql"]
        )
        if test_result:
            test_results.append(test_result)
        time.sleep(0.5)

    messages_required = [{
        "role": "user",
        "content": "Please execute on sqlite: SELECT COUNT(*) AS n FROM users;"
    }]
    print("\n-- SQL tool_choice=required --")
    for stream in [False, True]:
        stream_type = "streaming" if stream else "non-streaming"
        response_data, test_result = make_request(
            messages_required, tools=TOOLS, stream=stream, tool_choice="required",
            test_name=f"SQL required ({stream_type})",
            expected_tools=["execute_sql"]
        )
        if test_result:
            test_results.append(test_result)
        time.sleep(0.5)

    messages_targeted = [{
        "role": "user",
        "content": "On sqlite, run: SELECT name FROM sqlite_master WHERE type='table' LIMIT 3;"
    }]
    targeted_choice = {"type": "function", "function": {"name": "execute_sql"}}
    print("\n-- SQL tool_choice={type:function, name:execute_sql} --")
    for stream in [False, True]:
        stream_type = "streaming" if stream else "non-streaming"
        response_data, test_result = make_request(
            messages_targeted, tools=TOOLS, stream=stream, tool_choice=targeted_choice,
            test_name=f"SQL targeted ({stream_type})",
            expected_tools=["execute_sql"]
        )
        if test_result:
            test_results.append(test_result)
        time.sleep(0.5)


def test_tool_choice_variants():
    """Test tool_choice modes: auto, required, and targeted function name, in streaming and non-streaming."""
    print(f"\n{'#'*80}")
    print("TESTING TOOL_CHOICE VARIANTS")
    print(f"{'#'*80}")

    messages_auto = [{"role": "user", "content": "What's the weather like in San Francisco?"}]
    print("\n-- tool_choice=auto --")
    for stream in [False, True]:
        stream_type = "streaming" if stream else "non-streaming"
        response_data, test_result = make_request(
            messages_auto, tools=TOOLS, stream=stream, tool_choice="auto",
            test_name=f"Tool choice auto ({stream_type})",
            expected_tools=["get_weather"]
        )
        if test_result:
            test_results.append(test_result)
        time.sleep(0.5)

    # New Delhi weather test with 'detailed thinking off' enforced via system override
    messages_delhi = [{"role": "user", "content": "Temperature in new delhi?"}]
    print("\n-- tool_choice=auto (New Delhi, thinking off) --")
    for stream in [False, True]:
        stream_type = "streaming" if stream else "non-streaming"
        response_data, test_result = make_request(
            messages_delhi, tools=TOOLS, stream=stream, tool_choice="auto",
            system_override="detailed thinking off",
            test_name=f"Tool choice auto - thinking off ({stream_type})",
            expected_tools=["get_weather"]
        )
        if test_result:
            test_results.append(test_result)
        time.sleep(0.5)

    messages_required = [{"role": "user", "content": "Briefly tell me the weather in San Francisco."}]
    print("\n-- tool_choice=required --")
    for stream in [False, True]:
        stream_type = "streaming" if stream else "non-streaming"
        response_data, test_result = make_request(
            messages_required, tools=TOOLS, stream=stream, tool_choice="required",
            test_name=f"Tool choice required ({stream_type})",
            expected_tools=["get_weather"]  # Should be forced to use a tool
        )
        if test_result:
            test_results.append(test_result)
        time.sleep(0.5)

    messages_targeted = [{"role": "user", "content": "Calculate 15 * 23."}]
    targeted_choice = {"type": "function", "function": {"name": "calculate"}}
    print("\n-- tool_choice={type:function, name:calculate} --")
    for stream in [False, True]:
        stream_type = "streaming" if stream else "non-streaming"
        response_data, test_result = make_request(
            messages_targeted, tools=TOOLS, stream=stream, tool_choice=targeted_choice,
            test_name=f"Tool choice targeted ({stream_type})",
            expected_tools=["calculate"]
        )
        if test_result:
            test_results.append(test_result)
        time.sleep(0.5)


def run_selected_tests(only: str) -> None:
    if only == "all":
        test_normal_queries()
        test_single_tool_calling()
        test_parallel_tool_calling()
        test_complex_tool_scenario()
        test_edge_cases()
        test_tool_choice_variants()
        test_sql_tool_choice_variants()
    elif only == "tools":
        test_single_tool_calling()
        test_parallel_tool_calling()
        test_complex_tool_scenario()
        test_tool_choice_variants()
        test_sql_tool_choice_variants()
    elif only == "single":
        test_single_tool_calling()
    elif only == "parallel":
        test_parallel_tool_calling()
    elif only == "complex":
        test_complex_tool_scenario()
    elif only == "tool_choice":
        test_tool_choice_variants()
    elif only == "sql":
        test_sql_tool_choice_variants()


def main():
    """Run selected tests against configured server in one or both thinking modes."""
    global BASE_URL, CHAT_URL, MODELS_URL, HEALTH_URL, DEFAULT_MODEL, HEADERS, MAX_TOKENS, THINKING_MODE

    parser = argparse.ArgumentParser(description="Nemotron tool-calling test runner with thinking-mode toggle")
    parser.add_argument("--base-url", default=os.environ.get("SGLANG_BASE_URL", "http://localhost:8003"), help="Server base URL (no trailing slash)")
    parser.add_argument("--model", default=os.environ.get("SGLANG_MODEL"), help="Model name (optional; will auto-detect if omitted)")
    parser.add_argument("--api-key", default=os.environ.get("SGLANG_API_KEY"), help="Bearer API key (optional)")
    parser.add_argument("--only", choices=["all", "tools", "single", "parallel", "complex", "tool_choice", "sql"], default="all", help="Run a subset of tests")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max tokens for completions")
    parser.add_argument("--thinking-mode", choices=["on", "off", "both"], default="both", help="Toggle Nemotron thinking via system prompt")
    args = parser.parse_args()

    BASE_URL = args.base_url.rstrip("/")
    CHAT_URL = f"{BASE_URL}/v1/chat/completions"
    MODELS_URL = f"{BASE_URL}/v1/models"
    HEALTH_URL = f"{BASE_URL}/health"
    DEFAULT_MODEL = args.model
    HEADERS = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else None
    MAX_TOKENS = max(1, args.max_tokens)

    print("Starting Nemotron Tool Calling Tests")
    print(f"Base URL: {BASE_URL}")
    print(f"Chat URL: {CHAT_URL}")

    # Health/models check
    try:
        r = requests.get(HEALTH_URL, headers=HEADERS, timeout=5)
        print(f"Health check: {r.status_code}")
    except Exception as e:
        print(f"Warning: Could not reach health endpoint {HEALTH_URL}: {e}")
    try:
        r = requests.get(MODELS_URL, headers=HEADERS, timeout=5)
        if r.ok:
            print(f"Models: {r.json()}")
        else:
            print(f"Models endpoint returned status {r.status_code}")
    except Exception as e:
        print(f"Warning: Could not reach models endpoint {MODELS_URL}: {e}")

    modes = [args.thinking_mode] if args.thinking_mode in ("on", "off") else ["on", "off"]

    for mode in modes:
        THINKING_MODE = mode
        print(f"\n{'#'*80}")
        print(f"RUNNING IN THINKING MODE: {THINKING_MODE}  (system: '{get_system_prompt()}')")
        print(f"{'#'*80}")
        run_selected_tests(args.only)

    # Print test summary
    print_test_summary()
    
    print(f"\n{'#'*80}")
    print("ALL TESTS COMPLETED")
    print(f"{'#'*80}")


def print_test_summary():
    """Print a summary of all test results."""
    if not test_results:
        return
    
    print(f"\n{'='*80}")
    print("🧪 TEST SUMMARY")
    print(f"{'='*80}")
    
    passed = 0
    failed = 0
    
    for result in test_results:
        print(result.summary())
        if result.passed:
            passed += 1
        else:
            failed += 1
        
        # Show errors and warnings
        for error in result.errors:
            print(f"   ❌ {error}")
        for warning in result.warnings:
            print(f"   ⚠️  {warning}")
    
    print(f"\n{'-'*80}")
    print(f"Total: {len(test_results)} tests, {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed!")
    else:
        print(f"💥 {failed} test(s) failed")

if __name__ == "__main__":
    main()


