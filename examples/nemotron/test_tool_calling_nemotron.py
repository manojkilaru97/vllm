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
from typing import List, Dict, Any

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
    if THINKING_MODE == "off":
        base += (
            "\nWhen you decide to use tools, output exactly one tool call in this exact format and nothing else:\n"
            "<TOOLCALL>[{\"name\":\"FUNCTION_NAME\",\"arguments\":{...}}]</TOOLCALL>\n"
            "No prose, no code fences, no extra text."
        )
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


def make_request(messages: List[Dict], tools: List[Dict] = None,
                stream: bool = False, model: str = None, tool_choice: Any = None) -> Dict[str, Any]:
    """Make a request to the chat completions endpoint with Nemotron system prompt injected."""

    # Prepend system prompt to toggle thinking behavior
    system_message = {"role": "system", "content": get_system_prompt()}
    request_messages = [system_message] + messages.copy()

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

            return {"content": full_content, "tool_calls": assembled_tool_calls, "sse_events": sse_events}

        else:
            response = requests.post(CHAT_URL, json=payload, headers=HEADERS, timeout=60)
            response.raise_for_status()
            result = response.json()

            print("NON-STREAMING RESPONSE:")
            print(json.dumps(result, indent=2))

            if 'choices' in result and len(result['choices']) > 0:
                message = result['choices'][0]['message']
                content = message.get('content', '')
                tool_calls = message.get('tool_calls', [])
                if tool_calls:
                    print("\nNON-STREAM TOOL CALL SUMMARY:")
                    for tc in tool_calls:
                        fn = (tc.get('function') or {}).get('name')
                        args = (tc.get('function') or {}).get('arguments')
                        print(f"- {fn}({args})")

                return {"content": content, "tool_calls": tool_calls}

            return {"content": "", "tool_calls": []}

    except Exception as e:
        print(f"ERROR: {e}")
        return {"error": str(e)}


def test_normal_queries():
    """Test normal queries without tools."""
    print(f"\n{'#'*80}")
    print("TESTING NORMAL QUERIES (NO TOOLS)")
    print(f"{'#'*80}")

    messages = [{"role": "user", "content": "What is the capital of France? Explain briefly."}]

    for stream in [False, True]:
        make_request(messages, tools=None, stream=stream)
        time.sleep(1)


def test_single_tool_calling():
    """Test single tool calling."""
    print(f"\n{'#'*80}")
    print("TESTING SINGLE TOOL CALLING")
    print(f"{'#'*80}")

    messages = [{"role": "user", "content": "What's the weather like in San Francisco?"}]

    for stream in [False, True]:
        make_request(messages, tools=TOOLS, stream=stream, tool_choice="auto")
        time.sleep(1)


def test_parallel_tool_calling():
    """Test parallel tool calling."""
    print(f"\n{'#'*80}")
    print("TESTING PARALLEL TOOL CALLING")
    print(f"{'#'*80}")

    messages = [{"role": "user", "content": "I need the weather in both New York and Los Angeles, and also calculate 15 * 23."}]

    for stream in [False, True]:
        make_request(messages, tools=TOOLS, stream=stream, tool_choice="auto")
        time.sleep(1)


def test_complex_tool_scenario():
    """Test a complex multi-turn tool calling scenario."""
    print(f"\n{'#'*80}")
    print("TESTING COMPLEX MULTI-TURN TOOL SCENARIO")
    print(f"{'#'*80}")

    messages = [{"role": "user", "content": "What's the weather in Tokyo? Use Celsius."}]

    result = make_request(messages, tools=TOOLS, stream=False, tool_choice="auto")

    if result.get('tool_calls'):
        messages.append({
            "role": "assistant",
            "content": result.get('content', ''),
            "tool_calls": result['tool_calls']
        })

        for tool_call in result['tool_calls']:
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

        make_request(messages, tools=TOOLS, stream=True, tool_choice="auto")


def test_edge_cases():
    """Test edge cases and error scenarios."""
    print(f"\n{'#'*80}")
    print("TESTING EDGE CASES")
    print(f"{'#'*80}")

    messages = [{"role": "user", "content": "What's the weather?"}]
    make_request(messages, tools=[], stream=False)

    long_message = "Please " + "really " * 100 + "help me with the weather."
    messages = [{"role": "user", "content": long_message}]
    make_request(messages, tools=TOOLS, stream=False, tool_choice="auto")


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
        make_request(messages_auto, tools=TOOLS, stream=stream, tool_choice="auto")
        time.sleep(0.5)

    messages_required = [{
        "role": "user",
        "content": "Please execute on sqlite: SELECT COUNT(*) AS n FROM users;"
    }]
    print("\n-- SQL tool_choice=required --")
    for stream in [False, True]:
        make_request(messages_required, tools=TOOLS, stream=stream, tool_choice="required")
        time.sleep(0.5)

    messages_targeted = [{
        "role": "user",
        "content": "On sqlite, run: SELECT name FROM sqlite_master WHERE type='table' LIMIT 3;"
    }]
    targeted_choice = {"type": "function", "function": {"name": "execute_sql"}}
    print("\n-- SQL tool_choice={type:function, name:execute_sql} --")
    for stream in [False, True]:
        make_request(messages_targeted, tools=TOOLS, stream=stream, tool_choice=targeted_choice)
        time.sleep(0.5)


def test_tool_choice_variants():
    """Test tool_choice modes: auto, required, and targeted function name, in streaming and non-streaming."""
    print(f"\n{'#'*80}")
    print("TESTING TOOL_CHOICE VARIANTS")
    print(f"{'#'*80}")

    messages_auto = [{"role": "user", "content": "What's the weather like in San Francisco?"}]
    print("\n-- tool_choice=auto --")
    for stream in [False, True]:
        make_request(messages_auto, tools=TOOLS, stream=stream, tool_choice="auto")
        time.sleep(0.5)

    messages_required = [{"role": "user", "content": "Briefly tell me the weather in San Francisco."}]
    print("\n-- tool_choice=required --")
    for stream in [False, True]:
        make_request(messages_required, tools=TOOLS, stream=stream, tool_choice="required")
        time.sleep(0.5)

    messages_targeted = [{"role": "user", "content": "Calculate 15 * 23."}]
    targeted_choice = {"type": "function", "function": {"name": "calculate"}}
    print("\n-- tool_choice={type:function, name:calculate} --")
    for stream in [False, True]:
        make_request(messages_targeted, tools=TOOLS, stream=stream, tool_choice=targeted_choice)
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

    print(f"\n{'#'*80}")
    print("ALL TESTS COMPLETED")
    print(f"{'#'*80}")


if __name__ == "__main__":
    main()


