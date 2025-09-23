#!/usr/bin/env python3
"""
Script to test SSE assembly and see the complete response structure
for Nemotron reasoning parser debugging.
"""

import json
import requests
import argparse
from typing import Dict, List, Any, Tuple

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

def collect_sse_response(url: str, payload: Dict[str, Any], test_name: str) -> Tuple[Dict[str, Any], TestResult]:
    """Collect all SSE events and assemble the complete response."""
    
    result = TestResult(test_name)
    
    print(f"\n🔄 {test_name}")
    print(f"Making request to: {url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    print("=" * 80)
    
    try:
        response = requests.post(url, json=payload, stream=True, timeout=60)
        response.raise_for_status()
    except Exception as e:
        result.add_error(f"Request failed: {e}")
        return {}, result
    
    # Collect all SSE events
    sse_events = []
    assembled_content = ""
    assembled_reasoning_content = ""
    json_decode_errors = 0
    
    print("SSE EVENTS:")
    print("-" * 40)
    
    for line_num, line in enumerate(response.iter_lines(), 1):
        if line:
            line = line.decode('utf-8')
            if line.startswith('data: '):
                data_str = line[6:]
                if data_str.strip() == '[DONE]':
                    print(f"[{line_num:3d}] DONE")
                    break
                    
                try:
                    data = json.loads(data_str)
                    sse_events.append(data)
                    
                    # Extract delta content
                    if 'choices' in data and len(data['choices']) > 0:
                        delta = data['choices'][0].get('delta', {})
                        
                        # Show the SSE event
                        event_summary = f"[{line_num:3d}] "
                        if 'role' in delta:
                            event_summary += f"role='{delta['role']}' "
                        if 'content' in delta:
                            content = delta['content']
                            assembled_content += content
                            event_summary += f"content='{content}' (repr: {repr(content)})"
                        if 'reasoning_content' in delta:
                            reasoning = delta['reasoning_content']
                            assembled_reasoning_content += reasoning
                            event_summary += f"reasoning_content='{reasoning}' (repr: {repr(reasoning)})"
                        if 'finish_reason' in delta:
                            event_summary += f"finish_reason='{delta['finish_reason']}'"
                            
                        # print(event_summary)
                        
                except json.JSONDecodeError as e:
                    print(f"[{line_num:3d}] JSON decode error: {e}")
                    json_decode_errors += 1
                    result.add_warning(f"JSON decode error on line {line_num}: {e}")
                    continue
    
    print("-" * 40)
    print(f"Total SSE events: {len(sse_events)}")
    if json_decode_errors > 0:
        print(f"⚠️  JSON decode errors: {json_decode_errors}")
    print()
    
    # Show assembled response
    print("ASSEMBLED RESPONSE:")
    print("=" * 80)
    print(f"Content length: {len(assembled_content)}")
    print(f"Content: {repr(assembled_content)}")
    print()
    print(f"Reasoning content length: {len(assembled_reasoning_content)}")
    print(f"Reasoning content: {repr(assembled_reasoning_content)}")
    print()
    
    # Show final assembled message structure
    final_message = {
        "role": "assistant",
        "content": assembled_content if assembled_content else None,
        "reasoning_content": assembled_reasoning_content if assembled_reasoning_content else None
    }
    
    print("FINAL MESSAGE STRUCTURE:")
    print("=" * 80)
    print(json.dumps(final_message, indent=2))
    
    # Validate the response
    validate_sse_response(payload, sse_events, assembled_content, assembled_reasoning_content, result)
    
    response_data = {
        "sse_events": sse_events,
        "assembled_content": assembled_content,
        "assembled_reasoning_content": assembled_reasoning_content,
        "final_message": final_message,
        "json_decode_errors": json_decode_errors
    }
    
    result.details = response_data
    return response_data, result

def validate_sse_response(payload: Dict[str, Any], sse_events: List[Dict], 
                         assembled_content: str, assembled_reasoning_content: str, 
                         result: TestResult) -> None:
    """Validate SSE response based on expected behavior."""
    
    thinking_mode = None
    for msg in payload.get("messages", []):
        if msg.get("role") == "system":
            content = msg.get("content", "")
            if "detailed thinking on" in content:
                thinking_mode = "on"
            elif "detailed thinking off" in content:
                thinking_mode = "off"
            break
    
    # Basic validations
    if len(sse_events) == 0:
        result.add_error("No SSE events received")
        return
    
    if not assembled_content and not assembled_reasoning_content:
        result.add_error("No content assembled from SSE events")
        return
    
    # Thinking mode specific validations
    if thinking_mode == "on":
        if not assembled_reasoning_content:
            result.add_warning("Expected reasoning_content with 'detailed thinking on' but got none")
        else:
            print(f"✅ Reasoning content present ({len(assembled_reasoning_content)} chars)")
    
    elif thinking_mode == "off":
        if assembled_reasoning_content:
            result.add_warning(f"Got reasoning_content with 'detailed thinking off': {len(assembled_reasoning_content)} chars")
        else:
            print("✅ No reasoning content (as expected with thinking off)")
    
    # Content validation
    if assembled_content:
        print(f"✅ Response content present ({len(assembled_content)} chars)")
    else:
        result.add_warning("No response content assembled")
    
    # SSE structure validation
    has_role_event = any('role' in event for event in sse_events if isinstance(event, dict))
    has_content_event = any('content' in event for event in sse_events if isinstance(event, dict))
    has_finish_event = any('finish_reason' in event for event in sse_events if isinstance(event, dict))
    
    if not has_role_event:
        result.add_warning("No role event found in SSE stream")
    if not has_content_event and not assembled_reasoning_content:
        result.add_warning("No content events found in SSE stream")
    if not has_finish_event:
        result.add_warning("No finish_reason event found in SSE stream")
    
    # Set overall pass/fail
    result.set_passed(len(result.errors) == 0)

def test_simple_query():
    """Test simple query with detailed thinking on."""
    payload = {
        "model": "nemotron_ultra",
        "messages": [
            {"role": "system", "content": "detailed thinking on"},
            {"role": "user", "content": "Hello!"}
        ],
        "temperature": 0.6,
        "max_tokens": 1000,
        "stream": True
    }
    
    response_data, test_result = collect_sse_response(
        "http://localhost:8003/v1/chat/completions", 
        payload, 
        "Simple Query (thinking on)"
    )
    test_results.append(test_result)
    return response_data

def test_complex_query():
    """Test complex query with detailed thinking on."""
    payload = {
        "model": "nemotron_ultra",
        "messages": [
            {"role": "system", "content": "detailed thinking on"},
            {"role": "user", "content": "Solve sin(x)cos(x) = 1"}
        ],
        "temperature": 0.6,
        "max_tokens": 2000,
        "stream": True
    }
    
    response_data, test_result = collect_sse_response(
        "http://localhost:8003/v1/chat/completions", 
        payload, 
        "Complex Query (thinking on)"
    )
    test_results.append(test_result)
    return response_data

def test_thinking_off():
    """Test with detailed thinking off."""
    payload = {
        "model": "nemotron_ultra",
        "messages": [
            {"role": "system", "content": "detailed thinking off"},
            {"role": "user", "content": "Hello!"}
        ],
        "temperature": 0.6,
        "max_tokens": 1000,
        "stream": True
    }
    
    response_data, test_result = collect_sse_response(
        "http://localhost:8003/v1/chat/completions", 
        payload, 
        "Simple Query (thinking off)"
    )
    test_results.append(test_result)
    return response_data

def print_test_summary():
    """Print a summary of all test results."""
    print("\n" + "=" * 80)
    print("🧪 TEST SUMMARY")
    print("=" * 80)
    
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
    
    print("\n" + "-" * 80)
    print(f"Total: {len(test_results)} tests, {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed!")
    else:
        print(f"💥 {failed} test(s) failed")
    
    return failed == 0

def main():
    parser = argparse.ArgumentParser(description="Test SSE assembly for Nemotron reasoning parser")
    parser.add_argument("--test", choices=["simple", "complex", "off", "all"], default="all",
                       help="Which test to run")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    args = parser.parse_args()
    
    print("🚀 Starting Nemotron SSE Assembly Tests")
    print("=" * 80)
    
    try:
        if args.test in ["simple", "all"]:
            test_simple_query()
            
        if args.test in ["complex", "all"]:
            test_complex_query()
            
        if args.test in ["off", "all"]:
            test_thinking_off()
            
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1
    
    # Print summary
    success = print_test_summary()
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
