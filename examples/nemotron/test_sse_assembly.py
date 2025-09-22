#!/usr/bin/env python3
"""
Script to test SSE assembly and see the complete response structure
for Nemotron reasoning parser debugging.
"""

import json
import requests
import argparse
from typing import Dict, List, Any

def collect_sse_response(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Collect all SSE events and assemble the complete response."""
    
    print(f"Making request to: {url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    print("=" * 80)
    
    response = requests.post(url, json=payload, stream=True, timeout=60)
    response.raise_for_status()
    
    # Collect all SSE events
    sse_events = []
    assembled_content = ""
    assembled_reasoning_content = ""
    
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
                    continue
    
    print("-" * 40)
    print(f"Total SSE events: {len(sse_events)}")
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
    
    return {
        "sse_events": sse_events,
        "assembled_content": assembled_content,
        "assembled_reasoning_content": assembled_reasoning_content,
        "final_message": final_message
    }

def test_simple_query():
    """Test simple query with detailed thinking on."""
    print("🧪 TESTING SIMPLE QUERY: 'Hello!' with 'detailed thinking on'")
    print("=" * 80)
    
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
    
    return collect_sse_response("http://localhost:8003/v1/chat/completions", payload)

def test_complex_query():
    """Test complex query with detailed thinking on."""
    print("\n\n🧪 TESTING COMPLEX QUERY: Math problem with 'detailed thinking on'")
    print("=" * 80)
    
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
    
    return collect_sse_response("http://localhost:8003/v1/chat/completions", payload)

def test_thinking_off():
    """Test with detailed thinking off."""
    print("\n\n🧪 TESTING 'detailed thinking off'")
    print("=" * 80)
    
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
    
    return collect_sse_response("http://localhost:8003/v1/chat/completions", payload)

def main():
    parser = argparse.ArgumentParser(description="Test SSE assembly for Nemotron reasoning parser")
    parser.add_argument("--test", choices=["simple", "complex", "off", "all"], default="all",
                       help="Which test to run")
    args = parser.parse_args()
    
    try:
        if args.test in ["simple", "all"]:
            test_simple_query()
            
        if args.test in ["complex", "all"]:
            test_complex_query()
            
        if args.test in ["off", "all"]:
            test_thinking_off()
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    print("\n✅ All tests completed!")
    return 0

if __name__ == "__main__":
    exit(main())
