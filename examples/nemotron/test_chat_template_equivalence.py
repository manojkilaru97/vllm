#!/usr/bin/env python3
"""
Test script to compare chat template behaviors between:
1. Original model provider chat template (from tokenizer_config.json)
2. Nemotron generic tool-calling chat template (Jinja file)

Scenarios (non-tool only):
- Non-thinking requests (system: "detailed thinking off")
- Thinking requests (system: "detailed thinking on")
- With system prompts (prepend with thinking directive)
- All with add_generation_prompt=True

The assertion is strict: token-for-token identical output for non-tool scenarios.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional

from jinja2 import Environment, FileSystemLoader


TOKENIZER_JSON_PATH = \
    "/home/scratch.mkilaru_coreai/Llama-3_1-Nemotron-Ultra-253B-v1-FP8/tokenizer_config.json"
NEMOTRON_JINJA_PATH = \
    "/home/scratch.mkilaru_coreai/Llama-3_1-Nemotron-Ultra-253B-v1-FP8/llama_nemotron_ultra_generic_tool_calling.jinja"


def load_tokenizer_template(tokenizer_json_path: str):
    """Load chat_template string and bos_token from tokenizer_config.json."""
    with open(tokenizer_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    chat_template = data["chat_template"]
    bos_token = data.get("bos_token", "")
    env = Environment()
    template = env.from_string(chat_template)
    return template, bos_token


def load_file_template(template_path: str):
    """Load a Jinja2 template from file path."""
    template_dir = str(Path(template_path).parent)
    template_name = str(Path(template_path).name)
    env = Environment(loader=FileSystemLoader(template_dir))
    return env.get_template(template_name)


def render_template(template,
                    messages: List[Dict[str, str]],
                    bos_token: str,
                    system_prompt: Optional[str] = None,
                    add_generation_prompt: bool = True):
    """Render a template with given parameters for non-tool scenarios."""
    # Prepend system if provided (thinking control is via system message)
    if system_prompt is not None:
        messages = [{"role": "system", "content": system_prompt}] + messages

    context = {
        "messages": messages,
        "add_generation_prompt": add_generation_prompt,
        "bos_token": bos_token,
        "tools": None,  # No tools for these tests
    }
    return template.render(**context)


def test_scenario(name: str,
                  tokenizer_template,
                  nemotron_template,
                  bos_token: str,
                  messages: List[Dict[str, str]],
                  system_prompt: Optional[str] = None):
    print(f"\n{'='*60}")
    print(f"SCENARIO: {name}")
    print(f"{'='*60}")

    print(f"Messages: {json.dumps(messages, indent=2)}")
    if system_prompt:
        print(f"System Prompt: {system_prompt!r}")
    print(f"add_generation_prompt: True")

    print(f"\n{'-'*30} TOKENIZER TEMPLATE {'-'*30}")
    original_result = render_template(tokenizer_template, messages, bos_token, system_prompt)
    print(repr(original_result))

    print(f"\n{'-'*30} NEMOTRON TEMPLATE {'-'*30}")
    combined_result = render_template(nemotron_template, messages, bos_token, system_prompt)
    print(repr(combined_result))

    assert original_result == combined_result, (
        f"Templates differ for scenario '{name}':\n"
        f"Original: {repr(original_result)}\n"
        f"Nemotron: {repr(combined_result)}"
    )
    print(f"\n✅ ASSERTION PASSED: Templates produce identical output")
    print(f"\n{'-'*60}")


def main():
    tokenizer_template, bos_token = load_tokenizer_template(TOKENIZER_JSON_PATH)
    nemotron_template = load_file_template(NEMOTRON_JINJA_PATH)

    # Test messages
    simple_messages = [
        {"role": "user", "content": "What is 2 + 2?"}
    ]

    multi_turn_messages = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you!"},
        {"role": "user", "content": "Can you help me with math?"}
    ]

    # Scenarios (use system prompt to control thinking on/off per Nemotron)
    test_scenario(
        "Simple User Query - Non-thinking (system: off)",
        tokenizer_template, nemotron_template, bos_token,
        simple_messages,
        system_prompt="detailed thinking off",
    )

    test_scenario(
        "Simple User Query - Thinking (system: on)",
        tokenizer_template, nemotron_template, bos_token,
        simple_messages,
        system_prompt="detailed thinking on",
    )

    test_scenario(
        "With System Prompt - Non-thinking", 
        tokenizer_template, nemotron_template, bos_token,
        simple_messages,
        system_prompt="detailed thinking off\nYou are a helpful math tutor.",
    )

    test_scenario(
        "With System Prompt - Thinking",
        tokenizer_template, nemotron_template, bos_token,
        simple_messages,
        system_prompt="detailed thinking on\nYou are a helpful math tutor.",
    )

    test_scenario(
        "Multi-turn Conversation - Non-thinking",
        tokenizer_template, nemotron_template, bos_token,
        multi_turn_messages,
        system_prompt="detailed thinking off",
    )

    test_scenario(
        "Multi-turn Conversation - Thinking", 
        tokenizer_template, nemotron_template, bos_token,
        multi_turn_messages,
        system_prompt="detailed thinking on",
    )

    test_scenario(
        "Multi-turn with System - Thinking",
        tokenizer_template, nemotron_template, bos_token,
        multi_turn_messages, 
        system_prompt=(
            "detailed thinking on\n"
            "You are a helpful assistant that thinks step by step."
        ),
    )

    print(f"\n{'='*60}")
    print("🎉 ALL TESTS PASSED! Templates produce identical output for all non-tool scenarios.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
