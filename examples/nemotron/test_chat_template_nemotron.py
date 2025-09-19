#!/usr/bin/env python3
"""
Unit test to compare chat template behaviors between Nemotron templates:
1) Tokenizer chat_template (from tokenizer_config.json)
2) Jinja file chat template (llama_nemotron_ultra_generic_tool_calling.jinja)

We test only non-tool scenarios and toggle "thinking" behavior via system prompts:
  - detailed thinking on
  - detailed thinking off

For each scenario we set add_generation_prompt=True and assert identical outputs.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional

from jinja2 import Environment, FileSystemLoader


JINJA_TEMPLATE_PATH = \
    "/home/scratch.mkilaru_coreai/Llama-3_1-Nemotron-Ultra-253B-v1-FP8/llama_nemotron_ultra_generic_tool_calling.jinja"
TOKENIZER_CONFIG_PATH = \
    "/home/scratch.mkilaru_coreai/Llama-3_1-Nemotron-Ultra-253B-v1-FP8/tokenizer_config.json"


def load_file_template(template_path: str):
    template_dir = Path(template_path).parent
    template_name = Path(template_path).name
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    return env, env.get_template(template_name)


def load_tokenizer_template_and_bos(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    chat_template_str = cfg["chat_template"]
    bos_token = cfg.get("bos_token", "")
    env = Environment()
    template = env.from_string(chat_template_str)
    return env, template, bos_token


def render_with_thinking(env, template, messages: List[Dict], thinking_mode: Optional[str], bos_token: str,
                         extra_system_prompt: Optional[str] = None,
                         add_generation_prompt: bool = True) -> str:
    # Build system message content
    system_parts: List[str] = []
    if thinking_mode in ("on", "off"):
        system_parts.append("detailed thinking on" if thinking_mode == "on" else "detailed thinking off")
    if extra_system_prompt:
        system_parts.append(extra_system_prompt)

    # Prepend system message if any content
    if system_parts:
        sys_msg = {"role": "system", "content": "\n\n".join(part.strip() for part in system_parts if part is not None and part != "")}
        render_messages = [sys_msg] + messages
    else:
        render_messages = list(messages)

    context = {
        "messages": render_messages,
        "add_generation_prompt": add_generation_prompt,
        "bos_token": bos_token,
        "tools": None,  # Explicitly no tools
    }
    return template.render(**context)


def test_scenario(name: str,
                  tok_env, tok_tmpl, j_env, j_tmpl,
                  bos_token: str,
                  messages: List[Dict],
                  thinking_mode: Optional[str] = None,
                  extra_system_prompt: Optional[str] = None):
    print(f"\n{'='*60}")
    print(f"SCENARIO: {name}")
    print(f"{'='*60}")

    print(f"Messages: {json.dumps(messages, indent=2)}")
    if thinking_mode:
        print(f"Thinking mode: {thinking_mode}")
    if extra_system_prompt:
        print(f"Extra system: {extra_system_prompt}")
    print("add_generation_prompt: True")

    print(f"\n{'-'*30} TOKENIZER TEMPLATE {'-'*30}")
    tok_result = render_with_thinking(tok_env, tok_tmpl, messages, thinking_mode, bos_token, extra_system_prompt)
    print(repr(tok_result))

    print(f"\n{'-'*30} JINJA FILE TEMPLATE {'-'*30}")
    j_result = render_with_thinking(j_env, j_tmpl, messages, thinking_mode, bos_token, extra_system_prompt)
    print(repr(j_result))

    assert tok_result == j_result, (
        f"Templates differ for scenario '{name}':\n"
        f"Tokenizer: {repr(tok_result)}\n"
        f"Jinja:     {repr(j_result)}"
    )
    print("\n✅ ASSERTION PASSED: Templates produce identical output")
    print(f"\n{'-'*60}")


def main():
    # Load templates
    j_env, j_template = load_file_template(JINJA_TEMPLATE_PATH)
    tok_env, tok_template, bos_token = load_tokenizer_template_and_bos(TOKENIZER_CONFIG_PATH)

    # Messages
    simple_messages = [
        {"role": "user", "content": "What is 2 + 2?"}
    ]

    multi_turn_messages = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you!"},
        {"role": "user", "content": "Can you help me with math?"}
    ]

    # Scenarios (all add_generation_prompt=True)
    test_scenario(
        "Simple User Query - Thinking ON",
        tok_env, tok_template, j_env, j_template,
        bos_token,
        simple_messages,
        thinking_mode="on"
    )

    test_scenario(
        "Simple User Query - Thinking OFF",
        tok_env, tok_template, j_env, j_template,
        bos_token,
        simple_messages,
        thinking_mode="off"
    )

    test_scenario(
        "With System Prompt - Thinking ON",
        tok_env, tok_template, j_env, j_template,
        bos_token,
        simple_messages,
        thinking_mode="on",
        extra_system_prompt="You are a helpful math tutor."
    )

    test_scenario(
        "With System Prompt - Thinking OFF",
        tok_env, tok_template, j_env, j_template,
        bos_token,
        simple_messages,
        thinking_mode="off",
        extra_system_prompt="You are a helpful math tutor."
    )

    test_scenario(
        "Multi-turn Conversation - Thinking ON",
        tok_env, tok_template, j_env, j_template,
        bos_token,
        multi_turn_messages,
        thinking_mode="on"
    )

    test_scenario(
        "Multi-turn Conversation - Thinking OFF",
        tok_env, tok_template, j_env, j_template,
        bos_token,
        multi_turn_messages,
        thinking_mode="off"
    )

    test_scenario(
        "Multi-turn with Extra System - Thinking ON",
        tok_env, tok_template, j_env, j_template,
        bos_token,
        multi_turn_messages,
        thinking_mode="on",
        extra_system_prompt="You are a helpful assistant that thinks step by step."
    )

    print(f"\n{'='*60}")
    print("🎉 ALL TESTS PASSED! Templates produce identical output for all non-tool scenarios.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()


