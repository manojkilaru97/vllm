import json

from vllm.v1.structured_output.backend_xgrammar import _is_tool_call_array_schema


def test_detects_tool_call_array_schema():
    schema = {
        "type": "array",
        "items": {
            "anyOf": [
                {
                    "type": "object",
                    "properties": {
                        "name": {"const": "get_weather"},
                        "parameters": {
                            "type": "object",
                            "properties": {"location": {"type": "string"}},
                        },
                    },
                    "required": ["name", "parameters"],
                }
            ]
        },
    }

    assert _is_tool_call_array_schema(json.dumps(schema))


def test_does_not_detect_ordinary_array_schema_as_tool_calls():
    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "value": {"type": "string"},
            },
        },
    }

    assert not _is_tool_call_array_schema(json.dumps(schema))
