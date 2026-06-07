# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.structured_schema_bounds import bound_json_schema_for_constrained_decoding


def test_strict_object_properties_preserves_explicit_additional_properties():
    schema = {
        "type": "object",
        "properties": {
            "known": {"type": "string"},
            "open": {
                "type": "object",
                "additionalProperties": True,
            },
        },
    }

    bounded = bound_json_schema_for_constrained_decoding(
        schema,
        strict_object_properties=True,
    )

    assert bounded["additionalProperties"] is False
    assert bounded["properties"]["open"]["additionalProperties"] is True


def test_strict_object_properties_recurses_into_nested_tool_objects():
    schema = {
        "type": "object",
        "properties": {
            "filters": {
                "type": "object",
                "properties": {
                    "date_range": {
                        "type": "object",
                        "properties": {
                            "from": {"type": "string"},
                            "to": {"type": "string"},
                        },
                    },
                    "metadata": {"type": "object"},
                },
            },
        },
    }

    bounded = bound_json_schema_for_constrained_decoding(
        schema,
        strict_object_properties=True,
    )

    filters = bounded["properties"]["filters"]
    date_range = filters["properties"]["date_range"]
    metadata = filters["properties"]["metadata"]
    assert bounded["additionalProperties"] is False
    assert filters["additionalProperties"] is False
    assert date_range["additionalProperties"] is False
    assert "additionalProperties" not in metadata


def test_strict_object_properties_can_skip_root_object():
    schema = {
        "type": "object",
        "properties": {
            "filters": {
                "type": "object",
                "properties": {
                    "tags": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    }

    bounded = bound_json_schema_for_constrained_decoding(
        schema,
        strict_object_properties=True,
        strict_object_properties_min_depth=1,
    )

    assert "additionalProperties" not in bounded
    assert bounded["properties"]["filters"]["additionalProperties"] is False
