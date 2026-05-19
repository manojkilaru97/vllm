# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
from typing import Any

DEFAULT_SCHEMA_MAX_STRING_LENGTH = 4096
DEFAULT_SCHEMA_MAX_ARRAY_ITEMS = 32
_STRING_CONSTRAINT_KEYS = {
    "const",
    "contentEncoding",
    "contentMediaType",
    "enum",
    "format",
    "maxLength",
    "minLength",
    "pattern",
}


def bound_json_schema_for_constrained_decoding(schema: Any) -> Any:
    """Add finite bounds to JSON schemas before constrained decoding.

    Unbounded strings and arrays are valid JSON Schema, but they create a very
    large language for grammar-constrained decoding and can let a model generate
    until max_tokens. Explicit caller-provided bounds are preserved.
    """
    if isinstance(schema, list):
        return [bound_json_schema_for_constrained_decoding(item) for item in schema]
    if not isinstance(schema, dict):
        return schema

    bounded = copy.deepcopy(schema)
    schema_type = bounded.get("type")
    schema_types = schema_type if isinstance(schema_type, list) else [schema_type]
    if "string" in schema_types and "maxLength" not in bounded:
        bounded["maxLength"] = DEFAULT_SCHEMA_MAX_STRING_LENGTH
    if "array" in schema_types and "maxItems" not in bounded:
        bounded["maxItems"] = DEFAULT_SCHEMA_MAX_ARRAY_ITEMS

    for key in ("properties", "$defs", "definitions"):
        if key in bounded:
            value = bounded[key]
            if isinstance(value, dict):
                bounded[key] = {
                    name: bound_json_schema_for_constrained_decoding(subschema)
                    for name, subschema in value.items()
                }

    for key in ("items", "additionalProperties", "anyOf", "oneOf", "allOf"):
        if key in bounded:
            bounded[key] = bound_json_schema_for_constrained_decoding(bounded[key])
    return bounded


def json_schema_has_unconstrained_string_fields(schema: Any) -> bool:
    """Return true for schemas with unconstrained string fields.

    JSON grammars can only enforce syntax. With an unconstrained string schema,
    many semantically different strings are valid, so xgrammar can allow a
    valid-but-wrong continuation such as literal "n" where an escaped newline
    was intended. Constrained strings with enum/const/pattern/format stay on
    xgrammar.
    """

    def has_string_type(obj: dict[str, Any]) -> bool:
        schema_type = obj.get("type")
        if isinstance(schema_type, list):
            return "string" in schema_type
        return schema_type == "string"

    def is_unconstrained_string(obj: dict[str, Any]) -> bool:
        if not has_string_type(obj):
            return False
        pattern = obj.get("pattern")
        if isinstance(pattern, str) and ".*" in pattern:
            return True
        return not any(key in obj for key in _STRING_CONSTRAINT_KEYS)

    def visit(obj: Any) -> bool:
        if isinstance(obj, list):
            return any(visit(item) for item in obj)
        if not isinstance(obj, dict):
            return False

        if is_unconstrained_string(obj):
            return True

        properties = obj.get("properties")
        if isinstance(properties, dict):
            for subschema in properties.values():
                if visit(subschema):
                    return True

        for key in ("items", "additionalProperties", "anyOf", "oneOf", "allOf"):
            if key in obj and visit(obj[key]):
                return True

        for key in ("$defs", "definitions"):
            defs = obj.get(key)
            if isinstance(defs, dict) and any(
                visit(subschema) for subschema in defs.values()
            ):
                return True

        return False

    return visit(schema)
