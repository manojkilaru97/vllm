# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
from typing import Any

_STRING_CONSTRAINT_KEYS = {
    "const",
    "enum",
    "format",
    "pattern",
}


def bound_json_schema_for_constrained_decoding(schema: Any) -> Any:
    """Normalize JSON schemas before constrained decoding.

    Caller-provided schemas are behavioral constraints. Do not invent length or
    array bounds here; preserve the user's schema except for empty string
    constraints that constrained-decoding backends treat poorly.
    """
    if isinstance(schema, list):
        return [bound_json_schema_for_constrained_decoding(item) for item in schema]
    if not isinstance(schema, dict):
        return schema

    bounded = copy.deepcopy(schema)
    for key in ("format", "pattern"):
        if bounded.get(key) == "":
            bounded.pop(key)

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


def json_schema_has_unconstrained_string_fields(
    schema: Any,
    *,
    treat_length_bounds_as_constraint: bool = False,
) -> bool:
    """Return true for schemas with unconstrained string fields.

    JSON grammars can only enforce syntax. With an unconstrained string schema,
    many semantically different strings are valid, so xgrammar can allow a
    valid-but-wrong continuation such as literal "n" where an escaped newline
    was intended. Length-only bounds are not semantic constraints; they cap the
    language but still leave arbitrary content valid. Strings constrained by
    enum/const/pattern/format stay on xgrammar. For non-tool response schemas,
    caller-provided length bounds are often enough to keep xgrammar from
    running away, while avoiding guidance's weaker handling of punctuation-heavy
    strings.
    """

    def has_string_type(obj: dict[str, Any]) -> bool:
        schema_type = obj.get("type")
        if isinstance(schema_type, list):
            return "string" in schema_type
        return schema_type == "string"

    def is_unconstrained_string(obj: dict[str, Any]) -> bool:
        if not has_string_type(obj):
            return False
        if treat_length_bounds_as_constraint and (
            "maxLength" in obj or "minLength" in obj
        ):
            return False
        pattern = obj.get("pattern")
        if isinstance(pattern, str) and ".*" in pattern:
            return True
        return not any(
            _has_semantic_string_constraint(obj, key)
            for key in _STRING_CONSTRAINT_KEYS
        )

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


def json_schema_should_use_guidance_for_unconstrained_strings(schema: Any) -> bool:
    """Return true when guidance is safer than xgrammar for string-heavy schemas.

    xgrammar can choose valid-but-wrong arbitrary string continuations for
    simple payload schemas. Guidance avoids that, but currently handles some
    composition-heavy JSON schemas too loosely and can allow immediate EOS.
    Prefer xgrammar for non-tool schemas that rely on composition keywords.
    """
    if _is_tool_call_array_schema(schema):
        # Required tool_choice uses a tool-call array schema. Keep this on
        # xgrammar when possible: guidance is significantly slower for large
        # tool catalogs and can produce malformed argument JSON under complex
        # anyOf tool arrays.
        return False
    if not json_schema_has_unconstrained_string_fields(
        schema, treat_length_bounds_as_constraint=True
    ):
        return False
    return not _has_composition_keywords(schema)


def _has_semantic_string_constraint(obj: dict[str, Any], key: str) -> bool:
    if key not in obj:
        return False
    value = obj[key]
    if key in {"format", "pattern"}:
        return isinstance(value, str) and bool(value)
    return True


def _has_composition_keywords(schema: Any) -> bool:
    if isinstance(schema, list):
        return any(_has_composition_keywords(item) for item in schema)
    if not isinstance(schema, dict):
        return False
    if any(key in schema for key in ("anyOf", "oneOf", "allOf", "not")):
        return True
    return any(_has_composition_keywords(value) for value in schema.values())


def _is_tool_call_array_schema(schema: Any) -> bool:
    if not isinstance(schema, dict) or schema.get("type") != "array":
        return False
    items = schema.get("items")
    if not isinstance(items, dict):
        return False
    candidates = items.get("anyOf")
    if not isinstance(candidates, list):
        return False
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        properties = candidate.get("properties")
        required = candidate.get("required")
        if (
            isinstance(properties, dict)
            and "name" in properties
            and "parameters" in properties
            and isinstance(required, list)
            and {"name", "parameters"}.issubset(required)
        ):
            return True
    return False
