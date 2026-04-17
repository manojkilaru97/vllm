# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.utils.import_utils import LazyLoader
from vllm.v1.structured_output.backend_types import (
    StructuredOutputBackend,
    StructuredOutputGrammar,
    StructuredOutputOptions,
)
from vllm.v1.structured_output.request import get_structured_output_key

if TYPE_CHECKING:
    import llguidance
    import llguidance.hf as llguidance_hf
    import llguidance.torch as llguidance_torch
else:
    llguidance = LazyLoader("llguidance", globals(), "llguidance")
    llguidance_hf = LazyLoader("llguidance.hf", globals(), "llguidance.hf")
    llguidance_torch = LazyLoader("llguidance.torch", globals(), "llguidance.torch")

logger = init_logger(__name__)


def _walk_json_for_additional_properties(
    data: object,
    *,
    within_allof: bool = False,
):
    if isinstance(data, dict):
        current_within_allof = within_allof or isinstance(data.get("allOf"), list)
        for key, value in data.items():
            child_within_allof = current_within_allof or key == "allOf"
            _walk_json_for_additional_properties(
                value,
                within_allof=child_within_allof,
            )

        properties = data.get("properties")
        pattern_properties = data.get("patternProperties")
        required = data.get("required")

        has_properties = isinstance(properties, dict)
        has_pattern_properties = isinstance(pattern_properties, dict)
        if not has_properties and not has_pattern_properties:
            return

        if "additionalProperties" in data:
            return
        if current_within_allof:
            return
        if any(
            isinstance(data.get(keyword), list)
            for keyword in ("allOf", "anyOf", "oneOf")
        ):
            return
        if "$ref" in data:
            return

        property_keys = set(properties.keys()) if has_properties else set()
        if has_properties and not property_keys and not has_pattern_properties:
            return
        if isinstance(required, list) and any(
            isinstance(name, str) and name not in property_keys for name in required
        ):
            return

        if has_properties or has_pattern_properties:
            data["additionalProperties"] = False
    elif isinstance(data, list):
        for item in data:
            _walk_json_for_additional_properties(
                item,
                within_allof=within_allof,
            )


def _walk_json_for_oneof_disambiguation(data: object):
    if isinstance(data, dict):
        one_of = data.get("oneOf")
        if isinstance(one_of, list):
            branch_props: list[set[str]] = []
            for branch in one_of:
                if isinstance(branch, dict):
                    props = branch.get("properties")
                    if isinstance(props, dict):
                        branch_props.append(set(props.keys()))
                    else:
                        branch_props.append(set())
                else:
                    branch_props.append(set())

            for idx, branch in enumerate(one_of):
                if not isinstance(branch, dict):
                    continue
                props = branch.get("properties")
                if not isinstance(props, dict) or branch.get("required"):
                    continue
                other_props = set().union(
                    *(branch_props[j] for j in range(len(branch_props)) if j != idx)
                )
                unique_props = sorted(branch_props[idx] - other_props)
                # Guidance's oneOf coercion approximates oneOf as anyOf. For
                # simple object branches, require at least one branch-unique
                # property so {} does not satisfy every branch.
                if unique_props:
                    branch["required"] = unique_props
                    branch.setdefault("additionalProperties", False)

        for value in data.values():
            _walk_json_for_oneof_disambiguation(value)
    elif isinstance(data, list):
        for item in data:
            _walk_json_for_oneof_disambiguation(item)


def _rewrite_simple_not_types(data: object):
    if isinstance(data, dict):
        not_schema = data.get("not")
        not_type = (
            not_schema.get("type")
            if isinstance(not_schema, dict)
            else None
        )
        if isinstance(not_type, str):
            if not_type == "number":
                data.pop("not", None)
                data["anyOf"] = [
                    {"type": "string"},
                    {"type": "boolean"},
                    {"type": "object"},
                    {"type": "array"},
                    {"type": "null"},
                ]
            elif not_type == "integer":
                data.pop("not", None)
                data["anyOf"] = [
                    {"type": "string"},
                    {"type": "boolean"},
                    {"type": "object"},
                    {"type": "array"},
                    {"type": "null"},
                    {"type": "number", "minimum": 0, "exclusiveMinimum": True},
                    {"type": "number", "maximum": 0, "exclusiveMaximum": True},
                ]

        for value in data.values():
            _rewrite_simple_not_types(value)
    elif isinstance(data, list):
        for item in data:
            _rewrite_simple_not_types(item)


def has_guidance_unsupported_json_features(schema: dict[str, Any]) -> bool:
    """Check if JSON schema contains features unsupported by guidance/llguidance."""

    def check_object(obj: dict[str, Any]) -> bool:
        if not isinstance(obj, dict):
            return False

        # patternProperties is not supported by llguidance
        if "patternProperties" in obj:
            return True

        # Recursively check all nested objects and arrays
        for value in obj.values():
            if isinstance(value, dict):
                if check_object(value):
                    return True
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) and check_object(item):
                        return True

        return False

    return check_object(schema)


def process_for_additional_properties(
    guide_json: str | dict[str, Any],
) -> dict[str, Any]:
    if isinstance(guide_json, str):
        guide_json_obj = json.loads(guide_json)
    else:
        # copy for modifications
        guide_json_obj = copy.deepcopy(guide_json)
    _walk_json_for_additional_properties(guide_json_obj)
    return guide_json_obj


def process_for_oneof_disambiguation(
    guide_json: str | dict[str, Any],
) -> dict[str, Any]:
    if isinstance(guide_json, str):
        guide_json_obj = json.loads(guide_json)
    else:
        guide_json_obj = copy.deepcopy(guide_json)
    _walk_json_for_oneof_disambiguation(guide_json_obj)
    return guide_json_obj


def process_for_simple_not_types(
    guide_json: str | dict[str, Any],
) -> dict[str, Any]:
    if isinstance(guide_json, str):
        guide_json_obj = json.loads(guide_json)
    else:
        guide_json_obj = copy.deepcopy(guide_json)
    _rewrite_simple_not_types(guide_json_obj)
    return guide_json_obj


@dataclass
class GuidanceBackend(StructuredOutputBackend):
    def __post_init__(self):
        self.disable_any_whitespace = (
            self.vllm_config.structured_outputs_config.disable_any_whitespace
        )
        self.disable_additional_properties = (
            self.vllm_config.structured_outputs_config.disable_additional_properties
        )

        self.ll_tokenizer = llguidance_hf.from_tokenizer(
            self.tokenizer, max(self.vocab_size, len(self.tokenizer))
        )

    def compile_grammar(
        self,
        request_type: StructuredOutputOptions,
        grammar_spec: str,
        params: StructuredOutputsParams | None = None,
    ) -> StructuredOutputGrammar:
        disable_any_whitespace = self.disable_any_whitespace
        disable_additional_properties = self.disable_additional_properties
        if params is not None and params.disable_any_whitespace:
            disable_any_whitespace = True
        if params is not None and params.disable_additional_properties:
            disable_additional_properties = True

        last_error: Exception | None = None
        for lenient in (False, True):
            try:
                self.serialized_grammar = serialize_guidance_grammar(
                    request_type,
                    grammar_spec,
                    disable_any_whitespace,
                    disable_additional_properties,
                    lenient=lenient,
                )

                ll_matcher = llguidance.LLMatcher(
                    self.ll_tokenizer,
                    self.serialized_grammar,
                    log_level=int(os.environ.get("LLGUIDANCE_LOG_LEVEL", "1")),
                )

                r = GuidanceGrammar(
                    ll_matcher=ll_matcher,
                    ll_tokenizer=self.ll_tokenizer,
                    vocab_size=self.vocab_size,
                )

                err = r.ll_matcher.get_error()
                if err:
                    raise ValueError(f"Grammar error: {err}")
                return r
            except Exception as exc:
                last_error = exc
                if not lenient:
                    logger.warning(
                        "Strict guidance grammar compile failed; "
                        "retrying with lenient guidance options.",
                        exc_info=True,
                    )
                    continue
                raise

        assert last_error is not None
        raise last_error

    def allocate_token_bitmask(self, max_num_seqs: int):
        return llguidance_torch.allocate_token_bitmask(
            max_num_seqs, self.ll_tokenizer.vocab_size
        )

    def destroy(self):
        pass


@dataclass
class GuidanceGrammar(StructuredOutputGrammar):
    ll_matcher: llguidance.LLMatcher
    ll_tokenizer: llguidance.LLTokenizer
    vocab_size: int
    printed_error: bool = False
    terminated: bool = False
    rollback_lag: int = 0

    def check_error(self):
        if not self.printed_error:
            err = self.ll_matcher.get_error()
            if err:
                self.printed_error = True
                logger.warning("LLMatcher error: %s", err)

    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
        """Accepts a list of tokens and advances the parser.

        Returns True if the parser was advanced successfully.
        Returns False if the parser failed to advance.
        """

        if self.ll_tokenizer.eos_token in tokens:
            if self.ll_matcher.is_stopped() and not self.terminated:
                self.rollback_lag = 1
            self.terminated = True

        if self.ll_matcher.is_stopped():
            return True

        # TODO - Add jump decoding support in the future:
        # self.ll_matcher.compute_ff_bytes() - this should always work
        # self.ll_matcher.compute_ff_tokens() - this only works for
        #   "canonical" tokenizers
        # For conversion between the two, see
        # https://github.com/guidance-ai/llguidance/blob/main/docs/fast_forward.md

        r = self.ll_matcher.consume_tokens(tokens)

        self.check_error()

        return r

    def validate_tokens(self, tokens: list[int]) -> list[int]:
        """Checks if the list of tokens are accepted by the parser in sequence.
        Will not advance the parser.

        Returns the prefix list of tokens that are accepted by the parser.
        """
        if len(tokens) == 0:
            return []
        if self.ll_matcher.is_stopped():
            return []

        num_tokens = self.ll_matcher.validate_tokens(tokens)

        self.check_error()

        return tokens[:num_tokens]

    def rollback(self, num_tokens: int) -> None:
        if num_tokens > 0:
            self.ll_matcher.rollback(num_tokens - self.rollback_lag)
            self.terminated = False
            self.rollback_lag = 0
            self.check_error()

    def fill_bitmask(self, bitmask: torch.Tensor, idx: int) -> None:
        # this will automatically return [EOS] mask if the matcher is stopped
        # or otherwise in an error state
        llguidance_torch.fill_next_token_bitmask(self.ll_matcher, bitmask, idx)
        self.check_error()

    def is_terminated(self) -> bool:
        return self.terminated

    def reset(self):
        # This method may be not needed anymore? TODO
        self.ll_matcher.reset()


def serialize_guidance_grammar(
    request_type: StructuredOutputOptions,
    grammar_spec: str | dict[str, Any],
    disable_any_whitespace: bool = False,
    disable_additional_properties: bool = False,
    lenient: bool = False,
) -> str:
    def _process_schema(
        grammar_spec: str | dict[str, Any],
    ) -> str:
        if disable_additional_properties:
            grammar_spec = process_for_additional_properties(grammar_spec)
        grammar_spec = process_for_oneof_disambiguation(grammar_spec)
        grammar_spec = process_for_simple_not_types(grammar_spec)
        return llguidance.LLMatcher.grammar_from_json_schema(
            grammar_spec,
            defaults={
                "whitespace_flexible": not disable_any_whitespace,
                # llguidance rejects JSON Schema oneOf by default.
                # Approximate it as anyOf so common OpenAI-style request
                # schemas keep working instead of returning 400.
                "coerce_one_of": True,
                "lenient": lenient,
            },
        )

    if request_type == StructuredOutputOptions.JSON:
        return _process_schema(grammar_spec)
    elif request_type == StructuredOutputOptions.JSON_OBJECT:
        return llguidance.LLMatcher.grammar_from_json_schema(
            '{"type": "object"}',
            defaults={
                "whitespace_flexible": not disable_any_whitespace,
                "coerce_one_of": True,
                "lenient": lenient,
            },
        )
    else:
        if request_type == StructuredOutputOptions.REGEX:
            tp = "regex"
        elif request_type == StructuredOutputOptions.GRAMMAR:
            tp = "grammar"
        elif request_type == StructuredOutputOptions.CHOICE:
            tp = "choice"
        elif request_type == StructuredOutputOptions.STRUCTURAL_TAG:
            if isinstance(grammar_spec, str):
                s_tag = json.loads(grammar_spec)
            else:
                s_tag = grammar_spec
            triggers: list[str] = s_tag["triggers"]
            tags: list[llguidance.StructTag] = []
            for s in s_tag["structures"]:
                begin: str = s["begin"]
                trig = next((t for t in triggers if begin.startswith(t)), None)
                if trig is None:
                    raise ValueError(
                        f"Trigger {begin} not found in triggers {triggers}"
                    )
                tags.append(
                    llguidance.StructTag(
                        trigger=trig,
                        begin=s["begin"],
                        grammar=_process_schema(s["schema"]),
                        end=s["end"],
                    )
                )
            if not tags:
                raise ValueError("No structural tags found in the grammar spec.")
            return llguidance.StructTag.to_grammar(tags)
        else:
            logger.error(
                "Validation should have already occurred. Please file an issue."
            )
            raise ValueError(
                f"grammar is not of valid supported types. ({request_type!s})"
            )
        return llguidance.grammar_from(tp, grammar_spec)


def validate_guidance_grammar(
    sampling_params: SamplingParams, tokenizer: llguidance.LLTokenizer | None = None
) -> None:
    # if structured output is not enabled, there is nothing to validate
    if sampling_params.structured_outputs is None:
        return
    tp, grm = get_structured_output_key(sampling_params.structured_outputs)
    last_error: str | None = None
    for lenient in (False, True):
        guidance_grm = serialize_guidance_grammar(
            tp,
            grm,
            disable_any_whitespace=sampling_params.structured_outputs.disable_any_whitespace,
            disable_additional_properties=(
                sampling_params.structured_outputs.disable_additional_properties
            ),
            lenient=lenient,
        )
        err = llguidance.LLMatcher.validate_grammar(guidance_grm, tokenizer)
        if not err:
            return
        last_error = err
        if not lenient:
            logger.warning(
                "Strict guidance grammar validation failed; "
                "retrying with lenient guidance options: %s",
                err,
            )
            continue
        raise ValueError(f"Grammar error: {err}")

    if last_error:
        raise ValueError(f"Grammar error: {last_error}")
