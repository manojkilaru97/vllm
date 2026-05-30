# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
import json
import multiprocessing
from collections.abc import Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.reasoning import ReasoningParserManager
from vllm.structured_schema_bounds import (
    json_schema_should_use_guidance_for_unconstrained_strings,
)
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.utils.import_utils import LazyLoader
from vllm.v1.structured_output.backend_guidance import (
    GuidanceBackend,
    has_guidance_unsupported_json_features,
)
from vllm.v1.structured_output.backend_types import (
    StructuredOutputBackend,
    StructuredOutputGrammar,
    StructuredOutputOptions,
)
from vllm.v1.structured_output.backend_xgrammar import XgrammarBackend

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    import torch

    from vllm.reasoning import ReasoningParser
    from vllm.v1.request import Request
else:
    torch = LazyLoader("torch", globals(), "torch")


logger = init_logger(__name__)


class StructuredOutputManager:
    """Engine-level manager for structured output requests."""

    def __init__(self, vllm_config: VllmConfig):
        self.backend: StructuredOutputBackend | None = None
        # We only store the class of the reasoner in the manager.
        # The parser instance is request-scoped because some reasoning parsers
        # depend on per-request chat-template kwargs.
        self.reasoner_cls: type[ReasoningParser] | None = None
        self.vllm_config = vllm_config

        # When in external_launcher mode, async grammar compilation causes deadlocks
        # due to external_launcher mode having a scheduler for each TP rank.
        # Async grammar compilation causes the
        # WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR → WAITING transition to
        # happen at different times on different TP ranks,
        # breaking the determinism assumption that external_launcher relies on.
        self._use_async_grammar_compilation = (
            vllm_config.parallel_config.distributed_executor_backend
            != "external_launcher"
        )

        self._grammar_bitmask: torch.Tensor | None = None
        self._full_mask = torch.tensor(-1, dtype=torch.int32)

        max_batch_size = self.vllm_config.scheduler_config.max_num_seqs
        self.fill_bitmask_parallel_threshold = 128
        if self.fill_bitmask_parallel_threshold < max_batch_size:
            self.fill_bitmask_parallel_batch_size = 16
            # Use:
            # - at least 1 CPU
            # - at most half the number of CPUs or 8, whichever is less
            max_workers = max(1, min(multiprocessing.cpu_count() // 2, 8))
            self.executor_for_fillmask = ThreadPoolExecutor(max_workers=max_workers)

        if not self.vllm_config.model_config.skip_tokenizer_init:
            # The default max_workers if not specified is the number of
            # CPUs * 5, which is way too high since these tasks are CPU-bound,
            # not I/O bound. We also know we would never dominate CPU usage
            # with just grammar compilation, so we set it to half the number
            # of CPUs.
            max_workers = max(1, (multiprocessing.cpu_count() + 1) // 2)
            self.executor = ThreadPoolExecutor(max_workers=max_workers)
            self.tokenizer = cached_tokenizer_from_config(
                model_config=self.vllm_config.model_config
            )
            reasoning_parser_plugin = (
                self.vllm_config.structured_outputs_config.reasoning_parser_plugin
            )
            if reasoning_parser_plugin and len(reasoning_parser_plugin) > 3:
                ReasoningParserManager.import_reasoning_parser(reasoning_parser_plugin)

            reasoning_parser = (
                self.vllm_config.structured_outputs_config.reasoning_parser
            )
            if reasoning_parser:
                self.reasoner_cls = ReasoningParserManager.get_reasoning_parser(
                    reasoning_parser
                )

        self.enable_in_reasoning = (
            self.vllm_config.structured_outputs_config.enable_in_reasoning
        )

    def _get_reasoner(self, request: "Request") -> "ReasoningParser | None":
        structured_req = request.structured_output_request
        if structured_req is None or self.reasoner_cls is None:
            return None

        if structured_req.reasoner is None:
            # Lazily build the request-local parser so the structured-output
            # gate observes the same template kwargs used by the frontend.
            parser_kwargs = structured_req.reasoning_parser_kwargs or {}
            structured_req.reasoner = self.reasoner_cls(
                tokenizer=self.tokenizer,
                **parser_kwargs,
            )
        return structured_req.reasoner

    def grammar_init(self, request: "Request") -> None:
        if request.structured_output_request is None:
            return

        if TYPE_CHECKING:
            assert (
                request.sampling_params is not None
                and request.sampling_params.structured_outputs is not None
            )

        # Initialize the backend the first time it is needed.
        #
        # NOTE: We only support a single backend. We do NOT support different
        # backends on a per-request basis in V1 (for now, anyway...).
        # _backend is set in Processor._validate_structured_output
        if self.backend is None:
            assert request.sampling_params is not None
            structured_outputs = request.sampling_params.structured_outputs
            backend = structured_outputs._backend
            backend_was_auto = structured_outputs._backend_was_auto
            vocab_size = self.vllm_config.model_config.get_vocab_size()
            if backend == "xgrammar" or (
                backend == "guidance" and backend_was_auto
            ):
                # Auto mode may pick guidance for string-heavy schemas, but
                # composition-heavy schemas should still use xgrammar later in
                # the same process. Keep xgrammar as the primary manager backend
                # and compile guidance per-request when auto selected it.
                self.backend = XgrammarBackend(
                    self.vllm_config,
                    tokenizer=self.tokenizer,
                    vocab_size=vocab_size,
                )
            elif backend == "guidance":
                self.backend = GuidanceBackend(
                    self.vllm_config,
                    tokenizer=self.tokenizer,
                    vocab_size=vocab_size,
                )
            elif backend == "outlines":
                from vllm.v1.structured_output.backend_outlines import OutlinesBackend

                self.backend = OutlinesBackend(
                    self.vllm_config,
                    tokenizer=self.tokenizer,
                    vocab_size=vocab_size,
                )
            elif backend == "lm-format-enforcer":
                from vllm.v1.structured_output.backend_lm_format_enforcer import (  # noqa: E501
                    LMFormatEnforcerBackend,
                )

                self.backend = LMFormatEnforcerBackend(
                    self.vllm_config,
                    tokenizer=self.tokenizer,
                    vocab_size=vocab_size,
                )
            else:
                raise ValueError(f"Unsupported structured output backend: {backend}")

        try:
            if self._use_async_grammar_compilation:
                grammar = self.executor.submit(self._create_grammar, request)
            else:
                grammar = self._create_grammar(request)  # type: ignore[assignment]
            request.structured_output_request.grammar = grammar  # type: ignore[assignment]
        except Exception as exc:
            logger.error(
                "Failed to initialize structured output grammar for request_id=%s",
                request.request_id,
                exc_info=True,
            )
            request.structured_output_request._grammar_error = exc

    def _create_grammar(self, request: "Request") -> StructuredOutputGrammar:
        key = request.structured_output_request.structured_output_key  # type: ignore[union-attr]

        # Note that the request was validated in the engine core client,
        # so at this point we know it is a supported type of request.
        #
        # TODO: we still need to handle xgrammar compilation failures,
        # though it should be unlikely as we test that up front as well.
        request_type, grammar_spec = key

        assert self.backend is not None
        request_backend = None
        if request.sampling_params and request.sampling_params.structured_outputs:
            request_backend = request.sampling_params.structured_outputs._backend
        if request_backend == "guidance" and isinstance(self.backend, XgrammarBackend):
            return self._compile_with_guidance(request_type, grammar_spec)

        if self._should_use_guidance_for_xgrammar(request_type, grammar_spec):
            schema = json.loads(grammar_spec)
            logger.info(
                "using guidance for unconstrained string JSON schema request_id=%s",
                request.request_id,
            )
            return self._compile_with_guidance(
                request_type,
                grammar_spec,
                disable_any_whitespace=not (
                    _is_tool_call_array_schema(schema)
                    and _may_need_reasoning_handoff_whitespace(request)
                ),
            )

        try:
            if isinstance(self.backend, XgrammarBackend) and request_type in {
                StructuredOutputOptions.JSON,
                StructuredOutputOptions.JSON_OBJECT,
            }:
                # Compact JSON prevents whitespace-only continuations from
                # consuming the full max_tokens budget after a valid prefix.
                return self.backend.compile_grammar_with_whitespace(
                    request_type,
                    grammar_spec,
                    any_whitespace=False,
                )
            return self.backend.compile_grammar(request_type, grammar_spec)
        except Exception:
            if not isinstance(self.backend, XgrammarBackend):
                raise

            logger.warning(
                "xgrammar compile failed for request_id=%s; trying guidance fallback",
                request.request_id,
                exc_info=True,
            )
            return self._compile_with_guidance(request_type, grammar_spec)

    def _should_use_guidance_for_xgrammar(
        self, request_type: StructuredOutputOptions, grammar_spec: str
    ) -> bool:
        if not isinstance(self.backend, XgrammarBackend):
            return False
        if request_type != StructuredOutputOptions.JSON:
            return False
        try:
            schema = json.loads(grammar_spec)
        except Exception:
            return False
        if has_guidance_unsupported_json_features(schema):
            return False
        return json_schema_should_use_guidance_for_unconstrained_strings(schema)

    def _compile_with_guidance(
        self,
        request_type: StructuredOutputOptions,
        grammar_spec: str,
        *,
        disable_any_whitespace: bool | None = None,
    ) -> StructuredOutputGrammar:
        vocab_size = self.vllm_config.model_config.get_vocab_size()
        guidance_backend = GuidanceBackend(
            self.vllm_config,
            tokenizer=self.tokenizer,
            vocab_size=vocab_size,
        )
        if disable_any_whitespace is not None:
            guidance_backend.disable_any_whitespace = disable_any_whitespace
        return guidance_backend.compile_grammar(request_type, grammar_spec)

    def _fill_bitmasks(
        self, batch: Iterable[tuple[StructuredOutputGrammar, int, bool]]
    ) -> None:
        assert self._grammar_bitmask is not None
        for grammar, index, apply_bitmask in batch:
            if apply_bitmask and not grammar.is_terminated():
                grammar.fill_bitmask(self._grammar_bitmask, index)
            else:
                # Note that for thinking support, we will need to
                # reset the relevant part of the bitmask for consequent
                # requests here.
                self._grammar_bitmask[index].fill_(self._full_mask)

    def _async_submit_fill_bitmask(
        self, batch: list[tuple[StructuredOutputGrammar, int, bool]]
    ) -> Future:
        return self.executor_for_fillmask.submit(self._fill_bitmasks, batch)

    def grammar_bitmask(
        self,
        requests: dict[str, "Request"],
        structured_output_request_ids: list[str],
        scheduled_spec_decode_tokens: dict[str, list[int]],
    ) -> "npt.NDArray[np.int32] | None":
        # Prepare the structured output bitmask for this batch.
        if not structured_output_request_ids:
            return None

        max_num_spec_tokens = 0
        if self.vllm_config.speculative_config is not None:
            max_num_spec_tokens = (
                self.vllm_config.speculative_config.num_speculative_tokens
            )

        if self._grammar_bitmask is None:
            assert self.backend is not None
            max_batch_size = self.vllm_config.scheduler_config.max_num_seqs

            # Allocate a bitmask for each token needing to be checked:
            # one for each speculative position, and one more for the
            # bonus token / non-speculative token.
            self._grammar_bitmask = self.backend.allocate_token_bitmask(
                max_batch_size * (1 + max_num_spec_tokens)
            )

        # Generate a batched bitmask for all structured output requests.
        # When speculative decoding is enabled, we need to include multiple
        # masks for each request, one for each possible bonus token position.
        # These are stored inline in the tensor and unpacked by the gpu runner.
        cumulative_index = 0

        # Optimized parallel filling of bitmasks for
        # non-spec, large-batch-size cases
        if (
            len(structured_output_request_ids) > self.fill_bitmask_parallel_threshold
            and max_num_spec_tokens == 0
        ):
            promises = []
            batch = []
            for req_id in structured_output_request_ids:
                request = requests[req_id]
                structured_output_request = request.structured_output_request
                if TYPE_CHECKING:
                    assert structured_output_request is not None
                    assert structured_output_request.grammar is not None
                grammar = structured_output_request.grammar

                apply_bitmask = self.should_fill_bitmask(request)
                batch.append((grammar, cumulative_index, apply_bitmask))
                if len(batch) == self.fill_bitmask_parallel_batch_size:
                    promises.append(self._async_submit_fill_bitmask(batch))
                    batch = []

                cumulative_index += 1
            if batch:
                promises.append(self._async_submit_fill_bitmask(batch))

            # Wait for all bitmask filling tasks to complete.
            for promise in promises:
                promise.result()
        else:
            # Fallback to serial filling of bitmasks for small-batch-size cases
            for req_id in structured_output_request_ids:
                request = requests[req_id]
                structured_output_request = request.structured_output_request

                if TYPE_CHECKING:
                    assert structured_output_request is not None
                    assert structured_output_request.grammar is not None
                grammar = structured_output_request.grammar
                apply_bitmask = self.should_fill_bitmask(request)

                state_advancements = 0
                req_tokens = scheduled_spec_decode_tokens.get(req_id, ())
                for token in itertools.chain(req_tokens, (-1,)):
                    self._fill_bitmasks(((grammar, cumulative_index, apply_bitmask),))
                    if token == -1:
                        # Stop advancing the grammar once we hit a padding token.
                        apply_bitmask = False
                    if apply_bitmask and not grammar.is_terminated():
                        accepted = grammar.accept_tokens(req_id, [token])
                        assert accepted, (token, req_id, scheduled_spec_decode_tokens)
                        state_advancements += 1
                    cumulative_index += 1
                if state_advancements > 0:
                    grammar.rollback(state_advancements)

        bitmask_tensor = self._grammar_bitmask
        if cumulative_index < bitmask_tensor.shape[0]:
            bitmask_tensor = bitmask_tensor[:cumulative_index]

        # After finishing with the xgrammar operations, we convert to
        # np.ndarray, because that is much more efficient for serialization
        # and deserialization when sending this to the GPU workers.
        return bitmask_tensor.numpy()

    def should_fill_bitmask(self, request: "Request") -> bool:
        # NOTE (Hanchen) if enable_in_reasoning is True, it means that
        # the model needs to be constrained in reasoning. So we should always
        # enable the bitmask filling.
        reasoner = self._get_reasoner(request)
        if reasoner is not None:
            if self.enable_in_reasoning:
                return True
            assert request.structured_output_request is not None
            if request.structured_output_request.reasoning_ended is None:
                # This should be removed here, but since `openai_gptoss`
                # is an independent code path, it is kept for now.
                # After unifying the `openai_gptoss` and non-`openai_gptoss` styles,
                # it can be removed.
                request.structured_output_request.reasoning_ended = (
                    reasoner.is_reasoning_end(request.prompt_token_ids or [])
                )
            return request.structured_output_request.reasoning_ended
        return True

    def should_advance(self, request: "Request") -> bool:
        if not request.use_structured_output:
            return False

        # To determine whether we can advance the FSM.
        # Supports thinking usage where we skip the reasoning components.
        if TYPE_CHECKING:
            assert request.structured_output_request is not None
            assert request.structured_output_request.grammar is not None
        # by default, we should always advance
        # for cases that don't use thinking mode.
        reasoner = self._get_reasoner(request)
        if reasoner is None:
            return True

        # if the model needs structured in reasoning, we should advance
        if self.enable_in_reasoning:
            return True

        structured_req = request.structured_output_request
        if structured_req.reasoning_ended:
            return True

        # Check if reasoning ends in *this* step
        delta_from = request.num_computed_tokens - request.num_output_placeholders
        all_token_ids = request.all_token_ids
        start = (
            delta_from if delta_from >= 0 else max(len(all_token_ids) + delta_from, 0)
        )
        if reasoner.is_reasoning_end_streaming(
            all_token_ids, itertools.islice(all_token_ids, start, None)
        ):
            # Reasoning just ended, so we shouldn't advance til
            # next pass
            structured_req.reasoning_ended = True

        return False

    def clear_backend(self) -> None:
        if self.backend is not None:
            self.backend.destroy()


def _is_tool_call_array_schema(schema: object) -> bool:
    """Return true for the required-tool JSON array schema shape.

    Tool-call schemas can appear immediately after a reasoning suffix such as
    ``</think>\n\n``. Guidance's compact whitespace mode is useful for normal
    structured outputs, but it rejects that leading newline for tool arrays.
    """
    if not isinstance(schema, dict) or schema.get("type") != "array":
        return False
    items = schema.get("items")
    if not isinstance(items, dict):
        return False
    variants = items.get("anyOf")
    if not isinstance(variants, list) or not variants:
        return False
    for variant in variants:
        if not isinstance(variant, dict):
            return False
        properties = variant.get("properties")
        required = variant.get("required")
        if not isinstance(properties, dict) or not isinstance(required, list):
            return False
        if not {"name", "parameters"}.issubset(properties):
            return False
        if not {"name", "parameters"}.issubset(set(required)):
            return False
    return True


def _may_need_reasoning_handoff_whitespace(request: "Request") -> bool:
    """Return true when grammar may start after a reasoning suffix.

    Required-tool arrays in thinking mode can begin immediately after
    ``</think>\n\n``. Compact guidance whitespace rejects that leading newline.
    If the frontend already determined reasoning is ended or disabled, compact
    whitespace is safer because it avoids whitespace-only runaways.
    """
    structured_req = request.structured_output_request
    if structured_req is None:
        return False
    return structured_req.reasoning_ended is not True
