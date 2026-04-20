# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
import multiprocessing
from collections.abc import Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.reasoning import ReasoningParserManager
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.utils.mistral import is_mistral_tokenizer
from vllm.utils.import_utils import LazyLoader
from vllm.v1.structured_output.backend_guidance import GuidanceBackend
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


class PassthroughGrammar(StructuredOutputGrammar):
    """Safety-net grammar that leaves decoding unconstrained."""

    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
        return True

    def validate_tokens(self, tokens: list[int]) -> list[int]:
        return tokens

    def rollback(self, num_tokens: int) -> None:
        return None

    def fill_bitmask(self, bitmask: "torch.Tensor", batch_index: int) -> None:
        bitmask[batch_index].fill_(-1)

    def is_terminated(self) -> bool:
        return False

    def reset(self):
        return None



class StructuredOutputManager:
    """Engine-level manager for structured output requests."""

    def __init__(self, vllm_config: VllmConfig):
        self.backend: StructuredOutputBackend | None = None
        self.backend_name: str | None = None
        self.backends: dict[str, StructuredOutputBackend] = {}
        self.reasoner: ReasoningParser | None = None
        self.vllm_config = vllm_config

        # When in external_launcher mode, async grammar compilation causes deadlocks
        # due to external_launcher mode having a scheduler for each TP rank.
        # Async grammar compilation causes the WAITING_FOR_FSM → WAITING transition to
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
                reasoner_cls = ReasoningParserManager.get_reasoning_parser(
                    reasoning_parser
                )
                self.reasoner = reasoner_cls(tokenizer=self.tokenizer)

        self.enable_in_reasoning = (
            self.vllm_config.structured_outputs_config.enable_in_reasoning
        )

    def _create_backend(self, backend: str) -> StructuredOutputBackend:
        vocab_size = self.vllm_config.model_config.get_vocab_size()
        if backend == "xgrammar":
            return XgrammarBackend(
                self.vllm_config,
                tokenizer=self.tokenizer,
                vocab_size=vocab_size,
            )
        if backend == "guidance":
            return GuidanceBackend(
                self.vllm_config,
                tokenizer=self.tokenizer,
                vocab_size=vocab_size,
            )
        if backend == "outlines":
            from vllm.v1.structured_output.backend_outlines import OutlinesBackend

            return OutlinesBackend(
                self.vllm_config,
                tokenizer=self.tokenizer,
                vocab_size=vocab_size,
            )
        if backend == "lm-format-enforcer":
            from vllm.v1.structured_output.backend_lm_format_enforcer import (  # noqa: E501
                LMFormatEnforcerBackend,
            )

            return LMFormatEnforcerBackend(
                self.vllm_config,
                tokenizer=self.tokenizer,
                vocab_size=vocab_size,
            )
        raise ValueError(f"Unsupported structured output backend: {backend}")

    @staticmethod
    def _uses_structural_tag(request: "Request") -> bool:
        structured_req = request.structured_output_request
        if structured_req is None:
            return False
        key = structured_req.structured_output_key
        return bool(key and key[0] == StructuredOutputOptions.STRUCTURAL_TAG)

    @staticmethod
    def _request_backend_name(request: "Request") -> str:
        assert request.sampling_params is not None
        assert request.sampling_params.structured_outputs is not None
        backend = request.sampling_params.structured_outputs._backend
        if backend is None:
            raise ValueError("Structured output backend was not resolved")
        return backend

    def _get_backend_for_request(self, request: "Request") -> StructuredOutputBackend:
        assert request.sampling_params is not None
        assert request.sampling_params.structured_outputs is not None

        backend = request.sampling_params.structured_outputs._backend
        if backend is None:
            raise ValueError("Structured output backend was not resolved")

        backend_instance = self.backends.get(backend)
        if backend_instance is None:
            if self.backend is not None and not self.backends:
                # Backward compatibility for tests that inject a single mock
                # backend directly on the manager without populating backends.
                backend_instance = self.backend
            else:
                backend_instance = self._create_backend(backend)
            self.backends[backend] = backend_instance

        # Keep legacy attributes aligned for tests/debugging, but do not treat
        # them as the sole source of truth; multiple backends may coexist.
        self.backend = backend_instance
        self.backend_name = backend
        return backend_instance

    def _guidance_fallback_backend_supported(self) -> bool:
        return not (
            is_mistral_tokenizer(self.tokenizer)
            and not getattr(self.tokenizer, "is_tekken", False)
        )

    def _compile_with_guidance_fallback(
        self,
        request: "Request",
        request_type: StructuredOutputOptions,
        grammar_spec: str,
        *,
        disable_any_whitespace: bool | None,
        disable_additional_properties: bool | None,
        original_error: Exception,
    ) -> StructuredOutputGrammar:
        if not self._guidance_fallback_backend_supported():
            logger.warning(
                "Structured output xgrammar compilation failed for request %s and guidance fallback is unsupported for this tokenizer: %s",
                request.request_id,
                original_error,
            )
            return PassthroughGrammar()

        try:
            guidance_backend = self.backends.get("guidance")
            if guidance_backend is None:
                guidance_backend = self._create_backend("guidance")
                self.backends["guidance"] = guidance_backend
            grammar = guidance_backend.compile_grammar(
                request_type,
                grammar_spec,
                disable_any_whitespace=disable_any_whitespace,
                disable_additional_properties=disable_additional_properties,
            )
            logger.warning(
                "Structured output xgrammar compilation failed for request %s; falling back to guidance: %s",
                request.request_id,
                original_error,
            )
            return grammar
        except Exception:
            logger.exception(
                "Structured output grammar compilation failed for request %s in xgrammar and guidance fallback; falling back to unconstrained decoding.",
                request.request_id,
            )
            return PassthroughGrammar()

    def grammar_init(self, request: "Request") -> None:
        if request.structured_output_request is None:
            return

        if TYPE_CHECKING:
            assert (
                request.sampling_params is not None
                and request.sampling_params.structured_outputs is not None
            )

        # Requests can legitimately mix backends (for example regex/json via
        # LMFE and explicit grammars via xgrammar) within the same engine
        # lifetime, including concurrently. Cache backend instances by name.
        self._get_backend_for_request(request)

        if self._use_async_grammar_compilation:
            grammar = self.executor.submit(self._create_grammar, request)
        else:
            grammar = self._create_grammar(request)  # type: ignore[assignment]
        request.structured_output_request.grammar = grammar  # type: ignore[assignment]

    def _create_grammar(self, request: "Request") -> StructuredOutputGrammar:
        key = request.structured_output_request.structured_output_key  # type: ignore[union-attr]

        # Note that the request was validated in the engine core client,
        # so at this point we know it is a supported type of request.
        #
        # TODO: we still need to handle xgrammar compilation failures,
        # though it should be unlikely as we test that up front as well.
        request_type, grammar_spec, _, _, _ = key

        backend = self._get_backend_for_request(request)
        so_params = request.sampling_params.structured_outputs
        disable_any_whitespace = (
            None if so_params is None else so_params.disable_any_whitespace
        )
        disable_additional_properties = (
            None if so_params is None else so_params.disable_additional_properties
        )
        try:
            return backend.compile_grammar(
                request_type,
                grammar_spec,
                disable_any_whitespace=disable_any_whitespace,
                disable_additional_properties=disable_additional_properties,
            )
        except Exception as err:
            backend_name = self._request_backend_name(request)
            if backend_name == "xgrammar":
                return self._compile_with_guidance_fallback(
                    request,
                    request_type,
                    grammar_spec,
                    disable_any_whitespace=disable_any_whitespace,
                    disable_additional_properties=disable_additional_properties,
                    original_error=err,
                )
            logger.exception(
                "Structured output grammar compilation failed for request %s with backend %s; falling back to unconstrained decoding for this request.",
                request.request_id,
                backend_name,
            )
            return PassthroughGrammar()

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
            max_batch_size = self.vllm_config.scheduler_config.max_num_seqs
            bitmask_backend = (
                next(iter(self.backends.values()), None) or self.backend
            )
            assert bitmask_backend is not None

            # Allocate a bitmask for each token needing to be checked:
            # one for each speculative position, and one more for the
            # bonus token / non-speculative token.
            self._grammar_bitmask = bitmask_backend.allocate_token_bitmask(
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
        if self._uses_structural_tag(request):
            return True

        # NOTE (Hanchen) if enable_in_reasoning is True, it means that
        # the model needs to be constrained in reasoning. So we should always
        # enable the bitmask filling.
        if self.reasoner is not None:
            if self.enable_in_reasoning:
                return True
            assert request.structured_output_request is not None
            if request.structured_output_request.reasoning_ended is None:
                # This should be removed here, but since `openai_gptoss`
                # is an independent code path, it is kept for now.
                # After unifying the `openai_gptoss` and non-`openai_gptoss` styles,
                # it can be removed.
                request.structured_output_request.reasoning_ended = (
                    self.reasoner.is_reasoning_end(request.prompt_token_ids or [])
                )
            return request.structured_output_request.reasoning_ended
        return True

    def should_advance(self, request: "Request") -> bool:
        if not request.use_structured_output:
            return False

        if self._uses_structural_tag(request):
            return True

        # To determine whether we can advance the FSM.
        # Supports thinking usage where we skip the reasoning components.
        if TYPE_CHECKING:
            assert request.structured_output_request is not None
            assert request.structured_output_request.grammar is not None
        # by default, we should always advance
        # for cases that don't use thinking mode.
        if self.reasoner is None:
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
        if self.reasoner.is_reasoning_end_streaming(
            all_token_ids, itertools.islice(all_token_ids, start, None)
        ):
            # Reasoning just ended, so we shouldn't advance til
            # next pass
            structured_req.reasoning_ended = True

        return False

    def clear_backend(self) -> None:
        for backend in self.backends.values():
            backend.destroy()
        if self.backend is not None and not self.backends:
            self.backend.destroy()
        self.backends = {}
        self.backend = None
        self.backend_name = None
        self._grammar_bitmask = None
