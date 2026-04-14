# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for reasoning-aware structured output functionality (PR #25515)."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.config import ModelConfig, SchedulerConfig, VllmConfig
from vllm.reasoning import ReasoningParser
from vllm.sampling_params import StructuredOutputsParams
from vllm.v1.request import Request
from vllm.v1.structured_output.backend_lm_format_enforcer import (
    LMFormatEnforcerGrammar,
)
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.structured_output.backend_types import StructuredOutputOptions
from vllm.v1.structured_output.request import get_structured_output_key


class TestReasoningStructuredOutput:
    """Test reasoning-aware structured output functionality."""

    @pytest.fixture
    def mock_model_config(self):
        """Create a mock ModelConfig."""
        config = Mock(spec=ModelConfig)
        config.skip_tokenizer_init = True  # Skip tokenizer init to avoid network calls
        config.get_vocab_size = Mock(return_value=50000)
        # Add missing runner_type attribute that tokenizer initialization expects
        config.runner_type = "generate"
        # Add other attributes that tokenizer initialization might need
        config.tokenizer = "test-tokenizer"
        config.tokenizer_mode = "auto"
        config.trust_remote_code = False
        config.tokenizer_revision = None
        return config

    @pytest.fixture
    def mock_scheduler_config(self):
        """Create a mock SchedulerConfig."""
        config = Mock(spec=SchedulerConfig)
        config.max_num_seqs = 128
        return config

    @pytest.fixture
    def mock_vllm_config(self, mock_model_config, mock_scheduler_config):
        """Create a mock VllmConfig."""
        config = Mock(spec=VllmConfig)
        config.model_config = mock_model_config
        config.scheduler_config = mock_scheduler_config
        config.structured_outputs_config = Mock()
        config.structured_outputs_config.reasoning_parser = None
        config.structured_outputs_config.enable_in_reasoning = False
        config.speculative_config = None
        return config

    @pytest.fixture
    def mock_reasoning_parser(self):
        """Create a mock ReasoningParser."""
        parser = Mock(spec=ReasoningParser)
        parser.is_reasoning_end = Mock(return_value=False)
        return parser

    @pytest.fixture
    def mock_request_with_structured_output(self):
        """Create a mock request with structured output."""
        request = Mock(spec=Request)
        request.structured_output_request = Mock()
        request.structured_output_request.reasoning_ended = None
        request.structured_output_request.grammar = Mock()
        request.structured_output_request.grammar.is_terminated = Mock(
            return_value=False
        )
        request.use_structured_output = True
        request.prompt_token_ids = [1, 2, 3, 4, 5]
        request.all_token_ids = [1, 2, 3, 4, 5, 6, 7, 8]
        request.num_computed_tokens = 5
        request.num_output_placeholders = 0
        return request

    def test_should_fill_bitmask_with_enable_in_reasoning(
        self, mock_vllm_config, mock_request_with_structured_output
    ):
        """Test should_fill_bitmask when enable_in_reasoning is True."""
        # Enable enable_in_reasoning
        mock_vllm_config.structured_outputs_config.enable_in_reasoning = True

        manager = StructuredOutputManager(mock_vllm_config)

        # Should always return True when enable_in_reasoning is enabled
        result = manager.should_fill_bitmask(mock_request_with_structured_output)
        assert result is True

    def test_should_fill_bitmask_structural_tag_bypasses_reasoning_gate(
        self,
        mock_vllm_config,
        mock_request_with_structured_output,
        mock_reasoning_parser,
    ):
        manager = StructuredOutputManager(mock_vllm_config)
        manager.reasoner = mock_reasoning_parser

        (
            mock_request_with_structured_output.structured_output_request
        ).structured_output_key = (StructuredOutputOptions.STRUCTURAL_TAG, "tag")
        (
            mock_request_with_structured_output.structured_output_request
        ).reasoning_ended = False
        mock_reasoning_parser.is_reasoning_end.return_value = False

        assert manager.should_fill_bitmask(mock_request_with_structured_output) is True

    def test_should_fill_bitmask_without_enable_in_reasoning(
        self,
        mock_vllm_config,
        mock_request_with_structured_output,
        mock_reasoning_parser,
    ):
        """Test should_fill_bitmask when enable_in_reasoning is False."""
        # Keep enable_in_reasoning as False (default)
        config = mock_vllm_config.structured_outputs_config
        assert config.enable_in_reasoning is False

        manager = StructuredOutputManager(mock_vllm_config)
        manager.reasoner = mock_reasoning_parser

        # Mock reasoning not ended
        mock_reasoning_parser.is_reasoning_end.return_value = False

        result = manager.should_fill_bitmask(mock_request_with_structured_output)

        # Should set reasoning_ended and return its value
        assert (
            mock_request_with_structured_output.structured_output_request.reasoning_ended
            is False
        )
        assert result is False

    def test_should_fill_bitmask_no_reasoner(
        self, mock_vllm_config, mock_request_with_structured_output
    ):
        """Test should_fill_bitmask when no reasoner is configured."""
        manager = StructuredOutputManager(mock_vllm_config)
        manager.reasoner = None

        result = manager.should_fill_bitmask(mock_request_with_structured_output)

        # Should default to True when no reasoner
        assert result is True

    def test_should_advance_with_enable_in_reasoning(
        self,
        mock_vllm_config,
        mock_request_with_structured_output,
        mock_reasoning_parser,
    ):
        """Test should_advance when enable_in_reasoning is True."""
        # Enable enable_in_reasoning
        mock_vllm_config.structured_outputs_config.enable_in_reasoning = True

        manager = StructuredOutputManager(mock_vllm_config)
        manager.reasoner = mock_reasoning_parser

        # Should always return True when enable_in_reasoning is enabled
        result = manager.should_advance(mock_request_with_structured_output)
        assert result is True

    def test_should_advance_reasoning_not_ended(
        self,
        mock_vllm_config,
        mock_request_with_structured_output,
        mock_reasoning_parser,
    ):
        """Test should_advance when reasoning has not ended."""
        manager = StructuredOutputManager(mock_vllm_config)
        manager.reasoner = mock_reasoning_parser

        # Set reasoning as not ended
        (
            mock_request_with_structured_output.structured_output_request
        ).reasoning_ended = False
        mock_reasoning_parser.is_reasoning_end.return_value = False

        result = manager.should_advance(mock_request_with_structured_output)

        # Should return False since reasoning hasn't ended
        assert result is False

    def test_should_advance_reasoning_just_ended(
        self,
        mock_vllm_config,
        mock_request_with_structured_output,
        mock_reasoning_parser,
    ):
        """Test should_advance when reasoning ends in current step."""
        manager = StructuredOutputManager(mock_vllm_config)
        manager.reasoner = mock_reasoning_parser

        # Set reasoning as not ended initially, but ends in this step
        (
            mock_request_with_structured_output.structured_output_request
        ).reasoning_ended = False
        mock_reasoning_parser.is_reasoning_end.return_value = True

        result = manager.should_advance(mock_request_with_structured_output)

        # Should set reasoning_ended to True but return False for this step
        assert (
            mock_request_with_structured_output.structured_output_request.reasoning_ended
            is True
        )
        assert result is False

    def test_should_advance_reasoning_already_ended(
        self,
        mock_vllm_config,
        mock_request_with_structured_output,
        mock_reasoning_parser,
    ):
        """Test should_advance when reasoning has already ended."""
        manager = StructuredOutputManager(mock_vllm_config)
        manager.reasoner = mock_reasoning_parser

        # Set reasoning as already ended
        (
            mock_request_with_structured_output.structured_output_request
        ).reasoning_ended = True

        result = manager.should_advance(mock_request_with_structured_output)

        # Should return True since reasoning has ended
        assert result is True

    def test_should_advance_structural_tag_bypasses_reasoning_gate(
        self,
        mock_vllm_config,
        mock_request_with_structured_output,
        mock_reasoning_parser,
    ):
        manager = StructuredOutputManager(mock_vllm_config)
        manager.reasoner = mock_reasoning_parser

        (
            mock_request_with_structured_output.structured_output_request
        ).structured_output_key = (StructuredOutputOptions.STRUCTURAL_TAG, "tag")
        (
            mock_request_with_structured_output.structured_output_request
        ).reasoning_ended = False

        assert manager.should_advance(mock_request_with_structured_output) is True

    def test_create_grammar_uses_request_level_structured_output_flags(
        self,
        mock_vllm_config,
    ):
        manager = StructuredOutputManager(mock_vllm_config)
        backend = Mock()
        backend.compile_grammar.return_value = "compiled"
        manager.backend = backend
        manager.backends["xgrammar"] = backend
        manager.backend_name = "xgrammar"

        request = Mock(spec=Request)
        request.structured_output_request = Mock()
        request.structured_output_request.structured_output_key = (
            "json",
            '{"type":"object"}',
        )
        request.sampling_params = Mock()
        request.sampling_params.structured_outputs = Mock(
            _backend="xgrammar",
            disable_any_whitespace=True,
            disable_additional_properties=True,
        )

        grammar = manager._create_grammar(request)

        assert grammar == "compiled"
        backend.compile_grammar.assert_called_once_with(
            "json",
            '{"type":"object"}',
            disable_any_whitespace=True,
            disable_additional_properties=True,
        )

    def test_structured_output_key_includes_request_level_flags(self):
        params = StructuredOutputsParams(
            json='{"type":"object"}',
            disable_any_whitespace=True,
            disable_additional_properties=True,
            whitespace_pattern=r"\\s?",
        )

        assert get_structured_output_key(params) == (
            StructuredOutputOptions.JSON,
            '{"type":"object"}',
            True,
            True,
            r"\\s?",
        )

    def test_get_backend_for_request_caches_multiple_backends(self, mock_vllm_config):
        manager = StructuredOutputManager(mock_vllm_config)
        manager.tokenizer = Mock()

        first_backend = Mock()
        second_backend = Mock()
        regex_request = Mock(spec=Request)
        regex_request.structured_output_request = Mock()
        regex_request.sampling_params = Mock()
        regex_request.sampling_params.structured_outputs = Mock(
            _backend="lm-format-enforcer"
        )
        grammar_request = Mock(spec=Request)
        grammar_request.structured_output_request = Mock()
        grammar_request.sampling_params = Mock()
        grammar_request.sampling_params.structured_outputs = Mock(_backend="xgrammar")

        from vllm.v1 import structured_output as structured_output_module

        old_lmfe = structured_output_module.LMFormatEnforcerBackend
        old_xgrammar = structured_output_module.XgrammarBackend
        structured_output_module.LMFormatEnforcerBackend = Mock(
            return_value=first_backend
        )
        structured_output_module.XgrammarBackend = Mock(return_value=second_backend)
        try:
            regex_backend = manager._get_backend_for_request(regex_request)
            grammar_backend = manager._get_backend_for_request(grammar_request)
        finally:
            structured_output_module.LMFormatEnforcerBackend = old_lmfe
            structured_output_module.XgrammarBackend = old_xgrammar

        assert regex_backend is first_backend
        assert grammar_backend is second_backend
        assert manager.backends == {
            "lm-format-enforcer": first_backend,
            "xgrammar": second_backend,
        }
        assert manager.backend is second_backend
        assert manager.backend_name == "xgrammar"
        first_backend.destroy.assert_not_called()
        second_backend.destroy.assert_not_called()

    def test_clear_backend_resets_backend_state(self, mock_vllm_config):
        manager = StructuredOutputManager(mock_vllm_config)
        backend_one = Mock()
        backend_two = Mock()
        manager.backends = {
            "xgrammar": backend_one,
            "lm-format-enforcer": backend_two,
        }
        manager.backend = backend_two
        manager.backend_name = "lm-format-enforcer"
        manager._grammar_bitmask = torch.ones((1, 1), dtype=torch.int32)

        manager.clear_backend()

        backend_one.destroy.assert_called_once()
        backend_two.destroy.assert_called_once()
        assert manager.backends == {}
        assert manager.backend is None
        assert manager.backend_name is None
        assert manager._grammar_bitmask is None


def _bitmask_allows(bitmask_row: torch.Tensor, token_id: int) -> bool:
    element_index = token_id >> 5
    bit_index = token_id & 0x1F
    return bool(int(bitmask_row[element_index]) & (1 << bit_index))


class _FakeParser:
    def __init__(self, *, can_end: bool):
        self._can_end = can_end

    def can_end(self) -> bool:
        return self._can_end


class _FakeAllowedTokens:
    def __init__(self, vocab_size: int, tokens: list[int]):
        tensor_size = (vocab_size + 31) // 32
        self.allowed_tokens = torch.zeros(tensor_size, dtype=torch.int32)
        for token_id in tokens:
            element_index = token_id >> 5
            bit_index = token_id & 0x1F
            self.allowed_tokens[element_index] |= 1 << bit_index

    def is_token_allowed(self, token_id: int) -> bool:
        return _bitmask_allows(self.allowed_tokens, token_id)


class _FakeTokenEnforcer:
    def __init__(self, *, vocab_size: int = 64, eos_token_id: int = 7):
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id
        self.prefix_states: dict[tuple[int, ...], SimpleNamespace] = {}

    def get_allowed_tokens(self, token_sequence: list[int]) -> _FakeAllowedTokens:
        prefix = tuple(token_sequence)
        can_end = prefix == (1, 2)
        allowed = [5]
        if can_end:
            allowed.append(self.eos_token_id)
        self.prefix_states[prefix] = SimpleNamespace(parser=_FakeParser(can_end=can_end))
        return _FakeAllowedTokens(self.vocab_size, allowed)


def test_lmfe_fill_bitmask_forces_eos_once_parser_can_end():
    grammar = LMFormatEnforcerGrammar(token_enforcer=_FakeTokenEnforcer())
    bitmask = torch.zeros((1, 2), dtype=torch.int32)

    grammar.fill_bitmask(bitmask, 0)
    assert _bitmask_allows(bitmask[0], 5)
    assert not _bitmask_allows(bitmask[0], 7)

    grammar.current_tokens_prefix = [1, 2]
    grammar.fill_bitmask(bitmask, 0)
    assert _bitmask_allows(bitmask[0], 7)
    assert not _bitmask_allows(bitmask[0], 5)


def test_lmfe_is_not_terminated_until_eos_is_emitted():
    grammar = LMFormatEnforcerGrammar(token_enforcer=_FakeTokenEnforcer())

    grammar.current_tokens_prefix = [1, 2]
    assert grammar.is_terminated() is False

    grammar.current_tokens_prefix = [1, 2, 7]
    assert grammar.is_terminated() is True
