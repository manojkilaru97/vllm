# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for reasoning-aware structured output functionality (PR #25515)."""

import json
from unittest.mock import Mock

import pytest
import torch

from vllm.config import ModelConfig, SchedulerConfig, VllmConfig
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.structured_output.backend_types import StructuredOutputOptions


class MockReasoner:
    def __init__(self, tokenizer):
        self.is_reasoning_end = Mock(return_value=False)
        self.is_reasoning_end_streaming = Mock(return_value=False)


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
    def mock_request_with_structured_output(self):
        """Create a mock request with structured output."""
        request = Mock(spec=Request)
        request.structured_output_request = Mock()
        request.structured_output_request.reasoning_ended = None
        request.structured_output_request.reasoning_checked_token_count = 0
        request.structured_output_request.grammar = Mock()
        request.structured_output_request.reasoning_parser_kwargs = None
        request.structured_output_request.reasoner = None
        request.structured_output_request.grammar.is_terminated = Mock(
            return_value=False
        )
        request.use_structured_output = True
        request.prompt_token_ids = [1, 2, 3, 4, 5]
        request.all_token_ids = [1, 2, 3, 4, 5, 6, 7, 8]
        request.num_computed_tokens = 5
        request.num_output_placeholders = 0
        return request

    @pytest.fixture
    def manager_with_reasoner(self, mock_vllm_config):
        manager = StructuredOutputManager(mock_vllm_config)
        manager.reasoner_cls = MockReasoner
        manager.tokenizer = Mock()
        return manager

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

    def test_should_fill_bitmask_without_enable_in_reasoning(
        self,
        manager_with_reasoner,
        mock_request_with_structured_output,
    ):
        """Test should_fill_bitmask when enable_in_reasoning is False."""
        # Keep enable_in_reasoning as False (default)
        config = manager_with_reasoner.vllm_config.structured_outputs_config
        assert config.enable_in_reasoning is False

        result = manager_with_reasoner.should_fill_bitmask(
            mock_request_with_structured_output
        )

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

        result = manager.should_fill_bitmask(mock_request_with_structured_output)

        # Should default to True when no reasoner
        assert result is True

    def test_should_fill_bitmask_uses_request_reasoning_parser_kwargs(
        self, mock_vllm_config, mock_request_with_structured_output
    ):
        """Test request-level parser kwargs override the default reasoner."""

        class KwargReasoner:
            def __init__(self, tokenizer, chat_template_kwargs=None):
                self.chat_template_kwargs = chat_template_kwargs or {}

            def is_reasoning_end(self, input_ids):
                return not self.chat_template_kwargs.get("enable_thinking", False)

        manager = StructuredOutputManager(mock_vllm_config)
        manager.reasoner_cls = KwargReasoner
        manager.tokenizer = Mock()

        structured_req = mock_request_with_structured_output.structured_output_request
        structured_req.reasoning_parser_kwargs = {
            "chat_template_kwargs": {"enable_thinking": True}
        }

        result = manager.should_fill_bitmask(mock_request_with_structured_output)

        assert result is False
        assert (
            mock_request_with_structured_output.structured_output_request.reasoner
            is not None
        )

    def test_should_advance_with_enable_in_reasoning(
        self,
        manager_with_reasoner,
        mock_request_with_structured_output,
    ):
        """Test should_advance when enable_in_reasoning is True."""
        # Enable enable_in_reasoning
        manager_with_reasoner.enable_in_reasoning = True

        # Should always return True when enable_in_reasoning is enabled
        result = manager_with_reasoner.should_advance(
            mock_request_with_structured_output
        )
        assert result is True

    def test_should_advance_reasoning_not_ended(
        self,
        manager_with_reasoner,
        mock_request_with_structured_output,
    ):
        """Test should_advance when reasoning has not ended."""
        # Set reasoning as not ended
        (
            mock_request_with_structured_output.structured_output_request
        ).reasoning_ended = False

        result = manager_with_reasoner.should_advance(
            mock_request_with_structured_output
        )

        # Should return False since reasoning hasn't ended
        assert result is False

    def test_should_advance_reasoning_just_ended(
        self,
        manager_with_reasoner,
        mock_request_with_structured_output,
    ):
        """Test should_advance when reasoning ends in current step."""
        # Set reasoning as not ended initially, but ends in this step
        (
            mock_request_with_structured_output.structured_output_request
        ).reasoning_ended = False
        reasoner = MockReasoner(tokenizer=Mock())
        reasoner.is_reasoning_end_streaming.return_value = True
        structured_req = mock_request_with_structured_output.structured_output_request
        structured_req.reasoner = reasoner

        result = manager_with_reasoner.should_advance(
            mock_request_with_structured_output
        )

        # Should set reasoning_ended to True but return False for this step
        assert (
            mock_request_with_structured_output.structured_output_request.reasoning_ended
            is True
        )
        assert result is False

    def test_should_advance_reasoning_just_ended_with_spec_decode_structural_tag(
        self,
        manager_with_reasoner,
        mock_request_with_structured_output,
    ):
        """When reasoning ends this step, advance immediately for structural
        tags with speculative decoding."""
        structured_req = mock_request_with_structured_output.structured_output_request
        structured_req.reasoning_ended = False
        structured_req.structured_output_key = (
            StructuredOutputOptions.STRUCTURAL_TAG,
            "{}",
        )
        reasoner = MockReasoner(tokenizer=Mock())
        reasoner.is_reasoning_end_streaming.return_value = True
        structured_req.reasoner = reasoner

        manager_with_reasoner.vllm_config.speculative_config = Mock()

        result = manager_with_reasoner.should_advance(
            mock_request_with_structured_output
        )

        assert structured_req.reasoning_ended is True
        assert result is True

    def test_should_advance_reasoning_already_ended(
        self,
        manager_with_reasoner,
        mock_request_with_structured_output,
    ):
        """Test should_advance when reasoning has already ended."""
        # Set reasoning as already ended
        (
            mock_request_with_structured_output.structured_output_request
        ).reasoning_ended = True

        result = manager_with_reasoner.should_advance(
            mock_request_with_structured_output
        )

        # Should return True since reasoning has ended
        assert result is True

    def test_should_advance_spec_decode_scans_full_output(
        self,
        manager_with_reasoner,
        mock_request_with_structured_output,
    ):
        """Under spec decode, num_computed_tokens counts draft tokens absent from
        all_token_ids, so the windowed delta can land past a </think> buried in a
        multi-token acceptance. The fix scans the full output region instead."""
        req = mock_request_with_structured_output
        req.prompt_token_ids = [1, 2, 3, 4, 5]
        req.all_token_ids = [1, 2, 3, 4, 5, 6, 7, 8]
        # Overshoot: window start would be 11-4=7 -> delta=[8], missing 6,7.
        req.num_computed_tokens = 11
        req.num_output_placeholders = 4
        req.structured_output_request.reasoning_ended = False
        req.structured_output_request.structured_output_key = (
            StructuredOutputOptions.JSON,
            "{}",
        )
        reasoner = MockReasoner(tokenizer=Mock())
        reasoner.is_reasoning_end_streaming.return_value = True
        req.structured_output_request.reasoner = reasoner
        manager_with_reasoner.vllm_config.speculative_config = Mock()

        manager_with_reasoner.should_advance(req)

        # The delta passed must be the FULL output region, not the narrow window.
        call = reasoner.is_reasoning_end_streaming.call_args
        delta = list(call.args[1])
        assert delta == [6, 7, 8], f"expected full output region, got {delta}"
        # reasoning_ended flips even though </think> was buried mid-batch.
        assert req.structured_output_request.reasoning_ended is True

    def test_should_advance_spec_decode_uses_checked_cursor(
        self,
        manager_with_reasoner,
        mock_request_with_structured_output,
    ):
        req = mock_request_with_structured_output
        req.prompt_token_ids = [1, 2, 3, 4, 5]
        req.all_token_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        req.num_computed_tokens = 13
        req.num_output_placeholders = 4
        req.structured_output_request.reasoning_ended = False
        req.structured_output_request.reasoning_checked_token_count = 7
        reasoner = MockReasoner(tokenizer=Mock())
        reasoner.is_reasoning_end_streaming.return_value = False
        req.structured_output_request.reasoner = reasoner
        manager_with_reasoner.vllm_config.speculative_config = Mock()

        manager_with_reasoner.should_advance(req)

        call = reasoner.is_reasoning_end_streaming.call_args
        delta = list(call.args[1])
        assert delta == [8, 9]
        assert req.structured_output_request.reasoning_checked_token_count == 9

    def test_should_advance_without_spec_uses_window(
        self,
        manager_with_reasoner,
        mock_request_with_structured_output,
    ):
        """Without spec decode, the original 1-token windowed delta is used."""
        req = mock_request_with_structured_output
        req.prompt_token_ids = [1, 2, 3, 4, 5]
        req.all_token_ids = [1, 2, 3, 4, 5, 6, 7, 8]
        req.num_computed_tokens = 8
        req.num_output_placeholders = 1
        req.structured_output_request.reasoning_ended = False
        req.structured_output_request.structured_output_key = (
            StructuredOutputOptions.JSON,
            "{}",
        )
        reasoner = MockReasoner(tokenizer=Mock())
        reasoner.is_reasoning_end_streaming.return_value = False
        req.structured_output_request.reasoner = reasoner
        manager_with_reasoner.vllm_config.speculative_config = None

        manager_with_reasoner.should_advance(req)

        call = reasoner.is_reasoning_end_streaming.call_args
        delta = list(call.args[1])
        # window start = 8 - 1 = 7 -> delta = all_token_ids[7:] = [8]
        assert delta == [8], f"expected windowed delta, got {delta}"

    def test_token_ids_to_advance_returns_post_boundary_suffix(
        self,
        manager_with_reasoner,
        mock_request_with_structured_output,
    ):
        req = mock_request_with_structured_output
        new_token_ids = [10, 99, 123, 124]
        req.all_token_ids = [1, 2, 3, *new_token_ids]
        req.structured_output_request.reasoning_ended = False

        reasoner = MockReasoner(tokenizer=Mock())
        reasoner.is_reasoning_end_streaming.side_effect = (
            lambda input_ids, delta_ids: 99 in list(delta_ids)
        )
        req.structured_output_request.reasoner = reasoner

        token_ids = manager_with_reasoner.token_ids_to_advance(req, new_token_ids)

        assert token_ids == [123, 124]
        assert req.structured_output_request.reasoning_ended is True
        assert [
            list(call.args[1])
            for call in reasoner.is_reasoning_end_streaming.call_args_list
        ] == [[10, 99, 123, 124], [10], [99]]

    def test_token_ids_to_advance_ignores_unfinished_reasoning_tokens(
        self,
        manager_with_reasoner,
        mock_request_with_structured_output,
    ):
        req = mock_request_with_structured_output
        new_token_ids = [10, 11, 12]
        req.all_token_ids = [1, 2, 3, *new_token_ids]
        req.structured_output_request.reasoning_ended = False

        reasoner = MockReasoner(tokenizer=Mock())
        reasoner.is_reasoning_end_streaming.return_value = False
        req.structured_output_request.reasoner = reasoner

        token_ids = manager_with_reasoner.token_ids_to_advance(req, new_token_ids)

        assert token_ids == []
        assert req.structured_output_request.reasoning_ended is False

    def test_validate_spec_tokens_constrains_post_boundary_suffix(
        self,
        manager_with_reasoner,
        mock_request_with_structured_output,
    ):
        req = mock_request_with_structured_output
        req.all_token_ids = [1, 2, 3]
        req.num_computed_tokens = len(req.all_token_ids)
        req.num_output_placeholders = 0
        req.structured_output_request.reasoning_ended = False

        reasoner = MockReasoner(tokenizer=Mock())
        reasoner.is_reasoning_end_streaming.side_effect = (
            lambda input_ids, delta_ids: 99 in list(delta_ids)
        )
        req.structured_output_request.reasoner = reasoner
        req.structured_output_request.grammar.validate_tokens.return_value = [123]

        spec_token_ids = manager_with_reasoner.validate_spec_tokens(
            req, [10, 99, 123, 124]
        )

        assert spec_token_ids == [10, 99, 123]
        assert req.structured_output_request.reasoning_ended is False
        req.structured_output_request.grammar.validate_tokens.assert_called_once_with(
            [123, 124]
        )

    def test_validate_spec_tokens_after_accepted_reasoning_boundary(
        self,
        manager_with_reasoner,
        mock_request_with_structured_output,
    ):
        req = mock_request_with_structured_output
        req.all_token_ids = [1, 2, 3, 10, 99]
        req.prompt_token_ids = [1, 2, 3]
        req.num_computed_tokens = len(req.all_token_ids)
        req.num_output_placeholders = 3
        req.structured_output_request.reasoning_ended = False
        req.structured_output_request.structured_output_key = (
            StructuredOutputOptions.JSON,
            "{}",
        )

        reasoner = MockReasoner(tokenizer=Mock())
        reasoner.is_reasoning_end_streaming.side_effect = (
            lambda input_ids, delta_ids: 99 in list(delta_ids)
        )
        req.structured_output_request.reasoner = reasoner
        req.structured_output_request.grammar.validate_tokens.return_value = [123]
        manager_with_reasoner.vllm_config.speculative_config = Mock()

        spec_token_ids = manager_with_reasoner.validate_spec_tokens(req, [123, 124])

        assert spec_token_ids == [123]
        assert req.structured_output_request.reasoning_ended is True
        req.structured_output_request.grammar.validate_tokens.assert_called_once_with(
            [123, 124]
        )

    def test_guidance_json_schema_uses_compact_whitespace(
        self, mock_request_with_structured_output
    ):
        schema = {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
        }

        disable_any_whitespace = (
            StructuredOutputManager._guidance_disable_any_whitespace(
                StructuredOutputOptions.JSON,
                json.dumps(schema),
                mock_request_with_structured_output,
            )
        )

        assert disable_any_whitespace is True

    def test_guidance_tool_array_keeps_reasoning_handoff_whitespace(
        self, mock_request_with_structured_output
    ):
        schema = {
            "type": "array",
            "items": {
                "anyOf": [
                    {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "parameters": {"type": "object"},
                        },
                        "required": ["name", "parameters"],
                    }
                ]
            },
        }
        mock_request_with_structured_output.structured_output_request.reasoning_ended = (
            False
        )

        disable_any_whitespace = (
            StructuredOutputManager._guidance_disable_any_whitespace(
                StructuredOutputOptions.JSON,
                json.dumps(schema),
                mock_request_with_structured_output,
            )
        )

        assert disable_any_whitespace is False

    def test_grammar_bitmask_constrains_spec_tokens_after_reasoning_boundary(
        self,
        manager_with_reasoner,
        mock_request_with_structured_output,
    ):
        req = mock_request_with_structured_output
        req.all_token_ids = [1, 2, 3]
        req.structured_output_request.reasoning_ended = False

        reasoner = MockReasoner(tokenizer=Mock())
        reasoner.is_reasoning_end_streaming.side_effect = (
            lambda input_ids, delta_ids: 99 in list(delta_ids)
        )
        req.structured_output_request.reasoner = reasoner

        manager_with_reasoner.vllm_config.speculative_config = Mock(
            num_speculative_tokens=3
        )
        manager_with_reasoner._grammar_bitmask = torch.full(
            (4, 2), -1, dtype=torch.int32
        )

        grammar = req.structured_output_request.grammar

        def fill_bitmask(mask, index):
            mask[index].fill_(index)

        grammar.fill_bitmask.side_effect = fill_bitmask
        grammar.accept_tokens.return_value = True

        bitmask = manager_with_reasoner.grammar_bitmask(
            {"req": req}, ["req"], {"req": [10, 99, 123]}
        )

        assert bitmask.tolist() == [[-1, -1], [-1, -1], [2, 2], [3, 3]]
        assert [call.args for call in grammar.accept_tokens.call_args_list] == [
            ("req", [123])
        ]
        grammar.rollback.assert_called_once_with(1)
        assert req.structured_output_request.reasoning_ended is False
        assert [
            list(call.args[1])
            for call in reasoner.is_reasoning_end_streaming.call_args_list
        ] == [[10, 99, 123], [10], [99]]

    def test_grammar_bitmask_tolerates_invalid_post_boundary_spec_token(
        self,
        manager_with_reasoner,
        mock_request_with_structured_output,
    ):
        req = mock_request_with_structured_output
        req.all_token_ids = [1, 2, 3]
        req.structured_output_request.reasoning_ended = False

        reasoner = MockReasoner(tokenizer=Mock())
        reasoner.is_reasoning_end_streaming.side_effect = (
            lambda input_ids, delta_ids: 99 in list(delta_ids)
        )
        req.structured_output_request.reasoner = reasoner

        manager_with_reasoner.vllm_config.speculative_config = Mock(
            num_speculative_tokens=3
        )
        manager_with_reasoner._grammar_bitmask = torch.full(
            (4, 2), -1, dtype=torch.int32
        )

        grammar = req.structured_output_request.grammar

        def fill_bitmask(mask, index):
            mask[index].fill_(index)

        grammar.fill_bitmask.side_effect = fill_bitmask
        grammar.accept_tokens.return_value = False

        bitmask = manager_with_reasoner.grammar_bitmask(
            {"req": req}, ["req"], {"req": [99, 321, 322]}
        )

        assert bitmask.tolist() == [[-1, -1], [1, 1], [-1, -1], [-1, -1]]
        assert [call.args for call in grammar.accept_tokens.call_args_list] == [
            ("req", [321])
        ]
        grammar.rollback.assert_not_called()

    def test_grammar_bitmask_checks_reasoning_boundary_once_on_common_path(
        self,
        manager_with_reasoner,
        mock_request_with_structured_output,
    ):
        req = mock_request_with_structured_output
        req.all_token_ids = [1, 2, 3]
        req.structured_output_request.reasoning_ended = False

        reasoner = MockReasoner(tokenizer=Mock())
        reasoner.is_reasoning_end_streaming.return_value = False
        req.structured_output_request.reasoner = reasoner

        manager_with_reasoner.vllm_config.speculative_config = Mock(
            num_speculative_tokens=3
        )
        manager_with_reasoner._grammar_bitmask = torch.full(
            (4, 2), -1, dtype=torch.int32
        )

        grammar = req.structured_output_request.grammar

        bitmask = manager_with_reasoner.grammar_bitmask(
            {"req": req}, ["req"], {"req": [10, 11, 12]}
        )

        assert bitmask.tolist() == [[-1, -1], [-1, -1], [-1, -1], [-1, -1]]
        assert [
            list(call.args[1])
            for call in reasoner.is_reasoning_end_streaming.call_args_list
        ] == [[10, 11, 12]]
        grammar.accept_tokens.assert_not_called()
        grammar.rollback.assert_not_called()
