# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/lm-sys/FastChat/blob/168ccc29d3f7edc50823016105c024fe2282732a/fastchat/protocol/openai_api_protocol.py
import time
from typing import Any, Literal, TypeAlias

from openai.types.responses import (
    ResponseCodeInterpreterCallCodeDeltaEvent,
    ResponseCodeInterpreterCallCodeDoneEvent,
    ResponseCodeInterpreterCallCompletedEvent,
    ResponseCodeInterpreterCallInProgressEvent,
    ResponseCodeInterpreterCallInterpretingEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseFunctionToolCall,
    ResponseInputItemParam,
    ResponseMcpCallArgumentsDeltaEvent,
    ResponseMcpCallArgumentsDoneEvent,
    ResponseMcpCallCompletedEvent,
    ResponseMcpCallInProgressEvent,
    ResponseOutputItem,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponsePrompt,
    ResponseReasoningTextDeltaEvent,
    ResponseReasoningTextDoneEvent,
    ResponseStatus,
    ResponseWebSearchCallCompletedEvent,
    ResponseWebSearchCallInProgressEvent,
    ResponseWebSearchCallSearchingEvent,
)
from openai.types.responses import (
    ResponseCompletedEvent as OpenAIResponseCompletedEvent,
)
from openai.types.responses import ResponseCreatedEvent as OpenAIResponseCreatedEvent
from openai.types.responses import (
    ResponseInProgressEvent as OpenAIResponseInProgressEvent,
)
from openai.types.responses.tool import Tool
from openai_harmony import Message as OpenAIHarmonyMessage

# Backward compatibility for OpenAI client versions
try:  # For older openai versions (< 1.100.0)
    from openai.types.responses import ResponseTextConfig
except ImportError:  # For newer openai versions (>= 1.100.0)
    from openai.types.responses import ResponseFormatTextConfig as ResponseTextConfig

from openai.types.responses.response import IncompleteDetails, ToolChoice
from openai.types.responses.response_reasoning_item import (
    Content as ResponseReasoningTextContent,
)
from openai.types.shared import Metadata, Reasoning
from pydantic import (
    Field,
    ValidationError,
    field_serializer,
    model_validator,
)

from vllm.config import ModelConfig
from vllm.entrypoints.chat_utils import (
    ChatCompletionMessageParam,
    ChatTemplateContentFormatOption,
)
from vllm.entrypoints.openai.engine.protocol import OpenAIBaseModel
from vllm.exceptions import VLLMValidationError
from vllm.logger import init_logger
from vllm.renderers import ChatParams, TokenizeParams, merge_kwargs
from vllm.sampling_params import (
    RequestOutputKind,
    SamplingParams,
    StructuredOutputsParams,
)
from vllm.utils import random_uuid

logger = init_logger(__name__)

_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1


class InputTokensDetails(OpenAIBaseModel):
    cached_tokens: int
    input_tokens_per_turn: list[int] = Field(default_factory=list)
    cached_tokens_per_turn: list[int] = Field(default_factory=list)


class OutputTokensDetails(OpenAIBaseModel):
    reasoning_tokens: int = 0
    tool_output_tokens: int = 0
    output_tokens_per_turn: list[int] = Field(default_factory=list)
    tool_output_tokens_per_turn: list[int] = Field(default_factory=list)


class ResponseUsage(OpenAIBaseModel):
    input_tokens: int
    input_tokens_details: InputTokensDetails
    output_tokens: int
    output_tokens_details: OutputTokensDetails
    total_tokens: int


def serialize_message(msg):
    """
    Serializes a single message
    """
    if isinstance(msg, dict):
        return msg
    elif hasattr(msg, "to_dict"):
        return msg.to_dict()
    else:
        # fallback to pyandic dump
        return msg.model_dump_json()


def serialize_messages(msgs):
    """
    Serializes multiple messages
    """
    return [serialize_message(msg) for msg in msgs] if msgs else None


class ResponseRawMessageAndToken(OpenAIBaseModel):
    """Class to show the raw message.
    If message / tokens diverge, tokens is the source of truth"""

    message: str
    tokens: list[int]
    type: Literal["raw_message_tokens"] = "raw_message_tokens"


ResponseInputOutputMessage: TypeAlias = (
    list[ChatCompletionMessageParam] | list[ResponseRawMessageAndToken]
)
ResponseInputOutputItem: TypeAlias = ResponseInputItemParam | ResponseOutputItem


class ResponsesRequest(OpenAIBaseModel):
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/responses/create
    background: bool | None = False
    include: (
        list[
            Literal[
                "code_interpreter_call.outputs",
                "computer_call_output.output.image_url",
                "file_search_call.results",
                "message.input_image.image_url",
                "message.output_text.logprobs",
                "reasoning.encrypted_content",
            ],
        ]
        | None
    ) = None
    input: str | list[ResponseInputOutputItem]
    instructions: str | None = None
    max_output_tokens: int | None = None
    max_tool_calls: int | None = None
    metadata: Metadata | None = None
    model: str | None = None
    logit_bias: dict[str, float] | None = None
    parallel_tool_calls: bool | None = True
    previous_response_id: str | None = None
    prompt: ResponsePrompt | None = None
    reasoning: Reasoning | None = None
    service_tier: Literal["auto", "default", "flex", "scale", "priority"] = "auto"
    store: bool | None = True
    stream: bool | None = False
    temperature: float | None = None
    text: ResponseTextConfig | None = None
    tool_choice: ToolChoice = "auto"
    tools: list[Tool] = Field(default_factory=list)
    top_logprobs: int | None = 0
    top_p: float | None = None
    top_k: int | None = None
    truncation: Literal["auto", "disabled"] | None = "disabled"
    user: str | None = None
    skip_special_tokens: bool = True
    include_stop_str_in_output: bool = False
    prompt_cache_key: str | None = Field(
        default=None,
        description=(
            "A key that was used to read from or write to the prompt cache."
            "Note: This field has not been implemented yet "
            "and vLLM will ignore it."
        ),
    )

    # --8<-- [start:responses-extra-params]
    request_id: str = Field(
        default_factory=lambda: f"resp_{random_uuid()}",
        description=(
            "The request_id related to this request. If the caller does "
            "not set it, a random_uuid will be generated. This id is used "
            "through out the inference process and return in response."
        ),
    )
    media_io_kwargs: dict[str, dict[str, Any]] | None = Field(
        default=None,
        description=(
            "Additional kwargs to pass to the media IO connectors, "
            "keyed by modality. Merged with engine-level media_io_kwargs."
        ),
    )
    mm_processor_kwargs: dict[str, Any] | None = Field(
        default=None,
        description=("Additional kwargs to pass to the HF processor."),
    )
    priority: int = Field(
        default=0,
        ge=_INT64_MIN,
        le=_INT64_MAX,
        description=(
            "The priority of the request (lower means earlier handling; "
            "default: 0). Any priority other than 0 will raise an error "
            "if the served model does not use priority scheduling."
        ),
    )
    cache_salt: str | None = Field(
        default=None,
        description=(
            "If specified, the prefix cache will be salted with the provided "
            "string to prevent an attacker to guess prompts in multi-user "
            "environments. The salt should be random, protected from "
            "access by 3rd parties, and long enough to be "
            "unpredictable (e.g., 43 characters base64-encoded, corresponding "
            "to 256 bit)."
        ),
    )
    kv_transfer_params: dict[str, Any] | None = Field(
        default=None,
        description="KVTransfer parameters used for disaggregated serving.",
    )

    enable_response_messages: bool = Field(
        default=False,
        description=(
            "Dictates whether or not to return messages as part of the "
            "response object. Currently only supported for non-background."
        ),
    )
    chat_template_kwargs: dict[str, Any] | None = Field(
        default=None,
        description="Internal field used to pass resolved chat template kwargs.",
        exclude=True,
    )
    # similar to input_messages / output_messages in ResponsesResponse
    # we take in previous_input_messages (ie in harmony format)
    # this cannot be used in conjunction with previous_response_id
    # TODO: consider supporting non harmony messages as well
    previous_input_messages: list[OpenAIHarmonyMessage | dict] | None = None
    structured_outputs: StructuredOutputsParams | None = Field(
        default=None,
        description="Additional kwargs for structured outputs",
    )

    repetition_penalty: float | None = None
    seed: int | None = Field(None, ge=_INT64_MIN, le=_INT64_MAX)
    stop: str | list[str] | None = []
    ignore_eos: bool = False
    vllm_xargs: dict[str, str | int | float | list[str | int | float]] | None = Field(
        default=None,
        description=(
            "Additional request parameters with (list of) string or "
            "numeric values, used by custom extensions."
        ),
    )
    # --8<-- [end:responses-extra-params]

    def _get_reasoning_field(self, field_name: str) -> Any:
        if self.reasoning is None:
            return None
        return getattr(self.reasoning, field_name, None)

    def _get_effective_reasoning_effort(
        self,
    ) -> Literal["none", "low", "medium", "high"] | None:
        enabled = self._get_reasoning_field("enabled")
        if enabled is False:
            return "none"

        effort = self._get_reasoning_field("effort")
        if effort == "minimal":
            return "low"
        if effort in ("none", "low", "medium", "high"):
            return effort
        if enabled is True:
            return "medium"
        return None

    def _get_named_tool_schema(self) -> dict[str, Any] | None:
        normalized = self.get_normalized_tool_choice()
        if not isinstance(normalized, dict):
            return None
        function = normalized.get("function")
        if not isinstance(function, dict):
            return None
        target_name = function.get("name")
        if not isinstance(target_name, str) or not self.tools:
            return None

        for tool in self.tools:
            if getattr(tool, "type", None) != "function":
                continue
            if getattr(tool, "name", None) != target_name:
                continue
            params = getattr(tool, "parameters", None)
            if isinstance(params, dict):
                return params
            return {"type": "object", "additionalProperties": True}
        return None

    def _get_required_tools_schema(self) -> dict[str, Any] | None:
        if self.tool_choice != "required" or not self.tools:
            return None

        variants: list[dict[str, Any]] = []
        for tool in self.tools:
            if getattr(tool, "type", None) != "function":
                continue
            tool_name = getattr(tool, "name", None)
            if not isinstance(tool_name, str):
                continue
            params = getattr(tool, "parameters", None)
            if not isinstance(params, dict):
                params = {"type": "object", "additionalProperties": True}
            variants.append(
                {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "const": tool_name},
                        "parameters": params,
                    },
                    "required": ["name", "parameters"],
                    "additionalProperties": False,
                }
            )

        if not variants:
            return None

        return {
            "type": "array",
            "minItems": 1,
            "items": {"oneOf": variants},
        }

    def get_normalized_tool_choice(self) -> ToolChoice:
        tool_choice = self.tool_choice
        if isinstance(tool_choice, dict):
            if (
                tool_choice.get("type") == "function"
                and "name" in tool_choice
                and "function" not in tool_choice
            ):
                return {
                    "type": "function",
                    "function": {"name": tool_choice["name"]},
                }
            return tool_choice

        if (
            getattr(tool_choice, "type", None) == "function"
            and getattr(tool_choice, "name", None) is not None
            and getattr(tool_choice, "function", None) is None
        ):
            return {
                "type": "function",
                "function": {"name": getattr(tool_choice, "name")},
            }

        return tool_choice

    def build_chat_params(
        self,
        default_template: str | None,
        default_template_content_format: ChatTemplateContentFormatOption,
    ) -> ChatParams:
        from .utils import should_continue_final_message

        # Check if we should continue the final message (partial completion)
        # This enables Anthropic-style partial message completion where the
        # user provides an incomplete assistant message to continue from.
        continue_final = should_continue_final_message(self.input)

        reasoning_effort = self._get_effective_reasoning_effort()
        chat_template_kwargs: dict[str, Any] = {}
        if reasoning_effort == "none":
            chat_template_kwargs["enable_thinking"] = False
            chat_template_kwargs["force_nonempty_content"] = True
        elif reasoning_effort == "low":
            chat_template_kwargs["enable_thinking"] = True
            chat_template_kwargs["low_effort"] = True
        elif reasoning_effort in ("medium", "high"):
            chat_template_kwargs["enable_thinking"] = True

        resolved_chat_template_kwargs = merge_kwargs(
            chat_template_kwargs,
            dict(
                add_generation_prompt=not continue_final,
                continue_final_message=continue_final,
                reasoning_effort=reasoning_effort,
            ),
        )
        self.chat_template_kwargs = resolved_chat_template_kwargs

        return ChatParams(
            chat_template=default_template,
            chat_template_content_format=default_template_content_format,
            chat_template_kwargs=resolved_chat_template_kwargs,
            media_io_kwargs=self.media_io_kwargs,
        )

    def build_tok_params(self, model_config: ModelConfig) -> TokenizeParams:
        return TokenizeParams(
            max_total_tokens=model_config.max_model_len,
            max_output_tokens=self.max_output_tokens or 0,
            truncate_prompt_tokens=-1 if self.truncation != "disabled" else None,
            max_total_tokens_param="max_model_len",
            max_output_tokens_param="max_output_tokens",
        )

    _DEFAULT_SAMPLING_PARAMS = {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
    }

    def to_sampling_params(
        self,
        default_max_tokens: int,
        default_sampling_params: dict | None = None,
    ) -> SamplingParams:
        if self.max_output_tokens is None:
            max_tokens = default_max_tokens
        else:
            max_tokens = min(self.max_output_tokens, default_max_tokens)

        default_sampling_params = default_sampling_params or {}
        if (temperature := self.temperature) is None:
            temperature = default_sampling_params.get(
                "temperature", self._DEFAULT_SAMPLING_PARAMS["temperature"]
            )
        if (top_p := self.top_p) is None:
            top_p = default_sampling_params.get(
                "top_p", self._DEFAULT_SAMPLING_PARAMS["top_p"]
            )
        if (top_k := self.top_k) is None:
            top_k = default_sampling_params.get(
                "top_k", self._DEFAULT_SAMPLING_PARAMS["top_k"]
            )

        if (repetition_penalty := self.repetition_penalty) is None:
            repetition_penalty = default_sampling_params.get("repetition_penalty", 1.0)

        stop_token_ids = default_sampling_params.get("stop_token_ids")

        # Structured output
        structured_outputs = self.structured_outputs

        # Also check text.format for OpenAI-style json_schema
        if self.text is not None and self.text.format is not None:
            if structured_outputs is not None:
                raise VLLMValidationError(
                    "Cannot specify both structured_outputs and text.format",
                    parameter="structured_outputs",
                )
            response_format = self.text.format
            if (
                response_format.type == "json_schema"
                and response_format.schema_ is not None
            ):
                structured_outputs = StructuredOutputsParams(
                    json=response_format.schema_,  # type: ignore[call-arg]
                    disable_any_whitespace=True,
                    disable_additional_properties=True,
                    # --follow-imports skip hides the class definition but also hides
                    # multiple third party conflicts, so best of both evils
                )

        if structured_outputs is None:
            named_tool_schema = self._get_named_tool_schema()
            if named_tool_schema is not None:
                structured_outputs = StructuredOutputsParams(
                    json=named_tool_schema,
                    disable_any_whitespace=True,
                    disable_additional_properties=True,
                )
            else:
                required_tools_schema = self._get_required_tools_schema()
                if required_tools_schema is not None:
                    structured_outputs = StructuredOutputsParams(
                        json=required_tools_schema,
                        disable_any_whitespace=True,
                        disable_additional_properties=True,
                    )

        stop = self.stop if self.stop else []
        if isinstance(stop, str):
            stop = [stop]

        extra_args = dict(self.vllm_xargs or {})
        if self.kv_transfer_params:
            extra_args["kv_transfer_params"] = self.kv_transfer_params
        normalized_tool_choice = self.get_normalized_tool_choice()
        if self.tool_choice == "required" or (
            isinstance(normalized_tool_choice, dict)
            and normalized_tool_choice.get("type") == "function"
        ):
            extra_args["disable_spec_decode"] = True

        output_kind = (
            RequestOutputKind.DELTA if self.stream else RequestOutputKind.FINAL_ONLY
        )
        if self.stream and (
            self.tool_choice == "required"
            or (
                isinstance(normalized_tool_choice, dict)
                and normalized_tool_choice.get("type") == "function"
            )
        ):
            output_kind = RequestOutputKind.FINAL_ONLY
        return SamplingParams.from_optional(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            logprobs=self.top_logprobs if self.is_include_output_logprobs() else None,
            stop_token_ids=stop_token_ids,
            stop=stop,
            repetition_penalty=repetition_penalty,
            seed=self.seed,
            ignore_eos=self.ignore_eos,
            output_kind=output_kind,
            structured_outputs=structured_outputs,
            logit_bias=self.logit_bias,
            extra_args=extra_args or None,
            skip_clone=True,  # Created fresh per request, safe to skip clone
            skip_special_tokens=self.skip_special_tokens,
            include_stop_str_in_output=self.include_stop_str_in_output,
        )

    def is_include_output_logprobs(self) -> bool:
        """Check if the request includes output logprobs."""
        if self.include is None:
            return False
        return (
            isinstance(self.include, list)
            and "message.output_text.logprobs" in self.include
        )

    @model_validator(mode="before")
    @classmethod
    def validate_priority_field(cls, data):
        if not isinstance(data, dict):
            return data
        if data.get("priority", 0) != 0:
            raise VLLMValidationError(
                "Request body field `priority` is not supported.",
                parameter="priority",
                value=data.get("priority"),
            )
        return data

    @model_validator(mode="before")
    @classmethod
    def validate_background(cls, data):
        if not data.get("background"):
            return data
        if not data.get("store", True):
            raise VLLMValidationError(
                "background can only be used when `store` is true",
                parameter="background",
            )
        return data

    @model_validator(mode="before")
    @classmethod
    def validate_prompt(cls, data):
        if data.get("prompt") is not None:
            raise VLLMValidationError(
                "prompt template is not supported", parameter="prompt"
            )
        return data

    @model_validator(mode="before")
    @classmethod
    def check_cache_salt_support(cls, data):
        if data.get("cache_salt") is not None and (
            not isinstance(data["cache_salt"], str) or not data["cache_salt"]
        ):
            raise VLLMValidationError(
                "Parameter 'cache_salt' must be a non-empty string if provided.",
                parameter="cache_salt",
            )
        return data

    @model_validator(mode="before")
    @classmethod
    def function_call_parsing(cls, data):
        """Parse function_call dictionaries into ResponseFunctionToolCall objects.
        This ensures Pydantic can properly resolve union types in the input field.
        Function calls provided as dicts are converted to ResponseFunctionToolCall
        objects before validation, while invalid structures are left for Pydantic
        to reject with appropriate error messages.
        """

        input_data = data.get("input")

        # Early return for None, strings, or bytes
        # (strings are iterable but shouldn't be processed)
        if input_data is None or isinstance(input_data, (str, bytes)):
            return data

        # Convert iterators (like ValidatorIterator) to list
        if not isinstance(input_data, list):
            try:
                input_data = list(input_data)
            except TypeError:
                # Not iterable, leave as-is for Pydantic to handle
                return data

        processed_input = []
        for item in input_data:
            if isinstance(item, dict) and item.get("type") == "function_call":
                try:
                    processed_input.append(ResponseFunctionToolCall(**item))
                except ValidationError:
                    # Let Pydantic handle validation for malformed function calls
                    logger.debug(
                        "Failed to parse function_call to ResponseFunctionToolCall, "
                        "leaving for Pydantic validation"
                    )
                    processed_input.append(item)
            else:
                processed_input.append(item)

        data["input"] = processed_input
        return data

    @model_validator(mode="before")
    @classmethod
    def normalize_function_tools(cls, data):
        if not isinstance(data, dict):
            return data

        tools = data.get("tools")
        if not isinstance(tools, list):
            return data

        normalized_tools: list[Any] = []
        changed = False
        for tool in tools:
            if (
                isinstance(tool, dict)
                and tool.get("type") == "function"
                and isinstance(tool.get("function"), dict)
                and "name" not in tool
            ):
                fn = dict(tool["function"])
                normalized_tools.append(
                    {
                        "type": "function",
                        "name": fn.get("name"),
                        "description": fn.get("description"),
                        "parameters": fn.get("parameters"),
                        "strict": fn.get("strict"),
                    }
                )
                changed = True
            else:
                normalized_tools.append(tool)

        if changed:
            data["tools"] = normalized_tools
        return data


class ResponsesResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"resp_{random_uuid()}")
    created_at: int = Field(default_factory=lambda: int(time.time()))
    # error: Optional[ResponseError] = None
    incomplete_details: IncompleteDetails | None = None
    instructions: str | None = None
    metadata: Metadata | None = None
    model: str
    object: Literal["response"] = "response"
    output: list[ResponseOutputItem]
    parallel_tool_calls: bool
    temperature: float
    tool_choice: ToolChoice
    tools: list[Tool]
    top_p: float
    background: bool
    max_output_tokens: int
    max_tool_calls: int | None = None
    previous_response_id: str | None = None
    prompt: ResponsePrompt | None = None
    reasoning: Reasoning | None = None
    service_tier: Literal["auto", "default", "flex", "scale", "priority"]
    status: ResponseStatus
    text: ResponseTextConfig | None = None
    top_logprobs: int | None = None
    truncation: Literal["auto", "disabled"]
    usage: ResponseUsage | None = None
    user: str | None = None

    # --8<-- [start:responses-response-extra-params]
    # These are populated when enable_response_messages is set to True
    # NOTE: custom serialization is needed
    # see serialize_input_messages and serialize_output_messages
    input_messages: ResponseInputOutputMessage | None = Field(
        default=None,
        description=(
            "If enable_response_messages, we can show raw token input to model."
        ),
    )
    output_messages: ResponseInputOutputMessage | None = Field(
        default=None,
        description=(
            "If enable_response_messages, we can show raw token output of model."
        ),
    )
    # --8<-- [end:responses-response-extra-params]

    # NOTE: openAI harmony doesn't serialize TextContent properly,
    # TODO: this fixes for TextContent, but need to verify for tools etc
    # https://github.com/openai/harmony/issues/78
    @field_serializer("output_messages", when_used="json")
    def serialize_output_messages(self, msgs, _info):
        return serialize_messages(msgs)

    # NOTE: openAI harmony doesn't serialize TextContent properly, this fixes it
    # https://github.com/openai/harmony/issues/78
    @field_serializer("input_messages", when_used="json")
    def serialize_input_messages(self, msgs, _info):
        return serialize_messages(msgs)

    @classmethod
    def from_request(
        cls,
        request: ResponsesRequest,
        sampling_params: SamplingParams,
        model_name: str,
        created_time: int,
        output: list[ResponseOutputItem],
        status: ResponseStatus,
        usage: ResponseUsage | None = None,
        input_messages: ResponseInputOutputMessage | None = None,
        output_messages: ResponseInputOutputMessage | None = None,
    ) -> "ResponsesResponse":
        incomplete_details: IncompleteDetails | None = None
        if status == "incomplete":
            incomplete_details = IncompleteDetails(reason="max_output_tokens")
        # TODO: implement the other reason for incomplete_details,
        # which is content_filter
        # incomplete_details = IncompleteDetails(reason='content_filter')
        return cls(
            id=request.request_id,
            created_at=created_time,
            incomplete_details=incomplete_details,
            instructions=request.instructions,
            metadata=request.metadata,
            model=model_name,
            output=output,
            input_messages=input_messages,
            output_messages=output_messages,
            parallel_tool_calls=request.parallel_tool_calls,
            temperature=sampling_params.temperature,
            tool_choice=request.tool_choice,
            tools=request.tools,
            top_p=sampling_params.top_p,
            background=request.background,
            max_output_tokens=sampling_params.max_tokens,
            max_tool_calls=request.max_tool_calls,
            previous_response_id=request.previous_response_id,
            prompt=request.prompt,
            reasoning=request.reasoning,
            service_tier=request.service_tier,
            status=status,
            text=request.text,
            top_logprobs=sampling_params.logprobs,
            truncation=request.truncation,
            user=request.user,
            usage=usage,
        )


# TODO: this code can be removed once
# https://github.com/openai/openai-python/issues/2634 has been resolved
class ResponseReasoningPartDoneEvent(OpenAIBaseModel):
    content_index: int
    """The index of the content part that is done."""

    item_id: str
    """The ID of the output item that the content part was added to."""

    output_index: int
    """The index of the output item that the content part was added to."""

    part: ResponseReasoningTextContent
    """The content part that is done."""

    sequence_number: int
    """The sequence number of this event."""

    type: Literal["response.reasoning_part.done"]
    """The type of the event. Always `response.reasoning_part.done`."""


# TODO: this code can be removed once
# https://github.com/openai/openai-python/issues/2634 has been resolved
class ResponseReasoningPartAddedEvent(OpenAIBaseModel):
    content_index: int
    """The index of the content part that is done."""

    item_id: str
    """The ID of the output item that the content part was added to."""

    output_index: int
    """The index of the output item that the content part was added to."""

    part: ResponseReasoningTextContent
    """The content part that is done."""

    sequence_number: int
    """The sequence number of this event."""

    type: Literal["response.reasoning_part.added"]
    """The type of the event. Always `response.reasoning_part.added`."""


# vLLM Streaming Events
# Note: we override the response type with the vLLM ResponsesResponse type
class ResponseCompletedEvent(OpenAIResponseCompletedEvent):
    response: ResponsesResponse  # type: ignore[override]


class ResponseCreatedEvent(OpenAIResponseCreatedEvent):
    response: ResponsesResponse  # type: ignore[override]


class ResponseInProgressEvent(OpenAIResponseInProgressEvent):
    response: ResponsesResponse  # type: ignore[override]


StreamingResponsesResponse: TypeAlias = (
    ResponseCreatedEvent
    | ResponseInProgressEvent
    | ResponseCompletedEvent
    | ResponseOutputItemAddedEvent
    | ResponseOutputItemDoneEvent
    | ResponseContentPartAddedEvent
    | ResponseContentPartDoneEvent
    | ResponseReasoningTextDeltaEvent
    | ResponseReasoningTextDoneEvent
    | ResponseReasoningPartAddedEvent
    | ResponseReasoningPartDoneEvent
    | ResponseCodeInterpreterCallInProgressEvent
    | ResponseCodeInterpreterCallCodeDeltaEvent
    | ResponseWebSearchCallInProgressEvent
    | ResponseWebSearchCallSearchingEvent
    | ResponseWebSearchCallCompletedEvent
    | ResponseCodeInterpreterCallCodeDoneEvent
    | ResponseCodeInterpreterCallInterpretingEvent
    | ResponseCodeInterpreterCallCompletedEvent
    | ResponseMcpCallArgumentsDeltaEvent
    | ResponseMcpCallArgumentsDoneEvent
    | ResponseMcpCallInProgressEvent
    | ResponseMcpCallCompletedEvent
)
