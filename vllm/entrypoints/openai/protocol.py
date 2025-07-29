# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/lm-sys/FastChat/blob/168ccc29d3f7edc50823016105c024fe2282732a/fastchat/protocol/openai_api_protocol.py
import json
import time
from http import HTTPStatus
from typing import Annotated, Any, ClassVar, Literal, Optional, Union, List

import regex as re
import torch
from fastapi import HTTPException, UploadFile
from pydantic import (BaseModel, ConfigDict, Field, TypeAdapter,
                      ValidationInfo, field_validator, model_validator)
from typing_extensions import TypeAlias

from vllm import envs
from vllm.entrypoints.chat_utils import (ChatCompletionMessageParam,
                                         random_tool_call_id)
from vllm.logger import init_logger
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import (BeamSearchParams, GuidedDecodingParams,
                                  RequestOutputKind, SamplingParams)
from vllm.sequence import Logprob
from vllm.utils import random_uuid, resolve_obj_by_qualname

logger = init_logger(__name__)

_LONG_INFO = torch.iinfo(torch.long)


class OpenAIBaseModel(BaseModel):
    # OpenAI API does allow extra fields
    model_config = ConfigDict(extra="allow")

    # Cache class field names
    field_names: ClassVar[Optional[set[str]]] = None

    @model_validator(mode="wrap")
    @classmethod
    def __log_extra_fields__(cls, data, handler):
        result = handler(data)
        if not isinstance(data, dict):
            return result
        field_names = cls.field_names
        if field_names is None:
            # Get all class field names and their potential aliases
            field_names = set()
            for field_name, field in cls.model_fields.items():
                field_names.add(field_name)
                if alias := getattr(field, "alias", None):
                    field_names.add(alias)
            cls.field_names = field_names

        # Compare against both field names and aliases
        if any(k not in field_names for k in data):
            logger.warning(
                "The following fields were present in the request "
                "but ignored: %s",
                data.keys() - field_names,
            )
        return result


class ErrorResponse(OpenAIBaseModel):
    object: str = "error"
    message: str
    type: str
    param: Optional[str] = None
    code: int


class ModelPermission(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"modelperm-{random_uuid()}")
    object: str = "model_permission"
    created: int = Field(default_factory=lambda: int(time.time()))
    allow_create_engine: bool = False
    allow_sampling: bool = True
    allow_logprobs: bool = True
    allow_search_indices: bool = False
    allow_view: bool = True
    allow_fine_tuning: bool = False
    organization: str = "*"
    group: Optional[str] = None
    is_blocking: bool = False


class ModelCard(OpenAIBaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "vllm"
    root: Optional[str] = None
    parent: Optional[str] = None
    max_model_len: Optional[int] = None
    permission: list[ModelPermission] = Field(default_factory=list)


class ModelList(OpenAIBaseModel):
    object: str = "list"
    data: list[ModelCard] = Field(default_factory=list)


class PromptTokenUsageInfo(OpenAIBaseModel):
    cached_tokens: Optional[int] = None


class UsageInfo(OpenAIBaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0
    prompt_tokens_details: Optional[PromptTokenUsageInfo] = None


class RequestResponseMetadata(BaseModel):
    request_id: str
    final_usage_info: Optional[UsageInfo] = None


class JsonSchemaResponseFormat(OpenAIBaseModel):
    name: str
    description: Optional[str] = None
    # schema is the field in openai but that causes conflicts with pydantic so
    # instead use json_schema with an alias
    json_schema: Optional[dict[str, Any]] = Field(default=None, alias='schema')
    strict: Optional[bool] = None


class StructuralTag(OpenAIBaseModel):
    begin: str
    # schema is the field, but that causes conflicts with pydantic so
    # instead use structural_tag_schema with an alias
    structural_tag_schema: Optional[dict[str, Any]] = Field(default=None,
                                                            alias="schema")
    end: str


class StructuralTagResponseFormat(OpenAIBaseModel):
    type: Literal["structural_tag"]
    structures: list[StructuralTag]
    triggers: list[str]


class ResponseFormat(OpenAIBaseModel):
    # type must be "json_schema", "json_object", or "text"
    type: Literal["text", "json_object", "json_schema"]
    json_schema: Optional[JsonSchemaResponseFormat] = None


AnyResponseFormat = Union[ResponseFormat, StructuralTagResponseFormat]


class StreamOptions(OpenAIBaseModel):
    include_usage: Optional[bool] = True
    continuous_usage_stats: Optional[bool] = False


class FunctionDefinition(OpenAIBaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[dict[str, Any]] = None


class ChatCompletionToolsParam(OpenAIBaseModel):
    type: Literal["function"] = "function"
    function: FunctionDefinition


class ChatCompletionNamedFunction(OpenAIBaseModel):
    name: str


class ChatCompletionNamedToolChoiceParam(OpenAIBaseModel):
    function: ChatCompletionNamedFunction
    type: Literal["function"] = "function"


# extra="forbid" is a workaround to have kwargs as a field,
# see https://github.com/pydantic/pydantic/issues/3125
class LogitsProcessorConstructor(BaseModel):
    qualname: str
    args: Optional[list[Any]] = None
    kwargs: Optional[dict[str, Any]] = None

    model_config = ConfigDict(extra="forbid")


LogitsProcessors = list[Union[str, LogitsProcessorConstructor]]

NEWLINE_TOKENS = {12291, 8199, 20487, 57354, 71693, 14350, 126989, 2064, 10260, 26644, 114710, 10263, 106525, 81952, 116768, 10274, 51235, 28710, 94246, 79914, 86059, 2092, 79917, 18479, 18481, 96306, 32819, 2100, 14390, 67639, 73782, 94265, 110650, 122935, 2116, 12357, 114758, 86087, 67656, 4169, 100423, 104519, 106570, 114759, 124997, 102479, 12368, 14416, 14417, 38997, 98389, 34903, 67673, 102490, 26720, 94306, 12390, 61546, 53355, 32876, 98414, 26740, 90228, 94325, 39031, 127095, 18554, 47229, 6273, 51330, 4227, 2180, 98437, 100485, 112779, 96397, 108686, 43151, 4241, 110737, 69780, 49301, 86173, 4256, 118944, 18595, 16548, 102565, 12455, 114859, 92334, 104623, 6320, 118960, 118963, 55476, 67766, 14519, 55479, 123067, 61631, 32960, 2241, 63685, 114885, 116935, 63691, 125131, 57551, 104655, 73942, 100571, 16606, 73954, 16616, 114925, 14576, 51440, 20723, 4342, 67830, 14588, 123134, 33024, 4353, 39170, 26885, 67846, 123142, 129286, 55561, 28938, 37131, 84236, 12560, 88337, 43282, 6422, 24866, 90404, 14630, 45352, 20777, 31020, 20783, 14642, 65846, 8503, 4410, 100670, 14657, 53569, 2373, 20806, 90445, 6478, 104785, 41301, 20822, 115029, 8538, 6491, 47456, 94561, 24931, 72038, 117098, 31086, 43374, 43378, 80242, 14708, 61813, 82293, 53623, 92535, 110964, 39293, 94592, 104833, 72068, 72071, 10632, 84360, 98696, 55692, 22926, 27022, 31120, 35215, 117138, 74136, 123293, 37278, 86430, 106910, 37281, 125346, 113060, 68007, 65960, 88489, 98729, 43438, 33203, 39347, 102839, 57784, 49593, 76218, 108983, 68030, 92606, 16832, 100798, 14790, 43462, 37321, 4554, 86475, 96716, 14797, 43472, 29141, 12758, 2519, 109013, 14810, 16858, 72156, 10717, 61918, 125407, 80352, 84451, 12776, 76264, 92650, 6637, 80365, 125422, 8691, 104947, 92665, 23034, 14843, 12795, 14845, 72189, 14847, 107007, 66050, 64006, 4615, 12807, 74247, 80390, 123401, 14863, 25104, 90641, 35346, 37395, 2580, 94736, 84503, 2591, 102946, 76323, 74283, 18988, 70189, 121388, 125483, 14896, 98865, 100915, 90676, 19002, 35390, 80447, 47680, 98879, 86596, 100932, 27207, 2634, 43597, 2638, 107087, 2640, 4688, 121426, 82515, 113236, 8789, 21078, 8791, 88661, 127578, 21090, 82531, 19044, 84578, 94822, 33383, 96867, 115307, 25197, 62062, 64112, 10869, 72310, 8824, 14971, 17020, 43646, 2687, 78464, 66177, 12930, 27266, 86659, 96894, 119428, 129666, 8841, 123531, 29324, 103055, 2706, 113300, 111257, 31391, 82592, 37541, 15014, 49833, 2731, 96940, 68269, 2734, 115377, 94903, 21181, 62142, 31423, 86720, 6849, 43715, 45763, 72388, 76485, 92867, 94919, 17100, 51919, 49872, 60113, 58067, 27350, 117462, 41690, 10971, 53979, 56028, 10974, 31454, 53982, 115419, 39653, 78566, 127727, 31473, 35571, 60147, 64244, 35574, 97015, 119546, 2812, 31484, 84733, 127743, 45825, 2820, 21253, 78597, 27399, 29448, 11017, 97033, 6923, 27404, 35596, 86796, 101126, 117520, 84754, 2838, 35608, 8985, 76569, 82713, 125721, 70433, 49958, 29479, 82729, 2861, 60206, 37679, 66357, 113462, 2871, 74551, 119606, 19261, 13118, 78657, 97091, 74566, 113483, 6991, 47952, 35669, 11095, 35674, 105306, 95068, 13149, 2913, 35683, 9060, 11108, 31588, 66404, 23400, 107366, 123751, 33645, 127854, 58226, 9075, 47988, 25461, 21366, 80758, 19323, 62332, 107387, 115580, 31624, 103304, 25482, 54158, 68494, 50068, 93078, 11159, 13207, 80791, 115607, 56219, 97181, 117662, 21407, 41890, 33699, 84898, 86947, 129960, 78768, 25525, 19383, 19384, 60343, 74681, 119736, 58303, 113599, 19394, 95172, 5062, 119750, 109514, 60363, 15308, 95179, 43982, 121809, 76754, 9171, 23507, 62421, 80855, 23513, 70623, 76772, 74726, 27624, 100335, 111595, 1010, 7154, 60403, 80888, 44025, 29690, 97278, 7168, 11270, 5127, 62473, 58380, 54285, 76816, 107537, 107539, 3092, 56343, 39960, 44059, 3100, 115739, 19486, 44065, 101411, 31782, 89128, 23593, 105512, 109611, 11308, 125994, 68655, 117811, 87093, 121909, 7223, 9273, 44090, 7227, 66620, 107577, 130110, 3135, 11330, 9283, 80962, 97347, 99397, 23626, 25676, 95309, 123980, 29775, 115792, 50259, 68692, 87126, 27736, 42076, 19551, 19553, 99425, 58467, 103527, 66666, 50283, 95338, 19565, 99437, 44143, 130154, 99442, 117876, 40058, 50298, 111739, 11389, 11390, 7295, 68735, 7297, 87168, 66691, 97412, 23685, 58502, 126084, 78984, 19594, 23691, 76944, 19602, 11411, 25748, 101522, 109715, 130199, 130200, 40089, 52377, 115869, 19614, 17572, 9381, 76968, 107688, 79020, 33966, 13487, 50352, 111791, 58547, 93363, 91318, 54455, 52408, 56504, 5306, 76984, 83135, 70848, 25799, 11471, 38097, 21714, 60625, 40151, 79063, 91352, 5338, 111832, 77021, 46302, 9439, 50400, 3297, 3298, 25828, 52454, 23784, 36080, 107760, 1267, 3318, 17655, 118010, 15611, 70909, 111870, 120061, 97536, 60673, 87298, 93441, 115976, 124171, 7437, 7438, 48403, 5396, 34069, 58643, 64790, 66835, 109848, 46366, 52511, 21794, 60707, 21796, 44325, 83235, 93475, 105766, 23849, 101673, 46380, 27949, 19758, 103726, 68912, 56626, 19767, 40247, 1338, 5439, 73024, 87362, 25923, 54597, 7498, 1355, 116042, 9549, 9551, 5457, 56657, 70993, 81235, 1365, 34133, 19802, 34138, 52572, 32093, 109916, 120155, 89445, 73064, 11625, 56683, 111979, 109940, 21877, 3448, 56697, 71032, 103804, 64895, 71040, 111999, 60802, 73091, 48516, 62853, 17798, 73093, 7562, 32140, 109964, 85394, 36243, 118171, 7580, 13724, 13725, 21916, 60830, 71069, 71071, 114080, 118177, 3493, 128422, 21927, 73127, 3500, 1456, 21942, 120246, 1468, 36285, 32194, 3523, 36297, 50634, 19920, 15829, 93653, 30168, 128472, 3546, 17882, 69082, 114140, 101854, 1512, 11753, 9706, 3563, 13803, 46570, 60907, 1520, 36336, 22004, 69108, 124407, 48632, 95738, 52733, 85501, 75265, 30211, 99843, 24071, 48647, 87559, 122375, 32268, 128535, 1561, 48668, 11810, 32291, 1572, 83494, 38439, 95783, 114219, 89644, 85551, 120367, 101940, 75319, 3640, 52797, 1600, 91712, 28230, 3655, 20039, 22087, 34378, 38471, 46665, 65102, 75342, 56913, 38482, 116306, 120408, 1626, 34395, 7772, 73307, 83549, 30303, 91740, 54881, 28258, 120413, 13926, 83558, 1640, 1641, 104042, 106092, 11886, 28272, 56950, 65142, 44664, 24185, 99959, 106108, 128638, 59009, 11906, 26244, 65159, 24202, 56977, 34450, 61074, 24212, 102038, 93852, 81566, 18081, 46753, 79523, 50852, 9893, 69285, 85665, 104104, 61100, 7854, 28335, 128687, 61105, 102066, 26293, 24246, 38582, 3768, 61112, 57018, 81592, 93884, 22205, 126654, 124608, 89793, 38594, 114377, 28364, 130764, 20174, 63183, 14032, 50898, 9940, 28378, 46811, 102109, 57054, 124643, 83684, 44775, 55015, 3817, 79594, 48875, 67307, 3824, 110322, 57075, 67316, 9973, 16117, 20213, 12024, 55031, 120568, 69371, 32508, 100092, 104190, 130808, 65280, 38659, 12039, 1801, 28426, 77579, 53004, 61198, 32530, 108306, 59158, 67350, 3864, 30489, 130842, 102173, 5920, 16161, 1826, 69411, 87840, 87842, 16166, 24362, 3885, 18225, 128818, 32563, 1844, 89907, 53047, 71482, 108346, 44861, 69437, 71488, 69441, 89922, 94017, 100167, 32584, 1877, 110423, 124760, 106331, 53084, 63325, 42846, 65374, 100191, 20321, 38754, 106341, 118631, 100202, 116588, 112493, 14190, 36719, 46958, 71537, 102254, 89974, 104312, 12156, 38781, 10110, 3971, 71556, 40838, 112518, 65416, 36745, 57226, 87946, 10127, 92048, 3989, 34710, 75670, 118680, 20377, 24474, 71578, 38816, 1953, 71586, 24483, 6052, 83877, 110496, 128930, 67499, 53165, 28590, 122797, 20400, 126895, 102323, 30645, 10166, 63413, 92089, 114617, 49083, 57277, 120765, 55234, 20419, 40900, 79811, 88002, 38858, 4043, 38859, 65483, 57294, 83917, 104396, 2002, 26579, 30679, 120791, 88025, 57311, 26592, 30689, 12260, 100324, 53223, 71657, 2030, 4078, 14320, 22512, 28656, 8179, 4084, 32753, 67568, 79855, 61432, 122875, 116734}

class ThinkingLogitsProcessor:
    """A logits processor that limit the number of thinking tokens."""

    def __init__(self, max_tokens: int, min_tokens: int):
        self.is_thinking_done = False
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens

    def __call__(
        self,
        input_ids: List[int],
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Process the logits to enforce </think> token when needed.

        Args:
            input_ids: List of input token IDs.
            logits: Tensor of logits for the next token.

        Returns:
            Processed logits tensor.
        """
        if not self.is_thinking_done:
            if len(input_ids) >= self.min_tokens:
                if input_ids[-1] in NEWLINE_TOKENS or len(input_ids) >= self.max_tokens:
                    # "</think>\n\n" is [1885, 74045, 3318]
                    # 1. first we force </
                    # 2. then we force think>
                    # 3. finally we force \n\n

                    logits = torch.full_like(logits, float('-inf'))

                    # state 2
                    if input_ids[-1] == 1885:
                        logits[74045] = 1.0
                    # state 3
                    elif input_ids[-1] == 74045:
                        logits[3318] = 1.0
                        self.is_thinking_done = True
                    # state 1
                    else:
                        logits[1885] = 1.0

        return logits

def get_logits_processors(processors: Optional[LogitsProcessors],
                          pattern: Optional[str],
                          max_thinking_tokens: Optional[int] = None,
                          min_thinking_tokens: Optional[int] = None) -> Optional[list[Any]]:
    logits_processors = []
    if max_thinking_tokens is not None and min_thinking_tokens is not None:
        logits_processors.append(ThinkingLogitsProcessor(max_tokens=max_thinking_tokens, min_tokens=min_thinking_tokens))
    if processors and pattern:
        for processor in processors:
            qualname = processor if isinstance(processor,
                                               str) else processor.qualname
            if not re.match(pattern, qualname):
                raise ValueError(
                    f"Logits processor '{qualname}' is not allowed by this "
                    "server. See --logits-processor-pattern engine argument "
                    "for more information.")
            try:
                logits_processor = resolve_obj_by_qualname(qualname)
            except Exception as e:
                raise ValueError(
                    f"Logits processor '{qualname}' could not be resolved: {e}"
                ) from e
            if isinstance(processor, LogitsProcessorConstructor):
                logits_processor = logits_processor(*processor.args or [],
                                                    **processor.kwargs or {})
            logits_processors.append(logits_processor)
        return logits_processors
    elif processors:
        raise ValueError(
            "The `logits_processors` argument is not supported by this "
            "server. See --logits-processor-pattern engine argugment "
            "for more information.")
    return logits_processors


class ChatCompletionRequest(OpenAIBaseModel):
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/chat/create
    messages: list[ChatCompletionMessageParam]
    model: Optional[str] = None
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[dict[str, float]] = None
    logprobs: Optional[bool] = False
    top_logprobs: Optional[int] = 0
    # TODO(#9845): remove max_tokens when field is removed from OpenAI API
    max_tokens: Optional[int] = Field(
        default=None,
        deprecated=
        'max_tokens is deprecated in favor of the max_completion_tokens field')
    max_completion_tokens: Optional[int] = None
    n: Optional[int] = 1
    presence_penalty: Optional[float] = 0.0
    response_format: Optional[AnyResponseFormat] = None
    seed: Optional[int] = Field(None, ge=_LONG_INFO.min, le=_LONG_INFO.max)
    stop: Optional[Union[str, list[str]]] = []
    stream: Optional[bool] = False
    stream_options: Optional[StreamOptions] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    tools: Optional[list[ChatCompletionToolsParam]] = None
    tool_choice: Optional[Union[
        Literal["none"],
        Literal["auto"],
        Literal["required"],
        ChatCompletionNamedToolChoiceParam,
    ]] = "none"

    # NOTE this will be ignored by vLLM -- the model determines the behavior
    parallel_tool_calls: Optional[bool] = False
    user: Optional[str] = None

    # --8<-- [start:chat-completion-sampling-params]
    best_of: Optional[int] = None
    use_beam_search: bool = False
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    length_penalty: float = 1.0
    stop_token_ids: Optional[list[int]] = []
    include_stop_str_in_output: bool = False
    ignore_eos: bool = False
    min_tokens: int = 0
    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True
    truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]] = None
    prompt_logprobs: Optional[int] = None
    allowed_token_ids: Optional[list[int]] = None
    # --8<-- [end:chat-completion-sampling-params]

    # --8<-- [start:chat-completion-extra-params]
    echo: bool = Field(
        default=False,
        description=(
            "If true, the new message will be prepended with the last message "
            "if they belong to the same role."),
    )
    add_generation_prompt: bool = Field(
        default=True,
        description=
        ("If true, the generation prompt will be added to the chat template. "
         "This is a parameter used by chat template in tokenizer config of the "
         "model."),
    )
    continue_final_message: bool = Field(
        default=False,
        description=
        ("If this is set, the chat will be formatted so that the final "
         "message in the chat is open-ended, without any EOS tokens. The "
         "model will continue this message rather than starting a new one. "
         "This allows you to \"prefill\" part of the model's response for it. "
         "Cannot be used at the same time as `add_generation_prompt`."),
    )
    add_special_tokens: bool = Field(
        default=False,
        description=(
            "If true, special tokens (e.g. BOS) will be added to the prompt "
            "on top of what is added by the chat template. "
            "For most models, the chat template takes care of adding the "
            "special tokens so this should be set to false (as is the "
            "default)."),
    )
    documents: Optional[list[dict[str, str]]] = Field(
        default=None,
        description=
        ("A list of dicts representing documents that will be accessible to "
         "the model if it is performing RAG (retrieval-augmented generation)."
         " If the template does not support RAG, this argument will have no "
         "effect. We recommend that each document should be a dict containing "
         "\"title\" and \"text\" keys."),
    )
    chat_template: Optional[str] = Field(
        default=None,
        description=(
            "A Jinja template to use for this conversion. "
            "As of transformers v4.44, default chat template is no longer "
            "allowed, so you must provide a chat template if the tokenizer "
            "does not define one."),
    )
    chat_template_kwargs: Optional[dict[str, Any]] = Field(
        default=None,
        description=("Additional kwargs to pass to the template renderer. "
                     "Will be accessible by the chat template."),
    )
    mm_processor_kwargs: Optional[dict[str, Any]] = Field(
        default=None,
        description=("Additional kwargs to pass to the HF processor."),
    )
    guided_json: Optional[Union[str, dict, BaseModel]] = Field(
        default=None,
        description=("If specified, the output will follow the JSON schema."),
    )
    guided_regex: Optional[str] = Field(
        default=None,
        description=(
            "If specified, the output will follow the regex pattern."),
    )
    guided_choice: Optional[list[str]] = Field(
        default=None,
        description=(
            "If specified, the output will be exactly one of the choices."),
    )
    guided_grammar: Optional[str] = Field(
        default=None,
        description=(
            "If specified, the output will follow the context free grammar."),
    )
    structural_tag: Optional[str] = Field(
        default=None,
        description=(
            "If specified, the output will follow the structural tag schema."),
    )
    guided_decoding_backend: Optional[str] = Field(
        default=None,
        description=(
            "If specified, will override the default guided decoding backend "
            "of the server for this specific request. If set, must be either "
            "'outlines' / 'lm-format-enforcer'"),
    )
    guided_whitespace_pattern: Optional[str] = Field(
        default=None,
        description=(
            "If specified, will override the default whitespace pattern "
            "for guided json decoding."),
    )
    priority: int = Field(
        default=0,
        description=(
            "The priority of the request (lower means earlier handling; "
            "default: 0). Any priority other than 0 will raise an error "
            "if the served model does not use priority scheduling."),
    )
    request_id: str = Field(
        default_factory=lambda: f"{random_uuid()}",
        description=(
            "The request_id related to this request. If the caller does "
            "not set it, a random_uuid will be generated. This id is used "
            "through out the inference process and return in response."),
    )
    logits_processors: Optional[LogitsProcessors] = Field(
        default=None,
        description=(
            "A list of either qualified names of logits processors, or "
            "constructor objects, to apply when sampling. A constructor is "
            "a JSON object with a required 'qualname' field specifying the "
            "qualified name of the processor class/factory, and optional "
            "'args' and 'kwargs' fields containing positional and keyword "
            "arguments. For example: {'qualname': "
            "'my_module.MyLogitsProcessor', 'args': [1, 2], 'kwargs': "
            "{'param': 'value'}}."))
    return_tokens_as_token_ids: Optional[bool] = Field(
        default=None,
        description=(
            "If specified with 'logprobs', tokens are represented "
            " as strings of the form 'token_id:{token_id}' so that tokens "
            "that are not JSON-encodable can be identified."))
    cache_salt: Optional[str] = Field(
        default=None,
        description=(
            "If specified, the prefix cache will be salted with the provided "
            "string to prevent an attacker to guess prompts in multi-user "
            "environments. The salt should be random, protected from "
            "access by 3rd parties, and long enough to be "
            "unpredictable (e.g., 43 characters base64-encoded, corresponding "
            "to 256 bit). Not supported by vLLM engine V0."))
    kv_transfer_params: Optional[dict[str, Any]] = Field(
        default=None,
        description="KVTransfer parameters used for disaggregated serving.")
    min_thinking_tokens: Optional[int] = None
    max_thinking_tokens: Optional[int] = None

    # --8<-- [end:chat-completion-extra-params]

    # Default sampling parameters for chat completion requests
    _DEFAULT_SAMPLING_PARAMS: dict = {
        "repetition_penalty": 1.0,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
    }

    def to_beam_search_params(
            self,
            default_max_tokens: int,
            default_sampling_params: Optional[dict] = None
    ) -> BeamSearchParams:
        # TODO(#9845): remove max_tokens when field is removed from OpenAI API
        max_tokens = self.max_completion_tokens or self.max_tokens

        if default_sampling_params is None:
            default_sampling_params = {}
        n = self.n if self.n is not None else 1

        # Use minimum of context window, user request & server limit.
        max_tokens = min(
            val for val in (default_max_tokens, max_tokens,
                            default_sampling_params.get("max_tokens", None))
            if val is not None)

        if (temperature := self.temperature) is None:
            temperature = default_sampling_params.get(
                "temperature", self._DEFAULT_SAMPLING_PARAMS["temperature"])

        return BeamSearchParams(
            beam_width=n,
            max_tokens=max_tokens,
            ignore_eos=self.ignore_eos,
            temperature=temperature,
            length_penalty=self.length_penalty,
            include_stop_str_in_output=self.include_stop_str_in_output,
        )

    def to_sampling_params(
        self,
        default_max_tokens: int,
        logits_processor_pattern: Optional[str],
        default_sampling_params: Optional[dict] = None,
    ) -> SamplingParams:
        # TODO(#9845): remove max_tokens when field is removed from OpenAI API
        max_tokens = self.max_completion_tokens or self.max_tokens

        if default_sampling_params is None:
            default_sampling_params = {}

        # Use minimum of context window, user request & server limit.
        max_tokens = min(
            val for val in (default_max_tokens, max_tokens,
                            default_sampling_params.get("max_tokens", None))
            if val is not None)

        # Default parameters
        if (repetition_penalty := self.repetition_penalty) is None:
            repetition_penalty = default_sampling_params.get(
                "repetition_penalty",
                self._DEFAULT_SAMPLING_PARAMS["repetition_penalty"],
            )
        if (temperature := self.temperature) is None:
            temperature = default_sampling_params.get(
                "temperature", self._DEFAULT_SAMPLING_PARAMS["temperature"])
        if (top_p := self.top_p) is None:
            top_p = default_sampling_params.get(
                "top_p", self._DEFAULT_SAMPLING_PARAMS["top_p"])
        if (top_k := self.top_k) is None:
            top_k = default_sampling_params.get(
                "top_k", self._DEFAULT_SAMPLING_PARAMS["top_k"])
        if (min_p := self.min_p) is None:
            min_p = default_sampling_params.get(
                "min_p", self._DEFAULT_SAMPLING_PARAMS["min_p"])

        prompt_logprobs = self.prompt_logprobs
        if prompt_logprobs is None and self.echo:
            prompt_logprobs = self.top_logprobs

        guided_json_object = None
        if self.response_format is not None:
            if self.response_format.type == "json_object":
                guided_json_object = True
            elif self.response_format.type == "json_schema":
                json_schema = self.response_format.json_schema
                assert json_schema is not None
                self.guided_json = json_schema.json_schema
            elif self.response_format.type == "structural_tag":
                structural_tag = self.response_format
                assert structural_tag is not None and isinstance(
                    structural_tag, StructuralTagResponseFormat)
                s_tag_obj = structural_tag.model_dump(by_alias=True)
                self.structural_tag = json.dumps(s_tag_obj)

        guided_decoding = GuidedDecodingParams.from_optional(
            json=self._get_guided_json_from_tool() or self.guided_json,
            regex=self.guided_regex,
            choice=self.guided_choice,
            grammar=self.guided_grammar,
            json_object=guided_json_object,
            backend=self.guided_decoding_backend,
            whitespace_pattern=self.guided_whitespace_pattern,
            structural_tag=self.structural_tag,
        )

        if self.min_thinking_tokens is not None and self.max_thinking_tokens is None:
            self.max_thinking_tokens = self.min_thinking_tokens + 500

        if self.max_thinking_tokens is not None and self.min_thinking_tokens is None:
            self.min_thinking_tokens = max(self.max_thinking_tokens - 500, 0)

        if self.min_thinking_tokens is not None and self.max_thinking_tokens is not None:
            assert self.min_thinking_tokens < self.max_thinking_tokens, "min_thinking_tokens must be less than max_thinking_tokens"
            if len(self.messages) > 0:
                first_message = self.messages[0]["content"]
                last_message = self.messages[-1]["content"]
                if "/no_think" in first_message or "/no_think" in last_message:
                    raise ValueError("min_thinking_tokens and max_thinking_tokens cannot be set when /no_think is used")

        return SamplingParams.from_optional(
            n=self.n,
            best_of=self.best_of,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            seed=self.seed,
            stop=self.stop,
            stop_token_ids=self.stop_token_ids,
            logprobs=self.top_logprobs if self.logprobs else None,
            prompt_logprobs=prompt_logprobs,
            ignore_eos=self.ignore_eos,
            max_tokens=max_tokens,
            min_tokens=self.min_tokens,
            skip_special_tokens=self.skip_special_tokens,
            spaces_between_special_tokens=self.spaces_between_special_tokens,
            logits_processors=get_logits_processors(self.logits_processors,
                                                    logits_processor_pattern,
                                                    self.max_thinking_tokens,
                                                    self.min_thinking_tokens),
            include_stop_str_in_output=self.include_stop_str_in_output,
            truncate_prompt_tokens=self.truncate_prompt_tokens,
            output_kind=RequestOutputKind.DELTA if self.stream \
                else RequestOutputKind.FINAL_ONLY,
            guided_decoding=guided_decoding,
            logit_bias=self.logit_bias,
            allowed_token_ids=self.allowed_token_ids,
            extra_args=({"kv_transfer_params": self.kv_transfer_params}
                        if self.kv_transfer_params else None))

    def _get_guided_json_from_tool(
            self) -> Optional[Union[str, dict, BaseModel]]:
        # user has chosen to not use any tool
        if self.tool_choice == "none" or self.tools is None:
            return None

        # user has chosen to use a named tool
        if type(self.tool_choice) is ChatCompletionNamedToolChoiceParam:
            tool_name = self.tool_choice.function.name
            tools = {tool.function.name: tool.function for tool in self.tools}
            if tool_name not in tools:
                raise ValueError(
                    f"Tool '{tool_name}' has not been passed in `tools`.")
            tool = tools[tool_name]
            return tool.parameters

        if self.tool_choice == "required":
            # Pydantic schema generation cannot be used since the JSON schema
            # has to be constructed for a specific instantiation of a tool list
            # so that parameters of a function are correctly generated
            # based on the chosen function name
            def get_tool_schema(tool: ChatCompletionToolsParam) -> dict:
                return {
                    "properties": {
                        "name": {
                            "type": "string",
                            "enum": [tool.function.name]
                        },
                        # parameters are always generated as '{}' in the final
                        # output if they are missing from the request
                        # (i.e. are None or '{}') so the schema is
                        # updated to produce an empty object in that case
                        "parameters": tool.function.parameters
                        if tool.function.parameters else {
                            "type": "object",
                            "properties": {}
                        }
                    },
                    "required": ["name", "parameters"]
                }

            json_schema = {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "anyOf": [get_tool_schema(tool) for tool in self.tools]
                }
            }
            return json_schema

        return None

    @model_validator(mode="before")
    @classmethod
    def validate_stream_options(cls, data):
        if data.get("stream_options") and not data.get("stream"):
            raise ValueError(
                "Stream options can only be defined when `stream=True`.")

        return data

    @model_validator(mode="before")
    @classmethod
    def check_logprobs(cls, data):
        if (prompt_logprobs := data.get("prompt_logprobs")) is not None:
            if data.get("stream") and prompt_logprobs > 0:
                raise ValueError(
                    "`prompt_logprobs` are not available when `stream=True`.")

            if prompt_logprobs < 0:
                raise ValueError("`prompt_logprobs` must be a positive value.")

        if (top_logprobs := data.get("top_logprobs")) is not None:
            if top_logprobs < 0:
                raise ValueError("`top_logprobs` must be a positive value.")

            if top_logprobs > 0 and not data.get("logprobs"):
                raise ValueError(
                    "when using `top_logprobs`, `logprobs` must be set to true."
                )

        return data

    @model_validator(mode="before")
    @classmethod
    def check_guided_decoding_count(cls, data):
        if isinstance(data, ValueError):
            raise data

        guide_count = sum([
            "guided_json" in data and data["guided_json"] is not None,
            "guided_regex" in data and data["guided_regex"] is not None,
            "guided_choice" in data and data["guided_choice"] is not None
        ])
        # you can only use one kind of guided decoding
        if guide_count > 1:
            raise ValueError(
                "You can only use one kind of guided decoding "
                "('guided_json', 'guided_regex' or 'guided_choice').")
        # you can only either use guided decoding or tools, not both
        if guide_count > 1 and data.get("tool_choice", "none") not in (
                "none",
                "auto",
                "required",
        ):
            raise ValueError(
                "You can only either use guided decoding or tools, not both.")
        return data

    @model_validator(mode="before")
    @classmethod
    def check_tool_usage(cls, data):

        # if "tool_choice" is not specified but tools are provided,
        # default to "auto" tool_choice
        if "tool_choice" not in data and data.get("tools"):
            data["tool_choice"] = "auto"

        # if "tool_choice" is "none" -- ignore tools if present
        if "tool_choice" in data and data["tool_choice"] == "none":
            # ensure that no tools are present
            data.pop("tools", None)
            return data

        # if "tool_choice" is specified -- validation
        if "tool_choice" in data:

            # ensure that if "tool choice" is specified, tools are present
            if "tools" not in data or data["tools"] is None:
                raise ValueError(
                    "When using `tool_choice`, `tools` must be set.")

            # make sure that tool choice is either a named tool
            # OR that it's set to "auto" or "required"
            if data["tool_choice"] not in [
                    "auto", "required"
            ] and not isinstance(data["tool_choice"], dict):
                raise NotImplementedError(
                    f'Invalid value for `tool_choice`: {data["tool_choice"]}! '\
                    'Only named tools, "none", "auto" or "required" '\
                    'are supported.'
                )

            # ensure that if "tool_choice" is specified as an object,
            # it matches a valid tool
            correct_usage_message = 'Correct usage: `{"type": "function",' \
                ' "function": {"name": "my_function"}}`'
            if isinstance(data["tool_choice"], dict):
                valid_tool = False
                function = data["tool_choice"].get("function")
                if not isinstance(function, dict):
                    raise ValueError(
                        f"Invalid value for `function`: `{function}` in "
                        f"`tool_choice`! {correct_usage_message}")
                if "name" not in function:
                    raise ValueError(f"Expected field `name` in `function` in "
                                     f"`tool_choice`! {correct_usage_message}")
                function_name = function["name"]
                if not isinstance(function_name,
                                  str) or len(function_name) == 0:
                    raise ValueError(
                        f"Invalid `name` in `function`: `{function_name}`"
                        f" in `tool_choice`! {correct_usage_message}")
                for tool in data["tools"]:
                    if tool["function"]["name"] == function_name:
                        valid_tool = True
                        break
                if not valid_tool:
                    raise ValueError(
                        "The tool specified in `tool_choice` does not match any"
                        " of the specified `tools`")
        return data

    @model_validator(mode="before")
    @classmethod
    def check_generation_prompt(cls, data):
        if data.get("continue_final_message") and data.get(
                "add_generation_prompt"):
            raise ValueError("Cannot set both `continue_final_message` and "
                             "`add_generation_prompt` to True.")
        return data

    @model_validator(mode="before")
    @classmethod
    def check_cache_salt_support(cls, data):
        if data.get("cache_salt") is not None:
            if not envs.VLLM_USE_V1:
                raise ValueError(
                    "Parameter 'cache_salt' is not supported with "
                    "this instance of vLLM, which uses engine V0.")
            if not isinstance(data["cache_salt"],
                              str) or not data["cache_salt"]:
                raise ValueError("Parameter 'cache_salt' must be a "
                                 "non-empty string if provided.")
        return data


class CompletionRequest(OpenAIBaseModel):
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/completions/create
    model: Optional[str] = None
    prompt: Optional[Union[list[int], list[list[int]], str, list[str]]] = None
    prompt_embeds: Optional[Union[bytes, list[bytes]]] = None
    best_of: Optional[int] = None
    echo: Optional[bool] = False
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[dict[str, float]] = None
    logprobs: Optional[int] = None
    max_tokens: Optional[int] = 16
    n: int = 1
    presence_penalty: Optional[float] = 0.0
    seed: Optional[int] = Field(None, ge=_LONG_INFO.min, le=_LONG_INFO.max)
    stop: Optional[Union[str, list[str]]] = []
    stream: Optional[bool] = False
    stream_options: Optional[StreamOptions] = None
    suffix: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    user: Optional[str] = None

    # --8<-- [start:completion-sampling-params]
    use_beam_search: bool = False
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    length_penalty: float = 1.0
    stop_token_ids: Optional[list[int]] = []
    include_stop_str_in_output: bool = False
    ignore_eos: bool = False
    min_tokens: int = 0
    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True
    truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]] = None
    allowed_token_ids: Optional[list[int]] = None
    prompt_logprobs: Optional[int] = None
    # --8<-- [end:completion-sampling-params]

    # --8<-- [start:completion-extra-params]
    add_special_tokens: bool = Field(
        default=True,
        description=(
            "If true (the default), special tokens (e.g. BOS) will be added to "
            "the prompt."),
    )
    response_format: Optional[AnyResponseFormat] = Field(
        default=None,
        description=(
            "Similar to chat completion, this parameter specifies the format "
            "of output. Only {'type': 'json_object'}, {'type': 'json_schema'}"
            ", {'type': 'structural_tag'}, or {'type': 'text' } is supported."
        ),
    )
    guided_json: Optional[Union[str, dict, BaseModel]] = Field(
        default=None,
        description="If specified, the output will follow the JSON schema.",
    )
    guided_regex: Optional[str] = Field(
        default=None,
        description=(
            "If specified, the output will follow the regex pattern."),
    )
    guided_choice: Optional[list[str]] = Field(
        default=None,
        description=(
            "If specified, the output will be exactly one of the choices."),
    )
    guided_grammar: Optional[str] = Field(
        default=None,
        description=(
            "If specified, the output will follow the context free grammar."),
    )
    guided_decoding_backend: Optional[str] = Field(
        default=None,
        description=(
            "If specified, will override the default guided decoding backend "
            "of the server for this specific request. If set, must be one of "
            "'outlines' / 'lm-format-enforcer'"),
    )
    guided_whitespace_pattern: Optional[str] = Field(
        default=None,
        description=(
            "If specified, will override the default whitespace pattern "
            "for guided json decoding."),
    )
    priority: int = Field(
        default=0,
        description=(
            "The priority of the request (lower means earlier handling; "
            "default: 0). Any priority other than 0 will raise an error "
            "if the served model does not use priority scheduling."),
    )
    logits_processors: Optional[LogitsProcessors] = Field(
        default=None,
        description=(
            "A list of either qualified names of logits processors, or "
            "constructor objects, to apply when sampling. A constructor is "
            "a JSON object with a required 'qualname' field specifying the "
            "qualified name of the processor class/factory, and optional "
            "'args' and 'kwargs' fields containing positional and keyword "
            "arguments. For example: {'qualname': "
            "'my_module.MyLogitsProcessor', 'args': [1, 2], 'kwargs': "
            "{'param': 'value'}}."))

    return_tokens_as_token_ids: Optional[bool] = Field(
        default=None,
        description=(
            "If specified with 'logprobs', tokens are represented "
            " as strings of the form 'token_id:{token_id}' so that tokens "
            "that are not JSON-encodable can be identified."))

    kv_transfer_params: Optional[dict[str, Any]] = Field(
        default=None,
        description="KVTransfer parameters used for disaggregated serving.")

    # --8<-- [end:completion-extra-params]

    # Default sampling parameters for completion requests
    _DEFAULT_SAMPLING_PARAMS: dict = {
        "repetition_penalty": 1.0,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
    }

    def to_beam_search_params(
            self,
            default_max_tokens: int,
            default_sampling_params: Optional[dict] = None
    ) -> BeamSearchParams:
        max_tokens = self.max_tokens

        if default_sampling_params is None:
            default_sampling_params = {}
        n = self.n if self.n is not None else 1

        # Use minimum of context window, user request & server limit.
        max_tokens = min(
            val for val in (default_max_tokens, max_tokens,
                            default_sampling_params.get("max_tokens", None))
            if val is not None)

        if (temperature := self.temperature) is None:
            temperature = default_sampling_params.get("temperature", 1.0)

        return BeamSearchParams(
            beam_width=n,
            max_tokens=max_tokens,
            ignore_eos=self.ignore_eos,
            temperature=temperature,
            length_penalty=self.length_penalty,
            include_stop_str_in_output=self.include_stop_str_in_output,
        )

    def to_sampling_params(
        self,
        default_max_tokens: int,
        logits_processor_pattern: Optional[str],
        default_sampling_params: Optional[dict] = None,
    ) -> SamplingParams:
        max_tokens = self.max_tokens

        if default_sampling_params is None:
            default_sampling_params = {}

        # Use minimum of context window, user request & server limit.
        max_tokens = min(
            val for val in (default_max_tokens, max_tokens,
                            default_sampling_params.get("max_tokens", None))
            if val is not None)

        # Default parameters
        if (repetition_penalty := self.repetition_penalty) is None:
            repetition_penalty = default_sampling_params.get(
                "repetition_penalty",
                self._DEFAULT_SAMPLING_PARAMS["repetition_penalty"],
            )
        if (temperature := self.temperature) is None:
            temperature = default_sampling_params.get(
                "temperature", self._DEFAULT_SAMPLING_PARAMS["temperature"])
        if (top_p := self.top_p) is None:
            top_p = default_sampling_params.get(
                "top_p", self._DEFAULT_SAMPLING_PARAMS["top_p"])
        if (top_k := self.top_k) is None:
            top_k = default_sampling_params.get(
                "top_k", self._DEFAULT_SAMPLING_PARAMS["top_k"])
        if (min_p := self.min_p) is None:
            min_p = default_sampling_params.get(
                "min_p", self._DEFAULT_SAMPLING_PARAMS["min_p"])

        prompt_logprobs = self.prompt_logprobs
        if prompt_logprobs is None and self.echo:
            prompt_logprobs = self.logprobs

        echo_without_generation = self.echo and self.max_tokens == 0

        guided_json_object = None
        if (self.response_format is not None
                and self.response_format.type == "json_object"):
            guided_json_object = True

        guided_decoding = GuidedDecodingParams.from_optional(
            json=self.guided_json,
            regex=self.guided_regex,
            choice=self.guided_choice,
            grammar=self.guided_grammar,
            json_object=guided_json_object,
            backend=self.guided_decoding_backend,
            whitespace_pattern=self.guided_whitespace_pattern,
        )

        return SamplingParams.from_optional(
            n=self.n,
            best_of=self.best_of,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            seed=self.seed,
            stop=self.stop,
            stop_token_ids=self.stop_token_ids,
            logprobs=self.logprobs,
            ignore_eos=self.ignore_eos,
            max_tokens=max_tokens if not echo_without_generation else 1,
            min_tokens=self.min_tokens,
            prompt_logprobs=prompt_logprobs,
            skip_special_tokens=self.skip_special_tokens,
            spaces_between_special_tokens=self.spaces_between_special_tokens,
            include_stop_str_in_output=self.include_stop_str_in_output,
            logits_processors=get_logits_processors(self.logits_processors,
                                                    logits_processor_pattern),
            truncate_prompt_tokens=self.truncate_prompt_tokens,
            output_kind=RequestOutputKind.DELTA if self.stream \
                else RequestOutputKind.FINAL_ONLY,
            guided_decoding=guided_decoding,
            logit_bias=self.logit_bias,
            allowed_token_ids=self.allowed_token_ids,
            extra_args=({"kv_transfer_params": self.kv_transfer_params}
                        if self.kv_transfer_params else None))

    @model_validator(mode="before")
    @classmethod
    def check_guided_decoding_count(cls, data):
        guide_count = sum([
            "guided_json" in data and data["guided_json"] is not None,
            "guided_regex" in data and data["guided_regex"] is not None,
            "guided_choice" in data and data["guided_choice"] is not None
        ])
        if guide_count > 1:
            raise ValueError(
                "You can only use one kind of guided decoding "
                "('guided_json', 'guided_regex' or 'guided_choice').")
        return data

    @model_validator(mode="before")
    @classmethod
    def check_logprobs(cls, data):
        if (prompt_logprobs := data.get("prompt_logprobs")) is not None:
            if data.get("stream") and prompt_logprobs > 0:
                raise ValueError(
                    "`prompt_logprobs` are not available when `stream=True`.")

            if prompt_logprobs < 0:
                raise ValueError("`prompt_logprobs` must be a positive value.")

        if (logprobs := data.get("logprobs")) is not None and logprobs < 0:
            raise ValueError("`logprobs` must be a positive value.")

        return data

    @model_validator(mode="before")
    @classmethod
    def validate_stream_options(cls, data):
        if data.get("stream_options") and not data.get("stream"):
            raise ValueError(
                "Stream options can only be defined when `stream=True`.")

        return data

    @model_validator(mode="before")
    @classmethod
    def validate_prompt_and_prompt_embeds(cls, data):
        if data.get("prompt") is None and data.get("prompt_embeds") is None:
            raise ValueError(
                "At least one of `prompt` or `prompt_embeds` must be set.")
        return data


class EmbeddingCompletionRequest(OpenAIBaseModel):
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/embeddings
    model: Optional[str] = None
    input: Union[list[int], list[list[int]], str, list[str]]
    encoding_format: Literal["float", "base64"] = "float"
    dimensions: Optional[int] = None
    user: Optional[str] = None
    truncate_prompt_tokens: Optional[Annotated[int, Field(ge=-1)]] = None

    # --8<-- [start:embedding-pooling-params]
    additional_data: Optional[Any] = None
    # --8<-- [end:embedding-pooling-params]

    # --8<-- [start:embedding-extra-params]
    add_special_tokens: bool = Field(
        default=True,
        description=(
            "If true (the default), special tokens (e.g. BOS) will be added to "
            "the prompt."),
    )
    priority: int = Field(
        default=0,
        description=(
            "The priority of the request (lower means earlier handling; "
            "default: 0). Any priority other than 0 will raise an error "
            "if the served model does not use priority scheduling."),
    )

    # --8<-- [end:embedding-extra-params]

    def to_pooling_params(self):
        return PoolingParams(dimensions=self.dimensions,
                             additional_data=self.additional_data)


class EmbeddingChatRequest(OpenAIBaseModel):
    model: Optional[str] = None
    messages: list[ChatCompletionMessageParam]

    encoding_format: Literal["float", "base64"] = "float"
    dimensions: Optional[int] = None
    user: Optional[str] = None
    truncate_prompt_tokens: Optional[Annotated[int, Field(ge=-1)]] = None

    # --8<-- [start:chat-embedding-pooling-params]
    additional_data: Optional[Any] = None
    # --8<-- [end:chat-embedding-pooling-params]

    # --8<-- [start:chat-embedding-extra-params]
    add_special_tokens: bool = Field(
        default=False,
        description=(
            "If true, special tokens (e.g. BOS) will be added to the prompt "
            "on top of what is added by the chat template. "
            "For most models, the chat template takes care of adding the "
            "special tokens so this should be set to false (as is the "
            "default)."),
    )
    chat_template: Optional[str] = Field(
        default=None,
        description=(
            "A Jinja template to use for this conversion. "
            "As of transformers v4.44, default chat template is no longer "
            "allowed, so you must provide a chat template if the tokenizer "
            "does not define one."),
    )
    chat_template_kwargs: Optional[dict[str, Any]] = Field(
        default=None,
        description=("Additional kwargs to pass to the template renderer. "
                     "Will be accessible by the chat template."),
    )
    mm_processor_kwargs: Optional[dict[str, Any]] = Field(
        default=None,
        description=("Additional kwargs to pass to the HF processor."),
    )
    priority: int = Field(
        default=0,
        description=(
            "The priority of the request (lower means earlier handling; "
            "default: 0). Any priority other than 0 will raise an error "
            "if the served model does not use priority scheduling."),
    )
    # --8<-- [end:chat-embedding-extra-params]

    @model_validator(mode="before")
    @classmethod
    def check_generation_prompt(cls, data):
        if data.get("continue_final_message") and data.get(
                "add_generation_prompt"):
            raise ValueError("Cannot set both `continue_final_message` and "
                             "`add_generation_prompt` to True.")
        return data

    def to_pooling_params(self):
        return PoolingParams(dimensions=self.dimensions,
                             additional_data=self.additional_data)


EmbeddingRequest = Union[EmbeddingCompletionRequest, EmbeddingChatRequest]

PoolingCompletionRequest = EmbeddingCompletionRequest
PoolingChatRequest = EmbeddingChatRequest
PoolingRequest = Union[PoolingCompletionRequest, PoolingChatRequest]


class ScoreRequest(OpenAIBaseModel):
    model: Optional[str] = None
    text_1: Union[list[str], str]
    text_2: Union[list[str], str]
    truncate_prompt_tokens: Optional[Annotated[int, Field(ge=-1)]] = None

    # --8<-- [start:score-pooling-params]
    additional_data: Optional[Any] = None
    # --8<-- [end:score-pooling-params]

    # --8<-- [start:score-extra-params]
    priority: int = Field(
        default=0,
        description=(
            "The priority of the request (lower means earlier handling; "
            "default: 0). Any priority other than 0 will raise an error "
            "if the served model does not use priority scheduling."),
    )

    # --8<-- [end:score-extra-params]

    def to_pooling_params(self):
        return PoolingParams(additional_data=self.additional_data)


class RerankRequest(OpenAIBaseModel):
    model: Optional[str] = None
    query: str
    documents: list[str]
    top_n: int = Field(default_factory=lambda: 0)
    truncate_prompt_tokens: Optional[Annotated[int, Field(ge=-1)]] = None

    # --8<-- [start:rerank-pooling-params]
    additional_data: Optional[Any] = None
    # --8<-- [end:rerank-pooling-params]

    # --8<-- [start:rerank-extra-params]
    priority: int = Field(
        default=0,
        description=(
            "The priority of the request (lower means earlier handling; "
            "default: 0). Any priority other than 0 will raise an error "
            "if the served model does not use priority scheduling."),
    )

    # --8<-- [end:rerank-extra-params]

    def to_pooling_params(self):
        return PoolingParams(additional_data=self.additional_data)


class RerankDocument(BaseModel):
    text: str


class RerankResult(BaseModel):
    index: int
    document: RerankDocument
    relevance_score: float


class RerankUsage(BaseModel):
    total_tokens: int


class RerankResponse(OpenAIBaseModel):
    id: str
    model: str
    usage: RerankUsage
    results: list[RerankResult]


class CompletionLogProbs(OpenAIBaseModel):
    text_offset: list[int] = Field(default_factory=list)
    token_logprobs: list[Optional[float]] = Field(default_factory=list)
    tokens: list[str] = Field(default_factory=list)
    top_logprobs: list[Optional[dict[str,
                                     float]]] = Field(default_factory=list)


class CompletionResponseChoice(OpenAIBaseModel):
    index: int
    text: str
    logprobs: Optional[CompletionLogProbs] = None
    finish_reason: Optional[str] = None
    stop_reason: Optional[Union[int, str]] = Field(
        default=None,
        description=(
            "The stop string or token id that caused the completion "
            "to stop, None if the completion finished for some other reason "
            "including encountering the EOS token"),
    )
    prompt_logprobs: Optional[list[Optional[dict[int, Logprob]]]] = None


class CompletionResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{random_uuid()}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[CompletionResponseChoice]
    usage: UsageInfo
    kv_transfer_params: Optional[dict[str, Any]] = Field(
        default=None, description="KVTransfer parameters.")


class CompletionResponseStreamChoice(OpenAIBaseModel):
    index: int
    text: str
    logprobs: Optional[CompletionLogProbs] = None
    finish_reason: Optional[str] = None
    stop_reason: Optional[Union[int, str]] = Field(
        default=None,
        description=(
            "The stop string or token id that caused the completion "
            "to stop, None if the completion finished for some other reason "
            "including encountering the EOS token"),
    )


class CompletionStreamResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{random_uuid()}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[CompletionResponseStreamChoice]
    usage: Optional[UsageInfo] = Field(default=None)


class EmbeddingResponseData(OpenAIBaseModel):
    index: int
    object: str = "embedding"
    embedding: Union[list[float], str]


class EmbeddingResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"embd-{random_uuid()}")
    object: str = "list"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    data: list[EmbeddingResponseData]
    usage: UsageInfo


class PoolingResponseData(OpenAIBaseModel):
    index: int
    object: str = "pooling"
    data: Union[list[list[float]], list[float], str]


class PoolingResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"pool-{random_uuid()}")
    object: str = "list"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    data: list[PoolingResponseData]
    usage: UsageInfo


class ScoreResponseData(OpenAIBaseModel):
    index: int
    object: str = "score"
    score: float


class ScoreResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"embd-{random_uuid()}")
    object: str = "list"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    data: list[ScoreResponseData]
    usage: UsageInfo


class ClassificationRequest(OpenAIBaseModel):
    model: Optional[str] = None
    input: Union[list[str], str]
    truncate_prompt_tokens: Optional[int] = None
    user: Optional[str] = None

    # --8<-- [start:classification-pooling-params]
    additional_data: Optional[Any] = None
    # --8<-- [end:classification-pooling-params]

    # --8<-- [start:classification-extra-params]
    priority: int = Field(
        default=0,
        description=(
            "The priority of the request (lower means earlier handling; "
            "default: 0). Any priority other than 0 will raise an error "
            "if the served model does not use priority scheduling."),
    )

    # --8<-- [end:classification-extra-params]

    def to_pooling_params(self):
        return PoolingParams(additional_data=self.additional_data)


class ClassificationData(OpenAIBaseModel):
    index: int
    label: Optional[str]
    probs: list[float]
    num_classes: int


class ClassificationResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"classify-{random_uuid()}")
    object: str = "list"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    data: list[ClassificationData]
    usage: UsageInfo


class FunctionCall(OpenAIBaseModel):
    name: str
    arguments: str


class ToolCall(OpenAIBaseModel):
    id: str = Field(default_factory=random_tool_call_id)
    type: Literal["function"] = "function"
    function: FunctionCall


class DeltaFunctionCall(BaseModel):
    name: Optional[str] = None
    arguments: Optional[str] = None


# a tool call delta where everything is optional
class DeltaToolCall(OpenAIBaseModel):
    id: Optional[str] = None
    type: Optional[Literal["function"]] = None
    index: int
    function: Optional[DeltaFunctionCall] = None


class ExtractedToolCallInformation(BaseModel):
    # indicate if tools were called
    tools_called: bool

    # extracted tool calls
    tool_calls: list[ToolCall]

    # content - per OpenAI spec, content AND tool calls can be returned rarely
    # But some models will do this intentionally
    content: Optional[str] = None


class ChatMessage(OpenAIBaseModel):
    role: str
    reasoning_content: Optional[str] = None
    content: Optional[str] = None
    tool_calls: list[ToolCall] = Field(default_factory=list)


class ChatCompletionLogProb(OpenAIBaseModel):
    token: str
    logprob: float = -9999.0
    bytes: Optional[list[int]] = None


class ChatCompletionLogProbsContent(ChatCompletionLogProb):
    # Workaround: redefine fields name cache so that it's not
    # shared with the super class.
    field_names: ClassVar[Optional[set[str]]] = None
    top_logprobs: list[ChatCompletionLogProb] = Field(default_factory=list)


class ChatCompletionLogProbs(OpenAIBaseModel):
    content: Optional[list[ChatCompletionLogProbsContent]] = None


class ChatCompletionResponseChoice(OpenAIBaseModel):
    index: int
    message: ChatMessage
    logprobs: Optional[ChatCompletionLogProbs] = None
    # per OpenAI spec this is the default
    finish_reason: Optional[str] = "stop"
    # not part of the OpenAI spec but included in vLLM for legacy reasons
    stop_reason: Optional[Union[int, str]] = None


class ChatCompletionResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{random_uuid()}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionResponseChoice]
    usage: UsageInfo
    prompt_logprobs: Optional[list[Optional[dict[int, Logprob]]]] = None
    kv_transfer_params: Optional[dict[str, Any]] = Field(
        default=None, description="KVTransfer parameters.")


class DeltaMessage(OpenAIBaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    tool_calls: list[DeltaToolCall] = Field(default_factory=list)


class ChatCompletionResponseStreamChoice(OpenAIBaseModel):
    index: int
    delta: DeltaMessage
    logprobs: Optional[ChatCompletionLogProbs] = None
    finish_reason: Optional[str] = None
    stop_reason: Optional[Union[int, str]] = None


class ChatCompletionStreamResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{random_uuid()}")
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionResponseStreamChoice]
    usage: Optional[UsageInfo] = Field(default=None)


class TranscriptionResponseStreamChoice(OpenAIBaseModel):
    delta: DeltaMessage
    finish_reason: Optional[str] = None
    stop_reason: Optional[Union[int, str]] = None


class TranscriptionStreamResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"trsc-{random_uuid()}")
    object: Literal["transcription.chunk"] = "transcription.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[TranscriptionResponseStreamChoice]
    usage: Optional[UsageInfo] = Field(default=None)


BatchRequestInputBody = Union[ChatCompletionRequest, EmbeddingRequest,
                              ScoreRequest, RerankRequest]


class BatchRequestInput(OpenAIBaseModel):
    """
    The per-line object of the batch input file.

    NOTE: Currently only the `/v1/chat/completions` endpoint is supported.
    """

    # A developer-provided per-request id that will be used to match outputs to
    # inputs. Must be unique for each request in a batch.
    custom_id: str

    # The HTTP method to be used for the request. Currently only POST is
    # supported.
    method: str

    # The OpenAI API relative URL to be used for the request. Currently
    # /v1/chat/completions is supported.
    url: str

    # The parameters of the request.
    body: BatchRequestInputBody

    @field_validator('body', mode='plain')
    @classmethod
    def check_type_for_url(cls, value: Any, info: ValidationInfo):
        # Use url to disambiguate models
        url: str = info.data["url"]
        if url == "/v1/chat/completions":
            return ChatCompletionRequest.model_validate(value)
        if url == "/v1/embeddings":
            return TypeAdapter(EmbeddingRequest).validate_python(value)
        if url.endswith("/score"):
            return ScoreRequest.model_validate(value)
        if url.endswith("/rerank"):
            return RerankRequest.model_validate(value)
        return TypeAdapter(BatchRequestInputBody).validate_python(value)


class BatchResponseData(OpenAIBaseModel):
    # HTTP status code of the response.
    status_code: int = 200

    # An unique identifier for the API request.
    request_id: str

    # The body of the response.
    body: Optional[Union[ChatCompletionResponse, EmbeddingResponse,
                         ScoreResponse, RerankResponse]] = None


class BatchRequestOutput(OpenAIBaseModel):
    """
    The per-line object of the batch output and error files
    """

    id: str

    # A developer-provided per-request id that will be used to match outputs to
    # inputs.
    custom_id: str

    response: Optional[BatchResponseData]

    # For requests that failed with a non-HTTP error, this will contain more
    # information on the cause of the failure.
    error: Optional[Any]


class TokenizeCompletionRequest(OpenAIBaseModel):
    model: Optional[str] = None
    prompt: str

    add_special_tokens: bool = Field(
        default=True,
        description=(
            "If true (the default), special tokens (e.g. BOS) will be added to "
            "the prompt."),
    )
    return_token_strs: Optional[bool] = Field(
        default=False,
        description=("If true, also return the token strings "
                     "corresponding to the token ids."),
    )


class TokenizeChatRequest(OpenAIBaseModel):
    model: Optional[str] = None
    messages: list[ChatCompletionMessageParam]

    add_generation_prompt: bool = Field(
        default=True,
        description=
        ("If true, the generation prompt will be added to the chat template. "
         "This is a parameter used by chat template in tokenizer config of the "
         "model."),
    )
    return_token_strs: Optional[bool] = Field(
        default=False,
        description=("If true, also return the token strings "
                     "corresponding to the token ids."),
    )
    continue_final_message: bool = Field(
        default=False,
        description=
        ("If this is set, the chat will be formatted so that the final "
         "message in the chat is open-ended, without any EOS tokens. The "
         "model will continue this message rather than starting a new one. "
         "This allows you to \"prefill\" part of the model's response for it. "
         "Cannot be used at the same time as `add_generation_prompt`."),
    )
    add_special_tokens: bool = Field(
        default=False,
        description=(
            "If true, special tokens (e.g. BOS) will be added to the prompt "
            "on top of what is added by the chat template. "
            "For most models, the chat template takes care of adding the "
            "special tokens so this should be set to false (as is the "
            "default)."),
    )
    chat_template: Optional[str] = Field(
        default=None,
        description=(
            "A Jinja template to use for this conversion. "
            "As of transformers v4.44, default chat template is no longer "
            "allowed, so you must provide a chat template if the tokenizer "
            "does not define one."),
    )
    chat_template_kwargs: Optional[dict[str, Any]] = Field(
        default=None,
        description=("Additional kwargs to pass to the template renderer. "
                     "Will be accessible by the chat template."),
    )
    mm_processor_kwargs: Optional[dict[str, Any]] = Field(
        default=None,
        description=("Additional kwargs to pass to the HF processor."),
    )
    tools: Optional[list[ChatCompletionToolsParam]] = Field(
        default=None,
        description=("A list of tools the model may call."),
    )

    @model_validator(mode="before")
    @classmethod
    def check_generation_prompt(cls, data):
        if data.get("continue_final_message") and data.get(
                "add_generation_prompt"):
            raise ValueError("Cannot set both `continue_final_message` and "
                             "`add_generation_prompt` to True.")
        return data


TokenizeRequest = Union[TokenizeCompletionRequest, TokenizeChatRequest]


class TokenizeResponse(OpenAIBaseModel):
    count: int
    max_model_len: int
    tokens: list[int]
    token_strs: Optional[list[str]] = None


class DetokenizeRequest(OpenAIBaseModel):
    model: Optional[str] = None
    tokens: list[int]


class DetokenizeResponse(OpenAIBaseModel):
    prompt: str


class LoadLoRAAdapterRequest(BaseModel):
    lora_name: str
    lora_path: str


class UnloadLoRAAdapterRequest(BaseModel):
    lora_name: str
    lora_int_id: Optional[int] = Field(default=None)


## Protocols for Audio
AudioResponseFormat: TypeAlias = Literal["json", "text", "srt", "verbose_json",
                                         "vtt"]


class TranscriptionRequest(OpenAIBaseModel):
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/audio/createTranscription

    file: UploadFile
    """
    The audio file object (not file name) to transcribe, in one of these
    formats: flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, or webm.
    """

    model: Optional[str] = None
    """ID of the model to use.
    """

    language: Optional[str] = None
    """The language of the input audio.

    Supplying the input language in
    [ISO-639-1](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes) format
    will improve accuracy and latency.
    """

    prompt: str = Field(default="")
    """An optional text to guide the model's style or continue a previous audio
    segment.

    The [prompt](https://platform.openai.com/docs/guides/speech-to-text#prompting)
    should match the audio language.
    """

    response_format: AudioResponseFormat = Field(default="json")
    """
    The format of the output, in one of these options: `json`, `text`, `srt`,
    `verbose_json`, or `vtt`.
    """

    ## TODO (varun) : Support if set to 0, certain thresholds are met !!

    timestamp_granularities: list[Literal["word", "segment"]] = Field(
        alias="timestamp_granularities[]", default=[])
    """The timestamp granularities to populate for this transcription.

    `response_format` must be set `verbose_json` to use timestamp granularities.
    Either or both of these options are supported: `word`, or `segment`. Note:
    There is no additional latency for segment timestamps, but generating word
    timestamps incurs additional latency.
    """

    # --8<-- [start:transcription-extra-params]
    stream: Optional[bool] = False
    """Custom field not present in the original OpenAI definition. When set,
    it will enable output to be streamed in a similar fashion as the Chat
    Completion endpoint.
    """
    # Flattened stream option to simplify form data.
    stream_include_usage: Optional[bool] = False
    stream_continuous_usage_stats: Optional[bool] = False
    # --8<-- [end:transcription-extra-params]

    # --8<-- [start:transcription-sampling-params]
    temperature: float = Field(default=0.0)
    """The sampling temperature, between 0 and 1.

    Higher values like 0.8 will make the output more random, while lower values
    like 0.2 will make it more focused / deterministic. If set to 0, the model
    will use [log probability](https://en.wikipedia.org/wiki/Log_probability)
    to automatically increase the temperature until certain thresholds are hit.
    """

    top_p: Optional[float] = None
    """Enables nucleus (top-p) sampling, where tokens are selected from the
    smallest possible set whose cumulative probability exceeds `p`.
    """

    top_k: Optional[int] = None
    """Limits sampling to the `k` most probable tokens at each step."""

    min_p: Optional[float] = None
    """Filters out tokens with a probability lower than `min_p`, ensuring a
    minimum likelihood threshold during sampling.
    """

    seed: Optional[int] = Field(None, ge=_LONG_INFO.min, le=_LONG_INFO.max)
    """The seed to use for sampling."""

    frequency_penalty: Optional[float] = 0.0
    """The frequency penalty to use for sampling."""

    repetition_penalty: Optional[float] = None
    """The repetition penalty to use for sampling."""

    presence_penalty: Optional[float] = 0.0
    """The presence penalty to use for sampling."""
    # --8<-- [end:transcription-sampling-params]

    # Default sampling parameters for transcription requests.
    _DEFAULT_SAMPLING_PARAMS: dict = {
        "repetition_penalty": 1.0,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
    }

    def to_sampling_params(
            self,
            default_max_tokens: int,
            default_sampling_params: Optional[dict] = None) -> SamplingParams:
        # TODO(#9845): remove max_tokens when field is removed from OpenAI API
        max_tokens = default_max_tokens

        if default_sampling_params is None:
            default_sampling_params = {}

        # Default parameters
        if (temperature := self.temperature) is None:
            temperature = default_sampling_params.get(
                "temperature", self._DEFAULT_SAMPLING_PARAMS["temperature"])
        if (top_p := self.top_p) is None:
            top_p = default_sampling_params.get(
                "top_p", self._DEFAULT_SAMPLING_PARAMS["top_p"])
        if (top_k := self.top_k) is None:
            top_k = default_sampling_params.get(
                "top_k", self._DEFAULT_SAMPLING_PARAMS["top_k"])
        if (min_p := self.min_p) is None:
            min_p = default_sampling_params.get(
                "min_p", self._DEFAULT_SAMPLING_PARAMS["min_p"])

        if (repetition_penalty := self.repetition_penalty) is None:
            repetition_penalty = default_sampling_params.get(
                "repetition_penalty",
                self._DEFAULT_SAMPLING_PARAMS["repetition_penalty"])

        return SamplingParams.from_optional(temperature=temperature,
                                            max_tokens=max_tokens,
                                            seed=self.seed,
                                            top_p=top_p,
                                            top_k=top_k,
                                            min_p=min_p,
                                            frequency_penalty=self.frequency_penalty,
                                            repetition_penalty=repetition_penalty,
                                            presence_penalty=self.presence_penalty,
                                            output_kind=RequestOutputKind.DELTA
                                            if self.stream \
                                            else RequestOutputKind.FINAL_ONLY)

    @model_validator(mode="before")
    @classmethod
    def validate_transcription_request(cls, data):
        if isinstance(data.get("file"), str):
            raise HTTPException(
                status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
                detail="Expected 'file' to be a file-like object, not 'str'.",
            )

        stream_opts = ["stream_include_usage", "stream_continuous_usage_stats"]
        stream = data.get("stream", False)
        if any(bool(data.get(so, False)) for so in stream_opts) and not stream:
            raise ValueError(
                "Stream options can only be defined when `stream=True`.")

        return data


# Transcription response objects
class TranscriptionResponse(OpenAIBaseModel):
    text: str
    """The transcribed text."""


class TranscriptionWord(OpenAIBaseModel):
    end: float
    """End time of the word in seconds."""

    start: float
    """Start time of the word in seconds."""

    word: str
    """The text content of the word."""


class TranscriptionSegment(OpenAIBaseModel):
    id: int
    """Unique identifier of the segment."""

    avg_logprob: float
    """Average logprob of the segment.

    If the value is lower than -1, consider the logprobs failed.
    """

    compression_ratio: float
    """Compression ratio of the segment.

    If the value is greater than 2.4, consider the compression failed.
    """

    end: float
    """End time of the segment in seconds."""

    no_speech_prob: float
    """Probability of no speech in the segment.

    If the value is higher than 1.0 and the `avg_logprob` is below -1, consider
    this segment silent.
    """

    seek: int
    """Seek offset of the segment."""

    start: float
    """Start time of the segment in seconds."""

    temperature: float
    """Temperature parameter used for generating the segment."""

    text: str
    """Text content of the segment."""

    tokens: list[int]
    """Array of token IDs for the text content."""


class TranscriptionResponseVerbose(OpenAIBaseModel):
    duration: str
    """The duration of the input audio."""

    language: str
    """The language of the input audio."""

    text: str
    """The transcribed text."""

    segments: Optional[list[TranscriptionSegment]] = None
    """Segments of the transcribed text and their corresponding details."""

    words: Optional[list[TranscriptionWord]] = None
    """Extracted words and their corresponding timestamps."""
