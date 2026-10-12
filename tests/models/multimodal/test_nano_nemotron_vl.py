# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import patch

import numpy as np
import pytest
import torch

from vllm import envs
from vllm.model_executor.models.nano_nemotron_vl import (
    NanoNemotronVLMultiModalProcessor,
    NemotronH_Nano_VL_V2,
    _hf_radio_names_to_legacy,
)
from vllm.multimodal.parse import (
    MultiModalDataItems,
    MultiModalDataParser,
    VideoProcessorItems,
)


@pytest.mark.parametrize("input_key", ["image_embeds", "video_embeds"])
def test_precomputed_multimodal_embeddings(input_key: str):
    model = object.__new__(NemotronH_Nano_VL_V2)
    embeds = torch.randn(2, 4, 8)

    outputs = model.embed_multimodal(**{input_key: embeds})

    assert len(outputs) == len(embeds)
    assert all(torch.equal(output, embed) for output, embed in zip(outputs, embeds))


class _TextOnlyMultiModalConfig:
    def get_limit_per_prompt(self, modality: str) -> int:
        return 0


class _ImageOnlyMultiModalConfig:
    def get_limit_per_prompt(self, modality: str) -> int:
        return 1 if modality == "image" else 0


class _ModelConfig:
    multimodal_config = _TextOnlyMultiModalConfig()


class _ImageOnlyModelConfig:
    multimodal_config = _ImageOnlyMultiModalConfig()


class _LanguageModel:
    def __init__(self) -> None:
        self.loaded_weights: list[tuple[str, object]] = []

    def load_weights(self, weights):
        self.loaded_weights = list(weights)


class _MissingMultiModalModule:
    def named_parameters(self):
        raise AssertionError("multimodal weights should not be inspected")

    def load_weights(self, weights):
        raise AssertionError("multimodal weights should not be loaded")


class _AdapterModule:
    def named_parameters(self):
        return []


class _VisionModel:
    def __init__(self) -> None:
        self.loaded_weights: list[tuple[str, object]] = []

    def load_weights(self, weights):
        self.loaded_weights = list(weights)


class _FakeTensor:
    """Sentinel stand-in for torch.Tensor in load_weights tests. Supports the
    .detach().clone() chain used by load_weights for buffered mm weights;
    both methods return self so identity (and the existing equality
    assertions) are preserved through cloning."""

    def detach(self):
        return self

    def clone(self):
        return self


def test_nano_nemotron_vl_skips_multimodal_weights_in_text_only_mode():
    model = object.__new__(NemotronH_Nano_VL_V2)
    language_model = _LanguageModel()
    object.__setattr__(model, "model_config", _ModelConfig())
    object.__setattr__(model, "language_model", language_model)
    object.__setattr__(model, "mlp1", _AdapterModule())
    object.__setattr__(model, "vision_model", _MissingMultiModalModule())
    object.__setattr__(model, "vision_final_layernorm", _MissingMultiModalModule())
    object.__setattr__(model, "sound_encoder", None)

    language_weight = object()
    model.load_weights(
        [
            ("language_model.layers.0.weight", language_weight),
            ("mlp1.0.weight", object()),
            ("vision_model.radio_model.encoder.weight", object()),
            ("sound_encoder.encoder.weight", object()),
            ("vision_projector.vision_final_layernorm.weight", object()),
            ("vision_projector.vision_final_layernorm.bias", object()),
        ]
    )

    assert language_model.loaded_weights == [("layers.0.weight", language_weight)]


def test_nano_nemotron_vl_loads_vision_weights_without_sound_encoder():
    model = object.__new__(NemotronH_Nano_VL_V2)
    language_model = _LanguageModel()
    vision_model = _VisionModel()
    object.__setattr__(model, "model_config", _ImageOnlyModelConfig())
    object.__setattr__(model, "language_model", language_model)
    object.__setattr__(model, "mlp1", _AdapterModule())
    object.__setattr__(model, "vision_model", vision_model)
    object.__setattr__(model, "vision_final_layernorm", None)
    object.__setattr__(model, "sound_encoder", None)

    language_weight = object()
    vision_weight = _FakeTensor()
    model.load_weights(
        [
            ("language_model.layers.0.weight", language_weight),
            ("vision_model.radio_model.encoder.weight", vision_weight),
        ]
    )

    assert language_model.loaded_weights == [("layers.0.weight", language_weight)]
    assert vision_model.loaded_weights == [
        ("radio_model.encoder.weight", vision_weight)
    ]


def test_hf_v5_radio_names_map_to_fused_legacy_weights():
    """Transformers v5 re-saved checkpoints must not silently drop vision."""
    q, k, v = (torch.full((2, 4), float(i)) for i in range(3))
    converted = dict(
        _hf_radio_names_to_legacy(
            [
                ("language_model.lm_head.weight", torch.zeros(1)),
                ("vision_model.encoder.layer.3.attention.attention.key.weight", k),
                ("vision_model.encoder.layer.3.attention.attention.query.weight", q),
                ("vision_model.encoder.layer.3.attention.attention.value.weight", v),
                ("vision_model.encoder.layer.3.attention.output.dense.bias", q),
                ("vision_model.encoder.layer.3.layer_scale1.lambda1", torch.ones(4)),
                ("vision_model.encoder.layer.3.mlp.fc1.weight", q),
                ("vision_model.embeddings.patch_projection.weight", q),
                ("vision_model.embeddings.cls_register_token", q),
                ("vision_projector.mlp1.linear2.weight", q),
                ("vision_projector.vision_final_layernorm.bias", q),
            ]
        )
    )

    blocks = "vision_model.radio_model.model.blocks.3."
    assert torch.equal(converted[blocks + "attn.qkv.weight"], torch.cat([q, k, v]))
    assert set(converted) == {
        "language_model.lm_head.weight",
        blocks + "attn.qkv.weight",
        blocks + "attn.proj.bias",
        blocks + "mlp.fc1.weight",
        "vision_model.radio_model.model.patch_generator.embedder.weight",
        "vision_model.radio_model.model.patch_generator.cls_token.token",
        "mlp1.3.weight",
        "vision_projector.vision_final_layernorm.bias",
    }


def test_hf_v5_radio_rejects_unsupported_layer_scale_and_partial_qkv():
    with pytest.raises(ValueError, match="layer scale"):
        list(
            _hf_radio_names_to_legacy(
                [("vision_model.encoder.layer.0.layer_scale2.lambda1", torch.zeros(4))]
            )
        )
    query_bias = "vision_model.encoder.layer.0.attention.attention.query.bias"
    with pytest.raises(ValueError, match="query/key/value"):
        list(_hf_radio_names_to_legacy([(query_bias, torch.zeros(4))]))


class _RadioLikeVisionModel(_VisionModel):
    def named_parameters(self):
        return [("model.encoder.layers.0.attn.qkv.weight", None)]

    def load_weights(self, weights):
        super().load_weights(weights)
        return set()


def test_nano_nemotron_vl_fails_when_vision_parameters_are_not_loaded():
    model = object.__new__(NemotronH_Nano_VL_V2)
    object.__setattr__(model, "model_config", _ImageOnlyModelConfig())
    object.__setattr__(model, "language_model", _LanguageModel())
    object.__setattr__(model, "mlp1", _AdapterModule())
    object.__setattr__(model, "vision_model", _RadioLikeVisionModel())
    object.__setattr__(model, "sound_encoder", None)

    with pytest.raises(ValueError, match="vision tower parameters were not loaded"):
        model.load_weights([("vision_model.unknown.weight", _FakeTensor())])


def test_nano_nemotron_vl_requires_sound_encoder_for_sound_weights():
    model = object.__new__(NemotronH_Nano_VL_V2)
    language_model = _LanguageModel()
    vision_model = _VisionModel()
    object.__setattr__(model, "model_config", _ImageOnlyModelConfig())
    object.__setattr__(model, "language_model", language_model)
    object.__setattr__(model, "mlp1", _AdapterModule())
    object.__setattr__(model, "vision_model", vision_model)
    object.__setattr__(model, "vision_final_layernorm", None)
    object.__setattr__(model, "sound_encoder", None)

    with pytest.raises(AssertionError):
        model.load_weights([("sound_encoder.encoder.weight", object())])


def _vision_norm_model(use_layernorm=True):
    model = object.__new__(NemotronH_Nano_VL_V2)
    torch.nn.Module.__init__(model)
    model.model_config = _ImageOnlyModelConfig()
    model.language_model = _LanguageModel()
    model.vision_model = _VisionModel()
    model.sound_encoder = None
    model.mlp1 = torch.nn.Identity()
    model.vision_final_layernorm = (
        torch.nn.LayerNorm(4, eps=1e-6, dtype=torch.bfloat16) if use_layernorm else None
    )
    model.patch_size = 2
    model.downsample_ratio = 0.5
    model.ps_version = "v2"
    model.video_temporal_patch_size = 2
    return model


@pytest.mark.parametrize("mode", ["image", "dynamic", "video"])
@pytest.mark.parametrize("use_layernorm", [True, False])
def test_vision_layernorm_is_loaded_and_applied_before_pixel_shuffle(
    mode, use_layernorm
):
    model = _vision_norm_model(use_layernorm)
    weight = torch.tensor([2.0, 3.0, 4.0, 5.0], dtype=torch.bfloat16)
    bias = torch.tensor([0.5, -0.5, 1.0, -1.0], dtype=torch.bfloat16)
    model.load_weights(
        [
            ("vision_projector.vision_final_layernorm.weight", weight),
            ("vision_projector.vision_final_layernorm.bias", bias),
        ]
        if use_layernorm
        else []
    )
    features = torch.arange(16, dtype=torch.bfloat16).reshape(1, 4, 4)
    expected = (
        torch.nn.functional.layer_norm(features, (4,), weight, bias, eps=1e-6)
        if use_layernorm
        else features
    )
    expected = model.pixel_shuffle(expected.reshape(1, 2, 2, 4)).reshape(1, 1, 16)
    model.vision_model = lambda *args, **kwargs: (None, features)
    pixels = torch.zeros(2 if mode == "video" else 1, 3, 4, 4)
    if mode == "dynamic":
        output = model.extract_feature_dynamic(pixels, imgs_sizes=[(4, 4)])
    else:
        output = model.extract_feature(
            pixels, num_frames=2 if mode == "video" else None
        )
    torch.testing.assert_close(output, expected)


def _make_mm_items_with_video_bytes(
    video_bytes: bytes,
) -> MultiModalDataItems:
    """Build a minimal MultiModalDataItems with one video entry."""
    items = MultiModalDataItems()
    items["video"] = VideoProcessorItems(
        data=[None],
        metadata=[{"original_video_bytes": video_bytes}],
    )
    return items


def test_extract_audio_from_videos_passes_max_duration():
    """_extract_audio_from_videos must forward VLLM_MAX_AUDIO_DECODE_DURATION_S
    to load_audio_pyav so decompression-bomb audio is rejected."""
    dummy_audio = (np.zeros(16000, dtype=np.float32), 16000.0)
    mm_items = _make_mm_items_with_video_bytes(b"\x00" * 64)

    processor = object.__new__(NanoNemotronVLMultiModalProcessor)
    processor.data_parser = MultiModalDataParser(
        target_sr=dummy_audio[1], target_channels=1
    )

    target = "vllm.model_executor.models.nano_nemotron_vl.load_audio_pyav"
    pyav = "vllm.model_executor.models.nano_nemotron_vl._pyav_available"
    with (
        patch(pyav, return_value=True),
        patch(target, return_value=dummy_audio) as mock_load,
    ):
        processor._extract_audio_from_videos(mm_items)

    mock_load.assert_called_once()
    _, kwargs = mock_load.call_args
    assert kwargs["max_duration_s"] == envs.VLLM_MAX_AUDIO_DECODE_DURATION_S


def test_extract_audio_from_videos_rejects_oversized_audio():
    """When load_audio_pyav raises due to duration limit the video is
    marked as having no audio instead of crashing the server."""
    mm_items = _make_mm_items_with_video_bytes(b"\x00" * 64)

    processor = object.__new__(NanoNemotronVLMultiModalProcessor)

    target = "vllm.model_executor.models.nano_nemotron_vl.load_audio_pyav"
    pyav = "vllm.model_executor.models.nano_nemotron_vl._pyav_available"
    with (
        patch(pyav, return_value=True),
        patch(
            target,
            side_effect=ValueError("Audio exceeds maximum allowed duration"),
        ),
    ):
        _, audio_items, has_audio = processor._extract_audio_from_videos(mm_items)

    assert audio_items == []
    assert has_audio == [False]
