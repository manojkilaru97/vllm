# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.models.nano_nemotron_vl import (
    NemotronH_Nano_VL_V2,
    _hf_radio_names_to_legacy,
)


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
    object.__setattr__(model, "sound_encoder", None)

    language_weight = object()
    model.load_weights(
        [
            ("language_model.layers.0.weight", language_weight),
            ("mlp1.0.weight", object()),
            ("vision_model.radio_model.encoder.weight", object()),
            ("sound_encoder.encoder.weight", object()),
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
    object.__setattr__(model, "sound_encoder", None)

    with pytest.raises(AssertionError):
        model.load_weights([("sound_encoder.encoder.weight", object())])
