# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
from torch._inductor.exc import CppCompileError, InductorError

import vllm.utils.torch_utils as torch_utils
from vllm.utils.torch_utils import compile_with_eager_fallback

ARMV9_ERROR = "cc1plus: error: unknown value 'armv9-a+sve2' for '-march'"


class _FakeCompiled:
    def __init__(self, fn, exc: BaseException | None):
        self.fn = fn
        self.exc = exc
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        if self.exc is not None:
            raise self.exc
        return ("compiled", self.fn(*args, **kwargs))


def _patch_compile(monkeypatch, exc: BaseException | None):
    made: list[_FakeCompiled] = []
    seen_kwargs: list[dict] = []

    def fake_compile(fn, **kwargs):
        seen_kwargs.append(kwargs)
        made.append(_FakeCompiled(fn, exc))
        return made[-1]

    monkeypatch.setattr(torch_utils.torch, "compile", fake_compile)
    return made, seen_kwargs


def _inductor_error() -> InductorError:
    return InductorError(CppCompileError(["g++"], ARMV9_ERROR), None)


def test_uses_compiled_path_when_compilation_works(monkeypatch):
    made, seen_kwargs = _patch_compile(monkeypatch, None)

    @compile_with_eager_fallback(dynamic=True)
    def add_one(x):
        return x + 1

    assert seen_kwargs == [{"dynamic": True}]
    assert add_one(1) == ("compiled", 2)
    assert add_one(2) == ("compiled", 3)
    assert made[0].calls == 2


@pytest.mark.parametrize(
    "exc",
    [
        pytest.param(_inductor_error(), id="inductor-error"),
        pytest.param(CppCompileError(["g++"], ARMV9_ERROR), id="cpp-compile-error"),
    ],
)
def test_falls_back_to_eager_once_on_compiler_failure(monkeypatch, exc):
    made, _ = _patch_compile(monkeypatch, exc)
    warnings: list[tuple] = []
    monkeypatch.setattr(
        torch_utils.logger, "warning", lambda *args, **kw: warnings.append(args)
    )

    @compile_with_eager_fallback(dynamic=True)
    def add_one(x):
        return x + 1

    assert add_one(1) == 2
    assert add_one(5) == 6
    assert made[0].calls == 1
    assert len(warnings) == 1
    assert "add_one" in warnings[0][1]


def test_does_not_swallow_errors_from_the_function(monkeypatch):
    made, _ = _patch_compile(monkeypatch, ValueError("bad input"))

    @compile_with_eager_fallback
    def identity(x):
        return x

    for _ in range(2):
        with pytest.raises(ValueError, match="bad input"):
            identity(1)
    assert made[0].calls == 2


def test_method_decorator_falls_back(monkeypatch):
    _patch_compile(monkeypatch, _inductor_error())

    class Scaler:
        def __init__(self, k):
            self.k = k

        @compile_with_eager_fallback(dynamic=True)
        def scale(self, x):
            return self.k * x

    assert Scaler(3).scale(2) == 6
    assert Scaler(4).scale(2) == 8


def test_real_compile_matches_eager_on_cpu():
    def fn(x):
        return (x / 255.0 - 0.5).contiguous()

    wrapped = compile_with_eager_fallback(fn, dynamic=True)
    x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    torch.testing.assert_close(wrapped(x), fn(x))


def test_nemotron_vl_cpu_preprocessing_uses_fallback_wrapper():
    from vllm.model_executor.models.parakeet import ParakeetExtractor
    from vllm.transformers_utils.processors import nano_nemotron_vl

    wrapped = [
        nano_nemotron_vl._bicubic_resize_and_normalize,
        ParakeetExtractor._apply_mel_filters,
        ParakeetExtractor._apply_preemphasis,
        ParakeetExtractor._normalize_mel_features,
    ]
    for fn in wrapped:
        assert hasattr(fn, "_vllm_compiled"), fn.__qualname__
