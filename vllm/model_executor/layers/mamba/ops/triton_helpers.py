# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


@functools.cache
def _has_cvt_rs(device_index: int) -> bool:
    try:
        return (
            current_platform.is_cuda()
            and torch.cuda.get_device_capability(device_index)[0] == 10
        )
    except Exception:
        return False


def has_cvt_rs(device: torch.device) -> bool:
    """Whether `device` has the `cvt.rs` stochastic rounding PTX instruction
    (data center Blackwell). Other GPUs use the software emulation."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    return _has_cvt_rs(index)


@triton.jit
def _convert_rs_fp16_sw(x, rand):
    """Stochastically round fp32 `x` to fp16 with 23 bits of `rand`.

    Rounds up to the next fp16 away from zero with probability equal to the
    truncated remainder in fp16 ulps, quantized to 2^-23 at grid midpoints.
    The result is unbiased up to that quantization (at most 2^-24 ulp) for
    finite inputs in the fp16 range, including subnormals; inputs below
    2^-48 always round to zero. Inputs beyond the fp16 range saturate to the
    largest finite fp16.
    """
    lo = x.to(tl.float16, fp_downcast_rounding="rtz")
    hi = (lo.to(tl.int16, bitcast=True) + 1).to(tl.float16, bitcast=True)
    lo_f = lo.to(tl.float32)
    u = ((rand.to(tl.uint32, bitcast=True) >> 9).to(tl.float32) + 0.5) * (
        1.0 / 8388608.0
    )
    # |hi - lo| is a power of two, so the product is exact: u < |x - lo| / ulp.
    ulp = tl.abs(hi.to(tl.float32) - lo_f)
    return tl.where(u * ulp < tl.abs(x - lo_f), hi, lo)


@triton.jit
def convert_rs_fp16x2(x: tl.tensor, rand: tl.tensor, HW_RS: tl.constexpr):
    """Stochastically round fp32 `x` to fp16 using per-element `rand` bits,
    with `cvt.rs` when `HW_RS` (see `has_cvt_rs`), else in software."""
    if HW_RS:
        return tl.inline_asm_elementwise(
            asm="""{
cvt.rs.f16x2.f32 $0, $2, $1, $3;
}""",
            constraints="=r,r,r,r,r",
            args=(x, rand),
            dtype=tl.float16,
            is_pure=True,
            pack=2,
        )
    else:
        return _convert_rs_fp16_sw(x, rand)


@triton.jit
def fast_exp(x):
    """Faster alternative to tl.exp() using the hardware exp2 instruction.

    tl.math.exp2 maps directly to a single ex2.approx.f32 PTX instruction,
    while tl.exp goes through libdevice __nv_expf which adds function call
    overhead and extra range checking.
    """
    # exp(x) = exp2(x * log2(e)), where log2(e) = 1/ln(2) = 1.4426950408889634
    LOG2E = tl.constexpr(1.4426950408889634)
    return tl.math.exp2(LOG2E * x)
