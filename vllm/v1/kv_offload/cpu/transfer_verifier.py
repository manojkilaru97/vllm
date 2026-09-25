# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Debug-only byte verification of CPU KV offload transfers.

Enabled with ``VLLM_KV_OFFLOAD_VERIFY=1``. Stores record a digest per CPU
destination range after checking it equals the GPU source; loads check the CPU
source still has that digest and that the GPU destination equals it.
"""

import hashlib
import os
import threading

import numpy as np
import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

ENABLED = os.environ.get("VLLM_KV_OFFLOAD_VERIFY", "0") == "1"
_LOG_EVERY = int(os.environ.get("VLLM_KV_OFFLOAD_VERIFY_LOG_EVERY", "200"))

_digests: dict[tuple[int, int], bytes] = {}
_lock = threading.Lock()
_last_bad = 0
_next_log = 0
_counts = {
    "store_ops": 0,
    "store_copy_mismatch": 0,
    "load_ops": 0,
    "load_unknown": 0,
    "load_cpu_changed": 0,
    "load_copy_mismatch": 0,
}


def _view(base: torch.Tensor, ptr: int, size: int) -> torch.Tensor:
    # Rows may be strided (shared-memory CPU layout), so index by row/column.
    row, col = divmod(ptr - base.data_ptr(), base.stride(0))
    assert 0 <= row < base.shape[0] and col + size <= base.shape[1]
    return base[row, col : col + size]


def _digest(cpu_view: torch.Tensor) -> bytes:
    return hashlib.blake2b(cpu_view.numpy().tobytes(), digest_size=16).digest()


def verify_transfer(
    gpu_to_cpu: bool,
    src_tensors: list[torch.Tensor],
    dst_tensors: list[torch.Tensor],
    tensor_indices: np.ndarray,
    src_ptrs: np.ndarray,
    dst_ptrs: np.ndarray,
    sizes: np.ndarray,
    stream: torch.cuda.Stream,
) -> None:
    """Synchronously check a finished transfer's bytes (debug only)."""
    global _last_bad, _next_log
    stream.synchronize()
    with _lock:
        for t_idx, src_ptr, dst_ptr, size in zip(
            tensor_indices.tolist(), src_ptrs.tolist(), dst_ptrs.tolist(),
            sizes.tolist()
        ):
            src = _view(src_tensors[t_idx], src_ptr, size)
            dst = _view(dst_tensors[t_idx], dst_ptr, size)
            if gpu_to_cpu:
                _counts["store_ops"] += 1
                if not torch.equal(src.cpu(), dst):
                    _counts["store_copy_mismatch"] += 1
                _digests[(dst_ptr, size)] = _digest(dst)
            else:
                _counts["load_ops"] += 1
                expected = _digests.get((src_ptr, size))
                if expected is None:
                    _counts["load_unknown"] += 1
                elif _digest(src) != expected:
                    _counts["load_cpu_changed"] += 1
                if not torch.equal(dst.cpu(), src):
                    _counts["load_copy_mismatch"] += 1
        total = _counts["store_ops"] + _counts["load_ops"]
        bad = (
            _counts["store_copy_mismatch"]
            + _counts["load_cpu_changed"]
            + _counts["load_copy_mismatch"]
        )
        if bad > _last_bad or total >= _next_log:
            logger.warning("KV offload verify: %s", dict(_counts))
            _last_bad = bad
            _next_log = total + _LOG_EVERY
