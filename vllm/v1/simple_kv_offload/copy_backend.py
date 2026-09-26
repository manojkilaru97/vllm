# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DMA copy backend for GPU<->CPU block transfers."""

from __future__ import annotations

import queue
import threading

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.kv_offload.cpu import transfer_verifier
from vllm.v1.simple_kv_offload.cuda_mem_ops import (
    CU_MEMCPY_SRC_ACCESS_ORDER_ANY,
    CU_MEMCPY_SRC_ACCESS_ORDER_STREAM,
    BatchMemcpyParams,
    build_params,
    copy_blocks,
)

logger = init_logger(__name__)


class DmaCopyBackend:
    """cuMemcpyBatchAsync copy backend (background thread)."""

    def __init__(self) -> None:
        self._store_params: BatchMemcpyParams | None = None
        self._load_params: BatchMemcpyParams | None = None
        self._load_stream: torch.cuda.Stream | None = None
        self._store_stream: torch.cuda.Stream | None = None
        self._queue: queue.SimpleQueue | None = None
        self._thread: threading.Thread | None = None
        self._shutdown: bool = False

    def init(
        self,
        gpu_caches: dict[str, torch.Tensor],
        cpu_caches: dict[str, torch.Tensor],
        device: torch.device,
        load_stream: torch.cuda.Stream,
        store_stream: torch.cuda.Stream,
    ) -> None:
        self._load_stream = load_stream
        self._store_stream = store_stream

        # Stores read the live KV cache -> STREAM (paired with the compute-done
        # wait in get_finished); loads read stable pinned host memory -> ANY.
        self._store_params = build_params(
            gpu_caches,
            cpu_caches,
            store_stream,
            src_access_order=CU_MEMCPY_SRC_ACCESS_ORDER_STREAM,
        )
        self._load_params = build_params(
            cpu_caches,
            gpu_caches,
            load_stream,
            src_access_order=CU_MEMCPY_SRC_ACCESS_ORDER_ANY,
        )

        # Debug-only byte checks of every copy (VLLM_KV_OFFLOAD_VERIFY=1).
        verify_tensors = (
            (list(gpu_caches.values()), list(cpu_caches.values()))
            if transfer_verifier.ENABLED
            else None
        )
        self._queue = queue.SimpleQueue()
        self._thread = threading.Thread(
            target=self._copy_loop,
            args=(self._queue, device, load_stream, store_stream, verify_tensors),
            daemon=True,
        )
        self._thread.start()

    def launch_copy(
        self,
        src_blocks: list[int],
        dst_blocks: list[int],
        is_store: bool,
        event_idx: int,
        events_list: list[tuple[int, torch.Event]],
        wait_event: torch.Event | None = None,
    ) -> None:
        params = self._store_params if is_store else self._load_params
        assert params is not None and self._queue is not None
        self._queue.put(
            (
                src_blocks,
                dst_blocks,
                params,
                is_store,
                event_idx,
                events_list,
                wait_event,
            )
        )

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        if self._queue is not None:
            self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    @staticmethod
    def _verify_copy(
        verify_tensors: tuple[list[torch.Tensor], list[torch.Tensor]],
        src_blocks: list[int],
        dst_blocks: list[int],
        is_store: bool,
        stream: torch.cuda.Stream,
    ) -> None:
        gpu_tensors, cpu_tensors = verify_tensors
        src_tensors, dst_tensors = (
            (gpu_tensors, cpu_tensors) if is_store else (cpu_tensors, gpu_tensors)
        )
        # Every region is a [num_blocks, block_bytes] int8 view.
        num_tensors, num_blocks = len(src_tensors), len(src_blocks)
        t_idx = np.repeat(np.arange(num_tensors), num_blocks)
        bpb = np.array([t.stride(0) for t in src_tensors], dtype=np.int64)[t_idx]
        src_base = np.array([t.data_ptr() for t in src_tensors], dtype=np.int64)
        dst_base = np.array([t.data_ptr() for t in dst_tensors], dtype=np.int64)
        src_ptrs = src_base[t_idx] + np.tile(src_blocks, num_tensors) * bpb
        dst_ptrs = dst_base[t_idx] + np.tile(dst_blocks, num_tensors) * bpb
        transfer_verifier.verify_transfer(
            is_store,
            src_tensors,
            dst_tensors,
            t_idx,
            src_ptrs,
            dst_ptrs,
            bpb,
            stream,
        )

    @staticmethod
    def _copy_loop(
        q: queue.SimpleQueue,
        device: torch.device,
        load_stream: torch.cuda.Stream,
        store_stream: torch.cuda.Stream,
        verify_tensors: tuple[list[torch.Tensor], list[torch.Tensor]] | None = None,
    ) -> None:
        current_platform.set_device(device)
        while True:
            item = q.get()
            if item is None:
                return
            (
                src_blocks,
                dst_blocks,
                params,
                is_store,
                event_idx,
                events_list,
                wait_event,
            ) = item
            stream = store_stream if is_store else load_stream
            if wait_event is not None:
                stream.wait_event(wait_event)
            copy_blocks(src_blocks, dst_blocks, params)
            if verify_tensors is not None and src_blocks:
                # Before the event is published, so completion waits for the check.
                DmaCopyBackend._verify_copy(
                    verify_tensors, src_blocks, dst_blocks, is_store, stream
                )
            event = torch.Event()
            event.record(stream)
            events_list.append((event_idx, event))
