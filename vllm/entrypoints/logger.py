# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
import json
import re
from collections.abc import Sequence

import torch

from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import BeamSearchParams, SamplingParams

logger = init_logger(__name__)


class RequestLogger:
    def __init__(self, *, max_log_len: int | None) -> None:
        self.max_log_len = max_log_len

    def log_inputs(
        self,
        request_id: str,
        prompt: str | None,
        prompt_token_ids: list[int] | None,
        prompt_embeds: torch.Tensor | None,
        params: SamplingParams | PoolingParams | BeamSearchParams | None,
        lora_request: LoRARequest | None,
    ) -> None:
        if logger.isEnabledFor(logging.DEBUG):
            max_log_len = self.max_log_len
            if max_log_len is not None:
                if prompt is not None:
                    prompt = prompt[:max_log_len]

                if prompt_token_ids is not None:
                    prompt_token_ids = prompt_token_ids[:max_log_len]

            logger.debug(
                "Request %s details: prompt: %r, "
                "prompt_token_ids: %s, "
                "prompt_embeds shape: %s.",
                request_id,
                prompt,
                prompt_token_ids,
                prompt_embeds.shape if prompt_embeds is not None else None,
            )

        params_for_log: SamplingParams | PoolingParams | BeamSearchParams | str | None = (
            params
        )
        if isinstance(params, SamplingParams):
            params_for_log = self._summarize_sampling_params_for_log(params)

        logger.info(
            "Received request %s: params: %s, lora_request: %s.",
            request_id,
            params_for_log,
            lora_request,
        )

    def _summarize_sampling_params_for_log(self, params: SamplingParams) -> str:
        params_text = str(params)
        extra_args = params.extra_args
        if not isinstance(extra_args, dict):
            return params_text

        newline_token_ids = extra_args.get("newline_token_ids")
        if not (
            isinstance(newline_token_ids, Sequence)
            and not isinstance(newline_token_ids, (str, bytes, bytearray))
        ):
            return params_text

        try:
            count = len(newline_token_ids)
        except Exception:
            return params_text

        preview = []
        if hasattr(newline_token_ids, "__getitem__"):
            try:
                preview = list(newline_token_ids[:8])  # type: ignore[index]
            except Exception:
                preview = []
        summary = f"<{count} token_ids; preview={preview}>"

        return re.sub(
            r"'newline_token_ids': \[[^\]]*\]",
            f"'newline_token_ids': '{summary}'",
            params_text,
            count=1,
        )

    def log_http_request(
        self,
        request_id: str,
        *,
        method: str,
        path: str,
        headers: dict[str, str] | None,
        body: bytes | str | None,
    ) -> None:
        body_text: str | None
        if isinstance(body, bytes):
            body_text = body.decode("utf-8", errors="replace")
        else:
            body_text = body

        max_log_len = self.max_log_len
        if max_log_len is not None and body_text is not None:
            body_text = body_text[:max_log_len]

        payload = {
            "method": method,
            "path": path,
            "headers": headers or {},
            "body": body_text,
        }
        logger.info(
            "Received raw HTTP request %s: %s",
            request_id,
            json.dumps(payload, ensure_ascii=False),
        )

    def log_outputs(
        self,
        request_id: str,
        outputs: str,
        output_token_ids: Sequence[int] | None,
        finish_reason: str | None = None,
        is_streaming: bool = False,
        delta: bool = False,
    ) -> None:
        max_log_len = self.max_log_len
        if max_log_len is not None:
            if outputs is not None:
                outputs = outputs[:max_log_len]

            # We no longer log output_token_ids to reduce log volume
            # but we keep the parameter for backward compatibility with callers.

        stream_info = ""
        if is_streaming:
            stream_info = " (streaming delta)" if delta else " (streaming complete)"

        logger.info(
            "Generated response %s%s: output: %r, finish_reason: %s",
            request_id,
            stream_info,
            outputs,
            finish_reason,
        )
