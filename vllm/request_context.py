# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Optional

# Request-scoped context used for correlating async side effects (e.g. media mirroring)
# back to the originating OpenAI request.
_REQUEST_ID: ContextVar[Optional[str]] = ContextVar("vllm_request_id", default=None)


def set_request_id(rid: str | None) -> Token[Optional[str]]:
    return _REQUEST_ID.set(rid if rid else None)


def reset_request_id(token: Token[Optional[str]]) -> None:
    _REQUEST_ID.reset(token)


def get_request_id() -> str | None:
    return _REQUEST_ID.get()

