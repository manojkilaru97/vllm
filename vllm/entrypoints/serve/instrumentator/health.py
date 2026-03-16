# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.request_metrics import recent_failure_summary
from vllm.logger import init_logger
from vllm.v1.engine.exceptions import EngineDeadError
import logging

logger = init_logger(__name__)
payload_logger = logging.getLogger("vllm.payload")


router = APIRouter()

_HEALTH_STATE_KEY = "_vllm_health_state"


@dataclass
class HealthProbeState:
    interval_s: float
    timeout_s: float
    ready_ttl_s: float
    task: asyncio.Task[None] | None = None
    started_at: float = 0.0
    last_attempt_at: float = 0.0
    last_success_at: float = 0.0
    last_failure_at: float = 0.0
    consecutive_failures: int = 0
    last_error: str = ""
    last_unhealthy_log_at: float = 0.0

    def is_ready(self) -> bool:
        if self.last_success_at <= 0:
            return False
        return (time.time() - self.last_success_at) <= self.ready_ttl_s


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning("Invalid %s=%r, using default %s", name, value, default)
        return default


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


def _get_health_state(request: Request) -> HealthProbeState | None:
    return getattr(request.app.state, _HEALTH_STATE_KEY, None)


def _app_model_name(app) -> str | None:
    model_registry = getattr(app.state, "openai_serving_models", None)
    if model_registry is None:
        return None
    try:
        return model_registry.model_name()
    except Exception:
        return None


async def _run_probe_once(app) -> None:
    state: HealthProbeState = getattr(app.state, _HEALTH_STATE_KEY)
    state.last_attempt_at = time.time()

    client: EngineClient | None = getattr(app.state, "engine_client", None)
    if client is None:
        state.last_success_at = state.last_attempt_at
        state.consecutive_failures = 0
        state.last_error = ""
        return

    await client.check_health()

    handler = getattr(app.state, "openai_serving_chat", None)
    model_name = _app_model_name(app)
    if handler is None or model_name is None:
        state.last_success_at = time.time()
        state.consecutive_failures = 0
        state.last_error = ""
        return

    request = ChatCompletionRequest(
        model=model_name,
        messages=[{"role": "user", "content": "Reply with ok."}],
        temperature=0.0,
        max_completion_tokens=1,
        stream=False,
        request_id=f"health-probe-{int(time.time())}",
    )
    response = await handler.create_chat_completion(request, raw_request=None)
    if isinstance(response, ErrorResponse):
        raise RuntimeError(
            f"health probe returned {response.error.code}: {response.error.message}"
        )
    if not isinstance(response, ChatCompletionResponse):
        raise RuntimeError("health probe did not return a non-streaming response")

    state.last_success_at = time.time()
    state.consecutive_failures = 0
    state.last_error = ""


async def _health_probe_loop(app) -> None:
    state: HealthProbeState = getattr(app.state, _HEALTH_STATE_KEY)
    while True:
        try:
            await asyncio.wait_for(_run_probe_once(app), timeout=state.timeout_s)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            state.last_failure_at = time.time()
            state.consecutive_failures += 1
            state.last_error = str(exc)
            logger.warning("Internal health probe failed: %s", exc)
        await asyncio.sleep(state.interval_s)


def start_health_probe_loop(app) -> None:
    if hasattr(app.state, _HEALTH_STATE_KEY):
        state: HealthProbeState = getattr(app.state, _HEALTH_STATE_KEY)
        if state.task is not None:
            return

    state = HealthProbeState(
        interval_s=max(1.0, _env_float("VLLM_HEALTH_PROBE_INTERVAL_SEC", 30.0)),
        timeout_s=max(1.0, _env_float("VLLM_HEALTH_PROBE_TIMEOUT_SEC", 15.0)),
        ready_ttl_s=max(1.0, _env_float("VLLM_HEALTH_READY_TTL_SEC", 300.0)),
        started_at=time.time(),
    )
    setattr(app.state, _HEALTH_STATE_KEY, state)
    state.task = asyncio.create_task(_health_probe_loop(app))


async def stop_health_probe_loop(app) -> None:
    state: HealthProbeState | None = getattr(app.state, _HEALTH_STATE_KEY, None)
    if state is None or state.task is None:
        return
    state.task.cancel()
    try:
        await state.task
    except asyncio.CancelledError:
        pass
    state.task = None


def _ready_payload(raw_request: Request, state: HealthProbeState | None) -> dict[str, float | int | str | bool]:
    client = engine_client(raw_request)
    engine_alive = client is None or not client.errored
    return {
        "alive": engine_alive,
        "ready": bool(state and state.is_ready()),
        "last_attempt_at": state.last_attempt_at if state else 0.0,
        "last_success_at": state.last_success_at if state else 0.0,
        "last_failure_at": state.last_failure_at if state else 0.0,
        "consecutive_failures": state.consecutive_failures if state else 0,
        "last_error": state.last_error if state else "",
    }


def _health_context_payload(raw_request: Request, state: HealthProbeState | None) -> dict[str, Any]:
    payload: dict[str, Any] = _ready_payload(raw_request, state)
    payload["hostname"] = os.getenv("HOSTNAME", "")
    recent_failures = recent_failure_summary(window_s=1800.0)
    payload["recent_request_failures"] = recent_failures
    fingerprint_source = {
        "alive": payload["alive"],
        "ready": payload["ready"],
        "consecutive_failures": payload["consecutive_failures"],
        "last_error": payload["last_error"],
        "hostname": payload["hostname"],
        "failure_classes": recent_failures.get("failure_classes", {}),
        "recent_items": [
            {
                "rid": item.get("rid", ""),
                "failure_class": item.get("failure_class", ""),
                "status_code": item.get("status_code", 0),
                "shape_hash": item.get("shape_hash", ""),
            }
            for item in recent_failures.get("recent_items", [])
        ],
    }
    payload["health_fingerprint"] = hashlib.sha256(
        json.dumps(
            fingerprint_source, sort_keys=True, separators=(",", ":"), default=str
        ).encode("utf-8")
    ).hexdigest()[:16]
    return payload


def _maybe_log_unhealthy(path: str, payload: dict[str, Any], state: HealthProbeState | None) -> None:
    now = time.time()
    if state is not None and (now - state.last_unhealthy_log_at) < 60:
        return
    logger.warning(
        "health check unhealthy",
        extra={
            "endpoint": path,
            "health_payload": payload,
            "health_fingerprint": payload.get("health_fingerprint", ""),
        },
    )
    try:
        payload_logger.warning(
            "health.unhealthy",
            extra={
                "endpoint": path,
                "health_fingerprint": payload.get("health_fingerprint", ""),
                "health_payload": payload,
            },
        )
    except Exception:
        pass
    if state is not None:
        state.last_unhealthy_log_at = now


@router.get("/live", response_class=Response)
async def live(raw_request: Request) -> Response:
    """Liveness check: only fail when the engine is actually dead."""
    client = engine_client(raw_request)
    if client is None:
        return Response(status_code=200)
    try:
        await client.check_health()
        return Response(status_code=200)
    except EngineDeadError:
        state = _get_health_state(raw_request)
        payload = _health_context_payload(raw_request, state)
        _maybe_log_unhealthy("/live", payload, state)
        return Response(status_code=503)


@router.get("/ready")
async def ready(raw_request: Request) -> JSONResponse:
    """Readiness check backed by an internal synthetic completion probe."""
    client = engine_client(raw_request)
    state = _get_health_state(raw_request)
    if client is not None:
        try:
            await client.check_health()
        except EngineDeadError:
            payload = _health_context_payload(raw_request, state)
            _maybe_log_unhealthy("/ready", payload, state)
            return JSONResponse(
                payload,
                status_code=503,
            )

    payload = _health_context_payload(raw_request, state)
    if not payload["ready"]:
        _maybe_log_unhealthy("/ready", payload, state)
    return JSONResponse(payload, status_code=200 if payload["ready"] else 503)


@router.get("/health")
async def health(raw_request: Request) -> JSONResponse:
    """Backward-compatible readiness endpoint."""
    return await ready(raw_request)
