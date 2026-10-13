# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode a video's audio track with an audio-only FFmpeg CLI.

Imported only by ``video_audio.load_video_audio_without_pyav(decoder_cli=True)``,
which callers pass for audio-capable models (Nemotron Omni ``sound_config``).
The CLI (``NV_OMNI_AUDIO_FFMPEG``, default ``/opt/nvidia/omni-audio/bin/ffmpeg``)
runs as a bounded subprocess, so no decoder library is loaded into the serving
process. Tracks libsndfile decodes (Matroska Opus) keep that path; any other
track (AAC, MP3, Vorbis, FLAC, PCM), or an Opus track libsndfile cannot decode,
goes to the CLI. Without the CLI the libsndfile behaviour is unchanged.
"""

import os
import re
import selectors
import subprocess
import tempfile
import threading
import time

import numpy as np
import numpy.typing as npt

from vllm.multimodal.media.video_audio import (
    _COLLECTION_SLACK_S,
    UnsupportedVideoAudioError,
    _AudioTrack,
    _DurationExceeded,
    load_video_audio_without_pyav,
)

_AUDIO_CLI_ENV = "NV_OMNI_AUDIO_FFMPEG"
_DEFAULT_AUDIO_CLI = "/opt/nvidia/omni-audio/bin/ffmpeg"
_PRLIMIT = "/usr/bin/prlimit"
_SAMPLE_RATE = 48000
_TIMEOUT_S = 120.0
_MEMORY_BYTES = 1 << 30
_MAX_INPUT_BYTES = 512 << 20
_MAX_STDERR_BYTES = 64 << 10
# Encoder priming/padding (e.g. 1024 AAC samples) lengthens decoded output.
_DURATION_TOLERANCE_S = 0.25
_SLOTS = threading.BoundedSemaphore(4)


def audio_decoder_cli() -> str | None:
    path = os.environ.get(_AUDIO_CLI_ENV, _DEFAULT_AUDIO_CLI)
    if path and os.access(path, os.X_OK) and os.access(_PRLIMIT, os.X_OK):
        return path
    return None


def decode_with_cli_fallback(
    data: bytes, track: _AudioTrack, max_duration_s: float | None
) -> tuple[npt.NDArray, float]:
    """libsndfile first (Matroska Opus); the CLI for anything it cannot decode."""
    cli = audio_decoder_cli()
    if cli is None:
        result = load_video_audio_without_pyav(data, max_duration_s=max_duration_s)
        assert result is not None
        return result
    try:
        result = load_video_audio_without_pyav(data, max_duration_s=max_duration_s)
    except _DurationExceeded:
        raise
    except UnsupportedVideoAudioError:
        return decode_with_cli(cli, data, track, max_duration_s)
    assert result is not None
    return result


def _reason(stderr: bytes, returncode: int) -> str:
    lines = [
        re.sub(r"\[[^\]]* @ 0x[0-9a-f]+\]\s*", "", line).strip()
        for line in stderr.decode("utf-8", "replace").splitlines()
    ]
    lines = [line for line in lines if line]
    decoder_lines = [line for line in lines if "decoder" in line.lower()]
    return (decoder_lines or lines or [f"exit status {returncode}"])[-1][:200]


class _OutputLimit(Exception):
    pass


def _run_bounded(
    command: list[str], max_stdout: int | None
) -> tuple[int, bytes, bytes]:
    """Run ``command`` with bounded stdout/stderr and a deadline; always reap it."""
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None and process.stderr is not None
    stdout: list[bytes] = []
    stderr: list[bytes] = []
    sizes = {process.stdout: 0, process.stderr: 0}
    limits = {process.stdout: max_stdout, process.stderr: _MAX_STDERR_BYTES}
    buffers = {process.stdout: stdout, process.stderr: stderr}
    deadline = time.monotonic() + _TIMEOUT_S
    try:
        with selectors.DefaultSelector() as selector:
            for stream in buffers:
                selector.register(stream, selectors.EVENT_READ)
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError
                for key, _ in selector.select(remaining):
                    stream = key.fileobj
                    chunk = os.read(stream.fileno(), 1 << 20)
                    if not chunk:
                        selector.unregister(stream)
                        continue
                    sizes[stream] += len(chunk)
                    limit = limits[stream]
                    if limit is not None and sizes[stream] > limit:
                        raise _OutputLimit(stream is process.stdout)
                    buffers[stream].append(chunk)
        process.wait(timeout=max(deadline - time.monotonic(), 1.0))
    except (TimeoutError, subprocess.TimeoutExpired):
        raise UnsupportedVideoAudioError(
            "Decoding the video's audio track timed out."
        ) from None
    except _OutputLimit as exc:
        if exc.args[0]:
            return -1, b"", b""
        raise UnsupportedVideoAudioError(
            "Decoding the video's audio track produced too many errors."
        ) from None
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
        process.stdout.close()
        process.stderr.close()
    return process.returncode, b"".join(stdout), b"".join(stderr)


def decode_with_cli(
    cli: str, data: bytes, track: _AudioTrack, max_duration_s: float | None
) -> tuple[npt.NDArray, float]:
    """Decode the first audio stream to 48 kHz mono float32 with the FFmpeg CLI."""
    if len(data) > _MAX_INPUT_BYTES:
        raise UnsupportedVideoAudioError(
            f"The video is larger than {_MAX_INPUT_BYTES >> 20} MiB, the limit for "
            "extracting its audio track."
        )
    # Bound the output by sample count, not timestamps (which the client controls).
    max_samples = None
    filters = f"aresample={_SAMPLE_RATE},aformat=sample_fmts=flt:channel_layouts=mono"
    if max_duration_s is not None:
        max_samples = int((max_duration_s + _COLLECTION_SLACK_S) * _SAMPLE_RATE)
        filters += f",atrim=end_sample={max_samples}"
    if not _SLOTS.acquire(timeout=_TIMEOUT_S):
        raise RuntimeError("All audio decoder slots are busy.")
    try:
        with tempfile.TemporaryDirectory(prefix="vllm-video-audio-") as workdir:
            source = os.path.join(workdir, "video")
            try:
                with open(source, "wb") as staged:
                    staged.write(data)
            except OSError as exc:
                raise RuntimeError(
                    "Cannot stage the video for audio decoding."
                ) from exc
            # MP4 demuxing needs a seekable input (the moov box may follow mdat).
            command = [
                _PRLIMIT, f"--as={_MEMORY_BYTES}", "--",
                cli, "-nostdin", "-hide_banner", "-loglevel", "error", "-xerror",
                "-threads", "1", "-filter_threads", "1",
                "-protocol_whitelist", "file", "-i", source,
                "-map", "0:a:0", "-vn", "-sn", "-dn", "-af", filters,
                "-f", "f32le", "-c:a", "pcm_f32le", "pipe:1",
            ]  # fmt: skip
            max_bytes = None if max_samples is None else max_samples * 4
            returncode, stdout, stderr = _run_bounded(command, max_bytes)
    finally:
        _SLOTS.release()
    if returncode == -1:
        raise _DurationExceeded(max_duration_s)
    # At -loglevel error any diagnostic (e.g. "File ended prematurely") means the
    # track was not decoded in full, even when FFmpeg exits 0.
    if returncode != 0 or stderr.strip():
        raise UnsupportedVideoAudioError(
            f"Cannot decode the video's {track.codec} audio track: "
            f"{_reason(stderr, returncode)}"
        )
    audio = np.frombuffer(stdout, dtype="<f4")
    if audio.size == 0:
        raise UnsupportedVideoAudioError("The video's decoded audio track is empty.")
    if (
        max_duration_s is not None
        and audio.size / _SAMPLE_RATE > max_duration_s + _DURATION_TOLERANCE_S
    ):
        raise _DurationExceeded(max_duration_s)
    return audio.astype(np.float32), float(_SAMPLE_RATE)
