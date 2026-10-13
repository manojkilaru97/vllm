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

from vllm.logger import init_logger
from vllm.multimodal.media.video_audio import (
    _CODEC_ID,
    _COLLECTION_SLACK_S,
    _TRACK_ENTRY,
    _TRACK_TYPE,
    _TRACKS,
    UnsupportedVideoAudioError,
    _AudioTrack,
    _DurationExceeded,
    _ebml_uint,
    _iter_ebml,
    _iter_mp4_boxes,
    _matroska_segment,
    _segment_children,
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
# Output ceiling when the caller passes no duration limit.
_DEFAULT_MAX_DURATION_S = 3600.0
_CONTENT_ENCODINGS = 0x6D80
# Encoder priming/padding (e.g. 1024 AAC samples) lengthens decoded output.
_DURATION_TOLERANCE_S = 0.25
_SLOTS = threading.BoundedSemaphore(4)
# Codec names as reported by video_audio's track probe.
_DECODABLE = {
    "mp4": frozenset(
        {"aac", "opus", "flac", ".mp3", "lpcm", "sowt", "twos", "in24", "in32",
         "fl32", "fl64", "raw", "ulaw", "alaw", "ipcm", "fpcm"}
    ),
    "matroska": frozenset({"aac", "mpeg", "opus", "vorbis", "flac", "pcm"}),
}  # fmt: skip

logger = init_logger(__name__)


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
        logger.warning_once(
            "No executable audio-only FFmpeg CLI at %s (set %s); video audio "
            "tracks other than Matroska Opus will be rejected.",
            os.environ.get(_AUDIO_CLI_ENV, _DEFAULT_AUDIO_CLI),
            _AUDIO_CLI_ENV,
        )
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


def _matroska_audio_entry_problem(data: bytes) -> str | None:
    """Why FFmpeg would not decode the probed (first) Matroska audio entry."""
    segment = _matroska_segment(data)
    if segment is None:
        return None
    for element_id, body, end in _segment_children(data, *segment):
        if element_id != _TRACKS:
            continue
        for entry_id, entry, entry_end, _ in _iter_ebml(data, body, end):
            if entry_id != _TRACK_ENTRY:
                continue
            fields = {
                fid: (fb, fe) for fid, fb, fe, _ in _iter_ebml(data, entry, entry_end)
            }
            if _TRACK_TYPE not in fields or _ebml_uint(data, *fields[_TRACK_TYPE]) != 2:
                continue
            codec_id = data[slice(*fields.get(_CODEC_ID, (0, 0)))].rstrip(b"\0")
            # FFmpeg skips entries whose CodecID is not an audio id, so its first
            # audio stream would be a later track.
            if not codec_id.startswith(b"A_"):
                return "declares a non-audio codec id"
            # FFmpeg drops packets failing zlib decompression without an error.
            if _CONTENT_ENCODINGS in fields:
                return "uses Matroska content encoding (compression or encryption)"
            return None
        return None
    return None


def _mp4_audio_presentation_s(data: bytes) -> float | None:
    """The first sound track's edited duration (tkhd, in mvhd timescale)."""
    import struct

    def children(start: int, end: int):
        return _iter_mp4_boxes(data, start, end)

    moov = next(((b, e) for t, b, e in children(0, len(data)) if t == b"moov"), None)
    if moov is None:
        return None
    timescale = None
    for box, body, end in children(*moov):
        if box == b"mvhd" and end - body >= 32:
            version = data[body]
            timescale = struct.unpack_from(">I", data, body + (20 if version else 12))[
                0
            ]
        elif box == b"trak" and timescale:
            tkhd = mdia = None
            for sub, sbody, send in children(body, end):
                tkhd = (sbody, send) if sub == b"tkhd" else tkhd
                mdia = (sbody, send) if sub == b"mdia" else mdia
            if tkhd is None or mdia is None:
                continue
            handler = next(
                (
                    data[hb + 8 : hb + 12]
                    for t, hb, _ in children(*mdia)
                    if t == b"hdlr"
                ),
                None,
            )
            if handler != b"soun":
                continue
            version = data[tkhd[0]]
            fmt, offset = (">Q", 28) if version else (">I", 20)
            if tkhd[1] - tkhd[0] < offset + struct.calcsize(fmt):
                return None
            duration = struct.unpack_from(fmt, data, tkhd[0] + offset)[0]
            return duration / timescale if duration else None
    return None


def _reason(stderr: bytes, returncode: int) -> str:
    lines = [
        re.sub(r"\[[^\]]* @ 0x[0-9a-f]+\]\s*", "", line).strip()
        for line in stderr.decode("utf-8", "replace").splitlines()
    ]
    lines = [
        line for line in lines if line and not line.startswith("Last message repeated")
    ]
    decoder_lines = [line for line in lines if "decoder" in line.lower()]
    return (decoder_lines or lines or [f"exit status {returncode}"])[-1][:200]


class _OutputLimit(Exception):
    pass


def _run_bounded(command: list[str], max_stdout: int) -> tuple[int, bytes, bytes]:
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
            raise
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
    # Only tracks this CLI decodes. FFmpeg skips entries with a missing or
    # non-audio codec id, so its first audio stream would not be the probed one.
    if track.codec not in _DECODABLE.get(track.container, ()):
        raise UnsupportedVideoAudioError(
            f"The video's audio track uses codec '{track.codec or 'unknown'}' in a "
            f"{track.container} container, which this server cannot decode. Use "
            "AAC, MP3, Opus, Vorbis, FLAC or PCM audio."
        )
    problem = (
        _matroska_audio_entry_problem(data) if track.container == "matroska" else None
    )
    if problem is not None:
        raise UnsupportedVideoAudioError(
            f"The video's audio track {problem}, which this server does not decode."
        )
    if len(data) > _MAX_INPUT_BYTES:
        raise UnsupportedVideoAudioError(
            f"The video is larger than {_MAX_INPUT_BYTES >> 20} MiB, the limit for "
            "extracting its audio track."
        )
    # Bound the output by sample count, not timestamps (which the client controls).
    limit_s = _DEFAULT_MAX_DURATION_S if max_duration_s is None else max_duration_s
    max_samples = int((limit_s + _COLLECTION_SLACK_S) * _SAMPLE_RATE)
    filters = (
        f"aresample={_SAMPLE_RATE},aformat=sample_fmts=flt:channel_layouts=mono"
        f",atrim=end_sample={max_samples}"
    )
    # Restamp from the sample count so client timestamps never reach the muxer.
    filters += ",asetpts=N/SR/TB"
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
            try:
                returncode, stdout, stderr = _run_bounded(command, max_samples * 4)
            except _OutputLimit:
                raise _DurationExceeded(limit_s) from None
    finally:
        _SLOTS.release()
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
    decoded_s = audio.size / _SAMPLE_RATE
    if decoded_s > limit_s + _DURATION_TOLERANCE_S:
        raise _DurationExceeded(limit_s)
    # Edit lists are honoured (trims stay trimmed); a decode far shorter than the
    # track's edited duration means the sample table was inconsistent.
    expected_s = _mp4_audio_presentation_s(data) if track.container == "mp4" else None
    if expected_s is not None and decoded_s < min(expected_s, limit_s) * 0.9 - 0.1:
        raise UnsupportedVideoAudioError(
            f"The video's {track.codec} audio track decoded to {decoded_s:.2f} s of "
            f"its declared {expected_s:.2f} s."
        )
    return audio.astype(np.float32), float(_SAMPLE_RATE)
