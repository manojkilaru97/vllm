# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Extract the audio track of an encoded video without PyAV.

PyAV is the default decoder for ``use_audio_in_video``. Images that ship no
PyAV (and no software AAC/Opus decoder) can still decode Opus through
libsndfile, which reads Ogg Opus. This module finds the audio track of an
ISO-BMFF (MP4/MOV) or Matroska (WebM/MKV) container, remuxes a Matroska Opus
track into an in-memory Ogg Opus stream, and decodes it with ``soundfile``.

``load_video_audio_without_pyav`` returns ``None`` only when an MP4/MOV or
Matroska/WebM container has no audio track, and raises
``UnsupportedVideoAudioError`` when an audio track exists that this process
cannot decode or the container cannot be inspected, so callers never silently
drop narration.
"""

import struct
from dataclasses import dataclass
from io import BytesIO

import numpy as np
import numpy.typing as npt

from vllm.logger import init_logger

logger = init_logger(__name__)

_MAX_BOX_DEPTH = 8
_COLLECTION_SLACK_S = 1.0


_VISUAL_ONLY_REMEDY = (
    'Set mm_processor_kwargs {"use_audio_in_video": false} for a visual-only answer.'
)


class UnsupportedVideoAudioError(ValueError):
    """The video has an audio track that cannot be decoded here."""

    def __init__(self, message: str) -> None:
        if "use_audio_in_video" not in message:
            message = f"{message} {_VISUAL_ONLY_REMEDY}"
        super().__init__(message)


class _DurationExceeded(ValueError):
    def __init__(self, max_duration_s: float | None) -> None:
        super().__init__(
            f"Audio exceeds maximum allowed duration of {max_duration_s}s. "
            "Set VLLM_MAX_AUDIO_DECODE_DURATION_S to increase this limit."
        )


@dataclass
class _AudioTrack:
    container: str
    codec: str
    number: int | None = None
    codec_private: bytes | None = None
    channels: int | None = None
    sampling_rate: float | None = None


# ---------------------------------------------------------------------------
# ISO-BMFF (MP4 / MOV)
# ---------------------------------------------------------------------------

_MP4_CONTAINER_BOXES = {b"moov", b"trak", b"mdia", b"minf", b"stbl"}


def _iter_mp4_boxes(data: bytes, start: int, end: int, strict: bool = False):
    """Yield ``(type, body, end)`` boxes; ``strict`` raises on truncation."""
    offset = start
    while offset + 8 <= end:
        size, box_type = struct.unpack_from(">I4s", data, offset)
        header = 8
        if size == 1:
            if offset + 16 > end:
                if strict:
                    raise ValueError("truncated MP4 box header")
                return
            size = struct.unpack_from(">Q", data, offset + 8)[0]
            header = 16
        elif size == 0:
            size = end - offset
        if size < header or offset + size > end:
            if strict:
                raise ValueError("truncated or malformed MP4 box")
            return
        yield box_type, offset + header, offset + size
        offset += size


def _mp4_audio_track(data: bytes) -> _AudioTrack | None:
    def walk(
        start: int,
        end: int,
        track: dict[str, bytes],
        depth: int = 0,
        parent: bytes = b"",
    ):
        if depth > _MAX_BOX_DEPTH:
            raise ValueError("MP4 boxes are nested too deeply")
        for box_type, body, box_end in _iter_mp4_boxes(data, start, end, strict=True):
            if box_type == b"trak":
                inner: dict[str, bytes] = {}
                walk(body, box_end, inner, depth + 1, box_type)
                if inner.get("handler") == b"soun":
                    return inner.get("codec", b"????")
            elif box_type in _MP4_CONTAINER_BOXES:
                found = walk(body, box_end, track, depth + 1, box_type)
                if found is not None:
                    return found
            elif box_type == b"hdlr" and parent == b"mdia" and body + 12 <= box_end:
                # Only the media handler identifies the track type; QuickTime
                # also puts data-reference handlers (`url `, `alis`) in `minf`.
                track["handler"] = data[body + 8 : body + 12]
            elif box_type == b"stsd" and body + 16 <= box_end:
                track["codec"] = data[body + 12 : body + 16]
            elif box_type == b"cmov":
                raise ValueError("compressed QuickTime movie metadata (cmov)")
        return None

    codec = None
    moov_walked = False
    try:
        for box_type, body, box_end in _iter_mp4_boxes(data, 0, len(data), strict=True):
            if box_type == b"moov":
                codec = walk(body, box_end, {}, 1, box_type)
                if codec is not None:
                    break
                moov_walked = True
    except ValueError:
        # A cut or padded tail after a complete `moov` (e.g. a truncated
        # `mdat`) does not change the track list; without one, re-raise.
        if not moov_walked:
            raise
    if codec is None:
        if not moov_walked:
            raise ValueError("MP4 container has no moov box")
        return None
    name = {b"mp4a": "aac", b"Opus": "opus", b"fLaC": "flac", b"ac-3": "ac3"}.get(
        codec, codec.decode("latin-1").strip()
    )
    return _AudioTrack(container="mp4", codec=name)


# ---------------------------------------------------------------------------
# Matroska / WebM
# ---------------------------------------------------------------------------

_EBML_HEADER = 0x1A45DFA3
_SEGMENT = 0x18538067
_TRACKS = 0x1654AE6B
_TRACK_ENTRY = 0xAE
_TRACK_NUMBER = 0xD7
_TRACK_TYPE = 0x83
_CODEC_ID = 0x86
_CODEC_PRIVATE = 0x63A2
_AUDIO = 0xE1
_SAMPLING_FREQUENCY = 0xB5
_CHANNELS = 0x9F
_CLUSTER = 0x1F43B675
_SIMPLE_BLOCK = 0xA3
_BLOCK_GROUP = 0xA0
_BLOCK = 0xA1
_DISCARD_PADDING = 0x75A2
_SEGMENT_CHILDREN = frozenset(
    {
        0x114D9B74,  # SeekHead
        0x1549A966,  # Info
        _TRACKS,
        _CLUSTER,
        0x1C53BB6B,  # Cues
        0x1941A469,  # Attachments
        0x1043A770,  # Chapters
        0x1254C367,  # Tags
    }
)


def _read_vint(data: bytes, offset: int, keep_marker: bool) -> tuple[int, int, bool]:
    """Return ``(value, length, is_unknown_size)`` of an EBML variable int."""
    if offset >= len(data):
        raise ValueError("truncated EBML element")
    first = data[offset]
    length = 1
    mask = 0x80
    while length <= 8 and not first & mask:
        mask >>= 1
        length += 1
    if length > 8 or offset + length > len(data):
        raise ValueError("invalid EBML variable-length integer")
    value = first if keep_marker else first & (mask - 1)
    for byte in data[offset + 1 : offset + length]:
        value = (value << 8) | byte
    unknown = not keep_marker and value == (1 << (7 * length)) - 1
    return value, length, unknown


def _iter_ebml(data: bytes, start: int, end: int, stop_ids=frozenset()):
    offset = start
    while offset < end:
        element_id, id_len, _ = _read_vint(data, offset, keep_marker=True)
        if element_id in stop_ids:
            return
        size, size_len, unknown = _read_vint(data, offset + id_len, keep_marker=False)
        body = offset + id_len + size_len
        element_end = end if unknown else body + size
        if element_end > end:
            element_end = end
        yield element_id, body, element_end, unknown
        if unknown:
            return
        offset = element_end


def _ebml_uint(data: bytes, start: int, end: int) -> int:
    return int.from_bytes(data[start:end], "big") if end > start else 0


def _ebml_float(data: bytes, start: int, end: int) -> float:
    if end - start == 4:
        return struct.unpack(">f", data[start:end])[0]
    if end - start == 8:
        return struct.unpack(">d", data[start:end])[0]
    return 0.0


def _segment_children(data: bytes, start: int, end: int):
    """Yield ``(id, body, end)`` for each Segment child.

    An unknown-size child (a live/streamed Cluster) ends where the next
    Segment-level element starts, so later Tracks/Clusters stay reachable.
    """
    offset = start
    while offset < end:
        element_id, id_len, _ = _read_vint(data, offset, keep_marker=True)
        size, size_len, unknown = _read_vint(data, offset + id_len, keep_marker=False)
        body = offset + id_len + size_len
        if unknown:
            element_end = body
            for _, _, child_end, _ in _iter_ebml(
                data, body, end, stop_ids=_SEGMENT_CHILDREN
            ):
                element_end = child_end
        else:
            element_end = min(body + size, end)
        yield element_id, body, element_end
        if element_end <= offset:
            return
        offset = element_end


def _matroska_audio_track(data: bytes, segment: tuple[int, int]) -> _AudioTrack | None:
    seg_start, seg_end = segment
    for element_id, body, end in _segment_children(data, seg_start, seg_end):
        if element_id != _TRACKS:
            continue
        for entry_id, entry, entry_end, _ in _iter_ebml(data, body, end):
            if entry_id != _TRACK_ENTRY:
                continue
            fields: dict[int, tuple[int, int]] = {}
            audio: dict[int, tuple[int, int]] = {}
            for field_id, fbody, fend, _ in _iter_ebml(data, entry, entry_end):
                fields[field_id] = (fbody, fend)
                if field_id == _AUDIO:
                    for aid, abody, aend, _ in _iter_ebml(data, fbody, fend):
                        audio[aid] = (abody, aend)
            if _TRACK_TYPE not in fields or _ebml_uint(data, *fields[_TRACK_TYPE]) != 2:
                continue
            codec_id = (
                data[slice(*fields[_CODEC_ID])].decode("ascii", "replace").rstrip("\0")
                if _CODEC_ID in fields
                else ""
            )
            codec = {"A_OPUS": "opus", "A_VORBIS": "vorbis"}.get(codec_id)
            if codec is None:
                codec = codec_id.removeprefix("A_").split("/")[0].lower() or "unknown"
            return _AudioTrack(
                container="matroska",
                codec=codec,
                number=_ebml_uint(data, *fields[_TRACK_NUMBER])
                if _TRACK_NUMBER in fields
                else None,
                codec_private=data[slice(*fields[_CODEC_PRIVATE])]
                if _CODEC_PRIVATE in fields
                else None,
                channels=_ebml_uint(data, *audio[_CHANNELS])
                if _CHANNELS in audio
                else None,
                sampling_rate=_ebml_float(data, *audio[_SAMPLING_FREQUENCY])
                if _SAMPLING_FREQUENCY in audio
                else None,
            )
        return None
    return None


def _matroska_segment(data: bytes) -> tuple[int, int] | None:
    if not data.startswith(b"\x1a\x45\xdf\xa3"):
        return None
    elements = _iter_ebml(data, 0, len(data))
    first = next(elements, None)
    if first is None or first[0] != _EBML_HEADER:
        return None
    for element_id, body, end, _ in elements:
        if element_id == _SEGMENT:
            return body, end
    return None


def _split_laced_frames(payload: bytes, lacing: int) -> list[bytes]:
    if lacing == 0:
        return [payload]
    count = payload[0] + 1
    offset = 1
    if lacing == 2:
        size, rem = divmod(len(payload) - offset, count)
        if rem:
            raise ValueError("invalid fixed-size lacing")
        return [
            payload[offset + i * size : offset + (i + 1) * size] for i in range(count)
        ]
    sizes: list[int] = []
    if lacing == 1:
        for _ in range(count - 1):
            size = 0
            while True:
                byte = payload[offset]
                offset += 1
                size += byte
                if byte != 255:
                    break
            sizes.append(size)
    else:
        first, length, _ = _read_vint(payload, offset, keep_marker=False)
        offset += length
        sizes.append(first)
        for _ in range(count - 2):
            raw, length, _ = _read_vint(payload, offset, keep_marker=False)
            offset += length
            sizes.append(sizes[-1] + raw - ((1 << (7 * length - 1)) - 1))
    frames = []
    for size in sizes:
        frames.append(payload[offset : offset + size])
        offset += size
    frames.append(payload[offset:])
    return frames


@dataclass
class _OpusBlocks:
    packets: list[bytes]
    leading_discard_ns: int = 0
    trailing_discard_ns: int = 0
    trailing_block_samples: int = 0


def _check_block_padding(blocks: _OpusBlocks, discard_ns: int, first_block: bool):
    """Ogg can trim only the stream edges; reject padding anywhere else."""
    if blocks.trailing_discard_ns or (discard_ns < 0 and not first_block):
        raise UnsupportedVideoAudioError(
            "The video's Opus track uses DiscardPadding inside the stream, "
            "which this server cannot apply."
        )


def _matroska_track_packets(
    data: bytes,
    segment: tuple[int, int],
    track_number: int,
    max_duration_s: float | None = None,
) -> _OpusBlocks:
    """Collect the track's packets and the stream-edge ``DiscardPadding``.

    Matroska ``DiscardPadding`` applies to a whole (possibly laced) block:
    negative on the first block trims its start, positive on the last block
    trims its end. Ogg can only express those two edges (pre-skip and the
    final granule); padding on interior blocks is not representable.
    """
    blocks = _OpusBlocks(packets=[])
    first_block = True
    total_samples = 0
    # Packets include pre-skip and edge padding that decoding trims; allow some
    # slack here and enforce the exact limit on the decoded audio.
    max_samples = (
        None
        if max_duration_s is None
        else (max_duration_s + _COLLECTION_SLACK_S) * 48000
    )

    def block_packets(body: int, end: int, discard_ns: int) -> None:
        nonlocal first_block, total_samples
        number, length, _ = _read_vint(data, body, keep_marker=False)
        if number != track_number:
            return
        header = body + length
        flags = data[header + 2]
        frames = _split_laced_frames(data[header + 3 : end], (flags >> 1) & 3)
        block_samples = sum(_opus_packet_samples(f) for f in frames)
        total_samples += block_samples
        if max_samples is not None and total_samples > max_samples:
            raise _DurationExceeded(max_duration_s)
        _check_block_padding(blocks, discard_ns, first_block)
        blocks.packets.extend(frames)
        if first_block and discard_ns < 0:
            blocks.leading_discard_ns = -discard_ns
        first_block = False
        blocks.trailing_discard_ns = max(discard_ns, 0)
        blocks.trailing_block_samples = block_samples

    seg_start, seg_end = segment
    for element_id, body, end in _segment_children(data, seg_start, seg_end):
        if element_id != _CLUSTER:
            continue
        for child_id, cbody, cend, _ in _iter_ebml(data, body, end):
            if child_id == _SIMPLE_BLOCK:
                block_packets(cbody, cend, 0)
            elif child_id == _BLOCK_GROUP:
                block, discard_ns = None, 0
                for gid, gbody, gend, _ in _iter_ebml(data, cbody, cend):
                    if gid == _BLOCK:
                        block = (gbody, gend)
                    elif gid == _DISCARD_PADDING and gend > gbody:
                        discard_ns = int.from_bytes(
                            data[gbody:gend], "big", signed=True
                        )
                if block is not None:
                    block_packets(*block, discard_ns)
    return blocks


# ---------------------------------------------------------------------------
# Ogg Opus muxing (RFC 7845)
# ---------------------------------------------------------------------------


def _ogg_crc_table() -> list[int]:
    table = []
    for i in range(256):
        crc = i << 24
        for _ in range(8):
            crc = ((crc << 1) ^ 0x04C11DB7) if crc & 0x80000000 else crc << 1
        table.append(crc & 0xFFFFFFFF)
    return table


_OGG_CRC_TABLE = _ogg_crc_table()


def _ogg_crc(page: bytes) -> int:
    crc = 0
    for byte in page:
        crc = ((crc << 8) & 0xFFFFFFFF) ^ _OGG_CRC_TABLE[((crc >> 24) & 0xFF) ^ byte]
    return crc


def _ogg_page(packet: bytes, *, granule: int, sequence: int, header_type: int) -> bytes:
    lacing = bytes([255] * (len(packet) // 255) + [len(packet) % 255])
    if len(lacing) > 255:
        raise ValueError("Opus packet too large for a single Ogg page")
    header = struct.pack(
        "<4sBBqIIIB",
        b"OggS",
        0,
        header_type,
        granule,
        0x6F6D6E69,
        sequence,
        0,
        len(lacing),
    )
    page = bytearray(header + lacing + packet)
    struct.pack_into("<I", page, 22, _ogg_crc(page))
    return bytes(page)


_OPUS_FRAME_SAMPLES_48K = (
    [480, 960, 1920, 2880] * 3  # SILK-only: 10/20/40/60 ms
    + [480, 960] * 2  # hybrid: 10/20 ms
    + [120, 240, 480, 960] * 4  # CELT-only: 2.5/5/10/20 ms
)


def _opus_packet_samples(packet: bytes) -> int:
    if not packet:
        return 0
    toc = packet[0]
    frame_count = toc & 3
    if frame_count == 0:
        frames = 1
    elif frame_count in (1, 2):
        frames = 2
    else:
        frames = packet[1] & 0x3F if len(packet) > 1 else 0
    return frames * _OPUS_FRAME_SAMPLES_48K[toc >> 3]


def _ns_to_48k(ns: int) -> int:
    return round(ns * 48000 / 1_000_000_000)


def _opus_head(track: _AudioTrack, extra_pre_skip: int = 0) -> bytes:
    if track.codec_private and track.codec_private.startswith(b"OpusHead"):
        head = bytearray(track.codec_private)
    else:
        channels = track.channels or 1
        rate = int(track.sampling_rate or 48000)
        head = bytearray(
            struct.pack("<8sBBHIhB", b"OpusHead", 1, channels, 312, rate, 0, 0)
        )
    if extra_pre_skip and len(head) >= 12:
        pre_skip = struct.unpack_from("<H", head, 10)[0]
        struct.pack_into("<H", head, 10, min(pre_skip + extra_pre_skip, 0xFFFF))
    return bytes(head)


def _ogg_opus_from_packets(track: _AudioTrack, blocks: _OpusBlocks) -> bytes:
    vendor = b"vllm"
    tags = b"OpusTags" + struct.pack("<I", len(vendor)) + vendor + struct.pack("<I", 0)
    head = _opus_head(track, _ns_to_48k(blocks.leading_discard_ns))
    pages = [
        _ogg_page(head, granule=0, sequence=0, header_type=0x02),
        _ogg_page(tags, granule=0, sequence=1, header_type=0x00),
    ]
    granule = 0
    packets = blocks.packets
    for index, packet in enumerate(packets):
        granule += _opus_packet_samples(packet)
        last = index == len(packets) - 1
        if last and blocks.trailing_discard_ns:
            # RFC 7845 section 4.4: end trimming via a short final granule.
            granule -= min(
                _ns_to_48k(blocks.trailing_discard_ns), blocks.trailing_block_samples
            )
        pages.append(
            _ogg_page(
                packet,
                granule=granule,
                sequence=index + 2,
                header_type=0x04 if last else 0x00,
            )
        )
    return b"".join(pages)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _decode_with_soundfile(
    encoded: bytes, max_duration_s: float | None
) -> tuple[npt.NDArray, float]:
    import soundfile

    with soundfile.SoundFile(BytesIO(encoded)) as stream:
        native_sr = stream.samplerate
        if max_duration_s is not None and stream.frames / native_sr > max_duration_s:
            raise ValueError(
                f"Audio exceeds maximum allowed duration of {max_duration_s}s. "
                "Set VLLM_MAX_AUDIO_DECODE_DURATION_S to increase this limit."
            )
        audio = stream.read(dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)
    if audio.size == 0:
        raise UnsupportedVideoAudioError("The video's decoded audio track is empty.")
    return np.ascontiguousarray(audio, dtype=np.float32), float(native_sr)


def _is_iso_bmff(data: bytes) -> bool:
    return any(
        box_type in (b"ftyp", b"moov")
        for box_type, _, _ in _iter_mp4_boxes(data, 0, len(data))
    )


class _UnknownContainer(Exception):
    pass


def find_video_audio_track(data: bytes) -> _AudioTrack | None:
    """Locate the first audio track of an MP4/MOV or Matroska/WebM video.

    Raises:
        _UnknownContainer: ``data`` is neither ISO-BMFF nor Matroska.
    """
    segment = _matroska_segment(data)
    if segment is not None:
        return _matroska_audio_track(data, segment)
    if _is_iso_bmff(data):
        return _mp4_audio_track(data)
    raise _UnknownContainer


def load_video_audio_without_pyav(
    data: bytes, *, max_duration_s: float | None = None
) -> tuple[npt.NDArray, float] | None:
    """Decode the first audio track of ``data`` with libsndfile.

    Returns:
        ``(waveform, sample_rate)``, or ``None`` when the video has no audio
        track.

    Raises:
        UnsupportedVideoAudioError: the video has an audio track whose codec
            cannot be decoded without PyAV (e.g. AAC).
    """
    try:
        track = find_video_audio_track(data)
    except _UnknownContainer:
        raise UnsupportedVideoAudioError(
            "use_audio_in_video supports MP4/MOV and Matroska/WebM videos on this "
            "server; this container cannot be inspected for an audio track. Use "
            'an MP4 or WebM video, or set mm_processor_kwargs {"use_audio_in_video": '
            "false} for a visual-only answer."
        ) from None
    except Exception as exc:
        raise UnsupportedVideoAudioError(
            f"Cannot parse the video container to find its audio track: {exc}."
        ) from exc
    if track is None:
        return None
    segment = _matroska_segment(data) if track.container == "matroska" else None
    if segment is not None and track.codec == "opus":
        if track.number is None:
            raise UnsupportedVideoAudioError(
                "The video's Opus audio track has no TrackNumber."
            )
        try:
            blocks = _matroska_track_packets(
                data, segment, track.number, max_duration_s
            )
        except (_DurationExceeded, UnsupportedVideoAudioError):
            raise
        except Exception as exc:
            raise UnsupportedVideoAudioError(
                f"Cannot read the Opus audio track of the video: {exc}"
            ) from exc
        if not blocks.packets:
            return None
        try:
            return _decode_with_soundfile(
                _ogg_opus_from_packets(track, blocks), max_duration_s
            )
        except ValueError:
            raise
        except Exception as exc:
            raise UnsupportedVideoAudioError(
                f"Cannot decode the Opus audio track of the video: {exc}"
            ) from exc
    raise UnsupportedVideoAudioError(
        f"The video's audio track uses codec '{track.codec}' in a {track.container} "
        "container, which this server cannot decode. Use a WebM/Opus video, or set "
        'mm_processor_kwargs {"use_audio_in_video": false} for a visual-only answer '
        "(optionally sending the audio in that same request as input_audio in WAV "
        "or MP3, or as audio_url in WAV, MP3, FLAC or Ogg)."
    )
