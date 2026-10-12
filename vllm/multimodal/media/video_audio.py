# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Extract the audio track of an encoded video without PyAV.

PyAV is the default decoder for ``use_audio_in_video``. Images that ship no
PyAV (and no software AAC/Opus decoder) can still decode Opus through
libsndfile, which reads Ogg Opus. This module finds the audio track of an
ISO-BMFF (MP4/MOV) or Matroska (WebM/MKV) container, remuxes a Matroska Opus
track into an in-memory Ogg Opus stream, and decodes it with ``soundfile``.

``load_video_audio`` returns ``None`` when the container has no audio track
and raises ``UnsupportedVideoAudioError`` when an audio track exists that this
process cannot decode, so callers never silently drop narration.
"""

import struct
from dataclasses import dataclass
from io import BytesIO

import numpy as np
import numpy.typing as npt

from vllm.logger import init_logger

logger = init_logger(__name__)

_MAX_EBML_DEPTH = 8


class UnsupportedVideoAudioError(ValueError):
    """The video has an audio track that cannot be decoded here."""


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


def _iter_mp4_boxes(data: bytes, start: int, end: int):
    offset = start
    while offset + 8 <= end:
        size, box_type = struct.unpack_from(">I4s", data, offset)
        header = 8
        if size == 1:
            if offset + 16 > end:
                return
            size = struct.unpack_from(">Q", data, offset + 8)[0]
            header = 16
        elif size == 0:
            size = end - offset
        if size < header or offset + size > end:
            return
        yield box_type, offset + header, offset + size
        offset += size


def _mp4_audio_track(data: bytes) -> _AudioTrack | None:
    def walk(start: int, end: int, track: dict[str, bytes]):
        for box_type, body, box_end in _iter_mp4_boxes(data, start, end):
            if box_type == b"trak":
                inner: dict[str, bytes] = {}
                walk(body, box_end, inner)
                if inner.get("handler") == b"soun":
                    return inner.get("codec", b"????")
            elif box_type in _MP4_CONTAINER_BOXES:
                found = walk(body, box_end, track)
                if found is not None:
                    return found
            elif box_type == b"hdlr" and body + 12 <= box_end:
                track.setdefault("handler", data[body + 8 : body + 12])
            elif box_type == b"stsd" and body + 16 <= box_end:
                track["codec"] = data[body + 12 : body + 16]
        return None

    codec = walk(0, len(data), {})
    if codec is None:
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
_SEGMENT_CHILDREN = {
    0x114D9B74,  # SeekHead
    0x1549A966,  # Info
    _TRACKS,
    _CLUSTER,
    0x1C53BB6B,  # Cues
    0x1941A469,  # Attachments
    0x1043A770,  # Chapters
    0x1254C367,  # Tags
}


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


def _matroska_audio_track(data: bytes, segment: tuple[int, int]) -> _AudioTrack | None:
    seg_start, seg_end = segment
    for element_id, body, end, _ in _iter_ebml(data, seg_start, seg_end):
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


def _matroska_track_packets(
    data: bytes, segment: tuple[int, int], track_number: int
) -> list[bytes]:
    packets: list[bytes] = []

    def block_packets(body: int, end: int) -> None:
        number, length, _ = _read_vint(data, body, keep_marker=False)
        if number != track_number:
            return
        header = body + length
        flags = data[header + 2]
        packets.extend(_split_laced_frames(data[header + 3 : end], (flags >> 1) & 3))

    seg_start, seg_end = segment
    offset = seg_start
    while offset < seg_end:
        element_id, id_len, _ = _read_vint(data, offset, keep_marker=True)
        size, size_len, unknown = _read_vint(data, offset + id_len, keep_marker=False)
        body = offset + id_len + size_len
        end = seg_end if unknown else min(body + size, seg_end)
        if element_id != _CLUSTER:
            if unknown:
                break
            offset = end
            continue
        cursor = body
        for child_id, cbody, cend, cunknown in _iter_ebml(
            data,
            body,
            end,
            stop_ids=frozenset(_SEGMENT_CHILDREN) if unknown else frozenset(),
        ):
            cursor = cend
            if child_id == _SIMPLE_BLOCK:
                block_packets(cbody, cend)
            elif child_id == _BLOCK_GROUP:
                for gid, gbody, gend, _ in _iter_ebml(data, cbody, cend):
                    if gid == _BLOCK:
                        block_packets(gbody, gend)
            if cunknown:
                break
        offset = cursor if unknown else end
        if unknown and cursor >= seg_end:
            break
    return packets


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


def _opus_head(track: _AudioTrack) -> bytes:
    if track.codec_private and track.codec_private.startswith(b"OpusHead"):
        return track.codec_private
    channels = track.channels or 1
    rate = int(track.sampling_rate or 48000)
    return struct.pack("<8sBBHIhB", b"OpusHead", 1, channels, 312, rate, 0, 0)


def _ogg_opus_from_packets(track: _AudioTrack, packets: list[bytes]) -> bytes:
    vendor = b"vllm"
    tags = b"OpusTags" + struct.pack("<I", len(vendor)) + vendor + struct.pack("<I", 0)
    pages = [
        _ogg_page(_opus_head(track), granule=0, sequence=0, header_type=0x02),
        _ogg_page(tags, granule=0, sequence=1, header_type=0x00),
    ]
    granule = 0
    for index, packet in enumerate(packets):
        granule += _opus_packet_samples(packet)
        last = index == len(packets) - 1
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
        raise ValueError("decoded audio track is empty")
    return np.ascontiguousarray(audio, dtype=np.float32), float(native_sr)


def _is_iso_bmff(data: bytes) -> bool:
    return any(
        box_type in (b"ftyp", b"moov")
        for box_type, _, _ in _iter_mp4_boxes(data, 0, min(len(data), 1 << 20))
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
        logger.warning(
            "Video container is neither MP4/MOV nor Matroska/WebM; its audio "
            "track (if any) cannot be extracted without PyAV. Answering from "
            "the video frames only."
        )
        return None
    except (ValueError, IndexError, struct.error) as exc:
        raise UnsupportedVideoAudioError(
            f"Cannot parse the video container to find its audio track: {exc}"
        ) from exc
    if track is None:
        return None
    if track.container == "matroska" and track.codec == "opus":
        assert track.number is not None
        try:
            packets = _matroska_track_packets(
                data, _matroska_segment(data), track.number
            )
        except (ValueError, IndexError, struct.error) as exc:
            raise UnsupportedVideoAudioError(
                f"Cannot read the Opus audio track of the video: {exc}"
            ) from exc
        if not packets:
            return None
        try:
            return _decode_with_soundfile(
                _ogg_opus_from_packets(track, packets), max_duration_s
            )
        except ValueError:
            raise
        except Exception as exc:
            raise UnsupportedVideoAudioError(
                f"Cannot decode the Opus audio track of the video: {exc}"
            ) from exc
    raise UnsupportedVideoAudioError(
        f"The video's audio track uses codec '{track.codec}' in a {track.container} "
        "container, which this server cannot decode. Send the audio separately "
        "as audio_url/input_audio (WAV, MP3, FLAC or Ogg), use a WebM/Opus video, "
        'or set mm_processor_kwargs {"use_audio_in_video": false} for a '
        "visual-only answer."
    )
