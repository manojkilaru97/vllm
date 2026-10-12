# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PyAV-free audio-in-video extraction (Nemotron Omni ``use_audio_in_video``).

Fixtures are 1 s, 32x32 clips: a 440 Hz tone muxed as Opus in WebM (seekable
and live/unknown-size), as AAC in MP4, and silent MP4/WebM.
"""

from pathlib import Path

import numpy as np
import pytest

from vllm.multimodal.media.video_audio import (
    UnsupportedVideoAudioError,
    _iter_mp4_boxes,
    _matroska_segment,
    _ogg_crc,
    _opus_packet_samples,
    _segment_children,
    find_video_audio_track,
    load_video_audio_without_pyav,
)

pytestmark = pytest.mark.cpu_test

ASSETS = Path(__file__).parent.parent / "assets" / "audio_in_video"


def _read(name: str) -> bytes:
    return (ASSETS / name).read_bytes()


def _dominant_hz(audio: np.ndarray, sample_rate: float) -> float:
    spectrum = np.abs(np.fft.rfft(audio * np.hanning(len(audio))))
    return float(np.fft.rfftfreq(len(audio), 1 / sample_rate)[np.argmax(spectrum)])


@pytest.mark.parametrize("name", ["tone_opus.webm", "tone_opus_live.webm"])
def test_webm_opus_track_is_decoded(name):
    pytest.importorskip("soundfile")
    result = load_video_audio_without_pyav(_read(name), max_duration_s=600)

    assert result is not None
    audio, sample_rate = result
    assert audio.dtype == np.float32 and audio.ndim == 1
    # DiscardPadding end trimming (RFC 7845 section 4.4) keeps the exact length.
    assert abs(len(audio) / sample_rate - 1.0) < 0.005
    assert abs(_dominant_hz(audio, sample_rate) - 440) < 10


@pytest.mark.parametrize("name", ["silent.mp4", "silent.webm"])
def test_video_without_audio_track_returns_none(name):
    assert load_video_audio_without_pyav(_read(name)) is None


def test_undecodable_audio_track_is_rejected_not_dropped():
    with pytest.raises(UnsupportedVideoAudioError, match="'aac'"):
        load_video_audio_without_pyav(_read("tone_aac.mp4"))


def test_unknown_container_is_treated_as_visual_only():
    assert load_video_audio_without_pyav(b"\x00\x01not-a-video-container") is None


@pytest.mark.parametrize(
    "name", ["tone_opus.webm", "tone_opus_live.webm", "tone_aac.mp4"]
)
def test_truncated_containers_fail_with_value_errors_only(name):
    """Corrupt uploads must surface as client errors (ValueError -> HTTP 400)."""
    pytest.importorskip("soundfile")
    data = _read(name)
    for cut in range(1, len(data), max(1, len(data) // 97)):
        try:
            result = load_video_audio_without_pyav(data[:cut])
        except ValueError:
            continue
        assert result is None or result[0].ndim == 1


def test_duration_limit_is_enforced():
    pytest.importorskip("soundfile")
    with pytest.raises(ValueError, match="maximum allowed duration"):
        load_video_audio_without_pyav(_read("tone_opus.webm"), max_duration_s=0.1)


def test_opus_packet_durations_follow_the_toc_byte():
    assert _opus_packet_samples(bytes([(31 << 3) | 0])) == 960  # CELT FB 20 ms
    assert _opus_packet_samples(bytes([(1 << 3) | 1])) == 2 * 960  # SILK NB 20 ms x2
    assert _opus_packet_samples(bytes([(12 << 3) | 3, 3])) == 3 * 480  # hybrid 10 ms
    assert _opus_packet_samples(b"") == 0


def test_ogg_crc_matches_bitwise_reference():
    def reference(data: bytes) -> int:
        crc = 0
        for byte in data:
            crc ^= byte << 24
            for _ in range(8):
                crc = ((crc << 1) ^ 0x04C11DB7) if crc & 0x80000000 else crc << 1
                crc &= 0xFFFFFFFF
        return crc

    for data in (b"", b"OggS", b"123456789", bytes(range(256))):
        assert _ogg_crc(data) == reference(data)


def test_corrupted_ogg_page_is_rejected(monkeypatch):
    """libogg verifies page CRCs, so the decode test above proves the muxer."""
    pytest.importorskip("soundfile")
    import vllm.multimodal.media.video_audio as video_audio

    monkeypatch.setattr(video_audio, "_ogg_crc", lambda page: 0)
    with pytest.raises(UnsupportedVideoAudioError):
        load_video_audio_without_pyav(_read("tone_opus.webm"))


def test_nemotron_extraction_falls_back_to_libsndfile_without_pyav(monkeypatch):
    pytest.importorskip("soundfile")
    from vllm.model_executor.models import nano_nemotron_vl

    monkeypatch.setattr(nano_nemotron_vl, "_pyav_available", lambda: False)
    load = nano_nemotron_vl._load_audio_from_video_bytes

    assert load(_read("tone_opus.webm"), 0) is not None
    assert load(_read("silent.mp4"), 0) is None
    with pytest.raises(UnsupportedVideoAudioError):
        load(_read("tone_aac.mp4"), 0)


def test_nemotron_extraction_keeps_the_pyav_path_when_installed(monkeypatch):
    from vllm.model_executor.models import nano_nemotron_vl

    decoded = (np.zeros(4, dtype=np.float32), 16000.0)
    monkeypatch.setattr(nano_nemotron_vl, "_pyav_available", lambda: True)
    monkeypatch.setattr(nano_nemotron_vl, "load_audio_pyav", lambda stream: decoded)
    assert nano_nemotron_vl._load_audio_from_video_bytes(b"video", 0) is decoded

    def no_audio(stream):
        raise ValueError("No audio stream found.")

    monkeypatch.setattr(nano_nemotron_vl, "load_audio_pyav", no_audio)
    assert nano_nemotron_vl._load_audio_from_video_bytes(b"video", 0) is None


@pytest.mark.parametrize("pyav", [True, False])
def test_nemotron_data_parser_only_changes_for_audio_models(monkeypatch, pyav):
    from types import SimpleNamespace

    from vllm.model_executor.models import nano_nemotron_vl

    monkeypatch.setattr(nano_nemotron_vl, "_pyav_available", lambda: pyav)
    get_parser = nano_nemotron_vl.NanoNemotronVLProcessingInfo.get_data_parser

    def info(sound_config):
        return SimpleNamespace(
            sound_config=sound_config, _get_expected_hidden_size=lambda: 64
        )

    # Super 3.5 (no sound encoder): identical parser regardless of PyAV.
    vision_only = get_parser(info(None))
    assert vision_only.audio_resampler.target_sr is None
    assert vision_only.audio_resampler.method == "pyav"
    assert vision_only.target_channels is None

    omni = get_parser(info(SimpleNamespace(sampling_rate=16000)))
    assert omni.audio_resampler.target_sr == 16000
    assert omni.audio_resampler.method == ("pyav" if pyav else "scipy")


def test_tracks_after_an_unknown_size_cluster_are_found():
    """Matroska allows Tracks after the first Cluster (referenced by SeekHead)."""
    pytest.importorskip("soundfile")
    data = _read("tone_opus_live.webm")
    seg_start, seg_end = _matroska_segment(data)
    children, offset = [], seg_start
    for element_id, _, end in _segment_children(data, seg_start, seg_end):
        children.append((element_id, data[offset:end]))
        offset = end
    ids = [element_id for element_id, _ in children]
    tracks = ids.index(0x1654AE6B)
    first_cluster = ids.index(0x1F43B675)
    assert tracks < first_cluster
    reordered = children[:tracks] + children[tracks + 1 : first_cluster + 1]
    reordered += [children[tracks]] + children[first_cluster + 1 :]
    moved = data[:seg_start] + b"".join(body for _, body in reordered)

    assert find_video_audio_track(moved).codec == "opus"
    assert load_video_audio_without_pyav(moved) is not None


def test_mov_without_ftyp_and_large_leading_mdat_is_still_probed():
    data = _read("tone_aac.mp4")
    boxes = [
        data[body - 8 : end]
        for box_type, body, end in _iter_mp4_boxes(data, 0, len(data))
        if box_type != b"ftyp"
    ]
    filler = (8 + (2 << 20)).to_bytes(4, "big") + b"mdat" + bytes(2 << 20)
    with pytest.raises(UnsupportedVideoAudioError, match="'aac'"):
        load_video_audio_without_pyav(filler + b"".join(boxes))


def test_deeply_nested_mp4_boxes_are_a_client_error():
    nested = b""
    for _ in range(2000):
        nested = (8 + len(nested)).to_bytes(4, "big") + b"moov" + nested
    ftyp = (16).to_bytes(4, "big") + b"ftypisom" + bytes(4)
    with pytest.raises(UnsupportedVideoAudioError):
        load_video_audio_without_pyav(ftyp + nested)


def test_opus_track_without_track_number_is_a_client_error():
    data = _read("tone_opus.webm")
    track_number = bytes([0xD7, 0x81, 0x02])  # TrackNumber 2 (the Opus track)
    assert data.count(track_number) == 1
    patched = data.replace(track_number, bytes([0xEC, 0x81, 0x00]))  # EBML Void
    with pytest.raises(UnsupportedVideoAudioError, match="TrackNumber"):
        load_video_audio_without_pyav(patched)
