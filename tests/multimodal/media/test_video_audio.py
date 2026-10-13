# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PyAV-free audio-in-video extraction (Nemotron Omni ``use_audio_in_video``).

Fixtures are 1 s, 32x32 clips: a 440 Hz tone muxed as Opus in WebM (seekable
and live/unknown-size), as AAC in MP4, as Vorbis in WebM, as MP3 in Matroska,
and silent MP4/WebM. Tests use no audio-decoder CLI unless they request the
``audio_cli`` fixture (``NV_TEST_OMNI_AUDIO_FFMPEG``).
"""

import os
import subprocess
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


@pytest.fixture(autouse=True)
def _no_audio_cli(monkeypatch):
    monkeypatch.setenv("NV_OMNI_AUDIO_FFMPEG", "")


@pytest.fixture
def audio_cli(monkeypatch):
    cli = os.environ.get("NV_TEST_OMNI_AUDIO_FFMPEG", "")
    if not os.access(cli, os.X_OK) or not os.access("/usr/bin/prlimit", os.X_OK):
        pytest.skip("NV_TEST_OMNI_AUDIO_FFMPEG does not name an FFmpeg CLI")
    monkeypatch.setenv("NV_OMNI_AUDIO_FFMPEG", cli)
    calls = []
    real_popen = subprocess.Popen

    def popen(command, *args, **kwargs):
        calls.append(command)
        return real_popen(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", popen)
    return calls


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
    assert abs(len(audio) / sample_rate - 1.0) < 0.002
    assert abs(_dominant_hz(audio, sample_rate) - 440) < 10


@pytest.mark.parametrize("name", ["silent.mp4", "silent.webm"])
def test_video_without_audio_track_returns_none(name):
    assert load_video_audio_without_pyav(_read(name)) is None


def test_undecodable_audio_track_is_rejected_not_dropped():
    with pytest.raises(UnsupportedVideoAudioError, match="'aac'"):
        load_video_audio_without_pyav(_read("tone_aac.mp4"))


def test_uninspectable_container_is_rejected_not_dropped():
    with pytest.raises(UnsupportedVideoAudioError, match="use_audio_in_video"):
        load_video_audio_without_pyav(b"RIFF\x00\x01\x00\x00AVI LIST")


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


def test_discard_padding_edges_map_to_pre_skip_and_final_granule():
    from vllm.multimodal.media.video_audio import (
        _AudioTrack,
        _ogg_opus_from_packets,
        _OpusBlocks,
    )

    celt_20ms = bytes([(31 << 3) | 0, 0])
    laced_pair = bytes([(31 << 3) | 1, 0])  # two 20 ms frames in one packet
    track = _AudioTrack(container="matroska", codec="opus", channels=1)
    blocks = _OpusBlocks(
        packets=[celt_20ms, laced_pair],
        leading_discard_ns=5_000_000,
        trailing_discard_ns=25_000_000,
        trailing_block_samples=1920,
    )
    ogg = _ogg_opus_from_packets(track, blocks)

    pages = [p for p in ogg.split(b"OggS") if p]  # header fields follow "OggS"
    head = pages[0][23 + pages[0][22] :]
    assert int.from_bytes(head[10:12], "little") == 312 + 240
    final_granule = int.from_bytes(pages[-1][2:10], "little", signed=True)
    assert final_granule == 960 + 1920 - 1200


def test_duration_limit_stops_packet_extraction_early(monkeypatch):
    import vllm.multimodal.media.video_audio as video_audio

    muxed = []
    monkeypatch.setattr(video_audio, "_COLLECTION_SLACK_S", 0.0)
    monkeypatch.setattr(
        video_audio, "_ogg_opus_from_packets", lambda *a: muxed.append(a) or b""
    )
    with pytest.raises(ValueError, match="maximum allowed duration"):
        load_video_audio_without_pyav(_read("tone_opus.webm"), max_duration_s=0.1)
    assert muxed == []


def test_truncated_mp4_metadata_is_rejected_not_dropped():
    data = _read("tone_aac.mp4")
    moov = next(
        body - 8
        for box_type, body, _ in _iter_mp4_boxes(data, 0, len(data))
        if box_type == b"moov"
    )
    with pytest.raises(UnsupportedVideoAudioError):
        load_video_audio_without_pyav(data[: moov + 64])


def test_duration_limit_allows_audio_exactly_at_the_limit():
    pytest.importorskip("soundfile")
    audio, sample_rate = load_video_audio_without_pyav(
        _read("tone_opus.webm"), max_duration_s=1
    )
    assert len(audio) / sample_rate <= 1.0


def test_interior_discard_padding_is_rejected():
    from vllm.multimodal.media.video_audio import _check_block_padding, _OpusBlocks

    _check_block_padding(_OpusBlocks(packets=[]), -5_000_000, first_block=True)
    _check_block_padding(_OpusBlocks(packets=[b"x"]), 0, first_block=False)
    with pytest.raises(UnsupportedVideoAudioError, match="DiscardPadding"):
        _check_block_padding(_OpusBlocks(packets=[b"x"]), -1, first_block=False)
    earlier_trailing = _OpusBlocks(packets=[b"x"], trailing_discard_ns=10)
    with pytest.raises(UnsupportedVideoAudioError, match="DiscardPadding"):
        _check_block_padding(earlier_trailing, 0, first_block=False)


def test_quicktime_data_handler_in_minf_does_not_hide_the_audio_track():
    """QuickTime may place a data handler `hdlr` (`url `) in `minf` first."""

    def box(kind: bytes, body: bytes) -> bytes:
        return (8 + len(body)).to_bytes(4, "big") + kind + body

    def rebuild(data: bytes, start: int, end: int) -> bytes:
        out = b""
        for kind, body, box_end in _iter_mp4_boxes(data, start, end):
            if kind in (b"moov", b"trak", b"mdia"):
                inner = rebuild(data, body, box_end)
                if kind == b"mdia":
                    children = list(_iter_mp4_boxes(inner, 0, len(inner)))
                    minf = [inner[b - 8 : e] for k, b, e in children if k == b"minf"]
                    rest = [inner[b - 8 : e] for k, b, e in children if k != b"minf"]
                    url_handler = box(b"hdlr", bytes(8) + b"url " + bytes(12))
                    minf_body = url_handler + minf[0][8:]
                    inner = box(b"minf", minf_body) + b"".join(rest)
                out += box(kind, inner)
            else:
                out += data[body - 8 : box_end]
        return out

    data = _read("tone_aac.mp4")
    reordered = rebuild(data, 0, len(data))
    with pytest.raises(UnsupportedVideoAudioError, match="'aac'"):
        load_video_audio_without_pyav(reordered)


@pytest.mark.parametrize(
    "tail", [b"", bytes(16), (1 << 20).to_bytes(4, "big") + b"mdat"]
)
def test_silent_mp4_tail_damage_after_moov_stays_visual_only(tail):
    data = _read("silent.mp4")
    boxes = {k: data[b - 8 : e] for k, b, e in _iter_mp4_boxes(data, 0, len(data))}
    faststart = boxes[b"ftyp"] + boxes[b"moov"] + boxes[b"mdat"][:-50] + tail
    assert load_video_audio_without_pyav(faststart) is None


def test_compressed_quicktime_metadata_is_rejected_not_dropped():
    cmov = (16).to_bytes(4, "big") + b"cmov" + bytes(8)
    moov = (8 + len(cmov)).to_bytes(4, "big") + b"moov" + cmov
    ftyp = (16).to_bytes(4, "big") + b"ftypqt  " + bytes(4)
    with pytest.raises(UnsupportedVideoAudioError, match="cmov"):
        load_video_audio_without_pyav(ftyp + moov)


def test_every_rejection_names_the_visual_only_remedy():
    from vllm.multimodal.media.video_audio import _check_block_padding, _OpusBlocks

    with pytest.raises(UnsupportedVideoAudioError) as padding:
        _check_block_padding(_OpusBlocks(packets=[b"x"]), -1, first_block=False)
    with pytest.raises(UnsupportedVideoAudioError) as aac:
        load_video_audio_without_pyav(_read("tone_aac.mp4"))
    with pytest.raises(ValueError) as duration:
        load_video_audio_without_pyav(_read("tone_opus.webm"), max_duration_s=0.1)
    for error in (padding, aac, duration):
        assert str(error.value).count('{"use_audio_in_video": false}') == 1


def test_declared_opus_track_without_packets_is_rejected_not_dropped():
    data = _read("tone_opus.webm")
    seg_start, seg_end = _matroska_segment(data)
    kept, offset = [], seg_start
    for element_id, _, end in _segment_children(data, seg_start, seg_end):
        if element_id != 0x1F43B675:  # drop every Cluster, keep Tracks
            kept.append(data[offset:end])
        offset = end
    with pytest.raises(UnsupportedVideoAudioError, match="no audio data"):
        load_video_audio_without_pyav(data[:seg_start] + b"".join(kept))


def _cli_load(name_or_data, **kwargs):
    data = _read(name_or_data) if isinstance(name_or_data, str) else name_or_data
    return load_video_audio_without_pyav(data, decoder_cli=True, **kwargs)


@pytest.mark.parametrize("name", ["tone_aac.mp4", "tone_vorbis.webm", "tone_mp3.mkv"])
def test_non_opus_tracks_are_decoded_by_the_audio_cli(audio_cli, name):
    result = _cli_load(name, max_duration_s=600)

    assert result is not None
    audio, sample_rate = result
    assert sample_rate == 48000 and audio.dtype == np.float32 and audio.ndim == 1
    assert abs(len(audio) / sample_rate - 1.0) < 0.1
    assert abs(_dominant_hz(audio, sample_rate) - 440) < 10
    assert len(audio_cli) == 1 and "-protocol_whitelist" in audio_cli[0]


def test_audio_cli_requires_the_caller_opt_in(audio_cli):
    """Models without a sound encoder (Super 3.5) keep the pre-CLI behaviour."""
    with pytest.raises(UnsupportedVideoAudioError, match="'aac'"):
        load_video_audio_without_pyav(_read("tone_aac.mp4"))
    assert audio_cli == []


def test_nemotron_processor_allows_the_cli_only_for_sound_config_models(monkeypatch):
    from types import SimpleNamespace

    from vllm.model_executor.models import nano_nemotron_vl
    from vllm.multimodal.parse import MultiModalDataItems, VideoProcessorItems

    seen = []
    monkeypatch.setattr(
        nano_nemotron_vl,
        "_load_audio_from_video_bytes",
        lambda data, idx, decoder_cli=False: seen.append(decoder_cli),
    )
    frames = np.zeros((1, 2, 2, 3), dtype=np.uint8)
    for sound_config, expected in ((None, False), (SimpleNamespace(), True)):
        videos = VideoProcessorItems(
            [frames], metadata=[{"original_video_bytes": b"v"}]
        )
        processor = SimpleNamespace(info=SimpleNamespace(sound_config=sound_config))
        nano_nemotron_vl.NanoNemotronVLMultiModalProcessor._extract_audio_from_videos(
            processor, MultiModalDataItems({"video": videos})
        )
        assert seen.pop() is expected


@pytest.mark.parametrize("name", ["silent.mp4", "silent.webm", "tone_opus.webm"])
def test_audio_cli_is_not_used_for_silent_or_opus_videos(audio_cli, name):
    pytest.importorskip("soundfile")
    result = _cli_load(name, max_duration_s=600)

    assert (result is None) == name.startswith("silent")
    assert audio_cli == []


def test_audio_cli_retries_opus_tracks_libsndfile_cannot_decode(audio_cli, monkeypatch):
    import vllm.multimodal.media.video_audio as video_audio

    def unsupported(*args):
        raise UnsupportedVideoAudioError("interior DiscardPadding")

    monkeypatch.setattr(video_audio, "_decode_matroska_opus", unsupported)
    audio, sample_rate = _cli_load("tone_opus.webm", max_duration_s=600)
    assert abs(_dominant_hz(audio, sample_rate) - 440) < 10
    assert len(audio_cli) == 1


def test_audio_cli_decoder_failures_are_client_errors_without_paths(audio_cli):
    # AudioSpecificConfig with audio object type 0: the AAC decoder cannot open.
    config = bytes.fromhex("0580808005140856e5")
    data = _read("tone_aac.mp4").replace(config, config[:5] + b"\0\0" + config[7:])
    with pytest.raises(UnsupportedVideoAudioError, match="opening decoder") as raised:
        _cli_load(data)
    message = str(raised.value)
    assert "vllm-video-audio-" not in message
    assert '{"use_audio_in_video": false}' in message


def test_audio_cli_rejects_corrupt_packets_instead_of_dropping_narration(audio_cli):
    data = bytearray(_read("tone_aac.mp4"))
    mdat = next(
        b for k, b, e in _iter_mp4_boxes(bytes(data), 0, len(data)) if k == b"mdat"
    )
    middle = mdat + (len(data) - mdat) // 4
    data[middle : middle + 40] = bytes(40)
    with pytest.raises(UnsupportedVideoAudioError, match="decoder"):
        _cli_load(bytes(data))


def test_audio_cli_output_obeys_the_duration_limit(audio_cli):
    with pytest.raises(ValueError, match="maximum allowed duration"):
        _cli_load("tone_aac.mp4", max_duration_s=0.25)
    command = audio_cli[0]
    assert "atrim=end_sample=60000" in command[command.index("-af") + 1]


def test_audio_cli_output_is_bounded_despite_repeated_timestamps(audio_cli):
    data = _read("tone_mp3.mkv")
    seg_start, seg_end = _matroska_segment(data)
    clusters = [
        data[body - 12 : end]  # 4-byte ID + 8-byte size
        for element_id, body, end in _segment_children(data, seg_start, seg_end)
        if element_id == 0x1F43B675
    ]
    size_field = data[seg_start - 8 : seg_start]
    assert size_field[0] == 0x01  # 8-byte size vint
    forged = (
        data[: seg_start - 8]
        + bytes([0x01] + [0xFF] * 7)  # unknown size
        + data[seg_start:seg_end]
        + b"".join(clusters) * 20
    )
    with pytest.raises(ValueError, match="maximum allowed duration"):
        _cli_load(forged, max_duration_s=1)


def test_oversized_videos_are_rejected_before_staging(audio_cli, monkeypatch):
    import vllm.multimodal.media.video_audio as video_audio

    monkeypatch.setattr(video_audio, "_CLI_MAX_INPUT_BYTES", 1024)
    with pytest.raises(UnsupportedVideoAudioError, match="larger than"):
        _cli_load("tone_aac.mp4")
    assert audio_cli == []
