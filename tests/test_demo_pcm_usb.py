"""Synthetic client guards, not live microphone or board acceptance."""
import importlib.util
from pathlib import Path
import struct
import wave
import zlib

import pytest

SPEC = importlib.util.spec_from_file_location(
    "pcm_usb", Path(__file__).resolve().parents[1] / "scripts/demo_voice_usb.py")
DEMO = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DEMO)


def test_pcm_frame_is_additive_and_crc_binds_all_samples():
    pcm = struct.pack("<h", -32768) * DEMO.PCM_SAMPLES
    body = DEMO.operation_payload(DEMO.OP_RECOGNIZE_PCM, pcm)
    packet = DEMO.frame(19, body)
    assert len(body) == 96002
    assert packet[:20] == DEMO.HEADER.pack(b"EDGEAI01", 19, 96002, zlib.crc32(body))
    assert packet[20:22] == b"\x20\x01"
    assert packet[22:] == pcm
    for bad in (b"", pcm[:-2], pcm + b"\0\0"):
        with pytest.raises(ValueError):
            DEMO.operation_payload(DEMO.OP_RECOGNIZE_PCM, bad)


@pytest.mark.parametrize("rate,channels,count", [(16000, 1, 48000),
    (48000, 1, 48000), (16000, 2, 48000), (16000, 1, 47999), (16000, 1, 48001)])
def test_pcm_container_read_never_converts_pads_or_crops(tmp_path, rate, channels, count):
    path = tmp_path / "capture.wav"
    raw = struct.pack("<h", -32768) * count * channels
    with wave.open(str(path), "wb") as handle:
        handle.setparams((channels, 2, rate, count, "NONE", "NONE"))
        handle.writeframes(raw)
    if (rate, channels, count) == (16000, 1, 48000):
        assert DEMO.read_pcm_capture(path) == raw
    else:
        with pytest.raises(ValueError, match="no host conversion"):
            DEMO.read_pcm_capture(path)


def pcm_reply():
    return dict(protocol_version=1, request_id=4, status="OK", operation="RECOGNIZE_PCM",
        model_sha256=DEMO.MODEL_SHA, runtime_mode="ESP_NN_ANSI_C_REFERENCE",
        input_bytes=11920, output_bytes=192, input_scale=DEMO.INPUT_SCALE,
        input_zero_point=-12, output_scale=DEMO.OUTPUT_SCALE, output_zero_point=6,
        arena_capacity_bytes=131072, arena_used_bytes=38140,
        input_crc32=zlib.crc32(bytes(96000)), output_hex=bytes(192).hex(),
        invoke_us=53500, pcm_sample_rate_hz=16000, pcm_samples=48000,
        frontend_id=DEMO.FRONTEND_ID, frontend_us=70000, feature_crc32=123,
        top1="An", scores=[{"label": "An", "cosine": .8},
        {"label": "Quang", "cosine": .5}, {"label": "Tien", "cosine": .4}],
        telemetry=dict(version=1, uptime_us=1000000, internal_heap_total_bytes=150000,
        internal_heap_free_bytes=80000, internal_heap_min_free_bytes=20000,
        internal_heap_largest_free_block_bytes=70000, query_heap_free_before_bytes=80000,
        query_heap_free_after_bytes=80000, query_heap_delta_bytes=0, recognition_us=130000))


def test_pcm_response_requires_distinct_board_frontend_evidence():
    record = pcm_reply()
    assert DEMO.recognition_output(record, 4, bytes(96000), "RECOGNIZE_PCM")[1][0]["label"] == "An"
    for key, value in (("operation", "RECOGNIZE"), ("pcm_samples", 47999),
                       ("pcm_sample_rate_hz", 48000), ("frontend_us", 0),
                       ("frontend_id", "HOST"), ("feature_crc32", True),
                       ("input_crc32", 0)):
        with pytest.raises(ValueError):
            DEMO.recognition_output(dict(record, **{key: value}), 4, bytes(96000), "RECOGNIZE_PCM")
    record["telemetry"]["recognition_us"] = 100000
    with pytest.raises(ValueError, match="frontend and encoder"):
        DEMO.recognition_output(record, 4, bytes(96000), "RECOGNIZE_PCM")
