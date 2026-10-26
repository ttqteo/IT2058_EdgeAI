"""Synthetic telemetry contract guards; these tests do not exercise a board."""
import importlib.util
import json
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    "demo_voice_usb", Path(__file__).resolve().parents[1] / "scripts/demo_voice_usb.py")
DEMO = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(DEMO)


def telemetry(**overrides):
    value = {
        "version": 1,
        "uptime_us": 123456,
        "internal_heap_total_bytes": 200000,
        "internal_heap_free_bytes": 100000,
        "internal_heap_min_free_bytes": 90000,
        "internal_heap_largest_free_block_bytes": 80000,
    }
    value.update(overrides)
    return value


def reply(operation="STATUS", value=None):
    result = {
        "protocol_version": 1,
        "model_sha256": DEMO.MODEL_SHA,
        "runtime_mode": "ESP_NN_ANSI_C_REFERENCE",
        "input_bytes": DEMO.INPUT_BYTES,
        "output_bytes": DEMO.OUTPUT_BYTES,
        "input_scale": DEMO.INPUT_SCALE,
        "input_zero_point": DEMO.INPUT_ZERO,
        "output_scale": DEMO.OUTPUT_SCALE,
        "output_zero_point": DEMO.OUTPUT_ZERO,
        "arena_capacity_bytes": 131072,
        "arena_used_bytes": 38140,
        "operation": operation,
    }
    if value is not None:
        result["telemetry"] = value
    if operation == "RECOGNIZE":
        result["invoke_us"] = 53500
    return result


def test_missing_telemetry_is_backward_compatible():
    assert DEMO.validate_telemetry(reply()) is None
    DEMO.check_metadata(reply())


def test_status_telemetry_is_normalized_without_query_fields():
    value = telemetry()
    assert DEMO.validate_telemetry(reply(value=value)) == value


def test_recognize_telemetry_accepts_signed_delta_and_query_fields():
    value = telemetry(
        query_heap_free_before_bytes=95000,
        query_heap_free_after_bytes=100000,
        query_heap_delta_bytes=-5000,
        recognition_us=60000,
    )
    assert DEMO.validate_telemetry(reply("RECOGNIZE", value)) == value
    DEMO.check_metadata(reply("RECOGNIZE", value))


@pytest.mark.parametrize("key", [
    "version", "uptime_us", "internal_heap_total_bytes",
    "internal_heap_free_bytes", "internal_heap_min_free_bytes",
    "internal_heap_largest_free_block_bytes",
])
def test_rejects_boolean_base_fields(key):
    value = telemetry(**{key: True})
    with pytest.raises(ValueError):
        DEMO.validate_telemetry(reply(value=value))


def test_rejects_unknown_version_and_fields():
    with pytest.raises(ValueError, match="version"):
        DEMO.validate_telemetry(reply(value=telemetry(version=2)))
    with pytest.raises(ValueError, match="Malformed telemetry fields"):
        DEMO.validate_telemetry(reply(value=telemetry(extra_field=1)))


@pytest.mark.parametrize("overrides", [
    {"internal_heap_free_bytes": 200001},
    {"internal_heap_min_free_bytes": 100001},
    {"internal_heap_largest_free_block_bytes": 100001},
])
def test_rejects_impossible_capacity_relationships(overrides):
    with pytest.raises(ValueError, match="heap"):
        DEMO.validate_telemetry(reply(value=telemetry(**overrides)))


@pytest.mark.parametrize("key", [
    "query_heap_free_before_bytes", "query_heap_free_after_bytes",
    "query_heap_delta_bytes", "recognition_us",
])
def test_rejects_partial_query_metrics(key):
    value = telemetry(**{key: 1})
    with pytest.raises(ValueError, match="Malformed telemetry fields"):
        DEMO.validate_telemetry(reply("RECOGNIZE", value))


def test_rejects_query_delta_and_capacity_drift():
    value = telemetry(
        internal_heap_free_bytes=95000,
        query_heap_free_before_bytes=100000,
        query_heap_free_after_bytes=95000,
        query_heap_delta_bytes=-1,
        recognition_us=60000,
    )
    with pytest.raises(ValueError, match="delta"):
        DEMO.validate_telemetry(reply("RECOGNIZE", value))

    value["query_heap_delta_bytes"] = 5000
    value["query_heap_free_before_bytes"] = 200001
    with pytest.raises(ValueError, match="capacity"):
        DEMO.validate_telemetry(reply("RECOGNIZE", value))


def test_rejects_query_after_disagreeing_with_same_heap_snapshot():
    value = telemetry(
        query_heap_free_before_bytes=100000,
        query_heap_free_after_bytes=95000,
        query_heap_delta_bytes=5000,
        recognition_us=60000,
    )
    with pytest.raises(ValueError, match="snapshot"):
        DEMO.validate_telemetry(reply("RECOGNIZE", value))


@pytest.mark.parametrize("recognition_us", [0, 53500 - 1, True])
def test_rejects_impossible_query_timings(recognition_us):
    value = telemetry(
        internal_heap_free_bytes=95000,
        query_heap_free_before_bytes=100000,
        query_heap_free_after_bytes=95000,
        query_heap_delta_bytes=5000,
        recognition_us=recognition_us,
    )
    with pytest.raises(ValueError, match="timing|integer"):
        DEMO.validate_telemetry(reply("RECOGNIZE", value))


def test_rejects_malformed_telemetry_object():
    with pytest.raises(ValueError):
        DEMO.validate_telemetry({"telemetry": None})


def test_recognition_timer_excludes_payload_crc():
    source = (Path(__file__).resolve().parents[1] /
              "firmware/esp32_s3_smoke/main/main.cc").read_text()
    crc = source.index("const uint32_t input_crc32 = Crc32Ieee(input, kInputBytes);")
    timer = source.index("recognition_begin_us = esp_timer_get_time();", crc)
    invoke = source.index("if (!InvokeRequest(input, &invoke_us))", timer)
    assert crc < timer < invoke


@pytest.mark.parametrize("pcm_five", [False, True])
def test_synthetic_max_recognize_envelope_fits_firmware_response_buffer(pcm_five):
    value = telemetry(
        uptime_us=18446744073709551615,
        internal_heap_total_bytes=4294967295,
        internal_heap_free_bytes=4294967295,
        internal_heap_min_free_bytes=4294967295,
        internal_heap_largest_free_block_bytes=4294967295,
        query_heap_free_before_bytes=4294967295,
        query_heap_free_after_bytes=4294967295,
        query_heap_delta_bytes=-4294967295,
        recognition_us=9223372036854775807,
    )
    envelope = {
        "protocol_version": 1, "request_id": 4294967295, "status": "OK",
        "model_sha256": "a" * 64, "runtime_mode": "ESP_NN_ANSI_C_REFERENCE",
        "input_bytes": 11920, "output_bytes": 192,
        "input_scale": "0.12695927917957306", "input_zero_point": -12,
        "output_scale": "0.01833495683968067", "output_zero_point": 6,
        "arena_capacity_bytes": 131072, "arena_used_bytes": 131072,
        "operation": "RECOGNIZE", "input_crc32": 4294967295,
        "output_hex": "ff" * 192, "invoke_us": 9223372036854775807,
        "telemetry": value, "top1": "Quang",
        "scores": [{"label": label, "cosine": -1.0}
                   for label in ("Quang", "Tien", "An")],
    }
    if pcm_five:
        envelope.update(operation="RECOGNIZE_PCM", top1="Speaker",
            pcm_sample_rate_hz=16000, pcm_samples=48000,
            frontend_id=DEMO.FRONTEND_ID, frontend_us=9223372036854775807,
            feature_crc32=4294967295,
            scores=[{"label": label, "cosine": -1.0} for label in
                    ("Quang", "Tien", "An", "Speaker", "Person2")])
    text = "EDGEAI_RESPONSE " + json.dumps(envelope, separators=(",", ":"))
    text = text.replace('"input_scale":"0.12695927917957306"',
                        '"input_scale":0.12695927917957306')
    text = text.replace('"output_scale":"0.01833495683968067"',
                        '"output_scale":0.01833495683968067')
    text = text.replace("-1.0}", "-1.000000000}") + "\n"
    if not pcm_five:
        assert len(text) == 1419
        assert len(text) < 1536
    else:
        assert len(text) + 1 < 2048  # Include the C-string terminator.
