"""Static and envelope guards for the additive ESP32 raw-PCM operation."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "firmware" / "esp32_s3_smoke"
SOURCE = PROJECT / "main" / "main.cc"
CAPACITY = 2048


def test_pcm_operation_and_exact_wire_contract_are_present() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    cmake = (PROJECT / "main" / "CMakeLists.txt").read_text(encoding="utf-8")
    frontend = (PROJECT / "main" / "pcm_frontend.h").read_text(encoding="utf-8")

    assert 'SRCS "main.cc" "pcm_frontend.cc" "template_store.cc"' in cmake
    assert "kOperationRecognizePcm = 0x20U" in source
    assert "kOperationVersion = 1U" in source
    assert "kRecognizePcmPayloadBytes" in source
    assert "edgeai_pcm_frontend::kPcmBytes" in source
    assert "kPcmSamples = 48000U" in frontend
    assert "kPcmBytes = kPcmSamples * sizeof(std::int16_t)" in frontend
    assert "RECOGNIZE_PCM" in source
    assert "pcm_sample_rate_hz" in source
    assert "pcm_samples" in source
    assert "LOGMEL_16K_400_160_512_40_CMN_V1" in source
    assert "frontend_us" in source
    assert "feature_crc32" in source
    assert "Crc32Ieee(input, edgeai_pcm_frontend::kPcmBytes)" in source
    assert "input->params.scale" in source
    assert "input->params.zero_point" in source


def test_raw_request_is_heap_bounded_and_workspace_is_freed_before_reply() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    assert "g_request_payload[kLegacyMaxRequestPayloadBytes]" in source
    assert "heap_caps_malloc(" in source
    assert "frame->header.payload_length, kTelemetryHeapCaps" in source
    assert "edgeai_pcm_frontend::kWorkspaceBytes" in source
    workspace_alloc = source.index("edgeai_pcm_frontend::kWorkspaceBytes")
    workspace_free = source.index("heap_caps_free(workspace_memory)")
    response = source.index("if (!SendPcmRecognitionResponse(")
    assert workspace_alloc < workspace_free < response
    assert "std::vector" not in source
    assert "float query[edgeai_template::kEmbeddingDim]" in source


def test_pcm_response_maximum_envelope_fits_actual_buffer() -> None:
    telemetry = {
        "version": 1,
        "uptime_us": 18446744073709551615,
        "internal_heap_total_bytes": 4294967295,
        "internal_heap_free_bytes": 4294967295,
        "internal_heap_min_free_bytes": 4294967295,
        "internal_heap_largest_free_block_bytes": 4294967295,
        "query_heap_free_before_bytes": 4294967295,
        "query_heap_free_after_bytes": 4294967295,
        "query_heap_delta_bytes": -4294967295,
        "recognition_us": 9223372036854775807,
    }
    envelope = {
        "protocol_version": 1,
        "request_id": 4294967295,
        "status": "OK",
        "model_sha256": "a" * 64,
        "runtime_mode": "ESP_NN_ANSI_C_REFERENCE",
        "input_bytes": 11920,
        "output_bytes": 192,
        "input_scale": 0.12695927917957306,
        "input_zero_point": -12,
        "output_scale": 0.01833495683968067,
        "output_zero_point": 6,
        "arena_capacity_bytes": 131072,
        "arena_used_bytes": 131072,
        "operation": "RECOGNIZE_PCM",
        "input_crc32": 4294967295,
        "output_hex": "ff" * 192,
        "invoke_us": 9223372036854775807,
        "pcm_sample_rate_hz": 16000,
        "pcm_samples": 48000,
        "frontend_id": "LOGMEL_16K_400_160_512_40_CMN_V1",
        "frontend_us": 9223372036854775807,
        "feature_crc32": 4294967295,
        "telemetry": telemetry,
        "top1": "Quang",
        "scores": [
            {"label": label, "cosine": -1.0}
            for label in ("Quang", "Tien", "An")
        ],
    }
    text = "EDGEAI_RESPONSE " + json.dumps(envelope, separators=(",", ":"))
    text = text.replace("-1.0}", "-1.000000000}") + "\n"
    assert len(text) < CAPACITY


def test_legacy_recognition_still_uses_legacy_input_size_and_timer_path() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    assert "Crc32Ieee(input, kInputBytes)" in source
    assert 'AppendOperation(&response, operation)' in source
    assert '"RECOGNIZE", false' in source
    assert "nvs_open(kNvsNamespace, NVS_READONLY" in source
    assert "nvs_flash_erase" not in source


def test_pcm_timer_covers_frontend_invoke_and_ranking() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    timer = source.index("const int64_t recognition_begin_us = esp_timer_get_time();")
    allocation = source.index("heap_caps_aligned_alloc(", timer)
    release = source.index("heap_caps_free(workspace_memory)", allocation)
    feature_crc = source.index("const uint32_t feature_crc32", release)
    invoke = source.index("if (!InvokeRequest(", feature_crc)
    response = source.index("if (!SendPcmRecognitionResponse(", invoke)
    assert timer < allocation < release < feature_crc < invoke < response
    assert "feature_crc_us = esp_timer_get_time() - feature_crc_begin_us" in source
    assert "recognition_begin_us + feature_crc_us, frontend_us, feature_crc32" in source
    assert "recognition_us = esp_timer_get_time() - recognition_begin_us" in source


def test_pcm_invoke_skips_identical_tensor_copy() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    start = source.index("bool InvokeRequest(")
    end = source.index("bool SendInferenceResponse(", start)
    invoke = source[start:end]
    assert "if (reinterpret_cast<const uint8_t*>(input->data.int8) != input_bytes) {" in invoke
    assert invoke.index("!= input_bytes") < invoke.index("std::memcpy(")
    assert invoke.index("std::memcpy(") < invoke.index("g_interpreter->Invoke()")
