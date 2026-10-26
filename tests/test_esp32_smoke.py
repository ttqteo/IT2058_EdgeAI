from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts/prepare_esp32_smoke.py"
SPEC = importlib.util.spec_from_file_location("prepare_esp32_smoke", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
PREPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE)


def test_pinned_model_is_exact_real_artifact() -> None:
    model = ROOT / "artifacts/full100/20261015/baseline_kd/encoder_int8.tflite"
    if not model.is_file():
        model = ROOT / "models/full100-20261015/baseline_kd/encoder_int8.tflite"
    assert model.is_file()
    assert model.stat().st_size == PREPARE.MODEL_BYTES
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    assert digest == PREPARE.MODEL_SHA256


def test_fixture_contract_is_real_train_only() -> None:
    assert PREPARE.REAL_TRAIN_FIXTURES == (
        (0, "103-1240-0000@0000000000"),
        (33_859, "2691-156745-0028@0000000000"),
        (67_718, "4397-15668-0014@0000144000"),
        (101_577, "6836-61803-0036@0000000000"),
        (135_437, "911-130578-0020@0000096000"),
    )
    assert "real-train" in PREPARE.__doc__.lower()


def test_preparation_refuses_existing_output(tmp_path: Path) -> None:
    existing = tmp_path / "generated"
    existing.mkdir()
    with pytest.raises(PREPARE.PreparationError, match="refusing to overwrite"):
        PREPARE._prepare_output_dir(existing)


def test_firmware_contract_retains_smoke_gate_before_usb_service() -> None:
    project = ROOT / "firmware/esp32_s3_smoke"
    cmake = (project / "main/CMakeLists.txt").read_text(encoding="utf-8")
    source = (project / "main/main.cc").read_text(encoding="utf-8")
    component = (project / "main/idf_component.yml").read_text(encoding="utf-8")
    assert "EMBED_FILES" in cmake
    assert "encoder_int8.tflite" in cmake
    assert "real_train_004_expected.int8" in cmake
    assert "==1.4.0" in component
    assert "BOARD_ENCODER_SMOKE_ONLY" in source
    assert "131072U" in source
    assert "DRAM_ATTR" in source
    assert "model_sha256_expected" in source
    assert "BUILTIN_REF_NO_DELEGATES" in source
    assert "internal=DRAM" in source
    assert "esp_timer_get_time" in source
    assert "raw192bytes" in source
    assert "AddConv2D" in source
    assert "AddAveragePool2D" in source
    assert "AddReshape" in source
    assert "AddFullyConnected" in source
    # Persistence is intentional, bounded to one namespace/blob, and has no
    # erase/reformat recovery path.
    assert "kFrameTimeoutUs" in source
    assert "kOperationProvision" in source
    assert "kOperationRecognize" in source
    assert "kProvisionPayloadBytes" in source
    assert "operation_version=1" in source
    assert "Crc32Ieee(g_request_payload, kInputBytes)" in source
    assert "invalid_crc" in source
    assert "nvs_flash_init()" in source
    assert "nvs_open(kNvsNamespace, NVS_READONLY" in source
    assert "nvs_set_blob" in source
    assert "nvs_commit" in source
    assert "DequantizeAndNormalize" in source
    assert "CosineSimilarity" in source
    assert "already_provisioned_no_overwrite" in source
    assert "PERSISTENCE_UNCERTAIN" in source
    assert "record_fingerprint" in source
    assert "nvs_flash_erase" not in source
    assert "nvs_erase_all" not in source
    assert "if (!smoke_ok)" in source
    assert source.rindex("if (!smoke_ok)") < source.rindex("StartUsbService()")
    assert "EDGEAI_RESPONSE" in source
    assert "usb_serial_jtag_read_bytes" in source
    for forbidden in ("i2s", "esp_wifi", "gpio_set_level"):
        assert forbidden not in source


def test_main_task_stack_budget_covers_persistence_working_set() -> None:
    project = ROOT / "firmware/esp32_s3_smoke"
    defaults = (project / "sdkconfig.defaults").read_text(encoding="utf-8")
    active = (project / "sdkconfig").read_text(encoding="utf-8")
    assert "CONFIG_ESP_MAIN_TASK_STACK_SIZE=16384" in defaults
    assert "CONFIG_ESP_MAIN_TASK_STACK_SIZE=16384" in active
    # Boot caller plus ParseRecord validation candidate; each TemplateSet has
    # five float32 vectors, five eight-byte labels, count and alignment padding.
    required = 2 * (5 * 192 * 4 + 5 * 8 + 4) + 8192
    assert 16384 >= required
    source = (project / "main/main.cc").read_text(encoding="utf-8")
    assert "2U * sizeof(edgeai_template::TemplateSet)" in source
    assert "TemplateSet v3_candidate" not in source
    assert "kPersistenceStackSafetyMarginBytes = 8192U" in source
    assert "main task stack is too small for NVS template persistence" in source
