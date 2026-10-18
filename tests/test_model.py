from __future__ import annotations

import json

import numpy as np
import pytest


def test_model_module_is_lazy_about_tensorflow() -> None:
    import edgeai.model as model

    assert model.EMBEDDING_DIM == 192
    assert model.MODEL_INPUT_SHAPE == (1, 298, 40, 1)
    assert model.MODEL_OUTPUT_SHAPE == (1, 192)


def test_smoke_model_converts_inspects_and_invokes_on_desktop(tmp_path) -> None:
    pytest.importorskip("tensorflow")
    from edgeai.model import create_smoke_model

    report = create_smoke_model(tmp_path / "model-smoke")
    assert report["status"] == "UNTRAINED_SMOKE_MODEL"
    assert report["input_status"] == "SYNTHETIC_INPUT"
    assert report["hardware_status"] == "UNVERIFIED_HARDWARE"
    assert report["final_runtime_status"] == "CONTRACT_UNRESOLVED"
    assert report["metrics"] == {"accuracy": None, "eer": None, "far": None}

    int8 = report["interpreter"]["int8"]
    assert int8["input"]["shape"] == [1, 298, 40, 1]
    assert int8["output"]["shape"] == [1, 192]
    assert int8["input"]["dtype"] == "int8"
    assert int8["output"]["dtype"] == "int8"
    assert len(int8["input"]["quantization"]["scales"]) == 1
    assert len(int8["output"]["quantization"]["scales"]) == 1
    assert int8["operations"]
    assert int8["float32_tensor_names"] == []

    assert report["artifacts"]["fp32"]["bytes"] > 0
    assert report["artifacts"]["int8"]["bytes"] > 0
    assert report["desktop_invoke"]["int8"]["output_dtype"] == "int8"
    assert report["desktop_invoke"]["int8"]["unit_embedding_l2"] == pytest.approx(1.0)

    report_path = tmp_path / "model-smoke" / "model_smoke_report.json"
    persisted = json.loads(report_path.read_text(encoding="utf-8"))
    assert persisted["metrics"] == {"accuracy": None, "eer": None, "far": None}


def test_quantization_helpers_use_metadata_scale_and_zero_point() -> None:
    pytest.importorskip("tensorflow")
    from edgeai.model import dequantize_tensor, quantize_tensor

    detail = {
        "dtype": np.dtype(np.int8),
        "quantization_parameters": {
            "scales": np.array([0.25], dtype=np.float32),
            "zero_points": np.array([-3], dtype=np.int64),
            "quantized_dimension": 0,
        },
    }
    quantized = quantize_tensor(np.array([-100.0, -0.5, 0.0, 0.5, 100.0]), detail)
    np.testing.assert_array_equal(quantized, [-128, -5, -3, -1, 127])
    np.testing.assert_allclose(dequantize_tensor(quantized, detail), [-31.25, -0.5, 0.0, 0.5, 32.5])
