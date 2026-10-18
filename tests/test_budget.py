from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import edgeai.budget as budget_module
from edgeai.budget import BudgetProfile, STATIC_FAIL, STATIC_PASS, check_model


ROOT = Path(__file__).resolve().parents[1]


def _require_tensorflow() -> None:
    pytest.importorskip("tensorflow")


@pytest.fixture(scope="session")
def smoke_artifacts(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Build fresh smoke artifacts outside the checkout for every test run."""

    _require_tensorflow()
    from edgeai.model import create_smoke_model

    output_dir = tmp_path_factory.mktemp("budget-smoke-artifacts")
    create_smoke_model(output_dir)
    return {
        "int8": output_dir / "encoder_int8.tflite",
        "fp32": output_dir / "encoder_fp32.tflite",
    }


def test_real_int8_artifact_passes_independent_mac_oracle(
    smoke_artifacts: dict[str, Path],
) -> None:
    int8_model = smoke_artifacts["int8"]
    report = check_model(int8_model)

    assert report["status"] == STATIC_PASS
    assert report["runtime_status"] == "UNVERIFIED_HARDWARE"
    assert report["contract_status"] == "CONTRACT_UNRESOLVED"
    expected_sha256 = hashlib.sha256(int8_model.read_bytes()).hexdigest()
    assert report["model"]["sha256"] == expected_sha256
    assert report["macs"]["total"] == 360_480
    assert [(item["operator"], item["macs"]) for item in report["macs"]["breakdown"]] == [
        ("CONV_2D", 149 * 20 * 8 * 3 * 3 * 1),
        ("FULLY_CONNECTED", (19 * 5 * 8) * 192),
    ]
    assert report["memory_plan"]["planned_buffers"] == {
        "pcm16_psram_bytes": 96_000,
        "float_frontend_bytes": 47_680,
        "int8_input_bytes": 11_920,
        "float_output_bytes": 768,
    }
    assert report["memory_plan"]["status"] == "UNVERIFIED_HARDWARE"


def test_fp32_artifact_fails_integer_and_size_gates(
    smoke_artifacts: dict[str, Path],
) -> None:
    report = check_model(smoke_artifacts["fp32"])

    assert report["status"] == STATIC_FAIL
    assert report["static_budget_pass"] is False
    assert report["gates"]["model_bytes"]["status"] == "FAIL"
    assert any(error["check"] == "io_dtype" for error in report["errors"])
    # Missing/invalid per-tensor INT8 metadata is independently rejected.
    assert any(error["check"] == "io_quantization" for error in report["errors"])
    assert report["gates"]["macs"]["status"] == "UNRESOLVED"


def test_missing_bad_and_oversized_artifacts_fail_closed(
    tmp_path: Path, smoke_artifacts: dict[str, Path]
) -> None:
    missing = check_model(tmp_path / "missing.tflite")
    assert missing["status"] == STATIC_FAIL
    assert missing["static_budget_pass"] is False
    assert any(error["check"] == "model_read" for error in missing["errors"])

    bad_path = tmp_path / "bad.tflite"
    bad_path.write_bytes(b"not a TFLite FlatBuffer")
    bad = check_model(bad_path)
    assert bad["status"] == STATIC_FAIL
    assert bad["macs"]["total"] is None
    assert bad["static_structure_pass"] is False

    oversized_path = tmp_path / "oversized.tflite"
    content = smoke_artifacts["int8"].read_bytes()
    oversized_path.write_bytes(content + b"\x00" * (300_001 - len(content)))
    oversized = check_model(oversized_path)
    assert oversized["status"] == STATIC_FAIL
    assert oversized["gates"]["model_bytes"]["status"] == "FAIL"
    # Trailing bytes do not turn a size failure into a pass or alter the hash.
    assert oversized["model"]["bytes"] == 300_001


def test_cli_writes_failed_report_and_returns_exit_two(
    tmp_path: Path, smoke_artifacts: dict[str, Path]
) -> None:
    report_path = tmp_path / "budget.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "edgeai",
            "budget-check",
            "--model",
            str(smoke_artifacts["fp32"]),
            "--output",
            str(report_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    persisted = json.loads(report_path.read_text(encoding="utf-8"))
    assert persisted["status"] == STATIC_FAIL
    assert json.loads(result.stdout)["status"] == STATIC_FAIL


def _load_model_t(path: Path):
    _require_tensorflow()
    from tensorflow.lite.python import schema_py_generated as schema

    raw = path.read_bytes()
    root = schema.Model.GetRootAsModel(raw, 0)
    return schema, schema.ModelT.InitFromObj(root)


def _pack_model(model_t, path: Path) -> Path:
    import flatbuffers

    builder = flatbuffers.Builder(0)
    root = model_t.Pack(builder)
    builder.Finish(root, file_identifier=b"TFL3")
    path.write_bytes(bytes(builder.Output()))
    return path


def _mutated_model(
    source: Path, tmp_path: Path, name: str, mutator
) -> Path:
    schema, model_t = _load_model_t(source)
    mutator(model_t, schema)
    return _pack_model(model_t, tmp_path / f"{name}.tflite")


def test_opcode_uses_schema_compatibility_max_and_rejects_legacy_mismatch(
    tmp_path: Path, smoke_artifacts: dict[str, Path]
) -> None:
    int8_model = smoke_artifacts["int8"]

    def legacy_only(model_t, schema) -> None:
        code = model_t.operatorCodes[0]
        code.builtinCode = 0
        code.deprecatedBuiltinCode = int(schema.BuiltinOperator.CONV_2D)

    legacy_report = check_model(
        _mutated_model(int8_model, tmp_path, "legacy-only", legacy_only)
    )
    assert legacy_report["status"] == STATIC_PASS
    assert legacy_report["operator_codes"][0]["builtin_code"] == int(
        _load_model_t(int8_model)[0].BuiltinOperator.CONV_2D
    )

    def unsupported_legacy_mismatch(model_t, schema) -> None:
        code = model_t.operatorCodes[0]
        code.builtinCode = int(schema.BuiltinOperator.CONV_2D)
        code.deprecatedBuiltinCode = int(schema.BuiltinOperator.MAX_POOL_2D)

    mismatch_report = check_model(
        _mutated_model(
            int8_model,
            tmp_path,
            "unsupported-legacy-mismatch",
            unsupported_legacy_mismatch,
        )
    )
    assert mismatch_report["status"] == STATIC_FAIL
    assert any(
        error["check"] == "operator_allowlist"
        for error in mismatch_report["errors"]
    )


def _fully_connected_operator(model_t, schema):
    graph = model_t.subgraphs[0]
    fully_connected_code = int(schema.BuiltinOperator.FULLY_CONNECTED)
    for operator in graph.operators:
        opcode = model_t.operatorCodes[int(operator.opcodeIndex)]
        builtin_code = max(int(opcode.builtinCode), int(opcode.deprecatedBuiltinCode))
        if builtin_code == fully_connected_code:
            return graph, operator
    raise AssertionError("smoke model has no FULLY_CONNECTED operator")


@pytest.mark.parametrize(
    ("name", "expected_check"),
    [
        ("input-batch-2", "fully_connected_profile"),
        ("output-batch-2", "fully_connected_profile"),
        ("input-rank-3", "fully_connected_profile"),
        ("output-rank-3", "fully_connected_profile"),
        ("weights-rank-3", "fully_connected_profile"),
        ("keep-num-dims", "fully_connected_options"),
        ("asymmetric-quantize-inputs", "fully_connected_options"),
        ("quantized-bias-type", "fully_connected_options"),
    ],
)
def test_fully_connected_bounded_profile_rejects_other_shapes_and_modes(
    tmp_path: Path,
    smoke_artifacts: dict[str, Path],
    name: str,
    expected_check: str,
) -> None:
    """Exercise the local batch-1/rank-2 FC policy through real FlatBuffers."""

    def mutate(model_t, schema) -> None:
        graph, operator = _fully_connected_operator(model_t, schema)
        tensors = graph.tensors
        if name == "input-batch-2":
            tensors[int(operator.inputs[0])].shape = np.asarray(
                [2, 760], dtype=np.int32
            )
        elif name == "output-batch-2":
            tensors[int(operator.outputs[0])].shape = np.asarray(
                [2, 192], dtype=np.int32
            )
        elif name == "input-rank-3":
            tensors[int(operator.inputs[0])].shape = np.asarray(
                [1, 19, 40], dtype=np.int32
            )
        elif name == "output-rank-3":
            tensors[int(operator.outputs[0])].shape = np.asarray(
                [1, 1, 192], dtype=np.int32
            )
        elif name == "weights-rank-3":
            tensors[int(operator.inputs[1])].shape = np.asarray(
                [192, 760, 1], dtype=np.int32
            )
        elif name == "keep-num-dims":
            operator.builtinOptions.keepNumDims = True
        elif name == "asymmetric-quantize-inputs":
            operator.builtinOptions.asymmetricQuantizeInputs = True
        elif name == "quantized-bias-type":
            operator.builtinOptions.quantizedBiasType = int(schema.TensorType.INT16)
        else:  # pragma: no cover - test programming error
            raise AssertionError(name)

    report = check_model(
        _mutated_model(smoke_artifacts["int8"], tmp_path, name, mutate)
    )
    assert report["status"] == STATIC_FAIL
    assert report["static_budget_pass"] is False
    assert report["macs"]["total"] is None
    assert any(error["check"] == expected_check for error in report["errors"])
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    ("name", "expected_check"),
    [
        ("variable", "variable_tensor"),
        ("sparse", "sparse_tensor"),
        ("external-buffer-offset", "external_buffer"),
        ("external-buffer-size", "external_buffer"),
        ("unknown-version", "operator_version"),
        ("unsupported-mode", "convolution_options"),
        ("unknown-options-union", "operator_options"),
    ],
)
def test_unmodeled_storage_and_operator_modes_fail_closed(
    tmp_path: Path,
    smoke_artifacts: dict[str, Path],
    name: str,
    expected_check: str,
) -> None:
    def mutate(model_t, schema) -> None:
        if name == "variable":
            model_t.subgraphs[0].tensors[0].isVariable = True
        elif name == "sparse":
            model_t.subgraphs[0].tensors[0].sparsity = schema.SparsityParametersT()
        elif name == "external-buffer-offset":
            model_t.buffers[3].offset = 1
        elif name == "external-buffer-size":
            model_t.buffers[3].size = 1
        elif name == "unknown-version":
            model_t.operatorCodes[0].version = 99
        elif name == "unsupported-mode":
            model_t.subgraphs[0].operators[0].builtinOptions.strideH = 0
        elif name == "unknown-options-union":
            model_t.subgraphs[0].operators[0].builtinOptionsType = 255
        else:  # pragma: no cover - test programming error
            raise AssertionError(name)

    report = check_model(
        _mutated_model(smoke_artifacts["int8"], tmp_path, name, mutate)
    )
    assert report["status"] == STATIC_FAIL
    assert report["static_budget_pass"] is False
    assert any(error["check"] == expected_check for error in report["errors"])
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    ("name", "expected_check"),
    [
        ("nan-scale", "quantization_scales"),
        ("zero-scale", "quantization_scales"),
        ("negative-scale", "quantization_scales"),
        ("input-zero-point-range", "int8_zero_points"),
        ("pool-scale-mismatch", "activation_quantization_relation"),
        ("reshape-zero-point-mismatch", "activation_quantization_relation"),
        ("bias-scale-mismatch", "bias_quantization_relation"),
        ("bias-zero-point", "weight_zero_point"),
    ],
)
def test_invalid_quantization_metadata_fails_closed(
    tmp_path: Path,
    smoke_artifacts: dict[str, Path],
    name: str,
    expected_check: str,
) -> None:
    def mutate(model_t, schema) -> None:
        tensors = model_t.subgraphs[0].tensors

        if name == "nan-scale":
            tensors[0].quantization.scale = np.asarray([np.nan], dtype=np.float32)
        elif name == "zero-scale":
            tensors[0].quantization.scale = np.asarray([0.0], dtype=np.float32)
        elif name == "negative-scale":
            tensors[0].quantization.scale = np.asarray([-0.01], dtype=np.float32)
        elif name == "input-zero-point-range":
            tensors[0].quantization.zeroPoint = np.asarray([128], dtype=np.int64)
        elif name == "pool-scale-mismatch":
            scale = float(tensors[6].quantization.scale[0])
            tensors[6].quantization.scale = np.asarray(
                [scale * 1.01], dtype=np.float32
            )
        elif name == "reshape-zero-point-mismatch":
            tensors[7].quantization.zeroPoint = np.asarray([-127], dtype=np.int64)
        elif name == "bias-scale-mismatch":
            tensors[3].quantization.scale = np.asarray(
                tensors[3].quantization.scale * np.float32(1.01), dtype=np.float32
            )
        elif name == "bias-zero-point":
            tensors[3].quantization.zeroPoint = np.ones(8, dtype=np.int64)
        else:  # pragma: no cover - test programming error
            raise AssertionError(name)

    report = check_model(
        _mutated_model(smoke_artifacts["int8"], tmp_path, name, mutate)
    )
    assert report["status"] == STATIC_FAIL
    assert report["static_budget_pass"] is False
    assert any(error["check"] == expected_check for error in report["errors"])
    # Even NaN input bytes must produce a strict JSON-safe failure report.
    json.dumps(report, allow_nan=False)


def test_custom_mac_limit_rejects_without_large_model_allocation(
    monkeypatch: pytest.MonkeyPatch,
    smoke_artifacts: dict[str, Path],
) -> None:
    monkeypatch.setattr(
        budget_module,
        "get_profile",
        lambda name: BudgetProfile(max_macs=360_479),
    )
    report = budget_module.check_model(smoke_artifacts["int8"])
    assert report["status"] == STATIC_FAIL
    assert report["gates"]["macs"] == {
        "status": "FAIL",
        "actual": 360_480,
        "limit": 360_479,
        "unit": "MACs_per_inference",
    }
    assert any(error["check"] == "macs" for error in report["errors"])


def _convert_depthwise_variant(tmp_path: Path) -> Path:
    _require_tensorflow()
    import tensorflow as tf

    tf.keras.utils.set_random_seed(2058)
    inputs = tf.keras.Input(
        batch_shape=(1, 298, 40, 1), dtype=tf.float32, name="depthwise_input"
    )
    x = tf.keras.layers.DepthwiseConv2D(
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="same",
        depth_multiplier=2,
        use_bias=True,
    )(inputs)
    x = tf.keras.layers.AveragePooling2D(
        pool_size=(8, 4), strides=(8, 4), padding="same"
    )(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(192, use_bias=False)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_dataset():
        for _ in range(2):
            yield [np.zeros((1, 298, 40, 1), dtype=np.float32)]

    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    path = tmp_path / "depthwise.tflite"
    path.write_bytes(bytes(converter.convert()))
    return path


def test_depthwise_mac_oracle_on_real_converted_artifact(tmp_path: Path) -> None:
    report = check_model(_convert_depthwise_variant(tmp_path))
    assert report["status"] == STATIC_PASS
    assert [(item["operator"], item["macs"]) for item in report["macs"]["breakdown"]] == [
        ("DEPTHWISE_CONV_2D", (149 * 20 * 2) * 3 * 3),
        ("FULLY_CONNECTED", (19 * 5 * 2) * 192),
    ]
    assert report["macs"]["total"] == (149 * 20 * 2 * 3 * 3) + ((19 * 5 * 2) * 192)


def _convert_variant(tmp_path: Path, variant: str) -> Path:
    """Create only test-local synthetic variants of the known host model."""

    _require_tensorflow()
    import numpy as np
    import tensorflow as tf

    tf.keras.utils.set_random_seed(2058)
    if variant == "wrong-shape":
        input_shape = (298, 41, 1)
        inputs = tf.keras.Input(batch_shape=(1, *input_shape), dtype=tf.float32)
        x = tf.keras.layers.Flatten()(inputs)
        outputs = tf.keras.layers.Dense(192)(x)
    elif variant == "unsupported-op":
        input_shape = (298, 40, 1)
        inputs = tf.keras.Input(batch_shape=(1, *input_shape), dtype=tf.float32)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(inputs)
        x = tf.keras.layers.Flatten()(x)
        outputs = tf.keras.layers.Dense(192)(x)
    elif variant == "dynamic":
        inputs = tf.keras.Input(shape=(None, 40, 1), batch_size=1, dtype=tf.float32)
        x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
        outputs = tf.keras.layers.Dense(192)(x)
    else:  # pragma: no cover - test programming error
        raise AssertionError(variant)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if variant != "dynamic":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        def representative_dataset():
            shape = (1, 298, 41, 1) if variant == "wrong-shape" else (1, 298, 40, 1)
            for _ in range(2):
                yield [np.zeros(shape, dtype=np.float32)]

        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    path = tmp_path / f"{variant}.tflite"
    path.write_bytes(bytes(converter.convert()))
    return path


@pytest.mark.parametrize("variant", ["dynamic", "wrong-shape", "unsupported-op"])
def test_dynamic_shape_wrong_shape_and_unsupported_ops_fail_closed(
    tmp_path: Path, variant: str
) -> None:
    model_path = _convert_variant(tmp_path, variant)
    report = check_model(model_path)

    assert report["status"] == STATIC_FAIL
    assert report["static_budget_pass"] is False
    assert report["macs"]["status"] == "UNRESOLVED"
    expected_check = {
        "dynamic": "tensor_shape_signature",
        "wrong-shape": "input_shape",
        "unsupported-op": "operator_allowlist",
    }[variant]
    assert any(error["check"] == expected_check for error in report["errors"])
