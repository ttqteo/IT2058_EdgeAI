"""Lazy TensorFlow/TFLite smoke-model tooling for desktop-only verification.

The generated model is intentionally untrained and uses synthetic inputs.  It
is a converter/interpreter smoke artifact, not a trained speaker model and not
evidence for accuracy, EER, FAR, board support, or deployment readiness.
"""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np

from .frontend import (
    CONTRACT_DISCOVERY,
    CONTRACT_UNRESOLVED,
    EXPERIMENTAL_HOST_ONLY,
    NUM_FRAMES,
    NUM_MELS,
)
from .matching import normalize_embedding

EMBEDDING_DIM = 192
MODEL_INPUT_SHAPE = (1, NUM_FRAMES, NUM_MELS, 1)
MODEL_OUTPUT_SHAPE = (1, EMBEDDING_DIM)
UNTRAINED_SMOKE_MODEL = "UNTRAINED_SMOKE_MODEL"
SYNTHETIC_INPUT = "SYNTHETIC_INPUT"
UNVERIFIED_HARDWARE = "UNVERIFIED_HARDWARE"


def _tensorflow() -> Any:
    """Import TensorFlow only when an ML operation is explicitly requested."""

    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "TensorFlow is required for smoke-model; install the optional "
            "'ml' dependency in the lead-managed environment"
        ) from exc
    return tf


def build_smoke_encoder(seed: int = 2058) -> Any:
    """Build the tiny, deliberately untrained Keras encoder."""

    tf = _tensorflow()
    tf.keras.utils.set_random_seed(seed)
    inputs = tf.keras.Input(
        batch_shape=MODEL_INPUT_SHAPE,
        dtype=tf.float32,
        name="mel_input",
    )
    x = tf.keras.layers.Conv2D(
        filters=8,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="same",
        use_bias=True,
        name="conv",
    )(inputs)
    x = tf.keras.layers.ReLU(name="relu")(x)
    x = tf.keras.layers.AveragePooling2D(
        pool_size=(8, 4),
        strides=(8, 4),
        padding="same",
        name="pool",
    )(x)
    x = tf.keras.layers.Flatten(name="flatten")(x)
    outputs = tf.keras.layers.Dense(
        EMBEDDING_DIM,
        activation=None,
        name="embedding",
    )(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="untrained_smoke_encoder")


def synthetic_representative_inputs(
    count: int = 8, seed: int = 2058
) -> list[np.ndarray]:
    """Return deterministic synthetic calibration inputs, never real audio."""

    if count <= 0:
        raise ValueError("representative input count must be positive")
    rng = np.random.default_rng(seed)
    return [
        rng.normal(
            loc=0.0,
            scale=0.25,
            size=MODEL_INPUT_SHAPE,
        ).astype(np.float32)
        for _ in range(count)
    ]


def _convert_fp32(model: Any) -> bytes:
    tf = _tensorflow()
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        return bytes(converter.convert())


def _convert_int8(model: Any, representative_inputs: Iterable[np.ndarray]) -> bytes:
    tf = _tensorflow()
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_dataset() -> Iterable[list[np.ndarray]]:
        for value in representative_inputs:
            yield [np.asarray(value, dtype=np.float32)]

    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        return bytes(converter.convert())


def _json_dtype(dtype: Any) -> str:
    return np.dtype(dtype).name


def _describe_tensor(detail: dict[str, Any]) -> dict[str, Any]:
    params = detail.get("quantization_parameters", {})
    scales = np.asarray(params.get("scales", []))
    zero_points = np.asarray(params.get("zero_points", []))
    return {
        "name": str(detail.get("name", "")),
        "index": int(detail["index"]),
        "shape": [int(value) for value in np.asarray(detail["shape"]).tolist()],
        "shape_signature": [
            int(value)
            for value in np.asarray(detail.get("shape_signature", detail["shape"])).tolist()
        ],
        "dtype": _json_dtype(detail["dtype"]),
        "quantization": {
            "scales": [float(value) for value in scales.tolist()],
            "zero_points": [int(value) for value in zero_points.tolist()],
            "quantized_dimension": int(params.get("quantized_dimension", 0)),
        },
    }


def _per_tensor_quantization(detail: dict[str, Any]) -> tuple[float, int]:
    params = detail.get("quantization_parameters", {})
    scales = np.asarray(params.get("scales", []), dtype=np.float64)
    zero_points = np.asarray(params.get("zero_points", []), dtype=np.int64)
    if scales.size != 1 or zero_points.size != 1:
        raise ValueError(
            f"tensor {detail.get('name', '<unnamed>')} lacks per-tensor quantization"
        )
    scale = float(scales[0])
    zero_point = int(zero_points[0])
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("quantization scale must be finite and positive")
    return scale, zero_point


def quantize_tensor(values: np.ndarray, detail: dict[str, Any]) -> np.ndarray:
    """Quantize a float tensor using the actual per-tensor I/O metadata."""

    scale, zero_point = _per_tensor_quantization(detail)
    dtype = np.dtype(detail["dtype"])
    if not np.issubdtype(dtype, np.integer):
        raise ValueError("quantize_tensor requires an integer tensor metadata record")
    values = np.asarray(values, dtype=np.float32)
    if not np.all(np.isfinite(values)):
        raise ValueError("values to quantize must be finite")
    info = np.iinfo(dtype)
    quantized = np.rint(values.astype(np.float64) / scale) + zero_point
    return np.clip(quantized, info.min, info.max).astype(dtype)


def dequantize_tensor(values: np.ndarray, detail: dict[str, Any]) -> np.ndarray:
    """Dequantize a tensor using the actual per-tensor I/O metadata."""

    scale, zero_point = _per_tensor_quantization(detail)
    quantized = np.asarray(values)
    return ((quantized.astype(np.float32) - zero_point) * np.float32(scale)).astype(
        np.float32
    )


def inspect_tflite(model_content: bytes, *, require_int8_io: bool = False) -> dict[str, Any]:
    """Inspect actual interpreter I/O metadata and the internal op list."""

    tf = _tensorflow()
    interpreter = tf.lite.Interpreter(model_content=model_content)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    if len(input_details) != 1 or len(output_details) != 1:
        raise ValueError("smoke model must have exactly one input and one output")

    input_record = _describe_tensor(input_details[0])
    output_record = _describe_tensor(output_details[0])
    if tuple(input_details[0]["shape"]) != MODEL_INPUT_SHAPE:
        raise ValueError(
            f"unexpected model input shape {input_details[0]['shape']}; "
            f"expected {MODEL_INPUT_SHAPE}"
        )
    if tuple(output_details[0]["shape"]) != MODEL_OUTPUT_SHAPE:
        raise ValueError(
            f"unexpected model output shape {output_details[0]['shape']}; "
            f"expected {MODEL_OUTPUT_SHAPE}"
        )
    if require_int8_io:
        if input_details[0]["dtype"] != np.int8 or output_details[0]["dtype"] != np.int8:
            raise ValueError("full INT8 smoke model must expose int8 input and output")
        _per_tensor_quantization(input_details[0])
        _per_tensor_quantization(output_details[0])

    try:
        raw_ops = interpreter._get_ops_details()
    except AttributeError as exc:  # pragma: no cover - package API dependent
        raise RuntimeError("installed interpreter cannot expose its internal op list") from exc
    operations = [
        {
            "index": int(index),
            "op_name": str(op.get("op_name", "")),
            "inputs": [int(value) for value in np.asarray(op.get("inputs", [])).tolist()],
            "outputs": [int(value) for value in np.asarray(op.get("outputs", [])).tolist()],
        }
        for index, op in enumerate(raw_ops)
    ]
    if not operations or any(not op["op_name"] for op in operations):
        raise ValueError("interpreter returned an empty or unnamed internal op list")

    tensor_details = interpreter.get_tensor_details()
    float_tensor_names = [
        str(detail.get("name", ""))
        for detail in tensor_details
        if np.dtype(detail["dtype"]) == np.dtype(np.float32)
    ]
    if require_int8_io and float_tensor_names:
        raise ValueError(
            "full INT8 smoke model contains float32 tensors: "
            + ", ".join(float_tensor_names)
        )

    return {
        "input": input_record,
        "output": output_record,
        "operations": operations,
        "tensor_count": len(tensor_details),
        "float32_tensor_names": float_tensor_names,
    }


def _timing_summary(samples_ms: list[float]) -> dict[str, float]:
    values = np.asarray(samples_ms, dtype=np.float64)
    return {
        "count": int(values.size),
        "min_ms": float(np.min(values)),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "max_ms": float(np.max(values)),
    }


def invoke_desktop(
    model_content: bytes,
    input_values: np.ndarray,
    *,
    quantized: bool,
    repeats: int = 5,
) -> dict[str, Any]:
    """Invoke a TFLite artifact on the desktop and L2-normalize its output."""

    if repeats <= 0:
        raise ValueError("repeats must be positive")
    tf = _tensorflow()
    interpreter = tf.lite.Interpreter(model_content=model_content)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    values = np.asarray(input_values, dtype=np.float32)
    if values.shape != MODEL_INPUT_SHAPE:
        raise ValueError(f"input shape must be {MODEL_INPUT_SHAPE}; got {values.shape}")
    if quantized:
        tensor_input = quantize_tensor(values, input_detail)
    else:
        tensor_input = values.astype(np.float32)

    interpreter.set_tensor(input_detail["index"], tensor_input)
    interpreter.invoke()
    samples_ms: list[float] = []
    output = None
    for _ in range(repeats):
        interpreter.set_tensor(input_detail["index"], tensor_input)
        start = time.perf_counter_ns()
        interpreter.invoke()
        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000.0
        samples_ms.append(elapsed_ms)
        output = interpreter.get_tensor(output_detail["index"]).copy()
    if output is None:  # pragma: no cover - repeats validation prevents this
        raise AssertionError("desktop invoke produced no output")
    output_float = (
        dequantize_tensor(output, output_detail)
        if quantized
        else np.asarray(output, dtype=np.float32)
    )
    if output_float.shape != MODEL_OUTPUT_SHAPE:
        raise ValueError(f"unexpected output shape {output_float.shape}")
    unit_embedding = normalize_embedding(output_float[0])
    return {
        "timing_ms": _timing_summary(samples_ms),
        "raw_output_l2": float(np.linalg.norm(output_float.astype(np.float64))),
        "unit_embedding_l2": float(np.linalg.norm(unit_embedding.astype(np.float64))),
        "input_dtype": _json_dtype(tensor_input.dtype),
        "output_dtype": _json_dtype(output.dtype),
    }


def create_smoke_model(output_dir: str | Path, *, seed: int = 2058) -> dict[str, Any]:
    """Create, inspect, invoke, and report the untrained desktop smoke model."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model = build_smoke_encoder(seed=seed)
    representative_inputs = synthetic_representative_inputs(seed=seed)
    fp32_content = _convert_fp32(model)
    int8_content = _convert_int8(model, representative_inputs)

    fp32_file = output_path / "encoder_fp32.tflite"
    int8_file = output_path / "encoder_int8.tflite"
    fp32_file.write_bytes(fp32_content)
    int8_file.write_bytes(int8_content)

    fp32_metadata = inspect_tflite(fp32_content)
    int8_metadata = inspect_tflite(int8_content, require_int8_io=True)
    fp32_invoke = invoke_desktop(
        fp32_content,
        representative_inputs[0],
        quantized=False,
    )
    int8_invoke = invoke_desktop(
        int8_content,
        representative_inputs[0],
        quantized=True,
    )

    report: dict[str, Any] = {
        "status": UNTRAINED_SMOKE_MODEL,
        "input_status": SYNTHETIC_INPUT,
        "frontend_status": EXPERIMENTAL_HOST_ONLY,
        "final_runtime_status": CONTRACT_UNRESOLVED,
        "contract_discovery": CONTRACT_DISCOVERY,
        "hardware_status": UNVERIFIED_HARDWARE,
        "metrics": {"accuracy": None, "eer": None, "far": None},
        "model": {
            "input_shape": list(MODEL_INPUT_SHAPE),
            "output_shape": list(MODEL_OUTPUT_SHAPE),
            "embedding_dim": EMBEDDING_DIM,
            "trained": False,
            "training_data": None,
        },
        "artifacts": {
            "fp32": {"path": str(fp32_file), "bytes": len(fp32_content)},
            "int8": {"path": str(int8_file), "bytes": len(int8_content)},
        },
        "interpreter": {
            "fp32": fp32_metadata,
            "int8": int8_metadata,
        },
        "desktop_invoke": {"fp32": fp32_invoke, "int8": int8_invoke},
        "timing_scope": "desktop TFLite Interpreter only; no board measurement",
        "thresholds": None,
        "accuracy_scope": None,
    }
    report_file = output_path / "model_smoke_report.json"
    report["report_path"] = str(report_file)
    import json

    report_file.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


__all__ = [
    "CONTRACT_DISCOVERY",
    "CONTRACT_UNRESOLVED",
    "EMBEDDING_DIM",
    "EXPERIMENTAL_HOST_ONLY",
    "MODEL_INPUT_SHAPE",
    "MODEL_OUTPUT_SHAPE",
    "SYNTHETIC_INPUT",
    "UNTRAINED_SMOKE_MODEL",
    "UNVERIFIED_HARDWARE",
    "build_smoke_encoder",
    "create_smoke_model",
    "dequantize_tensor",
    "inspect_tflite",
    "invoke_desktop",
    "quantize_tensor",
    "synthetic_representative_inputs",
]
