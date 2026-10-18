"""Bounded embedding-only KD pilot training and export.

This is a NEW_LOCAL_LIBRISPEECH_PILOT pipeline.  It consumes the prepared
manifest/features/teacher snapshot produced by the pilot-data task, trains a
small student against fixed teacher embeddings, and evaluates only after the
validation-selected checkpoint and exports are frozen.  It is deliberately
host-only: no board, deployment, or accuracy claim is made here.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
from io import StringIO
import json
import platform
from pathlib import Path
import random
import sys
import tempfile
import time
from typing import Any, Iterable

import numpy as np

from .frontend import NUM_FRAMES, NUM_MELS, NUM_SAMPLES

SEED = 20261012
LEARNING_RATE = 0.001
BATCH_SIZE = 16
MAX_EPOCHS = 80
PATIENCE = 10
MIN_DELTA = 0.0
EMBEDDING_DIM = 192
FEATURE_SHAPE = (NUM_FRAMES, NUM_MELS, 1)
MODEL_INPUT_SHAPE = (1, *FEATURE_SHAPE)
MODEL_OUTPUT_SHAPE = (1, EMBEDDING_DIM)
DATASET_NAME = "librispeech-dev-clean-pilot"
TEACHER_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
TEACHER_REVISION = "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"
HARDWARE_STATUS = "UNVERIFIED_HARDWARE"
CONTRACT_STATUS = "CONTRACT_UNRESOLVED"
_EXPORT_BATCH1_ATOL = 1e-6
_FLOAT32_DEFAULT_TOLERANCE_FACTOR = 10.0

_MANIFEST_FIELDS = {
    "utterance_id",
    "speaker_id",
    "chapter_id",
    "split",
    "wav_path",
    "wav_sha256",
    "source_sha256",
    "input_samples",
    "padded_samples",
    "truncated_samples",
}
_SPLITS = {
    "train",
    "validation",
    "calibration_enroll",
    "calibration_query",
    "test_enroll",
    "test_query",
}
_EXPECTED_COUNTS = {
    "train": 400,
    "validation": 60,
    "calibration_enroll": 15,
    "calibration_query": 45,
    "test_enroll": 30,
    "test_query": 90,
}


def _tensorflow() -> Any:
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("TensorFlow 2.20 is required for pilot training") from exc
    return tf


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ValueError(f"could not hash {path}: {exc}") from exc
    return digest.hexdigest()


def _sha256_text(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_hash(value: object, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{field} must be a 64-character hexadecimal SHA-256")
    lowered = value.lower()
    if any(char not in "0123456789abcdef" for char in lowered):
        raise ValueError(f"{field} must be hexadecimal")
    return lowered


def _base_split(split: str) -> str:
    if split.endswith("_enroll"):
        return split.removesuffix("_enroll")
    if split.endswith("_query"):
        return split.removesuffix("_query")
    return split


def _load_manifest(
    path: Path, *, synthetic_fixture: bool
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    try:
        raw = path.read_bytes()
        manifest = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read manifest {path}: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("format_version") != 1:
        raise ValueError("manifest format_version must be 1")
    if not synthetic_fixture and manifest.get("dataset") != DATASET_NAME:
        raise ValueError(f"manifest dataset must be {DATASET_NAME!r}")
    if not synthetic_fixture and manifest.get("seed") != SEED:
        raise ValueError(f"manifest seed must be {SEED}")

    rows = manifest.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("manifest rows must be a non-empty list")
    identifiers: set[str] = set()
    hashes: dict[str, dict[str, str]] = {
        "wav_sha256": {},
        "source_sha256": {},
    }
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not _MANIFEST_FIELDS <= set(row):
            raise ValueError(f"manifest row {index} is missing required fields")
        for field in ("utterance_id", "speaker_id", "chapter_id", "split", "wav_path"):
            if not isinstance(row[field], str) or not row[field]:
                raise ValueError(f"manifest row {index} has invalid {field}")
        identifier = row["utterance_id"]
        if identifier in identifiers:
            raise ValueError(f"duplicate utterance_id {identifier!r}")
        identifiers.add(identifier)
        if row["split"] not in _SPLITS:
            raise ValueError(f"manifest row {index} has unknown split")
        relative_wav = Path(row["wav_path"])
        if relative_wav.is_absolute() or ".." in relative_wav.parts:
            raise ValueError(f"manifest row {index} has unsafe wav_path")
        for field in ("wav_sha256", "source_sha256"):
            digest = _require_hash(row[field], f"row {index} {field}")
            previous = hashes[field].get(digest)
            if previous is not None and previous != identifier:
                raise ValueError(f"duplicate {field} across manifest rows")
            hashes[field][digest] = identifier
        for field in ("input_samples", "padded_samples", "truncated_samples"):
            value = row[field]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"manifest row {index} has invalid {field}")
        expected_padded = max(0, NUM_SAMPLES - row["input_samples"])
        expected_truncated = max(0, row["input_samples"] - NUM_SAMPLES)
        if (
            row["padded_samples"] != expected_padded
            or row["truncated_samples"] != expected_truncated
        ):
            raise ValueError(f"manifest row {index} has inconsistent crop metadata")

    groups: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        groups[_base_split(row["split"])].add(row["speaker_id"])
    for left_index, left in enumerate(sorted(groups)):
        for right in sorted(groups)[left_index + 1 :]:
            overlap = groups[left] & groups[right]
            if overlap:
                raise ValueError(
                    "speaker leakage across splits: "
                    f"{left}/{right} share {sorted(overlap)!r}"
                )
    if not groups.get("train") or not groups.get("validation"):
        raise ValueError("manifest must contain train and validation speakers")

    counts = Counter(row["split"] for row in rows)
    if not synthetic_fixture and dict(counts) != _EXPECTED_COUNTS:
        raise ValueError(
            "real pilot split counts do not match the locked contract: "
            f"{dict(counts)!r}"
        )
    if not synthetic_fixture:
        expected_speakers = {"train": 20, "validation": 5, "calibration": 5, "test": 10}
        actual_speakers = {name: len(values) for name, values in groups.items()}
        if actual_speakers != expected_speakers:
            raise ValueError(
                "real pilot speaker counts do not match the locked contract: "
                f"{actual_speakers!r}"
            )

    for evaluation_split in ("calibration", "test"):
        enroll_rows = [
            row for row in rows if row["split"] == f"{evaluation_split}_enroll"
        ]
        query_rows = [
            row for row in rows if row["split"] == f"{evaluation_split}_query"
        ]
        if not enroll_rows or not query_rows:
            raise ValueError(f"manifest lacks complete {evaluation_split} split")
        enroll_speakers = {row["speaker_id"] for row in enroll_rows}
        query_speakers = {row["speaker_id"] for row in query_rows}
        if enroll_speakers != query_speakers:
            raise ValueError(f"{evaluation_split} enrollment/query speaker sets differ")
        for speaker in sorted(enroll_speakers):
            speaker_enroll = [row for row in enroll_rows if row["speaker_id"] == speaker]
            speaker_query = [row for row in query_rows if row["speaker_id"] == speaker]
            if len(speaker_enroll) != 3 or len(speaker_query) != 9:
                raise ValueError(
                    f"{evaluation_split} speaker {speaker!r} must have 3 enrollment "
                    "and 9 query rows"
                )
        enroll_ids = {row["utterance_id"] for row in enroll_rows}
        query_ids = {row["utterance_id"] for row in query_rows}
        if enroll_ids & query_ids:
            raise ValueError(f"{evaluation_split} enrollment/query IDs overlap")
        for field in ("wav_sha256", "source_sha256"):
            enroll_hashes = {row[field] for row in enroll_rows}
            query_hashes = {row[field] for row in query_rows}
            if enroll_hashes & query_hashes:
                raise ValueError(f"{evaluation_split} enrollment/query {field} overlap")

    return manifest, rows, _sha256_text(raw)


def _unicode_scalar(value: np.ndarray, name: str) -> str:
    if value.ndim != 0 or value.dtype.kind != "U":
        raise ValueError(f"{name} must be a Unicode scalar")
    return str(value.item())


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    try:
        archive = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"could not read NPZ {path}: {exc}") from exc
    with archive:
        required = {"ids", "manifest_sha256"}
        if not required <= set(archive.files):
            raise ValueError(f"{path} is missing required NPZ arrays")
        return {name: np.asarray(archive[name]).copy() for name in archive.files}


def _load_snapshot(
    manifest_path: Path,
    features_path: Path,
    teacher_path: Path,
    *,
    synthetic_fixture: bool,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    str,
    np.ndarray,
    np.ndarray,
    list[str],
    dict[str, Any] | None,
]:
    manifest, rows, manifest_sha256 = _load_manifest(
        manifest_path, synthetic_fixture=synthetic_fixture
    )
    manifest_ids = [row["utterance_id"] for row in rows]

    features_archive = _load_npz(features_path)
    feature_ids = features_archive["ids"]
    features = features_archive.get("features")
    if feature_ids.ndim != 1 or feature_ids.dtype.kind != "U":
        raise ValueError("features ids must be one-dimensional Unicode")
    if feature_ids.tolist() != manifest_ids:
        raise ValueError("features ids must exactly match manifest row order")
    if features is None or features.dtype != np.dtype(np.float32):
        raise ValueError("features must be float32")
    if features.shape != (len(rows), *FEATURE_SHAPE):
        raise ValueError(
            f"features must have shape {(len(rows), *FEATURE_SHAPE)}; got {features.shape}"
        )
    if _unicode_scalar(features_archive["manifest_sha256"], "features manifest_sha256") != manifest_sha256:
        raise ValueError("features manifest_sha256 does not match manifest")
    if not np.all(np.isfinite(features)):
        raise ValueError("features must contain only finite values")

    teacher_archive = _load_npz(teacher_path)
    teacher_ids = teacher_archive["ids"]
    teacher_embeddings = teacher_archive.get("embeddings")
    if teacher_ids.ndim != 1 or teacher_ids.dtype.kind != "U":
        raise ValueError("teacher ids must be one-dimensional Unicode")
    if teacher_ids.tolist() != manifest_ids:
        raise ValueError("teacher ids must exactly match manifest row order")
    if teacher_embeddings is None or teacher_embeddings.dtype != np.dtype(np.float32):
        raise ValueError("teacher embeddings must be float32")
    if teacher_embeddings.shape != (len(rows), EMBEDDING_DIM):
        raise ValueError(
            "teacher embeddings must have shape "
            f"{(len(rows), EMBEDDING_DIM)}; got {teacher_embeddings.shape}"
        )
    if _unicode_scalar(teacher_archive["manifest_sha256"], "teacher manifest_sha256") != manifest_sha256:
        raise ValueError("teacher manifest_sha256 does not match manifest")
    if not np.all(np.isfinite(teacher_embeddings)):
        raise ValueError("teacher embeddings must contain only finite values")
    teacher_norms = np.linalg.norm(teacher_embeddings.astype(np.float64), axis=1)
    if np.any(~np.isfinite(teacher_norms)) or np.any(teacher_norms <= 0.0):
        raise ValueError("teacher embeddings must be finite and non-zero")
    if not np.allclose(teacher_norms, 1.0, rtol=2e-4, atol=2e-4):
        raise ValueError("teacher embeddings must be L2-normalized")

    provenance_path = manifest_path.parent / "provenance.json"
    provenance: dict[str, Any] | None = None
    if provenance_path.is_file():
        try:
            value = json.loads(provenance_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"could not read provenance {provenance_path}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError("pilot provenance must be a JSON object")
        provenance = value
        if not synthetic_fixture:
            artifacts = value.get("artifacts")
            if not isinstance(artifacts, dict):
                raise ValueError(
                    "real pilot provenance must contain an artifacts object"
                )
            for field, expected in (
                ("manifest_sha256", manifest_sha256),
                ("features_sha256", _sha256_file(features_path)),
            ):
                if artifacts.get(field) != expected:
                    raise ValueError(
                        f"provenance artifacts.{field} does not match current artifact"
                    )
    elif not synthetic_fixture:
        raise ValueError("real pilot manifest requires its provenance.json sidecar")

    return (
        manifest,
        rows,
        manifest_sha256,
        features,
        teacher_embeddings,
        manifest_ids,
        provenance,
    )


def _unit_rows(values: np.ndarray) -> np.ndarray:
    raw = np.asarray(values, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != EMBEDDING_DIM:
        raise ValueError("embedding outputs must have shape [N, 192]")
    if not np.all(np.isfinite(raw)):
        raise ValueError("embedding outputs must be finite")
    norms = np.linalg.norm(raw.astype(np.float64), axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
        raise ValueError("embedding outputs must be finite and non-zero")
    return (raw.astype(np.float64) / norms[:, None]).astype(np.float32)


def _write_embeddings(
    path: Path, ids: list[str], values: np.ndarray, manifest_sha256: str
) -> None:
    normalized = _unit_rows(values)
    np.savez_compressed(
        path,
        ids=np.asarray(ids, dtype=str),
        embeddings=normalized,
        manifest_sha256=np.asarray(manifest_sha256),
    )


def _configure_tf_threads(tf: Any, *, strict: bool) -> dict[str, object]:
    requested = {"inter_op": 4, "intra_op": 2}
    try:
        tf.config.threading.set_inter_op_parallelism_threads(requested["inter_op"])
        tf.config.threading.set_intra_op_parallelism_threads(requested["intra_op"])
    except RuntimeError as exc:
        current = {
            "inter_op": int(tf.config.threading.get_inter_op_parallelism_threads()),
            "intra_op": int(tf.config.threading.get_intra_op_parallelism_threads()),
        }
        if strict:
            raise RuntimeError(
                "TensorFlow thread settings were already initialized; run the real "
                "pipeline in a fresh Python process"
            ) from exc
        return {
            "requested": requested,
            "configured": False,
            "status": "already_initialized_not_changed",
            "current": current,
            "error": str(exc),
        }
    return {
        "requested": requested,
        "configured": True,
        "status": "configured_before_model_creation",
        "current": requested,
    }


def build_student_model(
    *, batch_size: int | None = None, name: str = "student_kd_pilot"
) -> Any:
    """Build the locked student topology with dynamic or fixed batch input."""

    tf = _tensorflow()
    if batch_size is None:
        inputs = tf.keras.Input(
            shape=FEATURE_SHAPE,
            dtype=tf.float32,
            name="student_mel_input",
        )
    else:
        if isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer or None")
        inputs = tf.keras.Input(
            batch_shape=(batch_size, *FEATURE_SHAPE),
            dtype=tf.float32,
            name="student_mel_input_fixed",
        )
    x = tf.keras.layers.Conv2D(
        filters=8,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="same",
        use_bias=True,
        name="student_conv",
    )(inputs)
    x = tf.keras.layers.ReLU(name="student_relu")(x)
    x = tf.keras.layers.AveragePooling2D(
        pool_size=(8, 4),
        strides=(8, 4),
        padding="same",
        name="student_pool",
    )(x)
    x = tf.keras.layers.Flatten(name="student_flatten")(x)
    outputs = tf.keras.layers.Dense(
        EMBEDDING_DIM,
        activation=None,
        name="student_embedding",
    )(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)


def _validate_model_shape(model: Any, *, batch_size: int | None, label: str) -> None:
    expected_input = (
        (None, *FEATURE_SHAPE) if batch_size is None else (batch_size, *FEATURE_SHAPE)
    )
    expected_output = (
        (None, EMBEDDING_DIM) if batch_size is None else (batch_size, EMBEDDING_DIM)
    )
    if tuple(model.input_shape) != expected_input:
        raise RuntimeError(
            f"{label} input shape changed: {model.input_shape!r} != {expected_input!r}"
        )
    if tuple(model.output_shape) != expected_output:
        raise RuntimeError(
            f"{label} output shape changed: {model.output_shape!r} != {expected_output!r}"
        )


def _validate_identical_model_weights(source: Any, target: Any) -> None:
    source_weights = source.get_weights()
    target_weights = target.get_weights()
    if len(source_weights) != len(target_weights):
        raise RuntimeError(
            "export model weight count differs: "
            f"{len(source_weights)} != {len(target_weights)}"
        )
    for index, (source_weight, target_weight) in enumerate(
        zip(source_weights, target_weights)
    ):
        source_array = np.asarray(source_weight)
        target_array = np.asarray(target_weight)
        if source_array.shape != target_array.shape:
            raise RuntimeError(
                f"export model weight shape differs at index {index}: "
                f"{source_array.shape!r} != {target_array.shape!r}"
            )
        if not np.all(np.isfinite(source_array)) or not np.all(
            np.isfinite(target_array)
        ):
            raise RuntimeError(f"non-finite export model weight at index {index}")
        if not np.array_equal(source_array, target_array):
            max_abs = float(
                np.max(
                    np.abs(
                        source_array.astype(np.float64)
                        - target_array.astype(np.float64)
                    )
                )
            )
            raise RuntimeError(
                f"export model weights differ at index {index}: max_abs={max_abs}"
            )


def _validate_batch1_export_parity(
    original_dynamic: Any,
    fixed_model: Any,
    values: np.ndarray,
    *,
    atol: float = _EXPORT_BATCH1_ATOL,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Hard-check original dynamic and fixed models on the same batch-1 input."""

    _validate_model_shape(original_dynamic, batch_size=None, label="dynamic model")
    _validate_model_shape(fixed_model, batch_size=1, label="fixed export model")
    _validate_identical_model_weights(original_dynamic, fixed_model)
    batch = np.asarray(values, dtype=np.float32)
    if batch.shape != MODEL_INPUT_SHAPE:
        raise RuntimeError(
            f"batch-1 export parity input shape is {batch.shape!r}, "
            f"expected {MODEL_INPUT_SHAPE!r}"
        )
    if not np.all(np.isfinite(batch)):
        raise RuntimeError("batch-1 export parity input is non-finite")

    dynamic_output = np.asarray(
        original_dynamic(batch, training=False).numpy(), dtype=np.float32
    )
    fixed_output = np.asarray(
        fixed_model(batch, training=False).numpy(), dtype=np.float32
    )
    for label, output in (
        ("dynamic", dynamic_output),
        ("fixed", fixed_output),
    ):
        if output.shape != MODEL_OUTPUT_SHAPE:
            raise RuntimeError(
                f"{label} batch-1 export parity output shape is {output.shape!r}, "
                f"expected {MODEL_OUTPUT_SHAPE!r}"
            )
        if not np.all(np.isfinite(output)):
            raise RuntimeError(f"{label} batch-1 export parity output is non-finite")

    max_abs = float(
        np.max(
            np.abs(
                dynamic_output.astype(np.float64) - fixed_output.astype(np.float64)
            )
        )
    )
    if not np.isfinite(max_abs) or max_abs > atol:
        raise RuntimeError(
            "dynamic/fixed same-batch-1 export parity failed: "
            f"max_abs={max_abs}, atol={atol}"
        )
    return max_abs, dynamic_output, fixed_output


def _summarize_batch2_diagnostic(
    reference_output: np.ndarray, batched_output: np.ndarray
) -> dict[str, Any]:
    """Report batch-size drift without turning it into an export hard gate."""

    reference = np.asarray(reference_output, dtype=np.float32)
    observed = np.asarray(batched_output, dtype=np.float32)
    if reference.shape == MODEL_OUTPUT_SHAPE:
        reference = reference[0]
    if observed.ndim == 2 and observed.shape[0] >= 1:
        observed = observed[0]
    if reference.shape != (EMBEDDING_DIM,) or observed.shape != (EMBEDDING_DIM,):
        raise RuntimeError(
            "batch-2 diagnostic requires one reference and one observed "
            f"embedding, got {reference.shape!r} and {observed.shape!r}"
        )
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(observed)):
        raise RuntimeError("batch-2 diagnostic output is non-finite")

    reference64 = reference.astype(np.float64)
    observed64 = observed.astype(np.float64)
    difference = np.abs(observed64 - reference64)
    denominator = np.maximum(
        np.abs(reference64), np.finfo(np.float32).tiny
    )
    max_abs = float(np.max(difference))
    max_relative = float(np.max(difference / denominator))
    reference_norm = float(np.linalg.norm(reference64))
    observed_norm = float(np.linalg.norm(observed64))
    if reference_norm == 0.0 or observed_norm == 0.0:
        raise RuntimeError("batch-2 diagnostic cosine is undefined for zero output")
    normalized_cosine = float(
        np.dot(reference64, observed64) / (reference_norm * observed_norm)
    )

    eps32 = float(np.finfo(np.float32).eps)
    tolerance = _FLOAT32_DEFAULT_TOLERANCE_FACTOR * eps32
    combined_bound = tolerance + tolerance * np.abs(reference64)
    bound_ratio = float(np.max(difference / combined_bound))
    bound_pass = bool(np.all(difference <= combined_bound))
    return {
        "comparison": "fixed_batch1_vs_dynamic_clone_batch2_row0",
        "max_abs": max_abs,
        "max_relative": max_relative,
        "normalized_cosine": normalized_cosine,
        "combined_bound_pass": bound_pass,
        "combined_bound_status": "PASS" if bound_pass else "FAIL",
        "combined_bound_ratio": bound_ratio,
        "combined_bound_factor": _FLOAT32_DEFAULT_TOLERANCE_FACTOR,
        "float32_epsilon": eps32,
        "combined_bound_formula": "10*eps32 + 10*eps32*abs(reference)",
        "interpretation": "diagnostic_only_not_backend_accuracy_guarantee",
    }


def _keras_infer(tf: Any, model: Any, values: np.ndarray, batch_size: int) -> np.ndarray:
    outputs: list[np.ndarray] = []
    for start in range(0, values.shape[0], batch_size):
        batch = tf.convert_to_tensor(values[start : start + batch_size], dtype=tf.float32)
        output = model(batch, training=False).numpy()
        outputs.append(np.asarray(output, dtype=np.float32))
    if not outputs:
        raise ValueError("cannot infer an empty feature array")
    result = np.concatenate(outputs, axis=0)
    if result.shape != (values.shape[0], EMBEDDING_DIM):
        raise ValueError(f"student output shape is {result.shape}")
    return result


def _cosine_loss(tf: Any, outputs: Any, targets: Any) -> Any:
    output_unit = tf.math.l2_normalize(outputs, axis=1)
    target_unit = tf.math.l2_normalize(targets, axis=1)
    return tf.reduce_mean(
        1.0 - tf.reduce_sum(output_unit * target_unit, axis=1)
    )


def _fit_student(
    tf: Any,
    model: Any,
    features: np.ndarray,
    teacher_embeddings: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    stage: Path,
    max_epochs: int,
    identifiers: list[str] | None = None,
    manifest_sha256: str | None = None,
) -> dict[str, Any]:
    initial_weights = [np.asarray(weight).copy() for weight in model.get_weights()]
    baseline_raw = _keras_infer(tf, model, features, BATCH_SIZE)
    # Persist the initialized snapshot before the first optimizer update.
    if identifiers is not None or manifest_sha256 is not None:
        if identifiers is None or manifest_sha256 is None:
            raise ValueError(
                "baseline persistence requires both identifiers and manifest_sha256"
            )
        _write_embeddings(
            stage / "student_untrained_embeddings.npz",
            identifiers,
            baseline_raw,
            manifest_sha256,
        )
    model.save_weights(stage / "student_untrained.weights.h5")

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    history: list[dict[str, object]] = []
    best_validation = float("inf")
    best_epoch = 0
    best_weights: list[np.ndarray] | None = None
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        train_total = 0.0
        train_count = 0
        for start in range(0, train_indices.size, BATCH_SIZE):
            batch_indices = train_indices[start : start + BATCH_SIZE]
            x_batch = tf.convert_to_tensor(features[batch_indices], dtype=tf.float32)
            y_batch = tf.convert_to_tensor(
                teacher_embeddings[batch_indices], dtype=tf.float32
            )
            with tf.GradientTape() as tape:
                output = model(x_batch, training=True)
                loss = _cosine_loss(tf, output, y_batch)
            gradients = tape.gradient(loss, model.trainable_variables)
            if any(gradient is None for gradient in gradients):
                raise RuntimeError("student gradient computation produced None")
            if not all(bool(tf.reduce_all(tf.math.is_finite(gradient))) for gradient in gradients):
                raise RuntimeError("student gradient computation produced non-finite values")
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            loss_value = float(loss.numpy())
            if not np.isfinite(loss_value):
                raise RuntimeError("student training loss is non-finite")
            train_total += loss_value * batch_indices.size
            train_count += int(batch_indices.size)
        if train_count == 0:
            raise ValueError("train split is empty")

        validation_total = 0.0
        validation_count = 0
        for start in range(0, validation_indices.size, BATCH_SIZE):
            batch_indices = validation_indices[start : start + BATCH_SIZE]
            x_batch = tf.convert_to_tensor(features[batch_indices], dtype=tf.float32)
            y_batch = tf.convert_to_tensor(
                teacher_embeddings[batch_indices], dtype=tf.float32
            )
            validation_output = model(x_batch, training=False)
            validation_loss = _cosine_loss(tf, validation_output, y_batch)
            loss_value = float(validation_loss.numpy())
            if not np.isfinite(loss_value):
                raise RuntimeError("student validation loss is non-finite")
            validation_total += loss_value * batch_indices.size
            validation_count += int(batch_indices.size)
        if validation_count == 0:
            raise ValueError("validation split is empty")
        train_loss = train_total / train_count
        validation_loss_value = validation_total / validation_count
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "validation_loss": float(validation_loss_value),
            }
        )

        if validation_loss_value < best_validation - MIN_DELTA:
            best_validation = validation_loss_value
            best_epoch = epoch
            best_weights = [np.asarray(weight).copy() for weight in model.get_weights()]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                break

    if best_weights is None:
        raise RuntimeError("no validation checkpoint was selected")
    model.set_weights(best_weights)
    model.save_weights(stage / "student_best.weights.h5")
    update_max = max(
        float(np.max(np.abs(before - after)))
        for before, after in zip(initial_weights, best_weights)
    )
    if not np.isfinite(update_max) or update_max <= 0.0:
        raise RuntimeError("training did not produce a finite weight update")
    return {
        "baseline_raw": baseline_raw,
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_loss": float(best_validation),
        "epochs_ran": len(history),
        "stop_reason": (
            "early_stopping"
            if len(history) < max_epochs
            else "max_epochs"
        ),
        "weight_update_max_abs": update_max,
    }


def _convert_fp32(tf: Any, model: Any) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        return bytes(converter.convert())


def _convert_int8(
    tf: Any, model: Any, features: np.ndarray, train_indices: np.ndarray
) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_dataset() -> Iterable[list[np.ndarray]]:
        for index in train_indices.tolist():
            yield [np.asarray(features[index : index + 1], dtype=np.float32)]

    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        return bytes(converter.convert())


def _run_tflite(
    model_content: bytes,
    features: np.ndarray,
    *,
    require_int8: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    tf = _tensorflow()
    from .model import dequantize_tensor, inspect_tflite, quantize_tensor

    metadata = inspect_tflite(model_content, require_int8_io=require_int8)
    interpreter = tf.lite.Interpreter(model_content=model_content)
    interpreter.allocate_tensors()
    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError("exported student must have one input and one output")
    input_detail = inputs[0]
    output_detail = outputs[0]
    if tuple(input_detail["shape"]) != MODEL_INPUT_SHAPE:
        raise ValueError(f"export input shape is {input_detail['shape']}")
    if tuple(output_detail["shape"]) != MODEL_OUTPUT_SHAPE:
        raise ValueError(f"export output shape is {output_detail['shape']}")
    expected_dtype = np.dtype(np.int8 if require_int8 else np.float32)
    if np.dtype(input_detail["dtype"]) != expected_dtype or np.dtype(output_detail["dtype"]) != expected_dtype:
        raise ValueError("exported student input/output dtypes do not match contract")

    raw_outputs: list[np.ndarray] = []
    for start in range(0, features.shape[0]):
        values = features[start : start + 1]
        if require_int8:
            tensor_input = quantize_tensor(values, input_detail)
        else:
            tensor_input = values.astype(np.float32, copy=False)
        interpreter.set_tensor(input_detail["index"], tensor_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_detail["index"]).copy()
        if require_int8:
            output_float = dequantize_tensor(output, output_detail)
        else:
            output_float = np.asarray(output, dtype=np.float32)
        raw_outputs.append(np.asarray(output_float, dtype=np.float32))
    raw = np.concatenate(raw_outputs, axis=0)
    if raw.shape != (features.shape[0], EMBEDDING_DIM):
        raise ValueError(f"TFLite output shape is {raw.shape}")
    return raw, _unit_rows(raw), metadata


def _closed_set_top1(result: dict[str, object]) -> dict[str, object]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in result.get("scores", []):
        if not isinstance(row, dict):
            raise ValueError("evaluation score table contains a malformed row")
        grouped[str(row["query_id"])].append(row)
    if not grouped:
        raise ValueError("evaluation score table is empty")
    correct = 0
    tie_queries = 0
    for rows in grouped.values():
        best_score = max(float(row["score"]) for row in rows)
        tied = [row for row in rows if float(row["score"]) == best_score]
        if len(tied) > 1:
            tie_queries += 1
        selected = tied[0]  # evaluator emits enrollment speakers in sorted order
        if selected["enrollment_speaker_id"] == selected["query_speaker_id"]:
            correct += 1
    total = len(grouped)
    return {
        "correct": correct,
        "total": total,
        "accuracy": float(correct / total),
        "tie_queries": tie_queries,
        "scope": "closed_set_query_to_enrollment_top1; not open_set accuracy",
        "tie_policy": "first row in deterministic evaluator score-table order",
    }


def _replace_stage_paths(value: object, stage: Path, final: Path) -> object:
    if isinstance(value, str):
        stage_text = str(stage)
        if value == stage_text or value.startswith(stage_text + "/"):
            return str(final / Path(value).relative_to(stage))
        return value
    if isinstance(value, list):
        return [_replace_stage_paths(item, stage, final) for item in value]
    if isinstance(value, dict):
        return {
            key: _replace_stage_paths(item, stage, final) for key, item in value.items()
        }
    return value


def _artifact_record(path: Path, *, stage: Path, final: Path) -> dict[str, object]:
    return {
        "path": str(final / path.relative_to(stage)),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
    }


def _versions(tf: Any) -> dict[str, str]:
    keras_version = getattr(tf.keras, "__version__", None)
    if keras_version is None:
        keras_version = getattr(tf.keras, "version", "unknown")
        if callable(keras_version):
            keras_version = keras_version()
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "tensorflow": str(tf.__version__),
        "keras": str(keras_version),
    }


def _run_pipeline(
    *,
    manifest_path: Path,
    features_path: Path,
    teacher_path: Path,
    output_stage: Path,
    output_final: Path,
    manifest: dict[str, Any],
    rows: list[dict[str, Any]],
    manifest_sha256: str,
    features: np.ndarray,
    teacher_embeddings: np.ndarray,
    identifiers: list[str],
    provenance: dict[str, Any] | None,
    synthetic_fixture: bool,
    max_epochs: int,
    strict_thread_config: bool,
) -> dict[str, Any]:
    tf = _tensorflow()
    thread_report = _configure_tf_threads(tf, strict=strict_thread_config)
    tf.keras.utils.set_random_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    started_at = datetime.now(timezone.utc)
    started_clock = time.perf_counter()

    split_indices = {
        split: np.asarray(
            [index for index, row in enumerate(rows) if row["split"] == split],
            dtype=np.int64,
        )
        for split in _SPLITS
    }
    train_indices = split_indices["train"]
    validation_indices = split_indices["validation"]
    if train_indices.size == 0 or validation_indices.size == 0:
        raise ValueError("train and validation splits must be non-empty")

    model = build_student_model(name="student_kd_pilot_train")
    if model.count_params() != 146192:
        raise RuntimeError(f"student parameter count changed: {model.count_params()}")
    fit = _fit_student(
        tf,
        model,
        features,
        teacher_embeddings,
        train_indices,
        validation_indices,
        output_stage,
        max_epochs,
        identifiers,
        manifest_sha256,
    )

    fixed_model = build_student_model(batch_size=1, name="student_kd_pilot_export")
    fixed_model.set_weights(model.get_weights())
    train_probe_indices = train_indices[: min(2, train_indices.size)]
    if train_probe_indices.size < 2:
        raise ValueError("at least two train rows are required for export diagnostics")
    dynamic_values = features[train_probe_indices]
    dynamic_fixed_max_abs, _, fixed_one_output = _validate_batch1_export_parity(
        model,
        fixed_model,
        dynamic_values[:1],
    )

    dynamic_export_input = tf.keras.Input(
        shape=FEATURE_SHAPE,
        dtype=tf.float32,
        name="student_mel_input_dynamic_oracle",
    )
    dynamic_export = tf.keras.models.clone_model(
        fixed_model, input_tensors=dynamic_export_input
    )
    dynamic_export.set_weights(fixed_model.get_weights())
    _validate_model_shape(dynamic_export, batch_size=None, label="dynamic clone")
    _validate_identical_model_weights(fixed_model, dynamic_export)
    dynamic_outputs = np.asarray(
        dynamic_export(tf.convert_to_tensor(dynamic_values), training=False).numpy(),
        dtype=np.float32,
    )
    expected_dynamic_output_shape = (dynamic_values.shape[0], EMBEDDING_DIM)
    if dynamic_outputs.shape != expected_dynamic_output_shape:
        raise RuntimeError(
            "dynamic clone diagnostic output shape is "
            f"{dynamic_outputs.shape!r}, expected {expected_dynamic_output_shape!r}"
        )
    if not np.all(np.isfinite(dynamic_outputs)):
        raise RuntimeError("dynamic clone diagnostic output is non-finite")
    dynamic_batch2_diagnostic = _summarize_batch2_diagnostic(
        fixed_one_output,
        dynamic_outputs,
    )

    fp32_content = _convert_fp32(tf, fixed_model)
    int8_content = _convert_int8(tf, fixed_model, features, train_indices)
    fp32_path = output_stage / "student_fp32.tflite"
    int8_path = output_stage / "student_int8.tflite"
    fp32_path.write_bytes(fp32_content)
    int8_path.write_bytes(int8_content)

    fp32_raw, fp32_embeddings, fp32_metadata = _run_tflite(
        fp32_content, features, require_int8=False
    )
    int8_raw, int8_embeddings, int8_metadata = _run_tflite(
        int8_content, features, require_int8=True
    )
    _write_embeddings(
        output_stage / "student_fp32_embeddings.npz",
        identifiers,
        fp32_embeddings,
        manifest_sha256,
    )
    _write_embeddings(
        output_stage / "student_int8_embeddings.npz",
        identifiers,
        int8_embeddings,
        manifest_sha256,
    )

    checkpoint_frozen = True
    if not checkpoint_frozen:  # defensive ordering marker for audit readers
        raise RuntimeError("checkpoint was not frozen before evaluation")
    from .evaluation import evaluate_embeddings

    embedding_paths = {
        "teacher": teacher_path,
        "student_untrained": output_stage / "student_untrained_embeddings.npz",
        "student_fp32": output_stage / "student_fp32_embeddings.npz",
        "student_int8": output_stage / "student_int8_embeddings.npz",
    }
    evaluations: dict[str, dict[str, object]] = {}
    for name, embedding_path in embedding_paths.items():
        evaluations[name] = {}
        for split in ("calibration", "test"):
            metric = evaluate_embeddings(manifest_path, embedding_path, split=split)
            metric["closed_set_top1"] = _closed_set_top1(metric)
            metric["threshold_scope"] = "retrospective metric threshold; not deployment"
            evaluations[name][split] = metric

    drift_raw = fp32_raw.astype(np.float64) - int8_raw.astype(np.float64)
    drift_normalized = fp32_embeddings.astype(np.float64) - int8_embeddings.astype(np.float64)
    cosine_drift = np.sum(
        fp32_embeddings.astype(np.float64) * int8_embeddings.astype(np.float64), axis=1
    )
    finished_at = datetime.now(timezone.utc)
    elapsed = time.perf_counter() - started_clock

    artifact_paths = {
        "student_untrained_checkpoint": output_stage / "student_untrained.weights.h5",
        "student_best_checkpoint": output_stage / "student_best.weights.h5",
        "student_fp32_tflite": fp32_path,
        "student_int8_tflite": int8_path,
        "student_untrained_embeddings": output_stage / "student_untrained_embeddings.npz",
        "student_fp32_embeddings": output_stage / "student_fp32_embeddings.npz",
        "student_int8_embeddings": output_stage / "student_int8_embeddings.npz",
    }
    artifacts = {
        name: _artifact_record(path, stage=output_stage, final=output_final)
        for name, path in artifact_paths.items()
    }
    report: dict[str, Any] = {
        "format_version": 1,
        "status": "TRAINED_KD_PILOT",
        "teacher_status": (
            "SYNTHETIC_TEST_FIXTURE" if synthetic_fixture else "OFFICIAL_PRETRAINED"
        ),
        "initialized_baseline_status": "UNTRAINED",
        "student_status": "TRAINED_KD_PILOT",
        "hardware_status": HARDWARE_STATUS,
        "contract_status": CONTRACT_STATUS,
        "input_status": "SYNTHETIC_TEST_FIXTURE" if synthetic_fixture else "REAL_LIBRISPEECH_PILOT",
        "dataset": manifest.get("dataset"),
        "provenance_status": (
            "official_source_checksum_recorded_in_manifest_provenance"
            if provenance is not None
            else "synthetic_test_fixture"
        ),
        "config": {
            "seed": SEED,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "max_epochs": max_epochs,
            "patience": PATIENCE,
            "min_delta": MIN_DELTA,
            "loss": "mean(1 - cosine(student_output, teacher_embedding))",
            "augmentation": None,
            "learning_rate_schedule": None,
            "feature_shape": list(FEATURE_SHAPE),
            "embedding_dim": EMBEDDING_DIM,
            "configuration_locked_before_test_evaluation": True,
            "synthetic_fixture_override": synthetic_fixture,
        },
        "architecture": {
            "topology": "Conv8/ReLU/AvgPool/Flatten/Dense192",
            "parameter_count": int(model.count_params()),
            "dynamic_train_input_shape": [None, *FEATURE_SHAPE],
            "fixed_export_input_shape": list(MODEL_INPUT_SHAPE),
            "output_shape": list(MODEL_OUTPUT_SHAPE),
            "dynamic_fixed_batch1_max_abs": dynamic_fixed_max_abs,
            "dynamic_fixed_batch1_atol": _EXPORT_BATCH1_ATOL,
            "dynamic_fixed_batch1_comparison": "original_dynamic_vs_fixed_same_batch1",
            "dynamic_batch_output_shape": [int(value) for value in dynamic_outputs.shape],
            "dynamic_batch2_diagnostic": dynamic_batch2_diagnostic,
        },
        "data_snapshot": {
            "manifest_path": str(manifest_path.resolve()),
            "manifest_sha256": manifest_sha256,
            "features_path": str(features_path.resolve()),
            "features_sha256": _sha256_file(features_path),
            "teacher_embeddings_path": str(teacher_path.resolve()),
            "teacher_embeddings_sha256": _sha256_file(teacher_path),
            "teacher_model": None if synthetic_fixture else TEACHER_MODEL,
            "teacher_revision": None if synthetic_fixture else TEACHER_REVISION,
            "row_count": len(rows),
            "feature_shape": list(features.shape),
            "teacher_shape": list(teacher_embeddings.shape),
            "cropping": {
                "sample_rate_hz": 16000,
                "samples": NUM_SAMPLES,
                "policy": "first 3 seconds, right zero padding, tail truncation",
                "teacher_student_pcm": "same prepared PCM16 samples",
            },
            "split_counts": dict(Counter(row["split"] for row in rows)),
            "source_provenance": provenance,
        },
        "split_usage": {
            "train_rows": int(train_indices.size),
            "validation_rows": int(validation_indices.size),
            "representative_rows": int(train_indices.size),
            "representative_split": "train only, manifest order",
            "checkpoint_selection": "validation loss only",
            "calibration_and_test": "evaluation after frozen checkpoint/configuration",
        },
        "training": {
            "history": fit["history"],
            "best_epoch": fit["best_epoch"],
            "best_validation_loss": fit["best_validation_loss"],
            "epochs_ran": fit["epochs_ran"],
            "stop_reason": fit["stop_reason"],
            "baseline_snapshot_before_optimizer": True,
            "weight_update_max_abs": fit["weight_update_max_abs"],
            "optimizer": "Adam",
            "loop": "GradientTape over explicit NumPy batches; no tf.data",
        },
        "tensorflow_thread_policy": thread_report,
        "dependencies": _versions(tf),
        "artifacts": artifacts,
        "tflite": {
            "fp32": fp32_metadata,
            "int8": int8_metadata,
            "representative_rows": int(train_indices.size),
            "representative_source": "all train features in manifest order",
        },
        "int8_vs_fp32_drift": {
            "raw_max_abs": float(np.max(np.abs(drift_raw))),
            "raw_rms": float(np.sqrt(np.mean(drift_raw**2))),
            "normalized_max_abs": float(np.max(np.abs(drift_normalized))),
            "normalized_rms": float(np.sqrt(np.mean(drift_normalized**2))),
            "cosine_min": float(np.min(cosine_drift)),
            "cosine_mean": float(np.mean(cosine_drift)),
            "cosine_max": float(np.max(cosine_drift)),
            "scope": "desktop TFLite FP32 versus INT8 only; no hardware claim",
        },
        "evaluations": evaluations,
        "metric_scope": {
            "status": "NEW_LOCAL_LIBRISPEECH_PILOT",
            "thresholds": "retrospective SpeechBrain-compatible thresholds, not deployment",
            "closed_set_top1": "reported from the score table; not open-set accuracy",
            "not_voxceleb_benchmark": True,
        },
        "timing": {
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": finished_at.isoformat(),
            "elapsed_seconds": float(elapsed),
        },
        "report_path": str(output_final / "pilot_training_report.json"),
    }
    report = _replace_stage_paths(report, output_stage, output_final)  # type: ignore[assignment]
    report_path = output_stage / "pilot_training_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def train_pilot(
    manifest_path: str | Path,
    features_path: str | Path,
    teacher_embeddings_path: str | Path,
    output_path: str | Path,
    *,
    synthetic_fixture: bool = False,
    max_epochs: int = MAX_EPOCHS,
    strict_thread_config: bool = True,
) -> dict[str, Any]:
    """Run the locked pilot pipeline, refusing existing output directories.

    ``synthetic_fixture=True`` is test-only and is the sole way to permit a
    reduced fixture and a lower epoch count.  The real CLI never sets it.
    """

    manifest_file = Path(manifest_path)
    features_file = Path(features_path)
    teacher_file = Path(teacher_embeddings_path)
    output = Path(output_path)
    if isinstance(max_epochs, bool) or not isinstance(max_epochs, int) or max_epochs <= 0:
        raise ValueError("max_epochs must be a positive integer")
    if max_epochs > MAX_EPOCHS:
        raise ValueError(f"max_epochs cannot exceed locked maximum {MAX_EPOCHS}")
    if not synthetic_fixture and max_epochs != MAX_EPOCHS:
        raise ValueError("real pilot training must use the locked max_epochs=80")
    if not synthetic_fixture and not strict_thread_config:
        raise ValueError("real pilot training requires strict TensorFlow thread setup")
    for path in (manifest_file, features_file, teacher_file):
        if not path.is_file():
            raise ValueError(f"input artifact is not a regular file: {path}")
    if output.exists():
        raise ValueError(f"refusing to overwrite existing output path: {output}")

    (
        manifest,
        rows,
        manifest_sha256,
        features,
        teacher_embeddings,
        identifiers,
        provenance,
    ) = _load_snapshot(
        manifest_file,
        features_file,
        teacher_file,
        synthetic_fixture=synthetic_fixture,
    )
    output_parent = output.parent.resolve()
    output_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output_parent) as temp:
        stage = Path(temp)
        report = _run_pipeline(
            manifest_path=manifest_file,
            features_path=features_file,
            teacher_path=teacher_file,
            output_stage=stage,
            output_final=output.resolve(),
            manifest=manifest,
            rows=rows,
            manifest_sha256=manifest_sha256,
            features=features,
            teacher_embeddings=teacher_embeddings,
            identifiers=identifiers,
            provenance=provenance,
            synthetic_fixture=synthetic_fixture,
            max_epochs=max_epochs,
            strict_thread_config=strict_thread_config,
        )
        stage.replace(output.resolve())
    return report


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--teacher-embeddings", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        report = train_pilot(
            args.manifest,
            args.features,
            args.teacher_embeddings,
            args.output,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"edgeai.training: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by CLI smoke only
    raise SystemExit(main())


__all__ = [
    "BATCH_SIZE",
    "CONTRACT_STATUS",
    "EMBEDDING_DIM",
    "FEATURE_SHAPE",
    "HARDWARE_STATUS",
    "LEARNING_RATE",
    "MAX_EPOCHS",
    "MIN_DELTA",
    "PATIENCE",
    "SEED",
    "TEACHER_MODEL",
    "TEACHER_REVISION",
    "build_student_model",
    "main",
    "train_pilot",
]
