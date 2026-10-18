"""Bounded AAM/KD comparison training for the local host pilot.

This module is deliberately additive to :mod:`edgeai.training`.  It keeps the
existing embedding-only pilot untouched and adds two training-only classifier
heads: AAM with cosine KD, and standalone supervised AAM.  The exported
artifacts contain the encoder only; no classifier head is exported.

The comparison is a ``NEW_LOCAL_DESIGN`` for the prepared LibriSpeech pilot.
It does not establish a benchmark, a final model contract, or hardware
support.  In particular, this module intentionally does not call the
retrospective calibration/test evaluator.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import tempfile
import time
from typing import Any, Iterable

import numpy as np

from . import training as _training
from .frontend import NUM_FRAMES, NUM_MELS, NUM_SAMPLES


SEED = 20261012
LEARNING_RATE = 0.001
BATCH_SIZE = 20
MAX_EPOCHS = 80
PATIENCE = 15
AAM_MARGIN = 0.2
AAM_SCALE = 30.0
KD_WEIGHT = 1.0
NUM_CLASSES = 20
VALIDATION_ENROLL_ROWS = 3
VALIDATION_QUERY_ROWS = 9
EMBEDDING_DIM = 192
FEATURE_SHAPE = (NUM_FRAMES, NUM_MELS, 1)
MODEL_INPUT_SHAPE = (1, *FEATURE_SHAPE)
MODEL_OUTPUT_SHAPE = (1, EMBEDDING_DIM)
DATASET_NAME = "librispeech-dev-clean-pilot"
HARDWARE_STATUS = "UNVERIFIED_HARDWARE"
CONTRACT_STATUS = "NEW_LOCAL_DESIGN"
COMPARISON_STATUS = "NEW_LOCAL_COMPARISON"
SYNTHETIC_STATUS = "SYNTHETIC_TEST_FIXTURE"
TEACHER_MODEL = _training.TEACHER_MODEL
TEACHER_REVISION = _training.TEACHER_REVISION
_NORM_FLOOR = np.float32(1e-12)
_AAM_CLAMP = np.float32(1e-7)


def _tensorflow() -> Any:
    """Import TensorFlow only when the comparison is actually used."""

    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("TensorFlow 2.20 is required for comparative training") from exc
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


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _unicode_scalar(value: np.ndarray, name: str) -> str:
    if value.ndim != 0 or value.dtype.kind != "U":
        raise ValueError(f"{name} must be a Unicode scalar")
    return str(value.item())


def _json_write(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _path_exists(path: Path) -> bool:
    # Path.exists() is false for a dangling symlink.  Refusing both cases is
    # important because the output directory is an immutable experiment result.
    return path.exists() or os.path.lexists(path)


def _validate_float_hyperparameters(
    *, margin: float, scale: float
) -> tuple[float, float]:
    try:
        margin_value = float(margin)
        scale_value = float(scale)
    except (TypeError, ValueError) as exc:
        raise ValueError("AAM margin and scale must be numeric") from exc
    if not np.isfinite(margin_value) or not 0.0 <= margin_value < math.pi:
        raise ValueError("AAM margin must be finite and in [0, pi)")
    if not np.isfinite(scale_value) or scale_value <= 0.0:
        raise ValueError("AAM scale must be finite and positive")
    return margin_value, scale_value


def _strict_unit_tensor(tf: Any, values: Any, *, name: str) -> Any:
    """Normalize rows with PyTorch ``F.normalize``'s explicit eps floor.

    SpeechBrain's classifier uses ``torch.nn.functional.normalize`` with its
    default ``eps=1e-12``.  TensorFlow's ``l2_normalize`` uses a squared
    epsilon internally, so the denominator is written explicitly here.
    Zero/non-finite vectors are rejected before the local comparison rather
    than being silently repaired.  That is intentionally stricter than the
    upstream primitive.
    """

    tensor = tf.cast(tf.convert_to_tensor(values), tf.float32)
    if tensor.shape.rank != 2:
        raise ValueError(f"{name} must have rank 2; got {tensor.shape}")
    tf.debugging.assert_all_finite(tensor, f"{name} must be finite")
    norms = tf.sqrt(tf.reduce_sum(tf.square(tensor), axis=1, keepdims=True))
    tf.debugging.assert_all_finite(norms, f"{name} norms must be finite")
    tf.debugging.assert_greater(
        norms,
        tf.zeros_like(norms),
        message=f"{name} rows must be non-zero",
    )
    return tensor / tf.maximum(norms, tf.constant(_NORM_FLOOR, dtype=tf.float32))


def normalized_classifier(tf: Any, embeddings: Any, kernel: Any) -> Any:
    """Return normalized-classifier cosine scores.

    Parameters are intentionally explicit for independent arithmetic tests:
    ``embeddings`` is ``[B, D]`` and ``kernel`` is ``[D, C]``.  Class vectors
    are normalized column-wise, matching SpeechBrain's normalized ECAPA
    classifier.  The result is float32 ``[B, C]`` and has no bias term.
    """

    values = tf.cast(tf.convert_to_tensor(embeddings), tf.float32)
    weights = tf.cast(tf.convert_to_tensor(kernel), tf.float32)
    if values.shape.rank != 2 or weights.shape.rank != 2:
        raise ValueError(
            "normalized_classifier expects embeddings [B,D] and kernel [D,C]"
        )
    if values.shape[-1] is not None and weights.shape[0] is not None:
        if int(values.shape[-1]) != int(weights.shape[0]):
            raise ValueError(
                "embedding dimension and classifier kernel depth do not match"
            )
    tf.debugging.assert_equal(
        tf.shape(values)[1],
        tf.shape(weights)[0],
        message="embedding dimension and classifier kernel depth do not match",
    )
    embedding_unit = _strict_unit_tensor(tf, values, name="embeddings")
    # Kernel is [D,C], so the class vectors are its columns.
    kernel_unit = tf.transpose(
        _strict_unit_tensor(tf, tf.transpose(weights), name="classifier columns")
    )
    result = tf.linalg.matmul(embedding_unit, kernel_unit)
    tf.debugging.assert_all_finite(result, "classifier cosine output must be finite")
    return tf.cast(result, tf.float32)


def aam_logits(
    tf: Any,
    cosine: Any,
    labels: Any,
    *,
    margin: float = AAM_MARGIN,
    scale: float = AAM_SCALE,
    easy_margin: bool = False,
) -> Any:
    """Apply SpeechBrain-compatible additive angular margin to cosine scores.

    ``cosine`` must already be ``[B,C]`` classifier scores and ``labels`` an
    integer ``[B]`` vector.  The output is float32 ``[B,C]``.  The clamp,
    fallback branch, and target-only replacement mirror the verified
    SpeechBrain 1.1.1 implementation; local finite/range checks fail closed.
    """

    margin_value, scale_value = _validate_float_hyperparameters(
        margin=margin, scale=scale
    )
    scores = tf.cast(tf.convert_to_tensor(cosine), tf.float32)
    target = tf.convert_to_tensor(labels)
    if scores.shape.rank != 2 or target.shape.rank != 1:
        raise ValueError("aam_logits expects cosine [B,C] and labels [B]")
    if not target.dtype.is_integer:
        raise ValueError("AAM labels must have an integer dtype")
    if scores.shape[0] is not None and target.shape[0] is not None:
        if int(scores.shape[0]) != int(target.shape[0]):
            raise ValueError("AAM cosine and labels batch sizes differ")
    tf.debugging.assert_equal(
        tf.shape(scores)[0],
        tf.shape(target)[0],
        message="AAM cosine and labels batch sizes differ",
    )
    tf.debugging.assert_all_finite(scores, "AAM cosine scores must be finite")
    classes = scores.shape[1]
    if classes is None:
        raise ValueError("AAM requires a statically known class dimension")
    classes = int(classes)
    tf.debugging.assert_greater_equal(
        target, tf.zeros_like(target), message="AAM labels must be non-negative"
    )
    tf.debugging.assert_less(
        target,
        tf.cast(classes, target.dtype),
        message="AAM labels exceed classifier class count",
    )

    clamp = tf.constant(_AAM_CLAMP, dtype=tf.float32)
    clamped = tf.clip_by_value(scores, -1.0 + clamp, 1.0 - clamp)
    sine = tf.sqrt(tf.maximum(tf.constant(0.0, tf.float32), 1.0 - tf.square(clamped)))
    cos_margin = tf.constant(math.cos(margin_value), dtype=tf.float32)
    sin_margin = tf.constant(math.sin(margin_value), dtype=tf.float32)
    phi = clamped * cos_margin - sine * sin_margin

    if easy_margin:
        modified = tf.where(clamped > 0.0, phi, clamped)
    else:
        threshold = tf.constant(math.cos(math.pi - margin_value), dtype=tf.float32)
        correction = tf.constant(
            math.sin(math.pi - margin_value) * margin_value, dtype=tf.float32
        )
        modified = tf.where(
            clamped > threshold,
            phi,
            clamped - correction,
        )
    one_hot = tf.one_hot(target, depth=classes, dtype=tf.float32)
    output = tf.where(one_hot > 0.0, modified, clamped)
    output = tf.cast(output * tf.constant(scale_value, tf.float32), tf.float32)
    tf.debugging.assert_all_finite(output, "AAM logits must be finite")
    return output


def _aam_cross_entropy(tf: Any, logits: Any, labels: Any) -> Any:
    target = tf.convert_to_tensor(labels)
    depth = logits.shape[-1]
    if depth is None:
        raise ValueError("AAM cross-entropy requires a static class dimension")
    one_hot = tf.one_hot(target, depth=int(depth), dtype=tf.float32)
    log_probabilities = tf.nn.log_softmax(tf.cast(logits, tf.float32), axis=-1)
    losses = -tf.reduce_sum(one_hot * log_probabilities, axis=-1)
    result = tf.reduce_mean(losses)
    tf.debugging.assert_all_finite(result, "AAM cross-entropy must be finite")
    return tf.cast(result, tf.float32)


def _cosine_kd_loss(tf: Any, embeddings: Any, teacher_embeddings: Any) -> Any:
    student_unit = _strict_unit_tensor(tf, embeddings, name="student embeddings")
    teacher_unit = tf.stop_gradient(
        _strict_unit_tensor(tf, teacher_embeddings, name="teacher embeddings")
    )
    tf.debugging.assert_equal(
        tf.shape(student_unit),
        tf.shape(teacher_unit),
        message="student and teacher embedding shapes differ",
    )
    result = tf.reduce_mean(
        1.0 - tf.reduce_sum(student_unit * teacher_unit, axis=1)
    )
    tf.debugging.assert_all_finite(result, "KD loss must be finite")
    return tf.cast(result, tf.float32)


def _method_losses(
    tf: Any,
    embeddings: Any,
    labels: Any,
    kernel: Any,
    *,
    method: str,
    teacher_embeddings: Any | None = None,
) -> tuple[Any, Any, Any | None]:
    """Compute one train batch, with no teacher branch for supervised AAM."""

    if method not in {"kd_aam", "aam_supervised"}:
        raise ValueError(f"unknown comparison method {method!r}")
    cosine = normalized_classifier(tf, embeddings, kernel)
    logits = aam_logits(tf, cosine, labels)
    classification = _aam_cross_entropy(tf, logits, labels)
    if method == "aam_supervised":
        if teacher_embeddings is not None:
            raise ValueError("supervised AAM loss must not receive teacher embeddings")
        return classification, classification, None
    if teacher_embeddings is None:
        raise ValueError("KD+AAM loss requires teacher embeddings")
    kd = _cosine_kd_loss(tf, embeddings, teacher_embeddings)
    return classification + KD_WEIGHT * kd, classification, kd


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    try:
        archive = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"could not read NPZ {path}: {exc}") from exc
    with archive:
        if not {"ids", "manifest_sha256"} <= set(archive.files):
            raise ValueError(f"{path} is missing required NPZ arrays")
        return {name: np.asarray(archive[name]).copy() for name in archive.files}


def _load_provenance(
    manifest_path: Path,
    features_path: Path,
    manifest_sha256: str,
    *,
    synthetic_fixture: bool,
) -> dict[str, Any] | None:
    provenance_path = manifest_path.parent / "provenance.json"
    if not provenance_path.is_file():
        if synthetic_fixture:
            return None
        raise ValueError("real pilot manifest requires provenance.json")
    try:
        value = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read provenance {provenance_path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("provenance must be a JSON object")
    if not synthetic_fixture:
        artifacts = value.get("artifacts")
        if not isinstance(artifacts, dict):
            raise ValueError("real provenance must contain an artifacts object")
        if artifacts.get("manifest_sha256") != manifest_sha256:
            raise ValueError("provenance manifest hash does not match current manifest")
        if artifacts.get("features_sha256") != _sha256_file(features_path):
            raise ValueError("provenance features hash does not match current features")
    return value


def _load_features_only(
    manifest_path: Path,
    features_path: Path,
    *,
    synthetic_fixture: bool,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    str,
    np.ndarray,
    list[str],
    dict[str, Any] | None,
]:
    """Load only manifest/features; this is the supervised loader boundary."""

    manifest, rows, manifest_sha256 = _training._load_manifest(
        manifest_path, synthetic_fixture=synthetic_fixture
    )
    identifiers = [str(row["utterance_id"]) for row in rows]
    archive = _load_npz(features_path)
    feature_ids = archive["ids"]
    features = archive.get("features")
    if feature_ids.ndim != 1 or feature_ids.dtype.kind != "U":
        raise ValueError("feature ids must be one-dimensional Unicode")
    if feature_ids.tolist() != identifiers:
        raise ValueError("feature ids must exactly match manifest row order")
    if features is None or features.dtype != np.dtype(np.float32):
        raise ValueError("features must be float32")
    if features.shape != (len(rows), *FEATURE_SHAPE):
        raise ValueError(
            f"features must have shape {(len(rows), *FEATURE_SHAPE)}; got {features.shape}"
        )
    if _unicode_scalar(archive["manifest_sha256"], "features manifest_sha256") != manifest_sha256:
        raise ValueError("features manifest_sha256 does not match manifest")
    if not np.all(np.isfinite(features)):
        raise ValueError("features must contain only finite values")
    provenance = _load_provenance(
        manifest_path,
        features_path,
        manifest_sha256,
        synthetic_fixture=synthetic_fixture,
    )
    return manifest, rows, manifest_sha256, features, identifiers, provenance


def _load_teacher_only(
    teacher_path: Path,
    identifiers: list[str],
    manifest_sha256: str,
) -> tuple[np.ndarray, str]:
    """Load the KD-only teacher artifact after the standalone feature load."""

    archive = _load_npz(teacher_path)
    teacher_ids = archive["ids"]
    embeddings = archive.get("embeddings")
    if teacher_ids.ndim != 1 or teacher_ids.dtype.kind != "U":
        raise ValueError("teacher ids must be one-dimensional Unicode")
    if teacher_ids.tolist() != identifiers:
        raise ValueError("teacher ids must exactly match manifest row order")
    if embeddings is None or embeddings.dtype != np.dtype(np.float32):
        raise ValueError("teacher embeddings must be float32")
    if embeddings.shape != (len(identifiers), EMBEDDING_DIM):
        raise ValueError(
            "teacher embeddings must have shape "
            f"{(len(identifiers), EMBEDDING_DIM)}; got {embeddings.shape}"
        )
    if _unicode_scalar(archive["manifest_sha256"], "teacher manifest_sha256") != manifest_sha256:
        raise ValueError("teacher manifest_sha256 does not match manifest")
    if not np.all(np.isfinite(embeddings)):
        raise ValueError("teacher embeddings must be finite")
    norms = np.linalg.norm(embeddings.astype(np.float64), axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
        raise ValueError("teacher embeddings must be finite and non-zero")
    if not np.allclose(norms, 1.0, rtol=2e-4, atol=2e-4):
        raise ValueError("teacher embeddings must be L2-normalized")
    return embeddings, _sha256_file(teacher_path)


def _protocol_indices(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate and index the locked 20x20 train / 5x12 validation design."""

    train_by_speaker: dict[str, list[int]] = defaultdict(list)
    validation_by_speaker: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if row["split"] == "train":
            train_by_speaker[str(row["speaker_id"])].append(index)
        elif row["split"] == "validation":
            validation_by_speaker[str(row["speaker_id"])].append(index)
    train_speakers = sorted(train_by_speaker)
    validation_speakers = sorted(validation_by_speaker)
    if len(train_speakers) != NUM_CLASSES:
        raise ValueError(f"training requires exactly {NUM_CLASSES} train speakers")
    if len(validation_speakers) != 5:
        raise ValueError("validation requires exactly 5 speakers")
    if set(train_speakers) & set(validation_speakers):
        raise ValueError("train and validation speaker sets overlap")
    for speaker in train_speakers:
        if len(train_by_speaker[speaker]) != BATCH_SIZE:
            raise ValueError(
                f"train speaker {speaker!r} must have exactly {BATCH_SIZE} rows"
            )
        train_by_speaker[speaker].sort(key=lambda i: str(rows[i]["utterance_id"]))
    for speaker in validation_speakers:
        validation_by_speaker[speaker].sort(
            key=lambda i: str(rows[i]["utterance_id"])
        )
        if len(validation_by_speaker[speaker]) != 12:
            raise ValueError(f"validation speaker {speaker!r} must have exactly 12 rows")

    labels = np.full(len(rows), -1, dtype=np.int32)
    for label, speaker in enumerate(train_speakers):
        labels[train_by_speaker[speaker]] = label
    if np.any(labels[np.concatenate([indices for indices in train_by_speaker.values()])] < 0):
        raise ValueError("failed to assign all train speaker labels")

    enrollment_indices: list[int] = []
    query_indices: list[int] = []
    for speaker in validation_speakers:
        values = validation_by_speaker[speaker]
        enrollment_indices.extend(values[:VALIDATION_ENROLL_ROWS])
        query_indices.extend(values[VALIDATION_ENROLL_ROWS:])
    return {
        "train_by_speaker": {key: value[:] for key, value in train_by_speaker.items()},
        "validation_by_speaker": {
            key: value[:] for key, value in validation_by_speaker.items()
        },
        "train_speakers": train_speakers,
        "validation_speakers": validation_speakers,
        "labels": labels,
        "train_indices": np.asarray(
            [index for speaker in train_speakers for index in train_by_speaker[speaker]],
            dtype=np.int64,
        ),
        "validation_indices": np.asarray(
            [index for speaker in validation_speakers for index in validation_by_speaker[speaker]],
            dtype=np.int64,
        ),
        "enrollment_indices": np.asarray(enrollment_indices, dtype=np.int64),
        "query_indices": np.asarray(query_indices, dtype=np.int64),
    }


def balanced_epoch_schedule(
    rows: list[dict[str, Any]], *, epoch: int, seed: int = SEED
) -> np.ndarray:
    """Return ``[20 batches, 20 speakers]`` indices for one locked epoch.

    ``epoch`` is one-based.  Every train row appears once, and every batch has
    one row from each sorted train speaker.  The schedule is deterministic for
    the explicit seed/epoch pair.
    """

    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= 0:
        raise ValueError("epoch must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    protocol = _protocol_indices(rows)
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
    columns: list[np.ndarray] = []
    for speaker in protocol["train_speakers"]:
        column = np.asarray(protocol["train_by_speaker"][speaker], dtype=np.int64).copy()
        rng.shuffle(column)
        columns.append(column)
    schedule = np.stack(columns, axis=1)
    expected = set(protocol["train_indices"].tolist())
    if schedule.shape != (BATCH_SIZE, NUM_CLASSES):
        raise RuntimeError(f"balanced schedule has unexpected shape {schedule.shape}")
    if set(schedule.reshape(-1).tolist()) != expected:
        raise RuntimeError("balanced schedule does not cover train rows exactly once")
    return schedule


def validation_protocol(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Expose the sorted 3-enrollment/9-query validation split."""

    protocol = _protocol_indices(rows)
    protocol = dict(protocol)
    protocol["enrollment_ids"] = [rows[i]["utterance_id"] for i in protocol["enrollment_indices"]]
    protocol["query_ids"] = [rows[i]["utterance_id"] for i in protocol["query_indices"]]
    return protocol


def select_validation_checkpoint(
    validation_eers: Iterable[float], *, patience: int = PATIENCE
) -> dict[str, Any]:
    """Apply strict-improvement/first-tie validation checkpoint selection."""

    if isinstance(patience, bool) or not isinstance(patience, int) or patience <= 0:
        raise ValueError("patience must be a positive integer")
    values = [float(value) for value in validation_eers]
    if not values:
        raise ValueError("validation EER history cannot be empty")
    if any(not np.isfinite(value) for value in values):
        raise ValueError("validation EER history must be finite")
    best = float("inf")
    best_epoch = 0
    stale = 0
    stop_epoch = len(values)
    stop_reason = "max_epochs"
    for epoch, value in enumerate(values, start=1):
        if value < best:
            best = value
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                stop_epoch = epoch
                stop_reason = "early_stopping"
                break
    return {
        "best_epoch": best_epoch,
        "best_validation_eer": float(best),
        "stop_epoch": stop_epoch,
        "stop_reason": stop_reason,
        "strict_improvement": True,
        "tie_policy": "first minimum; equal EER is not an improvement",
    }


def _validation_eer(
    model: Any,
    features: np.ndarray,
    rows: list[dict[str, Any]],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    """Infer validation only and compute the locked retrospective EER."""

    raw = np.asarray(model(features[protocol["validation_indices"]], training=False).numpy(), dtype=np.float32)
    if raw.shape != (protocol["validation_indices"].size, EMBEDDING_DIM):
        raise RuntimeError(f"validation encoder output shape is {raw.shape}")
    if not np.all(np.isfinite(raw)):
        raise RuntimeError("validation encoder output is non-finite")
    norms = np.linalg.norm(raw.astype(np.float64), axis=1)
    if np.any(norms <= 0.0) or not np.all(np.isfinite(norms)):
        raise RuntimeError("validation encoder output contains zero vectors")
    unit = (raw.astype(np.float64) / norms[:, None]).astype(np.float32)
    by_index = {
        int(index): unit[position]
        for position, index in enumerate(protocol["validation_indices"].tolist())
    }
    templates: dict[str, np.ndarray] = {}
    for speaker in protocol["validation_speakers"]:
        enrollment = np.stack(
            [by_index[index] for index in protocol["validation_by_speaker"][speaker][:3]],
            axis=0,
        )
        template_raw = np.mean(enrollment.astype(np.float64), axis=0)
        template_norm = float(np.linalg.norm(template_raw))
        if not np.isfinite(template_norm) or template_norm <= 0.0:
            raise RuntimeError("validation enrollment template is zero/non-finite")
        templates[speaker] = (template_raw / template_norm).astype(np.float32)

    labels: list[int] = []
    scores: list[float] = []
    query_speakers = {index: rows[index]["speaker_id"] for index in protocol["query_indices"]}
    for query_index in protocol["query_indices"].tolist():
        query = by_index[query_index].astype(np.float64)
        for speaker in protocol["validation_speakers"]:
            score = float(np.dot(query, templates[speaker].astype(np.float64)))
            if not np.isfinite(score):
                raise RuntimeError("validation score is non-finite")
            labels.append(int(query_speakers[query_index] == speaker))
            scores.append(score)
    from .evaluation import evaluate_scores

    metric = evaluate_scores(labels, scores)
    return {
        "eer": float(metric["eer"]),
        "eer_threshold": float(metric["eer_threshold"]),
        "target_trials": int(metric["target_trials"]),
        "non_target_trials": int(metric["non_target_trials"]),
        "speaker_count": len(protocol["validation_speakers"]),
        "enrollment_per_speaker": VALIDATION_ENROLL_ROWS,
        "query_per_speaker": VALIDATION_QUERY_ROWS,
        "metric_provenance": "SpeechBrain-compatible evaluate_scores; validation-only",
        "selection_scope": "validation EER only; no classifier loss/top1/minDCF selection",
    }


def _build_classifier_head(tf: Any) -> Any:
    head = tf.keras.layers.Dense(
        NUM_CLASSES,
        use_bias=False,
        activation=None,
        name="aam_classifier",
    )
    head(tf.zeros((1, EMBEDDING_DIM), dtype=tf.float32))
    if tuple(head.kernel.shape) != (EMBEDDING_DIM, NUM_CLASSES):
        raise RuntimeError(f"AAM head kernel shape changed: {head.kernel.shape}")
    return head


def _finite_gradients(tf: Any, gradients: list[Any | None]) -> None:
    if any(gradient is None for gradient in gradients):
        raise RuntimeError("comparison gradient computation produced None")
    if not all(bool(tf.reduce_all(tf.math.is_finite(gradient))) for gradient in gradients):
        raise RuntimeError("comparison gradient computation produced non-finite values")


def _save_classifier_checkpoint(head: Any, path: Path) -> None:
    kernel = np.asarray(head.kernel.numpy(), dtype=np.float32)
    if kernel.shape != (EMBEDDING_DIM, NUM_CLASSES) or not np.all(np.isfinite(kernel)):
        raise RuntimeError("classifier checkpoint kernel is invalid")
    np.savez_compressed(path, kernel=kernel)


def _fit_method(
    tf: Any,
    *,
    method: str,
    encoder: Any,
    head: Any,
    features: np.ndarray,
    teacher_embeddings: np.ndarray | None,
    rows: list[dict[str, Any]],
    protocol: dict[str, Any],
    stage: Path,
    max_epochs: int,
) -> dict[str, Any]:
    initial_encoder = [np.asarray(value).copy() for value in encoder.get_weights()]
    initial_head = [np.asarray(value).copy() for value in head.get_weights()]
    encoder.save_weights(stage / "encoder_initial.weights.h5")
    _save_classifier_checkpoint(head, stage / "classifier_initial.npz")
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    variables = list(encoder.trainable_variables) + list(head.trainable_variables)
    history: list[dict[str, Any]] = []
    best_eer = float("inf")
    best_epoch = 0
    best_encoder: list[np.ndarray] | None = None
    best_head: list[np.ndarray] | None = None
    stale = 0

    for epoch in range(1, max_epochs + 1):
        schedule = balanced_epoch_schedule(rows, epoch=epoch)
        train_total = 0.0
        train_aam = 0.0
        train_kd = 0.0
        for batch_indices in schedule:
            x_batch = tf.convert_to_tensor(features[batch_indices], dtype=tf.float32)
            y_batch = tf.convert_to_tensor(protocol["labels"][batch_indices], dtype=tf.int32)
            teacher_batch = None
            if method == "kd_aam":
                if teacher_embeddings is None:
                    raise RuntimeError("KD method has no teacher snapshot")
                teacher_batch = tf.convert_to_tensor(
                    teacher_embeddings[batch_indices], dtype=tf.float32
                )
            with tf.GradientTape() as tape:
                embeddings = encoder(x_batch, training=True)
                total, classification, kd = _method_losses(
                    tf,
                    embeddings,
                    y_batch,
                    head.kernel,
                    method=method,
                    teacher_embeddings=teacher_batch,
                )
            gradients = tape.gradient(total, variables)
            _finite_gradients(tf, gradients)
            optimizer.apply_gradients(zip(gradients, variables))
            total_value = float(total.numpy())
            classification_value = float(classification.numpy())
            kd_value = 0.0 if kd is None else float(kd.numpy())
            if not all(np.isfinite(value) for value in (total_value, classification_value, kd_value)):
                raise RuntimeError("comparison training loss is non-finite")
            train_total += total_value
            train_aam += classification_value
            train_kd += kd_value

        validation = _validation_eer(encoder, features, rows, protocol)
        row = {
            "epoch": epoch,
            "train_total_loss": train_total / BATCH_SIZE,
            "train_aam_loss": train_aam / BATCH_SIZE,
            "train_kd_loss": None if method == "aam_supervised" else train_kd / BATCH_SIZE,
            "validation_eer": validation["eer"],
            "validation_eer_threshold": validation["eer_threshold"],
            "validation_target_trials": validation["target_trials"],
            "validation_non_target_trials": validation["non_target_trials"],
        }
        history.append(row)
        if validation["eer"] < best_eer:
            best_eer = validation["eer"]
            best_epoch = epoch
            best_encoder = [np.asarray(value).copy() for value in encoder.get_weights()]
            best_head = [np.asarray(value).copy() for value in head.get_weights()]
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break

    if best_encoder is None or best_head is None:
        raise RuntimeError("no finite validation checkpoint was selected")
    encoder.set_weights(best_encoder)
    head.set_weights(best_head)
    encoder.save_weights(stage / "encoder_best.weights.h5")
    _save_classifier_checkpoint(head, stage / "classifier_best.npz")
    weight_update_max = max(
        [
            float(np.max(np.abs(before - after)))
            for before, after in zip(initial_encoder, best_encoder)
        ]
        + [
            float(np.max(np.abs(before - after)))
            for before, after in zip(initial_head, best_head)
        ]
    )
    if not np.isfinite(weight_update_max) or weight_update_max <= 0.0:
        raise RuntimeError("comparison training produced no finite weight update")
    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_eer": float(best_eer),
        "epochs_ran": len(history),
        "stop_reason": "early_stopping" if stale >= PATIENCE else "max_epochs",
        "weight_update_max_abs": weight_update_max,
        "initial_encoder_weights": initial_encoder,
        "initial_head_weights": initial_head,
        "best_encoder_weights": best_encoder,
        "best_head_weights": best_head,
    }


def _artifact_record(path: Path, *, stage: Path, final: Path) -> dict[str, Any]:
    return {
        "path": str(final / path.relative_to(stage)),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
    }


def _weight_digest(weights: Iterable[np.ndarray]) -> str:
    """Digest weight arrays with dtype/shape framing, not just raw bytes."""

    digest = hashlib.sha256()
    for value in weights:
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _checkpoint_paths(stage: Path) -> dict[str, Path]:
    return {
        "encoder_initial": stage / "encoder_initial.weights.h5",
        "encoder_best": stage / "encoder_best.weights.h5",
        "classifier_initial": stage / "classifier_initial.npz",
        "classifier_best": stage / "classifier_best.npz",
    }


def _checkpoint_bindings(
    stage: Path, *, root_stage: Path, final_root: Path
) -> dict[str, dict[str, Any]]:
    """Capture the files selected before export, using final artifact paths."""

    bindings: dict[str, dict[str, Any]] = {}
    for name, path in _checkpoint_paths(stage).items():
        if not path.is_file():
            raise RuntimeError(f"checkpoint binding is not a regular file: {path}")
        relative = path.relative_to(root_stage)
        bindings[name] = {
            "path": str(final_root / relative),
            "stage_relative_path": str(relative),
            "bytes": int(path.stat().st_size),
            "sha256": _sha256_file(path),
        }
    return bindings


def _assert_checkpoint_stable(
    checkpoint_paths: dict[str, Path],
    expected: dict[str, dict[str, Any]],
    *,
    method: str,
    phase: str,
) -> None:
    for name, path in checkpoint_paths.items():
        if not path.is_file():
            raise RuntimeError(
                f"{phase}: checkpoint changed during comparison: {method}/{name}"
            )
        observed_bytes = int(path.stat().st_size)
        observed_sha256 = _sha256_file(path)
        binding = expected[name]
        if (
            observed_bytes != int(binding["bytes"])
            or observed_sha256 != binding["sha256"]
        ):
            raise RuntimeError(
                f"{phase}: checkpoint changed during comparison: {method}/{name}"
            )


def _source_hashes() -> dict[str, str]:
    return {
        label: _sha256_file(path) for label, path in _source_paths().items()
    }


def _source_paths() -> dict[str, Path]:
    package_dir = Path(__file__).resolve().parent
    return {
        "comparative_training.py": package_dir / "comparative_training.py",
        "training.py": package_dir / "training.py",
        "evaluation.py": package_dir / "evaluation.py",
        "model.py": package_dir / "model.py",
        "frontend.py": package_dir / "frontend.py",
    }


def _input_paths(
    manifest_path: Path,
    features_path: Path,
    teacher_path: Path | None,
    provenance_path: Path | None,
) -> dict[str, Path]:
    result = {
        "manifest_sha256": manifest_path,
        "features_sha256": features_path,
    }
    if teacher_path is not None:
        result["teacher_embeddings_sha256"] = teacher_path
    if provenance_path is not None:
        result["provenance_sha256"] = provenance_path
    return result


def _input_hashes(
    manifest_path: Path,
    features_path: Path,
    teacher_path: Path | None,
    *,
    provenance_path: Path | None = None,
) -> dict[str, str]:
    result = {
        "manifest_sha256": _sha256_file(manifest_path),
        "features_sha256": _sha256_file(features_path),
    }
    if teacher_path is not None:
        result["teacher_embeddings_sha256"] = _sha256_file(teacher_path)
    if provenance_path is not None:
        result["provenance_sha256"] = _sha256_file(provenance_path)
    return result


def _assert_stable(
    paths: dict[str, Path], expected: dict[str, str], *, phase: str
) -> None:
    for label, path in paths.items():
        observed = _sha256_file(path)
        if observed != expected[label]:
            raise RuntimeError(
                f"{phase}: input/source changed during comparison: {label}"
            )


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


def _export_encoder(
    tf: Any,
    *,
    encoder: Any,
    features: np.ndarray,
    identifiers: list[str],
    manifest_sha256: str,
    train_indices: np.ndarray,
    stage: Path,
) -> dict[str, Any]:
    """Freeze/export/invoke one encoder after both methods have been selected."""

    fixed = _training.build_student_model(batch_size=1, name="comparison_encoder_export")
    fixed.set_weights(encoder.get_weights())
    probe = features[train_indices[:1]]
    parity_max_abs, _, _ = _training._validate_batch1_export_parity(encoder, fixed, probe)
    fp32_content = _training._convert_fp32(tf, fixed)
    int8_content = _training._convert_int8(tf, fixed, features, train_indices)
    fp32_path = stage / "encoder_fp32.tflite"
    int8_path = stage / "encoder_int8.tflite"
    fp32_path.write_bytes(fp32_content)
    int8_path.write_bytes(int8_content)
    fp32_raw, fp32_embeddings, fp32_metadata = _training._run_tflite(
        fp32_content, features, require_int8=False
    )
    int8_raw, int8_embeddings, int8_metadata = _training._run_tflite(
        int8_content, features, require_int8=True
    )
    fp32_embeddings_path = stage / "encoder_fp32_embeddings.npz"
    int8_embeddings_path = stage / "encoder_int8_embeddings.npz"
    _training._write_embeddings(
        fp32_embeddings_path, identifiers, fp32_embeddings, manifest_sha256
    )
    _training._write_embeddings(
        int8_embeddings_path, identifiers, int8_embeddings, manifest_sha256
    )
    raw_delta = fp32_raw.astype(np.float64) - int8_raw.astype(np.float64)
    normalized_delta = fp32_embeddings.astype(np.float64) - int8_embeddings.astype(np.float64)
    cosine = np.sum(
        fp32_embeddings.astype(np.float64) * int8_embeddings.astype(np.float64), axis=1
    )
    return {
        "fp32_path": fp32_path,
        "int8_path": int8_path,
        "fp32_embeddings_path": fp32_embeddings_path,
        "int8_embeddings_path": int8_embeddings_path,
        "fp32_metadata": fp32_metadata,
        "int8_metadata": int8_metadata,
        "dynamic_fixed_batch1_max_abs": float(parity_max_abs),
        "dynamic_fixed_batch1_atol": 1e-6,
        "row_count": int(features.shape[0]),
        "representative_rows": int(train_indices.size),
        "representative_source": "all train features in manifest order",
        "int8_vs_fp32_drift": {
            "raw_max_abs": float(np.max(np.abs(raw_delta))),
            "raw_rms": float(np.sqrt(np.mean(raw_delta**2))),
            "normalized_max_abs": float(np.max(np.abs(normalized_delta))),
            "normalized_rms": float(np.sqrt(np.mean(normalized_delta**2))),
            "cosine_min": float(np.min(cosine)),
            "cosine_mean": float(np.mean(cosine)),
            "cosine_max": float(np.max(cosine)),
            "scope": "desktop TFLite FP32 versus INT8; no hardware claim",
        },
    }


def _method_report(
    *,
    method: str,
    fit: dict[str, Any],
    export: dict[str, Any],
    stage: Path,
    final: Path,
    source_hashes: dict[str, str],
    input_hashes: dict[str, str],
    manifest_path: Path,
    features_path: Path,
    manifest: dict[str, Any],
    rows: list[dict[str, Any]],
    protocol: dict[str, Any],
    provenance: dict[str, Any] | None,
    teacher_used: bool,
    synthetic_fixture: bool,
    max_epochs: int,
    checkpoint_bindings: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    artifacts_paths = {
        "encoder_initial_checkpoint": stage / "encoder_initial.weights.h5",
        "encoder_best_checkpoint": stage / "encoder_best.weights.h5",
        "training_head_initial_checkpoint": stage / "classifier_initial.npz",
        "training_head_best_checkpoint": stage / "classifier_best.npz",
        "encoder_fp32_tflite": export["fp32_path"],
        "encoder_int8_tflite": export["int8_path"],
        "encoder_fp32_embeddings": export["fp32_embeddings_path"],
        "encoder_int8_embeddings": export["int8_embeddings_path"],
    }
    data_snapshot: dict[str, Any] = {
        "manifest_path": str(manifest_path.resolve()),
        "features_path": str(features_path.resolve()),
        "manifest_sha256": input_hashes["manifest_sha256"],
        "features_sha256": input_hashes["features_sha256"],
        "row_count": len(rows),
        "feature_shape": list((len(rows), *FEATURE_SHAPE)),
        "source_provenance": provenance,
    }
    if "provenance_sha256" in input_hashes:
        provenance_path = manifest_path.parent / "provenance.json"
        data_snapshot["provenance_path"] = str(provenance_path.resolve())
        data_snapshot["provenance_sha256"] = input_hashes["provenance_sha256"]

    report: dict[str, Any] = {
        "format_version": 1,
        "status": COMPARISON_STATUS,
        "method": method,
        "teacher_used": teacher_used,
        "teacher_gradient": "stop_gradient" if teacher_used else None,
        "teacher_provenance": (
            (
                "synthetic fixture only; no official teacher provenance"
                if synthetic_fixture
                else "KD teacher snapshot loaded and hash-bound"
            )
            if teacher_used
            else None
        ),
        "teacher_model": (
            None if synthetic_fixture else TEACHER_MODEL
        ) if teacher_used else None,
        "teacher_revision": (
            None if synthetic_fixture else TEACHER_REVISION
        ) if teacher_used else None,
        "input_status": SYNTHETIC_STATUS if synthetic_fixture else "REAL_LIBRISPEECH_PILOT",
        "contract_status": CONTRACT_STATUS,
        "hardware_status": HARDWARE_STATUS,
        "dataset": manifest.get("dataset"),
        "provenance_status": (
            "synthetic_test_fixture" if synthetic_fixture else "prepared_manifest_features_provenance"
        ),
        "data_snapshot": data_snapshot,
        "source_hashes": source_hashes,
        "input_hashes": input_hashes,
        "checkpoint_bindings": checkpoint_bindings or {},
        "config": {
            "seed": SEED,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "max_epochs": max_epochs,
            "patience": PATIENCE,
            "aam_margin": AAM_MARGIN,
            "aam_scale": AAM_SCALE,
            "aam_easy_margin": False,
            "kd_weight": KD_WEIGHT if teacher_used else None,
            "optimizer": "Adam",
            "augmentation": None,
            "encoder_topology": "Conv8/ReLU/AvgPool/Flatten/Dense192",
            "encoder_parameter_count": 146192,
            "classifier_kernel_shape": [EMBEDDING_DIM, NUM_CLASSES],
            "classifier_bias": False,
            "classifier_normalization": "column L2 with PyTorch F.normalize eps=1e-12",
            "strict_local_zero_nonfinite_preflight": True,
        },
        "split_usage": {
            "train_rows": int(protocol["train_indices"].size),
            "validation_rows": int(protocol["validation_indices"].size),
            "train_speaker_ids": protocol["train_speakers"],
            "validation_speaker_ids": protocol["validation_speakers"],
            "validation_enrollment_rows_per_speaker": VALIDATION_ENROLL_ROWS,
            "validation_query_rows_per_speaker": VALIDATION_QUERY_ROWS,
            "representative_rows": int(protocol["train_indices"].size),
            "representative_source": "train only, manifest order",
            "validation_classifier_loss": "forbidden; validation EER only",
            "calibration": None,
            "test": None,
        },
        "training": {
            "history": fit["history"],
            "best_epoch": fit["best_epoch"],
            "best_validation_eer": fit["best_validation_eer"],
            "epochs_ran": fit["epochs_ran"],
            "stop_reason": fit["stop_reason"],
            "weight_update_max_abs": fit["weight_update_max_abs"],
            "checkpoint_selection": "strict minimum validation EER; first tie",
            "balanced_schedule": "20 batches x 20 speakers; every train row once/epoch",
            "teacher_in_supervised_loss": False if not teacher_used else None,
        },
        "architecture": {
            "dynamic_train_input_shape": [None, *FEATURE_SHAPE],
            "fixed_export_input_shape": list(MODEL_INPUT_SHAPE),
            "output_shape": list(MODEL_OUTPUT_SHAPE),
            "dynamic_fixed_batch1_max_abs": export["dynamic_fixed_batch1_max_abs"],
            "dynamic_fixed_batch1_atol": export["dynamic_fixed_batch1_atol"],
        },
        "tflite": {
            "fp32": export["fp32_metadata"],
            "int8": export["int8_metadata"],
            "representative_rows": export["representative_rows"],
            "representative_source": export["representative_source"],
        },
        "int8_vs_fp32_drift": export["int8_vs_fp32_drift"],
        "artifacts": {
            name: _artifact_record(path, stage=stage, final=final)
            for name, path in artifacts_paths.items()
        },
        "evaluation": {
            "calibration": None,
            "test": None,
            "status": "NOT_RUN_BY_COMPARATIVE_TRAINING",
            "downstream_evaluator_call": False,
        },
        "report_path": str(final / ("kd_aam_report.json" if teacher_used else "aam_supervised_report.json")),
    }
    return report


def _build_initial_pair(tf: Any) -> tuple[Any, Any, Any, Any]:
    """Create two exact copies before either optimizer is used."""

    tf.keras.utils.set_random_seed(SEED)
    encoder_kd = _training.build_student_model(name="comparison_kd_train")
    head_kd = _build_classifier_head(tf)
    encoder_initial = [np.asarray(value).copy() for value in encoder_kd.get_weights()]
    head_initial = [np.asarray(value).copy() for value in head_kd.get_weights()]
    encoder_supervised = _training.build_student_model(name="comparison_supervised_train")
    head_supervised = _build_classifier_head(tf)
    encoder_supervised.set_weights(encoder_initial)
    head_supervised.set_weights(head_initial)
    _training._validate_identical_model_weights(encoder_kd, encoder_supervised)
    if not all(
        np.array_equal(left, right)
        for left, right in zip(head_kd.get_weights(), head_supervised.get_weights())
    ):
        raise RuntimeError("KD and supervised classifier initial weights differ")
    return encoder_kd, head_kd, encoder_supervised, head_supervised


def train_comparative_pair(
    manifest_path: str | Path,
    features_path: str | Path,
    teacher_embeddings_path: str | Path,
    output_path: str | Path,
    *,
    synthetic_fixture: bool = False,
    max_epochs: int = MAX_EPOCHS,
    strict_thread_config: bool = True,
) -> dict[str, Any]:
    """Train/export both comparison methods into a new immutable directory.

    ``synthetic_fixture`` is test-only and labels all output as synthetic.  It
    may shorten the epoch count, but it does not relax the locked 20-speaker
    train/5-speaker validation protocol.  The supervised loader reads only
    manifest/features/provenance; the teacher file is loaded by the KD branch.
    """

    manifest_file = Path(manifest_path).resolve()
    features_file = Path(features_path).resolve()
    teacher_file = Path(teacher_embeddings_path).resolve()
    output = Path(output_path).resolve()
    if isinstance(max_epochs, bool) or not isinstance(max_epochs, int) or max_epochs <= 0:
        raise ValueError("max_epochs must be a positive integer")
    if max_epochs > MAX_EPOCHS:
        raise ValueError(f"max_epochs cannot exceed locked maximum {MAX_EPOCHS}")
    if not synthetic_fixture and max_epochs != MAX_EPOCHS:
        raise ValueError("real comparison training must use max_epochs=80")
    if not synthetic_fixture and not strict_thread_config:
        raise ValueError("real comparison training requires strict TensorFlow thread setup")
    for path in (manifest_file, features_file, teacher_file):
        if not path.is_file():
            raise ValueError(f"input artifact is not a regular file: {path}")
    if _path_exists(output):
        raise ValueError(f"refusing to overwrite existing output path: {output}")

    provenance_file = manifest_file.parent / "provenance.json"
    if not synthetic_fixture and not provenance_file.is_file():
        raise ValueError("real pilot manifest requires provenance.json")
    if _path_exists(provenance_file) and not provenance_file.is_file():
        raise ValueError(f"provenance path is not a regular file: {provenance_file}")
    bound_provenance = provenance_file if provenance_file.is_file() else None

    # Capture the file hashes before deserializing.  The later comparison binds
    # the loaded arrays to these exact bytes; an original-path replacement
    # during load cannot become the new expected input by accident.
    source_hashes = _source_hashes()
    input_paths = _input_paths(
        manifest_file,
        features_file,
        teacher_file,
        bound_provenance,
    )
    preload_input_hashes = _input_hashes(
        manifest_file,
        features_file,
        teacher_file,
        provenance_path=bound_provenance,
    )

    # This first load is shared by both methods, but it deliberately does not
    # touch the teacher path.  The standalone supervised method therefore has
    # no opportunity to claim teacher supervision or attribution.
    manifest, rows, manifest_sha256, features, identifiers, provenance = _load_features_only(
        manifest_file, features_file, synthetic_fixture=synthetic_fixture
    )
    protocol = _protocol_indices(rows)
    teacher_embeddings, teacher_sha256 = _load_teacher_only(
        teacher_file, identifiers, manifest_sha256
    )
    input_hashes = _input_hashes(
        manifest_file,
        features_file,
        teacher_file,
        provenance_path=bound_provenance,
    )
    if input_hashes != preload_input_hashes:
        raise RuntimeError("manifest/features/teacher inputs changed while loading")
    if manifest_sha256 != preload_input_hashes["manifest_sha256"]:
        raise RuntimeError("loaded manifest bytes do not match the pre-load hash")
    if teacher_sha256 != preload_input_hashes["teacher_embeddings_sha256"]:
        raise RuntimeError("loaded teacher bytes do not match the pre-load hash")
    source_paths = _source_paths()

    output.parent.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc)
    started_clock = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temp:
        stage = Path(temp)
        tf = _tensorflow()
        thread_report = _training._configure_tf_threads(tf, strict=strict_thread_config)
        np.random.seed(SEED)
        random.seed(SEED)
        encoder_kd, head_kd, encoder_supervised, head_supervised = _build_initial_pair(tf)
        if encoder_kd.count_params() != 146192 or encoder_supervised.count_params() != 146192:
            raise RuntimeError("comparison encoder parameter count changed")
        kd_stage = stage / "kd_aam"
        supervised_stage = stage / "aam_supervised"
        kd_stage.mkdir()
        supervised_stage.mkdir()

        kd_fit = _fit_method(
            tf,
            method="kd_aam",
            encoder=encoder_kd,
            head=head_kd,
            features=features,
            teacher_embeddings=teacher_embeddings,
            rows=rows,
            protocol=protocol,
            stage=kd_stage,
            max_epochs=max_epochs,
        )
        supervised_fit = _fit_method(
            tf,
            method="aam_supervised",
            encoder=encoder_supervised,
            head=head_supervised,
            features=features,
            teacher_embeddings=None,
            rows=rows,
            protocol=protocol,
            stage=supervised_stage,
            max_epochs=max_epochs,
        )

        checkpoint_paths = {
            "kd_aam": _checkpoint_paths(kd_stage),
            "aam_supervised": _checkpoint_paths(supervised_stage),
        }
        checkpoint_bindings = {
            method: _checkpoint_bindings(
                paths["encoder_initial"].parent,
                root_stage=stage,
                final_root=output,
            )
            for method, paths in checkpoint_paths.items()
        }
        paired_initial_weights = {
            "encoder": {
                "kd_aam_sha256": _weight_digest(kd_fit["initial_encoder_weights"]),
                "aam_supervised_sha256": _weight_digest(
                    supervised_fit["initial_encoder_weights"]
                ),
            },
            "classifier": {
                "kd_aam_sha256": _weight_digest(kd_fit["initial_head_weights"]),
                "aam_supervised_sha256": _weight_digest(
                    supervised_fit["initial_head_weights"]
                ),
            },
            "exact": all(
                np.array_equal(left, right)
                for left, right in zip(
                    kd_fit["initial_encoder_weights"],
                    supervised_fit["initial_encoder_weights"],
                )
            )
            and all(
                np.array_equal(left, right)
                for left, right in zip(
                    kd_fit["initial_head_weights"],
                    supervised_fit["initial_head_weights"],
                )
            ),
        }
        if not paired_initial_weights["exact"]:
            raise RuntimeError("KD and supervised initial weights are not exactly paired")

        # Both validation-selected checkpoints and their source/input hashes
        # are frozen before either encoder sees all rows or is converted.
        _assert_stable(source_paths, source_hashes, phase="pre-export")
        _assert_stable(input_paths, input_hashes, phase="pre-export")
        for method, paths in checkpoint_paths.items():
            _assert_checkpoint_stable(
                paths,
                checkpoint_bindings[method],
                method=method,
                phase="pre-export",
            )
        freeze = {
            "format_version": 1,
            "status": "CHECKPOINTS_AND_CONFIG_FROZEN_BEFORE_EXPORT_AND_ALL_ROW_INFERENCE",
            "source_hashes": source_hashes,
            "input_paths": {
                label: str(path) for label, path in input_paths.items()
            },
            "input_hashes": input_hashes,
            "paired_initial_weights": paired_initial_weights,
            "config": {
                "seed": SEED,
                "learning_rate": LEARNING_RATE,
                "batch_size": BATCH_SIZE,
                "max_epochs": max_epochs,
                "patience": PATIENCE,
                "aam_margin": AAM_MARGIN,
                "aam_scale": AAM_SCALE,
                "aam_easy_margin": False,
                "kd_weight": KD_WEIGHT,
                "selection_rule": "strict minimum validation EER; first tie; patience counts non-improvements",
            },
            "methods": {
                "kd_aam": {
                    "best_epoch": kd_fit["best_epoch"],
                    "checkpoints": checkpoint_bindings["kd_aam"],
                },
                "aam_supervised": {
                    "best_epoch": supervised_fit["best_epoch"],
                    "checkpoints": checkpoint_bindings["aam_supervised"],
                },
            },
            "evaluation": {
                "calibration": None,
                "test": None,
                "status": "NOT_RUN",
            },
        }
        _json_write(stage / "comparison_freeze.json", freeze)
        kd_export = _export_encoder(
            tf,
            encoder=encoder_kd,
            features=features,
            identifiers=identifiers,
            manifest_sha256=manifest_sha256,
            train_indices=protocol["train_indices"],
            stage=kd_stage,
        )
        supervised_export = _export_encoder(
            tf,
            encoder=encoder_supervised,
            features=features,
            identifiers=identifiers,
            manifest_sha256=manifest_sha256,
            train_indices=protocol["train_indices"],
            stage=supervised_stage,
        )
        _assert_stable(source_paths, source_hashes, phase="pre-commit")
        _assert_stable(input_paths, input_hashes, phase="pre-commit")
        for method, paths in checkpoint_paths.items():
            _assert_checkpoint_stable(
                paths,
                checkpoint_bindings[method],
                method=method,
                phase="pre-commit",
            )

        kd_input_hashes = dict(input_hashes)
        supervised_input_hashes = {
            "manifest_sha256": input_hashes["manifest_sha256"],
            "features_sha256": input_hashes["features_sha256"],
        }
        if "provenance_sha256" in input_hashes:
            supervised_input_hashes["provenance_sha256"] = input_hashes[
                "provenance_sha256"
            ]
        kd_report = _method_report(
            method="kd_aam",
            fit=kd_fit,
            export=kd_export,
            stage=kd_stage,
            final=output / "kd_aam",
            source_hashes=source_hashes,
            input_hashes=kd_input_hashes,
            manifest_path=manifest_file,
            features_path=features_file,
            manifest=manifest,
            rows=rows,
            protocol=protocol,
            provenance=provenance,
            teacher_used=True,
            synthetic_fixture=synthetic_fixture,
            max_epochs=max_epochs,
            checkpoint_bindings=checkpoint_bindings["kd_aam"],
        )
        supervised_report = _method_report(
            method="aam_supervised",
            fit=supervised_fit,
            export=supervised_export,
            stage=supervised_stage,
            final=output / "aam_supervised",
            source_hashes=source_hashes,
            input_hashes=supervised_input_hashes,
            manifest_path=manifest_file,
            features_path=features_file,
            manifest=manifest,
            rows=rows,
            protocol=protocol,
            provenance=provenance,
            teacher_used=False,
            synthetic_fixture=synthetic_fixture,
            max_epochs=max_epochs,
            checkpoint_bindings=checkpoint_bindings["aam_supervised"],
        )
        kd_report_path = kd_stage / "kd_aam_report.json"
        supervised_report_path = supervised_stage / "aam_supervised_report.json"
        _json_write(kd_report_path, kd_report)
        _json_write(supervised_report_path, supervised_report)

        finished_at = datetime.now(timezone.utc)
        root_data_snapshot: dict[str, Any] = {
            "manifest_path": str(manifest_file),
            "features_path": str(features_file),
            "row_count": len(rows),
            "feature_shape": list(features.shape),
            "split_counts": dict(Counter(row["split"] for row in rows)),
            "source_provenance": provenance,
        }
        if "provenance_sha256" in input_hashes:
            root_data_snapshot["provenance_path"] = str(provenance_file)
            root_data_snapshot["provenance_sha256"] = input_hashes[
                "provenance_sha256"
            ]
        root_report: dict[str, Any] = {
            "format_version": 1,
            "status": COMPARISON_STATUS,
            "contract_status": CONTRACT_STATUS,
            "hardware_status": HARDWARE_STATUS,
            "input_status": SYNTHETIC_STATUS if synthetic_fixture else "REAL_LIBRISPEECH_PILOT",
            "dataset": manifest.get("dataset"),
            "source_hashes": source_hashes,
            "input_paths": {
                label: str(path) for label, path in input_paths.items()
            },
            "input_hashes": input_hashes,
            "teacher_provenance": {
                "kd_teacher_sha256": teacher_sha256,
                "kd_teacher_model": None if synthetic_fixture else TEACHER_MODEL,
                "kd_teacher_revision": None if synthetic_fixture else TEACHER_REVISION,
                "supervised_teacher_used": False,
                "supervised_teacher_attribution": None,
                "published_teacher_model_performance": None,
            },
            "config": freeze["config"],
            "data_snapshot": root_data_snapshot,
            "methods": {
                "kd_aam": {"report_path": str(output / "kd_aam" / "kd_aam_report.json")},
                "aam_supervised": {
                    "report_path": str(output / "aam_supervised" / "aam_supervised_report.json")
                },
            },
            "evaluation": {
                "calibration": None,
                "test": None,
                "status": "NOT_RUN_BY_COMPARATIVE_TRAINING",
                "downstream_evaluator_call": False,
            },
            "tensorflow_thread_policy": thread_report,
            "dependencies": _versions(tf),
            "freeze": {
                "path": str(output / "comparison_freeze.json"),
                "status": freeze["status"],
                "input_paths": freeze["input_paths"],
                "input_hashes": freeze["input_hashes"],
                "paired_initial_weights": freeze["paired_initial_weights"],
                "checkpoint_bindings": {
                    method: value["checkpoints"]
                    for method, value in freeze["methods"].items()
                },
            },
            "publish": {
                "mode": "single-process staged rename",
                "final_path_recheck": True,
                "exclusive_writer_assumption": True,
                "cross_process_race_free_no_overwrite_guarantee": False,
            },
            "timing": {
                "started_at_utc": started_at.isoformat(),
                "finished_at_utc": finished_at.isoformat(),
                "elapsed_seconds": float(time.perf_counter() - started_clock),
            },
            "report_path": str(output / "comparison_training_report.json"),
        }
        _json_write(stage / "comparison_training_report.json", root_report)
        # The output path was checked before loading and is checked again at
        # publication.  This is a single-process/exclusive-writer policy; the
        # rename is not advertised as a cross-process race-free no-overwrite
        # primitive.
        if _path_exists(output):
            raise ValueError(f"refusing to overwrite existing output path at publish: {output}")
        stage.rename(output)
    return root_report


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
        report = train_comparative_pair(
            args.manifest,
            args.features,
            args.teacher_embeddings,
            args.output,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"edgeai.comparative_training: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI integration
    raise SystemExit(main())


__all__ = [
    "AAM_MARGIN",
    "AAM_SCALE",
    "BATCH_SIZE",
    "COMPARISON_STATUS",
    "CONTRACT_STATUS",
    "FEATURE_SHAPE",
    "HARDWARE_STATUS",
    "KD_WEIGHT",
    "MAX_EPOCHS",
    "NUM_CLASSES",
    "PATIENCE",
    "SEED",
    "aam_logits",
    "balanced_epoch_schedule",
    "main",
    "normalized_classifier",
    "select_validation_checkpoint",
    "train_comparative_pair",
    "validation_protocol",
]
