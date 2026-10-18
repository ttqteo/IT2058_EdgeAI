"""Bounded host-only enhancement experiments.

This module is an additive, single-method training pipeline for the local
speaker-embedding experiment.  It intentionally does not change the frozen
pilot/comparison implementations.  The only rows used for optimization,
validation, export, and embedding output are the TRAIN and VALIDATION rows.
Calibration and TEST rows are validated as part of the prepared manifest
boundary, but are never inferred or written here.

The ``conv3_two_bin`` topology and relational term are NEW_LOCAL_DESIGN
ablations.  They are not an upstream IDIR implementation, a final model
contract, or evidence of board support.  A real run still requires the
verified prepared 640-row manifest; a future mixed-source/120-speaker
manifest is deliberately rejected until its independent provenance/schema
boundary is supplied.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
from io import StringIO
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

from . import budget as _budget
from . import comparative_training as _comparative
from . import training as _training
from .frontend import NUM_FRAMES, NUM_MELS, NUM_SAMPLES


SEED = 20261012
LEARNING_RATE = 0.001
BATCH_SIZE = 20
MAX_EPOCHS = 80
PATIENCE = 15
AAM_MARGIN = 0.2
AAM_SCALE = 30.0
EMBEDDING_DIM = 192
FEATURE_SHAPE = (NUM_FRAMES, NUM_MELS, 1)
MODEL_INPUT_SHAPE = (1, *FEATURE_SHAPE)
MODEL_OUTPUT_SHAPE = (1, EMBEDDING_DIM)
VALIDATION_ENROLL_ROWS = 3
VALIDATION_QUERY_ROWS = 9
HARDWARE_STATUS = "UNVERIFIED_HARDWARE"
CONTRACT_STATUS = "NEW_LOCAL_DESIGN"
SYNTHETIC_STATUS = "SYNTHETIC_TEST_FIXTURE"
REAL_STATUS = "REAL_LIBRISPEECH_PILOT"
_NORM_FLOOR = np.float32(1e-12)
_AAM_CLAMP = np.float32(1e-7)
_REAL_SPLIT_COUNTS = {
    "train": 400,
    "validation": 60,
    "calibration_enroll": 15,
    "calibration_query": 45,
    "test_enroll": 30,
    "test_query": 90,
}


# These are intentionally public aliases with the previously frozen signature.
# Independent tests can import them from this module without depending on the
# moving implementation below.
normalized_classifier = _comparative.normalized_classifier
aam_logits = _comparative.aam_logits


def _tensorflow() -> Any:
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("TensorFlow 2.20 is required for enhancement training") from exc
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


def _json_write(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _path_exists(path: Path) -> bool:
    return path.exists() or os.path.lexists(path)


def _finite_nonzero_unit(values: np.ndarray, *, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 2 or raw.shape[1] != EMBEDDING_DIM:
        raise ValueError(f"{name} must have shape [N, {EMBEDDING_DIM}]")
    if not np.issubdtype(raw.dtype, np.number) or np.iscomplexobj(raw):
        raise ValueError(f"{name} must contain real numeric values")
    values64 = raw.astype(np.float64, copy=False)
    if not np.all(np.isfinite(values64)):
        raise ValueError(f"{name} must be finite")
    norms = np.linalg.norm(values64, axis=1)
    if not np.all(np.isfinite(norms)) or np.any(norms <= 0.0):
        raise ValueError(f"{name} must be finite and non-zero")
    return values64 / norms[:, None]


def _numpy_cosine_kd_loss(
    student_embeddings: np.ndarray, teacher_embeddings: np.ndarray
) -> float:
    student = _finite_nonzero_unit(student_embeddings, name="student embeddings")
    teacher = _finite_nonzero_unit(teacher_embeddings, name="teacher embeddings")
    if student.shape != teacher.shape:
        raise ValueError("student and teacher embedding shapes differ")
    result = float(np.mean(1.0 - np.sum(student * teacher, axis=1), dtype=np.float64))
    if not np.isfinite(result):
        raise ValueError("cosine KD loss is non-finite")
    return result


def relational_kd_loss_numpy(
    student_embeddings: np.ndarray,
    teacher_embeddings: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Independent NumPy oracle for the local different-speaker relation term.

    This is deliberately not advertised as the full IDIR objective.  It uses
    only upper-triangular, off-diagonal pairs whose integer train labels
    differ, and computes the mean squared cosine-Gram difference.
    """

    student = _finite_nonzero_unit(student_embeddings, name="student embeddings")
    teacher = _finite_nonzero_unit(teacher_embeddings, name="teacher embeddings")
    if student.shape != teacher.shape:
        raise ValueError("student and teacher embedding shapes differ")
    target = np.asarray(labels)
    if target.ndim != 1 or target.shape[0] != student.shape[0]:
        raise ValueError("labels must have one entry per embedding")
    if not np.issubdtype(target.dtype, np.integer):
        raise ValueError("labels must be integer-valued")
    if target.size < 2:
        raise ValueError("relational KD requires at least two rows")
    row, col = np.triu_indices(target.size, k=1)
    mask = target[row] != target[col]
    if not np.any(mask):
        raise ValueError("relational KD requires a non-empty different-speaker mask")
    student_cosine = np.sum(student[row] * student[col], axis=1)
    teacher_cosine = np.sum(teacher[row] * teacher[col], axis=1)
    differences = student_cosine[mask] - teacher_cosine[mask]
    result = float(np.mean(np.square(differences), dtype=np.float64))
    if not np.isfinite(result):
        raise ValueError("relational KD loss is non-finite")
    return result


def relational_kd_loss(
    tf: Any,
    student_embeddings: Any,
    teacher_embeddings: Any,
    labels: Any,
) -> Any:
    """TensorFlow implementation of the bounded relational KD term."""

    student = _comparative._strict_unit_tensor(
        tf, student_embeddings, name="student embeddings"
    )
    teacher = tf.stop_gradient(
        _comparative._strict_unit_tensor(
            tf, teacher_embeddings, name="teacher embeddings"
        )
    )
    target = tf.convert_to_tensor(labels)
    if target.shape.rank != 1:
        raise ValueError("relational KD labels must have rank 1")
    if not target.dtype.is_integer:
        raise ValueError("relational KD labels must be integer-valued")
    tf.debugging.assert_equal(
        tf.shape(target)[0],
        tf.shape(student)[0],
        message="relational KD labels and embeddings have different batch sizes",
    )
    student_gram = tf.linalg.matmul(student, student, transpose_b=True)
    teacher_gram = tf.linalg.matmul(teacher, teacher, transpose_b=True)
    count = tf.shape(target)[0]
    upper = tf.linalg.band_part(tf.ones((count, count), dtype=tf.bool), 0, -1)
    diagonal = tf.eye(count, dtype=tf.bool)
    different = tf.not_equal(target[:, None], target[None, :])
    mask = upper & ~diagonal & different
    selected = tf.boolean_mask(student_gram - teacher_gram, mask)
    tf.debugging.assert_positive(
        tf.size(selected), message="relational KD mask is empty"
    )
    result = tf.reduce_mean(tf.square(selected))
    tf.debugging.assert_all_finite(result, "relational KD loss must be finite")
    return tf.cast(result, tf.float32)


def enhancement_loss_components(
    tf: Any,
    embeddings: Any,
    labels: Any,
    kernel: Any,
    *,
    teacher_embeddings: Any | None,
    kd_weight: float,
    relational_weight: float,
    supervised: bool,
) -> dict[str, Any]:
    """Return AAM, cosine-KD, relation-KD, and total local loss terms."""

    if supervised and teacher_embeddings is not None:
        raise ValueError("supervised mode must not receive teacher embeddings")
    if supervised and (kd_weight != 0.0 or relational_weight != 0.0):
        raise ValueError("supervised mode requires kd_weight=relational_weight=0")
    if not supervised and teacher_embeddings is None and (
        kd_weight != 0.0 or relational_weight != 0.0
    ):
        raise ValueError("non-zero KD terms require teacher embeddings")
    cosine = normalized_classifier(tf, embeddings, kernel)
    logits = aam_logits(
        tf,
        cosine,
        labels,
        margin=AAM_MARGIN,
        scale=AAM_SCALE,
        easy_margin=False,
    )
    classification = _comparative._aam_cross_entropy(tf, logits, labels)
    kd = None
    relation = None
    if not supervised and teacher_embeddings is not None:
        if kd_weight != 0.0:
            kd = _comparative._cosine_kd_loss(
                tf, embeddings, tf.stop_gradient(teacher_embeddings)
            )
        if relational_weight != 0.0:
            relation = relational_kd_loss(
                tf, embeddings, tf.stop_gradient(teacher_embeddings), labels
            )
    total = classification
    if kd is not None:
        total = total + tf.cast(kd_weight, tf.float32) * kd
    if relation is not None:
        total = total + tf.cast(relational_weight, tf.float32) * relation
    tf.debugging.assert_all_finite(total, "enhancement total loss must be finite")
    return {
        "total": tf.cast(total, tf.float32),
        "aam": tf.cast(classification, tf.float32),
        "cosine_kd": None if kd is None else tf.cast(kd, tf.float32),
        "relational_kd": None if relation is None else tf.cast(relation, tf.float32),
    }


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    try:
        archive = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"could not read NPZ {path}: {exc}") from exc
    with archive:
        if not {"ids", "manifest_sha256"} <= set(archive.files):
            raise ValueError(f"{path} is missing required NPZ arrays")
        return {name: np.asarray(archive[name]).copy() for name in archive.files}


def _unicode_scalar(value: np.ndarray, name: str) -> str:
    if value.ndim != 0 or value.dtype.kind != "U":
        raise ValueError(f"{name} must be a Unicode scalar")
    return str(value.item())


def _load_features(
    manifest_path: Path, features_path: Path, *, synthetic_fixture: bool
) -> tuple[dict[str, Any], list[dict[str, Any]], str, np.ndarray, list[str], dict[str, Any] | None]:
    # The lower-level loader validates row fields, crop arithmetic, duplicate
    # hashes, split leakage, and complete calibration/test partitions.  It is
    # called in generic mode so the future 2460-row design is not accidentally
    # hard-coded into this additive module.  Real acceptance is narrowed below
    # to the currently independently prepared 640-row manifest.
    manifest, rows, manifest_sha256 = _training._load_manifest(
        manifest_path, synthetic_fixture=True
    )
    for row in rows:
        if int(row["input_samples"]) <= 0:
            raise ValueError("manifest PCM rows must have positive input_samples")
    if not synthetic_fixture:
        if manifest.get("dataset") != _training.DATASET_NAME:
            raise ValueError(
                "CONTRACT_UNRESOLVED: real enhancement input must use the verified "
                f"dataset {_training.DATASET_NAME!r}"
            )
        if manifest.get("seed") != SEED:
            raise ValueError("real enhancement manifest seed is not the locked seed")
        counts = dict(Counter(str(row["split"]) for row in rows))
        if counts != _REAL_SPLIT_COUNTS:
            raise ValueError(
                "CONTRACT_UNRESOLVED: only the verified 640-row real manifest is "
                f"accepted; observed split counts={counts!r}"
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
    expected_shape = (len(rows), *FEATURE_SHAPE)
    if features.shape != expected_shape:
        raise ValueError(f"features must have shape {expected_shape}; got {features.shape}")
    if _unicode_scalar(archive["manifest_sha256"], "features manifest_sha256") != manifest_sha256:
        raise ValueError("features manifest_sha256 does not match manifest")
    if not np.all(np.isfinite(features)):
        raise ValueError("features must contain only finite values")
    provenance = _comparative._load_provenance(
        manifest_path,
        features_path,
        manifest_sha256,
        synthetic_fixture=synthetic_fixture,
    )
    return manifest, rows, manifest_sha256, features, identifiers, provenance


def _load_teacher(
    teacher_path: Path, identifiers: list[str], manifest_sha256: str
) -> tuple[np.ndarray, str]:
    return _comparative._load_teacher_only(teacher_path, identifiers, manifest_sha256)


def _protocol_indices(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate generic Sx20 TRAIN and fixed 5x12 VALIDATION partitions."""

    train_by_speaker: dict[str, list[int]] = defaultdict(list)
    validation_by_speaker: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        split = str(row["split"])
        if split == "train":
            train_by_speaker[str(row["speaker_id"])].append(index)
        elif split == "validation":
            validation_by_speaker[str(row["speaker_id"])].append(index)
    train_speakers = sorted(train_by_speaker)
    validation_speakers = sorted(validation_by_speaker)
    if len(train_speakers) < BATCH_SIZE or len(train_speakers) % BATCH_SIZE != 0:
        raise ValueError(
            "TRAIN speaker count must be at least 20 and divisible by 20; "
            f"got {len(train_speakers)}"
        )
    if len(validation_speakers) != 5:
        raise ValueError("validation requires exactly 5 speakers")
    if set(train_speakers) & set(validation_speakers):
        raise ValueError("train and validation speaker sets overlap")
    for speaker in train_speakers:
        if len(train_by_speaker[speaker]) != BATCH_SIZE:
            raise ValueError(f"train speaker {speaker!r} must have exactly 20 rows")
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
    train_indices = np.asarray(
        [index for speaker in train_speakers for index in train_by_speaker[speaker]],
        dtype=np.int64,
    )
    validation_indices = np.asarray(
        [index for speaker in validation_speakers for index in validation_by_speaker[speaker]],
        dtype=np.int64,
    )
    if np.any(labels[train_indices] < 0):
        raise RuntimeError("failed to assign train speaker labels")
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
        "class_count": len(train_speakers),
        "labels": labels,
        "train_indices": train_indices,
        "validation_indices": validation_indices,
        "enrollment_indices": np.asarray(enrollment_indices, dtype=np.int64),
        "query_indices": np.asarray(query_indices, dtype=np.int64),
    }


def validation_protocol(rows: list[dict[str, Any]]) -> dict[str, Any]:
    protocol = _protocol_indices(rows)
    result = dict(protocol)
    result["enrollment_ids"] = [
        rows[index]["utterance_id"] for index in protocol["enrollment_indices"]
    ]
    result["query_ids"] = [
        rows[index]["utterance_id"] for index in protocol["query_indices"]
    ]
    return result


def balanced_epoch_schedule(
    rows: list[dict[str, Any]], *, epoch: int, seed: int = SEED
) -> np.ndarray:
    """Return deterministic [number-of-train-speakers, 20] row indices.

    Speakers are partitioned into groups of 20.  Each group yields 20
    batches, one row per group speaker; groups are concatenated, so a 20-row
    batch always contains 20 distinct speakers and every S*20 TRAIN row is
    used exactly once, including S=120.
    """

    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= 0:
        raise ValueError("epoch must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    protocol = _protocol_indices(rows)
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
    shuffled: dict[str, np.ndarray] = {}
    for speaker in protocol["train_speakers"]:
        values = np.asarray(protocol["train_by_speaker"][speaker], dtype=np.int64).copy()
        rng.shuffle(values)
        shuffled[speaker] = values
    speakers = protocol["train_speakers"][:]
    groups = [speakers[start : start + BATCH_SIZE] for start in range(0, len(speakers), BATCH_SIZE)]
    group_order = np.arange(len(groups), dtype=np.int64)
    rng.shuffle(group_order)
    batches: list[list[int]] = []
    for group_index in group_order.tolist():
        group = groups[group_index][:]
        for offset in range(BATCH_SIZE):
            batch = [int(shuffled[speaker][offset]) for speaker in group]
            rng.shuffle(batch)
            batches.append(batch)
    schedule = np.asarray(batches, dtype=np.int64)
    expected = set(protocol["train_indices"].tolist())
    if schedule.shape != (len(speakers), BATCH_SIZE):
        raise RuntimeError(f"balanced schedule has unexpected shape {schedule.shape}")
    if set(schedule.reshape(-1).tolist()) != expected:
        raise RuntimeError("balanced schedule does not cover TRAIN rows exactly once")
    for batch in schedule:
        batch_speakers = {rows[index]["speaker_id"] for index in batch.tolist()}
        if len(batch_speakers) != BATCH_SIZE:
            raise RuntimeError("balanced batch does not contain 20 distinct speakers")
    return schedule


def _validate_hyperparameters(
    *,
    architecture: str,
    kd_weight: float,
    relational_weight: float,
    supervised: bool,
    max_epochs: int,
) -> tuple[str, float, float, bool, int]:
    if architecture not in {"baseline", "conv3_two_bin"}:
        raise ValueError("architecture must be 'baseline' or 'conv3_two_bin'")
    try:
        kd = float(kd_weight)
        relation = float(relational_weight)
    except (TypeError, ValueError) as exc:
        raise ValueError("KD weights must be numeric") from exc
    if not np.isfinite(kd) or kd < 0.0 or not np.isfinite(relation) or relation < 0.0:
        raise ValueError("KD weights must be finite and non-negative")
    if not isinstance(supervised, (bool, np.bool_)):
        raise ValueError("supervised must be boolean")
    if bool(supervised) and (kd != 0.0 or relation != 0.0):
        raise ValueError("supervised mode requires kd_weight=relational_weight=0")
    if isinstance(max_epochs, bool) or not isinstance(max_epochs, int) or max_epochs <= 0:
        raise ValueError("max_epochs must be a positive integer")
    if max_epochs > MAX_EPOCHS:
        raise ValueError(f"max_epochs cannot exceed locked maximum {MAX_EPOCHS}")
    return architecture, kd, relation, bool(supervised), max_epochs


def _build_encoder(tf: Any, architecture: str, *, batch_size: int | None) -> Any:
    if architecture == "baseline":
        model = _training.build_student_model(
            batch_size=batch_size, name="enhancement_baseline_encoder"
        )
        expected_params = 146192
    elif architecture == "conv3_two_bin":
        if batch_size is None:
            inputs = tf.keras.Input(
                shape=FEATURE_SHAPE, dtype=tf.float32, name="enhancement_mel_input"
            )
        else:
            if isinstance(batch_size, bool) or batch_size <= 0:
                raise ValueError("batch_size must be a positive integer or None")
            inputs = tf.keras.Input(
                batch_shape=(batch_size, *FEATURE_SHAPE),
                dtype=tf.float32,
                name="enhancement_mel_input_fixed",
            )
        x = tf.keras.layers.Conv2D(
            8, (3, 3), strides=(2, 2), padding="same", use_bias=True,
            name="enhancement_conv8",
        )(inputs)
        x = tf.keras.layers.ReLU(name="enhancement_relu8")(x)
        x = tf.keras.layers.AveragePooling2D(
            pool_size=(4, 2), strides=(4, 2), padding="same", name="enhancement_pool4x2"
        )(x)
        x = tf.keras.layers.Conv2D(
            16, (3, 3), strides=(2, 2), padding="same", use_bias=True,
            name="enhancement_conv16",
        )(x)
        x = tf.keras.layers.ReLU(name="enhancement_relu16")(x)
        x = tf.keras.layers.AveragePooling2D(
            pool_size=(2, 1), strides=(2, 1), padding="same", name="enhancement_pool2x1"
        )(x)
        x = tf.keras.layers.Conv2D(
            64, (3, 3), strides=(1, 1), padding="same", use_bias=True,
            name="enhancement_conv64",
        )(x)
        x = tf.keras.layers.ReLU(name="enhancement_relu64")(x)
        x = tf.keras.layers.AveragePooling2D(
            pool_size=(5, 5), strides=(5, 5), padding="valid", name="enhancement_pool5x5"
        )(x)
        x = tf.keras.layers.Flatten(name="enhancement_flatten128")(x)
        outputs = tf.keras.layers.Dense(
            EMBEDDING_DIM, activation=None, name="enhancement_embedding"
        )(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="enhancement_conv3_two_bin")
        expected_params = 35296
    else:  # pragma: no cover - guarded by _validate_hyperparameters
        raise ValueError(f"unknown architecture {architecture!r}")
    if int(model.count_params()) != expected_params:
        raise RuntimeError(
            f"{architecture} parameter count changed: {model.count_params()} != {expected_params}"
        )
    expected_input = (
        (None, *FEATURE_SHAPE) if batch_size is None else (batch_size, *FEATURE_SHAPE)
    )
    expected_output = (
        (None, EMBEDDING_DIM) if batch_size is None else (batch_size, EMBEDDING_DIM)
    )
    if tuple(model.input_shape) != expected_input or tuple(model.output_shape) != expected_output:
        raise RuntimeError(
            f"{architecture} shape changed: input={model.input_shape!r}, output={model.output_shape!r}"
        )
    return model


def build_enhancement_encoder(
    architecture: str = "baseline", *, batch_size: int | None = None
) -> Any:
    """Build one of the two explicitly named local encoder topologies."""

    if architecture not in {"baseline", "conv3_two_bin"}:
        raise ValueError("architecture must be 'baseline' or 'conv3_two_bin'")
    return _build_encoder(_tensorflow(), architecture, batch_size=batch_size)


def _architecture_metadata(architecture: str, model: Any) -> dict[str, Any]:
    if architecture == "baseline":
        topology = "Conv8/ReLU/AvgPool8x4/Flatten/Dense192"
        macs = None
        expected_params = 146192
    else:
        topology = (
            "Conv8(3x3,s2,SAME)/ReLU/AvgPool4x2(s4x2,SAME)/"
            "Conv16(3x3,s2,SAME)/ReLU/AvgPool2x1(s2x1,SAME)/"
            "Conv64(3x3,s1,SAME)/ReLU/AvgPool5x5(s5x5,VALID)/Flatten128/Dense192"
        )
        macs = 809376
        expected_params = 35296
    return {
        "name": architecture,
        "topology": topology,
        "parameter_count": int(model.count_params()),
        "expected_parameter_count": expected_params,
        "conv_dense_macs_estimate": macs,
        "mac_scope": "convolution and dense multiply-accumulates only; not runtime timing/frontend",
        "dynamic_input_shape": [None, *FEATURE_SHAPE],
        "fixed_export_input_shape": list(MODEL_INPUT_SHAPE),
        "output_shape": list(MODEL_OUTPUT_SHAPE),
    }


def _save_classifier(head: Any, path: Path, class_count: int) -> None:
    kernel = np.asarray(head.kernel.numpy(), dtype=np.float32)
    expected = (EMBEDDING_DIM, class_count)
    if kernel.shape != expected or not np.all(np.isfinite(kernel)):
        raise RuntimeError(f"classifier kernel must be finite with shape {expected}")
    np.savez_compressed(path, kernel=kernel)


def _artifact_binding(path: Path, *, stage: Path, final: Path) -> dict[str, Any]:
    return {
        "path": str(final / path.relative_to(stage)),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
    }


def _checkpoint_stable(
    paths: dict[str, Path], bindings: dict[str, dict[str, Any]], *, phase: str
) -> None:
    for name, path in paths.items():
        if not path.is_file():
            raise RuntimeError(f"{phase}: checkpoint missing: {name}")
        observed = (int(path.stat().st_size), _sha256_file(path))
        expected = (int(bindings[name]["bytes"]), str(bindings[name]["sha256"]))
        if observed != expected:
            raise RuntimeError(f"{phase}: checkpoint changed: {name}")


def _source_paths() -> dict[str, Path]:
    package = Path(__file__).resolve().parent
    return {
        "enhancement_training.py": package / "enhancement_training.py",
        "comparative_training.py": package / "comparative_training.py",
        "training.py": package / "training.py",
        "frontend.py": package / "frontend.py",
        "evaluation.py": package / "evaluation.py",
        "model.py": package / "model.py",
        "budget.py": package / "budget.py",
    }


def _hash_paths(paths: dict[str, Path]) -> dict[str, str]:
    return {label: _sha256_file(path) for label, path in paths.items()}


def _assert_stable(paths: dict[str, Path], expected: dict[str, str], *, phase: str) -> None:
    for label, path in paths.items():
        observed = _sha256_file(path)
        if observed != expected[label]:
            raise RuntimeError(f"{phase}: source/input changed: {label}")


def _input_paths(
    manifest_path: Path,
    features_path: Path,
    provenance_path: Path | None,
    teacher_path: Path | None,
) -> dict[str, Path]:
    result: dict[str, Path] = {
        "manifest_sha256": manifest_path,
        "features_sha256": features_path,
    }
    if provenance_path is not None:
        result["provenance_sha256"] = provenance_path
    if teacher_path is not None:
        result["teacher_embeddings_sha256"] = teacher_path
    return result


def _version_report(tf: Any) -> dict[str, str]:
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


def _unit_output(raw: np.ndarray, *, name: str) -> np.ndarray:
    return _finite_nonzero_unit(raw, name=name).astype(np.float32)


def _validation_from_embeddings(
    embeddings: np.ndarray, rows: list[dict[str, Any]], protocol: dict[str, Any]
) -> dict[str, Any]:
    if np.asarray(embeddings).shape != (
        protocol["validation_indices"].size,
        EMBEDDING_DIM,
    ):
        raise ValueError("validation embeddings do not match the validation row count")
    values = _unit_output(embeddings, name="validation embeddings")
    by_index = {
        int(index): values[position]
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
    correct = 0
    total_queries = 0
    for query_index in protocol["query_indices"].tolist():
        query = by_index[query_index].astype(np.float64)
        query_speaker = str(rows[query_index]["speaker_id"])
        speaker_scores: list[tuple[str, float]] = []
        for speaker in protocol["validation_speakers"]:
            score = float(np.dot(query, templates[speaker].astype(np.float64)))
            if not np.isfinite(score):
                raise RuntimeError("validation score is non-finite")
            labels.append(int(query_speaker == speaker))
            scores.append(score)
            speaker_scores.append((speaker, score))
        best_score = max(score for _, score in speaker_scores)
        selected = next(speaker for speaker, score in speaker_scores if score == best_score)
        correct += int(selected == query_speaker)
        total_queries += 1
    from .evaluation import evaluate_scores

    metric = evaluate_scores(labels, scores)
    return {
        "eer": float(metric["eer"]),
        "eer_threshold": float(metric["eer_threshold"]),
        "target_trials": int(metric["target_trials"]),
        "non_target_trials": int(metric["non_target_trials"]),
        "top1": {
            "correct": correct,
            "total": total_queries,
            "accuracy": float(correct / total_queries),
            "scope": "validation closed-set diagnostic only; not model selection",
            "tie_policy": "sorted validation speaker order first maximum",
        },
        "metric_provenance": "local float64 evaluate_scores; strict score > threshold; not bit-exact SpeechBrain",
    }


def _geometry_diagnostics(
    embeddings: np.ndarray,
    rows: list[dict[str, Any]],
    indices: np.ndarray,
    *,
    teacher_embeddings: np.ndarray | None = None,
) -> dict[str, Any]:
    values = _unit_output(embeddings, name="geometry embeddings").astype(np.float64)
    labels = np.asarray([str(rows[index]["speaker_id"]) for index in indices.tolist()])
    row, col = np.triu_indices(values.shape[0], k=1)
    same = labels[row] == labels[col]
    cosine = np.sum(values[row] * values[col], axis=1)
    if not np.any(same) or not np.any(~same):
        raise ValueError("geometry diagnostics require within- and between-speaker pairs")
    centroid = np.mean(values, axis=0)
    centroid_norm = float(np.linalg.norm(centroid))
    if not np.isfinite(centroid_norm):
        raise ValueError("geometry centroid norm is non-finite")
    if centroid_norm > 0.0:
        centroid_unit = centroid / centroid_norm
        centroid_dispersion = float(
            np.mean(1.0 - np.sum(values * centroid_unit[None, :], axis=1), dtype=np.float64)
        )
    else:
        centroid_dispersion = None
    result: dict[str, Any] = {
        "row_count": int(values.shape[0]),
        "within_pairs": int(np.sum(same)),
        "between_pairs": int(np.sum(~same)),
        "within_cosine_mean": float(np.mean(cosine[same], dtype=np.float64)),
        "between_cosine_mean": float(np.mean(cosine[~same], dtype=np.float64)),
        "gap_within_minus_between": float(
            np.mean(cosine[same], dtype=np.float64)
            - np.mean(cosine[~same], dtype=np.float64)
        ),
        "centroid_norm": centroid_norm,
        "dispersion_mean_one_minus_centroid_cosine": centroid_dispersion,
        "scope": "NEW_LOCAL_DIAGNOSTICS; train and validation rows only",
    }
    if teacher_embeddings is not None:
        teacher = _unit_output(teacher_embeddings, name="teacher geometry embeddings").astype(np.float64)
        if teacher.shape != values.shape:
            raise ValueError("teacher geometry shape differs from student geometry")
        result["teacher_alignment_mean"] = float(
            np.mean(np.sum(values * teacher, axis=1), dtype=np.float64)
        )
    else:
        result["teacher_alignment_mean"] = None
    return result


def _export_encoder(
    tf: Any,
    *,
    architecture: str,
    encoder: Any,
    features: np.ndarray,
    selected_indices: np.ndarray,
    train_indices: np.ndarray,
    identifiers: list[str],
    manifest_sha256: str,
    stage: Path,
) -> dict[str, Any]:
    """Export/invoke only the selected TRAIN+VALIDATION rows."""

    fixed = _build_encoder(tf, architecture, batch_size=1)
    fixed.set_weights(encoder.get_weights())
    probe = features[train_indices[:1]]
    parity_max_abs, _, _ = _training._validate_batch1_export_parity(encoder, fixed, probe)
    fp32_content = _training._convert_fp32(tf, fixed)
    int8_content = _training._convert_int8(tf, fixed, features, train_indices)
    fp32_path = stage / "encoder_fp32.tflite"
    int8_path = stage / "encoder_int8.tflite"
    fp32_path.write_bytes(fp32_content)
    int8_path.write_bytes(int8_content)
    budget_report = _budget.check_model(int8_path)
    if not budget_report.get("static_budget_pass", False):
        raise RuntimeError(
            "INT8 export failed the static budget gate: "
            + json.dumps(budget_report, sort_keys=True, allow_nan=False)
        )
    selected_features = features[selected_indices]
    fp32_raw, fp32_embeddings, fp32_metadata = _training._run_tflite(
        fp32_content, selected_features, require_int8=False
    )
    int8_raw, int8_embeddings, int8_metadata = _training._run_tflite(
        int8_content, selected_features, require_int8=True
    )
    selected_ids = [identifiers[index] for index in selected_indices.tolist()]
    fp32_embeddings_path = stage / "encoder_fp32_embeddings.npz"
    int8_embeddings_path = stage / "encoder_int8_embeddings.npz"
    _training._write_embeddings(
        fp32_embeddings_path, selected_ids, fp32_embeddings, manifest_sha256
    )
    _training._write_embeddings(
        int8_embeddings_path, selected_ids, int8_embeddings, manifest_sha256
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
        "fp32_embeddings": fp32_embeddings,
        "int8_embeddings": int8_embeddings,
        "fp32_metadata": fp32_metadata,
        "int8_metadata": int8_metadata,
        "static_budget": budget_report,
        "dynamic_fixed_batch1_max_abs": float(parity_max_abs),
        "dynamic_fixed_batch1_atol": 1e-6,
        "row_count": int(selected_indices.size),
        "row_ids": selected_ids,
        "representative_rows": int(train_indices.size),
        "representative_source": "TRAIN only, manifest order",
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


def _replace_stage_paths(value: object, stage: Path, final: Path) -> object:
    if isinstance(value, str):
        stage_text = str(stage)
        if value == stage_text or value.startswith(stage_text + os.sep):
            return str(final / Path(value).relative_to(stage))
        return value
    if isinstance(value, list):
        return [_replace_stage_paths(item, stage, final) for item in value]
    if isinstance(value, dict):
        return {key: _replace_stage_paths(item, stage, final) for key, item in value.items()}
    return value


def _fit(
    tf: Any,
    *,
    encoder: Any,
    head: Any,
    architecture: str,
    features: np.ndarray,
    teacher_embeddings: np.ndarray | None,
    rows: list[dict[str, Any]],
    protocol: dict[str, Any],
    kd_weight: float,
    relational_weight: float,
    supervised: bool,
    stage: Path,
    max_epochs: int,
) -> dict[str, Any]:
    initial_encoder = [np.asarray(value).copy() for value in encoder.get_weights()]
    initial_head = [np.asarray(value).copy() for value in head.get_weights()]
    encoder_initial_path = stage / "encoder_initial.weights.h5"
    classifier_initial_path = stage / "classifier_initial.npz"
    encoder.save_weights(encoder_initial_path)
    _save_classifier(head, classifier_initial_path, protocol["class_count"])
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
        train_relation = 0.0
        for batch_indices in schedule:
            x_batch = tf.convert_to_tensor(features[batch_indices], dtype=tf.float32)
            y_batch = tf.convert_to_tensor(protocol["labels"][batch_indices], dtype=tf.int32)
            teacher_batch = None
            if not supervised:
                if teacher_embeddings is None:
                    raise RuntimeError("non-supervised enhancement has no teacher snapshot")
                teacher_batch = tf.convert_to_tensor(
                    teacher_embeddings[batch_indices], dtype=tf.float32
                )
            with tf.GradientTape() as tape:
                embeddings = encoder(x_batch, training=True)
                components = enhancement_loss_components(
                    tf,
                    embeddings,
                    y_batch,
                    head.kernel,
                    teacher_embeddings=teacher_batch,
                    kd_weight=kd_weight,
                    relational_weight=relational_weight,
                    supervised=supervised,
                )
            gradients = tape.gradient(components["total"], variables)
            _comparative._finite_gradients(tf, gradients)
            optimizer.apply_gradients(zip(gradients, variables))
            values = {
                key: (0.0 if components[key] is None else float(components[key].numpy()))
                for key in ("total", "aam", "cosine_kd", "relational_kd")
            }
            if not all(np.isfinite(value) for value in values.values()):
                raise RuntimeError("enhancement training loss is non-finite")
            train_total += values["total"]
            train_aam += values["aam"]
            train_kd += values["cosine_kd"]
            train_relation += values["relational_kd"]
        validation_raw = np.asarray(
            encoder(features[protocol["validation_indices"]], training=False).numpy(),
            dtype=np.float32,
        )
        validation = _validation_from_embeddings(
            validation_raw, rows, protocol
        )
        batch_count = int(schedule.shape[0])
        history.append(
            {
                "epoch": epoch,
                "train_total_loss": train_total / batch_count,
                "train_aam_loss": train_aam / batch_count,
                "train_cosine_kd_loss": None if supervised else train_kd / batch_count,
                "train_relational_kd_loss": None if supervised else train_relation / batch_count,
                "validation_eer": validation["eer"],
                "validation_eer_threshold": validation["eer_threshold"],
                "validation_top1": validation["top1"],
                "train_rows_seen": int(schedule.size),
                "batch_count": batch_count,
            }
        )
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
    encoder_best_path = stage / "encoder_best.weights.h5"
    classifier_best_path = stage / "classifier_best.npz"
    encoder.save_weights(encoder_best_path)
    _save_classifier(head, classifier_best_path, protocol["class_count"])
    update_max = max(
        [float(np.max(np.abs(before - after))) for before, after in zip(initial_encoder, best_encoder)]
        + [float(np.max(np.abs(before - after))) for before, after in zip(initial_head, best_head)]
    )
    if not np.isfinite(update_max) or update_max <= 0.0:
        raise RuntimeError("enhancement training produced no finite weight update")
    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_eer": float(best_eer),
        "best_validation": history[best_epoch - 1],
        "epochs_ran": len(history),
        "stop_reason": "early_stopping" if stale >= PATIENCE else "max_epochs",
        "weight_update_max_abs": update_max,
        "initial_encoder_weights": initial_encoder,
        "initial_head_weights": initial_head,
        "best_encoder_weights": best_encoder,
        "best_head_weights": best_head,
        "checkpoint_paths": {
            "encoder_initial": encoder_initial_path,
            "encoder_best": encoder_best_path,
            "classifier_initial": classifier_initial_path,
            "classifier_best": classifier_best_path,
        },
        "architecture": architecture,
    }


def _embedding_path_record(path: Path, *, stage: Path, final: Path) -> dict[str, Any]:
    return {
        "path": str(final / path.relative_to(stage)),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
    }


def train_enhancement(
    manifest_path: str | Path,
    features_path: str | Path,
    output_path: str | Path,
    *,
    teacher_embeddings_path: str | Path | None = None,
    architecture: str = "baseline",
    kd_weight: float = 1.0,
    relational_weight: float = 0.0,
    supervised: bool = False,
    synthetic_fixture: bool = False,
    max_epochs: int = MAX_EPOCHS,
    strict_thread_config: bool = True,
) -> dict[str, Any]:
    """Train, freeze, export, and report one enhancement method.

    ``synthetic_fixture=True`` is test-only.  It permits temporary synthetic
    inputs and shortened epochs; it is never an accuracy or provenance claim.
    In supervised mode ``teacher_embeddings_path`` is deliberately ignored,
    so an inaccessible teacher path cannot be read or attributed.
    """

    architecture, kd_weight, relational_weight, supervised, max_epochs = _validate_hyperparameters(
        architecture=architecture,
        kd_weight=kd_weight,
        relational_weight=relational_weight,
        supervised=supervised,
        max_epochs=max_epochs,
    )
    if not synthetic_fixture and max_epochs != MAX_EPOCHS:
        raise ValueError("real enhancement training must use max_epochs=80")
    if not synthetic_fixture and not strict_thread_config:
        raise ValueError("real enhancement training requires strict thread setup")
    manifest_file = Path(manifest_path)
    features_file = Path(features_path)
    output = Path(output_path)
    teacher_file = None if supervised else (
        None if teacher_embeddings_path is None else Path(teacher_embeddings_path)
    )
    if teacher_file is None and not supervised:
        raise ValueError("non-supervised enhancement requires teacher_embeddings_path")
    for path in (manifest_file, features_file):
        if not path.is_file():
            raise ValueError(f"input artifact is not a regular file: {path}")
    if teacher_file is not None and not teacher_file.is_file():
        raise ValueError(f"teacher artifact is not a regular file: {teacher_file}")
    if _path_exists(output):
        raise ValueError(f"refusing to overwrite existing output path: {output}")

    provenance_file = manifest_file.parent / "provenance.json"
    if not provenance_file.is_file():
        if not synthetic_fixture:
            raise ValueError("real enhancement manifest requires provenance.json")
        provenance_bound = None
    else:
        provenance_bound = provenance_file
    input_paths = _input_paths(manifest_file, features_file, provenance_bound, teacher_file)
    source_paths = _source_paths()
    input_hashes = _hash_paths(input_paths)
    source_hashes = _hash_paths(source_paths)
    manifest, rows, manifest_sha256, features, identifiers, provenance = _load_features(
        manifest_file, features_file, synthetic_fixture=synthetic_fixture
    )
    protocol = _protocol_indices(rows)
    teacher_embeddings: np.ndarray | None = None
    teacher_sha256: str | None = None
    if teacher_file is not None:
        teacher_embeddings, teacher_sha256 = _load_teacher(
            teacher_file, identifiers, manifest_sha256
        )
    _assert_stable(input_paths, input_hashes, phase="preload")
    _assert_stable(source_paths, source_hashes, phase="preload")
    if manifest_sha256 != input_hashes["manifest_sha256"]:
        raise RuntimeError("loaded manifest bytes do not match preload hash")
    if teacher_file is not None and teacher_sha256 != input_hashes["teacher_embeddings_sha256"]:
        raise RuntimeError("loaded teacher bytes do not match preload hash")

    output_parent = output.parent.resolve()
    output_parent.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc)
    started_clock = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output_parent) as temp:
        stage = Path(temp)
        tf = _tensorflow()
        thread_report = _training._configure_tf_threads(tf, strict=strict_thread_config)
        tf.keras.utils.set_random_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)
        encoder = _build_encoder(tf, architecture, batch_size=None)
        head = tf.keras.layers.Dense(
            protocol["class_count"], use_bias=False, activation=None, name="enhancement_aam_classifier"
        )
        head(tf.zeros((1, EMBEDDING_DIM), dtype=tf.float32))
        if tuple(head.kernel.shape) != (EMBEDDING_DIM, protocol["class_count"]):
            raise RuntimeError("enhancement classifier shape changed")
        fit = _fit(
            tf,
            encoder=encoder,
            head=head,
            architecture=architecture,
            features=features,
            teacher_embeddings=teacher_embeddings,
            rows=rows,
            protocol=protocol,
            kd_weight=kd_weight,
            relational_weight=relational_weight,
            supervised=supervised,
            stage=stage,
            max_epochs=max_epochs,
        )
        checkpoint_paths = fit["checkpoint_paths"]
        final_output = output.resolve()
        checkpoint_bindings = {
            name: _artifact_binding(path, stage=stage, final=final_output)
            for name, path in checkpoint_paths.items()
        }
        initial_hashes = {
            name: binding["sha256"]
            for name, binding in checkpoint_bindings.items()
            if "initial" in name
        }
        _assert_stable(source_paths, source_hashes, phase="pre-export")
        _assert_stable(input_paths, input_hashes, phase="pre-export")
        _checkpoint_stable(checkpoint_paths, checkpoint_bindings, phase="pre-export")
        config = {
            "seed": SEED,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "max_epochs": max_epochs,
            "patience": PATIENCE,
            "aam_margin": AAM_MARGIN,
            "aam_scale": AAM_SCALE,
            "aam_easy_margin": False,
            "kd_weight": None if supervised else kd_weight,
            "relational_weight": None if supervised else relational_weight,
            "supervised": supervised,
            "optimizer": "Adam",
            "augmentation": None,
            "selection_rule": "strictly minimum validation EER; earliest tie; patience counts non-improvements",
            "validation_metric": "local float64 evaluate_scores with strict score > threshold; not bit-exact SpeechBrain",
            "input_contract": "PCM16 mono16k prepared frontend features [298,40,1]",
        }
        freeze = {
            "format_version": 1,
            "status": "CHECKPOINTS_AND_CONFIG_FROZEN_BEFORE_EXPORT",
            "source_hashes": source_hashes,
            "input_paths": {label: str(path.resolve()) for label, path in input_paths.items()},
            "input_hashes": input_hashes,
            "initial_checkpoint_hashes": initial_hashes,
            "checkpoint_bindings": checkpoint_bindings,
            "config": config,
        }
        _json_write(stage / "enhancement_freeze.json", freeze)
        selected_indices = np.sort(
            np.concatenate((protocol["train_indices"], protocol["validation_indices"]))
        ).astype(np.int64)
        export = _export_encoder(
            tf,
            architecture=architecture,
            encoder=encoder,
            features=features,
            selected_indices=selected_indices,
            train_indices=protocol["train_indices"],
            identifiers=identifiers,
            manifest_sha256=manifest_sha256,
            stage=stage,
        )
        _assert_stable(source_paths, source_hashes, phase="pre-publish")
        _assert_stable(input_paths, input_hashes, phase="pre-publish")
        _checkpoint_stable(checkpoint_paths, checkpoint_bindings, phase="pre-publish")
        selected_position = {int(index): position for position, index in enumerate(selected_indices.tolist())}
        validation_positions = np.asarray(
            [selected_position[int(index)] for index in protocol["validation_indices"].tolist()],
            dtype=np.int64,
        )
        train_positions = np.asarray(
            [selected_position[int(index)] for index in protocol["train_indices"].tolist()],
            dtype=np.int64,
        )
        teacher_selected = None
        if teacher_embeddings is not None:
            teacher_selected = teacher_embeddings[selected_indices]
        fp32_validation = _validation_from_embeddings(
            export["fp32_embeddings"][validation_positions], rows, protocol
        )
        int8_validation = _validation_from_embeddings(
            export["int8_embeddings"][validation_positions], rows, protocol
        )
        selected_rows = [rows[index] for index in selected_indices.tolist()]
        fp32_geometry = _geometry_diagnostics(
            export["fp32_embeddings"], selected_rows, np.arange(selected_indices.size),
            teacher_embeddings=None if teacher_selected is None else teacher_selected,
        )
        int8_geometry = _geometry_diagnostics(
            export["int8_embeddings"], selected_rows, np.arange(selected_indices.size),
            teacher_embeddings=None if teacher_selected is None else teacher_selected,
        )
        artifact_paths = {
            **checkpoint_paths,
            "encoder_fp32_tflite": export["fp32_path"],
            "encoder_int8_tflite": export["int8_path"],
            "encoder_fp32_embeddings": export["fp32_embeddings_path"],
            "encoder_int8_embeddings": export["int8_embeddings_path"],
            "freeze": stage / "enhancement_freeze.json",
        }
        artifacts = {
            name: _embedding_path_record(path, stage=stage, final=final_output)
            for name, path in artifact_paths.items()
        }
        finished_at = datetime.now(timezone.utc)
        report: dict[str, Any] = {
            "format_version": 1,
            "status": "NEW_LOCAL_ENHANCEMENT_TRAINING",
            "contract_status": CONTRACT_STATUS,
            "hardware_status": HARDWARE_STATUS,
            "input_status": SYNTHETIC_STATUS if synthetic_fixture else REAL_STATUS,
            "dataset": manifest.get("dataset"),
            "architecture": _architecture_metadata(architecture, encoder),
            "config": config,
            "source_hashes": source_hashes,
            "input_paths": {label: str(path.resolve()) for label, path in input_paths.items()},
            "input_hashes": input_hashes,
            "provenance": {
                "status": "synthetic_test_fixture" if synthetic_fixture else "prepared_manifest_features_provenance",
                "sidecar_sha256": input_hashes.get("provenance_sha256"),
                "details": provenance,
            },
            "teacher_provenance": {
                "teacher_used": not supervised,
                "teacher_embeddings_sha256": teacher_sha256 if not supervised else None,
                "teacher_model": None if supervised or synthetic_fixture else _training.TEACHER_MODEL,
                "teacher_revision": None if supervised or synthetic_fixture else _training.TEACHER_REVISION,
                "supervised_teacher_used": False,
                "supervised_teacher_attribution": None,
                "synthetic_teacher_status": "synthetic_fixture_only" if synthetic_fixture and not supervised else None,
            },
            "data_snapshot": {
                "manifest_path": str(manifest_file.resolve()),
                "features_path": str(features_file.resolve()),
                "manifest_sha256": input_hashes["manifest_sha256"],
                "features_sha256": input_hashes["features_sha256"],
                "row_count": len(rows),
                "feature_shape": list(features.shape),
                "split_counts": dict(Counter(str(row["split"]) for row in rows)),
                "train_speaker_count": protocol["class_count"],
                "train_row_ids": [rows[index]["utterance_id"] for index in protocol["train_indices"].tolist()],
                "validation_row_ids": [rows[index]["utterance_id"] for index in protocol["validation_indices"].tolist()],
                "output_row_ids": export["row_ids"],
                "output_row_count": export["row_count"],
                "calibration_row_ids_written": None,
                "test_row_ids_written": None,
            },
            "split_usage": {
                "train_rows": int(protocol["train_indices"].size),
                "validation_rows": int(protocol["validation_indices"].size),
                "representative_rows": int(protocol["train_indices"].size),
                "representative_source": "TRAIN only",
                "validation_enrollment_rows_per_speaker": VALIDATION_ENROLL_ROWS,
                "validation_query_rows_per_speaker": VALIDATION_QUERY_ROWS,
                "validation_classifier_loss": "forbidden",
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
                "checkpoint_selection": config["selection_rule"],
                "teacher_stop_gradient": not supervised,
                "teacher_in_supervised_loss": False,
                "balanced_schedule": "20 distinct speakers per batch, one utterance per speaker, every TRAIN row once per epoch",
            },
            "checkpoints": checkpoint_bindings,
            "exports": {
                "artifacts": artifacts,
                "fp32": {
                    "tflite_metadata": export["fp32_metadata"],
                    "validation": fp32_validation,
                    "geometry": fp32_geometry,
                },
                "int8": {
                    "tflite_metadata": export["int8_metadata"],
                    "validation": int8_validation,
                    "geometry": int8_geometry,
                },
                "batch1_parity_max_abs": export["dynamic_fixed_batch1_max_abs"],
                "batch1_parity_atol": export["dynamic_fixed_batch1_atol"],
                "representative_rows": export["representative_rows"],
                "representative_source": export["representative_source"],
                "int8_vs_fp32_drift": export["int8_vs_fp32_drift"],
                "static_budget": export["static_budget"],
            },
            "evaluation": {
                "calibration": None,
                "test": None,
                "status": "NOT_RUN_BY_ENHANCEMENT_TRAINING",
                "downstream_evaluator_call": False,
                "validation_metric_note": "validation-only local float64 diagnostic; no final test/calibration score",
            },
            "tflm_runtime": {
                "status": "NOT_RUN_BY_ENHANCEMENT_TRAINING",
                "hardware_status": HARDWARE_STATUS,
                "runner_support_for_new_architecture": "UNVERIFIED",
            },
            "tensorflow_thread_policy": thread_report,
            "dependencies": _version_report(tf),
            "publish": {
                "mode": "single-process staged directory rename",
                "final_path_recheck": True,
                "exclusive_writer_assumption": True,
                "cross_process_race_free_no_overwrite_guarantee": False,
            },
            "timing": {
                "started_at_utc": started_at.isoformat(),
                "finished_at_utc": finished_at.isoformat(),
                "elapsed_seconds": float(time.perf_counter() - started_clock),
            },
            "report_path": str(final_output / "enhancement_training_report.json"),
        }
        report_for_write = _replace_stage_paths(report, stage, final_output)
        _json_write(stage / "enhancement_training_report.json", report_for_write)
        if _path_exists(final_output):
            raise ValueError(
                "refusing to overwrite output path at publish; exclusive-writer "
                "assumption is required"
            )
        stage.rename(final_output)
    return report_for_write  # type: ignore[return-value]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--teacher-embeddings", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--architecture", choices=("baseline", "conv3_two_bin"), default="baseline")
    parser.add_argument("--kd-weight", type=float, default=1.0)
    parser.add_argument("--relational-weight", type=float, default=0.0)
    parser.add_argument("--supervised", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--synthetic-fixture", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        report = train_enhancement(
            args.manifest,
            args.features,
            args.output,
            teacher_embeddings_path=args.teacher_embeddings,
            architecture=args.architecture,
            kd_weight=args.kd_weight,
            relational_weight=args.relational_weight,
            supervised=args.supervised,
            synthetic_fixture=args.synthetic_fixture,
            max_epochs=args.max_epochs,
            strict_thread_config=not args.synthetic_fixture,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"edgeai.enhancement_training: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI integration
    raise SystemExit(main())


__all__ = [
    "AAM_MARGIN",
    "AAM_SCALE",
    "BATCH_SIZE",
    "FEATURE_SHAPE",
    "HARDWARE_STATUS",
    "MAX_EPOCHS",
    "PATIENCE",
    "SEED",
    "aam_logits",
    "balanced_epoch_schedule",
    "build_enhancement_encoder",
    "enhancement_loss_components",
    "main",
    "normalized_classifier",
    "relational_kd_loss",
    "relational_kd_loss_numpy",
    "train_enhancement",
    "validation_protocol",
]

