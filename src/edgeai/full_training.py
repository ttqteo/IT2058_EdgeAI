"""Full train-clean-100 baseline training and bounded host export.

This module is deliberately additive.  It consumes the public
``full_data.FullDataset`` and ``full_teacher.load_full_teacher`` boundaries,
keeps the original 60-row validation set separate, and does not call any
calibration or test evaluator.  The real run is train-only: every train row
is visited exactly once per epoch, even when speaker window counts differ.

The public trainers publish only after a durable sibling stage has reached a
fully bound, exported state.  An interrupted stage is never silently reused;
the caller must pass ``resume_from`` (the CLI's ``--resume`` selects the
expected sibling stage explicitly).
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
from importlib import metadata as importlib_metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import tempfile
import time
from typing import Any, Literal

import numpy as np

from . import budget as _budget
from . import comparative_training as _comparative
from . import full_data as _full_data
from . import training as _training


SEED = 20261012
LEARNING_RATE = 1e-3
BATCH_SIZE = 20
ADAM_BETA_1 = 0.9
ADAM_BETA_2 = 0.999
ADAM_EPSILON = 1e-7
MAX_EPOCHS = 80
PATIENCE = 15
AAM_MARGIN = 0.2
AAM_SCALE = 30.0
KD_WEIGHT = 1.0
EMBEDDING_DIM = 192
NUM_CLASSES = 251
FEATURE_SHAPE = (298, 40, 1)
WINDOW_SAMPLES = 48_000
VALIDATION_ROWS = 60
VALIDATION_SPEAKERS = 5
VALIDATION_ENROLL_ROWS = 3
VALIDATION_QUERY_ROWS = 9
EXPECTED_TRAIN_ROWS = 135_438
PROGRESS_BATCH_INTERVAL = 256
SAMPLER_VERSION = "interleaved-active-speaker-round-robin-v1"
REPORT_FILENAME = "full_training_report.json"
STATE_FILENAME = "full_training_state.json"
CONFIG_FILENAME = "full_training_config.json"
HISTORY_FILENAME = "history.json"
FREEZE_FILENAME = "full_training_freeze.json"
STAGE_SUFFIX = ".stage"
METHODS = frozenset({"supervised_aam", "kd_aam"})


class FullTrainingError(RuntimeError):
    """Raised when the full-training contract cannot be satisfied."""


@dataclass(frozen=True)
class ValidationDataset:
    """The independent, original 60-row validation view.

    ``features`` is intentionally only the 60 validation rows.  The source
    prepared bundle may contain historical train/calibration/test rows, but
    this object exposes none of them to the trainer.
    """

    root: Path
    manifest: dict[str, Any]
    rows: list[dict[str, Any]]
    features: np.ndarray
    manifest_sha256: str
    features_sha256: str
    provenance: dict[str, Any]


def _tensorflow() -> Any:
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise FullTrainingError(
            "TensorFlow 2.20 is required for full training/export"
        ) from exc
    return tf


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise FullTrainingError("provenance/config value is not JSON-safe") from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise FullTrainingError(f"expected a regular non-symlink file: {path}")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise FullTrainingError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _sha256_array_rows(values: np.ndarray, *, chunk_rows: int = 256) -> str:
    """Hash an ndarray/mmap in bounded row chunks without making a full copy."""

    array = np.asarray(values)
    digest = hashlib.sha256()
    for start in range(0, int(array.shape[0]), chunk_rows):
        chunk = np.ascontiguousarray(array[start : start + chunk_rows])
        digest.update(chunk.tobytes(order="C"))
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise FullTrainingError("report contains a non-finite float")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        _json_safe(value),
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.{os.getpid()}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FullTrainingError(f"{label} is not a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FullTrainingError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise FullTrainingError(f"{label} must be a JSON object")
    return value


def _path_exists(path: Path) -> bool:
    return path.exists() or os.path.lexists(path)


def _validate_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise FullTrainingError(f"{label} must be a SHA256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise FullTrainingError(f"{label} must be a SHA256 hex digest") from exc
    return value.lower()


def _unicode_scalar(value: Any, label: str) -> str:
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in {"U", "S"}:
        raise FullTrainingError(f"{label} must be a scalar string")
    return str(array.item())


def load_validation_dataset(
    root: str | Path,
    *,
    verify_hashes: bool = True,
    test_fixture: bool = False,
) -> ValidationDataset:
    """Load only the original prepared 60-row validation view.

    The loader validates the complete original manifest/provenance boundary,
    then selects only ``split == validation`` features.  It never exposes or
    infers calibration/test rows.
    """

    validation_root = Path(root).expanduser().resolve()
    if validation_root.is_symlink() or not validation_root.is_dir():
        raise FullTrainingError(
            f"validation root is not a regular directory: {validation_root}"
        )
    manifest_path = validation_root / "manifest.json"
    features_path = validation_root / "features.npz"
    provenance_path = validation_root / "provenance.json"
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes.decode("utf-8"))
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FullTrainingError(
            f"cannot read original validation metadata from {validation_root}"
        ) from exc
    if not isinstance(manifest, dict) or not isinstance(provenance, dict):
        raise FullTrainingError("validation manifest/provenance must be objects")
    if set(manifest) != {"format_version", "dataset", "seed", "rows"}:
        raise FullTrainingError("original validation manifest fields are not exact")
    if manifest.get("format_version") != 1 or manifest.get("seed") != SEED:
        raise FullTrainingError("original validation manifest format/seed mismatch")
    expected_dataset = "synthetic-librispeech-dev-clean-pilot" if test_fixture else "librispeech-dev-clean-pilot"
    if manifest.get("dataset") != expected_dataset:
        raise FullTrainingError("validation dataset identity mismatch")
    if provenance.get("dataset") != expected_dataset:
        raise FullTrainingError("validation provenance dataset identity mismatch")
    manifest_sha256 = _sha256_bytes(manifest_bytes)
    rows = manifest.get("rows")
    if not isinstance(rows, list) or not rows:
        raise FullTrainingError("validation manifest rows must be a non-empty list")
    required_row_fields = {
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
    identifiers: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not required_row_fields <= set(row):
            raise FullTrainingError(f"validation manifest row {index} is malformed")
        identifier = row["utterance_id"]
        if not isinstance(identifier, str) or not identifier:
            raise FullTrainingError(f"validation row {index} has invalid utterance_id")
        identifiers.append(identifier)
        if not isinstance(row["speaker_id"], str) or not row["speaker_id"]:
            raise FullTrainingError(f"validation row {index} has invalid speaker_id")
    if len(set(identifiers)) != len(identifiers):
        raise FullTrainingError("validation manifest utterance IDs are not unique")
    counts = {split: sum(row["split"] == split for row in rows) for split in {
        "train",
        "validation",
        "calibration_enroll",
        "calibration_query",
        "test_enroll",
        "test_query",
    }}
    if not test_fixture and counts != {
        "train": 400,
        "validation": 60,
        "calibration_enroll": 15,
        "calibration_query": 45,
        "test_enroll": 30,
        "test_query": 90,
    }:
        raise FullTrainingError(f"original validation split counts changed: {counts}")
    selected_positions = [index for index, row in enumerate(rows) if row["split"] == "validation"]
    selected_rows = [dict(rows[index]) for index in selected_positions]
    if len(selected_rows) != VALIDATION_ROWS:
        raise FullTrainingError("validation must contain exactly 60 rows")
    speakers: dict[str, int] = {}
    for row in selected_rows:
        speakers[str(row["speaker_id"])] = speakers.get(str(row["speaker_id"]), 0) + 1
    if len(speakers) != VALIDATION_SPEAKERS or set(speakers.values()) != {12}:
        raise FullTrainingError("validation must contain 5 speakers with 12 rows each")
    if verify_hashes:
        for row in selected_rows:
            wav = (validation_root / row["wav_path"]).resolve()
            if not wav.is_relative_to(validation_root):
                raise FullTrainingError("validation WAV escapes prepared root")
            if _sha256_file(wav) != row["wav_sha256"]:
                raise FullTrainingError("validation WAV hash mismatch")
    if verify_hashes:
        features_sha256 = _sha256_file(features_path)
    else:
        features_sha256 = "UNVERIFIED"
    try:
        archive = np.load(features_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise FullTrainingError(f"cannot read validation features: {features_path}") from exc
    with archive:
        if set(archive.files) != {"ids", "features", "manifest_sha256"}:
            raise FullTrainingError("validation feature archive fields are not exact")
        archive_ids = np.asarray(archive["ids"])
        if archive_ids.ndim != 1 or archive_ids.dtype.kind not in {"U", "S"}:
            raise FullTrainingError("validation feature IDs are invalid")
        if archive_ids.tolist() != identifiers:
            raise FullTrainingError("validation feature IDs do not match manifest order")
        archive_manifest_sha256 = _unicode_scalar(
            archive["manifest_sha256"], "validation feature manifest_sha256"
        )
        if archive_manifest_sha256 != manifest_sha256:
            raise FullTrainingError("validation feature manifest hash mismatch")
        features_all = archive["features"]
        expected_shape = (len(rows), *FEATURE_SHAPE)
        if features_all.dtype != np.dtype(np.float32) or features_all.shape != expected_shape:
            raise FullTrainingError(
                f"validation feature shape/dtype mismatch: {features_all.shape} {features_all.dtype}"
            )
        features = np.asarray(features_all[selected_positions], dtype=np.float32).copy()
    if not np.all(np.isfinite(features)):
        raise FullTrainingError("validation features contain non-finite values")
    features.flags.writeable = False
    artifacts = provenance.get("artifacts")
    if not isinstance(artifacts, dict):
        raise FullTrainingError("validation provenance lacks artifacts binding")
    if artifacts.get("manifest_sha256") != manifest_sha256:
        raise FullTrainingError("validation provenance manifest hash mismatch")
    if verify_hashes and artifacts.get("features_sha256") != features_sha256:
        raise FullTrainingError("validation provenance feature hash mismatch")
    if artifacts.get("row_count") != len(rows):
        raise FullTrainingError("validation provenance row count mismatch")
    return ValidationDataset(
        root=validation_root,
        manifest=manifest,
        rows=selected_rows,
        features=features,
        manifest_sha256=manifest_sha256,
        features_sha256=features_sha256,
        provenance=provenance,
    )


def _validate_full_dataset(dataset: Any) -> tuple[dict[str, np.ndarray], list[str]]:
    if not isinstance(dataset, _full_data.FullDataset):
        raise FullTrainingError("dataset must be a validated full_data.FullDataset")
    rows = dataset.rows
    if not isinstance(rows, list) or not rows:
        raise FullTrainingError("full dataset rows are empty")
    if dataset.features.dtype != np.dtype(np.float32) or dataset.features.ndim != 4:
        raise FullTrainingError("full dataset features must be float32 rank-4")
    if dataset.features.shape[1:] != FEATURE_SHAPE:
        raise FullTrainingError("full dataset feature shape mismatch")
    if dataset.pcm.dtype != np.dtype("<i2") or dataset.pcm.shape != (len(rows), WINDOW_SAMPLES):
        raise FullTrainingError("full dataset PCM mmap shape/dtype mismatch")
    if bool(dataset.features.flags.writeable) or bool(dataset.pcm.flags.writeable):
        raise FullTrainingError("full dataset arrays must be read-only")
    if len(rows) != int(dataset.features.shape[0]) or len(rows) != int(dataset.pcm.shape[0]):
        raise FullTrainingError("full dataset row/array counts differ")
    if not _validate_sha256(dataset.manifest_sha256, "dataset manifest_sha256"):
        raise AssertionError("unreachable")
    by_speaker: dict[str, list[int]] = {}
    seen_windows: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("split") != "train":
            raise FullTrainingError("full dataset is not train-only")
        if row.get("global_index") != index:
            raise FullTrainingError("full dataset global_index is not contiguous")
        speaker = row.get("speaker_id")
        window_id = row.get("window_id")
        if not isinstance(speaker, str) or not speaker:
            raise FullTrainingError(f"full row {index} has invalid speaker_id")
        if not isinstance(window_id, str) or not window_id or window_id in seen_windows:
            raise FullTrainingError(f"full row {index} has invalid/duplicate window_id")
        seen_windows.add(window_id)
        by_speaker.setdefault(speaker, []).append(index)
    fixture = str(dataset.manifest.get("dataset", "")).startswith("synthetic")
    if not fixture:
        if len(rows) != EXPECTED_TRAIN_ROWS:
            raise FullTrainingError(
                f"full train row count changed: expected {EXPECTED_TRAIN_ROWS}, got {len(rows)}"
            )
        if len(by_speaker) != NUM_CLASSES:
            raise FullTrainingError(
                f"full train speaker count changed: expected {NUM_CLASSES}, got {len(by_speaker)}"
            )
    for start in range(0, len(rows), 256):
        if not np.all(np.isfinite(np.asarray(dataset.features[start : start + 256]))):
            raise FullTrainingError(f"full feature mmap contains non-finite values near row {start}")
    return {speaker: np.asarray(indices, dtype=np.int64) for speaker, indices in sorted(by_speaker.items())}, sorted(by_speaker)


def _validation_protocol(
    validation: ValidationDataset,
    train_speakers: Sequence[str],
) -> dict[str, Any]:
    if not isinstance(validation, ValidationDataset):
        raise FullTrainingError("validation must be a ValidationDataset")
    if len(validation.rows) != VALIDATION_ROWS:
        raise FullTrainingError("validation must contain exactly 60 rows")
    if validation.features.dtype != np.dtype(np.float32) or validation.features.shape != (
        VALIDATION_ROWS,
        *FEATURE_SHAPE,
    ):
        raise FullTrainingError("validation feature view shape/dtype mismatch")
    if bool(validation.features.flags.writeable):
        raise FullTrainingError("validation features must be read-only")
    validation_by_speaker: dict[str, list[int]] = {}
    seen_ids: set[str] = set()
    for index, row in enumerate(validation.rows):
        if row.get("split") != "validation":
            raise FullTrainingError("validation view contains a non-validation row")
        utterance_id = row.get("utterance_id")
        speaker = row.get("speaker_id")
        if not isinstance(utterance_id, str) or not utterance_id or utterance_id in seen_ids:
            raise FullTrainingError("validation utterance IDs are invalid/duplicate")
        if not isinstance(speaker, str) or not speaker:
            raise FullTrainingError("validation speaker IDs are invalid")
        seen_ids.add(utterance_id)
        validation_by_speaker.setdefault(speaker, []).append(index)
    validation_speakers = sorted(validation_by_speaker)
    if len(validation_speakers) != VALIDATION_SPEAKERS:
        raise FullTrainingError("validation must contain exactly 5 speakers")
    if set(train_speakers) & set(validation_speakers):
        raise FullTrainingError("train and validation speakers overlap")
    for speaker in validation_speakers:
        validation_by_speaker[speaker].sort(
            key=lambda index: str(validation.rows[index]["utterance_id"])
        )
        if len(validation_by_speaker[speaker]) != VALIDATION_ENROLL_ROWS + VALIDATION_QUERY_ROWS:
            raise FullTrainingError(f"validation speaker {speaker!r} must have 12 rows")
    validation_indices = np.asarray(
        [index for speaker in validation_speakers for index in validation_by_speaker[speaker]],
        dtype=np.int64,
    )
    enrollment_indices = np.asarray(
        [
            index
            for speaker in validation_speakers
            for index in validation_by_speaker[speaker][:VALIDATION_ENROLL_ROWS]
        ],
        dtype=np.int64,
    )
    query_indices = np.asarray(
        [
            index
            for speaker in validation_speakers
            for index in validation_by_speaker[speaker][VALIDATION_ENROLL_ROWS:]
        ],
        dtype=np.int64,
    )
    return {
        "validation_by_speaker": {key: value[:] for key, value in validation_by_speaker.items()},
        "validation_speakers": validation_speakers,
        "validation_indices": validation_indices,
        "enrollment_indices": enrollment_indices,
        "query_indices": query_indices,
    }


def _training_protocol(
    dataset: _full_data.FullDataset,
    validation: ValidationDataset,
) -> dict[str, Any]:
    by_speaker, train_speakers = _validate_full_dataset(dataset)
    validation_protocol = _validation_protocol(validation, train_speakers)
    labels = np.full(len(dataset.rows), -1, dtype=np.int32)
    for label, speaker in enumerate(train_speakers):
        labels[by_speaker[speaker]] = label
    train_indices = np.arange(len(dataset.rows), dtype=np.int64)
    if np.any(labels < 0):
        raise FullTrainingError("failed to assign every train speaker label")
    result: dict[str, Any] = {
        "train_by_speaker": by_speaker,
        "train_speakers": train_speakers,
        "class_count": len(train_speakers),
        "labels": labels,
        "train_indices": train_indices,
    }
    result.update(validation_protocol)
    return result


def interleaved_epoch_batches(
    dataset: _full_data.FullDataset | Mapping[str, np.ndarray],
    *,
    epoch: int,
    seed: int = SEED,
    batch_size: int = BATCH_SIZE,
) -> tuple[np.ndarray, ...]:
    """Return deterministic variable-count, no-replacement train batches.

    The active-speaker cursor emits one row per active speaker before a
    speaker can receive a second row in the same batch.  Once fewer than
    ``batch_size`` speakers remain active, a batch may contain repeated
    speakers, but never repeated rows.  A final partial batch is preserved.
    """

    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("epoch must be a non-negative integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if isinstance(dataset, _full_data.FullDataset):
        by_speaker, _ = _validate_full_dataset(dataset)
        expected_indices = set(range(len(dataset.rows)))
    elif isinstance(dataset, Mapping):
        by_speaker = {}
        expected_indices: set[int] = set()
        for speaker, values in sorted(dataset.items(), key=lambda item: str(item[0])):
            array = np.asarray(values)
            if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
                raise ValueError("speaker index arrays must be one-dimensional integers")
            if array.size == 0:
                raise ValueError("speaker index arrays cannot be empty")
            normalized = np.asarray(array, dtype=np.int64)
            if len(set(normalized.tolist())) != normalized.size:
                raise ValueError("speaker index arrays cannot contain duplicate rows")
            if expected_indices.intersection(normalized.tolist()):
                raise ValueError("speaker index arrays overlap")
            expected_indices.update(int(value) for value in normalized.tolist())
            by_speaker[str(speaker)] = normalized
        if not by_speaker:
            raise ValueError("speaker mapping cannot be empty")
    else:
        raise TypeError("dataset must be FullDataset or speaker-to-indices mapping")
    speakers = sorted(by_speaker)
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
    speaker_order = np.arange(len(speakers), dtype=np.int64)
    rng.shuffle(speaker_order)
    active: list[list[Any]] = []
    for speaker_index in speaker_order.tolist():
        values = np.asarray(by_speaker[speakers[int(speaker_index)]], dtype=np.int64).copy()
        rng.shuffle(values)
        active.append([values, 0])
    batches: list[np.ndarray] = []
    cursor = 0
    while active:
        batch: list[int] = []
        while len(batch) < batch_size and active:
            values, offset = active[cursor]
            batch.append(int(values[offset]))
            offset += 1
            if offset >= values.size:
                active.pop(cursor)
                if active:
                    cursor %= len(active)
            else:
                active[cursor][1] = offset
                cursor = (cursor + 1) % len(active)
        batches.append(np.asarray(batch, dtype=np.int64))
    flattened = np.concatenate(batches) if batches else np.empty(0, dtype=np.int64)
    if flattened.size != len(expected_indices) or set(flattened.tolist()) != expected_indices:
        raise FullTrainingError("epoch schedule does not cover every train row exactly once")
    if np.unique(flattened).size != flattened.size:
        raise FullTrainingError("epoch schedule contains a repeated train row")
    return tuple(batches)


def _schedule_digest(batches: Sequence[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for batch in batches:
        values = np.asarray(batch, dtype="<i8")
        digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _validate_teacher_values(teacher: Any, expected_rows: int) -> np.ndarray:
    values = np.asarray(teacher)
    if not isinstance(teacher, np.memmap):
        raise FullTrainingError("teacher loader must return a read-only mmap")
    if values.dtype != np.dtype(np.float32) or values.shape != (expected_rows, EMBEDDING_DIM):
        raise FullTrainingError("teacher shape/dtype does not match full train rows")
    if bool(values.flags.writeable):
        raise FullTrainingError("teacher mmap must be read-only")
    for start in range(0, expected_rows, 256):
        chunk = np.asarray(values[start : start + 256])
        if not np.all(np.isfinite(chunk)):
            raise FullTrainingError(f"teacher contains non-finite values near row {start}")
        norms = np.linalg.norm(chunk.astype(np.float64), axis=1)
        if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
            raise FullTrainingError(f"teacher contains zero/non-finite norms near row {start}")
        if np.any(np.abs(norms - 1.0) > 1e-5):
            raise FullTrainingError(
                f"teacher norm exceeds strict 1e-5 tolerance near row {start}"
            )
    return teacher


def _teacher_inputs(
    teacher_output_dir: Path,
    dataset: _full_data.FullDataset,
) -> tuple[np.ndarray, dict[str, Any], str]:
    from . import full_teacher as _full_teacher

    provenance_function = getattr(_full_teacher, "teacher_provenance", None)
    if not callable(provenance_function):
        raise FullTrainingError(
            "full_teacher.teacher_provenance is required before KD training"
        )
    try:
        provenance = provenance_function(teacher_output_dir, dataset)
    except Exception as exc:
        raise FullTrainingError("teacher provenance validation failed") from exc
    if not isinstance(provenance, dict):
        raise FullTrainingError("teacher_provenance must return a JSON object")
    try:
        _canonical_bytes(provenance)
    except FullTrainingError:
        raise
    try:
        teacher = _full_teacher.load_full_teacher(teacher_output_dir, dataset)
    except Exception as exc:
        raise FullTrainingError("full teacher loader rejected the dataset binding") from exc
    teacher = _validate_teacher_values(teacher, len(dataset.rows))
    teacher_hash = _sha256_array_rows(teacher)
    return teacher, _json_safe(provenance), teacher_hash


def _source_paths(method: str) -> dict[str, Path]:
    package = Path(__file__).resolve().parent
    names = [
        "full_training.py",
        "full_data.py",
        "training.py",
        "comparative_training.py",
        "evaluation.py",
        "budget.py",
        "model.py",
        "frontend.py",
    ]
    if method == "kd_aam":
        names.append("full_teacher.py")
    return {name: package / name for name in names}


def _source_bindings(method: str) -> dict[str, str]:
    return {name: _sha256_file(path) for name, path in _source_paths(method).items()}


def _dataset_binding(dataset: _full_data.FullDataset) -> dict[str, Any]:
    return {
        "root": str(dataset.root),
        "manifest_sha256": dataset.manifest_sha256,
        "manifest_content_sha256": _sha256_bytes(_canonical_bytes(dataset.manifest)),
        "row_count": len(dataset.rows),
        "feature_shape": list(dataset.features.shape),
        "pcm_shape": list(dataset.pcm.shape),
        "verification": "full_data.load_full_dataset(verify_hashes=True)",
    }


def _validation_binding(validation: ValidationDataset) -> dict[str, Any]:
    return {
        "root": str(validation.root),
        "manifest_sha256": validation.manifest_sha256,
        "features_sha256": validation.features_sha256,
        "row_count": len(validation.rows),
        "row_ids_sha256": _sha256_bytes(
            _canonical_bytes([row["utterance_id"] for row in validation.rows])
        ),
        "provenance_sha256": _sha256_bytes(_canonical_bytes(validation.provenance)),
        "verification": "load_validation_dataset(verify_hashes=True); validation split only",
    }


def _current_input_bindings(
    dataset: _full_data.FullDataset,
    validation: ValidationDataset,
    protocol: dict[str, Any],
    *,
    method: str,
    teacher_output_dir: Path | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "dataset": _dataset_binding(dataset),
        "validation": _validation_binding(validation),
        "train_speakers_sha256": _sha256_bytes(_canonical_bytes(protocol["train_speakers"])),
        "validation_speakers_sha256": _sha256_bytes(
            _canonical_bytes(protocol["validation_speakers"])
        ),
    }
    if method == "kd_aam":
        if teacher_output_dir is None:
            raise FullTrainingError("KD requires a teacher output directory")
        teacher, provenance, teacher_hash = _teacher_inputs(teacher_output_dir, dataset)
        del teacher
        result["teacher"] = {
            "output_dir": str(teacher_output_dir),
            "provenance": provenance,
            "provenance_sha256": _sha256_bytes(_canonical_bytes(provenance)),
            "embeddings_sha256": teacher_hash,
            "shape": [len(dataset.rows), EMBEDDING_DIM],
            "norm_tolerance": 1e-5,
        }
    return result


def _reload_and_bind_inputs(
    dataset: _full_data.FullDataset,
    validation: ValidationDataset,
    *,
    method: str,
    teacher_output_dir: Path | None,
) -> dict[str, Any]:
    """Re-run public validators before export/resume to catch file mutation."""

    fixture = str(dataset.manifest.get("dataset", "")).startswith("synthetic")
    if dataset.root.is_dir() and (dataset.root / "manifest.json").is_file():
        reloaded_dataset = _full_data.load_full_dataset(
            dataset.root,
            verify_hashes=True,
            test_fixture=fixture,
        )
    else:
        reloaded_dataset = dataset
    validation_fixture = str(validation.manifest.get("dataset", "")).startswith("synthetic")
    if validation.root.is_dir() and (validation.root / "manifest.json").is_file():
        reloaded_validation = load_validation_dataset(
            validation.root,
            verify_hashes=True,
            test_fixture=validation_fixture,
        )
    else:
        reloaded_validation = validation
    protocol = _training_protocol(reloaded_dataset, reloaded_validation)
    bindings = _current_input_bindings(
        reloaded_dataset,
        reloaded_validation,
        protocol,
        method=method,
        teacher_output_dir=teacher_output_dir,
    )
    return bindings


def _validate_hparams(
    *,
    method: str,
    seed: int,
    batch_size: int,
    learning_rate: float,
    aam_margin: float,
    aam_scale: float,
    max_epochs: int,
    patience: int,
) -> None:
    if method not in METHODS:
        raise ValueError(f"method must be one of {sorted(METHODS)}")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    numeric = {
        "learning_rate": learning_rate,
        "aam_margin": aam_margin,
        "aam_scale": aam_scale,
    }
    if any(not np.isfinite(float(value)) for value in numeric.values()):
        raise ValueError("learning rate/AAM values must be finite")
    if float(learning_rate) <= 0.0 or not 0.0 <= float(aam_margin) < math.pi or float(aam_scale) <= 0.0:
        raise ValueError("invalid learning rate or AAM parameters")
    if isinstance(max_epochs, bool) or not isinstance(max_epochs, int) or max_epochs <= 0:
        raise ValueError("max_epochs must be a positive integer")
    if isinstance(patience, bool) or not isinstance(patience, int) or patience <= 0:
        raise ValueError("patience must be a positive integer")


def _config(
    *,
    method: str,
    output_dir: Path,
    protocol: dict[str, Any],
    seed: int,
    batch_size: int,
    learning_rate: float,
    aam_margin: float,
    aam_scale: float,
    max_epochs: int,
    patience: int,
    source_bindings: dict[str, str],
    input_bindings: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "method": method,
        "output_dir": str(output_dir),
        "architecture": "baseline_conv8",
        "class_count": int(protocol["class_count"]),
        "embedding_dim": EMBEDDING_DIM,
        "feature_shape": list(FEATURE_SHAPE),
        "seed": seed,
        "batch_size": batch_size,
        "optimizer": {
            "name": "Adam",
            "learning_rate": float(learning_rate),
            "beta_1": ADAM_BETA_1,
            "beta_2": ADAM_BETA_2,
            "epsilon": ADAM_EPSILON,
            "amsgrad": False,
            "clipnorm": None,
        },
        "aam_margin": float(aam_margin),
        "aam_scale": float(aam_scale),
        "kd_weight": KD_WEIGHT if method == "kd_aam" else 0.0,
        "max_epochs": max_epochs,
        "patience": patience,
        "sampler": {
            "version": SAMPLER_VERSION,
            "no_replacement": True,
            "full_train_coverage": True,
            "variable_windows_per_speaker": True,
        },
        "selection": {
            "metric": "validation_eer",
            "rule": "strict minimum; earliest tie",
            "patience_counts": "non-improvements including ties",
            "validation_rows": VALIDATION_ROWS,
            "enrollment_rows_per_speaker": VALIDATION_ENROLL_ROWS,
            "query_rows_per_speaker": VALIDATION_QUERY_ROWS,
        },
        "source_bindings": source_bindings,
        "input_bindings": input_bindings,
        "runtime_versions": {
            "python": platform.python_version(),
            **{name: importlib_metadata.version(name) for name in ("numpy", "tensorflow", "keras")},
        },
    }


def _config_hash(config: Mapping[str, Any]) -> str:
    return _sha256_bytes(_canonical_bytes(config))


def _stage_paths(output_dir: Path, resume_from: Path | None) -> tuple[Path, bool]:
    output = output_dir.expanduser().resolve()
    if _path_exists(output):
        raise FileExistsError(f"refusing to overwrite final output: {output}")
    if resume_from is None:
        stage = output.with_name(output.name + STAGE_SUFFIX)
        if _path_exists(stage):
            raise FileExistsError(
                f"durable training stage exists; pass resume_from explicitly: {stage}"
            )
        stage.parent.mkdir(parents=True, exist_ok=True)
        stage.mkdir()
        return stage, True
    stage = Path(resume_from).expanduser().resolve()
    if stage.parent != output.parent or stage.name != output.name + STAGE_SUFFIX:
        raise FullTrainingError(
            "resume_from must be the output sibling stage named '<output>.stage'"
        )
    if stage.is_symlink() or not stage.is_dir():
        raise FullTrainingError(f"resume stage is not a regular directory: {stage}")
    return stage, False


def _tree_binding(root: Path) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        raise FullTrainingError(f"checkpoint directory is invalid: {root}")
    files = sorted(path for path in root.iterdir() if path.is_file() or path.is_symlink())
    if not files:
        raise FullTrainingError(f"checkpoint directory is empty: {root}")
    result: dict[str, Any] = {}
    for path in files:
        if path.is_symlink() or not path.is_file():
            raise FullTrainingError(f"checkpoint file is not regular: {path}")
        result[path.name] = {"bytes": int(path.stat().st_size), "sha256": _sha256_file(path)}
    return result


def _assert_tree_binding(root: Path, expected: Mapping[str, Any]) -> None:
    observed = _tree_binding(root)
    if observed != dict(expected):
        raise FullTrainingError(f"checkpoint binding changed: {root}")


def _artifact_binding(path: Path, *, final_root: Path, stage_root: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FullTrainingError(f"artifact is not a regular file: {path}")
    return {
        "path": str(final_root / path.relative_to(stage_root)),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
    }


def _configure_tf(tf: Any, *, strict: bool) -> dict[str, Any]:
    try:
        report = _training._configure_tf_threads(tf, strict=strict)
    except Exception as exc:
        raise FullTrainingError(
            "TensorFlow must be configured in a fresh process before model creation"
        ) from exc
    if strict and not report.get("configured", False):
        raise FullTrainingError("strict TensorFlow thread configuration was not applied")
    return _json_safe(report)


def _build_models(tf: Any, class_count: int) -> tuple[Any, Any]:
    encoder = _training.build_student_model(
        batch_size=None,
        name="full100_baseline_conv8_encoder",
    )
    if int(encoder.count_params()) != 146_192:
        raise FullTrainingError(
            f"baseline Conv8 parameter count changed: {encoder.count_params()}"
        )
    head = tf.keras.layers.Dense(
        class_count,
        use_bias=False,
        activation=None,
        name="full100_aam_classifier",
    )
    head(tf.zeros((1, EMBEDDING_DIM), dtype=tf.float32))
    if tuple(head.kernel.shape) != (EMBEDDING_DIM, class_count):
        raise FullTrainingError("AAM head shape changed")
    return encoder, head


def _optimizer(tf: Any, variables: Sequence[Any], *, learning_rate: float) -> Any:
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        beta_1=ADAM_BETA_1,
        beta_2=ADAM_BETA_2,
        epsilon=ADAM_EPSILON,
        amsgrad=False,
        clipnorm=None,
    )
    build = getattr(optimizer, "build", None)
    if callable(build):
        build(list(variables))
    return optimizer


def _checkpoint_objects(tf: Any, encoder: Any, head: Any, optimizer: Any) -> tuple[Any, Any, Any, Any, Any]:
    epoch = tf.Variable(0, dtype=tf.int64, trainable=False, name="completed_epoch")
    stale = tf.Variable(0, dtype=tf.int64, trainable=False, name="stale_epochs")
    best_epoch = tf.Variable(0, dtype=tf.int64, trainable=False, name="best_epoch")
    best_eer = tf.Variable(np.float64(np.inf), dtype=tf.float64, trainable=False, name="best_validation_eer")
    checkpoint = tf.train.Checkpoint(
        encoder=encoder,
        head=head,
        optimizer=optimizer,
        completed_epoch=epoch,
        stale_epochs=stale,
        best_epoch=best_epoch,
        best_validation_eer=best_eer,
    )
    return checkpoint, epoch, stale, best_epoch, best_eer


def _restore_checkpoint(checkpoint: Any, prefix: str) -> None:
    try:
        status = checkpoint.restore(prefix)
        status.assert_existing_objects_matched()
    except Exception as exc:
        raise FullTrainingError(f"cannot restore bound checkpoint: {prefix}") from exc


def _head_checkpoint(head: Any, path: Path) -> None:
    kernel = np.asarray(head.kernel.numpy(), dtype=np.float32)
    if kernel.ndim != 2 or not np.all(np.isfinite(kernel)):
        raise FullTrainingError("AAM head checkpoint is invalid")
    np.savez_compressed(path, kernel=kernel)


def _validation_metrics(
    tf: Any,
    encoder: Any,
    validation: ValidationDataset,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    raw = np.asarray(
        encoder(tf.convert_to_tensor(
            validation.features[protocol["validation_indices"]], dtype=tf.float32
        ), training=False).numpy(),
        dtype=np.float32,
    )
    if raw.shape != (VALIDATION_ROWS, EMBEDDING_DIM):
        raise FullTrainingError(f"validation encoder output shape changed: {raw.shape}")
    if not np.all(np.isfinite(raw)):
        raise FullTrainingError("validation encoder output is non-finite")
    norms = np.linalg.norm(raw.astype(np.float64), axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
        raise FullTrainingError(
            "validation encoder produced a zero/non-finite embedding; rows were retained"
        )
    unit = (raw.astype(np.float64) / norms[:, None]).astype(np.float32)
    by_index = {int(index): unit[position] for position, index in enumerate(protocol["validation_indices"].tolist())}
    templates: dict[str, np.ndarray] = {}
    for speaker in protocol["validation_speakers"]:
        enrollment = np.stack(
            [by_index[index] for index in protocol["validation_by_speaker"][speaker][:VALIDATION_ENROLL_ROWS]],
            axis=0,
        )
        template = np.mean(enrollment.astype(np.float64), axis=0)
        template_norm = float(np.linalg.norm(template))
        if not np.isfinite(template_norm) or template_norm <= 0.0:
            raise FullTrainingError("validation enrollment template is zero/non-finite")
        templates[speaker] = (template / template_norm).astype(np.float32)
    labels: list[int] = []
    scores: list[float] = []
    correct = 0
    total = 0
    ordered_speakers = list(protocol["validation_speakers"])
    for query_index in protocol["query_indices"].tolist():
        query = by_index[query_index].astype(np.float64)
        query_speaker = str(validation.rows[query_index]["speaker_id"])
        candidate_scores: list[tuple[str, float]] = []
        for speaker in ordered_speakers:
            score = float(np.dot(query, templates[speaker].astype(np.float64)))
            if not np.isfinite(score):
                raise FullTrainingError("validation score is non-finite")
            labels.append(int(query_speaker == speaker))
            scores.append(score)
            candidate_scores.append((speaker, score))
        selected = max(candidate_scores, key=lambda item: item[1])[0]
        correct += int(selected == query_speaker)
        total += 1
    metric = _comparative._validation_metric_from_scores(labels, scores) if hasattr(_comparative, "_validation_metric_from_scores") else None
    if metric is None:
        from .evaluation import evaluate_scores

        metric = evaluate_scores(labels, scores)
    return {
        "eer": float(metric["eer"]),
        "eer_threshold": float(metric["eer_threshold"]),
        "target_trials": int(metric["target_trials"]),
        "non_target_trials": int(metric["non_target_trials"]),
        "top1": {
            "correct": correct,
            "total": total,
            "accuracy": float(correct / total),
            "scope": "validation-only diagnostic; not model selection",
            "tie_policy": "sorted validation speaker order first maximum",
        },
        "metric_provenance": "local float64 evaluate_scores; validation-only",
    }


def _loss(
    tf: Any,
    embeddings: Any,
    labels: Any,
    kernel: Any,
    *,
    method: str,
    teacher_embeddings: Any | None,
    margin: float,
    scale: float,
) -> tuple[Any, Any | None]:
    try:
        cosine = _comparative.normalized_classifier(tf, embeddings, kernel)
        logits = _comparative.aam_logits(
            tf,
            cosine,
            labels,
            margin=margin,
            scale=scale,
        )
        classification = _comparative._aam_cross_entropy(tf, logits, labels)
        if method == "supervised_aam":
            if teacher_embeddings is not None:
                raise FullTrainingError("supervised AAM received a teacher tensor")
            return classification, None
        if teacher_embeddings is None:
            raise FullTrainingError("KD+AAM has no teacher tensor")
        kd = _comparative._cosine_kd_loss(tf, embeddings, teacher_embeddings)
        return classification + KD_WEIGHT * kd, kd
    except FullTrainingError:
        raise
    except Exception as exc:
        raise FullTrainingError(
            "normalized AAM/KD loss rejected a batch; rows were retained and not dropped"
        ) from exc


def _write_progress(stage: Path, value: Mapping[str, Any]) -> None:
    payload = dict(value)
    payload["updated_at_unix"] = time.time()
    _write_json(stage / "progress.json", payload)
    print(json.dumps(_json_safe(payload), sort_keys=True), flush=True)


def _batch_input(dataset: _full_data.FullDataset, indices: np.ndarray) -> np.ndarray:
    values = np.asarray(dataset.features[indices], dtype=np.float32)
    if values.shape != (indices.size, *FEATURE_SHAPE) or not np.all(np.isfinite(values)):
        raise FullTrainingError("bounded feature gather produced invalid values")
    return values


def _checkpoint_state(
    *,
    stage: Path,
    latest_manager: Any,
    best_manager: Any,
    epoch: int,
    stale: int,
    best_epoch: int,
    best_eer: float | None,
    history: list[dict[str, Any]],
    config: dict[str, Any],
    source_bindings: dict[str, str],
    input_bindings: dict[str, Any],
    status: str,
    stop_reason: str | None = None,
) -> dict[str, Any]:
    history_path = stage / HISTORY_FILENAME
    _write_json(history_path, history)
    latest_prefix = latest_manager.latest_checkpoint
    if latest_prefix is None:
        raise FullTrainingError("latest checkpoint manager has no checkpoint")
    latest_root = stage / "checkpoints" / "latest"
    latest_binding = _tree_binding(latest_root)
    best_prefix = best_manager.latest_checkpoint
    best_binding = None
    if best_prefix is not None:
        best_binding = _tree_binding(stage / "checkpoints" / "best")
    return {
        "format_version": 1,
        "status": status,
        "method": config["method"],
        "config_sha256": _config_hash(config),
        "source_bindings": source_bindings,
        "input_bindings": input_bindings,
        "completed_epoch": int(epoch),
        "stale_epochs": int(stale),
        "best_epoch": int(best_epoch),
        "best_validation_eer": None if best_eer is None else float(best_eer),
        "history_sha256": _sha256_file(history_path),
        "history_rows": len(history),
        "latest_checkpoint": str(Path(latest_prefix).relative_to(stage)),
        "latest_checkpoint_binding": latest_binding,
        "best_checkpoint": None if best_prefix is None else str(Path(best_prefix).relative_to(stage)),
        "best_checkpoint_binding": best_binding,
        "stop_reason": stop_reason,
    }


def _load_stage_state(stage: Path) -> dict[str, Any]:
    state = _read_json(stage / STATE_FILENAME, label="training state")
    config = _read_json(stage / CONFIG_FILENAME, label="training config")
    if state.get("config_sha256") != _config_hash(config):
        raise FullTrainingError("training state/config binding mismatch")
    history_path = stage / HISTORY_FILENAME
    if not history_path.is_file():
        raise FullTrainingError("training stage has no durable history")
    try:
        history = json.loads(history_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FullTrainingError("training history is unreadable") from exc
    if not isinstance(history, list) or _sha256_file(history_path) != state.get("history_sha256"):
        raise FullTrainingError("training history binding mismatch")
    state["_config"] = config
    state["_history"] = history
    return state


def _assert_resume_state(
    state: Mapping[str, Any],
    config: Mapping[str, Any],
    source_bindings: Mapping[str, str],
    input_bindings: Mapping[str, Any],
    stage: Path,
) -> None:
    if state.get("config_sha256") != _config_hash(config):
        raise FullTrainingError("resume rejected: config changed")
    if state.get("source_bindings") != dict(source_bindings):
        raise FullTrainingError("resume rejected: source bindings changed")
    if state.get("input_bindings") != dict(input_bindings):
        raise FullTrainingError("resume rejected: input bindings changed")
    latest = state.get("latest_checkpoint")
    latest_binding = state.get("latest_checkpoint_binding")
    if not isinstance(latest, str) or not isinstance(latest_binding, dict):
        raise FullTrainingError("resume rejected: latest checkpoint binding missing")
    _assert_tree_binding(stage / Path(latest).parent, latest_binding)
    best = state.get("best_checkpoint")
    best_binding = state.get("best_checkpoint_binding")
    if best is None or not isinstance(best_binding, dict):
        if int(state.get("completed_epoch", -1)) == 0:
            return
        raise FullTrainingError("resume rejected: best checkpoint binding missing")
    _assert_tree_binding(stage / Path(best).parent, best_binding)


def _best_portable_bindings(
    stage: Path,
    encoder: Any,
    head: Any,
    *,
    final_root: Path,
) -> dict[str, Any]:
    encoder_path = stage / "encoder_best.weights.h5"
    head_path = stage / "classifier_best.npz"
    encoder.save_weights(encoder_path)
    _head_checkpoint(head, head_path)
    return {
        "encoder": _artifact_binding(
            encoder_path, final_root=final_root, stage_root=stage
        ),
        "classifier": _artifact_binding(
            head_path, final_root=final_root, stage_root=stage
        ),
    }


def _export(
    tf: Any,
    *,
    encoder: Any,
    dataset: _full_data.FullDataset,
    validation: ValidationDataset,
    protocol: Mapping[str, Any],
    stage: Path,
    final_root: Path,
) -> dict[str, Any]:
    train_export: list[int] = []
    for speaker in protocol["train_speakers"]:
        values = sorted(
            protocol["train_by_speaker"][speaker].tolist(),
            key=lambda index: str(dataset.rows[index]["window_id"]),
        )
        if len(values) < 2:
            raise FullTrainingError(
                f"speaker {speaker!r} has fewer than two train windows for export"
            )
        train_export.extend(values[:2])
    train_export_indices = np.asarray(train_export, dtype=np.int64)
    validation_export_indices = np.asarray(protocol["validation_indices"], dtype=np.int64)
    fixed = _training.build_student_model(
        batch_size=1,
        name="full100_baseline_conv8_export",
    )
    fixed.set_weights(encoder.get_weights())
    probe = _batch_input(dataset, train_export_indices[:1])
    parity, _, _ = _training._validate_batch1_export_parity(encoder, fixed, probe)
    fp32_content = _training._convert_fp32(tf, fixed)
    int8_content = _training._convert_int8(
        tf,
        fixed,
        dataset.features,
        train_export_indices,
    )
    fp32_path = stage / "encoder_fp32.tflite"
    int8_path = stage / "encoder_int8.tflite"
    fp32_path.write_bytes(fp32_content)
    int8_path.write_bytes(int8_content)
    budget_report = _budget.check_model(int8_path)
    if not budget_report.get("static_budget_pass", False):
        raise FullTrainingError(
            "INT8 export failed the static budget gate: "
            + json.dumps(_json_safe(budget_report), sort_keys=True)
        )
    selected_features = np.concatenate(
        (
            _batch_input(dataset, train_export_indices),
            np.asarray(validation.features[validation_export_indices], dtype=np.float32),
        ),
        axis=0,
    )
    fp32_raw, fp32_embeddings, fp32_metadata = _training._run_tflite(
        fp32_content,
        selected_features,
        require_int8=False,
    )
    int8_raw, int8_embeddings, int8_metadata = _training._run_tflite(
        int8_content,
        selected_features,
        require_int8=True,
    )
    val_start = train_export_indices.size
    fp32_validation = _validation_metrics_from_embeddings(
        fp32_embeddings[val_start:], validation, protocol
    )
    int8_validation = _validation_metrics_from_embeddings(
        int8_embeddings[val_start:], validation, protocol
    )
    selected_ids = [
        f"train:{dataset.rows[index]['window_id']}" for index in train_export_indices.tolist()
    ] + [
        f"validation:{validation.rows[index]['utterance_id']}"
        for index in validation_export_indices.tolist()
    ]
    selected_ids_array = np.asarray(selected_ids, dtype=str)
    for path, values in (
        (stage / "encoder_fp32_embeddings.npz", fp32_embeddings),
        (stage / "encoder_int8_embeddings.npz", int8_embeddings),
    ):
        np.savez_compressed(
            path,
            ids=selected_ids_array,
            embeddings=np.asarray(values, dtype=np.float32),
            train_manifest_sha256=np.asarray(dataset.manifest_sha256),
            validation_manifest_sha256=np.asarray(validation.manifest_sha256),
        )
    drift = fp32_raw.astype(np.float64) - int8_raw.astype(np.float64)
    cosine = np.sum(
        fp32_embeddings.astype(np.float64) * int8_embeddings.astype(np.float64),
        axis=1,
    )
    return {
        "encoder_fp32": _artifact_binding(
            fp32_path, final_root=final_root, stage_root=stage
        ),
        "encoder_int8": _artifact_binding(
            int8_path, final_root=final_root, stage_root=stage
        ),
        "fp32_embeddings": _artifact_binding(
            stage / "encoder_fp32_embeddings.npz",
            final_root=final_root,
            stage_root=stage,
        ),
        "int8_embeddings": _artifact_binding(
            stage / "encoder_int8_embeddings.npz",
            final_root=final_root,
            stage_root=stage,
        ),
        "dynamic_fixed_batch1_max_abs": float(parity),
        "dynamic_fixed_batch1_atol": 1e-6,
        "input_shape": [1, *FEATURE_SHAPE],
        "output_shape": [1, EMBEDDING_DIM],
        "input_dtype": "int8",
        "output_dtype": "int8",
        "representative_rows": int(train_export_indices.size),
        "representative_source": "two deterministic TRAIN windows per speaker",
        "selected_rows": int(train_export_indices.size + validation_export_indices.size),
        "selected_train_rows": int(train_export_indices.size),
        "selected_validation_rows": int(validation_export_indices.size),
        "fp32_metadata": _json_safe(fp32_metadata),
        "int8_metadata": _json_safe(int8_metadata),
        "static_budget": _json_safe(budget_report),
        "validation": {
            "fp32": fp32_validation,
            "int8": int8_validation,
        },
        "int8_vs_fp32_drift": {
            "raw_max_abs": float(np.max(np.abs(drift))),
            "raw_rms": float(np.sqrt(np.mean(drift**2))),
            "normalized_max_abs": float(
                np.max(np.abs(fp32_embeddings.astype(np.float64) - int8_embeddings.astype(np.float64)))
            ),
            "normalized_rms": float(
                np.sqrt(
                    np.mean(
                        (
                            fp32_embeddings.astype(np.float64)
                            - int8_embeddings.astype(np.float64)
                        )
                        ** 2
                    )
                )
            ),
            "cosine_min": float(np.min(cosine)),
            "cosine_mean": float(np.mean(cosine)),
            "cosine_max": float(np.max(cosine)),
            "scope": "desktop TFLite FP32 versus INT8; no board claim",
        },
    }


def _validation_metrics_from_embeddings(
    embeddings: np.ndarray,
    validation: ValidationDataset,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    values = np.asarray(embeddings, dtype=np.float32)
    if values.shape != (VALIDATION_ROWS, EMBEDDING_DIM):
        raise FullTrainingError("export validation embedding shape changed")
    if not np.all(np.isfinite(values)):
        raise FullTrainingError("export validation embeddings are non-finite")
    norms = np.linalg.norm(values.astype(np.float64), axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
        raise FullTrainingError("export validation contains zero/non-finite embeddings")
    unit = (values.astype(np.float64) / norms[:, None]).astype(np.float32)
    by_index = {int(index): unit[position] for position, index in enumerate(protocol["validation_indices"].tolist())}
    templates: dict[str, np.ndarray] = {}
    for speaker in protocol["validation_speakers"]:
        enrollment = np.stack(
            [by_index[index] for index in protocol["validation_by_speaker"][speaker][:3]],
            axis=0,
        )
        template = np.mean(enrollment.astype(np.float64), axis=0)
        norm = float(np.linalg.norm(template))
        if not np.isfinite(norm) or norm <= 0.0:
            raise FullTrainingError("export validation enrollment is zero/non-finite")
        templates[speaker] = (template / norm).astype(np.float32)
    labels: list[int] = []
    scores: list[float] = []
    correct = 0
    total = 0
    for index in protocol["query_indices"].tolist():
        query = by_index[index].astype(np.float64)
        query_speaker = str(validation.rows[index]["speaker_id"])
        candidates = []
        for speaker in protocol["validation_speakers"]:
            score = float(np.dot(query, templates[speaker].astype(np.float64)))
            labels.append(int(query_speaker == speaker))
            scores.append(score)
            candidates.append((speaker, score))
        correct += int(max(candidates, key=lambda item: item[1])[0] == query_speaker)
        total += 1
    from .evaluation import evaluate_scores

    metric = evaluate_scores(labels, scores)
    return {
        "eer": float(metric["eer"]),
        "eer_threshold": float(metric["eer_threshold"]),
        "target_trials": int(metric["target_trials"]),
        "non_target_trials": int(metric["non_target_trials"]),
        "top1": {
            "correct": correct,
            "total": total,
            "accuracy": float(correct / total),
            "scope": "validation-only diagnostic; not model selection",
        },
    }


def _run_training(
    dataset: _full_data.FullDataset,
    validation: ValidationDataset,
    output_dir: str | Path,
    *,
    method: Literal["supervised_aam", "kd_aam"],
    teacher_output_dir: str | Path | None = None,
    resume_from: str | Path | None = None,
    seed: int = SEED,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    aam_margin: float = AAM_MARGIN,
    aam_scale: float = AAM_SCALE,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    strict_thread_config: bool = True,
    _interrupt_after_epoch: int | None = None,
) -> dict[str, Any]:
    _validate_hparams(
        method=method,
        seed=seed,
        batch_size=batch_size,
        learning_rate=learning_rate,
        aam_margin=aam_margin,
        aam_scale=aam_scale,
        max_epochs=max_epochs,
        patience=patience,
    )
    if method == "supervised_aam" and teacher_output_dir is not None:
        raise FullTrainingError("supervised_aam must not receive a teacher output")
    teacher_path = None if teacher_output_dir is None else Path(teacher_output_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    protocol = _training_protocol(dataset, validation)
    if method == "kd_aam" and teacher_path is None:
        raise FullTrainingError("kd_aam requires teacher_output_dir")
    source_bindings = _source_bindings(method)
    input_bindings = _current_input_bindings(
        dataset,
        validation,
        protocol,
        method=method,
        teacher_output_dir=teacher_path,
    )
    config = _config(
        method=method,
        output_dir=output,
        protocol=protocol,
        seed=seed,
        batch_size=batch_size,
        learning_rate=learning_rate,
        aam_margin=aam_margin,
        aam_scale=aam_scale,
        max_epochs=max_epochs,
        patience=patience,
        source_bindings=source_bindings,
        input_bindings=input_bindings,
    )
    stage, fresh = _stage_paths(output, None if resume_from is None else Path(resume_from))
    tf = _tensorflow()
    thread_report = _configure_tf(tf, strict=strict_thread_config)
    tf.keras.utils.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    encoder, head = _build_models(tf, int(protocol["class_count"]))
    variables = list(encoder.trainable_variables) + list(head.trainable_variables)
    optimizer = _optimizer(tf, variables, learning_rate=float(learning_rate))
    checkpoint, epoch_var, stale_var, best_epoch_var, best_eer_var = _checkpoint_objects(
        tf, encoder, head, optimizer
    )
    checkpoint_root = stage / "checkpoints"
    latest_manager = tf.train.CheckpointManager(
        checkpoint,
        str(checkpoint_root / "latest"),
        max_to_keep=2,
    )
    best_manager = tf.train.CheckpointManager(
        checkpoint,
        str(checkpoint_root / "best"),
        max_to_keep=1,
    )
    if fresh:
        history: list[dict[str, Any]] = []
        encoder.save_weights(stage / "encoder_initial.weights.h5")
        _head_checkpoint(head, stage / "classifier_initial.npz")
        latest_manager.save(checkpoint_number=0)
        _write_json(stage / CONFIG_FILENAME, config)
        state = _checkpoint_state(
            stage=stage,
            latest_manager=latest_manager,
            best_manager=best_manager,
            epoch=0,
            stale=0,
            best_epoch=0,
            best_eer=None,
            history=history,
            config=config,
            source_bindings=source_bindings,
            input_bindings=input_bindings,
            status="TRAINING",
        )
        _write_json(stage / STATE_FILENAME, state)
        _write_progress(
            stage,
            {
                "status": "TRAINING",
                "method": method,
                "completed_epoch": 0,
                "total_rows": len(dataset.rows),
            },
        )
    else:
        state = _load_stage_state(stage)
        if state.get("status") not in {
            "TRAINING",
            "FROZEN",
            "EXPORTING",
            "READY_TO_PUBLISH",
        }:
            raise FullTrainingError(f"resume stage has unsupported status: {state.get('status')}")
        _assert_resume_state(state, config, source_bindings, input_bindings, stage)
        history = list(state["_history"])
        latest_prefix = stage / state["latest_checkpoint"]
        _restore_checkpoint(checkpoint, str(latest_prefix))
        if int(epoch_var.numpy()) != int(state["completed_epoch"]):
            raise FullTrainingError("resume latest checkpoint epoch differs from state")
        if int(stale_var.numpy()) != int(state["stale_epochs"]):
            raise FullTrainingError("resume latest checkpoint stale count differs from state")
        if int(best_epoch_var.numpy()) != int(state["best_epoch"]):
            raise FullTrainingError("resume latest checkpoint best epoch differs from state")
        state_best = state.get("best_validation_eer")
        checkpoint_best = float(best_eer_var.numpy())
        if state_best is None or not math.isfinite(float(state_best)):
            if math.isfinite(checkpoint_best):
                raise FullTrainingError("resume best EER state differs from checkpoint")
        elif abs(float(state_best) - checkpoint_best) > 1e-7:
            raise FullTrainingError("resume best EER differs from checkpoint")

    teacher: np.ndarray | None = None
    if method == "kd_aam":
        teacher, _, _ = _teacher_inputs(teacher_path, dataset)  # type: ignore[arg-type]
    status = state.get("status", "TRAINING") if not fresh else "TRAINING"
    if status == "TRAINING" and int(stale_var.numpy()) < patience:
        start_epoch = int(epoch_var.numpy()) + 1
        for epoch in range(start_epoch, max_epochs + 1):
            batches = interleaved_epoch_batches(
                protocol["train_by_speaker"],
                epoch=epoch - 1,
                seed=seed,
                batch_size=batch_size,
            )
            train_total = 0.0
            train_aam = 0.0
            train_kd = 0.0
            rows_seen = 0
            for batch_number, batch_indices in enumerate(batches, start=1):
                x_batch = tf.convert_to_tensor(_batch_input(dataset, batch_indices), dtype=tf.float32)
                labels = tf.convert_to_tensor(protocol["labels"][batch_indices], dtype=tf.int32)
                teacher_batch = None
                if method == "kd_aam":
                    if teacher is None:
                        raise FullTrainingError("KD teacher snapshot disappeared")
                    teacher_batch = tf.convert_to_tensor(
                        np.asarray(teacher[batch_indices], dtype=np.float32), dtype=tf.float32
                    )
                with tf.GradientTape() as tape:
                    embeddings = encoder(x_batch, training=True)
                    total, kd_value = _loss(
                        tf,
                        embeddings,
                        labels,
                        head.kernel,
                        method=method,
                        teacher_embeddings=teacher_batch,
                        margin=aam_margin,
                        scale=aam_scale,
                    )
                gradients = tape.gradient(total, variables)
                if any(gradient is None for gradient in gradients):
                    raise FullTrainingError(f"missing gradient in epoch {epoch}, batch {batch_number}")
                if not all(bool(tf.reduce_all(tf.math.is_finite(gradient))) for gradient in gradients):
                    raise FullTrainingError(f"non-finite gradient in epoch {epoch}, batch {batch_number}")
                optimizer.apply_gradients(zip(gradients, variables))
                total_value = float(total.numpy())
                kd_float = 0.0 if kd_value is None else float(kd_value.numpy())
                aam_float = total_value - KD_WEIGHT * kd_float if method == "kd_aam" else total_value
                if not all(np.isfinite(value) for value in (total_value, kd_float, aam_float)):
                    raise FullTrainingError(f"non-finite loss in epoch {epoch}, batch {batch_number}")
                count = int(batch_indices.size)
                train_total += total_value * count
                train_aam += aam_float * count
                train_kd += kd_float * count
                rows_seen += count
                if batch_number % PROGRESS_BATCH_INTERVAL == 0 or batch_number == len(batches):
                    _write_progress(
                        stage,
                        {
                            "status": "TRAINING",
                            "method": method,
                            "epoch": epoch,
                            "batch": batch_number,
                            "batches": len(batches),
                            "rows_seen": rows_seen,
                            "total_rows": len(dataset.rows),
                        },
                    )
            if rows_seen != len(dataset.rows):
                raise FullTrainingError("epoch did not consume every train row")
            validation_metrics = _validation_metrics(tf, encoder, validation, protocol)
            eer = float(validation_metrics["eer"])
            improved = eer < float(best_eer_var.numpy())
            if improved:
                best_eer_var.assign(np.float64(eer))
                best_epoch_var.assign(epoch)
                stale_var.assign(0)
            else:
                stale_var.assign_add(1)
            epoch_var.assign(epoch)
            history.append(
                {
                    "epoch": epoch,
                    "schedule_digest": _schedule_digest(batches),
                    "batch_count": len(batches),
                    "rows_seen": rows_seen,
                    "train_total_loss": train_total / rows_seen,
                    "train_aam_loss": train_aam / rows_seen,
                    "train_kd_loss": None if method == "supervised_aam" else train_kd / rows_seen,
                    "validation_eer": eer,
                    "validation_eer_threshold": validation_metrics["eer_threshold"],
                    "validation_top1": validation_metrics["top1"],
                    "improved": improved,
                }
            )
            latest_manager.save(checkpoint_number=epoch)
            if improved:
                best_manager.save(checkpoint_number=epoch)
            best_value = float(best_eer_var.numpy())
            state = _checkpoint_state(
                stage=stage,
                latest_manager=latest_manager,
                best_manager=best_manager,
                epoch=epoch,
                stale=int(stale_var.numpy()),
                best_epoch=int(best_epoch_var.numpy()),
                best_eer=best_value,
                history=history,
                config=config,
                source_bindings=source_bindings,
                input_bindings=input_bindings,
                status="TRAINING",
                stop_reason=("early_stopping" if int(stale_var.numpy()) >= patience else None),
            )
            _write_json(stage / STATE_FILENAME, state)
            _write_progress(
                stage,
                {
                    "status": "EPOCH_COMPLETE",
                    "method": method,
                    "epoch": epoch,
                    "best_epoch": int(best_epoch_var.numpy()),
                    "best_validation_eer": best_value,
                    "stale_epochs": int(stale_var.numpy()),
                },
            )
            if _interrupt_after_epoch is not None and epoch >= _interrupt_after_epoch:
                raise RuntimeError(f"synthetic interruption after completed epoch {epoch}")
            if int(stale_var.numpy()) >= patience:
                break
    if best_manager.latest_checkpoint is None:
        raise FullTrainingError("no validation-selected best checkpoint exists")
    completed_epoch = int(epoch_var.numpy())
    completed_stale = int(stale_var.numpy())
    selected_best_epoch = int(best_epoch_var.numpy())
    selected_best_eer = float(best_eer_var.numpy())
    if not np.isfinite(selected_best_eer):
        raise FullTrainingError("best validation EER is non-finite")
    completed_stop_reason = (
        "early_stopping"
        if completed_stale >= patience
        else "max_epochs"
        if completed_epoch >= max_epochs
        else "resume_export"
    )
    _restore_checkpoint(checkpoint, best_manager.latest_checkpoint)
    best_portable = _best_portable_bindings(
        stage,
        encoder,
        head,
        final_root=output,
    )
    latest_binding = _tree_binding(stage / "checkpoints" / "latest")
    best_binding = _tree_binding(stage / "checkpoints" / "best")
    state = _checkpoint_state(
        stage=stage,
        latest_manager=latest_manager,
        best_manager=best_manager,
        epoch=completed_epoch,
        stale=completed_stale,
        best_epoch=selected_best_epoch,
        best_eer=selected_best_eer,
        history=history,
        config=config,
        source_bindings=source_bindings,
        input_bindings=input_bindings,
        status="FROZEN",
        stop_reason=completed_stop_reason,
    )
    state["portable_best_bindings"] = best_portable
    state["latest_checkpoint_binding"] = latest_binding
    state["best_checkpoint_binding"] = best_binding
    _write_json(stage / STATE_FILENAME, state)
    freeze_inputs = _reload_and_bind_inputs(
        dataset,
        validation,
        method=method,
        teacher_output_dir=teacher_path,
    )
    freeze_sources = _source_bindings(method)
    if freeze_inputs != input_bindings:
        raise FullTrainingError("pre-export input binding changed")
    if freeze_sources != source_bindings:
        raise FullTrainingError("pre-export source binding changed")
    _assert_tree_binding(stage / "checkpoints" / "latest", latest_binding)
    _assert_tree_binding(stage / "checkpoints" / "best", best_binding)
    freeze = {
        "format_version": 1,
        "status": "FROZEN_BEFORE_EXPORT",
        "config": config,
        "config_sha256": _config_hash(config),
        "source_bindings": source_bindings,
        "input_bindings": input_bindings,
        "checkpoint_bindings": {
            "latest": latest_binding,
            "best": best_binding,
            "portable_best": best_portable,
        },
        "evaluation": {
            "calibration": None,
            "test": None,
            "status": "NOT_RUN",
        },
    }
    _write_json(stage / FREEZE_FILENAME, freeze)
    state["status"] = "EXPORTING"
    _write_json(stage / STATE_FILENAME, state)
    export = _export(
        tf,
        encoder=encoder,
        dataset=dataset,
        validation=validation,
        protocol=protocol,
        stage=stage,
        final_root=output,
    )
    if _reload_and_bind_inputs(
        dataset,
        validation,
        method=method,
        teacher_output_dir=teacher_path,
    ) != input_bindings:
        raise FullTrainingError("pre-publish input binding changed")
    if _source_bindings(method) != source_bindings:
        raise FullTrainingError("pre-publish source binding changed")
    report = {
        "format_version": 1,
        "status": "FULL_TRAINING_COMPLETE",
        "method": method,
        "scope": "FULL_TRAIN_ONLY_HOST_EXPORT",
        "hardware_status": "UNVERIFIED_HARDWARE",
        "dataset": {
            "train_rows": len(dataset.rows),
            "train_speakers": len(protocol["train_speakers"]),
            "manifest_sha256": dataset.manifest_sha256,
            "windowing": "full consecutive non-overlap 3s windows; final padding from data loader",
        },
        "validation": {
            "rows": len(validation.rows),
            "speakers": len(protocol["validation_speakers"]),
            "manifest_sha256": validation.manifest_sha256,
            "features_sha256": validation.features_sha256,
            "enrollment_rows_per_speaker": VALIDATION_ENROLL_ROWS,
            "query_rows_per_speaker": VALIDATION_QUERY_ROWS,
            "selection_metric": "validation EER only",
        },
        "config": config,
        "source_bindings": source_bindings,
        "input_bindings": input_bindings,
        "teacher_provenance": None if method == "supervised_aam" else input_bindings["teacher"],
        "tensorflow_thread_policy": thread_report,
        "training": {
            "history": history,
            "epochs_ran": len(history),
            "best_epoch": selected_best_epoch,
            "best_validation_eer": selected_best_eer,
            "stop_reason": state["stop_reason"],
            "latest_checkpoint": state["latest_checkpoint"],
            "best_checkpoint": state["best_checkpoint"],
            "portable_best": best_portable,
        },
        "freeze": freeze,
        "export": export,
        "evaluation": {
            "validation": export["validation"],
            "calibration": None,
            "test": None,
            "status": "VALIDATION_ONLY; CALIBRATION_AND_TEST_NOT_RUN",
        },
        "paths": {
            "report": str(output / REPORT_FILENAME),
            "encoder_fp32": str(output / "encoder_fp32.tflite"),
            "encoder_int8": str(output / "encoder_int8.tflite"),
            "freeze": str(output / FREEZE_FILENAME),
        },
    }
    _write_json(stage / REPORT_FILENAME, report)
    report_binding = _artifact_binding(
        stage / REPORT_FILENAME,
        final_root=output,
        stage_root=stage,
    )
    state["status"] = "READY_TO_PUBLISH"
    state["report_binding"] = report_binding
    _write_json(stage / STATE_FILENAME, state)
    if _path_exists(output):
        raise FileExistsError(f"refusing to overwrite final output: {output}")
    os.replace(stage, output)
    report["paths"] = {
        "report": str(output / REPORT_FILENAME),
        "encoder_fp32": str(output / "encoder_fp32.tflite"),
        "encoder_int8": str(output / "encoder_int8.tflite"),
        "freeze": str(output / FREEZE_FILENAME),
    }
    return report


def train_supervised_aam(
    dataset: _full_data.FullDataset,
    validation: ValidationDataset,
    output_dir: str | Path,
    *,
    resume_from: str | Path | None = None,
    seed: int = SEED,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    aam_margin: float = AAM_MARGIN,
    aam_scale: float = AAM_SCALE,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    strict_thread_config: bool = True,
) -> dict[str, Any]:
    """Train the baseline 251-class supervised AAM objective."""

    return _run_training(
        dataset,
        validation,
        output_dir,
        method="supervised_aam",
        resume_from=resume_from,
        seed=seed,
        batch_size=batch_size,
        learning_rate=learning_rate,
        aam_margin=aam_margin,
        aam_scale=aam_scale,
        max_epochs=max_epochs,
        patience=patience,
        strict_thread_config=strict_thread_config,
    )


def train_kd_aam(
    dataset: _full_data.FullDataset,
    validation: ValidationDataset,
    teacher_output_dir: str | Path,
    output_dir: str | Path,
    *,
    resume_from: str | Path | None = None,
    seed: int = SEED,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    aam_margin: float = AAM_MARGIN,
    aam_scale: float = AAM_SCALE,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    strict_thread_config: bool = True,
) -> dict[str, Any]:
    """Train the baseline KD+AAM objective with the verified teacher cache."""

    return _run_training(
        dataset,
        validation,
        output_dir,
        method="kd_aam",
        teacher_output_dir=teacher_output_dir,
        resume_from=resume_from,
        seed=seed,
        batch_size=batch_size,
        learning_rate=learning_rate,
        aam_margin=aam_margin,
        aam_scale=aam_scale,
        max_epochs=max_epochs,
        patience=patience,
        strict_thread_config=strict_thread_config,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train full train-clean-100 baseline AAM or KD+AAM and export bounded host models"
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--method",
        choices=("supervised_aam", "kd_aam"),
        required=True,
    )
    parser.add_argument("--teacher-output", type=Path, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume only the explicit sibling stage '<output>.stage'; never overwrite final output",
    )
    return parser


def _main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.method == "supervised_aam" and args.teacher_output is not None:
        parser.error("--teacher-output is valid only for kd_aam")
    if args.method == "kd_aam" and args.teacher_output is None:
        parser.error("--teacher-output is required for kd_aam")
    resume_from = None
    if args.resume:
        resume_from = args.output.expanduser().resolve().with_name(
            args.output.expanduser().resolve().name + STAGE_SUFFIX
        )
    try:
        dataset = _full_data.load_full_dataset(
            args.dataset_root,
            verify_hashes=True,
            test_fixture=False,
        )
        validation = load_validation_dataset(
            args.validation_root,
            verify_hashes=True,
            test_fixture=False,
        )
        if args.method == "supervised_aam":
            report = train_supervised_aam(
                dataset,
                validation,
                args.output,
                resume_from=resume_from,
            )
        else:
            report = train_kd_aam(
                dataset,
                validation,
                args.teacher_output,
                args.output,
                resume_from=resume_from,
            )
    except Exception as exc:
        print(f"full_training failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2
    print(
        json.dumps(
            {
                "status": report["status"],
                "method": report["method"],
                "output_dir": str(args.output.expanduser().resolve()),
                "report": report["paths"]["report"],
                "encoder_int8": report["paths"]["encoder_int8"],
                "stage_contract": {
                    "fresh_output_refused": True,
                    "durable_sibling_stage": str(
                        args.output.expanduser().resolve().with_name(
                            args.output.expanduser().resolve().name + STAGE_SUFFIX
                        )
                    ),
                    "resume_flag": "explicit --resume only",
                },
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


__all__ = [
    "AAM_MARGIN",
    "AAM_SCALE",
    "BATCH_SIZE",
    "EXPECTED_TRAIN_ROWS",
    "FEATURE_SHAPE",
    "FullTrainingError",
    "KD_WEIGHT",
    "LEARNING_RATE",
    "MAX_EPOCHS",
    "PATIENCE",
    "SEED",
    "ValidationDataset",
    "interleaved_epoch_batches",
    "load_validation_dataset",
    "train_kd_aam",
    "train_supervised_aam",
]


if __name__ == "__main__":  # pragma: no cover - exercised as a CLI process.
    raise SystemExit(_main())
