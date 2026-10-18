"""Independent preparation and validation for the bounded expanded pilot.

This module deliberately does not use the original pilot-data or training
validators.  Those validators describe the fixed 640-row dev-clean pilot and
would either reject the expanded split or, worse, make its provenance look
like dev-clean.  The format here is a small, explicit v2 local experiment.

The real preparation path is two-pass: every archive member is checked before
any selected audio is written, and the archive digest is checked again before
publication.  Test fixtures must opt into ``test_fixture=True`` and are never
given official source provenance.
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import tarfile
import tempfile
import wave
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

import numpy as np

from .frontend import log_mel


FORMAT_VERSION = 2
SEED = 20261012
SAMPLE_RATE = 16_000
NUM_SAMPLES = 48_000
FEATURE_SHAPE = (298, 40, 1)
EMBEDDING_DIM = 192

TRAIN_ARCHIVE_URL = "https://www.openslr.org/resources/12/train-clean-100.tar.gz"
TRAIN_ARCHIVE_MD5 = "2a93770f6d5c6c964bc36631d331a522"
TRAIN_DATASET = "librispeech-train-clean-100-selected-pilot-v1"
EXPANDED_DATASET = "librispeech-dev-clean-plus-train-clean-100-pilot-v1"
OLD_DATASET = "librispeech-dev-clean-pilot"

MAX_ARCHIVE_BYTES = 8_000_000_000
MAX_DECOMPRESSED_BYTES = 20_000_000_000
MAX_MEMBER_COUNT = 1_000_000
MAX_MEMBER_BYTES = 100_000_000
MAX_METADATA_BYTES = 16_000_000

TOP_LEVEL_METADATA = frozenset(
    {"LICENSE.TXT", "README.TXT", "CHAPTERS.TXT", "SPEAKERS.TXT", "BOOKS.TXT"}
)
V2_SPLITS = frozenset({"train", "validation"})
V2_ROW_FIELDS = (
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
)
LEGACY_ROW_FIELDS = frozenset(V2_ROW_FIELDS)


class ExpandedDataError(ValueError):
    """Raised for an invalid archive, v2 manifest, or immutable input."""


@dataclass(frozen=True)
class ArchiveDigest:
    path: str
    size_bytes: int
    md5: str
    sha256: str
    expected_md5: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "bytes": self.size_bytes,
            "md5": self.md5,
            "expected_md5": self.expected_md5,
            "sha256": self.sha256,
            "md5_status": (
                "MATCHED_OFFICIAL_METADATA"
                if self.expected_md5 is not None
                else "SYNTHETIC_TEST_FIXTURE_NOT_OFFICIAL"
            ),
        }


@dataclass(frozen=True)
class SourceMember:
    name: str
    speaker_id: str
    chapter_id: str
    utterance_id: str
    size_bytes: int


@dataclass(frozen=True)
class ArchiveScan:
    members: tuple[SourceMember, ...]
    metadata: Mapping[str, bytes]
    metadata_sha256: Mapping[str, str]
    metadata_sizes: Mapping[str, int]
    total_member_bytes: int
    member_count: int


@dataclass(frozen=True)
class OriginalBase:
    root: Path
    manifest_path: Path
    manifest_sha256: str
    all_rows: tuple[dict[str, Any], ...]
    train_validation_rows: tuple[dict[str, Any], ...]
    features: np.ndarray
    feature_ids: tuple[str, ...]
    features_sha256: str
    provenance: Mapping[str, Any]

    @property
    def feature_by_id(self) -> dict[str, np.ndarray]:
        return {
            utterance_id: self.features[index]
            for index, utterance_id in enumerate(self.feature_ids)
        }


@dataclass(frozen=True)
class V2Manifest:
    path: Path
    manifest_sha256: str
    dataset: str
    seed: int
    rows: tuple[dict[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "format_version": FORMAT_VERSION,
            "dataset": self.dataset,
            "seed": self.seed,
            "rows": [dict(row) for row in self.rows],
        }


def sha256_file(path: str | Path) -> str:
    """Return SHA-256 for one regular file without loading it into memory."""

    file_path = Path(path)
    if file_path.is_symlink() or not file_path.is_file():
        raise ExpandedDataError(f"expected regular file: {file_path}")
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_tree(root: str | Path) -> dict[str, str]:
    """Hash every regular file below one immutable input directory."""

    directory = Path(root)
    if directory.is_symlink() or not directory.is_dir():
        raise ExpandedDataError(f"expected immutable input directory: {directory}")
    snapshot: dict[str, str] = {}
    for path in sorted(directory.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise ExpandedDataError(f"symlink in immutable input tree: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ExpandedDataError(f"non-regular input in immutable tree: {path}")
        snapshot[str(path.resolve())] = sha256_file(path)
    return snapshot


def _assert_tree_snapshot(
    root: str | Path, expected: Mapping[str, str], *, label: str
) -> None:
    observed = _snapshot_tree(root)
    if observed != dict(expected):
        raise ExpandedDataError(f"immutable {label} inputs changed during preparation")


def _assert_file_snapshot(path: str | Path, expected: str, *, label: str) -> None:
    observed = sha256_file(path)
    if observed != expected:
        raise ExpandedDataError(f"immutable {label} changed during preparation")


def _hash_file_pair(path: Path, *, expected_md5: str | None) -> ArchiveDigest:
    if path.is_symlink() or not path.is_file():
        raise ExpandedDataError(f"archive is not a regular file: {path}")
    size_bytes = path.stat().st_size
    if size_bytes <= 0 or size_bytes > MAX_ARCHIVE_BYTES:
        raise ExpandedDataError(
            f"archive size {size_bytes} outside 1..{MAX_ARCHIVE_BYTES} bytes"
        )
    md5 = hashlib.md5()  # noqa: S324 - OpenSLR publishes MD5 for this archive.
    sha256 = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            md5.update(chunk)
            sha256.update(chunk)
    observed_md5 = md5.hexdigest()
    if expected_md5 is not None and observed_md5 != expected_md5.lower():
        raise ExpandedDataError(
            f"archive MD5 mismatch: expected {expected_md5.lower()}, "
            f"observed {observed_md5}"
        )
    return ArchiveDigest(
        path=str(path.resolve()),
        size_bytes=size_bytes,
        md5=observed_md5,
        sha256=sha256.hexdigest(),
        expected_md5=expected_md5.lower() if expected_md5 is not None else None,
    )


def hash_archive(
    path: str | Path, *, expected_md5: str | None = TRAIN_ARCHIVE_MD5
) -> ArchiveDigest:
    """Hash and gate an archive before ``tarfile`` is allowed to read it."""

    if expected_md5 is not None and len(expected_md5) != 32:
        raise ValueError("expected_md5 must be a 32-character hexadecimal string")
    return _hash_file_pair(Path(path), expected_md5=expected_md5)


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ExpandedDataError(f"cannot read JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise ExpandedDataError(f"JSON object required: {path}")
    return value


def _safe_relative(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ExpandedDataError(f"unsafe {field}")
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ExpandedDataError(f"unsafe {field}: {value!r}")
    return relative.as_posix()


def _contained_file(root: Path, relative: str, *, require_hash: str | None = None) -> Path:
    safe = _safe_relative(relative, field="relative path")
    path = root.joinpath(*PurePosixPath(safe).parts)
    cursor = root
    for part in PurePosixPath(safe).parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ExpandedDataError(f"symlink in prepared tree: {relative}")
    if not path.is_file():
        raise ExpandedDataError(f"missing prepared file: {path}")
    if require_hash is not None:
        observed = sha256_file(path)
        if observed != require_hash:
            raise ExpandedDataError(
                f"hash mismatch for {relative}: expected {require_hash}, observed {observed}"
            )
    return path


def _scalar_string(value: Any, *, field: str) -> str:
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str) or not value:
        raise ExpandedDataError(f"{field} must be a non-empty string scalar")
    return value


def _validate_row_shape(row: Any, *, allowed_splits: frozenset[str]) -> dict[str, Any]:
    if not isinstance(row, dict) or set(row) != set(V2_ROW_FIELDS):
        raise ExpandedDataError("row fields do not exactly match the v2 schema")
    result = dict(row)
    for key in ("utterance_id", "speaker_id", "chapter_id"):
        if not isinstance(result[key], str) or not result[key] or "/" in result[key]:
            raise ExpandedDataError(f"invalid row {key}")
    if not isinstance(result["split"], str) or result["split"] not in allowed_splits:
        raise ExpandedDataError(f"unsupported split: {result['split']!r}")
    result["wav_path"] = _safe_relative(result["wav_path"], field="wav_path")
    for key in ("wav_sha256", "source_sha256"):
        value = result[key]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)
        ):
            raise ExpandedDataError(f"invalid {key}")
    for key in ("input_samples", "padded_samples", "truncated_samples"):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ExpandedDataError(f"invalid {key}")
    if result["input_samples"] <= 0:
        raise ExpandedDataError("input_samples must be positive")
    expected_padded = max(0, NUM_SAMPLES - result["input_samples"])
    expected_truncated = max(0, result["input_samples"] - NUM_SAMPLES)
    if result["padded_samples"] != expected_padded:
        raise ExpandedDataError("padded_samples does not match the fixed window")
    if result["truncated_samples"] != expected_truncated:
        raise ExpandedDataError("truncated_samples is inconsistent with input_samples")
    return result


def _validate_v2_counts(rows: list[dict[str, Any]], dataset: str, *, test_fixture: bool) -> None:
    if dataset not in {TRAIN_DATASET, EXPANDED_DATASET}:
        raise ExpandedDataError(f"unsupported v2 dataset tag: {dataset!r}")
    if test_fixture:
        if not rows:
            raise ExpandedDataError("fixture manifest cannot be empty")
        train_speakers = {row["speaker_id"] for row in rows if row["split"] == "train"}
        validation_speakers = {
            row["speaker_id"] for row in rows if row["split"] == "validation"
        }
        if train_speakers & validation_speakers:
            raise ExpandedDataError("train/validation speakers overlap")
        return
    counts = Counter(row["split"] for row in rows)
    speaker_counts = Counter(row["speaker_id"] for row in rows)
    if dataset == TRAIN_DATASET:
        if counts != Counter({"train": 2000}):
            raise ExpandedDataError(f"selected dataset counts are {counts}, expected 2000 train")
        if len(speaker_counts) != 100 or set(speaker_counts.values()) != {20}:
            raise ExpandedDataError("selected dataset must contain 100 speakers x 20 rows")
    else:
        if counts != Counter({"train": 2400, "validation": 60}):
            raise ExpandedDataError(
                f"expanded dataset counts are {counts}, expected 2400 train + 60 validation"
            )
        train_speakers = {row["speaker_id"] for row in rows if row["split"] == "train"}
        validation_speakers = {
            row["speaker_id"] for row in rows if row["split"] == "validation"
        }
        if len(train_speakers) != 120 or len(validation_speakers) != 5:
            raise ExpandedDataError("expanded speaker counts are not 120 train and 5 validation")
        if train_speakers & validation_speakers:
            raise ExpandedDataError("train/validation speakers overlap")
        train_counts = Counter(row["speaker_id"] for row in rows if row["split"] == "train")
        if set(train_counts.values()) != {20}:
            raise ExpandedDataError("expanded train speakers must each have 20 rows")
        if Counter(row["speaker_id"] for row in rows if row["split"] == "validation"):
            validation_counts = Counter(
                row["speaker_id"] for row in rows if row["split"] == "validation"
            )
            if set(validation_counts.values()) != {12}:
                raise ExpandedDataError("each validation speaker must have 12 rows")


def load_v2_manifest(
    path: str | Path,
    *,
    expected_dataset: str | None = None,
    verify_files: bool = True,
    test_fixture: bool = False,
) -> V2Manifest:
    """Validate a v2 manifest independently of the fixed pilot validators."""

    manifest_path = Path(path)
    raw = _json_object(manifest_path)
    if set(raw) != {"format_version", "dataset", "seed", "rows"}:
        raise ExpandedDataError("v2 manifest top-level fields are not exact")
    if raw["format_version"] != FORMAT_VERSION or raw["seed"] != SEED:
        raise ExpandedDataError("unsupported v2 format or seed")
    dataset = raw["dataset"]
    if expected_dataset is not None and dataset != expected_dataset:
        raise ExpandedDataError(f"unexpected dataset tag: {dataset!r}")
    if not isinstance(dataset, str):
        raise ExpandedDataError("dataset tag must be a string")
    if not isinstance(raw["rows"], list):
        raise ExpandedDataError("manifest rows must be a list")
    rows = [_validate_row_shape(row, allowed_splits=V2_SPLITS) for row in raw["rows"]]
    ids = [row["utterance_id"] for row in rows]
    wav_hashes = [row["wav_sha256"] for row in rows]
    source_hashes = [row["source_sha256"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ExpandedDataError("duplicate utterance_id")
    if len(set(wav_hashes)) != len(wav_hashes):
        raise ExpandedDataError("duplicate wav_sha256")
    if len(set(source_hashes)) != len(source_hashes):
        raise ExpandedDataError("duplicate source_sha256")
    _validate_v2_counts(rows, dataset, test_fixture=test_fixture)
    if verify_files:
        root = manifest_path.parent
        for row in rows:
            _contained_file(root, row["wav_path"], require_hash=row["wav_sha256"])
    return V2Manifest(
        path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        dataset=dataset,
        seed=raw["seed"],
        rows=tuple(rows),
    )


def validate_v2_manifest(
    path: str | Path,
    *,
    expected_dataset: str | None = None,
    verify_files: bool = True,
    test_fixture: bool = False,
) -> V2Manifest:
    """Stable alias for callers that prefer a verb-named validator."""

    return load_v2_manifest(
        path,
        expected_dataset=expected_dataset,
        verify_files=verify_files,
        test_fixture=test_fixture,
    )


def _read_pcm16_wav(path: Path) -> np.ndarray:
    try:
        with wave.open(str(path), "rb") as handle:
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            rate = handle.getframerate()
            frame_count = handle.getnframes()
            payload = handle.readframes(frame_count)
    except (OSError, wave.Error) as exc:
        raise ExpandedDataError(f"invalid WAV: {path}") from exc
    if channels != 1 or sample_width != 2 or rate != SAMPLE_RATE:
        raise ExpandedDataError(f"WAV is not mono PCM16/16kHz: {path}")
    values = np.frombuffer(payload, dtype="<i2")
    if values.size != frame_count:
        raise ExpandedDataError(f"WAV frame count mismatch: {path}")
    return values.copy()


def _validate_legacy_row(row: Any) -> dict[str, Any]:
    return _validate_row_shape(
        row,
        allowed_splits=frozenset(
            {
                "train",
                "validation",
                "calibration_enroll",
                "calibration_query",
                "test_enroll",
                "test_query",
            }
        ),
    )


def load_original_base(
    original_prepared_root: str | Path, *, test_fixture: bool = False
) -> OriginalBase:
    """Read the immutable old v1 prepared assets without its fixed-count loader."""

    root = Path(original_prepared_root)
    manifest_path = root / "manifest.json"
    raw = _json_object(manifest_path)
    if set(raw) != {"format_version", "dataset", "seed", "rows"}:
        raise ExpandedDataError("original manifest fields are not exact")
    if raw["format_version"] != 1 or raw["dataset"] != OLD_DATASET or raw["seed"] != SEED:
        raise ExpandedDataError("original manifest is not the pinned v1 pilot")
    if not isinstance(raw["rows"], list):
        raise ExpandedDataError("original rows must be a list")
    rows = [_validate_legacy_row(row) for row in raw["rows"]]
    ids = [row["utterance_id"] for row in rows]
    if not ids or len(set(ids)) != len(ids):
        raise ExpandedDataError("original manifest has duplicate/empty IDs")
    speakers_by_split: dict[str, set[str]] = {}
    for row in rows:
        speakers_by_split.setdefault(row["split"], set()).add(row["speaker_id"])
        _contained_file(root, row["wav_path"], require_hash=row["wav_sha256"])
        pcm = _read_pcm16_wav(root / row["wav_path"])
        expected_length = min(row["input_samples"], NUM_SAMPLES)
        if pcm.size != NUM_SAMPLES or expected_length <= 0:
            raise ExpandedDataError("original WAV does not contain the fixed window")
    if not test_fixture:
        expected = Counter(
            {
                "train": 400,
                "validation": 60,
                "calibration_enroll": 15,
                "calibration_query": 45,
                "test_enroll": 30,
                "test_query": 90,
            }
        )
        if Counter(row["split"] for row in rows) != expected:
            raise ExpandedDataError("original v1 row counts are not the verified 640-row pilot")
        if speakers_by_split.get("train", set()) & speakers_by_split.get("validation", set()):
            raise ExpandedDataError("original train/validation speakers overlap")
    features_path = root / "features.npz"
    if features_path.is_symlink() or not features_path.is_file():
        raise ExpandedDataError("original features.npz is missing")
    try:
        with np.load(features_path, allow_pickle=False) as archive:
            if set(archive.files) != {"ids", "features", "manifest_sha256"}:
                raise ExpandedDataError("original feature keys are not exact")
            feature_ids_array = np.asarray(archive["ids"])
            features = np.asarray(archive["features"])
            feature_manifest_hash = _scalar_string(
                archive["manifest_sha256"], field="original feature manifest hash"
            )
    except (OSError, ValueError, KeyError) as exc:
        raise ExpandedDataError("cannot read original features") from exc
    if feature_ids_array.ndim != 1 or feature_ids_array.dtype.kind not in {"U", "S"}:
        raise ExpandedDataError("original feature IDs must be one-dimensional strings")
    feature_ids = tuple(str(item) for item in feature_ids_array.tolist())
    if feature_ids != tuple(ids) or feature_manifest_hash != sha256_file(manifest_path):
        raise ExpandedDataError("original feature IDs or manifest binding mismatch")
    if features.dtype != np.float32 or features.shape != (len(rows),) + FEATURE_SHAPE:
        raise ExpandedDataError("original feature shape/dtype mismatch")
    if not np.isfinite(features).all():
        raise ExpandedDataError("original features contain non-finite values")
    provenance_path = root / "provenance.json"
    provenance = _json_object(provenance_path)
    if not test_fixture and not {
        "source",
        "selection",
        "audio_policy",
        "artifacts",
    }.issubset(provenance):
        raise ExpandedDataError("original provenance is incomplete")
    artifact_hashes = provenance.get("artifacts")
    if not isinstance(artifact_hashes, dict):
        raise ExpandedDataError("original provenance artifacts must be an object")
    for key, artifact_path in (
        ("manifest_sha256", manifest_path),
        ("features_sha256", features_path),
    ):
        recorded = artifact_hashes.get(key)
        if recorded is None and test_fixture:
            continue
        if (
            not isinstance(recorded, str)
            or len(recorded) != 64
            or any(char not in "0123456789abcdef" for char in recorded)
        ):
            raise ExpandedDataError(f"original provenance artifact {key} is invalid")
        observed = sha256_file(artifact_path)
        if recorded != observed:
            raise ExpandedDataError(
                f"original provenance artifact {key} mismatch: "
                f"expected {recorded}, observed {observed}"
            )
    train_validation = tuple(
        row for row in rows if row["split"] in {"train", "validation"}
    )
    return OriginalBase(
        root=root,
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        all_rows=tuple(rows),
        train_validation_rows=train_validation,
        features=features.copy(),
        feature_ids=feature_ids,
        features_sha256=sha256_file(features_path),
        provenance=provenance,
    )


def _validate_tar_name(member: tarfile.TarInfo) -> tuple[str, str | None]:
    name = member.name
    if not name or "\x00" in name or "\\" in name:
        raise ExpandedDataError(f"unsafe archive member name: {name!r}")
    path = PurePosixPath(name)
    parts = path.parts
    if path.is_absolute() or any(part in {"", ".", ".."} for part in parts):
        raise ExpandedDataError(f"unsafe archive member path: {name!r}")
    if member.issym() or member.islnk() or member.isdev() or not (member.isdir() or member.isfile()):
        raise ExpandedDataError(f"archive member is not a regular file/directory: {name!r}")
    if member.size < 0 or member.size > MAX_MEMBER_BYTES:
        raise ExpandedDataError(f"archive member too large: {name!r}")
    if parts == ("LibriSpeech",) or (
        len(parts) == 2 and parts[0] == "LibriSpeech" and parts[1] == "train-clean-100"
    ):
        if not member.isdir():
            raise ExpandedDataError(f"archive root must be a directory: {name!r}")
        return name, None
    if len(parts) == 2 and parts[0] == "LibriSpeech" and parts[1] in TOP_LEVEL_METADATA:
        if not member.isfile():
            raise ExpandedDataError(f"metadata member must be regular: {name!r}")
        return name, parts[1]
    if parts[:2] != ("LibriSpeech", "train-clean-100"):
        raise ExpandedDataError(f"unexpected archive root: {name!r}")
    if member.isdir() and len(parts) in {3, 4}:
        return name, None
    if not member.isfile() or len(parts) != 5:
        raise ExpandedDataError(f"unexpected archive member: {name!r}")
    speaker_id, chapter_id, filename = parts[2], parts[3], parts[4]
    if not speaker_id.isdigit() or not chapter_id.isdigit():
        raise ExpandedDataError(f"invalid speaker/chapter path: {name!r}")
    if filename.endswith(".trans.txt"):
        if not filename.startswith(f"{speaker_id}-{chapter_id}"):
            raise ExpandedDataError(f"invalid transcription name: {name!r}")
        return name, None
    if not filename.endswith(".flac"):
        raise ExpandedDataError(f"unexpected audio suffix: {name!r}")
    utterance_id = filename[:-5]
    utterance_parts = utterance_id.split("-")
    if (
        len(utterance_parts) != 3
        or utterance_parts[0] != speaker_id
        or utterance_parts[1] != chapter_id
        or not utterance_parts[2].isdigit()
    ):
        raise ExpandedDataError(f"invalid FLAC utterance name: {name!r}")
    del utterance_id
    return name, None


def _scan_archive(path: Path) -> ArchiveScan:
    names: set[str] = set()
    file_names: set[str] = set()
    metadata: dict[str, bytes] = {}
    metadata_sha256: dict[str, str] = {}
    metadata_sizes: dict[str, int] = {}
    audio: list[SourceMember] = []
    total_size = 0
    member_count = 0
    try:
        archive = tarfile.open(path, mode="r|gz")
    except (OSError, tarfile.TarError) as exc:
        raise ExpandedDataError(f"cannot open verified tar archive: {path}") from exc
    with archive:
        for member in archive:
            member_count += 1
            if member_count > MAX_MEMBER_COUNT:
                raise ExpandedDataError("archive contains too many members")
            name, metadata_name = _validate_tar_name(member)
            if name in names:
                raise ExpandedDataError(f"duplicate archive member: {name}")
            names.add(name)
            total_size += member.size
            if total_size > MAX_DECOMPRESSED_BYTES:
                raise ExpandedDataError("archive decompressed-size bound exceeded")
            if member.isfile():
                file_names.add(name)
            if metadata_name is not None:
                source = archive.extractfile(member)
                if source is None:
                    raise ExpandedDataError(f"cannot read metadata member: {name}")
                payload = source.read(MAX_METADATA_BYTES + 1)
                if len(payload) != member.size or len(payload) > MAX_METADATA_BYTES:
                    raise ExpandedDataError(f"metadata member exceeds bounded read: {name}")
                metadata[metadata_name] = payload
                metadata_sha256[metadata_name] = hashlib.sha256(payload).hexdigest()
                metadata_sizes[metadata_name] = len(payload)
            elif name.endswith(".flac"):
                parts = PurePosixPath(name).parts
                audio.append(
                    SourceMember(
                        name=name,
                        speaker_id=parts[2],
                        chapter_id=parts[3],
                        utterance_id=parts[4][:-5],
                        size_bytes=member.size,
                    )
                )
    if set(metadata) != TOP_LEVEL_METADATA:
        raise ExpandedDataError(
            f"archive metadata set is {set(metadata)}, expected {set(TOP_LEVEL_METADATA)}"
        )
    metadata_speakers = _metadata_speakers(metadata["SPEAKERS.TXT"])
    audio_speakers = {member.speaker_id for member in audio}
    if not audio_speakers.issubset(metadata_speakers):
        raise ExpandedDataError("audio speaker IDs are absent from SPEAKERS.TXT")
    for file_name in file_names:
        parts = PurePosixPath(file_name).parts
        for index in range(1, len(parts)):
            if PurePosixPath(*parts[:index]).as_posix() in file_names:
                raise ExpandedDataError(f"archive file shadows a directory: {file_name}")
    return ArchiveScan(
        members=tuple(audio),
        metadata=metadata,
        metadata_sha256=metadata_sha256,
        metadata_sizes=metadata_sizes,
        total_member_bytes=total_size,
        member_count=member_count,
    )


def _metadata_speakers(payload: bytes) -> set[str]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ExpandedDataError("SPEAKERS.TXT is not UTF-8") from exc
    speakers: set[str] = set()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        fields = [field.strip() for field in line.split("|")]
        if len(fields) < 3 or not fields[0].isdigit():
            raise ExpandedDataError(f"malformed SPEAKERS.TXT row: {line!r}")
        if fields[2] != "train-clean-100":
            continue
        if fields[0] in speakers:
            raise ExpandedDataError(f"duplicate speaker metadata: {fields[0]}")
        speakers.add(fields[0])
    if not speakers:
        raise ExpandedDataError("SPEAKERS.TXT has no train-clean-100 speakers")
    return speakers


def _choose_members(
    scan: ArchiveScan,
    *,
    excluded_speakers: set[str],
    test_fixture: bool,
    fixture_speakers: int | None,
    fixture_utterances_per_speaker: int | None,
) -> tuple[list[str], list[SourceMember], list[str]]:
    per_speaker = fixture_utterances_per_speaker if test_fixture else 20
    speaker_count = fixture_speakers if test_fixture else 100
    if per_speaker is None:
        per_speaker = 2
    if speaker_count is None:
        speaker_count = 2
    if per_speaker <= 0 or speaker_count <= 0:
        raise ValueError("fixture selection counts must be positive")
    archive_speakers = _metadata_speakers(scan.metadata["SPEAKERS.TXT"])
    by_speaker: dict[str, list[SourceMember]] = {}
    for member in scan.members:
        by_speaker.setdefault(member.speaker_id, []).append(member)
    eligible = sorted(
        speaker_id
        for speaker_id in archive_speakers
        if speaker_id not in excluded_speakers
        and len(by_speaker.get(speaker_id, [])) >= per_speaker
    )
    if len(eligible) < speaker_count:
        raise ExpandedDataError(
            f"only {len(eligible)} eligible speakers, need {speaker_count}"
        )
    selected = [str(value) for value in np.random.default_rng(SEED).permutation(eligible)[:speaker_count]]
    selected_members: list[SourceMember] = []
    for speaker_id in selected:
        members = sorted(by_speaker[speaker_id], key=lambda item: item.utterance_id)
        selected_members.extend(members[:per_speaker])
    if len({member.utterance_id for member in selected_members}) != len(selected_members):
        raise ExpandedDataError("selected archive utterance IDs are not unique")
    return selected, selected_members, eligible


def _write_pcm_wav(path: Path, pcm: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(np.asarray(pcm, dtype="<i2").tobytes())


def _decode_flac(payload: bytes, *, source_name: str) -> tuple[np.ndarray, int, int]:
    try:
        import soundfile as sf

        samples, sample_rate = sf.read(
            io.BytesIO(payload), dtype="int16", always_2d=False
        )
    except Exception as exc:  # soundfile exposes several backend-specific errors.
        raise ExpandedDataError(f"cannot decode FLAC member: {source_name}") from exc
    values = np.asarray(samples)
    if values.ndim != 1 or values.dtype.kind not in {"i", "u"}:
        raise ExpandedDataError(f"FLAC is not mono integer PCM: {source_name}")
    if int(sample_rate) != SAMPLE_RATE or values.size <= 0:
        raise ExpandedDataError(f"FLAC sample format is not 16kHz mono: {source_name}")
    input_samples = int(values.size)
    truncated = max(0, input_samples - NUM_SAMPLES)
    cropped = np.asarray(values[:NUM_SAMPLES], dtype="<i2")
    padded = max(0, NUM_SAMPLES - int(cropped.size))
    if padded:
        cropped = np.pad(cropped, (0, padded), mode="constant")
    return cropped.astype("<i2", copy=False), input_samples, truncated


def _extract_selected(
    archive_path: Path,
    selected_members: Iterable[SourceMember],
    destination_root: Path,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    selected_members = tuple(selected_members)
    by_name = {member.name: member for member in selected_members}
    rows_by_name: dict[str, dict[str, Any]] = {}
    features_by_name: dict[str, np.ndarray] = {}
    seen: set[str] = set()
    with tarfile.open(archive_path, mode="r|gz") as archive:
        for member in archive:
            _validate_tar_name(member)
            selected = by_name.get(member.name)
            if selected is None:
                continue
            source = archive.extractfile(member)
            if source is None:
                raise ExpandedDataError(f"cannot extract selected member: {member.name}")
            payload = source.read(member.size + 1)
            if len(payload) != member.size:
                raise ExpandedDataError(f"selected member changed while reading: {member.name}")
            pcm, input_samples, truncated = _decode_flac(payload, source_name=member.name)
            relative = (
                Path("wav")
                / "train-clean-100"
                / selected.speaker_id
                / selected.chapter_id
                / f"{selected.utterance_id}.wav"
            )
            target = destination_root / relative
            _write_pcm_wav(target, pcm)
            wav_hash = sha256_file(target)
            row = {
                "utterance_id": selected.utterance_id,
                "speaker_id": selected.speaker_id,
                "chapter_id": selected.chapter_id,
                "split": "train",
                "wav_path": relative.as_posix(),
                "wav_sha256": wav_hash,
                "source_sha256": hashlib.sha256(payload).hexdigest(),
                "input_samples": input_samples,
                "padded_samples": max(0, NUM_SAMPLES - min(input_samples, NUM_SAMPLES)),
                "truncated_samples": truncated,
            }
            rows_by_name[member.name] = row
            features_by_name[member.name] = np.asarray(
                log_mel(pcm.astype(np.float32) / 32768.0)[..., None], dtype=np.float32
            )
            seen.add(member.name)
    if seen != set(by_name):
        raise ExpandedDataError("selected archive members were not all extracted")
    if not rows_by_name:
        raise ExpandedDataError("selected extraction produced no rows")
    ordered_names = [member.name for member in selected_members]
    rows = [rows_by_name[name] for name in ordered_names]
    features = np.stack([features_by_name[name] for name in ordered_names]).astype(
        np.float32, copy=False
    )
    if features.shape[1:] != FEATURE_SHAPE or not np.isfinite(features).all():
        raise ExpandedDataError("frontend returned invalid feature shape or values")
    return rows, features


def _copy_old_rows(
    base: OriginalBase, rows: Iterable[dict[str, Any]], destination_root: Path
) -> list[dict[str, Any]]:
    copied: list[dict[str, Any]] = []
    for source_row in rows:
        source = _contained_file(
            base.root, source_row["wav_path"], require_hash=source_row["wav_sha256"]
        )
        relative = (
            Path("wav")
            / "original"
            / source_row["speaker_id"]
            / source_row["chapter_id"]
            / f"{source_row['utterance_id']}.wav"
        )
        target = destination_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if sha256_file(target) != source_row["wav_sha256"]:
            raise ExpandedDataError(f"old WAV copy changed: {source_row['utterance_id']}")
        copied_row = dict(source_row)
        copied_row["wav_path"] = relative.as_posix()
        copied.append(copied_row)
    return copied


def _copy_new_rows(source_root: Path, destination_root: Path, rows: Iterable[dict[str, Any]]) -> None:
    for row in rows:
        source = _contained_file(source_root, row["wav_path"], require_hash=row["wav_sha256"])
        target = destination_root / row["wav_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if sha256_file(target) != row["wav_sha256"]:
            raise ExpandedDataError(f"new WAV copy changed: {row['utterance_id']}")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_bundle(
    root: Path,
    *,
    dataset: str,
    rows: list[dict[str, Any]],
    features: np.ndarray,
    provenance: dict[str, Any],
    test_fixture: bool,
) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    features_path = root / "features.npz"
    if manifest_path.exists() or features_path.exists() or (root / "provenance.json").exists():
        raise ExpandedDataError(f"refusing to overwrite bundle: {root}")
    manifest_value = {
        "format_version": FORMAT_VERSION,
        "dataset": dataset,
        "seed": SEED,
        "rows": rows,
    }
    _write_json(manifest_path, manifest_value)
    manifest_hash = sha256_file(manifest_path)
    values = np.asarray(features, dtype=np.float32)
    if values.shape != (len(rows),) + FEATURE_SHAPE or not np.isfinite(values).all():
        raise ExpandedDataError(f"invalid features for {dataset}: {values.shape}")
    np.savez_compressed(
        features_path,
        ids=np.asarray([row["utterance_id"] for row in rows], dtype="U"),
        features=values,
        manifest_sha256=np.asarray(manifest_hash),
    )
    artifact_info = dict(provenance.get("artifacts", {}))
    artifact_info.update(
        {
            "manifest_sha256": manifest_hash,
            "features_sha256": sha256_file(features_path),
            "prepared_wav_index_sha256": hashlib.sha256(
                json.dumps(
                    [
                        [row["utterance_id"], row["wav_path"], row["wav_sha256"]]
                        for row in rows
                    ],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "rows": len(rows),
            "features_shape": list(values.shape),
        }
    )
    provenance["artifacts"] = artifact_info
    provenance_path = root / "provenance.json"
    _write_json(provenance_path, provenance)
    # Validate the just-written tree, including all WAV hashes, before it can publish.
    load_v2_manifest(
        manifest_path,
        expected_dataset=dataset,
        verify_files=True,
        test_fixture=test_fixture,
    )
    return {
        "dataset": dataset,
        "root": str(root),
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "features": str(features_path),
        "features_sha256": sha256_file(features_path),
        "provenance": str(provenance_path),
        "provenance_sha256": sha256_file(provenance_path),
        "rows": len(rows),
        "features_shape": list(values.shape),
    }


def _provenance(
    *,
    dataset: str,
    status: str,
    digest: ArchiveDigest,
    scan: ArchiveScan,
    base: OriginalBase,
    original_teacher_path: Path,
    selected_speakers: list[str],
    eligible_speakers: list[str],
    excluded_speakers: set[str],
    frontend_sha256: str,
    base_input_hashes: Mapping[str, str],
    frontend_path: Path,
) -> dict[str, Any]:
    old_source = dict(base.provenance.get("source", {}))
    return {
        "format_version": FORMAT_VERSION,
        "status": status,
        "dataset": dataset,
        "source": {
            "provider": "OpenSLR",
            "url": TRAIN_ARCHIVE_URL,
            "archive": digest.as_dict(),
            "official_expected_md5": TRAIN_ARCHIVE_MD5 if digest.expected_md5 else None,
            "observed_sha256_is_not_signature": True,
            "metadata_sha256": dict(scan.metadata_sha256),
            "metadata_bytes": dict(scan.metadata_sizes),
            "old_source": old_source,
        },
        "selection": {
            "seed": SEED,
            "eligible_speaker_count": len(eligible_speakers),
            "eligible_speaker_ids": eligible_speakers,
            "selected_speaker_ids": selected_speakers,
            "excluded_original_speaker_ids": sorted(excluded_speakers),
            "utterance_policy": "first 20 lexicographic FLAC utterance IDs per selected speaker",
            "chapter_policy": "preserve chapter IDs; no forced cross-chapter guarantee",
        },
        "audio": {
            "sample_rate": SAMPLE_RATE,
            "channels": 1,
            "source_format": "FLAC",
            "output_format": "little-endian PCM16 WAV",
            "window_samples": NUM_SAMPLES,
            "crop": "first 3 seconds",
            "pad": "right zero-pad",
            "frontend": "edgeai.frontend.log_mel",
            "frontend_sha256": frontend_sha256,
        },
        "base_artifacts": {
            "old_manifest_sha256": base.manifest_sha256,
            "old_features_sha256": base.features_sha256,
            "old_provenance_sha256": sha256_file(base.root / "provenance.json"),
            "old_teacher_sha256": sha256_file(original_teacher_path),
            "old_train_validation_rows": len(base.train_validation_rows),
        },
        "input_bindings": {
            "base_prepared_root": str(base.root.resolve()),
            "base_files": dict(sorted(base_input_hashes.items())),
            "original_teacher_path": str(original_teacher_path.resolve()),
            "original_teacher_sha256": sha256_file(original_teacher_path),
            "frontend_path": str(frontend_path.resolve()),
            "frontend_sha256": frontend_sha256,
        },
        "limits": {
            "archive_max_bytes": MAX_ARCHIVE_BYTES,
            "decompressed_member_bytes_max": MAX_DECOMPRESSED_BYTES,
            "member_max_bytes": MAX_MEMBER_BYTES,
            "member_count_max": MAX_MEMBER_COUNT,
        },
        "artifacts": {},
    }


def prepare_expanded_dataset(
    archive_path: str | Path,
    original_prepared_root: str | Path,
    output_root: str | Path,
    *,
    original_teacher_path: str | Path,
    test_fixture: bool = False,
    fixture_speakers: int | None = None,
    fixture_utterances_per_speaker: int | None = None,
) -> dict[str, Any]:
    """Prepare selected-new and old-plus-new bundles atomically.

    The real invocation requires the official archive MD5.  ``test_fixture``
    is intentionally explicit and only exists for small synthetic tar tests;
    its provenance is marked synthetic and cannot be mistaken for OpenSLR.
    """

    archive = Path(archive_path)
    output = Path(output_root)
    if output.exists() or output.is_symlink():
        raise ExpandedDataError(f"refusing to overwrite existing output: {output}")
    base_root = Path(original_prepared_root)
    base_input_hashes = _snapshot_tree(base_root)
    teacher_path = Path(original_teacher_path)
    teacher_sha256 = sha256_file(teacher_path)
    frontend_path = Path(__file__).with_name("frontend.py")
    frontend_sha256 = sha256_file(frontend_path)
    digest = hash_archive(
        archive,
        expected_md5=None if test_fixture else TRAIN_ARCHIVE_MD5,
    )
    base = load_original_base(base_root, test_fixture=test_fixture)
    scan = _scan_archive(archive)
    old_speakers = {row["speaker_id"] for row in base.all_rows}
    selected_speakers, selected_members, eligible = _choose_members(
        scan,
        excluded_speakers=old_speakers,
        test_fixture=test_fixture,
        fixture_speakers=fixture_speakers,
        fixture_utterances_per_speaker=fixture_utterances_per_speaker,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    status = "SYNTHETIC_TEST_FIXTURE" if test_fixture else "REAL_OFFICIAL_SOURCE_PENDING_TEACHER"
    with tempfile.TemporaryDirectory(prefix=".expanded-stage-", dir=str(output.parent)) as temporary:
        stage = Path(temporary) / "bundle"
        selected_root = stage / "selected"
        expanded_root = stage / "expanded"
        selected_rows, selected_features = _extract_selected(
            archive, selected_members, selected_root
        )
        old_rows = _copy_old_rows(base, base.train_validation_rows, expanded_root)
        _copy_new_rows(selected_root, expanded_root, selected_rows)
        expanded_rows = old_rows + selected_rows
        old_feature_map = base.feature_by_id
        old_features = np.stack([old_feature_map[row["utterance_id"]] for row in old_rows]).astype(
            np.float32, copy=False
        )
        expanded_features = np.concatenate((old_features, selected_features), axis=0)
        selected_provenance = _provenance(
            dataset=TRAIN_DATASET,
            status=status,
            digest=digest,
            scan=scan,
            base=base,
            original_teacher_path=teacher_path,
            selected_speakers=selected_speakers,
            eligible_speakers=eligible,
            excluded_speakers=old_speakers,
            frontend_sha256=frontend_sha256,
            base_input_hashes=base_input_hashes,
            frontend_path=frontend_path,
        )
        expanded_provenance = _provenance(
            dataset=EXPANDED_DATASET,
            status=status,
            digest=digest,
            scan=scan,
            base=base,
            original_teacher_path=teacher_path,
            selected_speakers=selected_speakers,
            eligible_speakers=eligible,
            excluded_speakers=old_speakers,
            frontend_sha256=frontend_sha256,
            base_input_hashes=base_input_hashes,
            frontend_path=frontend_path,
        )
        selected_info = _write_bundle(
            selected_root,
            dataset=TRAIN_DATASET,
            rows=selected_rows,
            features=selected_features,
            provenance=selected_provenance,
            test_fixture=test_fixture,
        )
        expanded_info = _write_bundle(
            expanded_root,
            dataset=EXPANDED_DATASET,
            rows=expanded_rows,
            features=expanded_features,
            provenance=expanded_provenance,
            test_fixture=test_fixture,
        )
        _assert_tree_snapshot(base_root, base_input_hashes, label="base")
        _assert_file_snapshot(teacher_path, teacher_sha256, label="original teacher")
        _assert_file_snapshot(frontend_path, frontend_sha256, label="frontend source")
        final_digest = hash_archive(
            archive,
            expected_md5=None if test_fixture else TRAIN_ARCHIVE_MD5,
        )
        if final_digest != digest:
            raise ExpandedDataError("archive changed between scan and publication")
        stage.rename(output)
    def _published_info(info: dict[str, Any], root: Path) -> dict[str, Any]:
        return {
            **info,
            "root": str(root),
            "manifest": str(root / "manifest.json"),
            "features": str(root / "features.npz"),
            "provenance": str(root / "provenance.json"),
        }

    return {
        "status": status,
        "selected": _published_info(selected_info, output / "selected"),
        "expanded": _published_info(expanded_info, output / "expanded"),
        "archive": digest.as_dict(),
        "rows": {"selected_new": len(selected_rows), "expanded": len(expanded_rows)},
    }


__all__ = [
    "ArchiveDigest",
    "ExpandedDataError",
    "EXPANDED_DATASET",
    "FEATURE_SHAPE",
    "FORMAT_VERSION",
    "NUM_SAMPLES",
    "OriginalBase",
    "SEED",
    "TRAIN_ARCHIVE_MD5",
    "TRAIN_ARCHIVE_URL",
    "TRAIN_DATASET",
    "V2Manifest",
    "hash_archive",
    "load_original_base",
    "load_v2_manifest",
    "prepare_expanded_dataset",
    "sha256_file",
    "validate_v2_manifest",
]
