"""Bounded full train-clean-100 preparation.

This module is additive to the fixed pilot and expanded-pilot paths.  The
full dataset is train-only: every source utterance is represented by
consecutive, non-overlapping 48,000-sample windows and the last window is
right padded with zeroes when necessary.  The archive is never extracted as a
directory.  Both archive passes use ``tarfile``'s streaming mode and keep at
most one member payload in memory.

The real contract is intentionally strict.  A real run must match the
independent train-clean-100 count oracle; synthetic fixtures must explicitly
opt in and are never given official-source provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

import numpy as np

from .frontend import NUM_FRAMES, NUM_MELS, NUM_SAMPLES, SAMPLE_RATE, log_mel


FORMAT_VERSION = 1
DATASET = "librispeech-train-clean-100-full-v1"
SYNTHETIC_DATASET = "synthetic-librispeech-train-clean-100-full-fixture-v1"
ARCHIVE_URL = "https://www.openslr.org/resources/12/train-clean-100.tar.gz"
OFFICIAL_MD5 = "2a93770f6d5c6c964bc36631d331a522"

WINDOW_SAMPLES = 48_000
FEATURE_SHAPE = (NUM_FRAMES, NUM_MELS, 1)
FEATURE_DTYPE = np.dtype(np.float32)
PCM_DTYPE = np.dtype("<i2")

EXPECTED_MEMBER_COUNT = 29_966
EXPECTED_AUDIO_COUNT = 28_539
EXPECTED_SPEAKER_COUNT = 251
EXPECTED_SOURCE_FRAMES = 5_794_034_668
EXPECTED_WINDOW_COUNT = 135_438
EXPECTED_PADDING_SAMPLES = 706_989_332
EXPECTED_MEMBER_BYTES = 6_618_476_609

MAX_ARCHIVE_BYTES = 8_000_000_000
MAX_DECOMPRESSED_BYTES = 20_000_000_000
MAX_MEMBER_COUNT = 1_000_000
MAX_MEMBER_BYTES = 100_000_000
MAX_METADATA_BYTES = 16_000_000
PROGRESS_INTERVAL = 1_000
HASH_CHUNK_BYTES = 1024 * 1024
FEATURE_CHECK_CHUNK = 256

_METADATA_NAMES = frozenset(
    {"LICENSE.TXT", "README.TXT", "CHAPTERS.TXT", "SPEAKERS.TXT", "BOOKS.TXT"}
)
_AUDIO_PREFIX = ("LibriSpeech", "train-clean-100")


class FullDataError(ValueError):
    """Raised when full-data preparation or validation violates the contract."""


@dataclass(frozen=True)
class FullDataset:
    """Validated full dataset backed by read-only NumPy memory maps."""

    root: Path
    manifest: dict[str, Any]
    rows: list[dict[str, Any]]
    pcm: np.ndarray
    features: np.ndarray
    manifest_sha256: str


@dataclass(frozen=True)
class _ArchiveDigest:
    path: Path
    size_bytes: int
    md5: str
    sha256: str
    expected_md5: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "bytes": self.size_bytes,
            "md5": self.md5,
            "sha256": self.sha256,
            "expected_md5": self.expected_md5,
            "md5_status": (
                "MATCHED_OFFICIAL_METADATA"
                if self.expected_md5 is not None
                else "SYNTHETIC_TEST_FIXTURE_NOT_OFFICIAL"
            ),
        }


@dataclass(frozen=True)
class _SourceMember:
    name: str
    speaker_id: str
    chapter_id: str
    utterance_id: str
    frames: int
    size_bytes: int
    source_sha256: str
    window_count: int = 0
    row_start: int = 0


@dataclass(frozen=True)
class _ScannedArchive:
    digest: _ArchiveDigest
    members: tuple[_SourceMember, ...]
    member_count: int
    total_member_bytes: int
    metadata_sha256: dict[str, str]
    member_sha256: dict[str, str]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise FullDataError(f"expected regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_bytes(path: Path) -> int:
    try:
        return int(path.stat().st_size)
    except OSError as exc:
        raise FullDataError(f"cannot stat file: {path}") from exc


def _archive_digest(path: Path, *, expected_md5: str | None) -> _ArchiveDigest:
    if path.is_symlink() or not path.is_file():
        raise FullDataError(f"archive is not a regular file: {path}")
    size = _file_bytes(path)
    if size <= 0 or size > MAX_ARCHIVE_BYTES:
        raise FullDataError(
            f"archive size {size} outside 1..{MAX_ARCHIVE_BYTES} bytes"
        )
    md5 = hashlib.md5()  # noqa: S324 - the source publishes an MD5 gate.
    sha256 = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
            md5.update(chunk)
            sha256.update(chunk)
    observed_md5 = md5.hexdigest()
    if expected_md5 is not None and observed_md5 != expected_md5.lower():
        raise FullDataError(
            f"archive MD5 mismatch: expected {expected_md5.lower()}, "
            f"observed {observed_md5}"
        )
    return _ArchiveDigest(
        path=path.resolve(),
        size_bytes=size,
        md5=observed_md5,
        sha256=sha256.hexdigest(),
        expected_md5=expected_md5.lower() if expected_md5 else None,
    )


def _soundfile_module() -> Any:
    try:
        import soundfile as sf
    except (ImportError, OSError) as exc:
        raise FullDataError(
            "soundfile is required for FLAC header/decode preparation"
        ) from exc
    return sf


def _read_member_payload(
    extracted: Any, size_bytes: int, *, name: str
) -> tuple[bytes, str]:
    if extracted is None:
        raise FullDataError(f"archive member has no readable payload: {name}")
    if size_bytes < 0 or size_bytes > MAX_MEMBER_BYTES:
        raise FullDataError(f"archive member exceeds bound: {name}")
    digest = hashlib.sha256()
    output = io.BytesIO()
    remaining = size_bytes
    while remaining:
        chunk = extracted.read(min(HASH_CHUNK_BYTES, remaining))
        if not chunk:
            raise FullDataError(f"truncated archive member: {name}")
        if len(chunk) > remaining:
            raise FullDataError(f"archive member over-read: {name}")
        digest.update(chunk)
        output.write(chunk)
        remaining -= len(chunk)
    if extracted.read(1):
        raise FullDataError(f"archive member has bytes beyond declared size: {name}")
    return output.getvalue(), digest.hexdigest()


def _safe_member_name(name: object) -> str:
    if not isinstance(name, str) or not name or "\x00" in name or "\\" in name:
        raise FullDataError(f"unsafe archive member name: {name!r}")
    directory_name = name[:-1] if name.endswith("/") else name
    if not directory_name:
        raise FullDataError("empty archive member name")
    if directory_name.startswith("/"):
        raise FullDataError(f"absolute archive member name: {name!r}")
    parts = directory_name.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise FullDataError(f"traversal archive member name: {name!r}")
    normalized = PurePosixPath(directory_name).as_posix()
    if normalized != directory_name:
        raise FullDataError(f"non-canonical archive member name: {name!r}")
    if parts[0] != "LibriSpeech":
        raise FullDataError(f"unexpected archive root: {name!r}")
    return normalized


def _classify_member(name: str, *, is_directory: bool) -> tuple[str, tuple[str, ...]]:
    parts = tuple(name.split("/"))
    if is_directory:
        return "directory", parts
    if len(parts) == 2 and parts[0] == "LibriSpeech" and parts[1] in _METADATA_NAMES:
        return "metadata", parts
    # The official archive contains one transcript regular member per numeric
    # speaker/chapter.  Keep this whitelist exact: arbitrary ``.trans.txt``
    # paths must not become an accidental extraction surface.
    if (
        len(parts) == 5
        and parts[:2] == _AUDIO_PREFIX
        and parts[2].isascii()
        and parts[2].isdigit()
        and parts[3].isascii()
        and parts[3].isdigit()
        and parts[4] == f"{parts[2]}-{parts[3]}.trans.txt"
    ):
        return "transcript", parts
    if len(parts) == 5 and parts[:2] == _AUDIO_PREFIX and parts[4].endswith(".flac"):
        filename = parts[4][:-5]
        pieces = filename.split("-", 2)
        if len(pieces) != 3 or pieces[0] != parts[2] or pieces[1] != parts[3]:
            raise FullDataError(f"audio member ID/path mismatch: {name!r}")
        if not pieces[2] or not all(piece.isdigit() for piece in pieces):
            raise FullDataError(f"invalid LibriSpeech audio ID: {name!r}")
        return "audio", parts
    raise FullDataError(f"unexpected regular archive member: {name!r}")


def _register_node(
    name: str,
    kind: str,
    nodes: dict[str, str],
    child_counts: dict[str, int],
) -> None:
    if name in nodes:
        raise FullDataError(f"duplicate/shadowing archive member: {name}")
    parts = name.split("/")
    for index in range(1, len(parts)):
        ancestor = "/".join(parts[:index])
        if nodes.get(ancestor) == "file":
            raise FullDataError(f"archive member is below a regular file: {name}")
    if kind == "file" and child_counts.get(name, 0):
        raise FullDataError(f"regular member shadows existing children: {name}")
    nodes[name] = kind
    for index in range(1, len(parts)):
        ancestor = "/".join(parts[:index])
        child_counts[ancestor] = child_counts.get(ancestor, 0) + 1


def _header_frames(payload: bytes, *, name: str) -> int:
    sf = _soundfile_module()
    try:
        with sf.SoundFile(io.BytesIO(payload)) as handle:
            if handle.samplerate != SAMPLE_RATE:
                raise FullDataError(
                    f"{name}: expected {SAMPLE_RATE} Hz, got {handle.samplerate}"
                )
            if handle.channels != 1:
                raise FullDataError(f"{name}: expected mono FLAC")
            if handle.format != "FLAC" or handle.subtype != "PCM_16":
                raise FullDataError(
                    f"{name}: expected FLAC PCM_16, got "
                    f"{handle.format}/{handle.subtype}"
                )
            frames = int(handle.frames)
    except FullDataError:
        raise
    except Exception as exc:
        raise FullDataError(f"cannot read FLAC header: {name}") from exc
    if frames <= 0:
        raise FullDataError(f"{name}: empty FLAC is not allowed")
    return frames


def _iter_secure_members(
    archive_path: Path,
) -> Iterator[tuple[str, str, bytes | None, str | None, int]]:
    """Yield validated members from one bounded streaming tar pass."""

    nodes: dict[str, str] = {}
    child_counts: dict[str, int] = {}
    total_member_bytes = 0
    member_count = 0
    try:
        archive = tarfile.open(archive_path, mode="r|gz")
    except (OSError, tarfile.TarError) as exc:
        raise FullDataError(f"cannot open gzip tar archive: {archive_path}") from exc
    with archive:
        for member in archive:
            member_count += 1
            if member_count > MAX_MEMBER_COUNT:
                raise FullDataError("archive member count exceeds bound")
            if member.size < 0 or member.size > MAX_MEMBER_BYTES:
                raise FullDataError(f"archive member size exceeds bound: {member.name!r}")
            total_member_bytes += int(member.size)
            if total_member_bytes > MAX_DECOMPRESSED_BYTES:
                raise FullDataError("archive decompressed-byte bound exceeded")
            name = _safe_member_name(member.name)
            if member.issym() or member.islnk() or member.isdev() or member.isfifo():
                raise FullDataError(f"links/devices are not allowed: {name}")
            kind, _ = _classify_member(name, is_directory=member.isdir())
            _register_node(
                name,
                "directory" if kind == "directory" else "file",
                nodes,
                child_counts,
            )
            if kind == "directory":
                yield name, kind, None, None, int(member.size)
                continue
            if kind in {"metadata", "transcript"} and member.size > MAX_METADATA_BYTES:
                raise FullDataError(f"metadata member exceeds bound: {name}")
            extracted = archive.extractfile(member)
            payload, source_hash = _read_member_payload(
                extracted, int(member.size), name=name
            )
            yield name, kind, payload, source_hash, int(member.size)


def _validate_real_scan(scan: _ScannedArchive) -> None:
    if scan.member_count != EXPECTED_MEMBER_COUNT:
        raise FullDataError(
            f"real archive member count mismatch: expected {EXPECTED_MEMBER_COUNT}, "
            f"got {scan.member_count}"
        )
    if scan.total_member_bytes != EXPECTED_MEMBER_BYTES:
        raise FullDataError(
            f"real archive member-byte count mismatch: expected "
            f"{EXPECTED_MEMBER_BYTES}, got {scan.total_member_bytes}"
        )
    if len(scan.members) != EXPECTED_AUDIO_COUNT:
        raise FullDataError(
            f"real FLAC count mismatch: expected {EXPECTED_AUDIO_COUNT}, "
            f"got {len(scan.members)}"
        )
    speakers = {member.speaker_id for member in scan.members}
    if len(speakers) != EXPECTED_SPEAKER_COUNT:
        raise FullDataError(
            f"real speaker count mismatch: expected {EXPECTED_SPEAKER_COUNT}, "
            f"got {len(speakers)}"
        )
    total_frames = sum(member.frames for member in scan.members)
    if total_frames != EXPECTED_SOURCE_FRAMES:
        raise FullDataError(
            f"real source-frame mismatch: expected {EXPECTED_SOURCE_FRAMES}, "
            f"got {total_frames}"
        )
    windows = sum((member.frames + WINDOW_SAMPLES - 1) // WINDOW_SAMPLES for member in scan.members)
    if windows != EXPECTED_WINDOW_COUNT:
        raise FullDataError(
            f"real window count mismatch: expected {EXPECTED_WINDOW_COUNT}, got {windows}"
        )
    padding = sum(
        ((WINDOW_SAMPLES - (member.frames % WINDOW_SAMPLES)) % WINDOW_SAMPLES)
        for member in scan.members
    )
    if padding != EXPECTED_PADDING_SAMPLES:
        raise FullDataError(
            f"real padding count mismatch: expected {EXPECTED_PADDING_SAMPLES}, got {padding}"
        )


def _scan_archive(archive_path: Path, *, test_fixture: bool) -> _ScannedArchive:
    expected_md5 = None if test_fixture else OFFICIAL_MD5
    # This gate deliberately precedes any tar reader.
    digest = _archive_digest(archive_path, expected_md5=expected_md5)
    if test_fixture and digest.md5 == OFFICIAL_MD5:
        raise FullDataError("test_fixture=True may not use the official archive")
    members: list[_SourceMember] = []
    metadata_hashes: dict[str, str] = {}
    member_hashes: dict[str, str] = {}
    transcript_chapters: set[tuple[str, str]] = set()
    member_count = 0
    total_member_bytes = 0
    for name, kind, payload, source_hash, size_bytes in _iter_secure_members(archive_path):
        member_count += 1
        total_member_bytes += size_bytes
        if source_hash is not None:
            member_hashes[name] = source_hash
        if kind == "metadata":
            if payload is None or source_hash is None:
                raise FullDataError(f"missing metadata payload: {name}")
            if len(payload) > MAX_METADATA_BYTES:
                raise FullDataError(f"metadata member exceeds bound: {name}")
            metadata_hashes[name.split("/", 1)[1]] = source_hash
            continue
        if kind == "transcript":
            parts = name.split("/")
            transcript_chapters.add((parts[2], parts[3]))
            continue
        if kind != "audio" or payload is None or source_hash is None:
            continue
        parts = name.split("/")
        utterance_id = parts[4][:-5]
        members.append(
            _SourceMember(
                name=name,
                speaker_id=parts[2],
                chapter_id=parts[3],
                utterance_id=utterance_id,
                frames=_header_frames(payload, name=name),
                size_bytes=size_bytes,
                source_sha256=source_hash,
            )
        )
    if not members:
        raise FullDataError("archive contains no train-clean-100 FLAC members")
    if len({member.utterance_id for member in members}) != len(members):
        raise FullDataError("archive contains duplicate utterance IDs")
    audio_chapters = {(member.speaker_id, member.chapter_id) for member in members}
    if not transcript_chapters.issubset(audio_chapters):
        raise FullDataError("transcript chapter is not contained in train-clean-100 audio")
    if not test_fixture:
        preliminary = _ScannedArchive(
            digest=digest,
            members=tuple(members),
            member_count=member_count,
            total_member_bytes=total_member_bytes,
            metadata_sha256=metadata_hashes,
            member_sha256=member_hashes,
        )
        _validate_real_scan(preliminary)
    return _ScannedArchive(
        digest=digest,
        members=tuple(sorted(members, key=lambda item: item.utterance_id)),
        member_count=member_count,
        total_member_bytes=total_member_bytes,
        metadata_sha256=metadata_hashes,
        member_sha256=member_hashes,
    )


def _rows_for_members(
    members: tuple[_SourceMember, ...],
) -> tuple[list[dict[str, Any]], dict[str, _SourceMember]]:
    rows: list[dict[str, Any]] = []
    indexed: dict[str, _SourceMember] = {}
    row_start = 0
    for source in members:
        window_count = (source.frames + WINDOW_SAMPLES - 1) // WINDOW_SAMPLES
        updated = _SourceMember(
            **{**source.__dict__, "window_count": window_count, "row_start": row_start}
        )
        indexed[source.name] = updated
        for window_index in range(window_count):
            start_sample = window_index * WINDOW_SAMPLES
            valid_samples = min(WINDOW_SAMPLES, source.frames - start_sample)
            padded_samples = WINDOW_SAMPLES - valid_samples
            rows.append(
                {
                    "global_index": len(rows),
                    "utterance_id": source.utterance_id,
                    "window_id": f"{source.utterance_id}@{start_sample:010d}",
                    "speaker_id": source.speaker_id,
                    "chapter_id": source.chapter_id,
                    "source_path": source.name,
                    "start_sample": start_sample,
                    "input_samples": source.frames,
                    "valid_samples": valid_samples,
                    "padded_samples": padded_samples,
                    "source_sha256": source.source_sha256,
                    "split": "train",
                }
            )
        row_start += window_count
    return rows, indexed


def _write_progress(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _decode_source(
    payload: bytes,
    source: _SourceMember,
    rows: list[dict[str, Any]],
    pcm: np.ndarray,
    features: np.ndarray,
) -> None:
    sf = _soundfile_module()
    try:
        handle = sf.SoundFile(io.BytesIO(payload))
    except Exception as exc:
        raise FullDataError(f"cannot open FLAC for decode: {source.name}") from exc
    with handle:
        if (
            handle.samplerate != SAMPLE_RATE
            or handle.channels != 1
            or handle.format != "FLAC"
            or handle.subtype != "PCM_16"
        ):
            raise FullDataError(f"FLAC header changed during decode: {source.name}")
        if int(handle.frames) != source.frames:
            raise FullDataError(f"FLAC frame count changed during decode: {source.name}")
        for local_index in range(source.window_count):
            row_index = source.row_start + local_index
            row = rows[row_index]
            valid_samples = int(row["valid_samples"])
            pcm_window = np.zeros(WINDOW_SAMPLES, dtype=PCM_DTYPE)
            decoded = handle.read(valid_samples, dtype="int16", always_2d=False)
            decoded = np.asarray(decoded, dtype=PCM_DTYPE)
            if decoded.shape != (valid_samples,):
                raise FullDataError(
                    f"decoded frame count mismatch for {source.name}: "
                    f"expected {valid_samples}, got {decoded.shape}"
                )
            pcm_window[:valid_samples] = decoded
            pcm[row_index] = pcm_window
            normalized = pcm_window.astype(np.float32) / np.float32(32768.0)
            try:
                feature = log_mel(normalized)[..., None]
            except (ValueError, FloatingPointError) as exc:
                raise FullDataError(
                    f"frontend failed for {row['window_id']}"
                ) from exc
            if feature.shape != FEATURE_SHAPE or feature.dtype != FEATURE_DTYPE:
                raise FullDataError(f"unexpected frontend shape/dtype for {row['window_id']}")
            features[row_index] = feature
        if int(handle.tell()) != source.frames:
            raise FullDataError(f"decoder did not consume source: {source.name}")


def _validate_row_schema(rows: Any, *, test_fixture: bool) -> tuple[list[dict[str, Any]], int, int]:
    if not isinstance(rows, list) or not rows:
        raise FullDataError("manifest rows must be a non-empty list")
    required = {
        "global_index",
        "utterance_id",
        "window_id",
        "speaker_id",
        "chapter_id",
        "source_path",
        "start_sample",
        "input_samples",
        "valid_samples",
        "padded_samples",
        "source_sha256",
        "split",
    }
    previous_key: tuple[str, int] | None = None
    seen_window_ids: set[str] = set()
    source_state: dict[str, dict[str, Any]] = {}
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        if not isinstance(raw, dict) or set(raw) != required:
            raise FullDataError(f"row {index} fields are not exact")
        row = dict(raw)
        if row["global_index"] != index:
            raise FullDataError(f"row {index} global_index is not contiguous")
        for field in ("utterance_id", "window_id", "speaker_id", "chapter_id", "source_path"):
            if not isinstance(row[field], str) or not row[field]:
                raise FullDataError(f"row {index} {field} must be non-empty text")
        if row["split"] != "train":
            raise FullDataError(f"row {index} is not train-only")
        for field in ("start_sample", "input_samples", "valid_samples", "padded_samples"):
            value = row[field]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise FullDataError(f"row {index} {field} must be a non-negative integer")
        if row["input_samples"] <= 0 or not 1 <= row["valid_samples"] <= WINDOW_SAMPLES:
            raise FullDataError(f"row {index} has invalid source/window lengths")
        if row["start_sample"] % WINDOW_SAMPLES != 0:
            raise FullDataError(f"row {index} start_sample is not a window boundary")
        if row["padded_samples"] != WINDOW_SAMPLES - row["valid_samples"]:
            raise FullDataError(f"row {index} padding does not match valid_samples")
        if row["start_sample"] + row["valid_samples"] > row["input_samples"]:
            raise FullDataError(f"row {index} exceeds source frame count")
        expected_window_id = f"{row['utterance_id']}@{row['start_sample']:010d}"
        if row["window_id"] != expected_window_id or row["window_id"] in seen_window_ids:
            raise FullDataError(f"row {index} window_id is not stable/unique")
        seen_window_ids.add(row["window_id"])
        source_hash = row["source_sha256"]
        if not isinstance(source_hash, str) or len(source_hash) != 64:
            raise FullDataError(f"row {index} source_sha256 is not SHA256")
        try:
            int(source_hash, 16)
        except ValueError as exc:
            raise FullDataError(f"row {index} source_sha256 is not SHA256") from exc
        key = (row["utterance_id"], row["start_sample"])
        if previous_key is not None and key <= previous_key:
            raise FullDataError("rows are not in stable utterance/start order")
        previous_key = key
        previous = source_state.get(row["utterance_id"])
        if previous is None:
            source_state[row["utterance_id"]] = {
                "speaker_id": row["speaker_id"],
                "chapter_id": row["chapter_id"],
                "source_path": row["source_path"],
                "input_samples": row["input_samples"],
                "source_sha256": source_hash,
                "next_start": 0,
                "rows": 0,
            }
            previous = source_state[row["utterance_id"]]
        for field in ("speaker_id", "chapter_id", "source_path", "input_samples", "source_sha256"):
            if row[field] != previous[field]:
                raise FullDataError(f"row {index} source binding changes within utterance")
        source_parts = row["source_path"].split("/")
        expected_source = [
            "LibriSpeech",
            "train-clean-100",
            row["speaker_id"],
            row["chapter_id"],
            f"{row['utterance_id']}.flac",
        ]
        if source_parts != expected_source:
            raise FullDataError(f"row {index} source_path is not immutable source identity")
        if row["start_sample"] != previous["next_start"]:
            raise FullDataError(f"row {index} has a gap/overlap in source coverage")
        previous["next_start"] += row["valid_samples"]
        previous["rows"] += 1
        normalized.append(row)
    for utterance_id, state in source_state.items():
        if state["next_start"] != state["input_samples"]:
            raise FullDataError(
                f"utterance {utterance_id} does not have complete source coverage"
            )
    if not test_fixture and len(source_state) != EXPECTED_AUDIO_COUNT:
        raise FullDataError(
            f"real utterance count mismatch: expected {EXPECTED_AUDIO_COUNT}, "
            f"got {len(source_state)}"
        )
    speaker_count = len({row["speaker_id"] for row in normalized})
    if not test_fixture and speaker_count != EXPECTED_SPEAKER_COUNT:
        raise FullDataError(
            f"real speaker count mismatch: expected {EXPECTED_SPEAKER_COUNT}, got {speaker_count}"
        )
    return normalized, len(source_state), speaker_count


def _manifest_array_record(path: Path, shape: tuple[int, ...], dtype: np.dtype) -> dict[str, Any]:
    return {
        "path": path.name,
        "shape": list(shape),
        "dtype": str(dtype),
        "bytes": _file_bytes(path),
        "sha256": _sha256_file(path),
    }


def _validate_artifact_record(
    root: Path,
    record: Any,
    *,
    expected_shape: tuple[int, ...],
    expected_dtype: np.dtype,
    verify_hashes: bool,
    label: str,
) -> tuple[Path, str]:
    if not isinstance(record, dict):
        raise FullDataError(f"manifest {label} artifact record is invalid")
    required = {"path", "shape", "dtype", "bytes", "sha256"}
    if set(record) != required:
        raise FullDataError(f"manifest {label} artifact fields are not exact")
    relative = record["path"]
    if not isinstance(relative, str) or Path(relative).name != relative or relative.startswith("."):
        raise FullDataError(f"manifest {label} artifact path is unsafe")
    path = root / relative
    if path.is_symlink() or not path.is_file():
        raise FullDataError(f"missing {label} artifact: {path}")
    if tuple(record["shape"]) != expected_shape or record["dtype"] != str(expected_dtype):
        raise FullDataError(f"manifest {label} shape/dtype mismatch")
    if record["bytes"] != _file_bytes(path):
        raise FullDataError(f"{label} artifact byte count mismatch")
    expected_hash = record["sha256"]
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise FullDataError(f"manifest {label} artifact hash is invalid")
    if verify_hashes and _sha256_file(path) != expected_hash:
        raise FullDataError(f"{label} artifact hash mismatch")
    return path, expected_hash


def _validate_real_manifest(manifest: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    if manifest.get("row_count") != EXPECTED_WINDOW_COUNT:
        raise FullDataError("real manifest row_count does not match oracle")
    if manifest.get("utterance_count") != EXPECTED_AUDIO_COUNT:
        raise FullDataError("real manifest utterance_count does not match oracle")
    if manifest.get("speaker_count") != EXPECTED_SPEAKER_COUNT:
        raise FullDataError("real manifest speaker_count does not match oracle")
    if len(rows) != EXPECTED_WINDOW_COUNT:
        raise FullDataError("real manifest rows do not match oracle")
    source_frames = sum(
        int(row["valid_samples"]) for row in rows
    )
    if source_frames != EXPECTED_SOURCE_FRAMES:
        raise FullDataError("real manifest source coverage does not match oracle")
    padding = sum(int(row["padded_samples"]) for row in rows)
    if padding != EXPECTED_PADDING_SAMPLES:
        raise FullDataError("real manifest padding does not match oracle")
    source = manifest["source"]
    expected_source = {
        "member_count": EXPECTED_MEMBER_COUNT,
        "audio_count": EXPECTED_AUDIO_COUNT,
        "speaker_count": EXPECTED_SPEAKER_COUNT,
        "source_frames": EXPECTED_SOURCE_FRAMES,
        "window_count": EXPECTED_WINDOW_COUNT,
        "padding_samples": EXPECTED_PADDING_SAMPLES,
    }
    for field, expected in expected_source.items():
        if source.get(field) != expected:
            raise FullDataError(f"real source {field} does not match oracle")


def load_full_dataset(
    root: str | Path,
    *,
    verify_hashes: bool = True,
    test_fixture: bool = False,
) -> FullDataset:
    """Load and validate a prepared full dataset using read-only mmaps."""

    dataset_candidate = Path(root).expanduser()
    if dataset_candidate.is_symlink():
        raise FullDataError(f"dataset root may not be a symlink: {dataset_candidate}")
    dataset_root = dataset_candidate.resolve()
    if not dataset_root.is_dir():
        raise FullDataError(f"dataset root is not a regular directory: {dataset_root}")
    manifest_path = dataset_root / "manifest.json"
    provenance_path = dataset_root / "provenance.json"
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes.decode("utf-8"))
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FullDataError(f"cannot read full dataset metadata: {dataset_root}") from exc
    if not isinstance(manifest, dict) or not isinstance(provenance, dict):
        raise FullDataError("manifest and provenance must be JSON objects")
    required_manifest = {
        "format_version",
        "dataset",
        "split",
        "sample_rate",
        "window_samples",
        "feature_shape",
        "row_count",
        "utterance_count",
        "speaker_count",
        "rows_sha256",
        "rows",
        "arrays",
        "source",
    }
    if set(manifest) != required_manifest:
        raise FullDataError("full manifest fields are not exact")
    if manifest["format_version"] != FORMAT_VERSION or manifest["split"] != "train":
        raise FullDataError("unsupported full manifest format/split")
    if manifest["sample_rate"] != SAMPLE_RATE or manifest["window_samples"] != WINDOW_SAMPLES:
        raise FullDataError("full dataset audio contract mismatch")
    if tuple(manifest["feature_shape"]) != FEATURE_SHAPE:
        raise FullDataError("full dataset feature shape mismatch")
    fixture_status = provenance.get("status") == "SYNTHETIC_TEST_FIXTURE"
    expected_status = (
        "SYNTHETIC_TEST_FIXTURE" if fixture_status else "PREPARED_OFFICIAL_SOURCE"
    )
    if provenance.get("status") != expected_status:
        raise FullDataError("full dataset provenance status is not recognized")
    if fixture_status != bool(test_fixture):
        raise FullDataError(
            "synthetic fixtures require test_fixture=True and real datasets require it False"
        )
    if manifest["dataset"] != (SYNTHETIC_DATASET if fixture_status else DATASET):
        raise FullDataError("full dataset identity/provenance mismatch")
    manifest_hash = _sha256_bytes(manifest_bytes)
    if provenance.get("manifest_sha256") != manifest_hash:
        raise FullDataError("provenance manifest hash mismatch")
    rows, utterance_count, speaker_count = _validate_row_schema(
        manifest["rows"], test_fixture=fixture_status
    )
    if manifest["row_count"] != len(rows) or manifest["utterance_count"] != utterance_count:
        raise FullDataError("manifest row/utterance count mismatch")
    if manifest["speaker_count"] != speaker_count:
        raise FullDataError("manifest speaker count mismatch")
    if _sha256_bytes(_canonical_bytes(rows)) != manifest["rows_sha256"]:
        raise FullDataError("manifest rows hash mismatch")
    if not fixture_status:
        _validate_real_manifest(manifest, rows)
    source = manifest["source"]
    if not isinstance(source, dict) or source.get("archive_sha256") != provenance.get("source", {}).get("archive_sha256"):
        raise FullDataError("source archive binding is missing or inconsistent")
    if not fixture_status:
        if source.get("expected_md5") != OFFICIAL_MD5 or source.get("md5_status") != "MATCHED_OFFICIAL_METADATA":
            raise FullDataError("real dataset lacks the official archive MD5 binding")
    arrays = manifest["arrays"]
    if not isinstance(arrays, dict) or set(arrays) != {"pcm", "features"}:
        raise FullDataError("full manifest array bindings are invalid")
    pcm_path, pcm_hash = _validate_artifact_record(
        dataset_root,
        arrays["pcm"],
        expected_shape=(len(rows), WINDOW_SAMPLES),
        expected_dtype=PCM_DTYPE,
        verify_hashes=verify_hashes,
        label="PCM",
    )
    feature_path, feature_hash = _validate_artifact_record(
        dataset_root,
        arrays["features"],
        expected_shape=(len(rows), *FEATURE_SHAPE),
        expected_dtype=FEATURE_DTYPE,
        verify_hashes=verify_hashes,
        label="feature",
    )
    if provenance.get("arrays") != arrays:
        raise FullDataError("provenance array bindings do not match manifest")
    try:
        pcm = np.load(pcm_path, mmap_mode="r", allow_pickle=False)
        features = np.load(feature_path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise FullDataError("cannot memory-map full dataset arrays") from exc
    if not isinstance(pcm, np.memmap) or not isinstance(features, np.memmap):
        raise FullDataError("full dataset arrays must be NumPy memory maps")
    if pcm.dtype != PCM_DTYPE or pcm.shape != (len(rows), WINDOW_SAMPLES):
        raise FullDataError("PCM mmap shape/dtype mismatch")
    if features.dtype != FEATURE_DTYPE or features.shape != (len(rows), *FEATURE_SHAPE):
        raise FullDataError("feature mmap shape/dtype mismatch")
    try:
        pcm.flags.writeable = False
        features.flags.writeable = False
    except ValueError as exc:
        raise FullDataError("full dataset mmap is not read-only") from exc
    for start in range(0, len(rows), FEATURE_CHECK_CHUNK):
        chunk = np.asarray(features[start : start + FEATURE_CHECK_CHUNK])
        if not np.all(np.isfinite(chunk)):
            raise FullDataError(f"feature mmap contains non-finite values near row {start}")
    if provenance.get("rows_sha256") != manifest["rows_sha256"]:
        raise FullDataError("provenance rows binding mismatch")
    if provenance.get("pcm_sha256") != pcm_hash or provenance.get("features_sha256") != feature_hash:
        raise FullDataError("provenance array hash binding mismatch")
    return FullDataset(
        root=dataset_root,
        manifest=manifest,
        rows=rows,
        pcm=pcm,
        features=features,
        manifest_sha256=manifest_hash,
    )


def prepare_full_dataset(
    archive_path: str | Path,
    output_dir: str | Path,
    *,
    test_fixture: bool = False,
) -> dict[str, Any]:
    """Prepare all train-clean-100 windows into one atomically published bundle."""

    archive = Path(archive_path).expanduser()
    output_candidate = Path(output_dir).expanduser()
    if output_candidate.is_symlink() or output_candidate.exists():
        raise FullDataError(f"refusing to overwrite existing output: {output_candidate}")
    output = output_candidate.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=str(output.parent)))
    started = time.perf_counter()
    try:
        scan = _scan_archive(archive, test_fixture=test_fixture)
        rows, indexed_sources = _rows_for_members(scan.members)
        if not rows:
            raise FullDataError("archive produced no window rows")
        if not test_fixture and len(rows) != EXPECTED_WINDOW_COUNT:
            raise FullDataError("real preparation rows do not match oracle")
        pcm_path = stage / "pcm_int16.npy"
        features_path = stage / "features.npy"
        pcm = np.lib.format.open_memmap(
            pcm_path, mode="w+", dtype=PCM_DTYPE, shape=(len(rows), WINDOW_SAMPLES)
        )
        features = np.lib.format.open_memmap(
            features_path,
            mode="w+",
            dtype=FEATURE_DTYPE,
            shape=(len(rows), *FEATURE_SHAPE),
        )
        progress_path = stage / "progress.json"
        progress: dict[str, Any] = {
            "status": "PREPARING",
            "rows_written": 0,
            "total_rows": len(rows),
            "utterances_decoded": 0,
            "total_utterances": len(scan.members),
            "elapsed_seconds": 0.0,
        }
        _write_progress(progress_path, progress)
        seen_sources: set[str] = set()
        next_progress = PROGRESS_INTERVAL
        second_member_count = 0
        second_member_bytes = 0
        for name, kind, payload, source_hash, size_bytes in _iter_secure_members(archive):
            second_member_count += 1
            second_member_bytes += size_bytes
            if kind != "directory":
                if source_hash is None or scan.member_sha256.get(name) != source_hash:
                    raise FullDataError(
                        f"archive member changed between archive passes: {name}"
                    )
            if kind != "audio":
                continue
            if payload is None or source_hash is None or name not in indexed_sources:
                raise FullDataError(f"unexpected audio member during decode: {name}")
            source = indexed_sources[name]
            if source_hash != source.source_sha256:
                raise FullDataError(f"source member changed between archive passes: {name}")
            _decode_source(payload, source, rows, pcm, features)
            seen_sources.add(name)
            progress["utterances_decoded"] = len(seen_sources)
            progress["rows_written"] += source.window_count
            progress["elapsed_seconds"] = time.perf_counter() - started
            if progress["rows_written"] >= next_progress or len(seen_sources) == len(indexed_sources):
                _write_progress(progress_path, progress)
                print(json.dumps(progress, sort_keys=True), flush=True)
                while next_progress <= progress["rows_written"]:
                    next_progress += PROGRESS_INTERVAL
        if seen_sources != set(indexed_sources):
            raise FullDataError("archive decode pass did not cover every audio member")
        if second_member_count != scan.member_count or second_member_bytes != scan.total_member_bytes:
            raise FullDataError("archive member inventory changed between passes")
        pcm.flush()
        features.flush()
        del pcm
        del features
        final_digest = _archive_digest(
            archive, expected_md5=None if test_fixture else OFFICIAL_MD5
        )
        if final_digest != scan.digest:
            raise FullDataError("archive changed during full preparation")
        pcm_record = _manifest_array_record(
            pcm_path, (len(rows), WINDOW_SAMPLES), PCM_DTYPE
        )
        feature_record = _manifest_array_record(
            features_path, (len(rows), *FEATURE_SHAPE), FEATURE_DTYPE
        )
        rows_hash = _sha256_bytes(_canonical_bytes(rows))
        source_record: dict[str, Any] = {
            "archive_url": ARCHIVE_URL if not test_fixture else None,
            "archive_sha256": scan.digest.sha256,
            "archive_md5": scan.digest.md5,
            "expected_md5": scan.digest.expected_md5,
            "md5_status": (
                "MATCHED_OFFICIAL_METADATA"
                if not test_fixture
                else "SYNTHETIC_TEST_FIXTURE_NOT_OFFICIAL"
            ),
            "member_count": scan.member_count,
            "audio_count": len(scan.members),
            "speaker_count": len({member.speaker_id for member in scan.members}),
            "source_frames": sum(member.frames for member in scan.members),
            "window_count": len(rows),
            "padding_samples": sum(int(row["padded_samples"]) for row in rows),
            "metadata_sha256": dict(scan.metadata_sha256),
        }
        manifest: dict[str, Any] = {
            "format_version": FORMAT_VERSION,
            "dataset": SYNTHETIC_DATASET if test_fixture else DATASET,
            "split": "train",
            "sample_rate": SAMPLE_RATE,
            "window_samples": WINDOW_SAMPLES,
            "feature_shape": list(FEATURE_SHAPE),
            "row_count": len(rows),
            "utterance_count": len(scan.members),
            "speaker_count": len({member.speaker_id for member in scan.members}),
            "rows_sha256": rows_hash,
            "rows": rows,
            "arrays": {"pcm": pcm_record, "features": feature_record},
            "source": source_record,
        }
        manifest_path = stage / "manifest.json"
        manifest_bytes = _canonical_bytes(manifest) + b"\n"
        manifest_path.write_bytes(manifest_bytes)
        manifest_hash = _sha256_bytes(manifest_bytes)
        provenance: dict[str, Any] = {
            "status": "SYNTHETIC_TEST_FIXTURE" if test_fixture else "PREPARED_OFFICIAL_SOURCE",
            "manifest_sha256": manifest_hash,
            "rows_sha256": rows_hash,
            "pcm_sha256": pcm_record["sha256"],
            "features_sha256": feature_record["sha256"],
            "arrays": {"pcm": pcm_record, "features": feature_record},
            "source": {"archive_sha256": scan.digest.sha256},
            "audio_policy": {
                "window_samples": WINDOW_SAMPLES,
                "windowing": "consecutive_nonoverlapping",
                "final_window": "right_zero_pad",
            },
        }
        (stage / "provenance.json").write_bytes(_canonical_bytes(provenance) + b"\n")
        progress.update(
            {
                "status": "READY",
                "rows_written": len(rows),
                "utterances_decoded": len(scan.members),
                "elapsed_seconds": time.perf_counter() - started,
                "manifest_sha256": manifest_hash,
            }
        )
        _write_progress(progress_path, progress)
        if output.exists() or output.is_symlink():
            raise FullDataError(f"refusing to overwrite existing output: {output}")
        os.replace(stage, output)
        stage = output
        # Publication validation is part of the write contract and keeps the
        # returned record tied to the same read-only bundle that was published.
        loaded = load_full_dataset(output, verify_hashes=True, test_fixture=test_fixture)
        return {
            "status": "SYNTHETIC_TEST_FIXTURE" if test_fixture else "FULL_DATASET_PREPARED",
            "root": str(output),
            "manifest": str(output / "manifest.json"),
            "manifest_sha256": loaded.manifest_sha256,
            "rows": len(loaded.rows),
            "speakers": len({row["speaker_id"] for row in loaded.rows}),
            "utterances": len({row["utterance_id"] for row in loaded.rows}),
            "pcm": dict(pcm_record),
            "features": dict(feature_record),
            "source": source_record,
        }
    except Exception:
        if stage != output:
            shutil.rmtree(stage, ignore_errors=True)
        raise


__all__ = [
    "DATASET",
    "FEATURE_DTYPE",
    "FEATURE_SHAPE",
    "FORMAT_VERSION",
    "FullDataError",
    "FullDataset",
    "OFFICIAL_MD5",
    "SAMPLE_RATE",
    "SYNTHETIC_DATASET",
    "WINDOW_SAMPLES",
    "load_full_dataset",
    "prepare_full_dataset",
]


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare the bounded full train-clean-100 window dataset"
    )
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--test-fixture",
        action="store_true",
        help="accept only a clearly synthetic fixture archive; never official provenance",
    )
    args = parser.parse_args(argv)
    result = prepare_full_dataset(
        args.archive,
        args.output,
        test_fixture=args.test_fixture,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a CLI process.
    raise SystemExit(_main())
