"""Deterministic LibriSpeech dev-clean pilot preparation.

The command in this module consumes an already-downloaded official OpenSLR
archive.  It never downloads data and refuses an archive whose published MD5
does not match.  Extraction is performed only after every tar member has been
checked for traversal, links, special files, and unexpected layout.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
from typing import Iterable

import numpy as np

from .frontend import NUM_SAMPLES, SAMPLE_RATE, log_mel

ARCHIVE_URL = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
ARCHIVE_MD5 = "42e2234ba48799c1f50f24a7926300a1"
DATASET_NAME = "librispeech-dev-clean-pilot"
LICENSE = "CC BY 4.0"
SEED = 20261012
EXPECTED_SPEAKERS = 40
TRAIN_SPEAKERS = 20
VALIDATION_SPEAKERS = 5
CALIBRATION_SPEAKERS = 5
TEST_SPEAKERS = 10
TRAIN_UTTERANCES = 20
EVAL_UTTERANCES = 12
ENROLL_UTTERANCES = 3
QUERY_UTTERANCES = 9
_ARCHIVE_ROOT = ("LibriSpeech", "dev-clean")
_SOURCE_SUFFIXES = {".flac", ".wav"}
_TOP_LEVEL_METADATA = {
    "LICENSE.TXT",
    "README.TXT",
    "CHAPTERS.TXT",
    "SPEAKERS.TXT",
    "BOOKS.TXT",
}


@dataclass(frozen=True)
class SourceUtterance:
    """A source audio file in the verified extracted corpus."""

    speaker_id: str
    chapter_id: str
    utterance_id: str
    path: Path


def _hash_file(path: Path) -> tuple[str, str, int]:
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                size += len(chunk)
                md5.update(chunk)
                sha256.update(chunk)
    except OSError as exc:
        raise ValueError(f"could not read archive/source file {path}: {exc}") from exc
    return md5.hexdigest(), sha256.hexdigest(), size


def _validate_member(member: tarfile.TarInfo) -> tuple[PurePosixPath, bool]:
    name = member.name
    if not name or "\x00" in name or "\\" in name:
        raise ValueError(f"unsafe tar member name {name!r}")
    relative = PurePosixPath(name)
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise ValueError(f"unsafe tar member path {name!r}")
    if member.issym() or member.islnk():
        raise ValueError(f"tar links are not allowed: {name!r}")
    if (
        len(relative.parts) == 2
        and relative.parts[0] == "LibriSpeech"
        and relative.parts[1] != "dev-clean"
    ):
        if relative.name not in _TOP_LEVEL_METADATA or not member.isfile():
            raise ValueError(f"unexpected top-level tar member {name!r}")
        return relative, False
    if relative.parts[:2] != _ARCHIVE_ROOT:
        raise ValueError(f"unexpected tar member root {name!r}")
    if member.isdir():
        if len(relative.parts) > 4:
            raise ValueError(f"unexpected tar directory layout {name!r}")
        return relative, True
    if not member.isfile():
        raise ValueError(f"tar special files are not allowed: {name!r}")
    if len(relative.parts) != 5:
        raise ValueError(f"unexpected tar file layout {name!r}")
    if not (
        Path(relative.name).suffix.lower() in _SOURCE_SUFFIXES
        or relative.name.endswith(".trans.txt")
    ):
        raise ValueError(f"unexpected source file suffix {name!r}")
    if member.size < 0:
        raise ValueError(f"negative tar member size {name!r}")
    return relative, False


def _validated_members(archive: tarfile.TarFile) -> list[tuple[tarfile.TarInfo, PurePosixPath, bool]]:
    members = archive.getmembers()
    if not members:
        raise ValueError("archive contains no members")
    validated: list[tuple[tarfile.TarInfo, PurePosixPath, bool]] = []
    names: set[PurePosixPath] = set()
    files: set[PurePosixPath] = set()
    for member in members:
        relative, is_directory = _validate_member(member)
        if relative in names:
            raise ValueError(f"duplicate tar member {member.name!r}")
        names.add(relative)
        if not is_directory:
            files.add(relative)
        validated.append((member, relative, is_directory))
    for file_path in files:
        for prefix_length in range(1, len(file_path.parts)):
            if PurePosixPath(*file_path.parts[:prefix_length]) in files:
                raise ValueError(f"tar file shadows a directory {file_path!s}")
    return validated


def _extract_verified_archive(archive_path: Path, destination: Path) -> int:
    """Validate all members, then extract only regular files/directories."""

    if destination.exists():
        if not destination.is_dir() or any(destination.iterdir()):
            raise ValueError(
                "extraction destination must be a fresh empty directory: "
                f"{destination}"
            )
    else:
        destination.mkdir(parents=True, exist_ok=False)
    try:
        with tarfile.open(archive_path, mode="r:*") as archive:
            members = _validated_members(archive)
            for member, relative, is_directory in members:
                target = destination.joinpath(*relative.parts)
                if is_directory:
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(f"could not read tar member {member.name!r}")
                with source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
                if target.stat().st_size != member.size:
                    raise ValueError(f"short extraction for tar member {member.name!r}")
                target.chmod(0o644)
    except (OSError, tarfile.TarError) as exc:
        raise ValueError(f"could not safely extract {archive_path}: {exc}") from exc
    return sum(not is_directory for _, _, is_directory in members)


def _discover_sources(extracted_root: Path) -> list[SourceUtterance]:
    corpus_root = extracted_root.joinpath(*_ARCHIVE_ROOT)
    if not corpus_root.is_dir():
        raise ValueError("extracted archive lacks LibriSpeech/dev-clean")
    sources: list[SourceUtterance] = []
    for path in sorted(corpus_root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"extracted source contains a symlink: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(corpus_root)
        if len(relative.parts) != 3:
            raise ValueError(f"unexpected extracted source path {relative!s}")
        speaker_id, chapter_id, filename = relative.parts
        if filename.endswith(".trans.txt"):
            continue
        if Path(filename).suffix.lower() not in _SOURCE_SUFFIXES:
            raise ValueError(f"unexpected extracted file {relative!s}")
        utterance_id = Path(filename).stem
        identifier_parts = utterance_id.split("-", 2)
        if len(identifier_parts) != 3 or identifier_parts[:2] != [speaker_id, chapter_id]:
            raise ValueError(f"source ID does not match path {relative!s}")
        sources.append(
            SourceUtterance(
                speaker_id=speaker_id,
                chapter_id=chapter_id,
                utterance_id=utterance_id,
                path=path,
            )
        )
    if not sources:
        raise ValueError("extracted corpus contains no audio files")
    if len({source.utterance_id for source in sources}) != len(sources):
        raise ValueError("source utterance IDs are not unique")
    return sources


def _speaker_plan(speakers: Iterable[str], seed: int) -> dict[str, list[str]]:
    speaker_ids = sorted(set(speakers))
    if len(speaker_ids) != EXPECTED_SPEAKERS:
        raise ValueError(
            f"expected exactly {EXPECTED_SPEAKERS} speakers; got {len(speaker_ids)}"
        )
    permutation = np.random.default_rng(seed).permutation(np.asarray(speaker_ids))
    ordered = [str(value) for value in permutation.tolist()]
    boundaries = {
        "train": TRAIN_SPEAKERS,
        "validation": VALIDATION_SPEAKERS,
        "calibration": CALIBRATION_SPEAKERS,
        "test": TEST_SPEAKERS,
    }
    plan: dict[str, list[str]] = {}
    offset = 0
    for name, count in boundaries.items():
        plan[name] = ordered[offset : offset + count]
        offset += count
    return plan


def _split_enrollment_query(
    candidates: list[SourceUtterance],
) -> tuple[list[SourceUtterance], list[SourceUtterance], dict[str, object]]:
    """Prefer distinct chapters, recording a deterministic fallback."""

    by_chapter: dict[str, list[SourceUtterance]] = {}
    for source in candidates:
        by_chapter.setdefault(source.chapter_id, []).append(source)
    for chapter_id in sorted(by_chapter):
        enrollment = by_chapter[chapter_id][:ENROLL_UTTERANCES]
        query_candidates = [
            source for source in candidates if source.chapter_id != chapter_id
        ]
        if len(enrollment) == ENROLL_UTTERANCES and len(query_candidates) >= QUERY_UTTERANCES:
            query = query_candidates[:QUERY_UTTERANCES]
            return enrollment, query, {
                "separated": True,
                "enrollment_chapters": sorted({item.chapter_id for item in enrollment}),
                "query_chapters": sorted({item.chapter_id for item in query}),
                "reason": "distinct chapter allocation",
            }

    selected = candidates[:EVAL_UTTERANCES]
    enrollment = selected[:ENROLL_UTTERANCES]
    query = selected[ENROLL_UTTERANCES:]
    return enrollment, query, {
        "separated": False,
        "enrollment_chapters": sorted({item.chapter_id for item in enrollment}),
        "query_chapters": sorted({item.chapter_id for item in query}),
        "reason": "insufficient distinct chapter coverage",
    }


def _select_sources(
    sources: list[SourceUtterance], seed: int
) -> tuple[list[tuple[SourceUtterance, str]], dict[str, object]]:
    by_speaker: dict[str, list[SourceUtterance]] = {}
    for source in sources:
        by_speaker.setdefault(source.speaker_id, []).append(source)
    for speaker_sources in by_speaker.values():
        speaker_sources.sort(key=lambda item: item.utterance_id)

    plan = _speaker_plan(by_speaker, seed)
    selected: list[tuple[SourceUtterance, str]] = []
    chapter_separation: dict[str, dict[str, object]] = {
        "calibration": {},
        "test": {},
    }
    split_speakers: dict[str, list[str]] = {}

    for split_name in ("train", "validation", "calibration", "test"):
        split_speakers[split_name] = list(plan[split_name])
        required = TRAIN_UTTERANCES if split_name == "train" else EVAL_UTTERANCES
        for speaker_id in plan[split_name]:
            candidates = by_speaker[speaker_id]
            if len(candidates) < required:
                raise ValueError(
                    f"speaker {speaker_id} has {len(candidates)} utterances; "
                    f"{split_name} requires {required}"
                )
            if split_name in {"calibration", "test"}:
                chosen = candidates
                enrollment, query, separation = _split_enrollment_query(chosen)
                if len(enrollment) != ENROLL_UTTERANCES or len(query) != QUERY_UTTERANCES:
                    raise ValueError(f"could not form 3/9 trials for speaker {speaker_id}")
                selected.extend((item, f"{split_name}_enroll") for item in enrollment)
                selected.extend((item, f"{split_name}_query") for item in query)
                chapter_separation[split_name][speaker_id] = separation
            else:
                selected.extend(
                    (item, split_name) for item in candidates[:required]
                )

    expected_rows = (
        TRAIN_SPEAKERS * TRAIN_UTTERANCES
        + VALIDATION_SPEAKERS * EVAL_UTTERANCES
        + CALIBRATION_SPEAKERS * EVAL_UTTERANCES
        + TEST_SPEAKERS * EVAL_UTTERANCES
    )
    if len(selected) != expected_rows:
        raise AssertionError(f"selection produced {len(selected)} rows, expected {expected_rows}")
    return selected, {
        "speaker_plan": split_speakers,
        "chapter_separation": chapter_separation,
        "selection_policy": {
            "speaker_order": "sorted IDs then NumPy default_rng permutation",
            "seed": seed,
            "train_max_utterances": TRAIN_UTTERANCES,
            "validation_utterances": EVAL_UTTERANCES,
            "calibration_utterances": EVAL_UTTERANCES,
            "test_utterances": EVAL_UTTERANCES,
            "enrollment_utterances": ENROLL_UTTERANCES,
            "query_utterances": QUERY_UTTERANCES,
        },
    }


def _read_segment(source: SourceUtterance) -> tuple[np.ndarray, int, int, int]:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError(
            "pilot preparation requires soundfile==0.13.1; install requirements-pilot.txt"
        ) from exc
    try:
        samples, sample_rate = sf.read(
            source.path, dtype="int16", always_2d=False
        )
    except Exception as exc:  # soundfile exposes several backend exception types.
        raise ValueError(f"could not decode {source.path}: {exc}") from exc
    pcm = np.asarray(samples)
    if sample_rate != SAMPLE_RATE:
        raise ValueError(
            f"{source.path} must be {SAMPLE_RATE} Hz; got {sample_rate}"
        )
    if pcm.ndim != 1 or pcm.size == 0:
        raise ValueError(f"{source.path} must be non-empty mono audio")
    if pcm.dtype.kind not in "iu":
        raise ValueError(f"{source.path} did not decode to integer PCM")
    pcm = np.asarray(pcm, dtype="<i2")
    input_samples = int(pcm.size)
    if input_samples < NUM_SAMPLES:
        padded_samples = NUM_SAMPLES - input_samples
        truncated_samples = 0
        prepared = np.pad(pcm, (0, padded_samples), mode="constant")
    else:
        padded_samples = 0
        truncated_samples = input_samples - NUM_SAMPLES
        prepared = pcm[:NUM_SAMPLES]
    return np.asarray(prepared, dtype="<i2"), input_samples, padded_samples, truncated_samples


def _write_wav(path: Path, pcm: np.ndarray) -> None:
    try:
        import soundfile as sf
        sf.write(path, pcm, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    except ImportError as exc:
        raise RuntimeError(
            "pilot preparation requires soundfile==0.13.1; install requirements-pilot.txt"
        ) from exc
    except Exception as exc:
        raise ValueError(f"could not write prepared WAV {path}: {exc}") from exc


def _manifest_text(manifest: dict[str, object]) -> bytes:
    return (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=False) + "\n"
    ).encode("utf-8")


def _write_prepared(
    prepared_root: Path,
    selected: list[tuple[SourceUtterance, str]],
    selection_metadata: dict[str, object],
    archive_path: Path,
    archive_md5: str,
    archive_sha256: str,
    archive_size: int,
    seed: int,
) -> dict[str, object]:
    wav_root = prepared_root / "wav"
    wav_root.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, object]] = []
    feature_values: list[np.ndarray] = []
    seen_source_hashes: set[str] = set()
    seen_wav_hashes: set[str] = set()

    for source, split in selected:
        source_md5, source_sha256, _ = _hash_file(source.path)
        del source_md5
        if source_sha256 in seen_source_hashes:
            raise ValueError("selected rows reuse a source file hash")
        seen_source_hashes.add(source_sha256)
        pcm, input_samples, padded_samples, truncated_samples = _read_segment(source)
        wav_path = wav_root / f"{source.utterance_id}.wav"
        _write_wav(wav_path, pcm)
        _, wav_sha256, _ = _hash_file(wav_path)
        if wav_sha256 in seen_wav_hashes:
            raise ValueError("selected rows reuse a prepared PCM hash")
        seen_wav_hashes.add(wav_sha256)
        feature_values.append(
            log_mel(pcm.astype(np.float32) / np.float32(32768.0))
        )
        rows.append(
            {
                "utterance_id": source.utterance_id,
                "speaker_id": source.speaker_id,
                "chapter_id": source.chapter_id,
                "split": split,
                "wav_path": f"wav/{source.utterance_id}.wav",
                "wav_sha256": wav_sha256,
                "source_sha256": source_sha256,
                "input_samples": input_samples,
                "padded_samples": padded_samples,
                "truncated_samples": truncated_samples,
            }
        )

    manifest: dict[str, object] = {
        "format_version": 1,
        "dataset": DATASET_NAME,
        "seed": seed,
        "rows": rows,
    }
    manifest_path = prepared_root / "manifest.json"
    manifest_path.write_bytes(_manifest_text(manifest))
    _, manifest_sha256, _ = _hash_file(manifest_path)

    features = np.stack(feature_values, axis=0).astype(np.float32)[..., None]
    features_path = prepared_root / "features.npz"
    np.savez_compressed(
        features_path,
        ids=np.asarray([row["utterance_id"] for row in rows], dtype=str),
        features=features,
        manifest_sha256=np.asarray(manifest_sha256),
    )
    _, features_sha256, _ = _hash_file(features_path)

    try:
        import soundfile as sf
        soundfile_version = str(sf.__version__)
    except ImportError as exc:
        raise RuntimeError(
            "pilot preparation requires soundfile==0.13.1; install requirements-pilot.txt"
        ) from exc
    provenance = {
        "format_version": 1,
        "dataset": DATASET_NAME,
        "source": {
            "provider": "OpenSLR SLR12",
            "url": ARCHIVE_URL,
            "archive_path": str(archive_path.resolve()),
            "archive_bytes": archive_size,
            "archive_md5": archive_md5,
            "archive_sha256": archive_sha256,
            "sha256_status": "observed_local_hash_not_signature",
            "license": LICENSE,
        },
        "audio_policy": {
            "reader": "soundfile",
            "soundfile_version": soundfile_version,
            "sample_rate": SAMPLE_RATE,
            "channels": 1,
            "input_encoding": "decoded integer PCM16",
            "prepared_encoding": "WAV PCM16 little-endian",
            "segment": "first 3 seconds, right zero padding, tail truncation",
            "teacher_student_pcm": "same prepared PCM16 samples",
        },
        "selection": selection_metadata,
        "artifacts": {
            "manifest_sha256": manifest_sha256,
            "features_sha256": features_sha256,
            "feature_shape": list(features.shape),
            "row_count": len(rows),
        },
    }
    provenance_path = prepared_root / "provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    split_counts: dict[str, int] = {}
    for row in rows:
        split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1
    return {
        "status": "NEW_LOCAL_LIBRISPEECH_PILOT_PREPARED",
        "dataset": DATASET_NAME,
        "manifest_path": str(manifest_path.resolve()),
        "features_path": str(features_path.resolve()),
        "provenance_path": str(provenance_path.resolve()),
        "row_count": len(rows),
        "feature_shape": list(features.shape),
        "split_counts": split_counts,
        "manifest_sha256": manifest_sha256,
        "features_sha256": features_sha256,
        "archive_md5": archive_md5,
        "archive_sha256": archive_sha256,
        "archive_sha256_status": "observed_local_hash_not_signature",
    }


def prepare_dataset(
    archive_path: str | Path,
    output_path: str | Path,
    *,
    seed: int = SEED,
) -> dict[str, object]:
    """Safely prepare the official dev-clean pilot into a fresh directory."""

    archive = Path(archive_path)
    output = Path(output_path)
    if not archive.is_file():
        raise ValueError(f"archive is not a regular file: {archive}")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    actual_md5, actual_sha256, archive_size = _hash_file(archive)
    if actual_md5.lower() != ARCHIVE_MD5:
        raise ValueError(
            f"archive MD5 mismatch: expected {ARCHIVE_MD5}, got {actual_md5}"
        )

    parent = output.parent
    parent.mkdir(parents=True, exist_ok=True)
    extracted_final = parent / "extracted"
    if extracted_final.exists() or output.exists():
        raise ValueError(
            f"refusing to overwrite existing extraction/prepared output: "
            f"{extracted_final} or {output}"
        )

    with tempfile.TemporaryDirectory(prefix=".pilot-extracted-", dir=parent) as extracted_name:
        with tempfile.TemporaryDirectory(prefix=".pilot-prepared-", dir=parent) as prepared_name:
            extracted_tmp = Path(extracted_name)
            prepared_tmp = Path(prepared_name)
            _extract_verified_archive(archive, extracted_tmp)
            sources = _discover_sources(extracted_tmp)
            selected, selection_metadata = _select_sources(sources, seed)
            result = _write_prepared(
                prepared_tmp,
                selected,
                selection_metadata,
                archive,
                actual_md5,
                actual_sha256,
                archive_size,
                seed,
            )
            extracted_tmp.rename(extracted_final)
            prepared_tmp.rename(output)

    for key, filename in (
        ("manifest_path", "manifest.json"),
        ("features_path", "features.npz"),
        ("provenance_path", "provenance.json"),
    ):
        result[key] = str((output / filename).resolve())
    result["extracted_path"] = str(extracted_final.resolve())
    result["output_path"] = str(output.resolve())
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        result = prepare_dataset(args.archive, args.output, seed=args.seed)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by CLI smoke tests.
    raise SystemExit(main())


__all__ = ["ARCHIVE_MD5", "ARCHIVE_URL", "SEED", "prepare_dataset"]
