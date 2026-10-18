from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from edgeai import expanded_data


def _write_wav(path: Path, samples: np.ndarray) -> str:
    import wave

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(np.asarray(samples, dtype="<i2").tobytes())
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _old_base(tmp_path: Path) -> tuple[Path, Path, tuple[str, ...]]:
    root = tmp_path / "old-prepared"
    root.mkdir()
    rows: list[dict[str, object]] = []
    ids: list[str] = []
    for index, (speaker, split) in enumerate((("9001", "train"), ("9002", "validation"))):
        utterance_id = f"{speaker}-001-{index:04d}"
        relative = Path("wav") / f"{utterance_id}.wav"
        wav_hash = _write_wav(root / relative, np.full(48_000, index, dtype=np.int16))
        ids.append(utterance_id)
        rows.append(
            {
                "utterance_id": utterance_id,
                "speaker_id": speaker,
                "chapter_id": "001",
                "split": split,
                "wav_path": relative.as_posix(),
                "wav_sha256": wav_hash,
                "source_sha256": hashlib.sha256(f"source-{index}".encode()).hexdigest(),
                "input_samples": 48_000,
                "padded_samples": 0,
                "truncated_samples": 0,
            }
        )
    manifest = {
        "format_version": 1,
        "dataset": expanded_data.OLD_DATASET,
        "seed": expanded_data.SEED,
        "rows": rows,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    manifest_hash = expanded_data.sha256_file(root / "manifest.json")
    features = np.arange(2 * 298 * 40, dtype=np.float32).reshape(2, 298, 40, 1)
    np.savez_compressed(
        root / "features.npz",
        ids=np.asarray(ids, dtype="U"),
        features=features,
        manifest_sha256=np.asarray(manifest_hash),
    )
    (root / "provenance.json").write_text(
        json.dumps(
            {
                "source": {"provider": "synthetic"},
                "selection": {},
                "audio": {},
                "artifacts": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    teacher = tmp_path / "old-teacher.npz"
    teacher.write_bytes(b"synthetic old teacher placeholder")
    return root, teacher, tuple(ids)


def _add_member(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    member = tarfile.TarInfo(name)
    member.size = len(payload)
    archive.addfile(member, io.BytesIO(payload))


def _flac_bytes(seed: int) -> bytes:
    samples = (np.arange(16_000, dtype=np.int32) + seed * 17) % 32_000
    samples = (samples - 16_000).astype(np.int16)
    buffer = io.BytesIO()
    sf.write(buffer, samples, 16_000, format="FLAC", subtype="PCM_16")
    return buffer.getvalue()


def _train_archive(tmp_path: Path, *, speakers: tuple[str, ...] = ("1001", "1002")) -> Path:
    archive_path = tmp_path / "train-clean-100.tar.gz"
    metadata = {
        "LICENSE.TXT": b"fixture license",
        "README.TXT": b"fixture readme",
        "CHAPTERS.TXT": b"fixture chapters",
        "BOOKS.TXT": b"fixture books",
        "SPEAKERS.TXT": (
            b"; id | gender | subset | minutes | name\n"
            + b"".join(
                f"{speaker} | F | train-clean-100 | 1.0 | Fixture {speaker}\n".encode()
                for speaker in speakers
            )
        ),
    }
    with tarfile.open(archive_path, "w:gz") as archive:
        for name in (
            "LICENSE.TXT",
            "README.TXT",
            "CHAPTERS.TXT",
            "SPEAKERS.TXT",
            "BOOKS.TXT",
        ):
            _add_member(archive, f"LibriSpeech/{name}", metadata[name])
        for speaker_index, speaker in enumerate(speakers):
            chapter = "001"
            for utterance_index in range(2):
                utterance = f"{speaker}-{chapter}-{utterance_index:04d}"
                _add_member(
                    archive,
                    f"LibriSpeech/train-clean-100/{speaker}/{chapter}/{utterance}.flac",
                    _flac_bytes(speaker_index * 10 + utterance_index),
                )
    return archive_path


def test_fixture_preparation_is_deterministic_and_keeps_old_rows_first(tmp_path: Path) -> None:
    old_root, old_teacher, old_ids = _old_base(tmp_path)
    archive = _train_archive(tmp_path)
    output = tmp_path / "expanded"

    result = expanded_data.prepare_expanded_dataset(
        archive,
        old_root,
        output,
        original_teacher_path=old_teacher,
        test_fixture=True,
        fixture_speakers=2,
        fixture_utterances_per_speaker=2,
    )

    selected = expanded_data.load_v2_manifest(
        output / "selected" / "manifest.json",
        expected_dataset=expanded_data.TRAIN_DATASET,
        test_fixture=True,
    )
    expanded = expanded_data.load_v2_manifest(
        output / "expanded" / "manifest.json",
        expected_dataset=expanded_data.EXPANDED_DATASET,
        test_fixture=True,
    )
    assert result["status"] == "SYNTHETIC_TEST_FIXTURE"
    assert Path(result["selected"]["manifest"]).is_file()
    assert Path(result["expanded"]["manifest"]).is_file()
    assert len(selected.rows) == 4
    assert tuple(row["utterance_id"] for row in expanded.rows[:2]) == old_ids
    assert tuple(row["utterance_id"] for row in expanded.rows[2:]) == tuple(
        row["utterance_id"] for row in selected.rows
    )
    assert all(".." not in row["wav_path"] for row in expanded.rows)
    with np.load(output / "expanded" / "features.npz", allow_pickle=False) as arrays:
        assert arrays["features"].shape == (6, 298, 40, 1)
        assert arrays["features"].dtype == np.float32
        assert arrays["ids"].tolist() == [row["utterance_id"] for row in expanded.rows]
        assert arrays["manifest_sha256"].item() == expanded.manifest_sha256
    provenance = json.loads((output / "expanded" / "provenance.json").read_text())
    assert provenance["status"] == "SYNTHETIC_TEST_FIXTURE"
    assert provenance["source"]["archive"]["expected_md5"] is None
    assert provenance["source"]["observed_sha256_is_not_signature"] is True


def test_archive_md5_gate_precedes_tar_scan_and_writes(tmp_path: Path) -> None:
    old_root, old_teacher, _ = _old_base(tmp_path)
    archive = _train_archive(tmp_path)
    output = tmp_path / "must-not-exist"

    with pytest.raises(expanded_data.ExpandedDataError, match="MD5 mismatch"):
        expanded_data.prepare_expanded_dataset(
            archive,
            old_root,
            output,
            original_teacher_path=old_teacher,
            test_fixture=False,
        )
    assert not output.exists()


@pytest.mark.parametrize("kind", ["traversal", "symlink", "duplicate", "shadow", "root"])
def test_archive_security_is_checked_before_selected_writes(tmp_path: Path, kind: str) -> None:
    old_root, old_teacher, _ = _old_base(tmp_path)
    archive = tmp_path / f"{kind}.tar.gz"
    metadata = {
        "LICENSE.TXT": b"license",
        "README.TXT": b"readme",
        "CHAPTERS.TXT": b"chapters",
        "SPEAKERS.TXT": b"1001 | F | train-clean-100 | 1.0 | Fixture\n",
        "BOOKS.TXT": b"books",
    }
    with tarfile.open(archive, "w:gz") as handle:
        for name in metadata:
            _add_member(handle, f"LibriSpeech/{name}", metadata[name])
        if kind == "traversal":
            _add_member(handle, "LibriSpeech/train-clean-100/1001/001/../escape.flac", b"bad")
        elif kind == "symlink":
            link = tarfile.TarInfo("LibriSpeech/train-clean-100/1001/001/1001-001-0000.flac")
            link.type = tarfile.SYMTYPE
            link.linkname = "/etc/passwd"
            handle.addfile(link)
        elif kind == "duplicate":
            _add_member(handle, "LibriSpeech/README.TXT", b"duplicate")
        elif kind == "shadow":
            _add_member(handle, "LibriSpeech/train-clean-100/1001", b"shadow")
            _add_member(handle, "LibriSpeech/train-clean-100/1001/001/1001-001-0000.flac", b"bad")
        else:
            _add_member(handle, "OtherRoot/README.TXT", b"wrong root")
    output = tmp_path / f"out-{kind}"
    with pytest.raises(expanded_data.ExpandedDataError):
        expanded_data.prepare_expanded_dataset(
            archive,
            old_root,
            output,
            original_teacher_path=old_teacher,
            test_fixture=True,
            fixture_speakers=1,
            fixture_utterances_per_speaker=1,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(f"out-{kind}*/*"))


def test_v2_validator_rejects_extra_fields_duplicates_and_split_leakage(tmp_path: Path) -> None:
    old_root, old_teacher, _ = _old_base(tmp_path)
    archive = _train_archive(tmp_path)
    output = tmp_path / "expanded"
    expanded_data.prepare_expanded_dataset(
        archive,
        old_root,
        output,
        original_teacher_path=old_teacher,
        test_fixture=True,
        fixture_speakers=2,
        fixture_utterances_per_speaker=2,
    )
    manifest_path = output / "selected" / "manifest.json"
    original = json.loads(manifest_path.read_text())

    extra = json.loads(json.dumps(original))
    extra["rows"][0]["unexpected"] = True
    manifest_path.write_text(json.dumps(extra), encoding="utf-8")
    with pytest.raises(expanded_data.ExpandedDataError, match="row fields"):
        expanded_data.load_v2_manifest(manifest_path, test_fixture=True)

    duplicate = json.loads(json.dumps(original))
    duplicate["rows"][1]["utterance_id"] = duplicate["rows"][0]["utterance_id"]
    manifest_path.write_text(json.dumps(duplicate), encoding="utf-8")
    with pytest.raises(expanded_data.ExpandedDataError, match="duplicate utterance"):
        expanded_data.load_v2_manifest(manifest_path, test_fixture=True)

    leakage = json.loads(json.dumps(original))
    leakage["dataset"] = expanded_data.EXPANDED_DATASET
    leakage["rows"][0]["split"] = "validation"
    leakage["rows"][1]["split"] = "train"
    manifest_path.write_text(json.dumps(leakage), encoding="utf-8")
    with pytest.raises(expanded_data.ExpandedDataError, match="speakers overlap"):
        expanded_data.load_v2_manifest(manifest_path, test_fixture=True)


def test_hash_archive_reports_observed_hash_without_signature_claim(tmp_path: Path) -> None:
    archive = _train_archive(tmp_path)
    digest = expanded_data.hash_archive(archive, expected_md5=None)
    assert len(digest.sha256) == 64
    assert digest.expected_md5 is None
    assert digest.as_dict()["md5_status"] == "SYNTHETIC_TEST_FIXTURE_NOT_OFFICIAL"


@pytest.mark.parametrize("mismatched_key", ["manifest_sha256", "features_sha256"])
def test_original_nested_artifact_hashes_are_verified(
    tmp_path: Path, mismatched_key: str
) -> None:
    root, _, _ = _old_base(tmp_path)
    manifest_path = root / "manifest.json"
    features_path = root / "features.npz"
    provenance_path = root / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["artifacts"] = {
        "manifest_sha256": expanded_data.sha256_file(manifest_path),
        "features_sha256": expanded_data.sha256_file(features_path),
    }
    provenance["artifacts"][mismatched_key] = "0" * 64
    provenance_path.write_text(json.dumps(provenance) + "\n", encoding="utf-8")

    with pytest.raises(expanded_data.ExpandedDataError, match="artifact .* mismatch"):
        expanded_data.load_original_base(root, test_fixture=True)


def test_preparation_rejects_changed_old_base_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old_root, old_teacher, _ = _old_base(tmp_path)
    archive = _train_archive(tmp_path)
    output = tmp_path / "changed-base-output"
    original_extract = expanded_data._extract_selected

    def extract_then_mutate(*args: object, **kwargs: object) -> object:
        result = original_extract(*args, **kwargs)
        features_path = old_root / "features.npz"
        features_path.write_bytes(features_path.read_bytes() + b"changed")
        return result

    monkeypatch.setattr(expanded_data, "_extract_selected", extract_then_mutate)
    with pytest.raises(expanded_data.ExpandedDataError, match="immutable base"):
        expanded_data.prepare_expanded_dataset(
            archive,
            old_root,
            output,
            original_teacher_path=old_teacher,
            test_fixture=True,
            fixture_speakers=2,
            fixture_utterances_per_speaker=2,
        )
    assert not output.exists()
