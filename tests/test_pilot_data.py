from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tarfile

import numpy as np
import pytest
import soundfile as sf

from edgeai import pilot_data


def _metadata_sources(tmp_path: Path) -> list[pilot_data.SourceUtterance]:
    sources: list[pilot_data.SourceUtterance] = []
    for speaker_index in range(40):
        speaker = f"{speaker_index:02d}"
        for utterance_index in range(20):
            chapter = "001" if utterance_index < 10 else "002"
            identifier = f"{speaker}-{chapter}-{utterance_index:04d}"
            sources.append(
                pilot_data.SourceUtterance(
                    speaker_id=speaker,
                    chapter_id=chapter,
                    utterance_id=identifier,
                    path=tmp_path / f"{identifier}.flac",
                )
            )
    return sources


def test_split_is_deterministic_disjoint_and_has_locked_counts(tmp_path) -> None:
    sources = _metadata_sources(tmp_path)
    selected_a, metadata_a = pilot_data._select_sources(sources, pilot_data.SEED)
    selected_b, metadata_b = pilot_data._select_sources(sources, pilot_data.SEED)

    assert [(item.utterance_id, split) for item, split in selected_a] == [
        (item.utterance_id, split) for item, split in selected_b
    ]
    assert metadata_a == metadata_b
    assert len(selected_a) == 640
    counts: dict[str, int] = {}
    for _, split in selected_a:
        counts[split] = counts.get(split, 0) + 1
    assert counts == {
        "train": 400,
        "validation": 60,
        "calibration_enroll": 15,
        "calibration_query": 45,
        "test_enroll": 30,
        "test_query": 90,
    }
    plan = metadata_a["speaker_plan"]
    groups = [set(plan[name]) for name in ("train", "validation", "calibration", "test")]
    assert sum(map(len, groups)) == 40
    assert sum(len(left & right) for index, left in enumerate(groups) for right in groups[index + 1 :]) == 0
    assert all(
        metadata_a["chapter_separation"][split][speaker]["separated"]
        for split in ("calibration", "test")
        for speaker in plan[split]
    )


@pytest.mark.parametrize("suffix", [".flac", ".wav"])
def test_soundfile_reads_small_fixture_and_records_crop(suffix: str, tmp_path: Path) -> None:
    source_path = tmp_path / f"12-34-00001{suffix}"
    samples = np.arange(16000, dtype=np.int16)
    sf.write(source_path, samples, 16000, format="FLAC" if suffix == ".flac" else "WAV", subtype="PCM_16")
    source = pilot_data.SourceUtterance("12", "34", "12-34-00001", source_path)

    prepared, input_samples, padded_samples, truncated_samples = pilot_data._read_segment(source)

    assert prepared.shape == (48000,)
    assert prepared.dtype == np.dtype("<i2")
    assert input_samples == 16000
    assert padded_samples == 32000
    assert truncated_samples == 0
    np.testing.assert_array_equal(prepared[:16000], samples)
    np.testing.assert_array_equal(prepared[16000:], np.zeros(32000, dtype=np.int16))


def test_safe_tar_validation_happens_before_any_member_is_written(tmp_path: Path) -> None:
    archive_path = tmp_path / "bad.tar"
    with tarfile.open(archive_path, "w") as archive:
        safe = tarfile.TarInfo("LibriSpeech/dev-clean/1/2/1-2-00001.wav")
        safe_data = b"safe"
        safe.size = len(safe_data)
        archive.addfile(safe, io.BytesIO(safe_data))
        traversal = tarfile.TarInfo("LibriSpeech/dev-clean/1/2/../escape.wav")
        traversal_data = b"bad"
        traversal.size = len(traversal_data)
        archive.addfile(traversal, io.BytesIO(traversal_data))
    destination = tmp_path / "extracted"

    with pytest.raises(ValueError, match="unsafe tar member path"):
        pilot_data._extract_verified_archive(archive_path, destination)
    assert not (destination / "LibriSpeech/dev-clean/1/2/1-2-00001.wav").exists()
    assert not (tmp_path / "escape.wav").exists()


def test_safe_tar_rejects_symlink(tmp_path: Path) -> None:
    archive_path = tmp_path / "link.tar"
    with tarfile.open(archive_path, "w") as archive:
        link = tarfile.TarInfo("LibriSpeech/dev-clean/1/2/1-2-00001.wav")
        link.type = tarfile.SYMTYPE
        link.linkname = "/etc/passwd"
        archive.addfile(link)
    with pytest.raises(ValueError, match="links"):
        pilot_data._extract_verified_archive(archive_path, tmp_path / "extracted")


def test_prepared_artifact_has_shared_manifest_and_features_fields(tmp_path: Path) -> None:
    source_path = tmp_path / "12-34-00001.wav"
    sf.write(source_path, np.zeros(48000, dtype=np.int16), 16000, format="WAV", subtype="PCM_16")
    source = pilot_data.SourceUtterance("12", "34", "12-34-00001", source_path)
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    archive = tmp_path / "dev-clean.tar.gz"
    archive.write_bytes(b"archive fixture")
    archive_md5 = hashlib.md5(archive.read_bytes()).hexdigest()
    archive_sha256 = hashlib.sha256(archive.read_bytes()).hexdigest()

    result = pilot_data._write_prepared(
        prepared,
        [(source, "train")],
        {"chapter_separation": {}},
        archive,
        archive_md5,
        archive_sha256,
        archive.stat().st_size,
        pilot_data.SEED,
    )

    manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
    with np.load(prepared / "features.npz", allow_pickle=False) as arrays:
        assert arrays["ids"].tolist() == ["12-34-00001"]
        assert arrays["features"].shape == (1, 298, 40, 1)
        assert arrays["features"].dtype == np.float32
        assert arrays["manifest_sha256"].item() == result["manifest_sha256"]
    assert manifest["rows"][0]["input_samples"] == 48000
    assert manifest["rows"][0]["padded_samples"] == 0
    assert manifest["rows"][0]["truncated_samples"] == 0
    provenance = json.loads((prepared / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["source"]["archive_md5"] == archive_md5
    assert provenance["source"]["archive_sha256"] == archive_sha256
    assert provenance["source"]["sha256_status"] == "observed_local_hash_not_signature"


def test_archive_checksum_rejection_precedes_extraction(tmp_path: Path) -> None:
    archive = tmp_path / "fixture.tar"
    with tarfile.open(archive, "w") as handle:
        member = tarfile.TarInfo("LibriSpeech/dev-clean/1/2/1-2-00001.wav")
        data = b"not audio"
        member.size = len(data)
        handle.addfile(member, io.BytesIO(data))
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match="MD5 mismatch"):
        pilot_data.prepare_dataset(archive, output)
    assert not output.exists()
    assert not (tmp_path / "extracted").exists()
