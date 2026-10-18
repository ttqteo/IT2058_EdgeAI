from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from edgeai.full_data import (
    FullDataError,
    FEATURE_SHAPE,
    WINDOW_SAMPLES,
    load_full_dataset,
    prepare_full_dataset,
)


def _member(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    archive.addfile(info, io.BytesIO(payload))


def _flac(frames: int, seed: int) -> bytes:
    values = ((np.arange(frames, dtype=np.int32) + seed * 101) % 30_000) - 15_000
    buffer = io.BytesIO()
    sf.write(buffer, values.astype(np.int16), 16_000, format="FLAC", subtype="PCM_16")
    return buffer.getvalue()


def _fixture_archive(tmp_path: Path) -> Path:
    path = tmp_path / "fixture.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        _member(archive, "LibriSpeech/README.TXT", b"synthetic fixture")
        _member(
            archive,
            "LibriSpeech/train-clean-100/1001/001/1001-001.trans.txt",
            b"1001-001-0000 synthetic transcript\n",
        )
        # Deliberately insert the higher ID first; preparation must sort by the
        # original utterance ID, not tar member encounter order.
        _member(
            archive,
            "LibriSpeech/train-clean-100/1002/001/1002-001-0001.flac",
            _flac(WINDOW_SAMPLES + 3, 2),
        )
        _member(
            archive,
            "LibriSpeech/train-clean-100/1001/001/1001-001-0000.flac",
            _flac(3, 1),
        )
    return path


def test_synthetic_full_preparation_has_stable_full_coverage_and_readonly_mmaps(
    tmp_path: Path,
) -> None:
    output = tmp_path / "full"
    result = prepare_full_dataset(
        _fixture_archive(tmp_path), output, test_fixture=True
    )
    dataset = load_full_dataset(output, test_fixture=True)

    assert result["status"] == "SYNTHETIC_TEST_FIXTURE"
    assert dataset.manifest["dataset"].startswith("synthetic-")
    assert len(dataset.rows) == 3
    assert [row["global_index"] for row in dataset.rows] == [0, 1, 2]
    assert [row["utterance_id"] for row in dataset.rows] == [
        "1001-001-0000",
        "1002-001-0001",
        "1002-001-0001",
    ]
    assert [row["start_sample"] for row in dataset.rows] == [0, 0, WINDOW_SAMPLES]
    assert dataset.rows[-1]["valid_samples"] == 3
    assert dataset.rows[-1]["padded_samples"] == WINDOW_SAMPLES - 3
    assert dataset.rows[-1]["window_id"] == "1002-001-0001@0000048000"
    assert dataset.pcm.shape == (3, WINDOW_SAMPLES)
    assert dataset.pcm.dtype == np.int16
    assert dataset.features.shape == (3, *FEATURE_SHAPE)
    assert dataset.features.dtype == np.float32
    assert not dataset.pcm.flags.writeable
    assert not dataset.features.flags.writeable
    assert np.isfinite(dataset.features).all()
    assert dataset.manifest["source"]["expected_md5"] is None
    assert json.loads((output / "provenance.json").read_text())["status"] == (
        "SYNTHETIC_TEST_FIXTURE"
    )


def test_full_loader_rejects_array_hash_and_finite_mutations(tmp_path: Path) -> None:
    output = tmp_path / "full"
    prepare_full_dataset(_fixture_archive(tmp_path), output, test_fixture=True)
    features = np.load(output / "features.npy", mmap_mode="r+")
    try:
        features[0, 0, 0, 0] = np.nan
        features.flush()
    finally:
        del features
    with pytest.raises(FullDataError, match="artifact hash mismatch"):
        load_full_dataset(output, test_fixture=True)
    with pytest.raises(FullDataError, match="non-finite"):
        load_full_dataset(output, verify_hashes=False, test_fixture=True)


def test_full_preparation_refuses_overwrite_and_archive_traversal(tmp_path: Path) -> None:
    archive = _fixture_archive(tmp_path)
    output = tmp_path / "full"
    prepare_full_dataset(archive, output, test_fixture=True)
    with pytest.raises(FullDataError, match="overwrite"):
        prepare_full_dataset(archive, output, test_fixture=True)

    bad_archive = tmp_path / "bad.tar.gz"
    with tarfile.open(bad_archive, "w:gz") as handle:
        _member(handle, "LibriSpeech/train-clean-100/1001/001/../escape.flac", b"bad")
    with pytest.raises(FullDataError, match="traversal"):
        prepare_full_dataset(bad_archive, tmp_path / "bad-output", test_fixture=True)


def test_transcript_whitelist_rejects_wrong_speaker_chapter_identity(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "wrong-transcript.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        _member(
            handle,
            "LibriSpeech/train-clean-100/1001/001/1001-002.trans.txt",
            b"wrong chapter identity",
        )
    with pytest.raises(FullDataError, match="unexpected regular archive member"):
        prepare_full_dataset(archive, tmp_path / "wrong-output", test_fixture=True)


def test_manifest_row_mutation_is_rejected_before_array_use(tmp_path: Path) -> None:
    output = tmp_path / "full"
    prepare_full_dataset(_fixture_archive(tmp_path), output, test_fixture=True)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["rows"][0]["window_id"] = "changed@0000000000"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(FullDataError):
        load_full_dataset(output, test_fixture=True)
