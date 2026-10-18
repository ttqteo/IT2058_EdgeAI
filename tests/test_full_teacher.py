from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from edgeai import full_teacher, teacher
from edgeai.full_data import load_full_dataset, prepare_full_dataset
from test_full_data import _fixture_archive


class _FixtureSnapshot:
    def as_dict(self) -> dict[str, object]:
        files = {
            name: {
                "path": f"fixture-model/{name}",
                "bytes": 1,
                "sha256": digest,
                "expected_sha256": digest,
                "hash_source": teacher.MODEL_FILE_HASH_SOURCE[name],
                "verified": True,
            }
            for name, digest in teacher.MODEL_FILE_SHA256.items()
        }
        return {
            "repo": teacher.MODEL_REPO,
            "revision": teacher.MODEL_REVISION,
            "path": "fixture-model",
            "revision_is_immutable_commit": True,
            "provenance_status": "PINNED_SNAPSHOT_HASH_MATCH_NO_CRYPTO_AUTHENTICATION",
            "files": files,
        }


class _FixtureObservation:
    def __init__(self, rows: int, *, valid: bool = True) -> None:
        self.embeddings = np.zeros((rows, 192), dtype=np.float32)
        if valid:
            self.embeddings[:, 0] = 1.0


def _prepared(tmp_path: Path) -> Path:
    root = tmp_path / "full"
    prepare_full_dataset(_fixture_archive(tmp_path), root, test_fixture=True)
    return root


def test_teacher_default_batch_four_publishes_mmap_and_public_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_root = _prepared(tmp_path)
    calls: list[tuple[int, int]] = []
    monkeypatch.setattr(full_teacher, "verify_model_snapshot", lambda _: _FixtureSnapshot())
    monkeypatch.setattr(full_teacher, "load_teacher", lambda *args, **kwargs: object())

    def fake_extract(_runtime: object, pcm: np.ndarray) -> _FixtureObservation:
        calls.append(tuple(pcm.shape))
        return _FixtureObservation(int(pcm.shape[0]))

    monkeypatch.setattr(full_teacher, "extract_batch", fake_extract)
    output = tmp_path / "teacher"
    report = full_teacher.extract_full_teacher(dataset_root, tmp_path / "model", output)
    dataset = load_full_dataset(dataset_root, test_fixture=True)
    embeddings = full_teacher.load_full_teacher(output, dataset)
    provenance = full_teacher.teacher_provenance(output, dataset)

    assert report["batch_size"] == 4
    assert report["threads"] == 4
    assert calls == [(3, 48_000)]
    assert embeddings.shape == (3, 192)
    assert embeddings.dtype == np.float32
    assert not embeddings.flags.writeable
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0)
    assert provenance["status"] == "VERIFIED_FULL_TEACHER_PROVENANCE"
    assert provenance["dataset"]["manifest_sha256"] == dataset.manifest_sha256
    assert provenance["teacher"]["sha256"] == report["teacher_embeddings_sha256"]
    assert set(provenance["extraction"]["dependencies"]) == {
        "numpy",
        "torch",
        "torchaudio",
        "speechbrain",
    }
    json.dumps(provenance)


def test_teacher_load_rejects_rebound_pinned_model_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_root = _prepared(tmp_path)
    monkeypatch.setattr(full_teacher, "verify_model_snapshot", lambda _: _FixtureSnapshot())
    monkeypatch.setattr(full_teacher, "load_teacher", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        full_teacher,
        "extract_batch",
        lambda _runtime, pcm: _FixtureObservation(int(np.asarray(pcm).shape[0])),
    )
    output = tmp_path / "teacher"
    full_teacher.extract_full_teacher(dataset_root, tmp_path / "model", output)
    config_path = output / "config.json"
    config = json.loads(config_path.read_text())
    config["model"]["revision"] = "not-the-pinned-revision"
    config["model_binding_sha256"] = full_teacher._model_binding(config["model"])
    config_path.write_text(json.dumps(config), encoding="utf-8")
    dataset = load_full_dataset(dataset_root, test_fixture=True)

    with pytest.raises(full_teacher.FullTeacherError, match="revision"):
        full_teacher.load_full_teacher(output, dataset)
    with pytest.raises(full_teacher.FullTeacherError, match="revision"):
        full_teacher.teacher_provenance(output, dataset)


def test_teacher_load_rejects_source_config_mutation_without_runtime_version_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_root = _prepared(tmp_path)
    monkeypatch.setattr(full_teacher, "verify_model_snapshot", lambda _: _FixtureSnapshot())
    monkeypatch.setattr(full_teacher, "load_teacher", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        full_teacher,
        "extract_batch",
        lambda _runtime, pcm: _FixtureObservation(int(np.asarray(pcm).shape[0])),
    )
    output = tmp_path / "teacher"
    full_teacher.extract_full_teacher(dataset_root, tmp_path / "model", output)
    config_path = output / "config.json"
    config = json.loads(config_path.read_text())
    config["extraction"]["sources"]["frontend"]["sha256"] = "0" * 64
    config_path.write_text(json.dumps(config), encoding="utf-8")
    dataset = load_full_dataset(dataset_root, test_fixture=True)

    # Consumer environments validate the persisted binding, but do not need
    # torch/torchaudio/SpeechBrain installed at their current versions.
    monkeypatch.setattr(
        full_teacher,
        "_installed_dependency_versions",
        lambda: {name: "different" for name in full_teacher._EXTRACTION_DISTRIBUTIONS},
    )
    with pytest.raises(full_teacher.FullTeacherError, match="source binding"):
        full_teacher.load_full_teacher(output, dataset)


def test_teacher_resume_keeps_verified_shard_after_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_root = _prepared(tmp_path)
    monkeypatch.setattr(full_teacher, "SHARD_ROWS", 2)
    monkeypatch.setattr(full_teacher, "verify_model_snapshot", lambda _: _FixtureSnapshot())
    monkeypatch.setattr(full_teacher, "load_teacher", lambda *args, **kwargs: object())
    invocations = {"count": 0}

    def fail_on_second_shard(_runtime: object, pcm: np.ndarray) -> _FixtureObservation:
        invocations["count"] += 1
        if invocations["count"] > 1:
            raise RuntimeError("synthetic interruption")
        return _FixtureObservation(int(pcm.shape[0]))

    monkeypatch.setattr(full_teacher, "extract_batch", fail_on_second_shard)
    output = tmp_path / "teacher"
    with pytest.raises(full_teacher.FullTeacherError, match="inference failed"):
        full_teacher.extract_full_teacher(dataset_root, tmp_path / "model", output)
    assert (output / "shards" / "shard-000000.npy").is_file()
    assert (output / "shards" / "shard-000000.json").is_file()
    assert not (output / "teacher_embeddings.npy").exists()

    resumed_calls: list[tuple[int, int]] = []

    def resume_extract(_runtime: object, pcm: np.ndarray) -> _FixtureObservation:
        resumed_calls.append(tuple(pcm.shape))
        return _FixtureObservation(int(pcm.shape[0]))

    monkeypatch.setattr(full_teacher, "extract_batch", resume_extract)
    full_teacher.extract_full_teacher(dataset_root, tmp_path / "model", output)
    assert resumed_calls == [(1, 48_000)]
    assert (output / "teacher_embeddings.npy").is_file()


def test_teacher_sidecar_mismatch_and_bad_norm_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert full_teacher.EMBEDDING_NORM_TOL == 1e-5
    dataset_root = _prepared(tmp_path)
    monkeypatch.setattr(full_teacher, "verify_model_snapshot", lambda _: _FixtureSnapshot())
    monkeypatch.setattr(full_teacher, "load_teacher", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        full_teacher,
        "extract_batch",
        lambda _runtime, pcm: _FixtureObservation(int(np.asarray(pcm).shape[0])),
    )
    output = tmp_path / "teacher"
    full_teacher.extract_full_teacher(dataset_root, tmp_path / "model", output)
    sidecar_path = output / "shards" / "shard-000000.json"
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["source_binding_sha256"] = "0" * 64
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    with pytest.raises(full_teacher.FullTeacherError, match="binding mismatch"):
        full_teacher.extract_full_teacher(dataset_root, tmp_path / "model", output)

    bad_output = tmp_path / "bad-teacher"
    monkeypatch.setattr(
        full_teacher,
        "extract_batch",
        lambda _runtime, pcm: _FixtureObservation(
            int(np.asarray(pcm).shape[0]), valid=False
        ),
    )
    with pytest.raises(full_teacher.FullTeacherError, match="norm"):
        full_teacher.extract_full_teacher(
            dataset_root, tmp_path / "model", bad_output
        )
    assert not (bad_output / "teacher_embeddings.npy").exists()


def test_teacher_refuses_completed_cache_with_different_batch_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_root = _prepared(tmp_path)
    monkeypatch.setattr(full_teacher, "verify_model_snapshot", lambda _: _FixtureSnapshot())
    monkeypatch.setattr(full_teacher, "load_teacher", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        full_teacher,
        "extract_batch",
        lambda _runtime, pcm: _FixtureObservation(int(np.asarray(pcm).shape[0])),
    )
    output = tmp_path / "teacher"
    full_teacher.extract_full_teacher(dataset_root, tmp_path / "model", output)
    with pytest.raises(full_teacher.FullTeacherError, match="config differs"):
        full_teacher.extract_full_teacher(
            dataset_root, tmp_path / "model", output, batch_size=3
        )
