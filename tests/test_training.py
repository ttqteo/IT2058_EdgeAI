from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from edgeai.training import (
    FEATURE_SHAPE,
    MAX_EPOCHS,
    _summarize_batch2_diagnostic,
    _validate_batch1_export_parity,
    build_student_model,
    train_pilot,
)


def _write_fixture(root: Path) -> tuple[Path, Path, Path, list[dict[str, object]]]:
    """Create a fully self-contained, explicitly synthetic pilot snapshot."""

    rows: list[dict[str, object]] = []

    def add_rows(speaker: str, split: str, count: int, chapter: str) -> None:
        for index in range(count):
            identifier = f"{speaker}-{chapter}-{split}-{index}"
            rows.append(
                {
                    "utterance_id": identifier,
                    "speaker_id": speaker,
                    "chapter_id": chapter,
                    "split": split,
                    "wav_path": f"wav/{identifier}.wav",
                    "wav_sha256": sha256(f"wav:{identifier}".encode()).hexdigest(),
                    "source_sha256": sha256(
                        f"source:{identifier}".encode()
                    ).hexdigest(),
                    "input_samples": 48000,
                    "padded_samples": 0,
                    "truncated_samples": 0,
                }
            )

    # Four disjoint speaker groups exercise the real leakage boundary while
    # keeping the focused fixture small.
    for speaker in ("tr0", "tr1"):
        add_rows(speaker, "train", 4, "train-chapter")
    for speaker in ("va0", "va1"):
        add_rows(speaker, "validation", 4, "validation-chapter")
    for speaker in ("ca0", "ca1"):
        add_rows(speaker, "calibration_enroll", 3, "calibration-enroll")
        add_rows(speaker, "calibration_query", 9, "calibration-query")
    for speaker in ("te0", "te1"):
        add_rows(speaker, "test_enroll", 3, "test-enroll")
        add_rows(speaker, "test_query", 9, "test-query")

    manifest = {
        "format_version": 1,
        "dataset": "synthetic-training-fixture",
        "seed": 20261012,
        "rows": rows,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    manifest_sha256 = sha256(manifest_path.read_bytes()).hexdigest()

    rng = np.random.default_rng(20261012)
    features = rng.normal(
        loc=0.0,
        scale=0.2,
        size=(len(rows), *FEATURE_SHAPE),
    ).astype(np.float32)
    teacher = rng.normal(size=(len(rows), 192)).astype(np.float32)
    teacher /= np.linalg.norm(teacher.astype(np.float64), axis=1)[:, None].astype(
        np.float32
    )
    ids = np.asarray([row["utterance_id"] for row in rows], dtype=str)
    features_path = root / "features.npz"
    np.savez_compressed(
        features_path,
        ids=ids,
        features=features,
        manifest_sha256=np.asarray(manifest_sha256),
    )
    teacher_path = root / "teacher_embeddings.npz"
    np.savez_compressed(
        teacher_path,
        ids=ids,
        embeddings=teacher,
        manifest_sha256=np.asarray(manifest_sha256),
    )
    return manifest_path, features_path, teacher_path, rows


@pytest.fixture
def snapshot(tmp_path: Path):
    return _write_fixture(tmp_path)


def test_student_topology_and_dynamic_fixed_parity() -> None:
    tf = pytest.importorskip("tensorflow")
    dynamic = build_student_model()
    fixed = build_student_model(batch_size=1)

    assert dynamic.count_params() == 146192
    assert dynamic.input_shape == (None, 298, 40, 1)
    assert dynamic.output_shape == (None, 192)
    assert fixed.input_shape == (1, 298, 40, 1)
    assert fixed.output_shape == (1, 192)

    fixed.set_weights(dynamic.get_weights())
    dynamic_clone = tf.keras.models.clone_model(
        fixed,
        input_tensors=tf.keras.Input(shape=FEATURE_SHAPE, dtype=tf.float32),
    )
    dynamic_clone.set_weights(fixed.get_weights())
    values = np.random.default_rng(4).normal(
        size=(2, *FEATURE_SHAPE)
    ).astype(np.float32)
    dynamic_max_abs, _, _ = _validate_batch1_export_parity(
        dynamic, fixed, values[:1]
    )
    clone_max_abs, _, _ = _validate_batch1_export_parity(
        dynamic_clone, fixed, values[:1]
    )
    assert dynamic_max_abs == 0.0
    assert clone_max_abs == 0.0
    dynamic_output = dynamic_clone(values, training=False).numpy()
    assert dynamic_output.shape == (2, 192)


def test_export_parity_rejects_mismatched_weights() -> None:
    pytest.importorskip("tensorflow")
    dynamic = build_student_model(name="parity_dynamic")
    fixed = build_student_model(batch_size=1, name="parity_fixed")
    fixed.set_weights(dynamic.get_weights())
    weights = fixed.get_weights()
    weights[0] = weights[0].copy()
    weights[0].flat[0] += np.float32(0.25)
    fixed.set_weights(weights)
    values = np.zeros((1, *FEATURE_SHAPE), dtype=np.float32)

    with pytest.raises(RuntimeError, match="export model weights differ"):
        _validate_batch1_export_parity(dynamic, fixed, values)


def test_batch2_drift_is_diagnostic_only() -> None:
    pytest.importorskip("tensorflow")
    dynamic = build_student_model(name="diagnostic_dynamic")
    fixed = build_student_model(batch_size=1, name="diagnostic_fixed")
    fixed.set_weights(dynamic.get_weights())
    values = np.random.default_rng(5).normal(
        size=(2, *FEATURE_SHAPE)
    ).astype(np.float32)
    _, _, fixed_output = _validate_batch1_export_parity(
        dynamic, fixed, values[:1]
    )

    # A clone's batch-2 output is intentionally a diagnostic comparison only.
    tf = pytest.importorskip("tensorflow")
    clone = tf.keras.models.clone_model(
        fixed,
        input_tensors=tf.keras.Input(shape=FEATURE_SHAPE, dtype=tf.float32),
    )
    clone.set_weights(fixed.get_weights())
    batched_output = clone(values, training=False).numpy()
    diagnostic = _summarize_batch2_diagnostic(fixed_output, batched_output)
    assert diagnostic["comparison"] == "fixed_batch1_vs_dynamic_clone_batch2_row0"
    assert diagnostic["interpretation"] == (
        "diagnostic_only_not_backend_accuracy_guarantee"
    )
    assert diagnostic["max_abs"] >= 0.0
    assert diagnostic["max_relative"] >= 0.0
    assert 0.0 <= diagnostic["normalized_cosine"] <= 1.0
    assert diagnostic["combined_bound_ratio"] >= 0.0
    assert diagnostic["combined_bound_status"] in {"PASS", "FAIL"}

    # Even a deliberately larger absolute drift is summarized, never treated
    # as the same-batch export hard-gate failure.
    oversized = batched_output.copy()
    oversized[0, 0] += np.float32(2e-6)
    oversized_diagnostic = _summarize_batch2_diagnostic(fixed_output, oversized)
    assert oversized_diagnostic["max_abs"] > 1e-6
    assert oversized_diagnostic["combined_bound_pass"] is False
    assert oversized_diagnostic["combined_bound_status"] == "FAIL"


def test_snapshot_id_and_hash_guards(snapshot, tmp_path: Path) -> None:
    manifest_path, features_path, teacher_path, _ = snapshot
    with np.load(features_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["ids"] = arrays["ids"].copy()
    arrays["ids"][0] = "wrong-id"
    bad_features = tmp_path / "bad-features.npz"
    np.savez_compressed(bad_features, **arrays)
    with pytest.raises(ValueError, match="features ids"):
        train_pilot(
            manifest_path,
            bad_features,
            teacher_path,
            tmp_path / "unused-output",
            synthetic_fixture=True,
            max_epochs=1,
            strict_thread_config=False,
        )

    with np.load(teacher_path, allow_pickle=False) as archive:
        teacher_arrays = {name: archive[name] for name in archive.files}
    teacher_arrays["manifest_sha256"] = np.asarray("0" * 64)
    bad_teacher = tmp_path / "bad-teacher.npz"
    np.savez_compressed(bad_teacher, **teacher_arrays)
    with pytest.raises(ValueError, match="teacher manifest_sha256"):
        train_pilot(
            manifest_path,
            features_path,
            bad_teacher,
            tmp_path / "unused-output-2",
            synthetic_fixture=True,
            max_epochs=1,
            strict_thread_config=False,
        )


def test_real_provenance_nested_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    source_root = Path("artifacts/datasets/librispeech-dev-clean/prepared")
    source_teacher = Path("artifacts/teacher/pilot/teacher_embeddings.npz")
    required = [
        source_root / "manifest.json",
        source_root / "features.npz",
        source_root / "provenance.json",
        source_teacher,
    ]
    if not all(path.is_file() for path in required):
        pytest.skip("Lead-provided real pilot snapshot is unavailable")

    root = tmp_path / "real-snapshot-copy"
    root.mkdir()
    for name in ("manifest.json", "features.npz", "provenance.json"):
        shutil.copyfile(source_root / name, root / name)
    teacher_copy = root / "teacher_embeddings.npz"
    shutil.copyfile(source_teacher, teacher_copy)
    provenance = json.loads((root / "provenance.json").read_text(encoding="utf-8"))
    provenance["artifacts"]["features_sha256"] = "0" * 64
    (root / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match=r"provenance artifacts\.features_sha256"):
        train_pilot(
            root / "manifest.json",
            root / "features.npz",
            teacher_copy,
            tmp_path / "unused-real-output",
        )


def test_manifest_speaker_leakage_and_real_epoch_lock(snapshot, tmp_path: Path) -> None:
    manifest_path, features_path, teacher_path, _ = snapshot
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["rows"][0]["speaker_id"] = "va0"
    leaked_manifest = tmp_path / "leaked-manifest.json"
    leaked_manifest.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="speaker leakage"):
        train_pilot(
            leaked_manifest,
            features_path,
            teacher_path,
            tmp_path / "unused-output-3",
            synthetic_fixture=True,
            max_epochs=1,
            strict_thread_config=False,
        )

    with pytest.raises(ValueError, match="locked max_epochs"):
        train_pilot(
            manifest_path,
            features_path,
            teacher_path,
            tmp_path / "unused-output-4",
            max_epochs=1,
            strict_thread_config=False,
        )


def test_synthetic_training_updates_exports_and_evaluates_after_lock(
    snapshot, tmp_path: Path
) -> None:
    pytest.importorskip("tensorflow")
    manifest_path, features_path, teacher_path, rows = snapshot
    output = tmp_path / "training-output"
    report = train_pilot(
        manifest_path,
        features_path,
        teacher_path,
        output,
        synthetic_fixture=True,
        max_epochs=2,
        strict_thread_config=False,
    )

    assert output.is_dir()
    persisted = json.loads(
        (output / "pilot_training_report.json").read_text(encoding="utf-8")
    )
    assert persisted["status"] == "TRAINED_KD_PILOT"
    assert report["status"] == "TRAINED_KD_PILOT"
    assert report["input_status"] == "SYNTHETIC_TEST_FIXTURE"
    assert report["teacher_status"] == "SYNTHETIC_TEST_FIXTURE"
    assert report["data_snapshot"]["teacher_model"] is None
    assert report["data_snapshot"]["teacher_revision"] is None
    assert report["config"]["max_epochs"] == 2
    assert report["config"]["synthetic_fixture_override"] is True
    assert report["architecture"]["parameter_count"] == 146192
    assert report["architecture"]["dynamic_fixed_batch1_max_abs"] <= 1e-6
    assert report["architecture"]["dynamic_fixed_batch1_comparison"] == (
        "original_dynamic_vs_fixed_same_batch1"
    )
    assert report["architecture"]["dynamic_batch_output_shape"] == [2, 192]
    batch2_diagnostic = report["architecture"]["dynamic_batch2_diagnostic"]
    assert batch2_diagnostic["comparison"] == (
        "fixed_batch1_vs_dynamic_clone_batch2_row0"
    )
    assert batch2_diagnostic["interpretation"] == (
        "diagnostic_only_not_backend_accuracy_guarantee"
    )
    assert batch2_diagnostic["combined_bound_ratio"] >= 0.0
    assert report["training"]["epochs_ran"] == 2
    assert report["training"]["weight_update_max_abs"] > 0.0
    assert report["split_usage"]["representative_rows"] == 8
    assert report["split_usage"]["representative_split"] == (
        "train only, manifest order"
    )

    artifact_names = (
        "student_untrained_embeddings",
        "student_fp32_embeddings",
        "student_int8_embeddings",
    )
    embeddings: dict[str, np.ndarray] = {}
    for name in artifact_names:
        artifact_path = Path(report["artifacts"][name]["path"])
        assert artifact_path.is_file()
        with np.load(artifact_path, allow_pickle=False) as archive:
            assert archive["ids"].dtype.kind == "U"
            assert archive["ids"].tolist() == [row["utterance_id"] for row in rows]
            assert archive["embeddings"].dtype == np.dtype(np.float32)
            assert archive["embeddings"].shape == (len(rows), 192)
            assert archive["manifest_sha256"].item() == report["data_snapshot"][
                "manifest_sha256"
            ]
            embeddings[name] = archive["embeddings"].copy()
        assert report["artifacts"][name]["sha256"]

    assert not np.allclose(
        embeddings["student_untrained_embeddings"],
        embeddings["student_fp32_embeddings"],
        atol=1e-7,
        rtol=1e-7,
    )
    for values in embeddings.values():
        np.testing.assert_allclose(
            np.linalg.norm(values.astype(np.float64), axis=1), 1.0, atol=2e-4
        )

    fp32 = report["tflite"]["fp32"]
    int8 = report["tflite"]["int8"]
    assert fp32["input"]["dtype"] == "float32"
    assert fp32["output"]["dtype"] == "float32"
    assert int8["input"]["dtype"] == "int8"
    assert int8["output"]["dtype"] == "int8"
    assert int8["input"]["shape"] == [1, 298, 40, 1]
    assert int8["output"]["shape"] == [1, 192]
    assert int8["operations"]
    assert int8["float32_tensor_names"] == []
    assert report["int8_vs_fp32_drift"]["raw_max_abs"] >= 0.0

    for model_name in (
        "teacher",
        "student_untrained",
        "student_fp32",
        "student_int8",
    ):
        for split in ("calibration", "test"):
            metric = report["evaluations"][model_name][split]
            assert metric["score_trials"] == 36
            assert len(metric["scores"]) == 36
            assert metric["closed_set_top1"]["total"] == 18
            assert metric["threshold_scope"].startswith("retrospective")

    assert report["hardware_status"] == "UNVERIFIED_HARDWARE"
    assert report["contract_status"] == "CONTRACT_UNRESOLVED"


def test_training_refuses_existing_output(snapshot, tmp_path: Path) -> None:
    manifest_path, features_path, teacher_path, _ = snapshot
    output = tmp_path / "already-there"
    output.mkdir()
    with pytest.raises(ValueError, match="refusing to overwrite"):
        train_pilot(
            manifest_path,
            features_path,
            teacher_path,
            output,
            synthetic_fixture=True,
            max_epochs=1,
            strict_thread_config=False,
        )


def test_locked_constant_is_not_silently_reduced_for_real_runs() -> None:
    assert MAX_EPOCHS == 80
