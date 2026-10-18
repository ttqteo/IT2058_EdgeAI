"""Focused tests for the additive comparative-training boundary.

The end-to-end case uses a deterministic 640-row synthetic fixture solely to
exercise the publication and immutability gates.  It is not pilot evidence and
these tests never run downstream evaluation or report quality metrics.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import edgeai.comparative_training as comparative


def _protocol_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for speaker_number in range(20):
        speaker = f"train-{speaker_number:02d}"
        for utterance_number in range(20):
            rows.append(
                {
                    "utterance_id": f"{speaker}-utt-{utterance_number:02d}",
                    "speaker_id": speaker,
                    "split": "train",
                }
            )
    for speaker_number in range(5):
        speaker = f"validation-{speaker_number:02d}"
        for utterance_number in range(12):
            rows.append(
                {
                    "utterance_id": f"{speaker}-utt-{utterance_number:02d}",
                    "speaker_id": speaker,
                    "split": "validation",
                }
            )
    return rows


def _full_synthetic_snapshot(tmp_path: Path) -> tuple[Path, Path]:
    """Create a loader-only fixture with the manifest's required split shape."""

    rows: list[dict[str, object]] = []

    def add_rows(split: str, speaker: str, count: int) -> None:
        for number in range(count):
            identifier = f"{split}-{speaker}-{number:02d}"
            digest = hashlib.sha256(identifier.encode("utf-8")).hexdigest()
            rows.append(
                {
                    "utterance_id": identifier,
                    "speaker_id": speaker,
                    "chapter_id": f"chapter-{speaker}",
                    "split": split,
                    "wav_path": f"wavs/{identifier}.wav",
                    "wav_sha256": digest,
                    "source_sha256": hashlib.sha256(
                        ("source-" + identifier).encode("utf-8")
                    ).hexdigest(),
                    "input_samples": 48000,
                    "padded_samples": 0,
                    "truncated_samples": 0,
                }
            )

    for speaker_number in range(20):
        add_rows("train", f"train-{speaker_number:02d}", 20)
    for speaker_number in range(5):
        add_rows("validation", f"validation-{speaker_number:02d}", 12)
    add_rows("calibration_enroll", "calibration-00", 3)
    add_rows("calibration_query", "calibration-00", 9)
    add_rows("test_enroll", "test-00", 3)
    add_rows("test_query", "test-00", 9)

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {"format_version": 1, "dataset": "synthetic", "seed": 0, "rows": rows},
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    manifest_sha256 = comparative._sha256_file(manifest_path)
    identifiers = np.asarray([row["utterance_id"] for row in rows], dtype=str)
    features_path = tmp_path / "features.npz"
    np.savez_compressed(
        features_path,
        ids=identifiers,
        features=np.zeros(
            (len(rows), *comparative.FEATURE_SHAPE), dtype=np.float32
        ),
        manifest_sha256=np.asarray(manifest_sha256),
    )
    return manifest_path, features_path


def _full_training_snapshot(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """Create all 640 rows needed by the locked protocol, for tests only."""

    rows: list[dict[str, object]] = []

    def add_rows(split: str, speaker: str, count: int) -> None:
        for number in range(count):
            identifier = f"{split}-{speaker}-{number:02d}"
            rows.append(
                {
                    "utterance_id": identifier,
                    "speaker_id": speaker,
                    "chapter_id": f"chapter-{speaker}",
                    "split": split,
                    "wav_path": f"wavs/{identifier}.wav",
                    "wav_sha256": hashlib.sha256(
                        ("wav-" + identifier).encode("utf-8")
                    ).hexdigest(),
                    "source_sha256": hashlib.sha256(
                        ("source-" + identifier).encode("utf-8")
                    ).hexdigest(),
                    "input_samples": 48000,
                    "padded_samples": 0,
                    "truncated_samples": 0,
                }
            )

    for speaker_number in range(20):
        add_rows("train", f"train-{speaker_number:02d}", 20)
    for speaker_number in range(5):
        add_rows("validation", f"validation-{speaker_number:02d}", 12)
    for speaker_number in range(5):
        add_rows("calibration_enroll", f"calibration-{speaker_number:02d}", 3)
        add_rows("calibration_query", f"calibration-{speaker_number:02d}", 9)
    for speaker_number in range(10):
        add_rows("test_enroll", f"test-{speaker_number:02d}", 3)
        add_rows("test_query", f"test-{speaker_number:02d}", 9)
    assert len(rows) == 640

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "dataset": "synthetic-comparison-fixture",
                "seed": 20261012,
                "rows": rows,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    manifest_sha256 = comparative._sha256_file(manifest_path)
    identifiers = np.asarray([row["utterance_id"] for row in rows], dtype=str)
    rng = np.random.default_rng(20261012)
    features = rng.normal(
        0.0,
        0.05,
        size=(len(rows), *comparative.FEATURE_SHAPE),
    ).astype(np.float32)
    features_path = tmp_path / "features.npz"
    np.savez_compressed(
        features_path,
        ids=identifiers,
        features=features,
        manifest_sha256=np.asarray(manifest_sha256),
    )
    teacher = rng.normal(size=(len(rows), comparative.EMBEDDING_DIM)).astype(np.float32)
    teacher /= np.linalg.norm(teacher, axis=1, keepdims=True).astype(np.float32)
    teacher_path = tmp_path / "teacher_embeddings.npz"
    np.savez_compressed(
        teacher_path,
        ids=identifiers,
        embeddings=teacher,
        manifest_sha256=np.asarray(manifest_sha256),
    )
    provenance_path = tmp_path / "provenance.json"
    provenance_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "status": comparative.SYNTHETIC_STATUS,
                "artifacts": {
                    "manifest_sha256": manifest_sha256,
                    "features_sha256": comparative._sha256_file(features_path),
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return manifest_path, features_path, teacher_path, provenance_path


def test_balanced_schedule_is_deterministic_and_covers_train_once() -> None:
    rows = _protocol_rows()
    first = comparative.balanced_epoch_schedule(rows, epoch=1)
    repeat = comparative.balanced_epoch_schedule(rows, epoch=1)
    next_epoch = comparative.balanced_epoch_schedule(rows, epoch=2)

    assert first.shape == (20, 20)
    np.testing.assert_array_equal(first, repeat)
    assert not np.array_equal(first, next_epoch)
    assert len(set(first.reshape(-1).tolist())) == 400
    train_indices = set(range(400))
    assert set(first.reshape(-1).tolist()) == train_indices
    for batch in first:
        assert {rows[index]["speaker_id"] for index in batch} == {
            f"train-{speaker:02d}" for speaker in range(20)
        }


def test_validation_protocol_is_sorted_and_speaker_disjoint() -> None:
    rows = _protocol_rows()
    protocol = comparative.validation_protocol(rows)

    assert protocol["train_indices"].size == 400
    assert protocol["validation_indices"].size == 60
    assert protocol["enrollment_indices"].size == 15
    assert protocol["query_indices"].size == 45
    assert set(protocol["train_indices"]).isdisjoint(
        set(protocol["validation_indices"])
    )
    for speaker in protocol["validation_speakers"]:
        values = protocol["validation_by_speaker"][speaker]
        assert [rows[index]["utterance_id"] for index in values[:3]] == sorted(
            rows[index]["utterance_id"] for index in values[:3]
        )
        assert len(set(values[:3]) & set(values[3:])) == 0
    assert all(rows[index]["split"] == "train" for index in protocol["train_indices"])
    assert all(
        rows[index]["split"] == "validation"
        for index in protocol["validation_indices"]
    )


def test_checkpoint_selection_uses_strict_improvement_and_patience() -> None:
    result = comparative.select_validation_checkpoint(
        [0.50, 0.40, 0.40, 0.41, 0.42], patience=2
    )
    assert result["best_epoch"] == 2
    assert result["best_validation_eer"] == pytest.approx(0.40)
    assert result["stop_epoch"] == 4
    assert result["stop_reason"] == "early_stopping"
    assert result["strict_improvement"] is True


def test_standalone_supervised_loader_does_not_need_teacher_file(tmp_path: Path) -> None:
    manifest_path, features_path = _full_synthetic_snapshot(tmp_path)

    loaded = comparative._load_features_only(
        manifest_path,
        features_path,
        synthetic_fixture=True,
    )
    _, rows, _, features, identifiers, provenance = loaded
    assert len(rows) == 484
    assert features.shape == (484, *comparative.FEATURE_SHAPE)
    assert len(identifiers) == 484
    assert provenance is None


def test_output_overwrite_is_rejected_before_loading_inputs(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    features = tmp_path / "features.npz"
    teacher = tmp_path / "teacher.npz"
    manifest.write_bytes(b"not used")
    features.write_bytes(b"not used")
    teacher.write_bytes(b"not used")
    output = tmp_path / "existing"
    output.mkdir()

    with pytest.raises(ValueError, match="refusing to overwrite"):
        comparative.train_comparative_pair(
            manifest,
            features,
            teacher,
            output,
            synthetic_fixture=True,
            max_epochs=1,
            strict_thread_config=False,
        )


def test_synthetic_full_pair_publishes_bound_exports_without_evaluation(
    tmp_path: Path,
) -> None:
    tf = pytest.importorskip("tensorflow")
    manifest, features, teacher, provenance = _full_training_snapshot(tmp_path)
    output = tmp_path / "comparison-trained"

    report = comparative.train_comparative_pair(
        manifest,
        features,
        teacher,
        output,
        synthetic_fixture=True,
        max_epochs=1,
        strict_thread_config=False,
    )
    assert output.is_dir()
    root = json.loads(
        (output / "comparison_training_report.json").read_text(encoding="utf-8")
    )
    freeze = json.loads(
        (output / "comparison_freeze.json").read_text(encoding="utf-8")
    )
    assert report["input_hashes"] == root["input_hashes"]
    assert root["input_status"] == comparative.SYNTHETIC_STATUS
    assert root["data_snapshot"]["provenance_path"] == str(provenance.resolve())
    assert root["data_snapshot"]["provenance_sha256"] == comparative._sha256_file(
        provenance
    )
    assert root["input_hashes"]["provenance_sha256"] == comparative._sha256_file(
        provenance
    )
    assert root["input_paths"]["provenance_sha256"] == str(provenance.resolve())
    assert root["evaluation"] == {
        "calibration": None,
        "test": None,
        "status": "NOT_RUN_BY_COMPARATIVE_TRAINING",
        "downstream_evaluator_call": False,
    }
    assert root["publish"]["final_path_recheck"] is True
    assert root["publish"]["exclusive_writer_assumption"] is True
    assert root["publish"]["cross_process_race_free_no_overwrite_guarantee"] is False
    assert freeze["config"]["learning_rate"] == comparative.LEARNING_RATE
    assert freeze["config"]["selection_rule"].startswith(
        "strict minimum validation EER"
    )
    assert freeze["input_paths"] == root["input_paths"]
    assert freeze["input_hashes"] == root["input_hashes"]
    assert freeze["paired_initial_weights"]["exact"] is True
    assert (
        freeze["paired_initial_weights"]["encoder"]["kd_aam_sha256"]
        == freeze["paired_initial_weights"]["encoder"]["aam_supervised_sha256"]
    )
    assert (
        freeze["paired_initial_weights"]["classifier"]["kd_aam_sha256"]
        == freeze["paired_initial_weights"]["classifier"]["aam_supervised_sha256"]
    )

    for method, report_name in (
        ("kd_aam", "kd_aam_report.json"),
        ("aam_supervised", "aam_supervised_report.json"),
    ):
        method_report = json.loads(
            (output / method / report_name).read_text(encoding="utf-8")
        )
        method_freeze = freeze["methods"][method]["checkpoints"]
        assert method_report["checkpoint_bindings"] == method_freeze
        assert method_report["data_snapshot"]["provenance_sha256"] == root[
            "input_hashes"
        ]["provenance_sha256"]
        if method == "kd_aam":
            assert method_report["teacher_used"] is True
            assert "teacher_embeddings_sha256" in method_report["input_hashes"]
        else:
            assert method_report["teacher_used"] is False
            assert method_report["teacher_provenance"] is None
            assert "teacher_embeddings_sha256" not in method_report["input_hashes"]
        assert method_report["evaluation"]["calibration"] is None
        assert method_report["evaluation"]["test"] is None
        for binding in method_freeze.values():
            artifact = Path(binding["path"])
            assert artifact.is_file()
            assert artifact.stat().st_size == binding["bytes"]
            assert comparative._sha256_file(artifact) == binding["sha256"]
        assert (output / method / "encoder_fp32.tflite").is_file()
        assert (output / method / "encoder_int8.tflite").is_file()
        for embeddings_name in (
            "encoder_fp32_embeddings.npz",
            "encoder_int8_embeddings.npz",
        ):
            with np.load(output / method / embeddings_name, allow_pickle=False) as archive:
                assert archive["ids"].shape == (640,)
                assert archive["embeddings"].shape == (640, comparative.EMBEDDING_DIM)

    # Re-open the two frozen initial files and prove the arrays, rather than
    # merely their report fields, are exactly paired before either optimizer.
    initial_models = []
    for method in ("kd_aam", "aam_supervised"):
        model = comparative._training.build_student_model(
            name=f"initial-weights-{method}"
        )
        model.load_weights(output / method / "encoder_initial.weights.h5")
        initial_models.append(model.get_weights())
    for left, right in zip(*initial_models):
        np.testing.assert_array_equal(left, right)
    with np.load(output / "kd_aam" / "classifier_initial.npz") as kd_archive:
        with np.load(
            output / "aam_supervised" / "classifier_initial.npz"
        ) as supervised_archive:
            np.testing.assert_array_equal(kd_archive["kernel"], supervised_archive["kernel"])

    with pytest.raises(ValueError, match="refusing to overwrite"):
        comparative.train_comparative_pair(
            manifest,
            features,
            teacher,
            output,
            synthetic_fixture=True,
            max_epochs=1,
            strict_thread_config=False,
        )
    tf.keras.backend.clear_session()


def test_pair_rejects_provenance_mutation_before_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("tensorflow")
    manifest, features, teacher, provenance = _full_training_snapshot(tmp_path)
    output = tmp_path / "comparison-provenance-mutation"
    original_fit = comparative._fit_method
    fit_calls = 0

    def mutate_after_fit(*args: object, **kwargs: object) -> object:
        nonlocal fit_calls
        result = original_fit(*args, **kwargs)
        fit_calls += 1
        if fit_calls == 2:
            provenance.write_bytes(provenance.read_bytes() + b"\nmutation")
        return result

    monkeypatch.setattr(comparative, "_fit_method", mutate_after_fit)
    with pytest.raises(
        RuntimeError,
        match=r"pre-export: input/source changed during comparison: provenance_sha256",
    ):
        comparative.train_comparative_pair(
            manifest,
            features,
            teacher,
            output,
            synthetic_fixture=True,
            max_epochs=1,
            strict_thread_config=False,
        )
    assert fit_calls == 2
    assert not output.exists()


def test_pair_rejects_checkpoint_mutation_before_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("tensorflow")
    manifest, features, teacher, _ = _full_training_snapshot(tmp_path)
    output = tmp_path / "comparison-checkpoint-mutation"
    original_export = comparative._export_encoder
    export_calls = 0

    def mutate_after_export(*args: object, **kwargs: object) -> object:
        nonlocal export_calls
        result = original_export(*args, **kwargs)
        export_calls += 1
        if export_calls == 1:
            stage = Path(kwargs["stage"])
            checkpoint = stage / "encoder_best.weights.h5"
            checkpoint.write_bytes(checkpoint.read_bytes() + b"\nmutation")
        return result

    monkeypatch.setattr(comparative, "_export_encoder", mutate_after_export)
    with pytest.raises(
        RuntimeError,
        match=r"pre-commit: checkpoint changed during comparison: kd_aam/encoder_best",
    ):
        comparative.train_comparative_pair(
            manifest,
            features,
            teacher,
            output,
            synthetic_fixture=True,
            max_epochs=1,
            strict_thread_config=False,
        )
    assert export_calls == 2
    assert not output.exists()


def test_supervised_loss_has_no_teacher_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    tf = pytest.importorskip("tensorflow")
    embeddings = tf.constant(np.ones((4, 192), dtype=np.float32))
    kernel = tf.constant(np.ones((192, 20), dtype=np.float32))
    labels = tf.constant([0, 1, 2, 3], dtype=tf.int32)

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("supervised AAM called the KD loss")

    monkeypatch.setattr(comparative, "_cosine_kd_loss", forbidden)
    total, classification, kd = comparative._method_losses(
        tf,
        embeddings,
        labels,
        kernel,
        method="aam_supervised",
    )
    assert kd is None
    assert float(total.numpy()) == pytest.approx(float(classification.numpy()))

    with pytest.raises(ValueError, match="must not receive teacher"):
        comparative._method_losses(
            tf,
            embeddings,
            labels,
            kernel,
            method="aam_supervised",
            teacher_embeddings=embeddings,
        )


def test_normalized_classifier_matches_pytorch_eps_floor() -> None:
    tf = pytest.importorskip("tensorflow")
    embeddings = np.asarray(
        [[3.0, 4.0], [1.0e-14, -2.0e-14]], dtype=np.float32
    )
    kernel = np.asarray(
        [[2.0, -1.0, 0.5], [1.0, 3.0, -2.0]], dtype=np.float32
    )
    actual = comparative.normalized_classifier(
        tf,
        tf.constant(embeddings),
        tf.constant(kernel),
    ).numpy()
    embedding_norm = np.linalg.norm(embeddings.astype(np.float64), axis=1, keepdims=True)
    kernel_norm = np.linalg.norm(kernel.astype(np.float64), axis=0, keepdims=True)
    expected = (
        embeddings.astype(np.float64) / np.maximum(embedding_norm, 1e-12)
    ) @ (kernel.astype(np.float64) / np.maximum(kernel_norm, 1e-12))
    np.testing.assert_allclose(actual, expected.astype(np.float32), rtol=2e-6, atol=2e-6)
    assert actual.dtype == np.float32

    with pytest.raises((tf.errors.InvalidArgumentError, ValueError)):
        comparative.normalized_classifier(
            tf,
            tf.constant([[0.0, 0.0]], dtype=tf.float32),
            tf.constant(kernel),
        )
    with pytest.raises((tf.errors.InvalidArgumentError, ValueError)):
        comparative.normalized_classifier(
            tf,
            tf.constant([[np.nan, 1.0]], dtype=tf.float32),
            tf.constant(kernel),
        )


def test_aam_logits_public_signature_returns_float32() -> None:
    tf = pytest.importorskip("tensorflow")
    cosine = tf.constant(
        [[0.2, -0.3, 0.8], [0.99, 0.1, -0.6]], dtype=tf.float32
    )
    labels = tf.constant([2, 0], dtype=tf.int32)
    logits = comparative.aam_logits(tf, cosine, labels)
    assert logits.shape == (2, 3)
    assert logits.dtype == tf.float32
    assert np.all(np.isfinite(logits.numpy()))
    with pytest.raises(ValueError, match="integer dtype"):
        comparative.aam_logits(tf, cosine, tf.constant([0.0, 1.0]))


def test_kd_teacher_is_stop_gradient_and_gradients_are_finite() -> None:
    tf = pytest.importorskip("tensorflow")
    rng = np.random.default_rng(7)
    embeddings = tf.Variable(rng.normal(size=(4, 192)).astype(np.float32))
    kernel = tf.Variable(rng.normal(size=(192, 20)).astype(np.float32))
    teacher = tf.Variable(rng.normal(size=(4, 192)).astype(np.float32))
    labels = tf.constant([0, 1, 2, 3], dtype=tf.int32)
    with tf.GradientTape(persistent=True) as tape:
        total, classification, kd = comparative._method_losses(
            tf,
            embeddings,
            labels,
            kernel,
            method="kd_aam",
            teacher_embeddings=teacher,
        )
    gradients = tape.gradient(total, [embeddings, kernel])
    teacher_gradient = tape.gradient(total, teacher)
    assert kd is not None
    assert all(gradient is not None for gradient in gradients)
    assert all(np.all(np.isfinite(gradient.numpy())) for gradient in gradients)
    assert teacher_gradient is None
    assert np.isfinite(float(classification.numpy()))


def test_source_contains_no_downstream_test_evaluator_call() -> None:
    source = Path(comparative.__file__).read_text(encoding="utf-8")
    assert "evaluate_embeddings" not in source
    assert "NOT_RUN_BY_COMPARATIVE_TRAINING" in source


def test_method_reports_separate_kd_and_supervised_provenance(tmp_path: Path) -> None:
    rows = _protocol_rows()
    protocol = comparative._protocol_indices(rows)
    source_hashes = {"comparative_training.py": "a" * 64, "training.py": "b" * 64}

    def make_method_stage(name: str) -> tuple[Path, dict[str, object]]:
        stage = tmp_path / name
        stage.mkdir()
        for filename in (
            "encoder_initial.weights.h5",
            "encoder_best.weights.h5",
            "classifier_initial.npz",
            "classifier_best.npz",
            "encoder_fp32.tflite",
            "encoder_int8.tflite",
            "encoder_fp32_embeddings.npz",
            "encoder_int8_embeddings.npz",
        ):
            (stage / filename).write_bytes(b"synthetic-test-artifact")
        export = {
            "fp32_path": stage / "encoder_fp32.tflite",
            "int8_path": stage / "encoder_int8.tflite",
            "fp32_embeddings_path": stage / "encoder_fp32_embeddings.npz",
            "int8_embeddings_path": stage / "encoder_int8_embeddings.npz",
            "fp32_metadata": {"input": {"dtype": "float32"}},
            "int8_metadata": {"input": {"dtype": "int8"}},
            "dynamic_fixed_batch1_max_abs": 0.0,
            "dynamic_fixed_batch1_atol": 1e-6,
            "representative_rows": 400,
            "representative_source": "train only",
            "int8_vs_fp32_drift": {"scope": "synthetic test only"},
        }
        return stage, export

    kd_stage, kd_export = make_method_stage("kd")
    supervised_stage, supervised_export = make_method_stage("supervised")
    fit = {
        "history": [],
        "best_epoch": 1,
        "best_validation_eer": 0.0,
        "epochs_ran": 1,
        "stop_reason": "max_epochs",
        "weight_update_max_abs": 1.0,
    }
    manifest = {"dataset": "synthetic"}
    kd = comparative._method_report(
        method="kd_aam",
        fit=fit,
        export=kd_export,
        stage=kd_stage,
        final=tmp_path / "out" / "kd_aam",
        source_hashes=source_hashes,
        input_hashes={"manifest_sha256": "c" * 64, "features_sha256": "d" * 64, "teacher_embeddings_sha256": "e" * 64},
        manifest_path=tmp_path / "manifest.json",
        features_path=tmp_path / "features.npz",
        manifest=manifest,
        rows=rows,
        protocol=protocol,
        provenance=None,
        teacher_used=True,
        synthetic_fixture=True,
        max_epochs=1,
    )
    supervised = comparative._method_report(
        method="aam_supervised",
        fit=fit,
        export=supervised_export,
        stage=supervised_stage,
        final=tmp_path / "out" / "aam_supervised",
        source_hashes=source_hashes,
        input_hashes={"manifest_sha256": "c" * 64, "features_sha256": "d" * 64},
        manifest_path=tmp_path / "manifest.json",
        features_path=tmp_path / "features.npz",
        manifest=manifest,
        rows=rows,
        protocol=protocol,
        provenance=None,
        teacher_used=False,
        synthetic_fixture=True,
        max_epochs=1,
    )
    assert kd["teacher_used"] is True
    assert kd["teacher_model"] is None
    assert kd["teacher_revision"] is None
    assert supervised["teacher_used"] is False
    assert supervised["teacher_provenance"] is None
    assert supervised["teacher_model"] is None
    assert "teacher_embeddings_sha256" not in supervised["input_hashes"]
    assert supervised["evaluation"]["test"] is None


def test_encoder_only_fp32_and_int8_conversion_and_invoke() -> None:
    tf = pytest.importorskip("tensorflow")
    model = comparative._training.build_student_model(
        batch_size=1, name="comparison_conversion_fixture"
    )
    features = np.random.default_rng(11).normal(
        size=(2, *comparative.FEATURE_SHAPE)
    ).astype(np.float32)
    fp32 = comparative._training._convert_fp32(tf, model)
    int8 = comparative._training._convert_int8(
        tf, model, features, np.asarray([0, 1], dtype=np.int64)
    )
    fp32_raw, fp32_unit, fp32_meta = comparative._training._run_tflite(
        fp32, features, require_int8=False
    )
    int8_raw, int8_unit, int8_meta = comparative._training._run_tflite(
        int8, features, require_int8=True
    )
    assert fp32_raw.shape == int8_raw.shape == (2, comparative.EMBEDDING_DIM)
    assert fp32_unit.shape == int8_unit.shape == fp32_raw.shape
    assert fp32_meta["input"]["shape"] == [1, *comparative.FEATURE_SHAPE]
    assert int8_meta["input"]["dtype"] == "int8"
    assert int8_meta["output"]["dtype"] == "int8"
