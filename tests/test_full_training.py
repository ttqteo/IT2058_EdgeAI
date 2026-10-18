"""Bounded synthetic checks for the full train-clean-100 trainer.

These tests never load the real archive, validation bundle, or teacher cache.
The fixtures are explicitly synthetic and only exercise the public trainer
contract plus the bounded scheduling/checkpoint/export machinery.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pytest

from edgeai import full_training as training
from edgeai.full_data import FullDataset


def _readonly_memmap(path: Path, shape: tuple[int, ...], dtype: np.dtype, values: np.ndarray) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    created = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    created[...] = np.asarray(values, dtype=dtype)
    created.flush()
    del created
    return np.load(path, mmap_mode="r", allow_pickle=False)


def _train_row(index: int, speaker: str) -> dict[str, object]:
    return {
        "split": "train",
        "global_index": index,
        "utterance_id": f"utt-{speaker}-{index:04d}",
        "window_id": f"utt-{speaker}-{index:04d}@{index:010d}",
        "speaker_id": speaker,
        "chapter_id": f"chapter-{speaker}",
        "start_sample": index * training.WINDOW_SAMPLES,
        "input_samples": training.WINDOW_SAMPLES,
        "valid_samples": training.WINDOW_SAMPLES,
        "padded_samples": 0,
        "source_sha256": "a" * 64,
    }


def _validation_row(index: int, speaker: str) -> dict[str, object]:
    return {
        "split": "validation",
        "utterance_id": f"val-{speaker}-{index:02d}",
        "speaker_id": speaker,
    }


def _synthetic_bundle(tmp_path: Path, counts: tuple[int, ...] = (2, 3, 5)) -> tuple[FullDataset, training.ValidationDataset]:
    rng = np.random.default_rng(20261012)
    train_rows: list[dict[str, object]] = []
    for speaker_index, count in enumerate(counts):
        speaker = f"train-{speaker_index:03d}"
        for _ in range(count):
            train_rows.append(_train_row(len(train_rows), speaker))
    train_features = rng.normal(
        0.0, 0.25, size=(len(train_rows), *training.FEATURE_SHAPE)
    ).astype(np.float32)
    train_pcm = np.zeros(
        (len(train_rows), training.WINDOW_SAMPLES), dtype=np.int16
    )
    train_root = tmp_path / "synthetic-full"
    train_root.mkdir()
    pcm = _readonly_memmap(
        train_root / "pcm_int16.npy",
        train_pcm.shape,
        np.dtype("<i2"),
        train_pcm,
    )
    features = _readonly_memmap(
        train_root / "features.npy",
        train_features.shape,
        np.dtype("<f4"),
        train_features,
    )
    dataset = FullDataset(
        root=train_root,
        manifest={"dataset": "synthetic-full"},
        rows=train_rows,
        pcm=pcm,
        features=features,
        manifest_sha256="b" * 64,
    )

    validation_rows = [
        _validation_row(index, f"validation-{speaker_index:02d}")
        for speaker_index in range(5)
        for index in range(12)
    ]
    validation_features = rng.normal(
        0.0, 0.25, size=(60, *training.FEATURE_SHAPE)
    ).astype(np.float32)
    validation_root = tmp_path / "synthetic-validation"
    validation_root.mkdir()
    validation_mmap = _readonly_memmap(
        validation_root / "features.npy",
        validation_features.shape,
        np.dtype("<f4"),
        validation_features,
    )
    validation = training.ValidationDataset(
        root=validation_root,
        manifest={"dataset": "synthetic-validation"},
        rows=validation_rows,
        features=validation_mmap,
        manifest_sha256="c" * 64,
        features_sha256="d" * 64,
        provenance={"status": "SYNTHETIC_TEST_FIXTURE"},
    )
    return dataset, validation


def test_interleaved_sampler_covers_variable_counts_and_final_partial_batch() -> None:
    speaker_rows = {
        "speaker-a": np.array([0, 1], dtype=np.int64),
        "speaker-b": np.array([2, 3, 4], dtype=np.int64),
        "speaker-c": np.array([5, 6, 7, 8, 9], dtype=np.int64),
    }
    first = training.interleaved_epoch_batches(
        speaker_rows, epoch=4, seed=training.SEED, batch_size=4
    )
    second = training.interleaved_epoch_batches(
        speaker_rows, epoch=4, seed=training.SEED, batch_size=4
    )
    assert [batch.tolist() for batch in first] == [batch.tolist() for batch in second]
    flat = np.concatenate(first)
    assert sorted(flat.tolist()) == list(range(10))
    assert len(first[-1]) == 2
    assert max(len(batch) for batch in first) == 4


def test_teacher_norm_check_is_strict_and_bounded(tmp_path: Path) -> None:
    values = np.zeros((3, training.EMBEDDING_DIM), dtype=np.float32)
    values[:, 0] = 1.0
    valid = _readonly_memmap(
        tmp_path / "teacher-valid.npy",
        values.shape,
        np.dtype("<f4"),
        values,
    )
    assert training._validate_teacher_values(valid, 3) is valid

    invalid_values = values.copy()
    invalid_values[1, 0] = np.float32(1.0 + 2e-5)
    invalid = _readonly_memmap(
        tmp_path / "teacher-invalid.npy",
        invalid_values.shape,
        np.dtype("<f4"),
        invalid_values,
    )
    with pytest.raises(training.FullTrainingError, match="strict 1e-5"):
        training._validate_teacher_values(invalid, 3)


def test_validation_mapping_is_invariant_to_manifest_row_order(tmp_path: Path) -> None:
    dataset, validation = _synthetic_bundle(tmp_path)
    import tensorflow as tf

    encoder = tf.keras.Sequential([
        tf.keras.layers.Input(shape=training.FEATURE_SHAPE),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(192),
    ])
    protocol = training._training_protocol(dataset, validation)
    expected = training._validation_metrics(tf, encoder, validation, protocol)
    permutation = np.random.default_rng(718).permutation(60)
    shuffled_features = validation.features[permutation].copy()
    shuffled_features.flags.writeable = False
    shuffled = training.ValidationDataset(
        root=validation.root, manifest=validation.manifest,
        rows=[validation.rows[i] for i in permutation], features=shuffled_features,
        manifest_sha256=validation.manifest_sha256,
        features_sha256=validation.features_sha256, provenance=validation.provenance,
    )
    actual = training._validation_metrics(
        tf, encoder, shuffled, training._training_protocol(dataset, shuffled)
    )
    assert actual == expected


def test_best_eer_checkpoint_preserves_float64_ties() -> None:
    import tensorflow as tf
    encoder, head = training._build_models(tf, 3)
    optimizer = training._optimizer(
        tf, list(encoder.trainable_variables) + list(head.trainable_variables), learning_rate=1e-3
    )
    *_, best_eer = training._checkpoint_objects(tf, encoder, head, optimizer)
    best_eer.assign(0.1361111111111111)
    assert best_eer.dtype == tf.float64
    assert not (0.1361111111111111 < float(best_eer.numpy()))


def test_supervised_public_api_has_no_teacher_and_isolated_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    signature = inspect.signature(training.train_supervised_aam)
    assert "teacher_output_dir" not in signature.parameters

    dataset, validation = _synthetic_bundle(tmp_path)
    monkeypatch.setattr(
        training,
        "_teacher_inputs",
        lambda *_args, **_kwargs: pytest.fail("supervised path opened teacher"),
    )
    monkeypatch.setattr(training, "_export", _fake_export)
    report = training.train_supervised_aam(
        dataset,
        validation,
        tmp_path / "supervised",
        max_epochs=1,
        patience=1,
        batch_size=4,
        strict_thread_config=False,
    )
    assert report["method"] == "supervised_aam"
    assert report["teacher_provenance"] is None


def _fake_export(
    _tf: object,
    *,
    encoder: object,
    dataset: FullDataset,
    validation: training.ValidationDataset,
    protocol: dict[str, object],
    stage: Path,
    final_root: Path,
) -> dict[str, object]:
    del encoder, dataset, validation, protocol, final_root
    fp32 = stage / "encoder_fp32.tflite"
    int8 = stage / "encoder_int8.tflite"
    fp32.write_bytes(b"synthetic-fp32")
    int8.write_bytes(b"synthetic-int8")
    return {
        "encoder_fp32": {"path": str(stage / fp32.name), "bytes": fp32.stat().st_size},
        "encoder_int8": {"path": str(stage / int8.name), "bytes": int8.stat().st_size},
        "validation": {
            "fp32": {"eer": 0.5, "top1": {"accuracy": 0.0}},
            "int8": {"eer": 0.5, "top1": {"accuracy": 0.0}},
        },
        "selected_train_rows": 6,
        "selected_validation_rows": 60,
    }


def test_kd_trainer_consumes_aligned_readonly_teacher(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset, validation = _synthetic_bundle(tmp_path)
    rng = np.random.default_rng(822)
    values = rng.normal(size=(len(dataset.rows), 192)).astype(np.float32)
    values /= np.linalg.norm(values.astype(np.float64), axis=1, keepdims=True)
    teacher_values = _readonly_memmap(tmp_path / "teacher.npy", values.shape, np.dtype("float32"), values)
    calls = []

    def teacher_inputs(path, observed_dataset):
        assert observed_dataset is dataset
        calls.append(path)
        return teacher_values, {"status": "SYNTHETIC_FIXTURE"}, training._sha256_array_rows(teacher_values)

    monkeypatch.setattr(training, "_teacher_inputs", teacher_inputs)
    monkeypatch.setattr(training, "_export", _fake_export)
    report = training.train_kd_aam(
        dataset, validation, tmp_path / "teacher-cache", tmp_path / "kd",
        max_epochs=1, patience=1, batch_size=4, strict_thread_config=False,
    )
    assert calls and report["teacher_provenance"] is not None
    assert report["training"]["history"][0]["rows_seen"] == len(dataset.rows)
    assert report["training"]["history"][0]["train_kd_loss"] is not None


def test_resume_restores_adam_slots_and_reaches_same_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset, validation = _synthetic_bundle(tmp_path)
    monkeypatch.setattr(training, "_export", _fake_export)
    uninterrupted = training.train_supervised_aam(
        dataset,
        validation,
        tmp_path / "uninterrupted",
        max_epochs=2,
        patience=10,
        batch_size=4,
        strict_thread_config=False,
    )

    interrupted_output = tmp_path / "resumed"
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        training._run_training(
            dataset,
            validation,
            interrupted_output,
            method="supervised_aam",
            max_epochs=2,
            patience=10,
            batch_size=4,
            strict_thread_config=False,
            _interrupt_after_epoch=1,
        )
    stage = interrupted_output.with_name(interrupted_output.name + ".stage")
    assert stage.is_dir()
    assert not interrupted_output.exists()
    resumed = training.train_supervised_aam(
        dataset,
        validation,
        interrupted_output,
        resume_from=stage,
        max_epochs=2,
        patience=10,
        batch_size=4,
        strict_thread_config=False,
    )
    assert resumed["training"]["history"] == uninterrupted["training"]["history"]
    assert resumed["training"]["best_epoch"] == uninterrupted["training"]["best_epoch"]

    # The latest checkpoints include encoder/head weights and all Adam slots.
    import tensorflow as tf

    left = tmp_path / "uninterrupted" / "checkpoints" / "latest"
    right = tmp_path / "resumed" / "checkpoints" / "latest"
    left_prefix = tf.train.latest_checkpoint(str(left))
    right_prefix = tf.train.latest_checkpoint(str(right))
    assert left_prefix is not None and right_prefix is not None
    def canonical_name(name: str) -> str:
        return re.sub(r"adam_\d+", "adam", name)

    left_variables = {
        canonical_name(name): shape
        for name, shape in tf.train.list_variables(left_prefix)
        if name not in {"_CHECKPOINTABLE_OBJECT_GRAPH", "save_counter/.ATTRIBUTES/VARIABLE_VALUE"}
    }
    right_variables = {
        canonical_name(name): shape
        for name, shape in tf.train.list_variables(right_prefix)
        if name not in {"_CHECKPOINTABLE_OBJECT_GRAPH", "save_counter/.ATTRIBUTES/VARIABLE_VALUE"}
    }
    assert left_variables == right_variables
    for name in left_variables:
        left_name = next(
            candidate
            for candidate, _shape in tf.train.list_variables(left_prefix)
                if canonical_name(candidate) == name
                and candidate != "save_counter/.ATTRIBUTES/VARIABLE_VALUE"
        )
        right_name = next(
            candidate
            for candidate, _shape in tf.train.list_variables(right_prefix)
                if canonical_name(candidate) == name
                and candidate != "save_counter/.ATTRIBUTES/VARIABLE_VALUE"
        )
        np.testing.assert_array_equal(
            tf.train.load_variable(left_prefix, left_name),
            tf.train.load_variable(right_prefix, right_name),
            err_msg=f"checkpoint variable differs after resume: {name}",
        )


def test_resume_fails_closed_on_source_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset, validation = _synthetic_bundle(tmp_path)
    monkeypatch.setattr(training, "_export", _fake_export)
    output = tmp_path / "mutated"
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        training._run_training(
            dataset,
            validation,
            output,
            method="supervised_aam",
            max_epochs=1,
            patience=1,
            batch_size=4,
            strict_thread_config=False,
            _interrupt_after_epoch=1,
        )
    original = training._source_bindings
    monkeypatch.setattr(
        training,
        "_source_bindings",
        lambda method: {**original(method), "full_training.py": "0" * 64},
    )
    with pytest.raises(
        training.FullTrainingError,
        match="(config changed|source bindings changed)",
    ):
        training.train_supervised_aam(
            dataset,
            validation,
            output,
            resume_from=output.with_name(output.name + ".stage"),
            max_epochs=1,
            patience=1,
            batch_size=4,
            strict_thread_config=False,
        )


def test_export_interruption_preserves_latest_epoch_for_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset, validation = _synthetic_bundle(tmp_path)
    calls = {"count": 0}

    def fixed_validation(*_args: object, **_kwargs: object) -> dict[str, object]:
        return {
            "eer": 0.5,
            "eer_threshold": 0.0,
            "target_trials": 60,
            "non_target_trials": 240,
            "top1": {"correct": 0, "total": 60, "accuracy": 0.0},
        }

    monkeypatch.setattr(training, "_validation_metrics", fixed_validation)

    def flaky_export(*args: object, **kwargs: object) -> dict[str, object]:
        if calls["count"] == 0:
            calls["count"] += 1
            raise RuntimeError("synthetic export interruption")
        return _fake_export(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(training, "_export", flaky_export)
    output = tmp_path / "export-resume"
    with pytest.raises(RuntimeError, match="synthetic export interruption"):
        training.train_supervised_aam(
            dataset,
            validation,
            output,
            max_epochs=2,
            patience=10,
            batch_size=4,
            strict_thread_config=False,
        )
    stage = output.with_name(output.name + ".stage")
    state = json.loads((stage / training.STATE_FILENAME).read_text())
    assert state["status"] == "EXPORTING"
    assert state["completed_epoch"] == 2
    assert state["best_epoch"] == 1

    report = training.train_supervised_aam(
        dataset,
        validation,
        output,
        resume_from=stage,
        max_epochs=2,
        patience=10,
        batch_size=4,
        strict_thread_config=False,
    )
    assert report["training"]["epochs_ran"] == 2
    assert output.is_dir()
    assert not stage.exists()


def test_real_synthetic_e2e_publishes_expected_report_and_int8_io(
    tmp_path: Path,
) -> None:
    dataset, validation = _synthetic_bundle(tmp_path, counts=(2, 3, 5))
    # This test intentionally uses the real converter and host TFLite runner.
    report = training.train_supervised_aam(
        dataset,
        validation,
        tmp_path / "e2e",
        max_epochs=1,
        patience=1,
        batch_size=4,
        strict_thread_config=False,
    )
    output = tmp_path / "e2e"
    assert report["status"] == "FULL_TRAINING_COMPLETE"
    assert (output / training.REPORT_FILENAME).is_file()
    assert (output / "encoder_int8.tflite").is_file()
    assert not output.with_name(output.name + ".stage").exists()
    assert report["export"]["input_dtype"] == "int8"
    assert report["export"]["output_dtype"] == "int8"
    assert report["export"]["selected_train_rows"] == 6
    assert report["export"]["selected_validation_rows"] == 60
    assert report["evaluation"]["calibration"] is None
    assert report["evaluation"]["test"] is None
    assert report["export"]["encoder_int8"]["path"] == str(
        output / "encoder_int8.tflite"
    )
    assert report["training"]["portable_best"]["encoder"]["path"] == str(
        output / "encoder_best.weights.h5"
    )
    artifact = output / "encoder_int8.tflite"
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert report["export"]["encoder_int8"]["sha256"] == digest
    assert json.loads((output / training.REPORT_FILENAME).read_text()) == report


def test_cli_help_contract() -> None:
    environment = os.environ.copy()
    source_root = str(Path(__file__).resolve().parents[1] / "src")
    environment["PYTHONPATH"] = source_root + os.pathsep + environment.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-m", "edgeai.full_training", "--help"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode == 0, result.stderr
    for flag in (
        "--dataset-root",
        "--validation-root",
        "--output",
        "--method",
        "--teacher-output",
        "--resume",
    ):
        assert flag in result.stdout
