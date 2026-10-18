"""Focused synthetic tests for the additive enhancement trainer.

These fixtures exercise protocol, loss, freeze, conversion, and holdout
boundaries only.  They are explicitly synthetic and are not accuracy or
deployment evidence.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from edgeai import comparative_training as comparative
from edgeai import enhancement_training as enhancement


def _row(
    index: int, speaker: str, split: str, utterance: str
) -> dict[str, object]:
    return {
        "utterance_id": utterance,
        "speaker_id": speaker,
        "chapter_id": f"chapter-{speaker}",
        "split": split,
        "wav_path": f"synthetic/{utterance}.wav",
        "wav_sha256": hashlib.sha256(f"wav-{index}".encode()).hexdigest(),
        "source_sha256": hashlib.sha256(f"source-{index}".encode()).hexdigest(),
        "input_samples": 48000,
        "padded_samples": 0,
        "truncated_samples": 0,
    }


def _protocol_rows(train_speakers: int = 20) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    index = 0
    for speaker_index in range(train_speakers):
        speaker = f"tr{speaker_index:03d}"
        for utterance_index in range(20):
            rows.append(
                _row(index, speaker, "train", f"train-{speaker}-{utterance_index:02d}")
            )
            index += 1
    for speaker_index in range(5):
        speaker = f"va{speaker_index:03d}"
        for utterance_index in range(12):
            rows.append(
                _row(index, speaker, "validation", f"validation-{speaker}-{utterance_index:02d}")
            )
            index += 1
    return rows


@pytest.fixture(scope="session")
def synthetic_snapshot(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path, list[dict[str, object]]]:
    root = tmp_path_factory.mktemp("enhancement-snapshot")
    rows = _protocol_rows(20)
    index = len(rows)
    for speaker_index in range(5):
        speaker = f"ca{speaker_index:03d}"
        for utterance_index in range(12):
            rows.append(
                _row(index, speaker, "calibration_enroll" if utterance_index < 3 else "calibration_query",
                     f"calibration-{speaker}-{utterance_index:02d}")
            )
            index += 1
    for speaker_index in range(10):
        speaker = f"te{speaker_index:03d}"
        for utterance_index in range(12):
            rows.append(
                _row(index, speaker, "test_enroll" if utterance_index < 3 else "test_query",
                     f"test-{speaker}-{utterance_index:02d}")
            )
            index += 1
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "format_version": 1,
                "dataset": "synthetic-enhancement-fixture",
                "seed": enhancement.SEED,
                "rows": rows,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    rng = np.random.default_rng(20261013)
    features = rng.normal(0.0, 0.2, size=(len(rows), *enhancement.FEATURE_SHAPE)).astype(
        np.float32
    )
    features_path = root / "features.npz"
    np.savez_compressed(
        features_path,
        ids=np.asarray([str(row["utterance_id"]) for row in rows]),
        features=features,
        manifest_sha256=np.asarray(manifest_sha),
    )
    teacher = rng.normal(0.0, 1.0, size=(len(rows), enhancement.EMBEDDING_DIM)).astype(
        np.float32
    )
    teacher /= np.linalg.norm(teacher.astype(np.float64), axis=1, keepdims=True).astype(
        np.float32
    )
    teacher_path = root / "teacher_embeddings.npz"
    np.savez_compressed(
        teacher_path,
        ids=np.asarray([str(row["utterance_id"]) for row in rows]),
        embeddings=teacher,
        manifest_sha256=np.asarray(manifest_sha),
    )
    provenance = root / "provenance.json"
    provenance.write_text(
        json.dumps(
            {
                "format_version": 1,
                "status": "SYNTHETIC_TEST_FIXTURE",
                "artifacts": {
                    "manifest_sha256": manifest_sha,
                    "features_sha256": hashlib.sha256(features_path.read_bytes()).hexdigest(),
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return manifest, features_path, teacher_path, rows


def test_balanced_schedule_covers_20_and_120_speakers() -> None:
    for speaker_count, expected_batches in ((20, 20), (120, 120)):
        rows = _protocol_rows(speaker_count)
        schedule = enhancement.balanced_epoch_schedule(rows, epoch=1)
        assert schedule.shape == (expected_batches, 20)
        assert len(set(schedule.reshape(-1).tolist())) == speaker_count * 20
        assert np.array_equal(
            schedule,
            enhancement.balanced_epoch_schedule(rows, epoch=1),
        )
        for batch in schedule:
            assert len({rows[index]["speaker_id"] for index in batch.tolist()}) == 20


def test_validation_is_sorted_and_rejects_speaker_leakage() -> None:
    rows = _protocol_rows(20)
    protocol = enhancement.validation_protocol(rows)
    assert protocol["class_count"] == 20
    assert protocol["enrollment_ids"] == sorted(protocol["enrollment_ids"])
    assert len(protocol["enrollment_ids"]) == 15
    assert len(protocol["query_ids"]) == 45
    leaked = [dict(row) for row in rows]
    for row in leaked:
        if row["speaker_id"] == "va004":
            row["speaker_id"] = "tr000"
    with pytest.raises(ValueError, match="overlap"):
        enhancement.validation_protocol(leaked)


def test_relational_numpy_oracle_and_teacher_stop_gradient() -> None:
    rng = np.random.default_rng(8)
    student = rng.normal(size=(6, 192)).astype(np.float32)
    teacher = rng.normal(size=(6, 192)).astype(np.float32)
    labels = np.asarray([0, 1, 0, 2, 3, 3], dtype=np.int32)
    student_unit = student.astype(np.float64) / np.linalg.norm(
        student.astype(np.float64), axis=1, keepdims=True
    )
    teacher_unit = teacher.astype(np.float64) / np.linalg.norm(
        teacher.astype(np.float64), axis=1, keepdims=True
    )
    row, col = np.triu_indices(6, k=1)
    mask = labels[row] != labels[col]
    expected = float(
        np.mean(
            (
                np.sum(student_unit[row] * student_unit[col], axis=1)[mask]
                - np.sum(teacher_unit[row] * teacher_unit[col], axis=1)[mask]
            )
            ** 2
        )
    )
    assert enhancement.relational_kd_loss_numpy(student, teacher, labels) == pytest.approx(
        expected, abs=1e-12
    )
    student_variable = tf.Variable(student)
    teacher_variable = tf.Variable(teacher)
    with tf.GradientTape(persistent=True) as tape:
        loss = enhancement.relational_kd_loss(
            tf, student_variable, teacher_variable, labels
        )
    student_gradient = tape.gradient(loss, student_variable)
    teacher_gradient = tape.gradient(loss, teacher_variable)
    assert teacher_gradient is None
    assert student_gradient is not None
    assert bool(tf.reduce_all(tf.math.is_finite(student_gradient)))
    with pytest.raises((tf.errors.InvalidArgumentError, ValueError)):
        enhancement.relational_kd_loss(
            tf, student, teacher, np.zeros(6, dtype=np.int32)
        ).numpy()


def test_beta_zero_matches_frozen_kd_aam_algebra() -> None:
    rng = np.random.default_rng(9)
    embeddings = tf.convert_to_tensor(rng.normal(size=(20, 192)).astype(np.float32))
    teacher = tf.convert_to_tensor(rng.normal(size=(20, 192)).astype(np.float32))
    kernel = tf.convert_to_tensor(rng.normal(size=(192, 20)).astype(np.float32))
    labels = tf.convert_to_tensor(np.arange(20, dtype=np.int32))
    local = enhancement.enhancement_loss_components(
        tf,
        embeddings,
        labels,
        kernel,
        teacher_embeddings=teacher,
        kd_weight=1.0,
        relational_weight=0.0,
        supervised=False,
    )
    frozen_total, frozen_aam, frozen_kd = comparative._method_losses(
        tf,
        embeddings,
        labels,
        kernel,
        method="kd_aam",
        teacher_embeddings=teacher,
    )
    np.testing.assert_array_equal(local["total"].numpy(), frozen_total.numpy())
    np.testing.assert_array_equal(local["aam"].numpy(), frozen_aam.numpy())
    np.testing.assert_array_equal(local["cosine_kd"].numpy(), frozen_kd.numpy())
    assert local["relational_kd"] is None


def test_same_seed_pairs_encoder_initialization() -> None:
    tf.keras.utils.set_random_seed(enhancement.SEED)
    first = enhancement.build_enhancement_encoder("conv3_two_bin")
    tf.keras.utils.set_random_seed(enhancement.SEED)
    second = enhancement.build_enhancement_encoder("conv3_two_bin")
    for left, right in zip(first.get_weights(), second.get_weights()):
        np.testing.assert_array_equal(left, right)


def test_invalid_binding_rejects_before_training(
    synthetic_snapshot: tuple[Path, Path, Path, list[dict[str, object]]], tmp_path: Path
) -> None:
    manifest, features, teacher, _ = synthetic_snapshot
    bad_features = tmp_path / "bad_features.npz"
    with np.load(features, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    arrays["manifest_sha256"] = np.asarray("0" * 64)
    np.savez_compressed(bad_features, **arrays)
    with pytest.raises(ValueError, match="manifest_sha256"):
        enhancement.train_enhancement(
            manifest,
            bad_features,
            tmp_path / "bad-output",
            teacher_embeddings_path=teacher,
            synthetic_fixture=True,
            max_epochs=1,
            strict_thread_config=False,
        )
    bad_teacher = tmp_path / "bad_teacher.npz"
    with np.load(teacher, allow_pickle=False) as archive:
        np.savez_compressed(
            bad_teacher,
            ids=np.asarray(archive["ids"])[::-1],
            embeddings=np.asarray(archive["embeddings"]),
            manifest_sha256=np.asarray(archive["manifest_sha256"]),
        )
    with pytest.raises(ValueError, match="teacher ids"):
        enhancement.train_enhancement(
            manifest,
            features,
            tmp_path / "bad-teacher-output",
            teacher_embeddings_path=bad_teacher,
            synthetic_fixture=True,
            max_epochs=1,
            strict_thread_config=False,
        )


def test_end_to_end_conv3_exports_only_train_validation(
    synthetic_snapshot: tuple[Path, Path, Path, list[dict[str, object]]],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest, features, teacher, _ = synthetic_snapshot
    output = tmp_path / "conv3-output"
    report = enhancement.train_enhancement(
        manifest,
        features,
        output,
        teacher_embeddings_path=teacher,
        architecture="conv3_two_bin",
        kd_weight=1.0,
        relational_weight=0.25,
        synthetic_fixture=True,
        max_epochs=1,
        strict_thread_config=False,
    )
    assert report["architecture"]["parameter_count"] == 35296
    assert report["data_snapshot"]["output_row_count"] == 460
    assert report["data_snapshot"]["calibration_row_ids_written"] is None
    assert report["data_snapshot"]["test_row_ids_written"] is None
    assert report["evaluation"]["calibration"] is None
    assert report["evaluation"]["test"] is None
    assert report["training"]["teacher_stop_gradient"] is True
    assert report["exports"]["static_budget"]["static_budget_pass"] is True
    for artifact in (
        "encoder_fp32_tflite",
        "encoder_int8_tflite",
        "encoder_fp32_embeddings",
        "encoder_int8_embeddings",
    ):
        assert Path(report["exports"]["artifacts"][artifact]["path"]).is_file()
    output_ids = set(report["data_snapshot"]["output_row_ids"])
    assert not any(identifier.startswith(("calibration-", "test-")) for identifier in output_ids)
    assert json.loads((output / "enhancement_training_report.json").read_text()) == report
    cli_output = tmp_path / "cli-conv3-output"
    assert enhancement.main(
        [
            "--manifest", str(manifest),
            "--features", str(features),
            "--teacher-embeddings", str(teacher),
            "--output", str(cli_output),
            "--architecture", "conv3_two_bin",
            "--relational-weight", "0.25",
            "--max-epochs", "1",
            "--synthetic-fixture",
        ]
    ) == 0
    cli_report = json.loads(capsys.readouterr().out)
    assert cli_report["architecture"]["name"] == "conv3_two_bin"
    assert cli_report["data_snapshot"]["output_row_count"] == 460


def test_supervised_is_teacher_free_and_refuses_overwrite(
    synthetic_snapshot: tuple[Path, Path, Path, list[dict[str, object]]], tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, features, _, _ = synthetic_snapshot

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("supervised mode attempted to read teacher")

    monkeypatch.setattr(enhancement, "_load_teacher", forbidden)
    output = tmp_path / "supervised-output"
    report = enhancement.train_enhancement(
        manifest,
        features,
        output,
        teacher_embeddings_path=tmp_path / "does-not-exist.npz",
        architecture="baseline",
        kd_weight=0.0,
        relational_weight=0.0,
        supervised=True,
        synthetic_fixture=True,
        max_epochs=1,
        strict_thread_config=False,
    )
    assert report["teacher_provenance"]["teacher_used"] is False
    assert "teacher_embeddings_sha256" not in report["input_hashes"]
    with pytest.raises(ValueError, match="overwrite"):
        enhancement.train_enhancement(
            manifest,
            features,
            output,
            supervised=True,
            kd_weight=0.0,
            relational_weight=0.0,
            synthetic_fixture=True,
            max_epochs=1,
            strict_thread_config=False,
        )

