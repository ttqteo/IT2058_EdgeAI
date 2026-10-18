"""Lead-owned verification of the saved, interrupted pilot implementation.

All generated inputs here are synthetic arithmetic fixtures, not evidence of
speaker-recognition quality. Real pilot quality is measured in a separate run.
"""

from pathlib import Path
import hashlib
import json

import numpy as np
import pytest

from edgeai import training


@pytest.fixture
def inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    rows = []
    for speaker, splits in (
        ("train", ["train"] * 2),
        ("val", ["validation"] * 2),
        ("cal", ["calibration_enroll"] * 3 + ["calibration_query"] * 9),
        ("test", ["test_enroll"] * 3 + ["test_query"] * 9),
    ):
        for index, split in enumerate(splits):
            name = f"{speaker}-{index}"
            rows.append({
                "utterance_id": name, "speaker_id": speaker, "chapter_id": "chapter",
                "split": split, "wav_path": f"wav/{name}.wav",
                "wav_sha256": hashlib.sha256((name + "wav").encode()).hexdigest(),
                "source_sha256": hashlib.sha256((name + "source").encode()).hexdigest(),
                "input_samples": 48000, "padded_samples": 0, "truncated_samples": 0,
            })
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"format_version": 1, "dataset": "SYNTHETIC_TEST_FIXTURE",
                                    "seed": training.SEED, "rows": rows}))
    bound = hashlib.sha256(manifest.read_bytes()).hexdigest()
    ids = np.asarray([row["utterance_id"] for row in rows])
    rng = np.random.default_rng(771)
    features = tmp_path / "features.npz"
    teacher = tmp_path / "teacher.npz"
    np.savez(features, ids=ids, manifest_sha256=np.asarray(bound),
             features=rng.normal(size=(len(rows), 298, 40, 1)).astype(np.float32))
    targets = rng.normal(size=(len(rows), 192)).astype(np.float32)
    targets /= np.linalg.norm(targets, axis=1, keepdims=True)
    np.savez(teacher, ids=ids, manifest_sha256=np.asarray(bound), embeddings=targets)
    return manifest, features, teacher


def test_snapshot_is_manifest_bound(inputs: tuple[Path, Path, Path]) -> None:
    snapshot = training._load_snapshot(*inputs, synthetic_fixture=True)
    assert snapshot[3].shape == (28, 298, 40, 1)
    assert snapshot[4].shape == (28, 192)


@pytest.mark.parametrize("fault", ["binding", "ids", "nonfinite", "dtype", "target_norm"])
def test_snapshot_rejects_invalid_arrays(inputs: tuple[Path, Path, Path], fault: str) -> None:
    path = inputs[2] if fault == "target_norm" else inputs[1]
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    if fault == "binding":
        arrays["manifest_sha256"] = np.asarray("0" * 64)
    elif fault == "ids":
        arrays["ids"] = arrays["ids"][::-1]
    elif fault == "nonfinite":
        arrays["features"][0, 0, 0, 0] = np.nan
    elif fault == "dtype":
        arrays["features"] = arrays["features"].astype(np.float64)
    else:
        arrays["embeddings"][0] *= 2
    np.savez(path, **arrays)
    with pytest.raises(ValueError):
        training._load_snapshot(*inputs, synthetic_fixture=True)


def test_speaker_leakage_rejected(inputs: tuple[Path, Path, Path]) -> None:
    manifest = inputs[0]
    payload = json.loads(manifest.read_text())
    payload["rows"][0]["speaker_id"] = "test"
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="speaker leakage"):
        training._load_snapshot(*inputs, synthetic_fixture=True)


def test_real_run_cannot_reduce_epochs(inputs: tuple[Path, Path, Path], tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="locked max_epochs"):
        training.train_pilot(*inputs, tmp_path / "out", max_epochs=2)


def test_closed_set_top1_is_explicit_and_ties_deterministic() -> None:
    result = training._closed_set_top1({"scores": [
        {"query_id": "q", "query_speaker_id": "b", "enrollment_speaker_id": "a", "score": .5},
        {"query_id": "q", "query_speaker_id": "b", "enrollment_speaker_id": "b", "score": .5},
    ]})
    assert result["accuracy"] == 0 and result["tie_queries"] == 1
    assert "not open_set" in result["scope"]


def test_gradient_checkpoint_export_and_actual_int8_invoke(tmp_path: Path) -> None:
    tf = pytest.importorskip("tensorflow")
    tf.keras.utils.set_random_seed(training.SEED)
    rng = np.random.default_rng(774)
    features = rng.normal(size=(6, 298, 40, 1)).astype(np.float32)
    targets = rng.normal(size=(6, 192)).astype(np.float32)
    targets /= np.linalg.norm(targets, axis=1, keepdims=True)
    model = training.build_student_model()
    fitted = training._fit_student(tf, model, features, targets,
                                  np.arange(4), np.arange(4, 6), tmp_path, 2)
    assert fitted["weight_update_max_abs"] > 0
    assert len(fitted["history"]) == 2 and fitted["best_epoch"] in (1, 2)
    assert (tmp_path / "student_best.weights.h5").is_file()
    fixed = training.build_student_model(batch_size=1)
    fixed.set_weights(model.get_weights())
    np.testing.assert_allclose(model(features[:1]).numpy(), fixed(features[:1]).numpy(), atol=1e-6)
    content = training._convert_int8(tf, fixed, features, np.arange(4))
    raw, unit, meta = training._run_tflite(content, features, require_int8=True)
    assert raw.shape == unit.shape == (6, 192)
    assert meta["input"]["dtype"] == meta["output"]["dtype"] == "int8"
    assert meta["input"]["shape"] == [1, 298, 40, 1]
    np.testing.assert_allclose(np.linalg.norm(unit, axis=1), 1., atol=1e-6)


def test_cosine_loss_independent_arithmetic() -> None:
    tf = pytest.importorskip("tensorflow")
    outputs = tf.constant([[2., 0.], [0., 3.]])
    targets = tf.constant([[1., 0.], [1., 0.]])
    assert float(training._cosine_loss(tf, outputs, targets)) == pytest.approx(.5)
