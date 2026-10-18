from __future__ import annotations

import hashlib
import json
import sys
from contextlib import nullcontext
from types import SimpleNamespace
import wave
from pathlib import Path

import numpy as np
import pytest

from edgeai import teacher


def _write_pcm_wav(path: Path, samples: np.ndarray, *, sample_rate: int = 16_000) -> None:
    pcm = np.asarray(samples, dtype=np.float32)
    encoded = np.clip(np.rint(pcm * 32767.0), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(encoded.tobytes())


def _manifest_fixture(tmp_path: Path, *, rows: int = 2) -> Path:
    wav_dir = tmp_path / "wav"
    wav_dir.mkdir()
    manifest_rows = []
    for index in range(rows):
        wav_path = wav_dir / f"u{index}.wav"
        samples = np.zeros(teacher.NUM_SAMPLES, dtype=np.float32)
        samples[index : index + 4] = 0.1 * (index + 1)
        _write_pcm_wav(wav_path, samples)
        wav_hash = hashlib.sha256(wav_path.read_bytes()).hexdigest()
        manifest_rows.append(
            {
                "utterance_id": f"u{index}",
                "speaker_id": f"s{index}",
                "chapter_id": f"c{index}",
                "split": "train",
                "wav_path": f"wav/u{index}.wav",
                "wav_sha256": wav_hash,
                "source_sha256": "a" * 64,
                "input_samples": teacher.NUM_SAMPLES,
                "padded_samples": 0,
                "truncated_samples": 0,
            }
        )
    manifest = {
        "format_version": 1,
        "dataset": teacher.MANIFEST_DATASET,
        "seed": teacher.MANIFEST_SEED,
        "rows": manifest_rows,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_strict_wav_reader_accepts_exact_pcm_and_rejects_wrong_length(tmp_path: Path) -> None:
    path = tmp_path / "exact.wav"
    _write_pcm_wav(path, np.zeros(teacher.NUM_SAMPLES, dtype=np.float32))
    decoded = teacher.read_pcm16_mono16k(path)
    assert decoded.shape == (teacher.NUM_SAMPLES,)
    assert decoded.dtype == np.float32
    assert np.all(decoded == 0)

    short_path = tmp_path / "short.wav"
    _write_pcm_wav(short_path, np.zeros(teacher.NUM_SAMPLES - 1, dtype=np.float32))
    with pytest.raises(teacher.TeacherContractError, match="exactly"):
        teacher.read_pcm16_mono16k(short_path)


def test_manifest_fixture_is_self_contained_and_hash_bound(tmp_path: Path) -> None:
    manifest_path = _manifest_fixture(tmp_path)
    loaded = teacher.load_manifest(manifest_path)
    assert loaded.sha256 == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert [row["utterance_id"] for row in loaded.rows] == ["u0", "u1"]
    assert all(Path(row["wav_path"]).is_absolute() for row in loaded.rows)

    wav_path = tmp_path / "wav/u0.wav"
    wav_path.write_bytes(wav_path.read_bytes() + b"changed")
    with pytest.raises(teacher.TeacherContractError, match="wav_sha256 mismatch"):
        teacher.load_manifest(manifest_path)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda data: data["rows"].__setitem__(0, {**data["rows"][0], "wav_path": "../escape.wav"}), "manifest directory"),
        (lambda data: data["rows"].__setitem__(0, {**data["rows"][0], "source_sha256": "bad"}), "64-character"),
        (lambda data: data["rows"].__setitem__(0, {**data["rows"][0], "input_samples": 50_000, "truncated_samples": 0}), "padding/truncation"),
        (lambda data: data["rows"].__setitem__(0, {**data["rows"][0], "split": "test"}), "unsupported split"),
    ],
)
def test_manifest_negative_conditions_fail_closed(
    tmp_path: Path, mutator, message: str
) -> None:
    manifest_path = _manifest_fixture(tmp_path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutator(data)
    manifest_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(teacher.TeacherContractError, match=message):
        teacher.load_manifest(manifest_path)


def test_invalid_manifest_cannot_reach_model_load(tmp_path: Path, monkeypatch) -> None:
    manifest_path = _manifest_fixture(tmp_path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["rows"][0]["wav_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(data), encoding="utf-8")
    called = False

    def unexpected_load(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("model load must not run for an invalid manifest")

    monkeypatch.setattr(teacher, "load_teacher", unexpected_load)
    with pytest.raises(teacher.TeacherContractError, match="wav_sha256 mismatch"):
        teacher.extract_manifest(
            manifest_path,
            tmp_path / "missing-model",
            tmp_path / "out" / "embeddings.npz",
        )
    assert called is False


def test_snapshot_verification_rejects_missing_or_tampered_files(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    with pytest.raises(teacher.TeacherContractError, match="required model file"):
        teacher.verify_model_snapshot(model_dir)

    for filename in teacher.REQUIRED_MODEL_FILES:
        (model_dir / filename).write_bytes(b"not-a-model")
    with pytest.raises(teacher.TeacherContractError, match="hash mismatch"):
        teacher.verify_model_snapshot(model_dir)


def test_pcm_batch_validation_and_normalized_shape_without_runtime() -> None:
    with pytest.raises(teacher.TeacherContractError, match="shape"):
        teacher._validate_pcm_batch(np.zeros((1, teacher.NUM_SAMPLES - 1), dtype=np.float32))
    bad = np.zeros((1, teacher.NUM_SAMPLES), dtype=np.float32)
    bad[0, 0] = np.nan
    with pytest.raises(teacher.TeacherContractError, match="finite"):
        teacher._validate_pcm_batch(bad)


def test_embedding_artifact_schema_is_manifest_bound(tmp_path: Path) -> None:
    manifest = teacher.load_manifest(_manifest_fixture(tmp_path))
    embeddings = np.zeros((len(manifest.rows), teacher.EMBEDDING_DIM), dtype=np.float32)
    embeddings[:, 0] = 1.0
    output = tmp_path / "teacher_embeddings.npz"
    np.savez(
        output,
        ids=np.asarray([row["utterance_id"] for row in manifest.rows], dtype=np.str_),
        embeddings=embeddings,
        manifest_sha256=np.asarray(manifest.sha256, dtype=np.str_),
    )
    checked = teacher._validate_embedding_artifact(output, manifest)
    assert checked["embeddings_shape"] == [len(manifest.rows), teacher.EMBEDDING_DIM]

    np.savez(
        output,
        ids=np.asarray([row["utterance_id"] for row in manifest.rows], dtype=np.str_),
        embeddings=embeddings,
        manifest_sha256=np.asarray("0" * 64, dtype=np.str_),
    )
    with pytest.raises(teacher.TeacherContractError, match="manifest_sha256"):
        teacher._validate_embedding_artifact(output, manifest)


class _FakeTensor:
    dtype = "float32"

    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape

    def detach(self):
        raise AssertionError("negative batch-shape test must fail before normalization")


class _FakeTorch:
    float32 = "float32"

    def __init__(self, feature_shape: tuple[int, ...], raw_shape: tuple[int, ...]) -> None:
        self.feature_shape = feature_shape
        self.raw_shape = raw_shape

    def from_numpy(self, value):
        return _FakeTensor(tuple(value.shape))

    def inference_mode(self):
        return nullcontext()

    def is_floating_point(self, value):
        return True


@pytest.mark.parametrize(
    ("feature_shape", "raw_shape", "message"),
    [
        ((1, 301, 80), (2, 1, teacher.EMBEDDING_DIM), "native feature batch count"),
        ((2, 301, 80), (1, 1, teacher.EMBEDDING_DIM), "raw output batch count"),
    ],
)
def test_extract_batch_rejects_backend_batch_mismatch(
    monkeypatch,
    feature_shape: tuple[int, ...],
    raw_shape: tuple[int, ...],
    message: str,
) -> None:
    fake_torch = _FakeTorch(feature_shape, raw_shape)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    classifier = SimpleNamespace(
        mods=SimpleNamespace(
            compute_features=lambda _wavs: _FakeTensor(fake_torch.feature_shape)
        ),
        encode_batch=lambda _wavs, wav_lens=None, normalize=False: _FakeTensor(
            fake_torch.raw_shape
        ),
    )
    runtime = SimpleNamespace(classifier=classifier)
    pcm = np.zeros((2, teacher.NUM_SAMPLES), dtype=np.float32)
    with pytest.raises(teacher.TeacherContractError, match=message):
        teacher.extract_batch(runtime, pcm)


def test_failure_report_preserves_existing_success_report(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "teacher_embeddings.npz"
    output.write_bytes(b"immutable-output-placeholder")
    prior_report = teacher._report_path(output)
    prior_report.parent.mkdir(parents=True)
    prior_bytes = b'{"status":"TEACHER_EXTRACTION_PASS"}\n'
    prior_report.write_bytes(prior_bytes)

    def fail_without_runtime(*_args, **_kwargs):
        raise teacher.TeacherContractError("refusing to overwrite existing output")

    monkeypatch.setattr(teacher, "extract_manifest", fail_without_runtime)
    result = teacher.main(
        [
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--model-dir",
            str(tmp_path / "model"),
            "--output",
            str(output),
        ]
    )
    assert result == 2
    assert prior_report.read_bytes() == prior_bytes
    failure_reports = list(prior_report.parent.glob("*.failure-*.json"))
    assert len(failure_reports) == 1
    failure = json.loads(failure_reports[0].read_text(encoding="utf-8"))
    assert failure["status"] == "TEACHER_EXTRACTION_FAIL"
    assert failure["preserved_report_path"] == str(prior_report)


def test_real_pinned_model_official_wav_smoke(tmp_path: Path) -> None:
    model_dir = Path("artifacts/teacher/model") / teacher.MODEL_REVISION
    example = model_dir / "example1.wav"
    if not model_dir.is_dir() or not example.is_file():
        pytest.skip("pinned teacher snapshot is not present")
    torch = pytest.importorskip("torch")
    torchaudio = pytest.importorskip("torchaudio")
    runtime = teacher.load_teacher(model_dir, cache_dir=tmp_path / "teacher-cache")
    waveform, sample_rate = torchaudio.load(str(example))
    assert sample_rate == teacher.SAMPLE_RATE
    pcm = waveform[:, : teacher.NUM_SAMPLES].numpy()
    observation = teacher.extract_batch(runtime, pcm)
    assert observation.feature_shape[0] == 1
    assert observation.feature_shape[-1] == 80
    assert observation.raw_shape == (1, 1, teacher.EMBEDDING_DIM)
    assert observation.embeddings.shape == (1, teacher.EMBEDDING_DIM)
    assert observation.embeddings.dtype == np.float32
    assert np.all(np.isfinite(observation.embeddings))
    assert np.isclose(np.linalg.norm(observation.embeddings[0]), 1.0, atol=3e-6)
    assert torch.get_num_threads() == 4
