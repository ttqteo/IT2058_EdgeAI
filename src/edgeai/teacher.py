"""Strict, host-only SpeechBrain ECAPA teacher extraction.

This module deliberately keeps the teacher boundary separate from the student
frontend.  It accepts only the prepared PCM16/mono/16 kHz/48,000-sample WAV
contract, loads one pinned local SpeechBrain snapshot offline, and writes the
manifest-bound embedding artifact described by ``docs/pilot-contract.md``.

The model and dependency files are evidence from a local pinned snapshot, not
a cryptographic attestation of the upstream repository.  This is a pilot
utility; it does not establish a firmware, board, or accuracy contract.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata as importlib_metadata
import json
import os
import platform
import sys
import time
import wave
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


MODEL_REPO = "speechbrain/spkrec-ecapa-voxceleb"
MODEL_REVISION = "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"
SAMPLE_RATE = 16_000
NUM_SAMPLES = 48_000
EMBEDDING_DIM = 192
MANIFEST_FORMAT_VERSION = 1
MANIFEST_DATASET = "librispeech-dev-clean-pilot"
MANIFEST_SEED = 2_026_1012
ALLOWED_SPLITS = frozenset(
    {
        "train",
        "validation",
        "calibration_enroll",
        "calibration_query",
        "test_enroll",
        "test_query",
    }
)

# These seven hashes are the exact local snapshot used for the desktop smoke
# and pilot.  The two large checkpoint hashes are independently published by
# the model card.  The remaining hashes bind this run to the observed files at
# the same immutable revision; they are not claimed as signed upstream data.
MODEL_FILE_SHA256: dict[str, str] = {
    "config.json": "15ff7afad09adec0936c31493a7397c13f855914f4579a8393fe82649778a664",
    "hyperparams.yaml": "6f78854fa04ba59e761437b76a2575d3aba5e5016de3e9b69f0c9a5077fb1a41",
    "embedding_model.ckpt": "0575cb64845e6b9a10db9bcb74d5ac32b326b8dc90352671d345e2ee3d0126a2",
    "classifier.ckpt": "fd9e3634fe68bd0a427c95e354c0c677374f62b3f434e45b78599950d860d535",
    "mean_var_norm_emb.ckpt": "cd70225b05b37be64fc5a95e24395d804231d43f74b2e1e5a513db7b69b34c33",
    "label_encoder.txt": "e13c3a167bb4112685670ee896d20e2b565af16b3a4ceeaa8689fa4d22adb8b9",
    "example1.wav": "bf2dde5cb516939ff619d62fc07d4f4bec5b5d521aee3d07ae51828c9d93be0b",
}
MODEL_FILE_HASH_SOURCE: dict[str, str] = {
    "embedding_model.ckpt": "published_model_card_sha256",
    "classifier.ckpt": "published_model_card_sha256",
    "config.json": "observed_local_download_at_pinned_revision",
    "hyperparams.yaml": "observed_local_download_at_pinned_revision",
    "mean_var_norm_emb.ckpt": "observed_local_download_at_pinned_revision",
    "label_encoder.txt": "observed_local_download_at_pinned_revision",
    "example1.wav": "observed_local_download_at_pinned_revision",
}
REQUIRED_MODEL_FILES = tuple(
    name for name in MODEL_FILE_SHA256 if name != "example1.wav"
)

_MANIFEST_FIELDS = frozenset({"format_version", "dataset", "seed", "rows"})
_ROW_FIELDS = frozenset(
    {
        "utterance_id",
        "speaker_id",
        "chapter_id",
        "split",
        "wav_path",
        "wav_sha256",
        "source_sha256",
        "input_samples",
        "padded_samples",
        "truncated_samples",
    }
)


class TeacherContractError(ValueError):
    """Raised when a teacher input, snapshot, or runtime violates the contract."""


@dataclass(frozen=True)
class ManifestData:
    path: Path
    sha256: str
    rows: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class ModelSnapshot:
    path: Path
    files: dict[str, dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "repo": MODEL_REPO,
            "revision": MODEL_REVISION,
            "path": str(self.path),
            "revision_is_immutable_commit": True,
            "provenance_status": (
                "PINNED_SNAPSHOT_HASH_MATCH_NO_CRYPTO_AUTHENTICATION"
            ),
            "files": self.files,
        }


@dataclass(frozen=True)
class LoadedTeacher:
    classifier: Any
    snapshot: ModelSnapshot
    cache_dir: Path
    threads: int
    interop_threads: int
    load_ms: float


@dataclass(frozen=True)
class BatchObservation:
    embeddings: np.ndarray
    feature_shape: tuple[int, ...]
    feature_dtype: str
    raw_shape: tuple[int, ...]
    raw_dtype: str
    inference_ms: float


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise TeacherContractError(f"{field} must be a 64-character SHA256 hex string")
    lowered = value.lower()
    try:
        int(lowered, 16)
    except ValueError as exc:
        raise TeacherContractError(f"{field} must be a SHA256 hex string") from exc
    return lowered


def _require_nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise TeacherContractError(f"{field} must be a non-empty string")
    return value


def _require_integer(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TeacherContractError(f"{field} must be an integer >= {minimum}")
    return value


def read_pcm16_mono16k(path: str | Path) -> np.ndarray:
    """Read one exact PCM16 mono 16 kHz, 48,000-sample WAV as float32."""

    wav_path = Path(path)
    try:
        with wave.open(str(wav_path), "rb") as handle:
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            sample_rate = handle.getframerate()
            compression = handle.getcomptype()
            frame_count = handle.getnframes()
            if channels != 1:
                raise TeacherContractError(
                    f"{wav_path}: expected mono WAV, got {channels} channels"
                )
            if sample_width != 2:
                raise TeacherContractError(
                    f"{wav_path}: expected 16-bit PCM, got {sample_width * 8}-bit"
                )
            if sample_rate != SAMPLE_RATE:
                raise TeacherContractError(
                    f"{wav_path}: expected {SAMPLE_RATE} Hz, got {sample_rate} Hz"
                )
            if compression != "NONE":
                raise TeacherContractError(
                    f"{wav_path}: compressed WAV is not allowed ({compression})"
                )
            if frame_count != NUM_SAMPLES:
                raise TeacherContractError(
                    f"{wav_path}: expected exactly {NUM_SAMPLES} frames, "
                    f"got {frame_count}"
                )
            raw = handle.readframes(NUM_SAMPLES)
    except TeacherContractError:
        raise
    except (OSError, EOFError, wave.Error) as exc:
        raise TeacherContractError(f"cannot read strict WAV {wav_path}: {exc}") from exc

    expected_bytes = NUM_SAMPLES * 2
    if len(raw) != expected_bytes:
        raise TeacherContractError(
            f"{wav_path}: expected {expected_bytes} PCM bytes, got {len(raw)}"
        )
    pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / np.float32(32768.0)
    if pcm.shape != (NUM_SAMPLES,) or not np.all(np.isfinite(pcm)):
        raise TeacherContractError(f"{wav_path}: decoded PCM is not finite mono data")
    if np.any(pcm < -1.0) or np.any(pcm > 1.0):
        raise TeacherContractError(f"{wav_path}: decoded PCM is outside [-1, 1]")
    return pcm


def _resolve_manifest_wav(manifest_path: Path, wav_path_value: object) -> Path:
    if not isinstance(wav_path_value, str) or not wav_path_value:
        raise TeacherContractError("wav_path must be a non-empty relative path")
    relative = Path(wav_path_value)
    if relative.is_absolute() or ".." in relative.parts:
        raise TeacherContractError("wav_path must stay below the manifest directory")
    root = manifest_path.parent.resolve()
    try:
        resolved = (root / relative).resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise TeacherContractError(
            f"wav_path does not resolve inside manifest directory: {wav_path_value}"
        ) from exc
    if not resolved.is_file():
        raise TeacherContractError(f"wav_path is not a regular file: {resolved}")
    return resolved


def _validate_manifest_row(
    manifest_path: Path,
    raw_row: object,
    seen_ids: set[str],
    seen_wavs: set[Path],
) -> dict[str, Any]:
    if not isinstance(raw_row, dict) or set(raw_row) != _ROW_FIELDS:
        raise TeacherContractError(
            "each manifest row must contain exactly "
            f"{sorted(_ROW_FIELDS)}"
        )
    row = dict(raw_row)
    utterance_id = _require_nonempty_string(row["utterance_id"], "utterance_id")
    if utterance_id in seen_ids:
        raise TeacherContractError(f"duplicate utterance_id: {utterance_id}")
    seen_ids.add(utterance_id)
    for field in ("speaker_id", "chapter_id"):
        _require_nonempty_string(row[field], field)
    split = _require_nonempty_string(row["split"], "split")
    if split not in ALLOWED_SPLITS:
        raise TeacherContractError(f"unsupported split: {split}")

    wav_path = _resolve_manifest_wav(manifest_path, row["wav_path"])
    if wav_path in seen_wavs:
        raise TeacherContractError(f"duplicate wav_path: {row['wav_path']}")
    seen_wavs.add(wav_path)
    expected_wav_hash = _require_sha256(row["wav_sha256"], "wav_sha256")
    source_hash = _require_sha256(row["source_sha256"], "source_sha256")
    input_samples = _require_integer(row["input_samples"], "input_samples")
    padded_samples = _require_integer(row["padded_samples"], "padded_samples")
    truncated_samples = _require_integer(row["truncated_samples"], "truncated_samples")
    expected_padded = max(NUM_SAMPLES - input_samples, 0)
    expected_truncated = max(input_samples - NUM_SAMPLES, 0)
    if (padded_samples, truncated_samples) != (expected_padded, expected_truncated):
        raise TeacherContractError(
            f"{utterance_id}: padding/truncation counts do not match input_samples"
        )

    observed_hash, _ = _sha256_file(wav_path)
    if observed_hash != expected_wav_hash:
        raise TeacherContractError(
            f"{utterance_id}: wav_sha256 mismatch; expected {expected_wav_hash}, "
            f"observed {observed_hash}"
        )
    # Strict decoding is intentionally part of manifest validation, before any
    # model import or checkpoint load.
    read_pcm16_mono16k(wav_path)
    row["wav_path"] = str(wav_path)
    row["wav_sha256"] = expected_wav_hash
    row["source_sha256"] = source_hash
    return row


def load_manifest(path: str | Path) -> ManifestData:
    """Validate the full manifest and all referenced WAVs before model load."""

    manifest_path = Path(path).expanduser().resolve()
    try:
        raw_bytes = manifest_path.read_bytes()
        raw = json.loads(raw_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TeacherContractError(f"cannot read manifest {manifest_path}: {exc}") from exc
    if not isinstance(raw, dict) or set(raw) != _MANIFEST_FIELDS:
        raise TeacherContractError(
            "manifest must contain exactly " f"{sorted(_MANIFEST_FIELDS)}"
        )
    if raw["format_version"] != MANIFEST_FORMAT_VERSION:
        raise TeacherContractError(
            f"unsupported manifest format_version: {raw['format_version']}"
        )
    if raw["dataset"] != MANIFEST_DATASET:
        raise TeacherContractError(f"unexpected dataset: {raw['dataset']!r}")
    if raw["seed"] != MANIFEST_SEED:
        raise TeacherContractError(f"unexpected manifest seed: {raw['seed']!r}")
    if not isinstance(raw["rows"], list) or not raw["rows"]:
        raise TeacherContractError("manifest rows must be a non-empty list")

    seen_ids: set[str] = set()
    seen_wavs: set[Path] = set()
    rows = tuple(
        _validate_manifest_row(manifest_path, row, seen_ids, seen_wavs)
        for row in raw["rows"]
    )
    return ManifestData(
        path=manifest_path,
        sha256=_sha256_bytes(raw_bytes),
        rows=rows,
    )


def verify_model_snapshot(path: str | Path) -> ModelSnapshot:
    """Verify every required pinned snapshot file without loading a checkpoint."""

    model_dir = Path(path).expanduser().resolve()
    if not model_dir.is_dir():
        raise TeacherContractError(f"model directory does not exist: {model_dir}")
    files: dict[str, dict[str, Any]] = {}
    for filename, expected in MODEL_FILE_SHA256.items():
        file_path = model_dir / filename
        if filename != "example1.wav" and not file_path.is_file():
            raise TeacherContractError(f"required model file is missing: {file_path}")
        if filename == "example1.wav" and not file_path.exists():
            continue
        try:
            observed, size = _sha256_file(file_path)
        except OSError as exc:
            raise TeacherContractError(f"cannot hash model file {file_path}: {exc}") from exc
        if observed != expected:
            raise TeacherContractError(
                f"model file hash mismatch for {filename}: expected {expected}, "
                f"observed {observed}"
            )
        files[filename] = {
            "path": str(file_path),
            "bytes": size,
            "sha256": observed,
            "expected_sha256": expected,
            "hash_source": MODEL_FILE_HASH_SOURCE[filename],
            "verified": True,
        }

    config_path = model_dir / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TeacherContractError(f"invalid model config.json: {exc}") from exc
    if not isinstance(config, dict) or config.get("speechbrain_interface") != "SpeakerRecognition":
        raise TeacherContractError(
            "model config does not identify SpeechBrain SpeakerRecognition"
        )
    return ModelSnapshot(path=model_dir, files=files)


def _set_torch_threads(torch: Any, threads: int) -> tuple[int, int]:
    if isinstance(threads, bool) or not isinstance(threads, int) or not 1 <= threads <= 64:
        raise TeacherContractError("threads must be an integer in [1, 64]")
    try:
        torch.set_num_threads(threads)
        current_interop = torch.get_num_interop_threads()
        if current_interop != threads:
            try:
                torch.set_num_interop_threads(threads)
            except RuntimeError as exc:
                raise TeacherContractError(
                    "cannot establish requested bounded torch interop thread count"
                ) from exc
        return int(torch.get_num_threads()), int(torch.get_num_interop_threads())
    except (AttributeError, RuntimeError) as exc:
        raise TeacherContractError(f"cannot configure torch CPU threads: {exc}") from exc


def load_teacher(
    model_dir: str | Path,
    *,
    cache_dir: str | Path | None = None,
    threads: int = 4,
) -> LoadedTeacher:
    """Load the pinned local model with SpeechBrain's offline API only."""

    snapshot = verify_model_snapshot(model_dir)
    try:
        import torch
    except (ImportError, OSError, RuntimeError) as exc:
        raise TeacherContractError(
            "teacher runtime is unavailable; install the isolated teacher requirements"
        ) from exc

    configured_threads, configured_interop = _set_torch_threads(torch, threads)
    if cache_dir is None:
        cache_path = snapshot.path.parent.parent / "cache" / "runtime"
    else:
        cache_path = Path(cache_dir).expanduser().resolve()
    try:
        cache_path.mkdir(parents=True, exist_ok=True)
        from speechbrain.inference.speaker import EncoderClassifier
        from speechbrain.utils.fetching import FetchConfig, LocalStrategy
    except (ImportError, OSError, RuntimeError) as exc:
        raise TeacherContractError(
            "SpeechBrain 1.1.1 offline runtime is unavailable"
        ) from exc

    old_offline = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    started = time.perf_counter()
    try:
        classifier = EncoderClassifier.from_hparams(
            source=str(snapshot.path),
            savedir=str(cache_path),
            overrides={"pretrained_path": str(snapshot.path)},
            run_opts={"device": "cpu"},
            local_strategy=LocalStrategy.COPY,
            fetch_config=FetchConfig(
                revision=MODEL_REVISION,
                allow_network=False,
                allow_updates=False,
            ),
        )
    except Exception as exc:
        raise TeacherContractError(
            "pinned SpeechBrain model failed to load offline; no fallback was used"
        ) from exc
    finally:
        if old_offline is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = old_offline
    try:
        classifier.eval()
    except Exception as exc:
        raise TeacherContractError("loaded teacher does not support eval mode") from exc
    return LoadedTeacher(
        classifier=classifier,
        snapshot=snapshot,
        cache_dir=cache_path,
        threads=configured_threads,
        interop_threads=configured_interop,
        load_ms=(time.perf_counter() - started) * 1000.0,
    )


def _validate_pcm_batch(pcm_batch: object) -> np.ndarray:
    pcm = np.asarray(pcm_batch)
    if pcm.ndim != 2 or pcm.shape[1] != NUM_SAMPLES or pcm.shape[0] == 0:
        raise TeacherContractError(
            f"PCM batch must have shape [B, {NUM_SAMPLES}], got {pcm.shape}"
        )
    if np.iscomplexobj(pcm):
        raise TeacherContractError("PCM batch must be real-valued")
    try:
        pcm = np.asarray(pcm, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise TeacherContractError("PCM batch must contain numeric values") from exc
    if not np.all(np.isfinite(pcm)) or np.any(pcm < -1.0) or np.any(pcm > 1.0):
        raise TeacherContractError("PCM batch must be finite and bounded in [-1, 1]")
    return np.ascontiguousarray(pcm)


def extract_batch(runtime: LoadedTeacher, pcm_batch: object) -> BatchObservation:
    """Invoke native SpeechBrain features and return normalized [B,192] output."""

    pcm = _validate_pcm_batch(pcm_batch)
    try:
        import torch
    except (ImportError, OSError) as exc:
        raise TeacherContractError("torch is required for teacher inference") from exc

    wavs = torch.from_numpy(pcm)
    try:
        with torch.inference_mode():
            native_features = runtime.classifier.mods.compute_features(wavs)
            started = time.perf_counter()
            raw = runtime.classifier.encode_batch(
                wavs, wav_lens=None, normalize=False
            )
            inference_ms = (time.perf_counter() - started) * 1000.0
    except (RuntimeError, ValueError, TypeError, AttributeError) as exc:
        raise TeacherContractError("SpeechBrain teacher invocation failed") from exc

    try:
        feature_shape = tuple(int(value) for value in native_features.shape)
        raw_shape = tuple(int(value) for value in raw.shape)
        feature_dtype = str(native_features.dtype).replace("torch.", "")
        raw_dtype = str(raw.dtype).replace("torch.", "")
    except AttributeError as exc:
        raise TeacherContractError("teacher returned an invalid tensor object") from exc
    batch_count = int(pcm.shape[0])
    if not feature_shape or feature_shape[0] != batch_count:
        raise TeacherContractError(
            "native feature batch count mismatch: "
            f"expected {batch_count}, got shape {feature_shape}"
        )
    if not raw_shape or raw_shape[0] != batch_count:
        raise TeacherContractError(
            "raw output batch count mismatch: "
            f"expected {batch_count}, got shape {raw_shape}"
        )
    if len(raw_shape) != 3 or raw_shape[1:] != (1, EMBEDDING_DIM):
        raise TeacherContractError(
            f"unexpected raw teacher output shape: {raw_shape}; expected [B,1,192]"
        )
    if not torch.is_floating_point(raw) or raw.dtype != torch.float32:
        raise TeacherContractError(
            f"unexpected raw teacher dtype: {raw_dtype}; expected float32"
        )
    if not bool(torch.isfinite(raw).all().item()):
        raise TeacherContractError("teacher returned non-finite raw embeddings")
    raw_np = raw.detach().cpu().numpy()
    squeezed = np.asarray(raw_np[:, 0, :], dtype=np.float64)
    norms = np.linalg.norm(squeezed, axis=1)
    if not np.all(np.isfinite(norms)) or np.any(norms <= 0.0):
        raise TeacherContractError("teacher returned zero/non-finite embedding norm")
    embeddings = (squeezed / norms[:, None]).astype(np.float32)
    if not np.all(np.isfinite(embeddings)):
        raise TeacherContractError("normalized teacher embeddings are not finite")
    return BatchObservation(
        embeddings=embeddings,
        feature_shape=feature_shape,
        feature_dtype=feature_dtype,
        raw_shape=raw_shape,
        raw_dtype=raw_dtype,
        inference_ms=float(inference_ms),
    )


def embed_waveform(runtime: LoadedTeacher, pcm: object) -> np.ndarray:
    """Embed one exact waveform, returning the public float32 [192] vector."""

    vector = np.asarray(pcm)
    if vector.ndim != 1:
        raise TeacherContractError(f"waveform must be one-dimensional, got {vector.shape}")
    return extract_batch(runtime, vector.reshape(1, -1)).embeddings[0]


def _read_manifest_pcm(row: Mapping[str, Any]) -> np.ndarray:
    path = Path(str(row["wav_path"]))
    observed_hash, _ = _sha256_file(path)
    if observed_hash != row["wav_sha256"]:
        raise TeacherContractError(
            f"{row['utterance_id']}: WAV changed after manifest validation"
        )
    return read_pcm16_mono16k(path)


def _dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for module_name, distribution_name in (
        ("numpy", "numpy"),
        ("torch", "torch"),
        ("torchaudio", "torchaudio"),
        ("speechbrain", "speechbrain"),
        ("huggingface_hub", "huggingface-hub"),
        ("soundfile", "soundfile"),
        ("scipy", "scipy"),
        ("HyperPyYAML", "HyperPyYAML"),
    ):
        try:
            versions[module_name] = str(importlib_metadata.version(distribution_name))
        except importlib_metadata.PackageNotFoundError:
            versions[module_name] = "unavailable"
    return versions


def _requirements_hash() -> str | None:
    requirements_path = Path(__file__).resolve().parents[2] / "requirements-teacher.txt"
    if not requirements_path.is_file():
        return None
    return _sha256_file(requirements_path)[0]


def _peak_rss_bytes() -> int | None:
    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform == "darwin":
            return value
        return value * 1024
    except (ImportError, OSError, ValueError):
        return None


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _report_path(output_path: Path) -> Path:
    return output_path.parent / "reports" / f"{output_path.stem}.report.json"


def _manifest_report(manifest: ManifestData) -> dict[str, Any]:
    return {
        "path": str(manifest.path),
        "sha256": manifest.sha256,
        "dataset": MANIFEST_DATASET,
        "seed": MANIFEST_SEED,
        "row_count": len(manifest.rows),
        "split_counts": dict(Counter(str(row["split"]) for row in manifest.rows)),
        "wav_hashes_verified": True,
        "strict_pcm_contract_verified": True,
        "source_hashes_present": True,
        "source_hashes_revalidated": False,
        "source_hash_note": (
            "source_sha256 values are preserved from the prepared manifest; the "
            "source archive is not present at this extraction boundary"
        ),
        "inputs": [
            {
                "utterance_id": row["utterance_id"],
                "speaker_id": row["speaker_id"],
                "split": row["split"],
                "wav_sha256": row["wav_sha256"],
                "source_sha256": row["source_sha256"],
            }
            for row in manifest.rows
        ],
    }


def _validate_embedding_artifact(path: Path, manifest: ManifestData) -> dict[str, Any]:
    try:
        with np.load(path, allow_pickle=False) as artifact:
            if set(artifact.files) != {"ids", "embeddings", "manifest_sha256"}:
                raise TeacherContractError("teacher artifact has an unexpected key set")
            ids = artifact["ids"]
            embeddings = artifact["embeddings"]
            manifest_hash = artifact["manifest_sha256"]
    except (OSError, ValueError, EOFError) as exc:
        raise TeacherContractError(f"cannot read written teacher artifact: {exc}") from exc
    if ids.dtype.kind != "U" or ids.shape != (len(manifest.rows),):
        raise TeacherContractError("teacher artifact ids schema mismatch")
    expected_ids = np.asarray([str(row["utterance_id"]) for row in manifest.rows], dtype=np.str_)
    if not np.array_equal(ids, expected_ids):
        raise TeacherContractError("teacher artifact ids do not match manifest order")
    if embeddings.dtype != np.float32 or embeddings.shape != (len(manifest.rows), EMBEDDING_DIM):
        raise TeacherContractError("teacher artifact embeddings schema mismatch")
    if not np.all(np.isfinite(embeddings)):
        raise TeacherContractError("teacher artifact contains non-finite embeddings")
    norms = np.linalg.norm(embeddings.astype(np.float64), axis=1)
    if not np.all(np.isfinite(norms)) or np.any(norms <= 0.0):
        raise TeacherContractError("teacher artifact contains zero/non-finite norms")
    if not np.allclose(norms, 1.0, rtol=0.0, atol=3e-6):
        raise TeacherContractError("teacher artifact embeddings are not L2-normalized")
    if np.asarray(manifest_hash).shape != () or str(np.asarray(manifest_hash).item()) != manifest.sha256:
        raise TeacherContractError("teacher artifact manifest_sha256 does not match input")
    observed, size = _sha256_file(path)
    return {
        "path": str(path),
        "bytes": size,
        "sha256": observed,
        "ids_shape": list(ids.shape),
        "embeddings_shape": list(embeddings.shape),
        "embeddings_dtype": str(embeddings.dtype),
        "manifest_sha256": manifest.sha256,
        "verified": True,
    }


def extract_manifest(
    manifest_path: str | Path,
    model_dir: str | Path,
    output_path: str | Path,
    *,
    batch_size: int = 4,
    threads: int = 4,
) -> dict[str, Any]:
    """Extract all manifest rows and write one checked teacher NPZ/report."""

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 1 <= batch_size <= 64:
        raise TeacherContractError("batch_size must be an integer in [1, 64]")
    # This is deliberately first: malformed/mutated inputs cannot trigger a
    # checkpoint load or a partial desktop result.
    manifest = load_manifest(manifest_path)
    snapshot = verify_model_snapshot(model_dir)
    output = Path(output_path).expanduser().resolve()
    if output.exists():
        raise TeacherContractError(f"refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    runtime = load_teacher(
        snapshot.path,
        cache_dir=output.parent.parent / "cache" / "runtime-pilot",
        threads=threads,
    )

    all_embeddings: list[np.ndarray] = []
    first_observation: BatchObservation | None = None
    batch_times: list[float] = []
    total_started = time.perf_counter()
    try:
        for start in range(0, len(manifest.rows), batch_size):
            rows = manifest.rows[start : start + batch_size]
            pcm_batch = np.stack([_read_manifest_pcm(row) for row in rows], axis=0)
            batch_started = time.perf_counter()
            observation = extract_batch(runtime, pcm_batch)
            batch_times.append((time.perf_counter() - batch_started) * 1000.0)
            if first_observation is None:
                first_observation = observation
            all_embeddings.append(observation.embeddings)
    except (OSError, RuntimeError, ValueError, TeacherContractError) as exc:
        raise TeacherContractError(f"pilot extraction failed: {exc}") from exc
    if first_observation is None:
        raise TeacherContractError("pilot manifest produced no batches")

    embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32, copy=False)
    ids = np.asarray([str(row["utterance_id"]) for row in manifest.rows], dtype=np.str_)
    if embeddings.shape != (len(ids), EMBEDDING_DIM):
        raise TeacherContractError("internal embedding count/shape mismatch")
    # Re-read the manifest bytes immediately before writing so a path mutation
    # cannot silently bind a different JSON to the output artifact.
    current_manifest_bytes = manifest.path.read_bytes()
    current_manifest_hash = _sha256_bytes(current_manifest_bytes)
    if current_manifest_hash != manifest.sha256:
        raise TeacherContractError("manifest changed during extraction")
    np.savez(
        output,
        ids=ids,
        embeddings=embeddings,
        manifest_sha256=np.asarray(manifest.sha256, dtype=np.str_),
    )
    artifact = _validate_embedding_artifact(output, manifest)
    total_ms = (time.perf_counter() - total_started) * 1000.0
    report = {
        "status": "TEACHER_EXTRACTION_PASS",
        "scope": "EXPERIMENTAL_HOST_ONLY",
        "runtime_status": "DESKTOP_CPU_VERIFIED",
        "hardware_status": "UNVERIFIED_HARDWARE",
        "model": snapshot.as_dict(),
        "manifest": _manifest_report(manifest),
        "dependencies": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "versions": _dependency_versions(),
            "requirements_teacher_sha256": _requirements_hash(),
        },
        "runtime": {
            "device": "cpu",
            "torch_threads": runtime.threads,
            "torch_interop_threads": runtime.interop_threads,
            "batch_size": batch_size,
            "normalize_argument": False,
            "squeeze_axis": 1,
            "native_feature_shape_first_batch": list(first_observation.feature_shape),
            "native_feature_dtype": first_observation.feature_dtype,
            "raw_output_shape_first_batch": list(first_observation.raw_shape),
            "raw_output_dtype": first_observation.raw_dtype,
            "public_embedding_shape": [len(ids), EMBEDDING_DIM],
            "public_embedding_dtype": "float32",
            "l2_normalized": True,
            "model_load_ms": runtime.load_ms,
            "first_batch_inference_ms": first_observation.inference_ms,
            "batch_time_ms": batch_times,
            "total_extraction_ms": total_ms,
            "peak_process_rss_bytes": _peak_rss_bytes(),
            "input_sample_rate": SAMPLE_RATE,
            "input_samples_per_clip": NUM_SAMPLES,
        },
        "published_model_metrics": {
            "eer_percent": 0.80,
            "source": "https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb",
            "local_reproduction": False,
            "note": "Published model-card metric; not a local pilot result.",
        },
        "local_metrics": {
            "accuracy": None,
            "eer": None,
            "far": None,
            "min_dcf": None,
            "test_evaluation": "NOT_RUN_BY_AUTHORIZATION",
        },
        "output": artifact,
    }
    report_path = _report_path(output)
    report["report_path"] = str(report_path)
    _write_json(report_path, report)
    return report


def _failure_report(
    *,
    output_path: Path,
    error: BaseException,
    manifest_path: Path,
    model_dir: Path,
) -> dict[str, Any]:
    prior_report_path = _report_path(output_path)
    report_path = prior_report_path
    if report_path.exists():
        timestamp = time.time_ns()
        suffix = 0
        while True:
            suffix_text = f"-{suffix}" if suffix else ""
            candidate = report_path.with_name(
                f"{report_path.stem}.failure-{timestamp}{suffix_text}{report_path.suffix}"
            )
            if not candidate.exists():
                report_path = candidate
                break
            suffix += 1
    report = {
        "status": "TEACHER_EXTRACTION_FAIL",
        "scope": "EXPERIMENTAL_HOST_ONLY",
        "runtime_status": "NOT_RUN_OR_FAILED",
        "hardware_status": "UNVERIFIED_HARDWARE",
        "error": {"type": type(error).__name__, "message": str(error)},
        "manifest_path": str(manifest_path),
        "model_dir": str(model_dir),
        "output_path": str(output_path),
        "test_evaluation": "NOT_RUN_BY_AUTHORIZATION",
        "report_path": str(report_path),
    }
    if report_path != prior_report_path:
        report["preserved_report_path"] = str(prior_report_path)
    try:
        _write_json(report_path, report)
    except OSError:
        # The JSON on stdout still gives the caller a structured failure when
        # even the requested report location is not writable.
        report["report_write_error"] = True
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest_path = args.manifest.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    try:
        report = extract_manifest(
            manifest_path,
            model_dir,
            output_path,
            batch_size=args.batch_size,
            threads=args.threads,
        )
    except Exception as exc:
        report = _failure_report(
            output_path=output_path,
            error=exc,
            manifest_path=manifest_path,
            model_dir=model_dir,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the CLI command
    raise SystemExit(main())
