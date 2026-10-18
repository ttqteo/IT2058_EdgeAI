"""Crash-resumable offline teacher extraction for :mod:`edgeai.full_data`.

The teacher cache is a persistent output directory, not a disposable staging
tree.  Each 256-row shard is published only after its row IDs, source hashes,
PCM slice, model binding, and embedding file hash have been recorded.  A
restart verifies those bindings and computes only missing shards.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata as importlib_metadata
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .full_data import FullDataError, FullDataset, load_full_dataset


EMBEDDING_DIM = 192
DEFAULT_BATCH_SIZE = 4
DEFAULT_THREADS = 4
SHARD_ROWS = 256
EMBEDDING_NORM_TOL = 1e-5
FORMAT_VERSION = 1
EXTRACTION_CONTRACT_VERSION = 1
_EXTRACTION_DISTRIBUTIONS = {
    "numpy": "numpy",
    "torch": "torch",
    "torchaudio": "torchaudio",
    "speechbrain": "speechbrain",
}
_HASH_CHUNK_BYTES = 1024 * 1024
_CHECK_CHUNK = 256


class FullTeacherError(ValueError):
    """Raised when full teacher inputs, cache state, or outputs are invalid."""


def verify_model_snapshot(path: str | Path) -> Any:
    """Use the existing pinned offline model verifier without importing torch."""

    from .teacher import verify_model_snapshot as verify

    return verify(path)


def load_teacher(
    path: str | Path,
    *,
    cache_dir: str | Path | None = None,
    threads: int = DEFAULT_THREADS,
) -> Any:
    """Use the existing pinned CPU/offline teacher loader."""

    from .teacher import load_teacher as load

    return load(path, cache_dir=cache_dir, threads=threads)


def extract_batch(runtime: Any, pcm_batch: object) -> Any:
    """Use the existing normalized SpeechBrain batch API."""

    from .teacher import extract_batch as extract

    return extract(runtime, pcm_batch)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise FullTeacherError(f"expected regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _installed_dependency_versions() -> dict[str, str | None]:
    """Read extraction-environment versions without importing ML runtimes."""

    versions: dict[str, str | None] = {}
    for label, distribution in _EXTRACTION_DISTRIBUTIONS.items():
        try:
            versions[label] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            # Synthetic tests may intentionally omit the teacher environment.
            # A real extraction will fail at the existing offline loader boundary
            # if a required runtime is absent; the absence is still persisted.
            versions[label] = None
    return versions


def _source_record(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    return {
        "path": str(resolved),
        "bytes": int(resolved.stat().st_size),
        "sha256": _sha256_file(resolved),
    }


def _extraction_source_records(dataset: FullDataset) -> dict[str, dict[str, Any]]:
    # These are deliberately source files rather than imported package objects;
    # provenance/load consumers can verify them without importing torch.
    from . import frontend as frontend_module
    from . import teacher as teacher_module
    from . import full_data as full_data_module

    return {
        "full_teacher": _source_record(Path(__file__)),
        "full_data": _source_record(Path(full_data_module.__file__)),
        "teacher": _source_record(Path(teacher_module.__file__)),
        "frontend": _source_record(Path(frontend_module.__file__)),
        "dataset_manifest": _source_record(dataset.root / "manifest.json"),
        "dataset_provenance": _source_record(dataset.root / "provenance.json"),
    }


def _extraction_binding(
    dataset: FullDataset, *, compare_runtime_dependencies: bool
) -> dict[str, Any]:
    return {
        "version": EXTRACTION_CONTRACT_VERSION,
        "sources": _extraction_source_records(dataset),
        "dependencies": (
            _installed_dependency_versions()
            if compare_runtime_dependencies
            else None
        ),
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(_canonical_bytes(dict(value)) + b"\n")
    os.replace(temporary, path)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FullTeacherError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise FullTeacherError(f"{label} must be a JSON object: {path}")
    return value


def _validate_positive_int(value: object, field: str, *, maximum: int = 64) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise FullTeacherError(f"{field} must be an integer in [1, {maximum}]")
    return value


def _load_dataset_for_teacher(root: str | Path) -> FullDataset:
    dataset_candidate = Path(root).expanduser()
    if dataset_candidate.is_symlink():
        raise FullTeacherError("full dataset root may not be a symlink")
    dataset_root = dataset_candidate.resolve()
    provenance_path = dataset_root / "provenance.json"
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FullTeacherError("cannot identify full dataset provenance") from exc
    if not isinstance(provenance, dict):
        raise FullTeacherError("full dataset provenance must be a JSON object")
    is_fixture = provenance.get("status") == "SYNTHETIC_TEST_FIXTURE"
    try:
        return load_full_dataset(dataset_root, verify_hashes=True, test_fixture=is_fixture)
    except FullDataError as exc:
        raise FullTeacherError(f"invalid full dataset input: {exc}") from exc


def _snapshot_record(snapshot: Any) -> dict[str, Any]:
    try:
        record = snapshot.as_dict()
    except (AttributeError, TypeError) as exc:
        raise FullTeacherError("verified model snapshot has no stable as_dict API") from exc
    if not isinstance(record, dict):
        raise FullTeacherError("verified model snapshot is not a JSON object")
    return record


def _dataset_binding(dataset: FullDataset) -> dict[str, Any]:
    arrays = dataset.manifest["arrays"]
    source = dataset.manifest["source"]
    return {
        "manifest_sha256": dataset.manifest_sha256,
        "rows_sha256": dataset.manifest["rows_sha256"],
        "row_count": len(dataset.rows),
        "pcm_sha256": arrays["pcm"]["sha256"],
        "features_sha256": arrays["features"]["sha256"],
        "source_archive_sha256": source["archive_sha256"],
    }


def _model_binding(snapshot_record: Mapping[str, Any]) -> str:
    return _sha256_bytes(_canonical_bytes(dict(snapshot_record)))


def _validate_pinned_model_record(snapshot_record: Mapping[str, Any]) -> None:
    """Validate the persisted snapshot metadata against the pinned teacher contract."""

    if not isinstance(snapshot_record, Mapping):
        raise FullTeacherError("teacher model snapshot must be an object")
    from . import teacher as pinned_teacher

    if snapshot_record.get("repo") != pinned_teacher.MODEL_REPO:
        raise FullTeacherError("teacher model repository is not the pinned snapshot")
    if snapshot_record.get("revision") != pinned_teacher.MODEL_REVISION:
        raise FullTeacherError("teacher model revision is not the pinned snapshot")
    if snapshot_record.get("revision_is_immutable_commit") is not True:
        raise FullTeacherError("teacher model revision is not immutable")
    if snapshot_record.get("provenance_status") != (
        "PINNED_SNAPSHOT_HASH_MATCH_NO_CRYPTO_AUTHENTICATION"
    ):
        raise FullTeacherError("teacher model provenance is not pinned")
    files = snapshot_record.get("files")
    if not isinstance(files, Mapping):
        raise FullTeacherError("teacher model file bindings are missing")
    expected_files = pinned_teacher.MODEL_FILE_SHA256
    required_files = set(pinned_teacher.REQUIRED_MODEL_FILES)
    observed_files = set(files)
    if not required_files.issubset(observed_files) or not observed_files.issubset(
        set(expected_files)
    ):
        raise FullTeacherError("teacher model file set is not pinned")
    for filename, expected_sha256 in expected_files.items():
        if filename not in files:
            continue
        record = files[filename]
        if not isinstance(record, Mapping):
            raise FullTeacherError(f"teacher model binding is invalid: {filename}")
        if (
            record.get("sha256") != expected_sha256
            or record.get("expected_sha256") != expected_sha256
            or record.get("verified") is not True
        ):
            raise FullTeacherError(
                f"teacher model file is not pinned: {filename}"
            )


def _validate_cache_config(
    dataset: FullDataset,
    config: Mapping[str, Any],
    *,
    compare_runtime_dependencies: bool,
) -> None:
    if config.get("format_version") != FORMAT_VERSION:
        raise FullTeacherError("teacher cache format version is unsupported")
    if config.get("dataset") != _dataset_binding(dataset):
        raise FullTeacherError("teacher output dataset/source binding mismatch")
    model = config.get("model")
    if not isinstance(model, Mapping):
        raise FullTeacherError("teacher cache model binding is missing")
    if config.get("model_binding_sha256") != _model_binding(model):
        raise FullTeacherError("teacher cache model binding hash mismatch")
    _validate_pinned_model_record(model)

    extraction = config.get("extraction")
    if not isinstance(extraction, Mapping):
        raise FullTeacherError("teacher cache extraction binding is missing")
    if extraction.get("version") != EXTRACTION_CONTRACT_VERSION:
        raise FullTeacherError("teacher cache extraction contract version mismatch")
    if extraction.get("sources") != _extraction_source_records(dataset):
        raise FullTeacherError("teacher cache extraction source binding mismatch")
    dependencies = extraction.get("dependencies")
    if not isinstance(dependencies, Mapping) or set(dependencies) != set(
        _EXTRACTION_DISTRIBUTIONS
    ):
        raise FullTeacherError("teacher cache dependency version binding is invalid")
    if any(value is not None and not isinstance(value, str) for value in dependencies.values()):
        raise FullTeacherError("teacher cache dependency versions are invalid")
    if compare_runtime_dependencies and dict(dependencies) != _installed_dependency_versions():
        raise FullTeacherError("teacher extraction dependency versions changed")


def _row_binding(rows: list[dict[str, Any]]) -> str:
    return _sha256_bytes(_canonical_bytes(rows))


def _source_binding(rows: list[dict[str, Any]]) -> str:
    return _sha256_bytes(
        _canonical_bytes(
            [
                {
                    "window_id": row["window_id"],
                    "source_sha256": row["source_sha256"],
                }
                for row in rows
            ]
        )
    )


def _shards_binding(sidecars: list[Mapping[str, Any]]) -> str:
    return _sha256_bytes(
        _canonical_bytes(
            [
                {
                    "shard_index": int(sidecar["shard_index"]),
                    "global_start": int(sidecar["global_start"]),
                    "global_end": int(sidecar["global_end"]),
                    "sha256": sidecar["sha256"],
                }
                for sidecar in sorted(
                    sidecars, key=lambda item: int(item["shard_index"])
                )
            ]
        )
    )


def _pcm_slice_binding(dataset: FullDataset, start: int, end: int) -> str:
    values = np.ascontiguousarray(dataset.pcm[start:end])
    return _sha256_bytes(values.tobytes(order="C"))


def _cache_config(
    dataset: FullDataset,
    snapshot_record: Mapping[str, Any],
    *,
    batch_size: int,
    threads: int,
) -> dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION,
        "dataset": _dataset_binding(dataset),
        "model": dict(snapshot_record),
        "model_binding_sha256": _model_binding(snapshot_record),
        "extraction": _extraction_binding(
            dataset, compare_runtime_dependencies=True
        ),
        "batch_size": batch_size,
        "threads": threads,
        "shard_rows": SHARD_ROWS,
        "embedding_shape": [SHARD_ROWS, EMBEDDING_DIM],
        "embedding_dtype": "float32",
        "runtime_cache_relative": "teacher-runtime-cache",
    }


def _ensure_output_dir(output: Path) -> None:
    if output.is_symlink() or (output.exists() and not output.is_dir()):
        raise FullTeacherError(f"teacher output is not a regular directory: {output}")
    if not output.exists():
        output.mkdir(parents=True)
        return
    config_path = output / "config.json"
    if not config_path.exists() and any(output.iterdir()):
        raise FullTeacherError(
            f"existing teacher output has no resumable config: {output}"
        )


def _config_hash(config: Mapping[str, Any]) -> str:
    value = dict(config)
    # ``final`` is publication state, not an input/configuration binding.  It
    # is added only after all shards have been verified and must not make an
    # otherwise compatible restart look like a different run.
    value.pop("final", None)
    return _sha256_bytes(_canonical_bytes(value))


def _check_or_write_config(output: Path, config: dict[str, Any]) -> str:
    path = output / "config.json"
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file():
            raise FullTeacherError("teacher config is not a regular file")
        existing = _read_json(path, label="teacher config")
        existing_base = dict(existing)
        existing_final = existing_base.pop("final", None)
        if existing_base != config:
            raise FullTeacherError(
                "existing teacher cache config differs; refusing resume/overwrite"
            )
        if existing_final is not None:
            config["final"] = existing_final
    else:
        _write_json(path, config)
    return _config_hash(config)


def _recheck_inputs_before_publish(
    dataset_root: str | Path,
    model_dir: str | Path,
    dataset: FullDataset,
    config: Mapping[str, Any],
) -> FullDataset:
    """Rehash mutable data/model/source inputs once before consolidation."""

    if _dataset_binding(dataset) != config.get("dataset"):
        raise FullTeacherError("teacher dataset changed during extraction")
    # This is intentionally one full dataset verification after all shards,
    # rather than a 19-GB array hash for every 256-row shard.
    current_dataset = _load_dataset_for_teacher(dataset_root)
    _validate_cache_config(
        current_dataset, config, compare_runtime_dependencies=True
    )
    try:
        current_snapshot = _snapshot_record(verify_model_snapshot(model_dir))
    except Exception as exc:
        raise FullTeacherError(
            "teacher model changed or is no longer a verified pinned snapshot"
        ) from exc
    _validate_pinned_model_record(current_snapshot)
    if (
        current_snapshot != config.get("model")
        or _model_binding(current_snapshot) != config.get("model_binding_sha256")
    ):
        raise FullTeacherError("teacher model changed during extraction")
    return current_dataset


def _shard_paths(shards_dir: Path, index: int) -> tuple[Path, Path]:
    stem = f"shard-{index:06d}"
    return shards_dir / f"{stem}.npy", shards_dir / f"{stem}.json"


def _expected_shard_record(
    dataset: FullDataset,
    config: Mapping[str, Any],
    *,
    index: int,
    start: int,
    end: int,
    embedding_hash: str,
) -> dict[str, Any]:
    rows = dataset.rows[start:end]
    return {
        "format_version": FORMAT_VERSION,
        "shard_index": index,
        "global_start": start,
        "global_end": end,
        "row_count": end - start,
        "path": f"shard-{index:06d}.npy",
        "shape": [end - start, EMBEDDING_DIM],
        "dtype": "float32",
        "sha256": embedding_hash,
        "dataset_manifest_sha256": config["dataset"]["manifest_sha256"],
        "dataset_rows_sha256": config["dataset"]["rows_sha256"],
        "pcm_array_sha256": config["dataset"]["pcm_sha256"],
        "source_archive_sha256": config["dataset"]["source_archive_sha256"],
        "model_binding_sha256": config["model_binding_sha256"],
        "batch_size": config["batch_size"],
        "threads": config["threads"],
        "shard_rows": config["shard_rows"],
        "row_binding_sha256": _row_binding(rows),
        "source_binding_sha256": _source_binding(rows),
        "pcm_slice_sha256": _pcm_slice_binding(dataset, start, end),
    }


def _validate_embedding_values(values: object, expected_rows: int) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype != np.float32 or array.shape != (expected_rows, EMBEDDING_DIM):
        raise FullTeacherError(
            f"teacher output must be float32 [{expected_rows},{EMBEDDING_DIM}], "
            f"got {array.shape} {array.dtype}"
        )
    if not np.all(np.isfinite(array)):
        raise FullTeacherError("teacher output contains non-finite values")
    norms = np.linalg.norm(array.astype(np.float64), axis=1)
    if not np.all(np.isfinite(norms)) or np.any(norms <= 0.0):
        raise FullTeacherError("teacher output contains zero/non-finite norms")
    if np.any(np.abs(norms - 1.0) > EMBEDDING_NORM_TOL):
        raise FullTeacherError(
            f"teacher output is not unit norm within {EMBEDDING_NORM_TOL}"
        )
    return np.ascontiguousarray(array)


def _verify_shard(
    data_path: Path,
    sidecar_path: Path,
    dataset: FullDataset,
    config: Mapping[str, Any],
    *,
    index: int,
    start: int,
    end: int,
) -> dict[str, Any]:
    if data_path.is_symlink() or sidecar_path.is_symlink():
        raise FullTeacherError(f"shard {index} contains a symlink")
    if not data_path.is_file() or not sidecar_path.is_file():
        raise FullTeacherError(f"shard {index} is incomplete")
    sidecar = _read_json(sidecar_path, label=f"shard {index} sidecar")
    expected_fields = {
        "format_version",
        "shard_index",
        "global_start",
        "global_end",
        "row_count",
        "path",
        "shape",
        "dtype",
        "sha256",
        "dataset_manifest_sha256",
        "dataset_rows_sha256",
        "pcm_array_sha256",
        "source_archive_sha256",
        "model_binding_sha256",
        "batch_size",
        "threads",
        "shard_rows",
        "row_binding_sha256",
        "source_binding_sha256",
        "pcm_slice_sha256",
    }
    if set(sidecar) != expected_fields:
        raise FullTeacherError(f"shard {index} sidecar fields are not exact")
    expected_shape = (end - start, EMBEDDING_DIM)
    if sidecar.get("global_start") != start or sidecar.get("global_end") != end:
        raise FullTeacherError(f"shard {index} row range binding mismatch")
    if sidecar.get("row_count") != end - start or tuple(sidecar.get("shape", [])) != expected_shape:
        raise FullTeacherError(f"shard {index} shape/count binding mismatch")
    if sidecar.get("path") != data_path.name or sidecar.get("dtype") != "float32":
        raise FullTeacherError(f"shard {index} path/dtype binding mismatch")
    expected = _expected_shard_record(
        dataset,
        config,
        index=index,
        start=start,
        end=end,
        embedding_hash=str(sidecar.get("sha256", "")),
    )
    for key, expected_value in expected.items():
        if sidecar.get(key) != expected_value:
            raise FullTeacherError(f"shard {index} binding mismatch: {key}")
    if _sha256_file(data_path) != sidecar["sha256"]:
        raise FullTeacherError(f"shard {index} embedding hash mismatch")
    try:
        values = np.load(data_path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise FullTeacherError(f"cannot map shard {index}") from exc
    if not isinstance(values, np.memmap):
        raise FullTeacherError(f"shard {index} is not a NumPy mmap")
    checked = _validate_embedding_values(values, end - start)
    del checked
    try:
        values.flags.writeable = False
    except ValueError as exc:
        raise FullTeacherError(f"shard {index} is not readonly") from exc
    return sidecar


def _remove_orphan(data_path: Path, sidecar_path: Path) -> None:
    # An orphan is a crash between the two atomic publications and is not a
    # completed shard.  Completed sidecar+array pairs are never overwritten.
    if data_path.exists() and not sidecar_path.exists():
        data_path.unlink()
    if sidecar_path.exists() and not data_path.exists():
        sidecar_path.unlink()


def _existing_shards(
    dataset: FullDataset,
    config: Mapping[str, Any],
    shards_dir: Path,
) -> tuple[list[dict[str, Any]], list[tuple[int, int, int]]]:
    completed: list[dict[str, Any]] = []
    missing: list[tuple[int, int, int]] = []
    total = len(dataset.rows)
    shard_index = 0
    for start in range(0, total, SHARD_ROWS):
        end = min(start + SHARD_ROWS, total)
        data_path, sidecar_path = _shard_paths(shards_dir, shard_index)
        if data_path.exists() != sidecar_path.exists():
            _remove_orphan(data_path, sidecar_path)
        if data_path.exists() or sidecar_path.exists():
            if not data_path.is_file() or not sidecar_path.is_file():
                raise FullTeacherError(f"shard {shard_index} is not a regular pair")
            completed.append(
                _verify_shard(
                    data_path,
                    sidecar_path,
                    dataset,
                    config,
                    index=shard_index,
                    start=start,
                    end=end,
                )
            )
        else:
            missing.append((shard_index, start, end))
        shard_index += 1
    unexpected = {
        path.name
        for path in shards_dir.iterdir()
        if path.name not in {
            _shard_paths(shards_dir, index)[0].name
            for index in range(shard_index)
        }
        and path.name not in {
            _shard_paths(shards_dir, index)[1].name
            for index in range(shard_index)
        }
        and not path.name.startswith(".")
    }
    if unexpected:
        raise FullTeacherError(f"unexpected files in teacher shard cache: {sorted(unexpected)}")
    return completed, missing


def _write_progress(
    output: Path,
    config_hash: str,
    completed: list[dict[str, Any]],
    *,
    total_rows: int,
    started: float,
    status: str,
) -> None:
    rows_completed = sum(int(item["row_count"]) for item in completed)
    value = {
        "status": status,
        "config_sha256": config_hash,
        "completed_shards": [int(item["shard_index"]) for item in completed],
        "completed_rows": rows_completed,
        "total_rows": total_rows,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_json(output / "progress.json", value)


def _extract_one_shard(
    dataset: FullDataset,
    runtime: Any,
    config: Mapping[str, Any],
    shards_dir: Path,
    *,
    index: int,
    start: int,
    end: int,
) -> dict[str, Any]:
    data_path, sidecar_path = _shard_paths(shards_dir, index)
    if data_path.exists() or sidecar_path.exists():
        raise FullTeacherError(f"refusing to overwrite existing shard {index}")
    temporary = shards_dir / f".shard-{index:06d}.{os.getpid()}.tmp.npy"
    try:
        values = np.lib.format.open_memmap(
            temporary,
            mode="w+",
            dtype=np.float32,
            shape=(end - start, EMBEDDING_DIM),
        )
        for batch_start in range(start, end, int(config["batch_size"])):
            batch_end = min(batch_start + int(config["batch_size"]), end)
            pcm = np.asarray(dataset.pcm[batch_start:batch_end], dtype=np.float32)
            pcm = np.ascontiguousarray(pcm / np.float32(32768.0))
            try:
                observation = extract_batch(runtime, pcm)
                raw_embeddings = getattr(observation, "embeddings", observation)
                embeddings = _validate_embedding_values(
                    raw_embeddings, batch_end - batch_start
                )
            except FullTeacherError:
                raise
            except Exception as exc:
                raise FullTeacherError(
                    f"teacher inference failed in shard {index}, rows "
                    f"{batch_start}:{batch_end}"
                ) from exc
            values[batch_start - start : batch_end - start] = embeddings
        values.flush()
        del values
        os.replace(temporary, data_path)
        embedding_hash = _sha256_file(data_path)
        sidecar = _expected_shard_record(
            dataset,
            config,
            index=index,
            start=start,
            end=end,
            embedding_hash=embedding_hash,
        )
        _write_json(sidecar_path, sidecar)
        _verify_shard(
            data_path,
            sidecar_path,
            dataset,
            config,
            index=index,
            start=start,
            end=end,
        )
        return sidecar
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _final_path(output: Path) -> Path:
    return output / "teacher_embeddings.npy"


def _validate_final(
    output: Path,
    dataset: FullDataset,
    config: Mapping[str, Any],
) -> np.memmap:
    path = _final_path(output)
    if path.is_symlink() or not path.is_file():
        raise FullTeacherError(f"missing consolidated teacher output: {path}")
    final_record = config.get("final")
    if not isinstance(final_record, dict) or final_record.get("path") != path.name:
        raise FullTeacherError("teacher config has no consolidated output binding")
    expected_shape = [len(dataset.rows), EMBEDDING_DIM]
    if (
        final_record.get("shape") != expected_shape
        or final_record.get("dtype") != "float32"
        or final_record.get("dataset_manifest_sha256")
        != config.get("dataset", {}).get("manifest_sha256")
        or final_record.get("dataset_rows_sha256")
        != config.get("dataset", {}).get("rows_sha256")
        or final_record.get("model_binding_sha256")
        != config.get("model_binding_sha256")
        or not isinstance(final_record.get("shards_sha256"), str)
        or len(final_record["shards_sha256"]) != 64
    ):
        raise FullTeacherError("consolidated teacher binding/schema mismatch")
    try:
        expected_bytes = int(final_record["bytes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise FullTeacherError("consolidated teacher byte binding is invalid") from exc
    if expected_bytes != int(path.stat().st_size):
        raise FullTeacherError("consolidated teacher byte count mismatch")
    if final_record.get("sha256") != _sha256_file(path):
        raise FullTeacherError("consolidated teacher hash mismatch")
    try:
        values = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise FullTeacherError("cannot map consolidated teacher output") from exc
    if not isinstance(values, np.memmap):
        raise FullTeacherError("consolidated teacher output is not a NumPy mmap")
    expected_shape = (len(dataset.rows), EMBEDDING_DIM)
    if values.dtype != np.float32 or values.shape != expected_shape:
        raise FullTeacherError("consolidated teacher shape/dtype mismatch")
    _validate_embedding_values(values, len(dataset.rows))
    try:
        values.flags.writeable = False
    except ValueError as exc:
        raise FullTeacherError("consolidated teacher mmap is not readonly") from exc
    return values


def _assert_consolidated_matches_shards(
    output: Path,
    dataset: FullDataset,
    values: np.ndarray,
    verified: list[Mapping[str, Any]],
) -> None:
    if values.shape != (len(dataset.rows), EMBEDDING_DIM):
        raise FullTeacherError("consolidated teacher row shape is not dataset-bound")
    expected_shards_hash = _shards_binding(verified)
    # The final record is checked by its caller; this comparison binds the
    # actual mmap bytes to the persistent, individually verified shards.
    config = _read_json(output / "config.json", label="teacher config")
    final = config.get("final")
    if not isinstance(final, dict) or final.get("shards_sha256") != expected_shards_hash:
        raise FullTeacherError("consolidated teacher shard-set binding mismatch")
    for sidecar in verified:
        index = int(sidecar["shard_index"])
        start = int(sidecar["global_start"])
        end = int(sidecar["global_end"])
        shard_path, _ = _shard_paths(output / "shards", index)
        shard = np.load(shard_path, mmap_mode="r", allow_pickle=False)
        if not np.array_equal(values[start:end], shard):
            raise FullTeacherError(
                f"consolidated teacher differs from verified shard {index}"
            )


def _consolidate(
    output: Path,
    dataset: FullDataset,
    config: dict[str, Any],
    verified: list[dict[str, Any]],
) -> tuple[np.memmap, str]:
    path = _final_path(output)
    if path.exists() or path.is_symlink():
        if path.is_symlink():
            raise FullTeacherError("refusing to overwrite symlinked consolidated output")
        if not isinstance(config.get("final"), dict):
            # A crash can occur after the final mmap is atomically published
            # but before the config journal records its binding.  Recover only
            # when the existing file is exactly the verified shard concat.
            try:
                recovered = np.load(path, mmap_mode="r", allow_pickle=False)
            except (OSError, ValueError) as exc:
                raise FullTeacherError("cannot recover orphan consolidated output") from exc
            if not isinstance(recovered, np.memmap):
                raise FullTeacherError("orphan consolidated output is not a NumPy mmap")
            if recovered.dtype != np.float32 or recovered.shape != (len(dataset.rows), EMBEDDING_DIM):
                raise FullTeacherError("orphan consolidated output shape/dtype mismatch")
            _validate_embedding_values(recovered, len(dataset.rows))
            for sidecar in verified:
                index = int(sidecar["shard_index"])
                start = int(sidecar["global_start"])
                end = int(sidecar["global_end"])
                shard_path, _ = _shard_paths(output / "shards", index)
                shard = np.load(shard_path, mmap_mode="r", allow_pickle=False)
                if not np.array_equal(recovered[start:end], shard):
                    raise FullTeacherError("orphan consolidated output differs from verified shards")
            final_hash = _sha256_file(path)
            config["final"] = {
                "path": path.name,
                "shape": [len(dataset.rows), EMBEDDING_DIM],
                "dtype": "float32",
                "bytes": int(path.stat().st_size),
                "sha256": final_hash,
                "dataset_manifest_sha256": config["dataset"]["manifest_sha256"],
                "dataset_rows_sha256": config["dataset"]["rows_sha256"],
                "model_binding_sha256": config["model_binding_sha256"],
                "shards_sha256": _shards_binding(verified),
            }
            _write_json(output / "config.json", config)
        values = _validate_final(output, dataset, config)
        _assert_consolidated_matches_shards(output, dataset, values, verified)
        return values, str(config["final"]["sha256"])
    temporary = output / f".teacher_embeddings.{os.getpid()}.tmp.npy"
    try:
        consolidated = np.lib.format.open_memmap(
            temporary,
            mode="w+",
            dtype=np.float32,
            shape=(len(dataset.rows), EMBEDDING_DIM),
        )
        for sidecar in sorted(verified, key=lambda item: int(item["shard_index"])):
            index = int(sidecar["shard_index"])
            start = int(sidecar["global_start"])
            end = int(sidecar["global_end"])
            shard_path, sidecar_path = _shard_paths(output / "shards", index)
            _verify_shard(
                shard_path,
                sidecar_path,
                dataset,
                config,
                index=index,
                start=start,
                end=end,
            )
            shard = np.load(shard_path, mmap_mode="r", allow_pickle=False)
            consolidated[start:end] = shard
        consolidated.flush()
        del consolidated
        final_hash = _sha256_file(temporary)
        if path.exists() or path.is_symlink():
            raise FullTeacherError("consolidated teacher appeared during publication")
        os.replace(temporary, path)
        config["final"] = {
            "path": path.name,
            "shape": [len(dataset.rows), EMBEDDING_DIM],
            "dtype": "float32",
            "bytes": int(path.stat().st_size),
            "sha256": final_hash,
            "dataset_manifest_sha256": config["dataset"]["manifest_sha256"],
            "dataset_rows_sha256": config["dataset"]["rows_sha256"],
            "model_binding_sha256": config["model_binding_sha256"],
            "shards_sha256": _shards_binding(verified),
        }
        # Updating config after publishing the final file is itself atomic.  A
        # crash leaves all verified shards and simply rebuilds the small final
        # map on the next invocation.
        _write_json(output / "config.json", config)
        values = _validate_final(output, dataset, config)
        _assert_consolidated_matches_shards(output, dataset, values, verified)
        return values, final_hash
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def load_full_teacher(
    output_dir: str | Path,
    dataset: FullDataset,
) -> np.ndarray:
    """Load the consolidated teacher matrix as a read-only float32 mmap."""

    if not isinstance(dataset, FullDataset):
        raise FullTeacherError("dataset must be a validated FullDataset")
    output_candidate = Path(output_dir).expanduser()
    if output_candidate.is_symlink():
        raise FullTeacherError("teacher output may not be a symlink")
    output = output_candidate.resolve()
    if not output.is_dir():
        raise FullTeacherError(f"teacher output is not a regular directory: {output}")
    config = _read_json(output / "config.json", label="teacher config")
    _validate_cache_config(
        dataset, config, compare_runtime_dependencies=False
    )
    binding = config.get("dataset")
    if binding != _dataset_binding(dataset):
        raise FullTeacherError("teacher output dataset/source binding mismatch")
    final = config.get("final")
    if not isinstance(final, dict):
        raise FullTeacherError("teacher output is not consolidated")
    if final.get("model_binding_sha256") != config.get("model_binding_sha256"):
        raise FullTeacherError("teacher final model binding mismatch")
    shards_dir = output / "shards"
    if shards_dir.is_symlink() or not shards_dir.is_dir():
        raise FullTeacherError("teacher shard directory is missing")
    verified, missing = _existing_shards(dataset, config, shards_dir)
    if missing or len(verified) != (len(dataset.rows) + SHARD_ROWS - 1) // SHARD_ROWS:
        raise FullTeacherError("teacher shard cache is incomplete")
    values = _validate_final(output, dataset, config)
    _assert_consolidated_matches_shards(output, dataset, values, verified)
    return values


def teacher_provenance(
    output_dir: str | Path,
    dataset: FullDataset,
) -> dict[str, Any]:
    """Return verified public provenance for a consolidated teacher cache.

    Consumers receive stable bindings for the source, prepared dataset, model,
    shard set, final matrix, and report.  They do not need to depend on the
    internal report or sidecar layout.
    """

    if not isinstance(dataset, FullDataset):
        raise FullTeacherError("dataset must be a validated FullDataset")
    output_candidate = Path(output_dir).expanduser()
    if output_candidate.is_symlink():
        raise FullTeacherError("teacher output may not be a symlink")
    output = output_candidate.resolve()
    if not output.is_dir():
        raise FullTeacherError(f"teacher output is not a regular directory: {output}")
    config = _read_json(output / "config.json", label="teacher config")
    _validate_cache_config(
        dataset, config, compare_runtime_dependencies=False
    )
    if config.get("dataset") != _dataset_binding(dataset):
        raise FullTeacherError("teacher provenance dataset/source binding mismatch")
    if config.get("shard_rows") != SHARD_ROWS:
        raise FullTeacherError("teacher provenance shard configuration mismatch")
    shards_dir = output / "shards"
    if shards_dir.is_symlink() or not shards_dir.is_dir():
        raise FullTeacherError("teacher provenance shard directory is missing")
    verified_shards, missing = _existing_shards(
        dataset, config, shards_dir
    )
    if missing or len(verified_shards) != (len(dataset.rows) + SHARD_ROWS - 1) // SHARD_ROWS:
        raise FullTeacherError("teacher provenance has incomplete shards")
    final_values = _validate_final(output, dataset, config)
    _assert_consolidated_matches_shards(
        output, dataset, final_values, verified_shards
    )
    del final_values
    report_path = output / "teacher_report.json"
    report = _read_json(report_path, label="teacher report")
    final = config["final"]
    report_bindings = {
        "path": report_path.name,
        "sha256": _sha256_file(report_path),
        "dataset_manifest_sha256": report.get("dataset_manifest_sha256"),
        "teacher_embeddings_sha256": report.get("teacher_embeddings_sha256"),
        "model_binding_sha256": report.get("model_binding_sha256"),
        "verified": True,
    }
    if (
        report_bindings["dataset_manifest_sha256"] != dataset.manifest_sha256
        or report_bindings["teacher_embeddings_sha256"] != final.get("sha256")
        or report_bindings["model_binding_sha256"] != config.get("model_binding_sha256")
    ):
        raise FullTeacherError("teacher report bindings do not match verified output")
    result: dict[str, Any] = {
        "status": "VERIFIED_FULL_TEACHER_PROVENANCE",
        "extraction": dict(config["extraction"]),
        "dataset": {
            "root": str(dataset.root),
            **_dataset_binding(dataset),
        },
        "source": dict(dataset.manifest["source"]),
        "model": {
            "binding_sha256": config["model_binding_sha256"],
            "snapshot": config["model"],
        },
        "teacher": {
            "output_dir": str(output),
            "path": final["path"],
            "shape": list(final["shape"]),
            "dtype": final["dtype"],
            "bytes": final["bytes"],
            "sha256": final["sha256"],
            "shard_rows": SHARD_ROWS,
            "shards": len(verified_shards),
            "verified": True,
        },
        "report": report_bindings,
    }
    try:
        json.dumps(result, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise FullTeacherError("teacher provenance is not JSON serializable") from exc
    return result


def extract_full_teacher(
    dataset_root: str | Path,
    model_dir: str | Path,
    output_dir: str | Path,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    threads: int = DEFAULT_THREADS,
) -> dict[str, Any]:
    """Extract full train-only teacher embeddings with persistent resumption."""

    batch_size = _validate_positive_int(batch_size, "batch_size")
    threads = _validate_positive_int(threads, "threads")
    dataset = _load_dataset_for_teacher(dataset_root)
    output_candidate = Path(output_dir).expanduser()
    if output_candidate.is_symlink():
        raise FullTeacherError("teacher output may not be a symlink")
    output = output_candidate.resolve()
    _ensure_output_dir(output)
    (output / "shards").mkdir(exist_ok=True)
    snapshot = verify_model_snapshot(model_dir)
    snapshot_record = _snapshot_record(snapshot)
    _validate_pinned_model_record(snapshot_record)
    config = _cache_config(
        dataset,
        snapshot_record,
        batch_size=batch_size,
        threads=threads,
    )
    _validate_cache_config(
        dataset, config, compare_runtime_dependencies=True
    )
    config_hash = _check_or_write_config(output, config)
    started = time.perf_counter()
    shards_dir = output / "shards"
    completed, missing = _existing_shards(dataset, config, shards_dir)
    if missing:
        runtime = load_teacher(
            model_dir,
            cache_dir=output / "teacher-runtime-cache",
            threads=threads,
        )
        for index, start, end in missing:
            sidecar = _extract_one_shard(
                dataset,
                runtime,
                config,
                shards_dir,
                index=index,
                start=start,
                end=end,
            )
            completed.append(sidecar)
            completed.sort(key=lambda item: int(item["shard_index"]))
            _write_progress(
                output,
                config_hash,
                completed,
                total_rows=len(dataset.rows),
                started=started,
                status="IN_PROGRESS",
            )
            print(
                json.dumps(
                    {
                        "status": "SHARD_COMPLETE",
                        "shard": index,
                        "rows": end - start,
                        "completed_rows": sum(int(item["row_count"]) for item in completed),
                        "total_rows": len(dataset.rows),
                        "elapsed_seconds": time.perf_counter() - started,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    # Rehash mutable dataset/model/source inputs once, then verify every
    # persistent pair immediately before consolidation.
    dataset = _recheck_inputs_before_publish(
        dataset_root, model_dir, dataset, config
    )
    verified, remaining = _existing_shards(dataset, config, shards_dir)
    if remaining or len(verified) != (len(dataset.rows) + SHARD_ROWS - 1) // SHARD_ROWS:
        raise FullTeacherError("teacher shard cache is incomplete after extraction")
    values, final_hash = _consolidate(output, dataset, config, verified)
    del values
    _write_progress(
        output,
        config_hash,
        verified,
        total_rows=len(dataset.rows),
        started=started,
        status="READY",
    )
    report = {
        "status": "TEACHER_EXTRACTED" if missing else "TEACHER_CACHE_REUSED",
        "output_dir": str(output),
        "extraction": dict(config["extraction"]),
        "dataset_manifest_sha256": dataset.manifest_sha256,
        "dataset_rows": len(dataset.rows),
        "batch_size": batch_size,
        "threads": threads,
        "shard_rows": SHARD_ROWS,
        "shards": len(verified),
        "teacher_embeddings": str(_final_path(output)),
        "teacher_embeddings_sha256": final_hash,
        "model_binding_sha256": config["model_binding_sha256"],
        "model": snapshot_record,
        "elapsed_seconds": time.perf_counter() - started,
    }
    report_path = output / "teacher_report.json"
    if not report_path.exists():
        _write_json(report_path, report)
    return report


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_THREADS",
    "EMBEDDING_DIM",
    "EMBEDDING_NORM_TOL",
    "FullTeacherError",
    "SHARD_ROWS",
    "extract_full_teacher",
    "load_full_teacher",
    "teacher_provenance",
]


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Resume-safe offline full train-clean-100 teacher extraction"
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    args = parser.parse_args(argv)
    result = extract_full_teacher(
        args.dataset_root,
        args.model_dir,
        args.output,
        batch_size=args.batch_size,
        threads=args.threads,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a CLI process.
    raise SystemExit(_main())
