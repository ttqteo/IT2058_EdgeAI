"""Prepare immutable FULL100 real-train vectors for the ESP32-S3 smoke app.

The output is deliberately a small, self-contained generated tree consumed by
``firmware/esp32_s3_smoke``.  It contains the pinned INT8 model, five real
train-clean-100 feature windows, and the corresponding desktop TFLite
BUILTIN_REF outputs.  No validation, calibration, or test row is inferred.

The script refuses to overwrite an existing output directory.  This is an
intentional regeneration guard: choose a new generated directory for a new
source binding, or remove the old generated directory only after reviewing
its manifest.  All generated files are written below ``--output-dir``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "artifacts/full100/20261015/baseline_kd/encoder_int8.tflite"
DEFAULT_DATASET = ROOT / "artifacts/full100/20261015/data"
DEFAULT_HOST_RUNNER = ROOT / "artifacts/tflm-host/edgeai_tflm_host"
DEFAULT_OUTPUT = ROOT / "firmware/esp32_s3_smoke/generated"

MODEL_SHA256 = "aa42a3140f375f3a2c93648893132b880dd3e9f4be4ca8f82cde09fc6c440727"
MODEL_BYTES = 154_000
INPUT_SHAPE = (1, 298, 40, 1)
OUTPUT_SHAPE = (1, 192)
INPUT_BYTES = 11_920
OUTPUT_BYTES = 192
ARENA_BYTES = 131_072
LSB_CUTOFF = 1

# These are the exact rows already checked by the FULL100 host run.  Keeping
# the row IDs here prevents a future dataset reorder from silently changing
# the board smoke input while retaining the same integer index.
REAL_TRAIN_FIXTURES: tuple[tuple[int, str], ...] = (
    (0, "103-1240-0000@0000000000"),
    (33_859, "2691-156745-0028@0000000000"),
    (67_718, "4397-15668-0014@0000144000"),
    (101_577, "6836-61803-0036@0000000000"),
    (135_437, "911-130578-0020@0000096000"),
)


class PreparationError(ValueError):
    """Raised when a source or generated-output contract is not satisfied."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise PreparationError(f"expected a regular file, not a symlink: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_binding(path: Path) -> dict[str, Any]:
    return {"bytes": int(path.stat().st_size), "sha256": _sha256_file(path)}


def _write_bytes(path: Path, value: bytes) -> dict[str, Any]:
    if path.exists() or path.is_symlink():
        raise PreparationError(f"refusing to overwrite generated file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return _file_binding(path)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise PreparationError(f"refusing to overwrite generated file: {path}")
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _regular_source(path: Path, label: str) -> Path:
    candidate = path.expanduser()
    if candidate.is_symlink() or not candidate.is_file():
        raise PreparationError(f"{label} must be an existing regular file: {candidate}")
    return candidate.resolve()


def _prepare_output_dir(candidate: Path) -> Path:
    output = candidate.expanduser()
    if output.exists() or output.is_symlink():
        raise PreparationError(
            f"refusing to overwrite existing generated output directory: {output}; "
            "choose a new directory after reviewing the old manifest"
        )
    if not output.parent.is_dir():
        raise PreparationError(f"generated output parent does not exist: {output.parent}")
    output.mkdir()
    return output.resolve()


def _load_full_train_dataset(dataset_root: Path) -> Any:
    # Keep the preparation script runnable from a source checkout without
    # relying on an editable install of the repository package.
    source_root = ROOT / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    try:
        from edgeai.full_data import load_full_dataset
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise PreparationError(
            "the repository edgeai package could not be imported; run with the "
            "project virtual environment"
        ) from exc
    try:
        dataset = load_full_dataset(dataset_root, verify_hashes=True)
    except Exception as exc:
        raise PreparationError(
            f"FULL100 dataset validation failed; refusing to prepare vectors: {exc}"
        ) from exc
    manifest = dataset.manifest
    if manifest.get("split") != "train" or manifest.get("dataset") != "librispeech-train-clean-100-full-v1":
        raise PreparationError("the source dataset is not the published FULL100 TRAIN split")
    if manifest.get("row_count") != 135_438:
        raise PreparationError("the source dataset row count is not the FULL100 TRAIN oracle")
    expected_features_shape = (manifest["row_count"],) + INPUT_SHAPE[1:]
    if tuple(dataset.features.shape) != expected_features_shape:
        # Retaining the source shape check at this boundary prevents a future
        # dataset layout change from silently selecting the wrong bytes.
        raise PreparationError(f"unexpected feature array shape: {dataset.features.shape}")
    return dataset


def _reference_interpreter(model_bytes: bytes) -> tuple[Any, dict[str, Any], dict[str, Any], str]:
    source_root = ROOT / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    try:
        from edgeai.tflm import _reference_interpreter as make_reference
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise PreparationError(
            "the repository TFLite reference helper could not be imported"
        ) from exc
    try:
        interpreter, input_metadata, output_metadata, mode = make_reference(model_bytes)
    except Exception as exc:
        raise PreparationError(
            f"desktop BUILTIN_REF oracle unavailable; refusing to generate vectors: {exc}"
        ) from exc
    if mode != "BUILTIN_REF_NO_DELEGATES":
        raise PreparationError(f"unexpected desktop reference mode: {mode}")
    if tuple(input_metadata.get("shape", ())) != INPUT_SHAPE or input_metadata.get("dtype") != "int8":
        raise PreparationError(f"desktop input metadata does not match {INPUT_SHAPE}/int8")
    if tuple(output_metadata.get("shape", ())) != OUTPUT_SHAPE or output_metadata.get("dtype") != "int8":
        raise PreparationError(f"desktop output metadata does not match {OUTPUT_SHAPE}/int8")
    return interpreter, input_metadata, output_metadata, mode


def _quantize_feature(value: Any, input_metadata: dict[str, Any]) -> Any:
    import numpy as np

    array = np.asarray(value, dtype=np.float32)
    if array.shape != INPUT_SHAPE or not np.all(np.isfinite(array)):
        raise PreparationError("selected TRAIN feature is non-finite or has the wrong shape")
    scale = float(input_metadata["scale"])
    zero_point = int(input_metadata["zero_point"])
    quantized = np.clip(np.rint(array / scale) + zero_point, -128, 127).astype(np.int8)
    if quantized.shape != INPUT_SHAPE or quantized.nbytes != INPUT_BYTES:
        raise PreparationError("quantized TRAIN feature does not match the firmware input contract")
    return quantized


def _run_host_runner(
    runner: Path,
    model: Path,
    input_path: Path,
    output_path: Path,
    expected: Any,
) -> dict[str, Any]:
    import numpy as np

    if runner.is_symlink() or not runner.is_file() or not os.access(runner, os.X_OK):
        raise PreparationError(f"host TFLM runner must be an executable regular file: {runner}")
    command = [str(runner), str(model), str(input_path), str(output_path), str(ARENA_BYTES)]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PreparationError(f"host TFLM runner failed to execute: {exc}") from exc
    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise PreparationError(
            f"host TFLM runner did not return JSON: {completed.stdout[:1000]!r}"
        ) from exc
    if completed.returncode != 0 or not isinstance(report, dict) or report.get("status") != "PASS":
        raise PreparationError(
            f"host TFLM runner rejected real TRAIN fixture: rc={completed.returncode}, "
            f"report={report!r}, stderr={completed.stderr[:1000]!r}"
        )
    if not output_path.is_file() or output_path.stat().st_size != OUTPUT_BYTES:
        raise PreparationError("host TFLM runner did not produce exactly 192 output bytes")
    actual = np.frombuffer(output_path.read_bytes(), dtype=np.int8)
    reference = np.asarray(expected, dtype=np.int8).reshape(-1)
    maximum = int(np.max(np.abs(actual.astype(np.int16) - reference.astype(np.int16))))
    if maximum > LSB_CUTOFF:
        raise PreparationError(
            f"host TFLM parity exceeded the {LSB_CUTOFF}-LSB cutoff: {maximum}"
        )
    arena_used = report.get("arena_used_bytes")
    if not isinstance(arena_used, int) or not 0 < arena_used <= ARENA_BYTES:
        raise PreparationError(f"host TFLM runner returned an invalid arena usage: {report!r}")
    return {
        "status": "PASS",
        "max_absdiff_lsb": maximum,
        "arena_used_bytes": arena_used,
        "runner_sha256": _sha256_file(runner),
        "runner_report": report,
    }


def _dataset_provenance(dataset: Any) -> dict[str, Any]:
    root = Path(dataset.root)
    manifest_path = root / "manifest.json"
    provenance_path = root / "provenance.json"
    manifest = dataset.manifest
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("status") != "PREPARED_OFFICIAL_SOURCE":
        raise PreparationError("the source dataset is not marked as official prepared data")
    arrays = manifest.get("arrays")
    if not isinstance(arrays, dict) or "features" not in arrays:
        raise PreparationError("the source dataset has no bound features artifact")
    return {
        "root": str(root),
        "status": provenance["status"],
        "manifest_sha256": dataset.manifest_sha256,
        "manifest_bytes": manifest_path.stat().st_size,
        "provenance_sha256": _sha256_file(provenance_path),
        "rows_sha256": manifest["rows_sha256"],
        "features": arrays["features"],
        "row_count": manifest["row_count"],
        "split": manifest["split"],
    }


def prepare(
    *,
    model_path: Path = DEFAULT_MODEL,
    dataset_root: Path = DEFAULT_DATASET,
    host_runner: Path = DEFAULT_HOST_RUNNER,
    output_dir: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    """Generate the model/vectors and return the immutable manifest record."""

    model_source = _regular_source(model_path, "model")
    if model_source.stat().st_size != MODEL_BYTES:
        raise PreparationError(
            f"model size mismatch: expected {MODEL_BYTES}, got {model_source.stat().st_size}"
        )
    model_source_sha256 = _sha256_file(model_source)
    if model_source_sha256 != MODEL_SHA256:
        raise PreparationError(
            f"model SHA-256 mismatch: expected {MODEL_SHA256}, got {model_source_sha256}"
        )

    output = _prepare_output_dir(output_dir)
    try:
        dataset = _load_full_train_dataset(dataset_root)
        rows = dataset.rows
        for index, expected_window_id in REAL_TRAIN_FIXTURES:
            if index < 0 or index >= len(rows):
                raise PreparationError(f"fixture row index is outside the TRAIN dataset: {index}")
            row = rows[index]
            if row.get("split") != "train" or row.get("window_id") != expected_window_id:
                raise PreparationError(
                    f"fixture binding mismatch at row {index}: expected TRAIN window "
                    f"{expected_window_id!r}, got {row.get('split')!r}/{row.get('window_id')!r}"
                )

        model_bytes = model_source.read_bytes()
        interpreter, input_metadata, output_metadata, reference_mode = _reference_interpreter(model_bytes)
        model_output = output / "encoder_int8.tflite"
        model_binding = _write_bytes(model_output, model_bytes)

        fixture_records: list[dict[str, Any]] = []
        with tempfile.TemporaryDirectory(prefix=".host-parity-", dir=str(output)) as temporary:
            temporary_root = Path(temporary)
            for fixture_number, (index, expected_window_id) in enumerate(REAL_TRAIN_FIXTURES):
                import numpy as np

                quantized = _quantize_feature(dataset.features[index : index + 1], input_metadata)
                interpreter.set_tensor(input_metadata["index"], quantized)
                interpreter.invoke()
                expected = np.asarray(
                    interpreter.get_tensor(output_metadata["index"]), dtype=np.int8
                ).reshape(-1)
                if expected.shape != OUTPUT_SHAPE[1:] or expected.nbytes != OUTPUT_BYTES:
                    raise PreparationError("desktop reference returned an unexpected output vector")

                input_output = output / f"real_train_{fixture_number:03d}_input.int8"
                expected_output = output / f"real_train_{fixture_number:03d}_expected.int8"
                input_binding = _write_bytes(input_output, quantized.tobytes())
                expected_binding = _write_bytes(expected_output, expected.tobytes())
                host_output = temporary_root / f"real_train_{fixture_number:03d}_output.int8"
                host_parity = _run_host_runner(
                    _regular_source(host_runner, "host runner"),
                    model_output,
                    input_output,
                    host_output,
                    expected,
                )
                fixture_records.append(
                    {
                        "fixture_number": fixture_number,
                        "row_index": index,
                        "window_id": expected_window_id,
                        "split": "train",
                        "input_file": input_output.name,
                        "expected_file": expected_output.name,
                        "input": input_binding,
                        "expected": expected_binding,
                        "host_parity": host_parity,
                    }
                )

        script_binding = _file_binding(Path(__file__).resolve())
        host_runner_path = _regular_source(host_runner, "host runner")
        host_manifest_path = Path(f"{host_runner_path}.build.json")
        host_manifest_binding = _file_binding(host_manifest_path) if host_manifest_path.is_file() else None
        manifest: dict[str, Any] = {
            "format_version": 1,
            "scope": "BOARD_ENCODER_SMOKE_ONLY",
            "status": "PREPARED_REAL_TRAIN_HOST_PARITY_PASS",
            "model": {
                "source": str(model_source),
                "expected_sha256": MODEL_SHA256,
                "expected_bytes": MODEL_BYTES,
                "source_binding": {"bytes": MODEL_BYTES, "sha256": model_source_sha256},
                "embedded_file": model_output.name,
                "embedded_binding": model_binding,
            },
            "dataset": _dataset_provenance(dataset),
            "reference": {
                "mode": reference_mode,
                "input": input_metadata,
                "output": output_metadata,
                "max_absdiff_lsb_cutoff": LSB_CUTOFF,
            },
            "host_runner": {
                "path": str(host_runner_path),
                "binding": _file_binding(host_runner_path),
                "build_manifest": host_manifest_binding,
                "arena_capacity_bytes": ARENA_BYTES,
            },
            "fixtures": fixture_records,
            "generator": {
                "script": str(Path(__file__).resolve()),
                "script_binding": script_binding,
                "regeneration_policy": "REFUSE_EXISTING_OUTPUT_DIRECTORY",
                "generated_only_under_output_dir": True,
            },
            "contract": {
                "input_dtype": "int8",
                "input_shape": list(INPUT_SHAPE),
                "output_dtype": "int8",
                "output_shape": list(OUTPUT_SHAPE),
                "fixture_policy": "REAL_TRAIN_ONLY_NO_CALIBRATION_VALIDATION_OR_TEST_INFERENCE",
                "runtime_note": "ESP-NN board runtime may differ from the desktop reference and host TFLM runner.",
            },
        }
        _write_json(output / "manifest.json", manifest)
        return manifest
    except Exception:
        # Do not delete a partially generated tree: retaining it makes a
        # failed run visible and the next run will refuse to overwrite it.
        raise


def _main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--host-runner", type=Path, default=DEFAULT_HOST_RUNNER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        result = prepare(
            model_path=args.model,
            dataset_root=args.dataset_root,
            host_runner=args.host_runner,
            output_dir=args.output_dir,
        )
    except PreparationError as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a CLI process.
    raise SystemExit(_main())
