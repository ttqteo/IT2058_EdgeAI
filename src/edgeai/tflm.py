"""Bounded host-only checks for the pinned TFLite Micro runner.

This module deliberately keeps the desktop TFLite interpreter as an
independent comparison oracle.  A desktop result is never substituted for a
missing or failing TFLM runner, and no result here establishes board support
or model accuracy.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any

import numpy as np

from .budget import (
    CONTRACT_UNRESOLVED,
    HOST_SCOPE,
    STATIC_FAIL,
    RUNTIME_UNVERIFIED,
    STATIC_PASS,
    check_model,
)

TFLM_SOURCE_COMMIT = "230c9d9531bd9b08d445b4f2ed0760756ffd5f78"
# Compatibility alias for callers that imported the old constant.  Reports
# deliberately use the unambiguous ``*_git_commit`` names below.
TFLM_SOURCE_SHA = TFLM_SOURCE_COMMIT
MIN_ARENA_BYTES = 8192
MAX_ARENA_BYTES = 131072
DEFAULT_ARENA_BYTES = MAX_ARENA_BYTES
INPUT_SHAPE = (1, 298, 40, 1)
OUTPUT_SHAPE = (1, 192)
INPUT_BYTES = 11920
OUTPUT_BYTES = 192
LSB_CUTOFF = 1
RUNNER_TIMEOUT_SECONDS = 30
GIT_CHECK_TIMEOUT_SECONDS = 10
BUILD_MANIFEST_FORMAT_VERSION = 1
BUILD_POLICY = "FORCED_REBUILD_PINNED_SOURCE"
_HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")
_MANIFEST_KEYS = (
    "format_version",
    "tflm_git_commit",
    "tflm_checkout",
    "tflm_library",
    "tflm_library_sha256",
    "runner_source_sha256",
    "runner_makefile_sha256",
    "build_script_sha256",
    "runner_binary_sha256",
    "compiler_version",
    "compile_flags",
    "build_policy",
)


def _default_runner() -> Path:
    root = Path(__file__).resolve().parents[2]
    return root / "artifacts" / "tflm-host" / "edgeai_tflm_host"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _manifest_path(runner_path: Path) -> Path:
    return Path(f"{runner_path}.build.json")


def _sha256(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    except (OSError, ValueError):
        return None


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _empty_build_provenance(runner_path: Path) -> dict[str, Any]:
    manifest_path = _manifest_path(runner_path)
    try:
        manifest_exists = manifest_path.is_file()
    except OSError:
        manifest_exists = False
    return {
        "status": "NOT_VERIFIED",
        "manifest_path": str(manifest_path),
        "manifest_exists": manifest_exists,
        "expected_tflm_git_commit": TFLM_SOURCE_COMMIT,
        "observed_tflm_git_commit": None,
        "runner_reported_tflm_git_commit": None,
        "expected_hashes": {},
        "observed_hashes": {},
        "errors": [],
    }


def _base_report(
    budget_report: dict[str, Any],
    model_path: Path,
    runner_path: Path,
    arena_bytes: int,
    model_sha256: str | None,
) -> dict[str, Any]:
    try:
        runner_exists = runner_path.is_file()
        runner_executable = runner_exists and runner_path.stat().st_mode & 0o111 != 0
    except OSError:
        runner_exists = False
        runner_executable = False
    runner_sha256 = _sha256(runner_path) if runner_exists else None
    return {
        "status": "TFLM_HOST_FAIL",
        "host_runtime_status": "NOT_RUN",
        "runtime_status": RUNTIME_UNVERIFIED,
        "contract_status": CONTRACT_UNRESOLVED,
        "scope": HOST_SCOPE,
        "model_training_status": "NOT_ESTABLISHED",
        "input_status": "SYNTHETIC_INPUT",
        "model": budget_report.get("model", {"path": str(model_path)}),
        "model_sha256": model_sha256,
        "expected_tflm_git_commit": TFLM_SOURCE_COMMIT,
        "binary_sha256": runner_sha256,
        "arena_capacity_bytes": arena_bytes,
        "build_provenance": _empty_build_provenance(runner_path),
        "runner": {
            "path": str(runner_path),
            "exists": runner_exists,
            "executable": runner_executable,
            "sha256": runner_sha256,
        },
        "budget": budget_report,
        "fixtures": [],
        "errors": [],
    }


def _failure(report: dict[str, Any], message: str, *, status: str | None = None) -> dict[str, Any]:
    report["status"] = status or "TFLM_HOST_FAIL"
    report["host_runtime_status"] = "FAIL"
    report["errors"].append(message)
    return _json_safe(report)


def _metadata(detail: dict[str, Any], role: str) -> dict[str, Any]:
    dtype = np.dtype(detail["dtype"])
    shape = tuple(int(item) for item in np.asarray(detail["shape"]).tolist())
    parameters = detail.get("quantization_parameters", {})
    scales = np.asarray(parameters.get("scales", []), dtype=np.float64)
    zero_points = np.asarray(parameters.get("zero_points", []), dtype=np.int64)
    if dtype != np.dtype(np.int8):
        raise ValueError(f"{role} dtype is {dtype.name}, expected int8")
    if role == "input" and shape != INPUT_SHAPE:
        raise ValueError(f"input shape is {shape}, expected {INPUT_SHAPE}")
    if role == "output" and shape != OUTPUT_SHAPE:
        raise ValueError(f"output shape is {shape}, expected {OUTPUT_SHAPE}")
    if scales.size != 1 or zero_points.size != 1:
        raise ValueError(f"{role} must expose one per-tensor quantization pair")
    scale = float(scales[0])
    zero_point = int(zero_points[0])
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"{role} quantization scale must be finite and positive")
    if zero_point < -128 or zero_point > 127:
        raise ValueError(f"{role} quantization zero point is outside int8 range")
    return {
        "index": int(detail["index"]),
        "dtype": dtype.name,
        "shape": [int(item) for item in shape],
        "scale": scale,
        "zero_point": zero_point,
    }


def _reference_interpreter(model_bytes: bytes) -> tuple[Any, dict[str, Any], dict[str, Any], str]:
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(f"TensorFlow desktop reference unavailable: {exc}") from exc

    kwargs: dict[str, Any] = {
        "model_content": model_bytes,
        "experimental_delegates": [],
    }
    resolver_type = getattr(getattr(tf.lite, "experimental", None), "OpResolverType", None)
    builtin_ref = getattr(resolver_type, "BUILTIN_REF", None)
    if builtin_ref is None:
        raise RuntimeError(
            "desktop reference unavailable: TensorFlow does not expose "
            "OpResolverType.BUILTIN_REF"
        )
    kwargs["experimental_op_resolver_type"] = builtin_ref
    mode = "BUILTIN_REF_NO_DELEGATES"
    try:
        interpreter = tf.lite.Interpreter(**kwargs)
    except Exception as exc:  # pragma: no cover - package dependent
        raise RuntimeError(
            "desktop BUILTIN_REF reference interpreter setup failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    try:
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        if len(input_details) != 1 or len(output_details) != 1:
            raise ValueError("desktop reference must expose exactly one input and output")
        input_metadata = _metadata(input_details[0], "input")
        output_metadata = _metadata(output_details[0], "output")
    except Exception:
        del interpreter
        raise
    return interpreter, input_metadata, output_metadata, mode


def _fixtures(input_zero_point: int) -> list[tuple[str, np.ndarray]]:
    total = int(np.prod(INPUT_SHAPE))
    zero_quant_point = np.full(INPUT_SHAPE, input_zero_point, dtype=np.int8)
    rng = np.random.default_rng(2058)
    deterministic_random = rng.integers(-128, 128, size=total, dtype=np.int16).astype(np.int8)
    edge_values = np.array([-128, -127, -1, 0, 1, 126, 127], dtype=np.int8)
    deterministic_edge = np.resize(edge_values, total).reshape(INPUT_SHAPE)
    return [
        ("zero_quant_point", zero_quant_point),
        ("deterministic_random", deterministic_random.reshape(INPUT_SHAPE)),
        ("int8_edges", deterministic_edge),
    ]


def _clip_diagnostic(value: str, limit: int = 2000) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "...<truncated>"


def _run_runner(
    runner_path: Path,
    model_path: Path,
    input_path: Path,
    output_path: Path,
    arena_bytes: int,
) -> tuple[int, dict[str, Any] | None, str, str]:
    command = [
        str(runner_path),
        str(model_path),
        str(input_path),
        str(output_path),
        str(arena_bytes),
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=RUNNER_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = _clip_diagnostic((exc.stdout or "") if isinstance(exc.stdout, str) else "")
        stderr = _clip_diagnostic((exc.stderr or "") if isinstance(exc.stderr, str) else "")
        return 124, None, stdout, stderr + "runner timed out after 30 seconds"
    except OSError as exc:
        return 127, None, "", f"could not start TFLM runner: {type(exc).__name__}: {exc}"
    stdout = _clip_diagnostic(completed.stdout)
    stderr = _clip_diagnostic(completed.stderr)
    parsed: dict[str, Any] | None = None
    try:
        candidate = json.loads(completed.stdout)
        if isinstance(candidate, dict):
            parsed = candidate
    except json.JSONDecodeError:
        parsed = None
    return completed.returncode, parsed, stdout, stderr


def _git_command(checkout: Path, *arguments: str) -> tuple[str | None, str | None]:
    """Run a read-only git query and return (stdout, error)."""

    try:
        completed = subprocess.run(
            ["git", "-C", str(checkout), *arguments],
            capture_output=True,
            text=True,
            timeout=GIT_CHECK_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None, f"git {' '.join(arguments)} timed out"
    except OSError as exc:
        return None, f"could not run git: {type(exc).__name__}: {exc}"
    if completed.returncode != 0:
        diagnostic = _clip_diagnostic((completed.stderr or completed.stdout).strip())
        return None, (
            f"git {' '.join(arguments)} failed with exit {completed.returncode}"
            + (f": {diagnostic}" if diagnostic else "")
        )
    return completed.stdout.strip(), None


def _verify_build_manifest(
    runner_path: Path, runner_sha256: str | None
) -> dict[str, Any]:
    """Verify the local build-bound sidecar without treating it as attestation."""

    result = _empty_build_provenance(runner_path)
    manifest_path = _manifest_path(runner_path)
    errors: list[str] = []
    expected_hashes: dict[str, str] = {}
    observed_hashes: dict[str, str | None] = {}
    result["expected_hashes"] = expected_hashes
    result["observed_hashes"] = observed_hashes

    try:
        manifest_text = manifest_path.read_text(encoding="utf-8")
    except OSError as exc:
        result["status"] = "FAIL"
        result["errors"] = [
            f"cannot read build manifest: {type(exc).__name__}: {exc}"
        ]
        return result
    try:
        manifest_value = json.loads(manifest_text)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        result["status"] = "FAIL"
        result["errors"] = [f"build manifest is not valid JSON: {exc}"]
        return result
    if not isinstance(manifest_value, dict):
        result["status"] = "FAIL"
        result["errors"] = ["build manifest root must be a JSON object"]
        return result

    manifest = manifest_value
    result["manifest"] = {
        key: manifest[key] for key in _MANIFEST_KEYS if key in manifest
    }
    missing = [key for key in _MANIFEST_KEYS if key not in manifest]
    if missing:
        errors.append("build manifest is missing required keys: " + ", ".join(missing))

    if type(manifest.get("format_version")) is not int or manifest.get("format_version") != BUILD_MANIFEST_FORMAT_VERSION:
        errors.append("build manifest format_version is not 1")

    string_keys = (
        "tflm_git_commit",
        "tflm_checkout",
        "tflm_library",
        "compiler_version",
        "compile_flags",
        "build_policy",
    )
    for key in string_keys:
        value = manifest.get(key)
        if not isinstance(value, str) or not value:
            errors.append(f"build manifest {key} must be a non-empty string")

    commit = manifest.get("tflm_git_commit")
    if isinstance(commit, str):
        result["expected_tflm_git_commit"] = TFLM_SOURCE_COMMIT
        if commit != TFLM_SOURCE_COMMIT:
            errors.append("build manifest TFLM commit does not match the pinned source")

    if manifest.get("build_policy") != BUILD_POLICY:
        errors.append(f"build manifest build_policy must be {BUILD_POLICY}")

    path_values: dict[str, Path] = {}
    for key in ("tflm_checkout", "tflm_library"):
        value = manifest.get(key)
        if isinstance(value, str):
            try:
                candidate = Path(value)
            except (TypeError, ValueError):
                candidate = None
            if candidate is None or not candidate.is_absolute():
                errors.append(f"build manifest {key} must be an absolute path")
            else:
                path_values[key] = candidate
    result["tflm_checkout"] = str(path_values["tflm_checkout"]) if "tflm_checkout" in path_values else None
    result["tflm_library"] = str(path_values["tflm_library"]) if "tflm_library" in path_values else None

    hash_keys = (
        "tflm_library_sha256",
        "runner_source_sha256",
        "runner_makefile_sha256",
        "build_script_sha256",
        "runner_binary_sha256",
    )
    for key in hash_keys:
        value = manifest.get(key)
        if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
            errors.append(f"build manifest {key} must be a 64-digit hexadecimal hash")
        else:
            expected_hashes[key] = value.lower()

    checkout = path_values.get("tflm_checkout")
    observed_commit: str | None = None
    if checkout is not None:
        head, head_error = _git_command(checkout, "rev-parse", "HEAD")
        if head_error is not None:
            errors.append(f"TFLM checkout verification failed: {head_error}")
        else:
            observed_commit = head
            if head != TFLM_SOURCE_COMMIT:
                errors.append("TFLM checkout HEAD does not match the pinned source")
        clean_status, clean_error = _git_command(
            checkout, "status", "--porcelain", "--untracked-files=all"
        )
        if clean_error is not None:
            errors.append(f"TFLM checkout cleanliness check failed: {clean_error}")
        elif clean_status:
            errors.append("TFLM checkout is dirty or contains untracked files")
    result["observed_tflm_git_commit"] = observed_commit

    canonical_paths = {
        "runner_source_sha256": _repo_root() / "firmware" / "tflm_host" / "runner.cc",
        "runner_makefile_sha256": _repo_root() / "firmware" / "tflm_host" / "Makefile",
        "build_script_sha256": _repo_root() / "scripts" / "build_tflm_host.sh",
    }
    hash_paths: dict[str, Path] = dict(canonical_paths)
    if "tflm_library" in path_values:
        hash_paths["tflm_library_sha256"] = path_values["tflm_library"]
    hash_paths["runner_binary_sha256"] = runner_path
    for key, path in hash_paths.items():
        observed = _sha256(path)
        observed_hashes[key] = observed
        expected = expected_hashes.get(key)
        if observed is None:
            errors.append(f"cannot hash artifact for manifest field {key}: {path}")
        elif expected is None or observed != expected:
            errors.append(f"manifest hash mismatch for {key}")

    # Keep the caller's initial observation visible, but never use it as proof
    # when the binary changed while the manifest was being checked.
    if runner_sha256 is not None and observed_hashes.get("runner_binary_sha256") != runner_sha256:
        errors.append("runner binary changed during provenance verification")

    result["errors"] = errors
    result["status"] = "BUILD_BOUND" if not errors else "FAIL"
    return result


def _float32_equal(left: Any, right: Any) -> bool:
    try:
        left_value = np.float32(float(left))
        right_value = np.float32(float(right))
    except (TypeError, ValueError, OverflowError):
        return False
    return bool(np.isfinite(left_value) and np.isfinite(right_value) and left_value == right_value)


def _validate_runner_report(
    runner_report: dict[str, Any],
    *,
    budget_report: dict[str, Any],
    input_metadata: dict[str, Any],
    output_metadata: dict[str, Any],
    model_bytes: bytes,
    arena_bytes: int,
) -> list[str]:
    """Validate the C peer protocol before comparing output bytes."""

    errors: list[str] = []
    if runner_report.get("status") != "PASS":
        errors.append("runner report status is not PASS")
    if runner_report.get("scope") != HOST_SCOPE:
        errors.append("runner report scope is not EXPERIMENTAL_HOST_ONLY")
    reported_commit = runner_report.get("tflm_source_sha")
    if reported_commit != TFLM_SOURCE_COMMIT:
        errors.append("runner-reported TFLM Git commit does not match the pinned source")

    schema = budget_report.get("schema")
    schema_version = schema.get("model_version") if isinstance(schema, dict) else None
    if type(schema_version) is not int or runner_report.get("schema_version") != schema_version:
        errors.append("runner schema version does not match the checked model")
    if type(runner_report.get("model_bytes")) is not int or runner_report.get("model_bytes") != len(model_bytes):
        errors.append("runner model byte count does not match the checked snapshot")
    if type(runner_report.get("input_bytes")) is not int or runner_report.get("input_bytes") != INPUT_BYTES:
        errors.append("runner input byte count is not 11920")
    if type(runner_report.get("output_bytes")) is not int or runner_report.get("output_bytes") != OUTPUT_BYTES:
        errors.append("runner output byte count is not 192")

    capacity = runner_report.get("arena_capacity_bytes")
    if type(capacity) is not int or capacity != arena_bytes:
        errors.append("runner arena capacity does not match the requested capacity")
    used = runner_report.get("arena_used_bytes")
    if type(used) is not int or not 0 < used <= arena_bytes:
        errors.append("runner arena usage must satisfy 0 < used <= capacity")

    expected_metadata = {
        "input": input_metadata,
        "output": output_metadata,
    }
    for role, expected in expected_metadata.items():
        actual = runner_report.get(role)
        if not isinstance(actual, dict):
            errors.append(f"runner {role} metadata is missing")
            continue
        if actual.get("dtype") != expected["dtype"]:
            errors.append(f"runner {role} dtype does not match the checked model")
        if actual.get("shape") != expected["shape"]:
            errors.append(f"runner {role} shape does not match the checked model")
        if type(actual.get("zero_point")) is not int or actual.get("zero_point") != expected["zero_point"]:
            errors.append(f"runner {role} zero point does not match the checked model")
        actual_scale = actual.get("scale")
        if (
            isinstance(actual_scale, bool)
            or not isinstance(actual_scale, (int, float))
            or not _float32_equal(actual_scale, expected["scale"])
        ):
            errors.append(f"runner {role} scale does not match the checked model")
    return errors


def _minimal_budget_failure(model_path: Path, message: str) -> dict[str, Any]:
    """Build a structured static failure when the independent checker cannot run."""

    error = {"check": "model_snapshot", "status": "FAIL", "message": message}
    return {
        "status": STATIC_FAIL,
        "runtime_status": RUNTIME_UNVERIFIED,
        "contract_status": CONTRACT_UNRESOLVED,
        "scope": HOST_SCOPE,
        "model": {
            "path": str(model_path),
            "exists": False,
            "bytes": None,
            "size_bytes": None,
            "sha256": None,
        },
        "checks": [{"name": "model_snapshot", **error}],
        "errors": [error],
        "macs": {"status": "UNRESOLVED", "total": None, "total_macs": None},
        "static_structure_pass": False,
        "static_budget_pass": False,
    }


def _mark_budget_failure(
    budget_report: dict[str, Any], message: str, *, check: str = "model_snapshot"
) -> dict[str, Any]:
    """Force a report into a failed state without discarding independent details."""

    budget_report["status"] = STATIC_FAIL
    budget_report["static_budget_pass"] = False
    budget_report.setdefault("checks", []).append(
        {"name": check, "status": "FAIL", "message": message}
    )
    budget_report.setdefault("errors", []).append(
        {"check": check, "status": "FAIL", "message": message}
    )
    return budget_report


def check_tflm(
    model_path: str | Path,
    *,
    runner_path: str | Path | None = None,
    arena_bytes: int = DEFAULT_ARENA_BYTES,
) -> dict[str, Any]:
    """Run bounded TFLM host inference and compare it with desktop TFLite.

    The initial gate is the stable :func:`edgeai.budget.check_model` report.
    A successful desktop comparison without a successful TFLM subprocess is
    always reported as failure.
    """

    if isinstance(arena_bytes, bool) or not isinstance(arena_bytes, int):
        raise ValueError("arena_bytes must be an integer")
    if arena_bytes < MIN_ARENA_BYTES or arena_bytes > MAX_ARENA_BYTES:
        raise ValueError("arena_bytes must be within the inclusive 8192..131072-byte host bound")
    try:
        model = Path(model_path)
    except TypeError as exc:
        raise ValueError("model_path must be path-like") from exc
    try:
        runner = _default_runner() if runner_path is None else Path(runner_path)
    except TypeError as exc:
        raise ValueError("runner_path must be path-like") from exc

    # Read once and immediately snapshot.  The static checker, desktop
    # reference, and peer process all consume this byte sequence, so a later
    # replacement of the caller's path cannot bypass the static gate.
    try:
        model_bytes = model.read_bytes()
    except OSError as exc:
        budget_report = _minimal_budget_failure(
            model, f"cannot snapshot model: {type(exc).__name__}: {exc}"
        )
        report = _base_report(budget_report, model, runner, arena_bytes, None)
        report["model_snapshot"] = {"status": "FAIL", "bytes": None, "sha256": None}
        return _failure(report, "cannot snapshot model", status="TFLM_STATIC_FAIL")

    model_sha256 = hashlib.sha256(model_bytes).hexdigest()
    report: dict[str, Any] | None = None
    try:
        with tempfile.TemporaryDirectory(prefix="edgeai-tflm-host-") as temporary_directory:
            snapshot_path = Path(temporary_directory) / "model.tflite"
            try:
                snapshot_path.write_bytes(model_bytes)
            except OSError as exc:
                budget_report = _minimal_budget_failure(
                    model, f"cannot write model snapshot: {type(exc).__name__}: {exc}"
                )
                report = _base_report(budget_report, model, runner, arena_bytes, model_sha256)
                report["model_snapshot"] = {
                    "status": "FAIL",
                    "bytes": len(model_bytes),
                    "sha256": model_sha256,
                }
                return _failure(report, "cannot write model snapshot", status="TFLM_STATIC_FAIL")

            try:
                budget_report = check_model(snapshot_path)
            except Exception as exc:  # independent checker must not turn into a pass
                budget_report = _minimal_budget_failure(
                    model, f"static model checker failed: {type(exc).__name__}: {exc}"
                )
            if not isinstance(budget_report, dict):
                budget_report = _minimal_budget_failure(
                    model, "static model checker returned a non-object report"
                )

            snapshot_model = budget_report.get("model")
            observed_snapshot_sha256 = (
                snapshot_model.get("sha256")
                if isinstance(snapshot_model, dict)
                else None
            )
            if observed_snapshot_sha256 != model_sha256:
                _mark_budget_failure(
                    budget_report,
                    "independent static report hash does not match the model snapshot",
                    check="model_snapshot_hash",
                )

            report = _base_report(budget_report, model, runner, arena_bytes, model_sha256)
            report["model_snapshot"] = {
                "status": "BOUND",
                "bytes": len(model_bytes),
                "sha256": model_sha256,
            }
            model_report = report.get("model")
            if not isinstance(model_report, dict):
                model_report = {"path": str(model)}
                report["model"] = model_report
            model_report["path"] = str(model)
            model_report["bytes"] = len(model_bytes)
            model_report["size_bytes"] = len(model_bytes)
            model_report["sha256"] = model_sha256

            if (
                budget_report.get("status") != STATIC_PASS
                or not budget_report.get("static_budget_pass", False)
            ):
                return _failure(
                    report,
                    "static budget/model gate did not pass",
                    status="TFLM_STATIC_FAIL",
                )

            try:
                runner_available = runner.is_file() and runner.stat().st_mode & 0o111 != 0
            except OSError as exc:
                runner_available = False
                report["runner"]["probe_error"] = f"{type(exc).__name__}: {exc}"
            if not runner_available:
                report["build_provenance"]["status"] = "UNAVAILABLE"
                report["build_provenance"]["errors"] = [
                    "TFLM runner binary is missing or not executable"
                ]
                return _failure(
                    report,
                    "TFLM runner binary is missing or not executable",
                    status="TFLM_RUNNER_UNAVAILABLE",
                )

            try:
                provenance = _verify_build_manifest(
                    runner, report.get("binary_sha256")
                )
            except Exception as exc:  # provenance failure is never a runtime pass
                provenance = _empty_build_provenance(runner)
                provenance["status"] = "FAIL"
                provenance["errors"] = [
                    f"build manifest verification failed: {type(exc).__name__}: {exc}"
                ]
            report["build_provenance"] = provenance
            if provenance.get("status") != "BUILD_BOUND":
                return _failure(
                    report,
                    "local TFLM build manifest verification failed",
                    status="TFLM_HOST_FAIL",
                )
            observed_hashes = provenance.get("observed_hashes")
            verified_runner_sha256 = (
                observed_hashes.get("runner_binary_sha256")
                if isinstance(observed_hashes, dict)
                else None
            )

            try:
                interpreter, input_metadata, output_metadata, reference_mode = _reference_interpreter(
                    model_bytes
                )
            except Exception as exc:  # includes missing/unsupported TF APIs
                return _failure(
                    report,
                    f"desktop reference unavailable: {type(exc).__name__}: {exc}",
                    status="TFLM_REFERENCE_UNAVAILABLE",
                )
            report["reference"] = {
                "mode": reference_mode,
                "input": input_metadata,
                "output": output_metadata,
            }

            all_passed = True
            temporary_root = Path(temporary_directory)
            for fixture_name, fixture in _fixtures(input_metadata["zero_point"]):
                input_path = temporary_root / f"{fixture_name}.input.int8"
                output_path = temporary_root / f"{fixture_name}.output.int8"
                fixture_report: dict[str, Any] = {
                    "name": fixture_name,
                    "runner_returncode": None,
                    "runner_report": None,
                    "runner_stdout": "",
                    "runner_stderr": "",
                    "status": "FAIL",
                }
                try:
                    input_bytes = np.asarray(fixture, dtype=np.int8).tobytes()
                    input_path.write_bytes(input_bytes)
                    fixture_report["input_bytes"] = len(input_bytes)
                    fixture_report["input_sha256"] = hashlib.sha256(input_bytes).hexdigest()
                except OSError as exc:
                    fixture_report["comparison_error"] = (
                        f"could not write runner input: {type(exc).__name__}: {exc}"
                    )
                    all_passed = False
                    report["fixtures"].append(fixture_report)
                    continue

                if _sha256(snapshot_path) != model_sha256:
                    fixture_report["comparison_error"] = (
                        "model snapshot changed after the static gate"
                    )
                    all_passed = False
                    report["fixtures"].append(fixture_report)
                    continue
                if _sha256(runner) != verified_runner_sha256:
                    fixture_report["comparison_error"] = (
                        "runner binary changed after provenance verification"
                    )
                    all_passed = False
                    report["fixtures"].append(fixture_report)
                    continue

                try:
                    returncode, runner_report, stdout, stderr = _run_runner(
                        runner, snapshot_path, input_path, output_path, arena_bytes
                    )
                except Exception as exc:  # subprocess/process failures fail closed
                    returncode, runner_report, stdout, stderr = (
                        127,
                        None,
                        "",
                        f"could not run TFLM runner: {type(exc).__name__}: {exc}",
                    )
                fixture_report.update(
                    {
                        "runner_returncode": int(returncode),
                        "runner_report": runner_report,
                        "runner_stdout": stdout,
                        "runner_stderr": stderr,
                    }
                )
                if isinstance(runner_report, dict):
                    report["build_provenance"]["runner_reported_tflm_git_commit"] = (
                        runner_report.get("tflm_source_sha")
                    )

                artifact_binding_errors: list[str] = []
                if _sha256(snapshot_path) != model_sha256:
                    artifact_binding_errors.append(
                        "model snapshot changed during runner invocation"
                    )
                if _sha256(runner) != verified_runner_sha256:
                    artifact_binding_errors.append(
                        "runner binary changed during runner invocation"
                    )
                if artifact_binding_errors:
                    fixture_report["comparison_error"] = "; ".join(artifact_binding_errors)
                elif returncode != 0 or runner_report is None:
                    fixture_report["comparison_error"] = (
                        "TFLM runner did not return a successful JSON report"
                    )
                else:
                    protocol_errors = _validate_runner_report(
                        runner_report,
                        budget_report=budget_report,
                        input_metadata=input_metadata,
                        output_metadata=output_metadata,
                        model_bytes=model_bytes,
                        arena_bytes=arena_bytes,
                    )
                    if protocol_errors:
                        fixture_report["comparison_error"] = "; ".join(protocol_errors)
                    else:
                        try:
                            if not output_path.is_file() or output_path.stat().st_size != OUTPUT_BYTES:
                                raise ValueError("runner did not produce exactly 192 output bytes")
                            interpreter.set_tensor(input_metadata["index"], fixture)
                            interpreter.invoke()
                            reference_output = np.asarray(
                                interpreter.get_tensor(output_metadata["index"]), dtype=np.int8
                            ).reshape(OUTPUT_SHAPE)
                            tflm_output = np.frombuffer(
                                output_path.read_bytes(), dtype=np.int8
                            ).reshape(OUTPUT_SHAPE)
                            difference = tflm_output.astype(np.int16) - reference_output.astype(
                                np.int16
                            )
                            absolute = np.abs(difference)
                            max_abs_lsb = int(np.max(absolute))
                            rms_lsb = float(
                                np.sqrt(np.mean(np.square(difference, dtype=np.float64)))
                            )
                            mean_abs_lsb = float(np.mean(absolute, dtype=np.float64))
                            fixture_report.update(
                                {
                                    "max_abs_lsb": max_abs_lsb,
                                    "rms_lsb": rms_lsb,
                                    "mean_abs_lsb": mean_abs_lsb,
                                    "status": "PASS" if max_abs_lsb <= LSB_CUTOFF else "FAIL",
                                }
                            )
                            if fixture_report["status"] != "PASS":
                                fixture_report["comparison_error"] = (
                                    f"desktop/TFLM output differs by {max_abs_lsb} LSB"
                                )
                        except Exception as exc:
                            fixture_report["comparison_error"] = str(exc)
                if fixture_report["status"] != "PASS":
                    all_passed = False
                report["fixtures"].append(fixture_report)

            report["host_runtime_status"] = "PASS" if all_passed else "FAIL"
            report["status"] = "TFLM_HOST_PASS" if all_passed else "TFLM_HOST_FAIL"
            return _json_safe(report)
    except OSError as exc:
        if report is None:
            budget_report = _minimal_budget_failure(
                model, f"host temporary workspace failed: {type(exc).__name__}: {exc}"
            )
            report = _base_report(budget_report, model, runner, arena_bytes, model_sha256)
            report["model_snapshot"] = {
                "status": "FAIL",
                "bytes": len(model_bytes),
                "sha256": model_sha256,
            }
        return _failure(report, f"host temporary workspace failed: {exc}")


__all__ = [
    "DEFAULT_ARENA_BYTES",
    "INPUT_BYTES",
    "INPUT_SHAPE",
    "LSB_CUTOFF",
    "MAX_ARENA_BYTES",
    "MIN_ARENA_BYTES",
    "OUTPUT_BYTES",
    "OUTPUT_SHAPE",
    "TFLM_SOURCE_SHA",
    "check_tflm",
]
