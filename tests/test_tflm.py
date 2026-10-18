"""Self-contained checks for the bounded, host-only TFLM wrapper.

The smoke model is generated in pytest's temporary directory.  No test reads
or overwrites the repository's ignored model or runner artifacts.  Fake
executables are used only to exercise negative protocol/provenance paths; an
actual runner test is skipped until the C peer supplies its local build-bound
manifest.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat
import sys

import numpy as np
import pytest

import edgeai.tflm as tflm_module
from edgeai.budget import HOST_SCOPE, check_model
from edgeai.tflm import DEFAULT_ARENA_BYTES, OUTPUT_BYTES, check_tflm


@pytest.fixture(scope="session")
def smoke_artifacts(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Create the real synthetic INT8/FP32 smoke artifacts outside the repo."""

    pytest.importorskip("tensorflow")
    from edgeai.model import create_smoke_model

    output_dir = tmp_path_factory.mktemp("tflm-smoke-artifacts")
    create_smoke_model(output_dir)
    return {
        "int8": output_dir / "encoder_int8.tflite",
        "fp32": output_dir / "encoder_fp32.tflite",
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fake_runner(path: Path, report: dict[str, object] | None = None) -> Path:
    payload = json.dumps(report if report is not None else {"status": "REJECTED"})
    script = (
        f"#!{sys.executable}\n"
        "from pathlib import Path\n"
        "import json\n"
        "import sys\n"
        "Path(sys.argv[3]).write_bytes(bytes(192))\n"
        f"print({payload!r})\n"
    )
    path.write_text(script, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _manifest_payload(tmp_path: Path) -> dict[str, object]:
    zero_hash = "0" * 64
    return {
        "format_version": 1,
        "tflm_git_commit": tflm_module.TFLM_SOURCE_COMMIT,
        "tflm_checkout": str(tmp_path / "missing-tflm-checkout"),
        "tflm_library": str(tmp_path / "missing-tflm-library.a"),
        "tflm_library_sha256": zero_hash,
        "runner_source_sha256": zero_hash,
        "runner_makefile_sha256": zero_hash,
        "build_script_sha256": zero_hash,
        "runner_binary_sha256": zero_hash,
        "compiler_version": "test compiler",
        "compile_flags": "test flags",
        "build_policy": tflm_module.BUILD_POLICY,
    }


def _write_manifest(runner: Path, payload: dict[str, object]) -> Path:
    manifest_path = Path(f"{runner}.build.json")
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    return manifest_path


def _bound_manifest(runner: Path, runner_sha256: str | None) -> dict[str, object]:
    """Test-only provenance stub; it is never used as acceptance evidence."""

    return {
        "status": "BUILD_BOUND",
        "manifest_path": str(Path(f"{runner}.build.json")),
        "manifest_exists": True,
        "expected_tflm_git_commit": tflm_module.TFLM_SOURCE_COMMIT,
        "observed_tflm_git_commit": tflm_module.TFLM_SOURCE_COMMIT,
        "runner_reported_tflm_git_commit": None,
        "expected_hashes": {},
        "observed_hashes": {"runner_binary_sha256": runner_sha256},
        "errors": [],
    }


def _protocol_report(model: Path, *, source: str | None = None, used: int = 1) -> dict[str, object]:
    """Derive fake protocol metadata from the actual checked model for negatives."""

    budget_report = check_model(model)
    _, input_metadata, output_metadata, _ = tflm_module._reference_interpreter(
        model.read_bytes()
    )
    return {
        "status": "PASS",
        "scope": HOST_SCOPE,
        "tflm_source_sha": (
            tflm_module.TFLM_SOURCE_COMMIT if source is None else source
        ),
        "schema_version": budget_report["schema"]["model_version"],
        "model_bytes": model.stat().st_size,
        "input_bytes": 11920,
        "output_bytes": OUTPUT_BYTES,
        "arena_capacity_bytes": DEFAULT_ARENA_BYTES,
        "arena_used_bytes": used,
        "input": input_metadata,
        "output": output_metadata,
    }


def _require_runner() -> Path:
    root = Path(__file__).resolve().parents[1]
    runner = root / "artifacts" / "tflm-host" / "edgeai_tflm_host"
    if not runner.is_file() or runner.stat().st_mode & 0o111 == 0:
        pytest.skip("C TFLM host runner is unavailable; runtime verification is deferred")
    manifest = Path(f"{runner}.build.json")
    if not manifest.is_file():
        pytest.skip("C TFLM build manifest is unavailable; C handback is pending")
    return runner


def test_missing_runner_returns_structured_report(
    tmp_path: Path, smoke_artifacts: dict[str, Path]
) -> None:
    model = smoke_artifacts["int8"]
    report = check_tflm(model, runner_path=tmp_path / "missing-runner")

    json.dumps(report)
    assert report["status"] == "TFLM_RUNNER_UNAVAILABLE"
    assert report["host_runtime_status"] == "FAIL"
    assert report["runtime_status"] == "UNVERIFIED_HARDWARE"
    assert report["build_provenance"]["status"] == "UNAVAILABLE"
    assert report["model_snapshot"] == {
        "status": "BOUND",
        "bytes": model.stat().st_size,
        "sha256": _sha256(model),
    }
    assert "source_sha256" not in report


def test_static_gate_rejects_fp32_without_desktop_pass(
    tmp_path: Path, smoke_artifacts: dict[str, Path]
) -> None:
    report = check_tflm(smoke_artifacts["fp32"], runner_path=tmp_path / "missing-runner")

    json.dumps(report)
    assert report["status"] == "TFLM_STATIC_FAIL"
    assert report["host_runtime_status"] == "FAIL"
    assert report["fixtures"] == []
    assert report["model_snapshot"]["sha256"] == _sha256(smoke_artifacts["fp32"])


@pytest.mark.parametrize("kind", ["missing", "malformed"])
def test_invalid_model_is_a_structured_static_failure(
    tmp_path: Path, smoke_artifacts: dict[str, Path], kind: str
) -> None:
    if kind == "missing":
        model = tmp_path / "missing.tflite"
    else:
        model = tmp_path / "malformed.tflite"
        model.write_bytes(b"not-a-tflite-model")

    report = check_tflm(model, runner_path=tmp_path / "missing-runner")

    json.dumps(report)
    assert report["status"] == "TFLM_STATIC_FAIL"
    assert report["host_runtime_status"] == "FAIL"
    assert report["fixtures"] == []
    assert report["errors"]


def test_static_gate_and_runtime_inputs_bind_to_one_snapshot(
    tmp_path: Path, smoke_artifacts: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "model.tflite"
    model.write_bytes(smoke_artifacts["int8"].read_bytes())
    original_hash = _sha256(model)
    actual_check_model = tflm_module.check_model

    def mutate_original_after_snapshot(snapshot_path: Path) -> dict[str, object]:
        assert Path(snapshot_path) != model
        model.write_bytes(b"replacement-after-snapshot")
        return actual_check_model(snapshot_path)

    monkeypatch.setattr(tflm_module, "check_model", mutate_original_after_snapshot)
    report = check_tflm(model, runner_path=tmp_path / "missing-runner")

    assert report["status"] == "TFLM_RUNNER_UNAVAILABLE"
    assert report["model_sha256"] == original_hash
    assert report["model"]["sha256"] == original_hash
    assert report["budget"]["model"]["sha256"] == original_hash
    assert report["budget"]["model"]["path"] == str(model)
    assert report["model_snapshot"]["sha256"] == original_hash


def test_missing_manifest_fails_before_desktop_fallback(
    tmp_path: Path, smoke_artifacts: dict[str, Path]
) -> None:
    runner = _write_fake_runner(tmp_path / "runner")
    report = check_tflm(smoke_artifacts["int8"], runner_path=runner)

    json.dumps(report)
    assert report["status"] == "TFLM_HOST_FAIL"
    assert report["host_runtime_status"] == "FAIL"
    assert report["build_provenance"]["status"] == "FAIL"
    assert any("manifest" in error.lower() for error in report["build_provenance"]["errors"])
    assert "reference" not in report


def test_malformed_manifest_fails_closed(
    tmp_path: Path, smoke_artifacts: dict[str, Path]
) -> None:
    runner = _write_fake_runner(tmp_path / "runner")
    Path(f"{runner}.build.json").write_text("{not-json", encoding="utf-8")

    report = check_tflm(smoke_artifacts["int8"], runner_path=runner)

    assert report["status"] == "TFLM_HOST_FAIL"
    assert report["build_provenance"]["status"] == "FAIL"
    assert "valid JSON" in report["build_provenance"]["errors"][0]
    assert "reference" not in report


def test_manifest_hash_mismatch_fails_closed(
    tmp_path: Path, smoke_artifacts: dict[str, Path]
) -> None:
    runner = _write_fake_runner(tmp_path / "runner")
    _write_manifest(runner, _manifest_payload(tmp_path))

    report = check_tflm(smoke_artifacts["int8"], runner_path=runner)

    assert report["status"] == "TFLM_HOST_FAIL"
    assert report["build_provenance"]["status"] == "FAIL"
    assert any("hash mismatch" in error for error in report["build_provenance"]["errors"])
    assert "reference" not in report


def test_runner_source_mismatch_is_protocol_failure(
    tmp_path: Path, smoke_artifacts: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    model = smoke_artifacts["int8"]
    runner = _write_fake_runner(
        tmp_path / "runner", _protocol_report(model, source="wrong-source")
    )
    monkeypatch.setattr(tflm_module, "_verify_build_manifest", _bound_manifest)

    report = check_tflm(model, runner_path=runner)

    assert report["status"] == "TFLM_HOST_FAIL"
    assert report["fixtures"]
    assert all(
        "Git commit does not match" in fixture["comparison_error"]
        for fixture in report["fixtures"]
    )
    assert report["build_provenance"]["runner_reported_tflm_git_commit"] == "wrong-source"


def test_runner_zero_arena_usage_is_protocol_failure(
    tmp_path: Path, smoke_artifacts: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    model = smoke_artifacts["int8"]
    runner = _write_fake_runner(tmp_path / "runner", _protocol_report(model, used=0))
    monkeypatch.setattr(tflm_module, "_verify_build_manifest", _bound_manifest)

    report = check_tflm(model, runner_path=runner)

    assert report["status"] == "TFLM_HOST_FAIL"
    assert all("0 < used <= capacity" in fixture["comparison_error"] for fixture in report["fixtures"])


def test_runner_oserror_is_reported_without_false_pass(
    tmp_path: Path, smoke_artifacts: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _write_fake_runner(tmp_path / "runner")
    monkeypatch.setattr(tflm_module, "_verify_build_manifest", _bound_manifest)

    def raise_oserror(*args: object, **kwargs: object) -> object:
        raise OSError("simulated process start failure")

    monkeypatch.setattr(tflm_module, "_run_runner", raise_oserror)
    report = check_tflm(smoke_artifacts["int8"], runner_path=runner)

    assert report["status"] == "TFLM_HOST_FAIL"
    assert report["host_runtime_status"] == "FAIL"
    assert len(report["fixtures"]) == 3
    assert all("simulated process start failure" in fixture["runner_stderr"] for fixture in report["fixtures"])


def test_actual_runner_parity_is_deferred_without_c_manifest(
    smoke_artifacts: dict[str, Path],
) -> None:
    runner = _require_runner()
    model = smoke_artifacts["int8"]
    report = check_tflm(model, runner_path=runner)

    json.dumps(report)
    assert report["status"] == "TFLM_HOST_PASS"
    assert report["host_runtime_status"] == "PASS"
    assert report["runtime_status"] == "UNVERIFIED_HARDWARE"
    assert report["contract_status"] == "CONTRACT_UNRESOLVED"
    assert report["model_training_status"] == "NOT_ESTABLISHED"
    assert report["input_status"] == "SYNTHETIC_INPUT"
    assert report["build_provenance"]["status"] == "BUILD_BOUND"
    assert report["reference"]["mode"] == "BUILTIN_REF_NO_DELEGATES"
    assert report["model_sha256"] == _sha256(model)
    assert "source_sha256" not in report
    assert all(fixture["max_abs_lsb"] <= 1 for fixture in report["fixtures"])
