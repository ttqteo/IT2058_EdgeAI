"""CLI integration errors must not report hardware success."""

import json

import pytest

from edgeai.cli import main


@pytest.mark.parametrize("arena_bytes", ["0", "1", "131073"])
def test_tflm_cli_persists_invalid_arena_failure(tmp_path, capsys, arena_bytes):
    output = tmp_path / "report.json"
    result = main([
        "tflm-check", "--model", str(tmp_path / "missing.tflite"),
        "--output", str(output), "--arena-bytes", arena_bytes,
    ])
    assert result == 2
    report = json.loads(output.read_text())
    assert report == json.loads(capsys.readouterr().out)
    assert report["status"] == "TFLM_INVALID_ARGUMENT"
    assert report["host_runtime_status"] == "NOT_RUN"
    assert report["runtime_status"] == "UNVERIFIED_HARDWARE"


def test_tflm_cli_missing_model_persists_static_failure(tmp_path, capsys):
    output = tmp_path / "report.json"
    result = main([
        "tflm-check", "--model", str(tmp_path / "missing.tflite"),
        "--output", str(output),
    ])
    assert result == 2
    report = json.loads(output.read_text())
    assert report == json.loads(capsys.readouterr().out)
    assert report["status"] == "TFLM_STATIC_FAIL"
    assert report["fixtures"] == []
    assert report["runtime_status"] == "UNVERIFIED_HARDWARE"
