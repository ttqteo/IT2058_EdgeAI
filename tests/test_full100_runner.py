"""Synthetic orchestration tests, not full-data/training acceptance evidence."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_full100.py"
spec = importlib.util.spec_from_file_location("full100_runner", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "SOURCE_NAMES", ())
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "audit_full100_data.py").write_text("# synthetic")
    validation = tmp_path / "validation"
    validation.mkdir()
    for name in ("manifest.json", "features.npz", "provenance.json"):
        (validation / name).write_bytes(b"synthetic")
    for name in ("archive", "runner", "runner.build.json"):
        (tmp_path / name).write_bytes(b"synthetic")
    return {"archive": str(tmp_path / "archive"), "validation": str(validation),
            "runner": str(tmp_path / "runner"), "teacher_model": str(tmp_path / "model"),
            "output": str(tmp_path / "output")}


def test_commands_isolation_and_defaults(config):
    plan = runner.commands(config, resume=False)
    assert "--teacher-output" not in plan["supervised"]
    assert "--teacher-output" in plan["kd"]
    assert ".venv-teacher/bin/python" in plan["teacher"][0]
    assert plan["teacher"][-4:] == ["--batch-size", "4", "--threads", "4"]
    assert len(plan) == 7
    assert "--resume" in runner.commands(config, resume=True)["kd"]


def test_complete_and_resume_does_not_reexecute(config, monkeypatch):
    monkeypatch.setattr(runner, "PHASES", ("prepare",))
    calls = []

    def fake(command, **kwargs):
        calls.append(command)
        output = Path(config["output"]) / "data"
        output.mkdir()
        (output / "manifest.json").write_text("synthetic")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", fake)
    assert runner.run(config)["status"] == "COMPLETE"
    assert runner.run(config, resume=True)["status"] == "COMPLETE"
    assert len(calls) == 1
    with pytest.raises(FileExistsError):
        runner.run(config)
    (Path(config["output"]) / "data/manifest.json").write_text("mutated")
    with pytest.raises(ValueError, match="mutated"):
        runner.run(config, resume=True)


def test_failed_process_recorded_and_explicit_resume(config, monkeypatch):
    monkeypatch.setattr(runner, "PHASES", ("supervised",))
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=7))
    with pytest.raises(RuntimeError, match="exited 7"):
        runner.run(config)
    status = json.loads((Path(config["output"]) / "run_status.json").read_text())
    assert status["status"] == "FAILED"

    def resumed(command, **kwargs):
        assert "--resume" in command
        directory = Path(config["output"]) / "baseline_supervised"
        directory.mkdir()
        (directory / "report.json").write_text("synthetic")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", resumed)
    assert runner.run(config, resume=True)["status"] == "COMPLETE"
    assert (Path(config["output"]) / "logs/supervised.001.log").is_file()
    assert (Path(config["output"]) / "logs/supervised.002.log").is_file()


def test_source_or_input_mutation_rejected(config, monkeypatch):
    monkeypatch.setattr(runner, "PHASES", ())
    runner.run(config)
    Path(config["archive"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="binding mismatch"):
        runner.run(config, resume=True)


def test_symlinks_fail_closed(tmp_path):
    real = tmp_path / "real"
    real.write_bytes(b"x")
    link = tmp_path / "link"
    link.symlink_to(real)
    with pytest.raises(ValueError, match="non-symlink"):
        runner.binding(link)


@pytest.mark.parametrize("lsb,accepted", [(1, True), (2, False)])
def test_host_real_fixture_gate_and_budget_boundary(tmp_path, monkeypatch, lsb, accepted):
    from edgeai import tflm

    model, binary = tmp_path / "model.tflite", tmp_path / "runner"
    for path in (model, binary, Path(str(binary) + ".build.json")):
        path.write_bytes(b"synthetic")
    features = np.zeros((5, 298, 40, 1), dtype=np.float32)
    dataset = SimpleNamespace(features=features, rows=[{"window_id": str(i)} for i in range(5)],
                              manifest_sha256="synthetic")
    monkeypatch.setitem(sys.modules, "edgeai.full_data", SimpleNamespace(load_full_dataset=lambda root: dataset))
    budget = {"schema": "distinct-budget-sentinel"}
    monkeypatch.setattr(tflm, "check_tflm", lambda *a, **kw: {
        "status": "TFLM_HOST_PASS", "build_provenance": {"status": "BUILD_BOUND"}, "budget": budget})

    class Reference:
        def set_tensor(self, index, value):
            assert value.dtype == np.int8 and value.shape == (1, 298, 40, 1)

        def invoke(self):
            pass

        def get_tensor(self, index):
            return np.zeros((1, 192), dtype=np.int8)

    metadata = {"index": 0, "scale": 0.1, "zero_point": 0}
    monkeypatch.setattr(tflm, "_reference_interpreter", lambda content: (Reference(), metadata, metadata, "FAKE"))

    def run_fake(binary, model, input_file, output_file, arena):
        output_file.write_bytes(np.full(192, lsb, dtype=np.int8).tobytes())
        return 0, {"arena_used_bytes": 38624}, "", ""

    def validate_fake(report, **kwargs):
        assert kwargs["budget_report"] is budget
        return []

    monkeypatch.setattr(tflm, "_run_runner", run_fake)
    monkeypatch.setattr(tflm, "_validate_runner_report", validate_fake)
    output = tmp_path / "report.json"
    if accepted:
        result = runner.host_check(model, tmp_path, binary, output)
        assert result["status"] == "PASS" and len(result["train_fixtures"]) == 5
        assert output.is_file()
    else:
        with pytest.raises(RuntimeError, match="one LSB"):
            runner.host_check(model, tmp_path, binary, output)
        assert not output.exists()
