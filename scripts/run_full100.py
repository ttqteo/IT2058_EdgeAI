#!/usr/bin/env python3
"""Sequential, resumable full train-clean-100 run; no board or final-test claim."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
DEFAULT_ARCHIVE = ROOT / "artifacts/datasets/librispeech-train-clean-100/download/train-clean-100.tar.gz"
DEFAULT_VALIDATION = ROOT / "artifacts/datasets/librispeech-dev-clean/prepared"
DEFAULT_MODEL = ROOT / "artifacts/teacher/model/0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"
DEFAULT_RUNNER = ROOT / "artifacts/tflm-host/edgeai_tflm_host"
DEFAULT_OUTPUT = ROOT / "artifacts/full100/20261015"
PHASES = ("prepare", "audit_data", "supervised", "host_supervised", "teacher", "kd", "host_kd")
SOURCE_NAMES = (
    "full_data", "full_teacher", "full_training", "training", "teacher",
    "comparative_training", "enhancement_training", "frontend", "evaluation",
    "model", "budget", "tflm", "expanded_data", "pilot_data",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def binding(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected regular non-symlink file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def inventory(paths: list[Path]) -> dict[str, Any]:
    return {str(p): binding(p) for p in sorted(paths)}


def tree_inventory(root: Path) -> dict[str, Any]:
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"Missing final output directory: {root}")
    paths = sorted(p for p in root.rglob("*") if p.is_file() or p.is_symlink())
    if not paths:
        raise ValueError(f"Empty output directory: {root}")
    return {str(p.relative_to(root)): binding(p) for p in paths}


def write_json(path: Path, value: Any) -> None:
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def commands(config: dict[str, str], *, resume: bool) -> dict[str, list[str]]:
    out = Path(config["output"])
    py = str(ROOT / ".venv/bin/python")
    teacher_py = str(ROOT / ".venv-teacher/bin/python")
    train = [py, "-u", "-m", "edgeai.full_training", "--dataset-root", str(out / "data"),
             "--validation-root", config["validation"]]
    result = {
        "prepare": [py, "-u", "-m", "edgeai.full_data", "--archive", config["archive"],
                    "--output", str(out / "data")],
        "audit_data": [py, "-u", str(ROOT / "scripts/audit_full100_data.py"), "--root", str(out / "data"),
                       "--archive", config["archive"], "--output", str(out / "independent_data_audit.json")],
        "supervised": train + ["--method", "supervised_aam", "--output", str(out / "baseline_supervised")],
        "teacher": [teacher_py, "-u", "-m", "edgeai.full_teacher", "--dataset-root", str(out / "data"),
                    "--model-dir", config["teacher_model"], "--output", str(out / "teacher"),
                    "--batch-size", "4", "--threads", "4"],
        "kd": train + ["--method", "kd_aam", "--teacher-output", str(out / "teacher"),
                       "--output", str(out / "baseline_kd")],
    }
    for name, directory in (("host_supervised", "baseline_supervised"), ("host_kd", "baseline_kd")):
        result[name] = [py, "-u", str(Path(__file__).resolve()), "--host-check",
                        str(out / directory / "encoder_int8.tflite"), "--dataset-root", str(out / "data"),
                        "--runner", config["runner"], "--host-output", str(out / f"{name}.json")]
    if resume:
        for phase in ("supervised", "kd"):
            result[phase].append("--resume")
    return result


def phase_target(out: Path, phase: str) -> Path:
    return out / {"prepare": "data", "supervised": "baseline_supervised", "teacher": "teacher",
                  "audit_data": "independent_data_audit.json", "kd": "baseline_kd",
                  "host_supervised": "host_supervised.json", "host_kd": "host_kd.json"}[phase]


def target_binding(target: Path) -> dict[str, Any]:
    return tree_inventory(target) if target.is_dir() else binding(target)


def run(config: dict[str, str], *, resume: bool = False) -> dict[str, Any]:
    out = Path(config["output"])
    if not resume:
        out.mkdir(parents=True, exist_ok=False)
    elif not out.is_dir() or out.is_symlink():
        raise ValueError("Resume requires the existing run directory")
    with (out / ".run.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sources = inventory([ROOT / f"src/edgeai/{name}.py" for name in SOURCE_NAMES] +
                            [Path(__file__).resolve(), ROOT / "scripts/audit_full100_data.py"])
        inputs = inventory([Path(config["archive"]), Path(config["runner"]),
                            Path(config["runner"] + ".build.json")] +
                           [Path(config["validation"]) / name for name in ("manifest.json", "features.npz", "provenance.json")])
        state_path = out / "run_status.json"
        if resume:
            state = json.loads(state_path.read_text())
            if state["config"] != config or state["sources"] != sources or state["inputs"] != inputs:
                raise ValueError("Resume binding mismatch: config, source or input changed")
            for phase, previous in state["phases"].items():
                if previous["status"] == "COMPLETE" and previous["output_binding"] != target_binding(phase_target(out, phase)):
                    raise ValueError(f"Previously completed output mutated: {phase}")
        else:
            state = {"format_version": 1, "created_at": now(), "config": config, "sources": sources,
                     "inputs": inputs, "phases": {}, "status": "RUNNING", "scope": "EXPERIMENTAL_HOST_ONLY",
                     "hardware_status": "UNVERIFIED_HARDWARE", "calibration": None, "test": None}
        state["status"] = "RUNNING"
        write_json(state_path, state)
        (out / "logs").mkdir(exist_ok=True)
        env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1",
                   OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
        plan = commands(config, resume=resume)
        for phase in PHASES:
            previous = state["phases"].get(phase, {})
            if previous.get("status") == "COMPLETE":
                continue
            # Training stages decide whether a resume checkpoint actually exists.
            command = plan[phase]
            if "--resume" in command and not previous:
                command = [arg for arg in command if arg != "--resume"]
            attempt = previous.get("attempt", 0) + 1
            log = out / "logs" / f"{phase}.{attempt:03d}.log"
            state["phases"][phase] = {"status": "RUNNING", "started_at": now(), "attempt": attempt,
                                      "command": command, "log": str(log)}
            write_json(state_path, state)
            print(f"{now()} START {phase}; log={log}", flush=True)
            try:
                if sources != inventory([Path(p) for p in sources]):
                    raise ValueError("Source mutation during run")
                if inputs != inventory([Path(p) for p in inputs]):
                    raise ValueError("Input mutation during run")
                with log.open("xb") as stream:
                    completed = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                               stdin=subprocess.DEVNULL, check=False)
                if completed.returncode:
                    raise RuntimeError(f"{phase} exited {completed.returncode}; see {log}")
                if sources != inventory([Path(p) for p in sources]):
                    raise ValueError("Source mutation during phase")
                state["phases"][phase].update(status="COMPLETE", finished_at=now(),
                                               output_binding=target_binding(phase_target(out, phase)))
                write_json(state_path, state)
                print(f"{now()} COMPLETE {phase}", flush=True)
            except BaseException as exc:
                state["status"] = "INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "FAILED"
                state["phases"][phase].update(status=state["status"], error=f"{type(exc).__name__}: {exc}")
                write_json(state_path, state)
                raise
        if inputs != inventory([Path(p) for p in inputs]):
            state.update(status="FAILED", error="Input mutation before run completion")
            write_json(state_path, state)
            raise ValueError("Input mutation before run completion")
        state.update(status="COMPLETE", finished_at=now())
        write_json(state_path, state)
        return state


def host_check(model: Path, dataset_root: Path, runner: Path, output: Path) -> dict[str, Any]:
    import numpy as np
    from edgeai import tflm
    from edgeai.full_data import load_full_dataset

    if output.exists():
        raise FileExistsError(output)
    before = inventory([model, runner, Path(str(runner) + ".build.json")])
    report = tflm.check_tflm(model, runner_path=runner)
    if report.get("status") != "TFLM_HOST_PASS" or report.get("build_provenance", {}).get("status") != "BUILD_BOUND":
        raise RuntimeError(f"TFLM host gate failed: {report}")
    data = load_full_dataset(dataset_root)
    content = model.read_bytes()
    interpreter, inp, out, mode = tflm._reference_interpreter(content)
    fixtures = []
    with tempfile.TemporaryDirectory(prefix="edgeai-full100-parity-") as temporary:
        for index in np.linspace(0, len(data.rows) - 1, 5, dtype=int):
            value = np.asarray(data.features[index:index + 1], dtype=np.float32)
            quantized = np.clip(np.rint(value / inp["scale"]) + inp["zero_point"], -128, 127).astype(np.int8)
            interpreter.set_tensor(inp["index"], quantized)
            interpreter.invoke()
            expected = interpreter.get_tensor(out["index"]).reshape(-1)
            input_file, output_file = Path(temporary) / "input.bin", Path(temporary) / "output.bin"
            if output_file.exists():
                output_file.unlink()
            input_file.write_bytes(quantized.tobytes())
            code, parsed, _, stderr = tflm._run_runner(runner, model, input_file, output_file, 131072)
            errors = tflm._validate_runner_report(parsed, budget_report=report["budget"], input_metadata=inp,
                       output_metadata=out, model_bytes=content, arena_bytes=131072) if parsed else ["Missing runner report"]
            if code or errors or not output_file.is_file() or output_file.stat().st_size != 192:
                raise RuntimeError(f"Real fixture failed: {code}, {errors}, {stderr}")
            actual = np.frombuffer(output_file.read_bytes(), dtype=np.int8)
            maximum = int(np.max(np.abs(actual.astype(np.int16) - expected.astype(np.int16))))
            if maximum > 1:
                raise RuntimeError(f"TFLM parity exceeded one LSB: {maximum}")
            fixtures.append({"row_index": int(index), "window_id": data.rows[index]["window_id"],
                             "input_sha256": binding(input_file)["sha256"], "max_lsb": maximum,
                             "arena_used_bytes": parsed["arena_used_bytes"]})
    if before != inventory([Path(p) for p in before]):
        raise ValueError("Model or runner mutation during host checks")
    result = {"status": "PASS", "scope": "EXPERIMENTAL_HOST_ONLY", "hardware_status": "UNVERIFIED_HARDWARE",
              "bindings": before, "synthetic_tflm": report, "reference_mode": mode,
              "train_fixtures": fixtures, "manifest_sha256": data.manifest_sha256}
    write_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--teacher-model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--host-check", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--dataset-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--host-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.host_check:
        if not args.dataset_root or not args.host_output:
            parser.error("Host checks require --dataset-root and --host-output")
        host_check(args.host_check.resolve(), args.dataset_root.resolve(), args.runner.resolve(), args.host_output.resolve())
        return
    config = {name: str(getattr(args, name).resolve()) for name in ("archive", "validation", "teacher_model", "runner", "output")}
    if args.dry_run:
        print(json.dumps(commands(config, resume=args.resume), indent=2))
    else:
        run(config, resume=args.resume)


if __name__ == "__main__":
    main()
