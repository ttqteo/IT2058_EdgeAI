"""Command-line entry points for the bounded host prototype."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .frontend import (
    CONTRACT_DISCOVERY,
    CONTRACT_UNRESOLVED,
    EXPERIMENTAL_HOST_ONLY,
    load_wav,
    log_mel,
)
from .matching import MatchResult, enroll, match_1_to_1, match_1_to_n


def _write_json_stdout(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _demo(output_dir: Path) -> dict[str, Any]:
    """Run a deterministic local simulation with no device/transport layer."""

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2058)
    alice_axis = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    bob_axis = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    alice_samples = np.stack(
        [alice_axis + rng.normal(0.0, 0.025, size=8) for _ in range(3)]
    ).astype(np.float32)
    bob_samples = np.stack(
        [bob_axis + rng.normal(0.0, 0.025, size=8) for _ in range(3)]
    ).astype(np.float32)
    prototypes = {"alice": enroll(alice_samples), "bob": enroll(bob_samples)}
    thresholds = {"cosine": 0.85, "margin": 0.10}
    ledger: set[str] = set()
    rows: list[dict[str, object]] = []

    def append_enrollment(label: str, sample_count: int) -> None:
        rows.append(
            {
                "event": "enrollment",
                "mode": "enroll",
                "label": label,
                "accepted": "",
                "score": "",
                "second_score": "",
                "margin": "",
                "reason": "prototype_created",
                "recorded": "",
                "duplicate_suppressed": "",
                "sample_count": sample_count,
            }
        )

    def append_decision(mode: str, result: MatchResult) -> None:
        label = result.label
        duplicate = bool(result.accepted and label in ledger)
        recorded = bool(result.accepted and not duplicate and label is not None)
        if recorded and label is not None:
            ledger.add(label)
        rows.append(
            {
                "event": "match",
                "mode": mode,
                "label": label or "",
                "accepted": result.accepted,
                "score": "" if result.score is None else f"{result.score:.8f}",
                "second_score": (
                    "" if result.second_score is None else f"{result.second_score:.8f}"
                ),
                "margin": "" if result.margin is None else f"{result.margin:.8f}",
                "reason": "duplicate_suppressed" if duplicate else result.reason,
                "recorded": recorded,
                "duplicate_suppressed": duplicate,
                "sample_count": "",
            }
        )

    append_enrollment("alice", int(alice_samples.shape[0]))
    append_enrollment("bob", int(bob_samples.shape[0]))

    alice_query = normalize_for_demo(alice_axis + rng.normal(0.0, 0.02, size=8))
    append_decision(
        "1:N",
        match_1_to_n(
            alice_query,
            prototypes,
            threshold=thresholds["cosine"],
            margin=thresholds["margin"],
        ),
    )
    append_decision(
        "1:N",
        match_1_to_n(
            alice_query,
            prototypes,
            threshold=thresholds["cosine"],
            margin=thresholds["margin"],
        ),
    )
    unknown_query = normalize_for_demo(alice_axis + bob_axis)
    append_decision(
        "1:N",
        match_1_to_n(
            unknown_query,
            prototypes,
            threshold=thresholds["cosine"],
            margin=thresholds["margin"],
        ),
    )

    bob_query = normalize_for_demo(bob_axis + rng.normal(0.0, 0.02, size=8))
    append_decision(
        "1:1",
        match_1_to_1(
            bob_query,
            prototypes["bob"],
            threshold=thresholds["cosine"],
            label="bob",
        ),
    )
    append_decision(
        "1:1",
        match_1_to_1(
            bob_query,
            prototypes["bob"],
            threshold=thresholds["cosine"],
            label="bob",
        ),
    )

    csv_path = output_dir / "attendance.csv"
    fieldnames = [
        "event",
        "mode",
        "label",
        "accepted",
        "score",
        "second_score",
        "margin",
        "reason",
        "recorded",
        "duplicate_suppressed",
        "sample_count",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "status": "SIMULATED_ATTENDANCE_DEMO",
        "simulated": True,
        "transport": None,
        "hardware": None,
        "frontend_status": EXPERIMENTAL_HOST_ONLY,
        "final_runtime_status": CONTRACT_UNRESOLVED,
        "contract_discovery": CONTRACT_DISCOVERY,
        "thresholds": thresholds,
        "thresholds_are_synthetic_demo_only": True,
        "enrolled_labels": sorted(prototypes),
        "recorded_labels": sorted(ledger),
        "duplicate_suppression": True,
        "attendance_csv": str(csv_path),
        "events": len(rows),
    }
    report_path = output_dir / "demo_report.json"
    report["report_path"] = str(report_path)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def normalize_for_demo(values: np.ndarray) -> np.ndarray:
    """Normalize only synthetic demo vectors without changing matching rules."""

    norm = float(np.linalg.norm(values))
    if norm <= 0.0 or not np.isfinite(norm):
        raise ValueError("demo vector must have a finite, non-zero norm")
    return (values / norm).astype(np.float32)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bounded EdgeAI host prototype")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess = subparsers.add_parser(
        "preprocess", help="load strict PCM16 mono16k WAV and write log-mel .npy"
    )
    preprocess.add_argument("wav", type=Path)
    preprocess.add_argument("--output", type=Path, required=True)

    demo = subparsers.add_parser(
        "demo", help="run a clearly simulated local attendance demo"
    )
    demo.add_argument("--output", type=Path, required=True)

    smoke = subparsers.add_parser(
        "smoke-model", help="build and invoke the untrained ML smoke model"
    )
    smoke.add_argument("--output", type=Path, required=True)

    budget = subparsers.add_parser(
        "budget-check",
        help="statically inspect a TFLite artifact against the ESP32-S3 budget",
    )
    budget.add_argument("--model", type=Path, required=True)
    budget.add_argument("--output", type=Path, required=True)
    budget.add_argument("--profile", default="esp32s3-budget")

    tflm = subparsers.add_parser(
        "tflm-check", help="run real TFLite Micro host inference in a bounded arena"
    )
    tflm.add_argument("--model", type=Path, required=True)
    tflm.add_argument("--output", type=Path, required=True)
    tflm.add_argument("--runner", type=Path)
    tflm.add_argument(
        "--arena-bytes", type=int, default=131072,
        help="host arena capacity, 8192..131072 bytes (default: 131072)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "preprocess":
        pcm, wav_report = load_wav(args.wav)
        features = log_mel(pcm)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output, features)
        _write_json_stdout(
            {
                "output": str(args.output),
                "shape": list(features.shape),
                "dtype": str(features.dtype),
                "wav": wav_report.as_dict(),
                "frontend_status": EXPERIMENTAL_HOST_ONLY,
                "final_runtime_status": CONTRACT_UNRESOLVED,
                "contract_discovery": CONTRACT_DISCOVERY,
            }
        )
        return 0
    if args.command == "demo":
        _write_json_stdout(_demo(args.output))
        return 0
    if args.command == "smoke-model":
        from .model import create_smoke_model

        _write_json_stdout(create_smoke_model(args.output))
        return 0
    if args.command == "budget-check":
        from .budget import STATIC_PASS, check_model

        report = check_model(args.model, profile=args.profile)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _write_json_stdout(report)
        return 0 if report["status"] == STATIC_PASS else 2
    if args.command == "tflm-check":
        from .tflm import check_tflm

        try:
            report = check_tflm(
                args.model, runner_path=args.runner, arena_bytes=args.arena_bytes
            )
        except ValueError as exc:
            report = {
                "status": "TFLM_INVALID_ARGUMENT",
                "host_runtime_status": "NOT_RUN",
                "runtime_status": "UNVERIFIED_HARDWARE",
                "contract_status": CONTRACT_UNRESOLVED,
                "scope": EXPERIMENTAL_HOST_ONLY,
                "errors": [str(exc)],
            }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _write_json_stdout(report)
        return 0 if report["status"] == "TFLM_HOST_PASS" else 2
    raise AssertionError(f"unhandled command {args.command!r}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
