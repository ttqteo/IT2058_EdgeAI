"""Write the local, evidence-oriented TFLM host build manifest.

This sidecar records the files and checkout used by one local build.  It is
not a signature, a GPG attestation, or a board acceptance record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import os
import re
import subprocess


TFLM_SOURCE_SHA = "230c9d9531bd9b08d445b4f2ed0760756ffd5f78"
BUILD_POLICY = "FORCED_REBUILD_PINNED_SOURCE"
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _file_path(value: str, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute path: {value}")
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"{label} is not a regular file: {path}")
    return path


def _checkout_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"tflm checkout must be an absolute path: {value}")
    path = path.resolve()
    if not path.is_dir() or not (path / ".git").exists():
        raise ValueError(f"tflm checkout is not a git directory: {path}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    if not HEX64.fullmatch(value):
        raise AssertionError(f"unexpected SHA-256 encoding for {path}")
    return value


def _git(checkout: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(checkout), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def build_manifest(args: argparse.Namespace) -> tuple[Path, dict[str, object]]:
    runner = _file_path(args.runner, "runner")
    checkout = _checkout_path(args.tflm_checkout)
    library = _file_path(args.tflm_library, "tflm library")
    runner_source = _file_path(args.runner_source, "runner source")
    runner_makefile = _file_path(args.runner_makefile, "runner Makefile")
    build_script = _file_path(args.build_script, "build script")
    expected_commit = args.tflm_commit
    if expected_commit != TFLM_SOURCE_SHA:
        raise ValueError("manifest writer is pinned to the approved TFLM source SHA")
    actual_commit = _git(checkout, "rev-parse", "HEAD")
    if actual_commit != expected_commit:
        raise ValueError(
            f"tflm checkout SHA mismatch: expected {expected_commit}, got {actual_commit}"
        )
    status = _git(checkout, "status", "--porcelain", "--untracked-files=all")
    if status:
        raise ValueError("tflm checkout is dirty; manifest refuses unclean source")
    if not args.compiler_version.strip():
        raise ValueError("compiler_version must be non-empty")
    if not args.compile_flags.strip():
        raise ValueError("compile_flags must be non-empty")

    manifest = {
        "build_policy": BUILD_POLICY,
        "compiler_version": args.compiler_version,
        "compile_flags": args.compile_flags,
        "format_version": 1,
        "runner_binary_sha256": _sha256(runner),
        "runner_makefile_sha256": _sha256(runner_makefile),
        "runner_source_sha256": _sha256(runner_source),
        "build_script_sha256": _sha256(build_script),
        "tflm_checkout": str(checkout),
        "tflm_git_commit": actual_commit,
        "tflm_library": str(library),
        "tflm_library_sha256": _sha256(library),
    }
    required_keys = {
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
    }
    if set(manifest) != required_keys:
        raise AssertionError("manifest key set does not match the approved contract")
    return runner.with_name(runner.name + ".build.json"), manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", required=True)
    parser.add_argument("--tflm-checkout", required=True)
    parser.add_argument("--tflm-commit", required=True)
    parser.add_argument("--tflm-library", required=True)
    parser.add_argument("--runner-source", required=True)
    parser.add_argument("--runner-makefile", required=True)
    parser.add_argument("--build-script", required=True)
    parser.add_argument("--compiler-version", required=True)
    parser.add_argument("--compile-flags", required=True)
    args = parser.parse_args()
    try:
        manifest_path, manifest = build_manifest(args)
        temporary_path = manifest_path.with_name(manifest_path.name + ".tmp")
        temporary_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, manifest_path)
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        parser.error(str(error))
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
