"""Direct fail-closed checks for the experimental host-only TFLM runner.

The tests never build or download anything.  FlatBuffer mutations use the
official TensorFlow generated ``ModelT``/``Pack`` path and are written only to
pytest's temporary directory.  A passing mutation test proves rejection by
the local runner's preflight, not board or model suitability.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import subprocess
from typing import Callable

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "artifacts" / "tflm-host" / "edgeai_tflm_host"
MANIFEST = ROOT / "artifacts" / "tflm-host" / "edgeai_tflm_host.build.json"
INPUT_BYTES = 1 * 298 * 40 * 1
OUTPUT_BYTES = 1 * 192
MIN_ARENA_BYTES = 8192
DEFAULT_ARENA_BYTES = 131072
MAX_ARENA_BYTES = 131072
TFLM_SOURCE_SHA = "230c9d9531bd9b08d445b4f2ed0760756ffd5f78"


@pytest.fixture(scope="session")
def smoke_artifacts(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Generate both real smoke models in pytest's temporary directory.

    TensorFlow/converter failure is a test failure, rather than a skip: a
    built runner must exercise the actual persisted semantic checks and must
    not silently pass because an ignored repository artifact is absent.
    """

    output_dir = tmp_path_factory.mktemp("tflm-runner-smoke")
    try:
        from edgeai.model import create_smoke_model

        create_smoke_model(output_dir)
    except Exception as exc:  # pragma: no cover - environment failure path
        pytest.fail(
            "could not generate the self-contained TensorFlow smoke models: "
            f"{type(exc).__name__}: {exc}",
            pytrace=False,
        )
    artifacts = {
        "int8": output_dir / "encoder_int8.tflite",
        "fp32": output_dir / "encoder_fp32.tflite",
    }
    for label, path in artifacts.items():
        if not path.is_file() or path.stat().st_size == 0:
            pytest.fail(f"generated {label} smoke model is missing or empty: {path}")
    return artifacts


def _require_runner() -> Path:
    if not RUNNER.is_file() or RUNNER.stat().st_mode & 0o111 == 0:
        pytest.skip("local TFLM runner is unavailable; run the build script")
    return RUNNER


def _run(
    runner: Path,
    model: Path,
    input_path: Path,
    output_path: Path,
    arena_bytes: int,
) -> tuple[int, dict[str, object] | None]:
    completed = subprocess.run(
        [str(runner), str(model), str(input_path), str(output_path), str(arena_bytes)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError:
        report = None
    return completed.returncode, report if isinstance(report, dict) else None


def _write_input(path: Path, size: int = INPUT_BYTES) -> None:
    path.write_bytes(bytes(size))


def _official_schema():
    try:
        from tensorflow.lite.python import schema_py_generated
        import flatbuffers  # noqa: F401  # required by the official Pack path
    except ImportError as exc:  # pragma: no cover - environment failure path
        pytest.fail(
            "official TensorFlow FlatBuffer tooling is required for persisted "
            f"semantic mutations: {exc}",
            pytrace=False,
        )
    return schema_py_generated


def _pack_mutated_model(
    source: Path,
    destination: Path,
    mutate: Callable[[object, object], None],
) -> None:
    schema = _official_schema()
    import flatbuffers

    raw = source.read_bytes()
    model = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(raw, 0))
    mutate(model, schema)
    builder = flatbuffers.Builder(1024)
    root = model.Pack(builder)
    builder.Finish(root, file_identifier=b"TFL3")
    destination.write_bytes(bytes(builder.Output()))


def _smoke_copy(tmp_path: Path, model: Path) -> Path:
    destination = tmp_path / "smoke-copy.tflite"
    destination.write_bytes(model.read_bytes())
    return destination


def test_valid_smoke_invocation_reports_bounded_host_result(
    tmp_path: Path, smoke_artifacts: dict[str, Path]
) -> None:
    model = smoke_artifacts["int8"]
    runner = _require_runner()
    input_path = tmp_path / "input.int8"
    output_path = tmp_path / "output.int8"
    _write_input(input_path)

    returncode, report = _run(
        runner, model, input_path, output_path, DEFAULT_ARENA_BYTES
    )

    assert returncode == 0, report
    assert report is not None
    assert report["status"] == "PASS"
    assert report["tflm_source_sha"] == TFLM_SOURCE_SHA
    assert report["arena_capacity_bytes"] == DEFAULT_ARENA_BYTES
    assert 0 < report["arena_used_bytes"] <= DEFAULT_ARENA_BYTES
    assert output_path.stat().st_size == OUTPUT_BYTES
    assert {"CONV_2D", "AVERAGE_POOL_2D", "RESHAPE", "FULLY_CONNECTED"}.issubset(
        set(report["operations"])
    )


def test_eight_kib_arena_constructs_and_fails_allocate_tensors(
    tmp_path: Path, smoke_artifacts: dict[str, Path]
) -> None:
    model = smoke_artifacts["int8"]
    runner = _require_runner()
    input_path = tmp_path / "input.int8"
    output_path = tmp_path / "output.int8"
    _write_input(input_path)

    returncode, report = _run(
        runner, model, input_path, output_path, MIN_ARENA_BYTES
    )

    assert returncode != 0
    assert report is not None
    assert report["status"] == "ALLOCATE_TENSORS_FAILED"
    assert report["arena_capacity_bytes"] == MIN_ARENA_BYTES
    assert not output_path.exists()


@pytest.mark.parametrize("arena_bytes", [1, MAX_ARENA_BYTES + 1])
def test_arena_bounds_reject_before_interpreter_and_output(
    tmp_path: Path, smoke_artifacts: dict[str, Path], arena_bytes: int
) -> None:
    model = smoke_artifacts["int8"]
    runner = _require_runner()
    input_path = tmp_path / f"input-{arena_bytes}.int8"
    output_path = tmp_path / f"output-{arena_bytes}.int8"
    _write_input(input_path)

    returncode, report = _run(runner, model, input_path, output_path, arena_bytes)

    assert returncode != 0
    assert report is not None
    assert report["status"] == "REJECTED"
    assert not output_path.exists()


@pytest.mark.parametrize("input_size", [INPUT_BYTES - 1, INPUT_BYTES + 1])
def test_input_length_is_exact_and_rejection_is_atomic(
    tmp_path: Path, smoke_artifacts: dict[str, Path], input_size: int
) -> None:
    model = smoke_artifacts["int8"]
    runner = _require_runner()
    input_path = tmp_path / f"input-{input_size}.int8"
    output_path = tmp_path / f"output-{input_size}.int8"
    _write_input(input_path, input_size)

    returncode, report = _run(
        runner, model, input_path, output_path, DEFAULT_ARENA_BYTES
    )

    assert returncode != 0
    assert report is not None
    assert report["status"] == "REJECTED"
    assert not output_path.exists()


def test_malformed_and_fp32_models_are_rejected_without_output(
    tmp_path: Path, smoke_artifacts: dict[str, Path]
) -> None:
    runner = _require_runner()
    input_path = tmp_path / "input.int8"
    _write_input(input_path)

    malformed_model = tmp_path / "malformed.tflite"
    malformed_model.write_bytes(b"not-a-tflite-model")
    malformed_output = tmp_path / "malformed.out"
    returncode, report = _run(
        runner, malformed_model, input_path, malformed_output, DEFAULT_ARENA_BYTES
    )
    assert returncode != 0
    assert report is not None and report["status"] == "REJECTED"
    assert not malformed_output.exists()

    fp32_output = tmp_path / "fp32.out"
    returncode, report = _run(
        runner,
        smoke_artifacts["fp32"],
        input_path,
        fp32_output,
        DEFAULT_ARENA_BYTES,
    )
    assert returncode != 0
    assert report is not None and report["status"] == "REJECTED"
    assert not fp32_output.exists()


def _mutate_variable(model: object, _: object) -> None:
    model.subgraphs[0].tensors[0].isVariable = True


def _mutate_dynamic(model: object, _: object) -> None:
    model.subgraphs[0].tensors[0].shapeSignature = [-1, 298, 40, 1]


def _mutate_sparse(model: object, schema: object) -> None:
    model.subgraphs[0].tensors[0].sparsity = schema.SparsityParametersT()


def _mutate_variant_metadata(model: object, schema: object) -> None:
    model.subgraphs[0].tensors[0].variantTensors = [schema.VariantSubTypeT()]


def _mutate_internal_dtype(model: object, schema: object) -> None:
    model.subgraphs[0].tensors[2].type = schema.TensorType.FLOAT32


def _mutate_shape(model: object, _: object) -> None:
    model.subgraphs[0].tensors[2].shape = [0, 760]


def _mutate_options(model: object, schema: object) -> None:
    operation = model.subgraphs[0].operators[0]
    operation.builtinOptionsType = schema.BuiltinOptions.NONE
    operation.builtinOptions = None


def _mutate_options2(model: object, schema: object) -> None:
    operation = model.subgraphs[0].operators[0]
    operation.builtinOptions2Type = schema.BuiltinOptions2.DilateOptions
    operation.builtinOptions2 = schema.DilateOptionsT()


def _mutate_fc_quantized_bias_type(model: object, schema: object) -> None:
    # The generated fixture's final operator is FULLY_CONNECTED.  Its actual
    # bias tensor remains INT32; this mutation changes only the schema option.
    operation = model.subgraphs[0].operators[3]
    operation.builtinOptions.quantizedBiasType = schema.TensorType.INT32


def _mutate_arity(model: object, _: object) -> None:
    model.subgraphs[0].operators[0].inputs = [0, 4]


def _mutate_quantization(model: object, _: object) -> None:
    model.subgraphs[0].tensors[0].quantization.zeroPoint = [128]


def _mutate_buffer(model: object, _: object) -> None:
    buffer = model.buffers[model.subgraphs[0].tensors[2].buffer]
    buffer.data = bytes(buffer.data)[:-1]


def _set_fc_input_shape(model: object, shape: list[int]) -> None:
    """Keep the fixture RESHAPE valid while changing the FC input layout."""

    subgraph = model.subgraphs[0]
    reshape = subgraph.operators[2]
    output_tensor = subgraph.tensors[int(reshape.outputs[0])]
    shape_tensor = subgraph.tensors[int(reshape.inputs[1])]
    output_tensor.shape = list(shape)
    if output_tensor.shapeSignature is not None:
        output_tensor.shapeSignature = list(shape)
    shape_tensor.shape = [len(shape)]
    if shape_tensor.shapeSignature is not None:
        shape_tensor.shapeSignature = [len(shape)]
    shape_buffer = model.buffers[shape_tensor.buffer]
    shape_buffer.data = b"".join(struct.pack("<i", dimension) for dimension in shape)


def _mutate_fc_input_batch(model: object, _: object) -> None:
    # The product remains 760, so RESHAPE succeeds and FC owns the rejection.
    _set_fc_input_shape(model, [2, 380])


def _mutate_fc_input_rank(model: object, _: object) -> None:
    # The product remains 760, so RESHAPE succeeds and FC sees rank three.
    _set_fc_input_shape(model, [1, 38, 20])


def _mutate_fc_output_batch(model: object, _: object) -> None:
    output_tensor = model.subgraphs[0].tensors[8]
    output_tensor.shape = [2, 192]
    if output_tensor.shapeSignature is not None:
        output_tensor.shapeSignature = [2, 192]


def _mutate_fc_output_rank(model: object, _: object) -> None:
    output_tensor = model.subgraphs[0].tensors[8]
    output_tensor.shape = [1, 192, 1]
    if output_tensor.shapeSignature is not None:
        output_tensor.shapeSignature = [1, 192, 1]


@pytest.mark.parametrize(
    ("label", "mutator"),
    [
        ("variable", _mutate_variable),
        ("dynamic", _mutate_dynamic),
        ("sparse", _mutate_sparse),
        ("variant", _mutate_variant_metadata),
        ("internal_dtype", _mutate_internal_dtype),
        ("shape", _mutate_shape),
        ("options", _mutate_options),
        ("options2", _mutate_options2),
        ("fc_quantized_bias_type", _mutate_fc_quantized_bias_type),
        ("arity", _mutate_arity),
        ("quantization", _mutate_quantization),
        ("buffer", _mutate_buffer),
        ("fc_input_batch", _mutate_fc_input_batch),
        ("fc_input_rank", _mutate_fc_input_rank),
        ("fc_output_batch", _mutate_fc_output_batch),
        ("fc_output_rank", _mutate_fc_output_rank),
    ],
)
def test_semantic_mutations_reject_before_constructor(
    tmp_path: Path,
    smoke_artifacts: dict[str, Path],
    label: str,
    mutator: Callable[[object, object], None],
) -> None:
    source = _smoke_copy(tmp_path, smoke_artifacts["int8"])
    runner = _require_runner()
    mutated_model = tmp_path / f"invalid-{label}.tflite"
    _pack_mutated_model(source, mutated_model, mutator)
    input_path = tmp_path / f"input-{label}.int8"
    output_path = tmp_path / f"output-{label}.int8"
    _write_input(input_path)

    returncode, report = _run(
        runner, mutated_model, input_path, output_path, DEFAULT_ARENA_BYTES
    )

    assert returncode != 0
    assert report is not None, f"runner emitted no JSON for {label}"
    assert report["status"] == "REJECTED", report
    assert not output_path.exists()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_build_manifest_binds_current_local_artifacts() -> None:
    if not MANIFEST.is_file():
        pytest.skip("build manifest is written after the reproducible build/test sequence")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    required = {
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
    assert set(manifest) == required
    assert manifest["format_version"] == 1
    assert manifest["build_policy"] == "FORCED_REBUILD_PINNED_SOURCE"
    assert manifest["tflm_git_commit"] == TFLM_SOURCE_SHA
    assert Path(manifest["tflm_checkout"]).is_absolute()
    assert Path(manifest["tflm_library"]).is_absolute()
    assert manifest["tflm_library_sha256"] == _sha256(Path(manifest["tflm_library"]))
    assert manifest["runner_source_sha256"] == _sha256(
        ROOT / "firmware" / "tflm_host" / "runner.cc"
    )
    assert manifest["runner_makefile_sha256"] == _sha256(
        ROOT / "firmware" / "tflm_host" / "Makefile"
    )
    assert manifest["build_script_sha256"] == _sha256(
        ROOT / "scripts" / "build_tflm_host.sh"
    )
    assert manifest["runner_binary_sha256"] == _sha256(RUNNER)
    assert manifest["compiler_version"]
    assert manifest["compile_flags"]
