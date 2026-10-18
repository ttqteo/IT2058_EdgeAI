"""Static ESP32-S3 budget checks for a real TFLite FlatBuffer.

This module deliberately does not emulate an ESP32, TFLM arena planner, or
runtime.  It checks only properties that can be established from the model
bytes and the locked local budget policy.  TensorFlow is imported lazily so
the host frontend and matching tests do not require the optional ML extra.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
import struct
from typing import Any, Iterable


STATIC_PASS = "STATIC_BUDGET_PASS_RUNTIME_UNVERIFIED"
STATIC_FAIL = "STATIC_BUDGET_FAIL_RUNTIME_UNVERIFIED"
RUNTIME_UNVERIFIED = "UNVERIFIED_HARDWARE"
CONTRACT_UNRESOLVED = "CONTRACT_UNRESOLVED"
HOST_SCOPE = "EXPERIMENTAL_HOST_ONLY"
CONTRACT_DISCOVERY = "BOOTSTRAP-CONTRACT-01"

SCHEMA_SOURCE = (
    "https://github.com/tensorflow/tensorflow/blob/v2.20.0/"
    "tensorflow/compiler/mlir/lite/schema/schema.fbs"
)
QUANTIZATION_SOURCE = (
    "https://developers.google.com/edge/litert/conversion/tensorflow/"
    "quantization/quantization_spec"
)

EXPECTED_INPUT_SHAPE = (1, 298, 40, 1)
EXPECTED_OUTPUT_SHAPE = (1, 192)


@dataclass(frozen=True)
class BudgetProfile:
    """The explicitly approved local policy, not a chip guarantee."""

    name: str = "esp32s3-budget"
    max_model_bytes: int = 300_000
    max_macs: int = 50_000_000
    target_sram_bytes: int = 200_000
    module_flash_bytes: int = 16 * 1024 * 1024
    module_psram_bytes: int = 8 * 1024 * 1024

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def get_profile(name: str = "esp32s3-budget") -> BudgetProfile:
    """Return a locked profile; unknown names fail rather than fall back."""

    if name != "esp32s3-budget":
        raise ValueError(f"unknown budget profile: {name!r}")
    return BudgetProfile()


def _load_schema() -> Any:
    """Import TensorFlow's generated schema only when a check is requested."""

    from tensorflow.lite.python import schema_py_generated as schema

    return schema


def _enum_value(enum_type: Any, name: str) -> int:
    value = getattr(enum_type, name, None)
    if value is None:
        raise RuntimeError(f"installed schema has no enum value {name}")
    return int(value)


def _product(shape: Iterable[int]) -> int:
    result = 1
    for value in shape:
        result *= int(value)
    return result


def _safe_decode(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _float32_bits(value: float) -> int | None:
    """Return the serialized float32 bit pattern for a finite value."""

    if not math.isfinite(value):
        return None
    try:
        return struct.unpack("<I", struct.pack("<f", float(value)))[0]
    except (OverflowError, struct.error):
        return None


def _float32_close(left: float, right: float, *, max_ulps: int = 1) -> bool:
    """Compare serialized float32 values within a one-rounding ULP bound."""

    left_bits = _float32_bits(left)
    right_bits = _float32_bits(right)
    return (
        left_bits is not None
        and right_bits is not None
        and abs(left_bits - right_bits) <= max_ulps
    )


def _tensor_type_names(schema: Any) -> dict[int, str]:
    names = (
        "FLOAT32",
        "FLOAT16",
        "INT32",
        "UINT8",
        "INT64",
        "STRING",
        "BOOL",
        "INT16",
        "COMPLEX64",
        "INT8",
        "FLOAT64",
        "COMPLEX128",
        "UINT64",
        "RESOURCE",
        "VARIANT",
        "UINT32",
        "UINT16",
        "INT4",
    )
    return {
        int(getattr(schema.TensorType, name)): name
        for name in names
        if hasattr(schema.TensorType, name)
    }


def _builtin_names(schema: Any) -> dict[int, str]:
    names = (
        "AVERAGE_POOL_2D",
        "CONV_2D",
        "DEPTHWISE_CONV_2D",
        "FULLY_CONNECTED",
        "RELU",
        "RELU6",
        "RESHAPE",
        "CUSTOM",
        "DELEGATE",
    )
    return {
        int(getattr(schema.BuiltinOperator, name)): name
        for name in names
        if hasattr(schema.BuiltinOperator, name)
    }


def _memory_plan(profile: BudgetProfile) -> dict[str, Any]:
    return {
        "status": RUNTIME_UNVERIFIED,
        "target_sram_bytes": profile.target_sram_bytes,
        "module_flash_bytes": profile.module_flash_bytes,
        "module_psram_bytes": profile.module_psram_bytes,
        "planned_buffers": {
            "pcm16_psram_bytes": 48_000 * 2,
            "float_frontend_bytes": 298 * 40 * 4,
            "int8_input_bytes": 298 * 40,
            "float_output_bytes": 192 * 4,
        },
        "interpretation": (
            "planned host/firmware buffers only; these values are not summed "
            "and presented as an actual TFLM arena; module flash/PSRAM values "
            "are planning targets, not whole-allocation guarantees"
        ),
    }


class _Analyzer:
    """Fail-closed schema walk used by :func:`check_model`."""

    _ALLOWED_VERSIONS = {
        "CONV_2D": {1, 2, 3},
        "DEPTHWISE_CONV_2D": {1, 2, 3},
        "AVERAGE_POOL_2D": {1, 2},
        "RELU": {1},
        "RELU6": {1},
        "RESHAPE": {1, 2},
        "FULLY_CONNECTED": {1, 2, 3, 4},
    }

    def __init__(self, data: bytes, schema: Any, profile: BudgetProfile) -> None:
        self.data = data
        self.schema = schema
        self.profile = profile
        self.model: Any = None
        self.subgraph: Any = None
        self.tensor_records: list[dict[str, Any]] = []
        self.tensor_by_index: dict[int, dict[str, Any]] = {}
        self.opcode_records: list[dict[str, Any]] = []
        self.operations: list[dict[str, Any]] = []
        self.mac_breakdown: list[dict[str, Any]] = []
        self.total_macs = 0
        self.macs_complete = True
        self.errors: list[dict[str, Any]] = []
        self.checks: list[dict[str, Any]] = []
        self.schema_version: int | None = None

    def _check(
        self,
        name: str,
        status: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        record: dict[str, Any] = {"name": name, "status": status, "message": message}
        if details:
            record["details"] = details
        self.checks.append(record)

    def fail(
        self,
        name: str,
        message: str,
        details: dict[str, Any] | None = None,
        *,
        unresolved: bool = False,
    ) -> None:
        status = "UNRESOLVED" if unresolved else "FAIL"
        error: dict[str, Any] = {
            "check": name,
            "status": status,
            "message": message,
        }
        if details:
            error["details"] = details
        self.errors.append(error)
        self._check(name, status, message, details)

    def pass_check(
        self, name: str, message: str, details: dict[str, Any] | None = None
    ) -> None:
        self._check(name, "PASS", message, details)

    def run(self) -> None:
        try:
            identifier_check = getattr(
                self.schema.Model, "ModelBufferHasIdentifier", None
            )
            if not callable(identifier_check) or not identifier_check(self.data, 0):
                self.fail(
                    "flatbuffer_identifier",
                    "model bytes do not contain the TFLite FlatBuffer identifier",
                    unresolved=True,
                )
                return
            self.model = self.schema.Model.GetRootAsModel(self.data, 0)
            if self.model is None:
                self.fail("flatbuffer_parse", "schema returned no Model", unresolved=True)
                return
            self._validate_model_header()
            self._read_tensors()
            self._read_opcodes()
            self._validate_io()
            self._validate_graph()
        except Exception as exc:  # malformed/unexpected schema data fails closed
            self.fail(
                "flatbuffer_analysis",
                f"unexpected schema parse failure: {type(exc).__name__}: {exc}",
                unresolved=True,
            )

    def _validate_model_header(self) -> None:
        self.schema_version = int(self.model.Version())
        if self.schema_version != 3:
            self.fail(
                "schema_version",
                "unsupported FlatBuffer schema version",
                {"actual": self.schema_version, "required": 3},
                unresolved=True,
            )
        else:
            self.pass_check("schema_version", "FlatBuffer schema version is 3")

        subgraph_count = int(self.model.SubgraphsLength())
        if subgraph_count != 1:
            self.fail(
                "subgraph_count",
                "exactly one subgraph is required for static inspection",
                {"actual": subgraph_count, "required": 1},
                unresolved=True,
            )
        else:
            self.subgraph = self.model.Subgraphs(0)
            if self.subgraph is None:
                self.fail("subgraph", "subgraph table is missing", unresolved=True)
            else:
                self.pass_check("subgraph_count", "model has one subgraph")
        self._validate_buffers()

    def _validate_buffers(self) -> None:
        """Reject storage modes this bounded analyzer does not model."""

        if self.model is None:
            return
        buffer_count = int(self.model.BuffersLength())
        for index in range(buffer_count):
            buffer = self.model.Buffers(index)
            if buffer is None:
                self.fail(
                    "buffer_table",
                    "buffer table is missing",
                    {"buffer_index": index},
                    unresolved=True,
                )
                continue
            offset_method = getattr(buffer, "Offset", None)
            size_method = getattr(buffer, "Size", None)
            if not callable(offset_method) or not callable(size_method):
                self.fail(
                    "buffer_storage_mode",
                    "installed schema does not expose external-buffer metadata",
                    {"buffer_index": index},
                    unresolved=True,
                )
                continue
            offset = int(offset_method())
            size = int(size_method())
            if offset != 0 or size != 0:
                self.fail(
                    "external_buffer",
                    "external/offset-backed buffers are outside the static policy",
                    {
                        "buffer_index": index,
                        "offset": offset,
                        "size": size,
                    },
                    unresolved=True,
                )
        if not self.errors:
            self.pass_check(
                "buffer_storage_mode",
                "all buffers use inline FlatBuffer storage",
            )

    def _read_tensors(self) -> None:
        if self.subgraph is None:
            return
        type_names = _tensor_type_names(self.schema)
        tensor_count = int(self.subgraph.TensorsLength())
        for index in range(tensor_count):
            tensor = self.subgraph.Tensors(index)
            if tensor is None:
                self.fail(
                    "tensor_table",
                    "tensor table is missing",
                    {"tensor_index": index},
                    unresolved=True,
                )
                continue
            record: dict[str, Any] = {
                "index": index,
                "name": _safe_decode(tensor.Name()),
                "type_code": int(tensor.Type()),
                "type": type_names.get(int(tensor.Type()), "UNKNOWN"),
                "buffer": int(tensor.Buffer()),
            }
            shape_length = int(tensor.ShapeLength())
            shape = [int(tensor.Shape(i)) for i in range(shape_length)]
            signature_length = int(tensor.ShapeSignatureLength())
            signature = [
                int(tensor.ShapeSignature(i)) for i in range(signature_length)
            ]
            record["shape"] = shape
            record["shape_signature"] = signature
            if bool(tensor.IsVariable()):
                self.fail(
                    "variable_tensor",
                    "variable tensors are not modeled by the static artifact policy",
                    {"tensor_index": index},
                    unresolved=True,
                )
            sparsity = tensor.Sparsity()
            if sparsity is not None:
                self.fail(
                    "sparse_tensor",
                    "sparse tensor metadata is not modeled by the static analyzer",
                    {"tensor_index": index},
                    unresolved=True,
                )
            variant_length = int(tensor.VariantTensorsLength())
            if variant_length:
                self.fail(
                    "variant_tensor",
                    "variant tensor metadata is not modeled by the static analyzer",
                    {"tensor_index": index, "count": variant_length},
                    unresolved=True,
                )
            if record["type"] not in {"INT8", "INT32"}:
                self.fail(
                    "integer_graph_types",
                    "every graph tensor must be INT8 or INT32 for this profile",
                    {
                        "tensor_index": index,
                        "type": record["type"],
                    },
                    unresolved=True,
                )
            if not shape or any(dimension <= 0 for dimension in shape):
                self.fail(
                    "tensor_static_shape",
                    "all tensors must have non-empty, positive static shapes",
                    {"tensor_index": index, "shape": shape},
                    unresolved=True,
                )
            if signature and signature != shape:
                self.fail(
                    "tensor_shape_signature",
                    "dynamic or ambiguous shape signature is not accepted",
                    {
                        "tensor_index": index,
                        "shape": shape,
                        "shape_signature": signature,
                    },
                    unresolved=True,
                )
            record["quantization"] = self._read_quantization(tensor, record)
            self.tensor_records.append(record)
            self.tensor_by_index[index] = record

        if not self.errors:
            self.pass_check(
                "tensor_metadata",
                "tensor shapes, types, and relevant quantization metadata are valid",
            )

    def _read_quantization(
        self, tensor: Any, record: dict[str, Any]
    ) -> dict[str, Any]:
        quantization = tensor.Quantization()
        if quantization is None:
            quant = {
                "present": False,
                "scales": [],
                "zero_points": [],
                "quantized_dimension": None,
                "per_tensor": False,
            }
            if record["type"] == "INT8":
                self.fail(
                    "int8_quantization",
                    "every INT8 tensor requires finite positive quantization metadata",
                    {"tensor_index": record["index"]},
                )
            return quant

        scale_count = int(quantization.ScaleLength())
        zero_point_count = int(quantization.ZeroPointLength())
        scales = [float(quantization.Scale(i)) for i in range(scale_count)]
        zero_points = [int(quantization.ZeroPoint(i)) for i in range(zero_point_count)]
        qdim = int(quantization.QuantizedDimension())
        quant = {
            "present": True,
            "scales": scales,
            "zero_points": zero_points,
            "quantized_dimension": qdim,
            "per_tensor": scale_count == 1 and zero_point_count == 1,
        }

        # TensorFlow commonly emits an empty QuantizationParameters table for
        # INT32 shape tensors.  That is different from malformed quantization
        # on an INT8 tensor and is allowed for integer metadata tensors.
        if scale_count == 0 and zero_point_count == 0:
            quant["present"] = False
            if record["type"] == "INT8":
                self.fail(
                    "int8_quantization",
                    "every INT8 tensor requires finite positive quantization metadata",
                    {"tensor_index": record["index"]},
                )
            return quant

        if scale_count != zero_point_count or scale_count == 0:
            self.fail(
                "quantization_lengths",
                "quantization scales and zero points must be non-empty and paired",
                {
                    "tensor_index": record["index"],
                    "scale_count": scale_count,
                    "zero_point_count": zero_point_count,
                },
            )
            return quant
        if any(not math.isfinite(scale) or scale <= 0.0 for scale in scales):
            self.fail(
                "quantization_scales",
                "quantization scales must be finite and positive",
                {
                    "tensor_index": record["index"],
                    "scales": [
                        scale if math.isfinite(scale) else None for scale in scales
                    ],
                },
            )
        shape = record["shape"]
        if scale_count > 1:
            if qdim < 0 or qdim >= len(shape) or scale_count != shape[qdim]:
                self.fail(
                    "quantization_dimension",
                    "per-axis quantization dimension does not match tensor shape",
                    {
                        "tensor_index": record["index"],
                        "quantized_dimension": qdim,
                        "scale_count": scale_count,
                        "shape": shape,
                    },
                )
        if record["type"] == "INT8" and any(
            zero_point < -128 or zero_point > 127 for zero_point in zero_points
        ):
            self.fail(
                "int8_zero_points",
                "INT8 zero points must fit the signed 8-bit range",
                {"tensor_index": record["index"], "zero_points": zero_points},
            )
        if record["type"] == "INT32" and any(
            zero_point < -(2**31) or zero_point > 2**31 - 1
            for zero_point in zero_points
        ):
            self.fail(
                "int32_zero_points",
                "INT32 zero points must fit the signed 32-bit range",
                {"tensor_index": record["index"], "zero_points": zero_points},
            )
        return quant

    def _read_opcodes(self) -> None:
        if self.model is None:
            return
        builtin_names = _builtin_names(self.schema)
        opcode_count = int(self.model.OperatorCodesLength())
        for index in range(opcode_count):
            code = self.model.OperatorCodes(index)
            if code is None:
                self.fail(
                    "opcode_table",
                    "operator code table is missing",
                    {"opcode_index": index},
                    unresolved=True,
                )
                continue
            modern_builtin_code = int(code.BuiltinCode())
            deprecated_builtin_code = int(code.DeprecatedBuiltinCode())
            # This mirrors TFLM's schema-compatibility helper: old generated
            # models may carry the meaningful code in either field.
            builtin_code = max(modern_builtin_code, deprecated_builtin_code)
            custom_code = _safe_decode(code.CustomCode())
            name = builtin_names.get(builtin_code, "UNKNOWN")
            version = int(code.Version())
            record = {
                "index": index,
                "builtin_code": builtin_code,
                "builtin_code_field": modern_builtin_code,
                "deprecated_builtin_code": deprecated_builtin_code,
                "name": name,
                "version": version,
                "custom_code": custom_code or None,
            }
            self.opcode_records.append(record)
            if custom_code or name in {"CUSTOM", "DELEGATE", "UNKNOWN"}:
                self.fail(
                    "operator_allowlist",
                    "custom, delegate, or unknown operator cannot establish support",
                    record,
                    unresolved=True,
                )
            elif name not in self._ALLOWED_VERSIONS:
                self.fail(
                    "operator_allowlist",
                    "operator is outside the bounded static allowlist",
                    record,
                    unresolved=True,
                )
            elif version not in self._ALLOWED_VERSIONS[name]:
                self.fail(
                    "operator_version",
                    "operator version is outside the checked allowlist",
                    {"operator": name, "version": version},
                    unresolved=True,
                )

    def _validate_io(self) -> None:
        if self.subgraph is None:
            return
        input_count = int(self.subgraph.InputsLength())
        output_count = int(self.subgraph.OutputsLength())
        if input_count != 1 or output_count != 1:
            self.fail(
                "io_count",
                "exactly one model input and one model output are required",
                {"inputs": input_count, "outputs": output_count},
                unresolved=True,
            )
            return
        input_index = int(self.subgraph.Inputs(0))
        output_index = int(self.subgraph.Outputs(0))
        input_tensor = self.tensor_by_index.get(input_index)
        output_tensor = self.tensor_by_index.get(output_index)
        if input_tensor is None or output_tensor is None:
            self.fail(
                "io_tensor_indices",
                "model input/output references an unknown tensor",
                {"input": input_index, "output": output_index},
                unresolved=True,
            )
            return
        io_ok = True
        if tuple(input_tensor["shape"]) != EXPECTED_INPUT_SHAPE:
            io_ok = False
            self.fail(
                "input_shape",
                "input shape does not match the experimental host tensor contract",
                {
                    "actual": input_tensor["shape"],
                    "required": list(EXPECTED_INPUT_SHAPE),
                },
                unresolved=True,
            )
        if tuple(output_tensor["shape"]) != EXPECTED_OUTPUT_SHAPE:
            io_ok = False
            self.fail(
                "output_shape",
                "output shape does not match the experimental host tensor contract",
                {
                    "actual": output_tensor["shape"],
                    "required": list(EXPECTED_OUTPUT_SHAPE),
                },
                unresolved=True,
            )
        if input_tensor["type"] != "INT8" or output_tensor["type"] != "INT8":
            io_ok = False
            self.fail(
                "io_dtype",
                "model input and output must both be INT8",
                {
                    "input": input_tensor["type"],
                    "output": output_tensor["type"],
                },
            )
        for role, tensor in (("input", input_tensor), ("output", output_tensor)):
            quant = tensor["quantization"]
            if not quant["per_tensor"]:
                io_ok = False
                self.fail(
                    "io_quantization",
                    "model input/output quantization must be per-tensor",
                    {"role": role, "tensor_index": tensor["index"]},
                )
        if io_ok:
            self.pass_check(
                "host_io_contract",
                "static INT8 I/O matches the experimental host-only shape contract",
                {
                    "input_shape": list(EXPECTED_INPUT_SHAPE),
                    "output_shape": list(EXPECTED_OUTPUT_SHAPE),
                    "layout": "NHWC",
                },
            )

    def _tensor(self, index: int, role: str) -> dict[str, Any] | None:
        if index < 0:
            self.fail(
                "tensor_reference",
                "optional or missing tensor references are not accepted for this profile",
                {"role": role, "tensor_index": index},
                unresolved=True,
            )
            return None
        tensor = self.tensor_by_index.get(index)
        if tensor is None:
            self.fail(
                "tensor_reference",
                "operator references an unknown tensor",
                {"role": role, "tensor_index": index},
                unresolved=True,
            )
        return tensor

    def _require_types(
        self, tensors: Iterable[dict[str, Any] | None], expected: str, role: str
    ) -> bool:
        valid = True
        for tensor in tensors:
            if tensor is None:
                valid = False
            elif tensor["type"] != expected:
                valid = False
                self.fail(
                    "operator_tensor_type",
                    f"{role} tensor must be {expected}",
                    {"tensor_index": tensor["index"], "actual": tensor["type"]},
                )
        return valid

    def _require_options(
        self, op: Any, operator_index: int, expected_name: str
    ) -> Any | None:
        expected = _enum_value(self.schema.BuiltinOptions, expected_name)
        actual = int(op.BuiltinOptionsType())
        if actual != expected:
            self.fail(
                "operator_options",
                "operator options union does not match the expected schema table",
                {
                    "operator_index": operator_index,
                    "expected": expected_name,
                    "actual": actual,
                },
                unresolved=True,
            )
            return None
        table = op.BuiltinOptions()
        if table is None:
            self.fail(
                "operator_options",
                "operator options table is missing",
                {"operator_index": operator_index},
                unresolved=True,
            )
            return None
        option_class = getattr(self.schema, expected_name, None)
        if option_class is None:
            self.fail(
                "operator_options",
                "installed schema does not expose the required option table",
                {"option_table": expected_name},
                unresolved=True,
            )
            return None
        options = option_class()
        options.Init(table.Bytes, table.Pos)
        return options

    def _require_no_options(self, op: Any, operator_index: int) -> bool:
        expected = _enum_value(self.schema.BuiltinOptions, "NONE")
        actual = int(op.BuiltinOptionsType())
        if actual != expected:
            self.fail(
                "operator_options",
                "this operator must not carry an unknown options union",
                {"operator_index": operator_index, "actual": actual},
                unresolved=True,
            )
            return False
        return True

    def _check_activation(
        self, value: int, operator_index: int, operator_name: str
    ) -> bool:
        allowed = {
            _enum_value(self.schema.ActivationFunctionType, "NONE"),
            _enum_value(self.schema.ActivationFunctionType, "RELU"),
            _enum_value(self.schema.ActivationFunctionType, "RELU6"),
        }
        if value not in allowed:
            self.fail(
                "fused_activation",
                "unsupported fused activation mode",
                {
                    "operator_index": operator_index,
                    "operator": operator_name,
                    "activation": value,
                },
                unresolved=True,
            )
            return False
        return True

    def _check_activation_quant(
        self, tensor: dict[str, Any], role: str
    ) -> bool:
        quant = tensor["quantization"]
        valid = quant["per_tensor"]
        if not valid:
            self.fail(
                "activation_quantization",
                "activation tensors must use per-tensor quantization",
                {"role": role, "tensor_index": tensor["index"]},
            )
        if len(quant["scales"]) != 1 or any(
            not math.isfinite(scale) or scale <= 0.0
            for scale in quant["scales"]
        ):
            valid = False
        return valid

    def _check_activation_edge(
        self,
        input_tensor: dict[str, Any],
        output_tensor: dict[str, Any],
        operator_index: int,
        operator_name: str,
        *,
        require_same_quantization: bool,
    ) -> bool:
        valid = self._check_activation_quant(input_tensor, f"{operator_name} input")
        valid = self._check_activation_quant(
            output_tensor, f"{operator_name} output"
        ) and valid
        if not require_same_quantization or not valid:
            return valid
        input_quant = input_tensor["quantization"]
        output_quant = output_tensor["quantization"]
        same_scale = _float32_close(
            input_quant["scales"][0], output_quant["scales"][0]
        )
        same_zero_point = input_quant["zero_points"] == output_quant["zero_points"]
        if not same_scale or not same_zero_point:
            self.fail(
                "activation_quantization_relation",
                "AVERAGE_POOL_2D and RESHAPE must preserve activation scale and zero point",
                {
                    "operator_index": operator_index,
                    "operator": operator_name,
                    "input_scale": input_quant["scales"][0],
                    "output_scale": output_quant["scales"][0],
                    "input_zero_point": input_quant["zero_points"][0],
                    "output_zero_point": output_quant["zero_points"][0],
                },
            )
            return False
        return valid

    def _check_weight_quant(
        self, tensor: dict[str, Any], role: str, expected_axis: int
    ) -> bool:
        raw = self._buffer_bytes(tensor["buffer"])
        bytes_per_element = 1 if tensor["type"] == "INT8" else 4
        expected_bytes = _product(tensor["shape"]) * bytes_per_element
        if raw is None or len(raw) != expected_bytes:
            self.fail(
                "constant_buffer_size",
                "constant weight/bias buffer does not match its static tensor shape",
                {
                    "role": role,
                    "tensor_index": tensor["index"],
                    "actual_bytes": None if raw is None else len(raw),
                    "expected_bytes": expected_bytes,
                },
                unresolved=True,
            )
            return False
        quant = tensor["quantization"]
        valid = quant["present"] and bool(quant["scales"])
        if not valid:
            self.fail(
                "weight_quantization",
                "weight/bias tensor lacks usable quantization metadata",
                {"role": role, "tensor_index": tensor["index"]},
            )
            return False
        if any(zero_point != 0 for zero_point in quant["zero_points"]):
            self.fail(
                "weight_zero_point",
                "integer weight/bias zero points must be zero",
                {"role": role, "tensor_index": tensor["index"]},
            )
            valid = False
        if len(quant["scales"]) > 1 and quant["quantized_dimension"] != expected_axis:
            self.fail(
                "weight_quantized_dimension",
                "per-axis weight quantization uses an unexpected dimension",
                {
                    "role": role,
                    "tensor_index": tensor["index"],
                    "actual": quant["quantized_dimension"],
                    "expected": expected_axis,
                },
            )
            valid = False
        return valid

    def _check_bias_scale_relation(
        self,
        input_tensor: dict[str, Any],
        weight_tensor: dict[str, Any],
        bias_tensor: dict[str, Any],
        operator_index: int,
        operator_name: str,
    ) -> bool:
        input_quant = input_tensor["quantization"]
        weight_quant = weight_tensor["quantization"]
        bias_quant = bias_tensor["quantization"]
        if not self._check_activation_quant(
            input_tensor, f"{operator_name} input"
        ):
            return False
        scales = weight_quant["scales"]
        actual_scales = bias_quant["scales"]
        if not scales or not actual_scales:
            return False
        expected_scales = [input_quant["scales"][0] * scale for scale in scales]
        if len(actual_scales) != len(expected_scales) or any(
            not _float32_close(actual, expected)
            for actual, expected in zip(actual_scales, expected_scales)
        ):
            self.fail(
                "bias_quantization_relation",
                "bias scales must equal input scale multiplied by weight scales "
                "within one float32 ULP",
                {
                    "operator_index": operator_index,
                    "operator": operator_name,
                    "expected_scales": [
                        _float32_bits(scale) for scale in expected_scales
                    ],
                    "actual_scales": [_float32_bits(scale) for scale in actual_scales],
                },
            )
            return False
        return True

    def _record_mac(
        self,
        operator_index: int,
        operator_name: str,
        version: int,
        macs: int,
        formula: str,
        details: dict[str, Any],
    ) -> None:
        self.total_macs += int(macs)
        self.mac_breakdown.append(
            {
                "operator_index": operator_index,
                "operator": operator_name,
                "version": version,
                "macs": int(macs),
                "formula": formula,
                **details,
            }
        )

    def _spatial_output(
        self,
        input_size: int,
        kernel: int,
        stride: int,
        padding: int,
        dilation: int = 1,
    ) -> int | None:
        if input_size <= 0 or kernel <= 0 or stride <= 0 or dilation <= 0:
            return None
        try:
            same = _enum_value(self.schema.Padding, "SAME")
            valid = _enum_value(self.schema.Padding, "VALID")
        except Exception:
            return None
        effective_kernel = (kernel - 1) * dilation + 1
        if padding == same:
            return (input_size + stride - 1) // stride
        if padding == valid:
            if input_size < effective_kernel:
                return 0
            return (input_size - effective_kernel) // stride + 1
        return None

    def _validate_graph(self) -> None:
        if self.subgraph is None:
            return
        operator_count = int(self.subgraph.OperatorsLength())
        if operator_count == 0:
            self.fail("operator_count", "subgraph has no operators", unresolved=True)
            self.macs_complete = False
            return
        for operator_index in range(operator_count):
            op = self.subgraph.Operators(operator_index)
            if op is None:
                self.fail(
                    "operator_table",
                    "operator table is missing",
                    {"operator_index": operator_index},
                    unresolved=True,
                )
                self.macs_complete = False
                continue
            opcode_index = int(op.OpcodeIndex())
            if opcode_index < 0 or opcode_index >= len(self.opcode_records):
                self.fail(
                    "operator_opcode_reference",
                    "operator references an unknown opcode",
                    {"operator_index": operator_index, "opcode_index": opcode_index},
                    unresolved=True,
                )
                self.macs_complete = False
                continue
            opcode = self.opcode_records[opcode_index]
            name = opcode["name"]
            version = opcode["version"]
            inputs = [int(op.Inputs(i)) for i in range(int(op.InputsLength()))]
            outputs = [int(op.Outputs(i)) for i in range(int(op.OutputsLength()))]
            operation: dict[str, Any] = {
                "index": operator_index,
                "opcode_index": opcode_index,
                "operator": name,
                "version": version,
                "inputs": inputs,
                "outputs": outputs,
                "builtin_options_type": int(op.BuiltinOptionsType()),
            }
            self.operations.append(operation)
            options2_type_method = getattr(op, "BuiltinOptions2Type", None)
            if not callable(options2_type_method):
                self.fail(
                    "operator_options2",
                    "installed schema does not expose the secondary options mode",
                    {"operator_index": operator_index},
                    unresolved=True,
                )
                self.macs_complete = False
                continue
            options2_type = int(options2_type_method())
            if options2_type != 0:
                self.fail(
                    "operator_options2",
                    "secondary operator options mode is not modeled",
                    {
                        "operator_index": operator_index,
                        "builtin_options2_type": options2_type,
                    },
                    unresolved=True,
                )
                self.macs_complete = False
                continue
            if name == "CONV_2D":
                valid = self._validate_conv(op, operation, depthwise=False)
            elif name == "DEPTHWISE_CONV_2D":
                valid = self._validate_conv(op, operation, depthwise=True)
            elif name == "AVERAGE_POOL_2D":
                valid = self._validate_pool(op, operation)
            elif name in {"RELU", "RELU6"}:
                valid = self._validate_relu(op, operation)
            elif name == "RESHAPE":
                valid = self._validate_reshape(op, operation)
            elif name == "FULLY_CONNECTED":
                valid = self._validate_fully_connected(op, operation)
            else:
                self.fail(
                    "operator_allowlist",
                    "operator cannot be counted or treated as TFLM-supported",
                    {"operator_index": operator_index, "operator": name},
                    unresolved=True,
                )
                valid = False
            if not valid:
                self.macs_complete = False

    def _validate_conv(
        self, op: Any, operation: dict[str, Any], *, depthwise: bool
    ) -> bool:
        name = operation["operator"]
        index = operation["index"]
        inputs = operation["inputs"]
        outputs = operation["outputs"]
        if len(inputs) != 3 or len(outputs) != 1:
            self.fail(
                "operator_arity",
                "convolution requires input, weights, bias, and one output",
                {"operator_index": index, "inputs": inputs, "outputs": outputs},
                unresolved=True,
            )
            return False
        option_name = "DepthwiseConv2DOptions" if depthwise else "Conv2DOptions"
        options = self._require_options(op, index, option_name)
        input_tensor = self._tensor(inputs[0], "convolution input")
        weight_tensor = self._tensor(inputs[1], "convolution weights")
        bias_tensor = self._tensor(inputs[2], "convolution bias")
        output_tensor = self._tensor(outputs[0], "convolution output")
        if options is None or None in (
            input_tensor,
            weight_tensor,
            bias_tensor,
            output_tensor,
        ):
            return False
        if not self._require_types(
            (input_tensor, weight_tensor, output_tensor), "INT8", "convolution"
        ) or not self._require_types((bias_tensor,), "INT32", "convolution bias"):
            return False
        valid = self._check_activation_edge(
            input_tensor,
            output_tensor,
            index,
            name,
            require_same_quantization=False,
        )
        activation = int(options.FusedActivationFunction())
        valid = self._check_activation(activation, index, name) and valid
        padding = int(options.Padding())
        stride_h = int(options.StrideH())
        stride_w = int(options.StrideW())
        dilation_h = int(options.DilationHFactor())
        dilation_w = int(options.DilationWFactor())
        if stride_h <= 0 or stride_w <= 0 or dilation_h <= 0 or dilation_w <= 0:
            self.fail(
                "convolution_options",
                "convolution strides and dilations must be positive",
                {"operator_index": index},
            )
            valid = False
        input_shape = input_tensor["shape"]
        weight_shape = weight_tensor["shape"]
        output_shape = output_tensor["shape"]
        if len(input_shape) != 4 or len(weight_shape) != 4 or len(output_shape) != 4:
            self.fail(
                "convolution_rank",
                "convolution tensors must be rank-4 NHWC/HWIO-compatible tensors",
                {
                    "operator_index": index,
                    "input_shape": input_shape,
                    "weight_shape": weight_shape,
                    "output_shape": output_shape,
                },
                unresolved=True,
            )
            return False
        if depthwise:
            input_channels = input_shape[3]
            depth_multiplier = int(options.DepthMultiplier())
            expected_output_channels = input_channels * depth_multiplier
            kernel_h, kernel_w = weight_shape[1], weight_shape[2]
            weights_ok = (
                weight_shape[0] == 1
                and weight_shape[3] == expected_output_channels
            )
        else:
            input_channels = input_shape[3]
            expected_output_channels = weight_shape[0]
            kernel_h, kernel_w = weight_shape[1], weight_shape[2]
            weights_ok = weight_shape[3] == input_channels
        if not weights_ok or expected_output_channels != output_shape[3]:
            self.fail(
                "convolution_channels",
                "convolution channels/groups are not statically unambiguous",
                {
                    "operator_index": index,
                    "input_channels": input_channels,
                    "weight_shape": weight_shape,
                    "output_channels": output_shape[3],
                },
                unresolved=True,
            )
            valid = False
        expected_h = self._spatial_output(
            input_shape[1], kernel_h, stride_h, padding, dilation_h
        )
        expected_w = self._spatial_output(
            input_shape[2], kernel_w, stride_w, padding, dilation_w
        )
        expected_shape = [input_shape[0], expected_h, expected_w, expected_output_channels]
        if expected_h is None or expected_w is None or output_shape != expected_shape:
            self.fail(
                "convolution_output_shape",
                "convolution output shape does not follow its options",
                {
                    "operator_index": index,
                    "actual": output_shape,
                    "expected": expected_shape,
                },
                unresolved=True,
            )
            valid = False
        if _product(bias_tensor["shape"]) != expected_output_channels:
            self.fail(
                "convolution_bias_shape",
                "convolution bias length must equal output channels",
                {"operator_index": index, "shape": bias_tensor["shape"]},
                unresolved=True,
            )
            valid = False
        weight_axis = 3 if depthwise else 0
        valid = (
            self._check_weight_quant(weight_tensor, "convolution weights", weight_axis)
            and self._check_weight_quant(bias_tensor, "convolution bias", 0)
            and self._check_bias_scale_relation(
                input_tensor,
                weight_tensor,
                bias_tensor,
                index,
                name,
            )
            and valid
        )
        if valid:
            output_elements = _product(output_shape)
            if depthwise:
                macs = output_elements * kernel_h * kernel_w
                formula = "output_elements * kernel_h * kernel_w"
            else:
                macs = output_elements * kernel_h * kernel_w * input_channels
                formula = (
                    "output_elements * kernel_h * kernel_w * input_channels "
                    "(groups=1)"
                )
            self._record_mac(
                index,
                name,
                operation["version"],
                macs,
                formula,
                {
                    "input_shape": input_shape,
                    "weight_shape": weight_shape,
                    "output_shape": output_shape,
                    "kernel": [kernel_h, kernel_w],
                    "input_channels": input_channels,
                    "padding": padding,
                    "stride": [stride_h, stride_w],
                    "dilation": [dilation_h, dilation_w],
                },
            )
        return valid

    def _validate_pool(self, op: Any, operation: dict[str, Any]) -> bool:
        index = operation["index"]
        inputs = operation["inputs"]
        outputs = operation["outputs"]
        if len(inputs) != 1 or len(outputs) != 1:
            self.fail(
                "operator_arity",
                "average pooling requires one input and one output",
                {"operator_index": index},
                unresolved=True,
            )
            return False
        options = self._require_options(op, index, "Pool2DOptions")
        input_tensor = self._tensor(inputs[0], "pool input")
        output_tensor = self._tensor(outputs[0], "pool output")
        if options is None or input_tensor is None or output_tensor is None:
            return False
        valid = self._require_types((input_tensor, output_tensor), "INT8", "pool")
        valid = self._check_activation_edge(
            input_tensor,
            output_tensor,
            index,
            operation["operator"],
            require_same_quantization=True,
        ) and valid
        valid = self._check_activation(
            int(options.FusedActivationFunction()), index, operation["operator"]
        ) and valid
        input_shape = input_tensor["shape"]
        output_shape = output_tensor["shape"]
        filter_h = int(options.FilterHeight())
        filter_w = int(options.FilterWidth())
        stride_h = int(options.StrideH())
        stride_w = int(options.StrideW())
        expected_h = (
            self._spatial_output(input_shape[1], filter_h, stride_h, int(options.Padding()))
            if len(input_shape) == 4
            else None
        )
        expected_w = (
            self._spatial_output(input_shape[2], filter_w, stride_w, int(options.Padding()))
            if len(input_shape) == 4
            else None
        )
        expected_shape = (
            [input_shape[0], expected_h, expected_w, input_shape[3]]
            if len(input_shape) == 4
            else None
        )
        if expected_shape is None or output_shape != expected_shape:
            self.fail(
                "pool_output_shape",
                "average-pool output shape does not follow its options",
                {
                    "operator_index": index,
                    "actual": output_shape,
                    "expected": expected_shape,
                },
                unresolved=True,
            )
            valid = False
        if filter_h <= 0 or filter_w <= 0 or stride_h <= 0 or stride_w <= 0:
            self.fail(
                "pool_options",
                "pool filter and stride values must be positive",
                {"operator_index": index},
            )
            valid = False
        return valid

    def _validate_relu(self, op: Any, operation: dict[str, Any]) -> bool:
        index = operation["index"]
        inputs = operation["inputs"]
        outputs = operation["outputs"]
        if len(inputs) != 1 or len(outputs) != 1:
            self.fail(
                "operator_arity",
                "activation requires one input and one output",
                {"operator_index": index},
                unresolved=True,
            )
            return False
        input_tensor = self._tensor(inputs[0], "activation input")
        output_tensor = self._tensor(outputs[0], "activation output")
        if input_tensor is None or output_tensor is None:
            return False
        valid = self._require_types(
            (input_tensor, output_tensor), "INT8", "activation"
        )
        valid = self._check_activation_edge(
            input_tensor,
            output_tensor,
            index,
            operation["operator"],
            require_same_quantization=False,
        ) and valid
        valid = self._require_no_options(op, index) and valid
        if input_tensor["shape"] != output_tensor["shape"]:
            self.fail(
                "activation_shape",
                "activation must preserve tensor shape",
                {"operator_index": index},
                unresolved=True,
            )
            valid = False
        return valid

    def _validate_reshape(self, op: Any, operation: dict[str, Any]) -> bool:
        index = operation["index"]
        inputs = operation["inputs"]
        outputs = operation["outputs"]
        if len(inputs) != 2 or len(outputs) != 1:
            self.fail(
                "operator_arity",
                "reshape requires data, shape, and one output",
                {"operator_index": index},
                unresolved=True,
            )
            return False
        data_tensor = self._tensor(inputs[0], "reshape data")
        shape_tensor = self._tensor(inputs[1], "reshape shape")
        output_tensor = self._tensor(outputs[0], "reshape output")
        if data_tensor is None or shape_tensor is None or output_tensor is None:
            return False
        valid = self._require_types((data_tensor, output_tensor), "INT8", "reshape")
        valid = self._check_activation_edge(
            data_tensor,
            output_tensor,
            index,
            operation["operator"],
            require_same_quantization=True,
        ) and valid
        valid = self._require_types((shape_tensor,), "INT32", "reshape shape") and valid
        options_type = int(op.BuiltinOptionsType())
        none_type = _enum_value(self.schema.BuiltinOptions, "NONE")
        reshape_type = _enum_value(self.schema.BuiltinOptions, "ReshapeOptions")
        if options_type not in {none_type, reshape_type}:
            self.fail(
                "reshape_options",
                "reshape options mode is not understood",
                {"operator_index": index, "actual": options_type},
                unresolved=True,
            )
            valid = False
        shape_values = self._reshape_shape_values(shape_tensor, index)
        if shape_values is None or shape_values != output_tensor["shape"]:
            self.fail(
                "reshape_shape",
                "reshape shape tensor does not exactly describe its output",
                {
                    "operator_index": index,
                    "shape_tensor": shape_values,
                    "output_shape": output_tensor["shape"],
                },
                unresolved=True,
            )
            valid = False
        if _product(data_tensor["shape"]) != _product(output_tensor["shape"]):
            self.fail(
                "reshape_element_count",
                "reshape must preserve the number of elements",
                {"operator_index": index},
                unresolved=True,
            )
            valid = False
        if options_type == reshape_type:
            table = op.BuiltinOptions()
            options = self.schema.ReshapeOptions()
            options.Init(table.Bytes, table.Pos)
            option_shape = [
                int(options.NewShape(i))
                for i in range(int(options.NewShapeLength()))
            ]
            if option_shape and option_shape != output_tensor["shape"]:
                self.fail(
                    "reshape_options_shape",
                    "ReshapeOptions.NewShape disagrees with the output shape",
                    {"operator_index": index, "new_shape": option_shape},
                    unresolved=True,
                )
                valid = False
        return valid

    def _buffer_bytes(self, buffer_index: int) -> bytes | None:
        if self.model is None or buffer_index < 0:
            return None
        if buffer_index >= int(self.model.BuffersLength()):
            return None
        buffer = self.model.Buffers(buffer_index)
        if buffer is None or buffer.DataLength() == 0:
            return b""
        try:
            return bytes(buffer.DataAsNumpy())
        except Exception:
            values = [int(buffer.Data(i)) for i in range(int(buffer.DataLength()))]
            return bytes(values)

    def _reshape_shape_values(
        self, shape_tensor: dict[str, Any], operator_index: int
    ) -> list[int] | None:
        raw = self._buffer_bytes(shape_tensor["buffer"])
        if raw is None or len(raw) % 4 != 0:
            self.fail(
                "reshape_shape_buffer",
                "reshape shape tensor buffer is absent or not INT32-aligned",
                {"operator_index": operator_index},
                unresolved=True,
            )
            return None
        count = len(raw) // 4
        try:
            return list(struct.unpack("<" + "i" * count, raw))
        except struct.error:
            self.fail(
                "reshape_shape_buffer",
                "reshape shape tensor buffer cannot be decoded as INT32",
                {"operator_index": operator_index},
                unresolved=True,
            )
            return None

    def _validate_fully_connected_options(
        self, options: Any, operator_index: int
    ) -> bool:
        """Validate the deliberately narrow FC option contract before MACs."""

        valid = True
        default_weights = _enum_value(
            self.schema.FullyConnectedOptionsWeightsFormat, "DEFAULT"
        )
        try:
            weights_format = int(options.WeightsFormat())
        except Exception as exc:
            self.fail(
                "fully_connected_options",
                "fully-connected weights format cannot be inspected",
                {"operator_index": operator_index, "error": str(exc)},
                unresolved=True,
            )
            valid = False
        else:
            if weights_format != default_weights:
                self.fail(
                    "fully_connected_weights_format",
                    "only the default fully-connected weight layout is understood",
                    {"operator_index": operator_index, "actual": weights_format},
                    unresolved=True,
                )
                valid = False

        for method_name, field_name in (
            ("KeepNumDims", "keep_num_dims"),
            ("AsymmetricQuantizeInputs", "asymmetric_quantize_inputs"),
        ):
            method = getattr(options, method_name, None)
            if not callable(method):
                self.fail(
                    "fully_connected_options",
                    f"installed schema cannot inspect {field_name}",
                    {"operator_index": operator_index, "field": field_name},
                    unresolved=True,
                )
                valid = False
                continue
            try:
                value = bool(method())
            except Exception as exc:
                self.fail(
                    "fully_connected_options",
                    f"fully-connected {field_name} cannot be read",
                    {
                        "operator_index": operator_index,
                        "field": field_name,
                        "error": str(exc),
                    },
                    unresolved=True,
                )
                valid = False
                continue
            if value:
                self.fail(
                    "fully_connected_options",
                    f"fully-connected {field_name}=true is outside the bounded profile",
                    {"operator_index": operator_index, "field": field_name},
                    unresolved=True,
                )
                valid = False

        bias_type_method = getattr(options, "QuantizedBiasType", None)
        expected_bias_type = getattr(self.schema.TensorType, "FLOAT32", None)
        if not callable(bias_type_method) or expected_bias_type is None:
            self.fail(
                "fully_connected_options",
                "installed schema cannot establish the supported quantized bias mode",
                {"operator_index": operator_index},
                unresolved=True,
            )
            valid = False
        else:
            try:
                bias_type = int(bias_type_method())
            except Exception as exc:
                self.fail(
                    "fully_connected_options",
                    "fully-connected quantized bias mode cannot be read",
                    {"operator_index": operator_index, "error": str(exc)},
                    unresolved=True,
                )
                valid = False
            else:
                if bias_type != int(expected_bias_type):
                    self.fail(
                        "fully_connected_options",
                        "non-default quantized bias type is outside the bounded profile",
                        {
                            "operator_index": operator_index,
                            "actual": bias_type,
                            "required": int(expected_bias_type),
                        },
                        unresolved=True,
                    )
                    valid = False

        # The pinned TFLM schema adds quant_spec at vtable offset 14, while
        # TensorFlow 2.20's generated Python class does not expose an accessor.
        # Inspect the generated accessor when present, otherwise inspect that
        # official FlatBuffer vtable slot; a non-empty opaque contract is never
        # silently counted as ordinary fully-connected quantization.
        quant_spec_length: int | None = None
        quant_spec_length_method = getattr(options, "QuantSpecLength", None)
        if callable(quant_spec_length_method):
            try:
                quant_spec_length = int(quant_spec_length_method())
            except Exception as exc:
                self.fail(
                    "fully_connected_options",
                    "fully-connected quant_spec cannot be read",
                    {"operator_index": operator_index, "error": str(exc)},
                    unresolved=True,
                )
                valid = False
        else:
            table = getattr(options, "_tab", None)
            offset_method = getattr(table, "Offset", None)
            vector_length_method = getattr(table, "VectorLen", None)
            table_position = getattr(table, "Pos", None)
            if (
                not callable(offset_method)
                or not callable(vector_length_method)
                or table_position is None
            ):
                self.fail(
                    "fully_connected_options",
                    "installed schema cannot inspect the pinned quant_spec field",
                    {"operator_index": operator_index},
                    unresolved=True,
                )
                valid = False
            else:
                try:
                    quant_spec_offset = int(offset_method(14))
                    quant_spec_length = (
                        0
                        if quant_spec_offset == 0
                        else int(
                            vector_length_method(
                                quant_spec_offset + int(table_position)
                            )
                        )
                    )
                except Exception as exc:
                    self.fail(
                        "fully_connected_options",
                        "fully-connected quant_spec cannot be inspected",
                        {"operator_index": operator_index, "error": str(exc)},
                        unresolved=True,
                    )
                    valid = False
        if quant_spec_length is not None and quant_spec_length != 0:
            self.fail(
                "fully_connected_options",
                "non-empty fully-connected quant_spec is outside the bounded profile",
                {
                    "operator_index": operator_index,
                    "quant_spec_bytes": quant_spec_length,
                },
                unresolved=True,
            )
            valid = False
        return valid

    def _validate_fully_connected(self, op: Any, operation: dict[str, Any]) -> bool:
        index = operation["index"]
        inputs = operation["inputs"]
        outputs = operation["outputs"]
        if len(inputs) != 3 or len(outputs) != 1:
            self.fail(
                "operator_arity",
                "fully connected requires input, weights, bias, and one output",
                {"operator_index": index, "inputs": inputs, "outputs": outputs},
                unresolved=True,
            )
            return False
        options = self._require_options(op, index, "FullyConnectedOptions")
        input_tensor = self._tensor(inputs[0], "fully-connected input")
        weight_tensor = self._tensor(inputs[1], "fully-connected weights")
        # The generated smoke graph uses the schema's optional-bias sentinel
        # (-1).  An absent bias is well-defined for FullyConnected and does
        # not make the MAC count ambiguous; an actual bias must be INT32.
        bias_tensor = (
            None
            if inputs[2] == -1
            else self._tensor(inputs[2], "fully-connected bias")
        )
        output_tensor = self._tensor(outputs[0], "fully-connected output")
        if (
            options is None
            or input_tensor is None
            or weight_tensor is None
            or output_tensor is None
        ):
            return False
        if not self._validate_fully_connected_options(options, index):
            return False
        input_shape = input_tensor["shape"]
        weight_shape = weight_tensor["shape"]
        output_shape = output_tensor["shape"]
        bounded_layout = (
            len(input_shape) == 2
            and input_shape[0] == 1
            and len(output_shape) == 2
            and output_shape[0] == 1
            and len(weight_shape) == 2
        )
        if not bounded_layout:
            self.fail(
                "fully_connected_profile",
                "bounded fully-connected profile requires input [1, depth], "
                "output [1, units], and weights [units, depth]",
                {
                    "operator_index": index,
                    "input_shape": input_shape,
                    "weight_shape": weight_shape,
                    "output_shape": output_shape,
                },
                unresolved=True,
            )
            return False
        valid = self._require_types(
            (input_tensor, weight_tensor, output_tensor), "INT8", "fully-connected"
        )
        valid = self._check_activation_edge(
            input_tensor,
            output_tensor,
            index,
            operation["operator"],
            require_same_quantization=False,
        ) and valid
        if bias_tensor is not None:
            valid = self._require_types((bias_tensor,), "INT32", "fully-connected bias") and valid
        valid = self._check_activation(
            int(options.FusedActivationFunction()), index, operation["operator"]
        ) and valid
        input_depth = _product(input_shape[1:])
        output_elements = _product(output_shape)
        if weight_shape[1] != input_depth or weight_shape[0] != output_elements:
            self.fail(
                "fully_connected_shape",
                "fully-connected weights do not match input depth/output elements",
                {
                    "operator_index": index,
                    "input_depth": input_depth,
                    "weight_shape": weight_shape,
                    "output_elements": output_elements,
                },
                unresolved=True,
            )
            valid = False
        if bias_tensor is not None and _product(bias_tensor["shape"]) != output_elements:
            self.fail(
                "fully_connected_bias_shape",
                "fully-connected bias length must equal output elements",
                {"operator_index": index, "shape": bias_tensor["shape"]},
                unresolved=True,
            )
            valid = False
        valid = self._check_weight_quant(
            weight_tensor, "fully-connected weights", 0
        ) and valid
        if bias_tensor is not None:
            valid = self._check_weight_quant(
                bias_tensor, "fully-connected bias", 0
            ) and valid
            valid = self._check_bias_scale_relation(
                input_tensor,
                weight_tensor,
                bias_tensor,
                index,
                operation["operator"],
            ) and valid
        if valid:
            macs = output_elements * input_depth
            self._record_mac(
                index,
                operation["operator"],
                operation["version"],
                macs,
                "output_elements * input_depth",
                {
                    "input_shape": input_shape,
                    "weight_shape": weight_shape,
                    "output_shape": output_shape,
                    "input_depth": input_depth,
                    "output_elements": output_elements,
                },
            )
        return valid

    def result(self) -> dict[str, Any]:
        structure_pass = not self.errors
        macs_complete = self.macs_complete and structure_pass
        total_macs = self.total_macs if macs_complete else None
        if not self.errors:
            self.pass_check(
                "graph_structure",
                "all operators, options, shapes, and integer metadata passed",
            )
        return {
            "schema": {
                "model_version": self.schema_version,
                "source": SCHEMA_SOURCE,
            },
            "operators": self.operations,
            "operator_codes": self.opcode_records,
            "macs": {
                "status": "COMPLETE" if macs_complete else "UNRESOLVED",
                "total": total_macs,
                "total_macs": total_macs,
                "limit": self.profile.max_macs,
                "breakdown": self.mac_breakdown,
                "scope": (
                    "CONV_2D/DEPTHWISE_CONV_2D/FULLY_CONNECTED multiply-accumulate "
                    "counts only; excludes pooling, activation, reshape, FFT/frontend, "
                    "firmware cycles, and timing"
                ),
            },
            "checks": self.checks,
            "errors": self.errors,
            "static_structure_pass": structure_pass,
        }


def _base_report(
    path: Path, data: bytes | None, profile: BudgetProfile
) -> dict[str, Any]:
    size = None if data is None else len(data)
    digest = None if data is None else hashlib.sha256(data).hexdigest()
    size_pass = size is not None and size <= profile.max_model_bytes
    return {
        "status": STATIC_FAIL,
        "runtime_status": RUNTIME_UNVERIFIED,
        "contract_status": CONTRACT_UNRESOLVED,
        "contract_discovery": CONTRACT_DISCOVERY,
        "scope": HOST_SCOPE,
        "profile": profile.as_dict(),
        "model": {
            "path": str(path),
            "exists": data is not None,
            "bytes": size,
            "size_bytes": size,
            "sha256": digest,
        },
        "evidence": {
            "budget_source": "docs/bai-toan-diem-danh.md, budget table line 99",
            "host_scope_source": "docs/host-prototype-contract.md",
            "schema_source": SCHEMA_SOURCE,
            "quantization_source": QUANTIZATION_SOURCE,
            "operator_version_policy": (
                "NEW_LOCAL_STATIC_ALLOWLIST; not a claim of TFLM version support; "
                "actual host invoke/runtime remains a separate oracle"
            ),
            "interpretation": (
                "the host tensor layout is experimental host-only evidence; final "
                "frontend/layout/runtime and board contract remain unresolved"
            ),
        },
        "memory_plan": _memory_plan(profile),
        "gates": {
            "model_bytes": {
                "status": "PASS" if size_pass else "FAIL",
                "actual": size,
                "limit": profile.max_model_bytes,
                "unit": "decimal_bytes",
            },
            "macs": {
                "status": "UNRESOLVED",
                "actual": None,
                "limit": profile.max_macs,
                "unit": "MACs_per_inference",
            },
            "sram": {
                "status": RUNTIME_UNVERIFIED,
                "target": profile.target_sram_bytes,
                "reason": "this static checker does not measure TFLM arena or total board SRAM",
            },
        },
        "checks": [],
        "errors": [],
        "macs": {
            "status": "UNRESOLVED",
            "total": None,
            "total_macs": None,
            "limit": profile.max_macs,
            "breakdown": [],
        },
        "static_structure_pass": False,
        "static_budget_pass": False,
    }


def check_model(
    model_path: str | Path, *, profile: str = "esp32s3-budget"
) -> dict[str, Any]:
    """Inspect a model and always return a JSON-serializable report.

    Missing files, malformed FlatBuffers, unsupported operators, and invalid
    metadata are represented as failed/unresolved reports instead of being
    treated as a zero-cost graph or raising through the CLI.
    """

    path = Path(model_path)
    try:
        selected_profile = get_profile(profile)
    except ValueError as exc:
        selected_profile = BudgetProfile(name=profile)
        report = _base_report(path, None, selected_profile)
        report["errors"].append(
            {"check": "profile", "status": "FAIL", "message": str(exc)}
        )
        report["checks"].append(
            {"name": "profile", "status": "FAIL", "message": str(exc)}
        )
        return report

    try:
        data = path.read_bytes()
    except (FileNotFoundError, IsADirectoryError, PermissionError, OSError) as exc:
        report = _base_report(path, None, selected_profile)
        message = f"cannot read model: {type(exc).__name__}: {exc}"
        report["errors"].append(
            {"check": "model_read", "status": "FAIL", "message": message}
        )
        report["checks"].append(
            {"name": "model_read", "status": "FAIL", "message": message}
        )
        return report

    report = _base_report(path, data, selected_profile)
    if len(data) <= selected_profile.max_model_bytes:
        report["checks"].append(
            {
                "name": "model_bytes",
                "status": "PASS",
                "message": "model bytes are within the decimal profile limit",
            }
        )
    else:
        report["checks"].append(
            {
                "name": "model_bytes",
                "status": "FAIL",
                "message": "model exceeds the decimal profile limit",
            }
        )
        report["errors"].append(
            {
                "check": "model_bytes",
                "status": "FAIL",
                "message": "model exceeds the decimal profile limit",
                "details": {
                    "actual": len(data),
                    "limit": selected_profile.max_model_bytes,
                },
            }
        )

    try:
        schema = _load_schema()
    except Exception as exc:
        message = (
            "TensorFlow generated schema is unavailable; static artifact check "
            f"cannot proceed: {type(exc).__name__}: {exc}"
        )
        report["checks"].append(
            {"name": "schema_import", "status": "UNRESOLVED", "message": message}
        )
        report["errors"].append(
            {"check": "schema_import", "status": "UNRESOLVED", "message": message}
        )
        return report

    analyzer = _Analyzer(data, schema, selected_profile)
    analyzer.run()
    analysis = analyzer.result()
    report["schema"] = analysis["schema"]
    report["operators"] = analysis["operators"]
    report["operator_codes"] = analysis["operator_codes"]
    report["checks"].extend(analysis["checks"])
    report["errors"].extend(analysis["errors"])
    report["static_structure_pass"] = analysis["static_structure_pass"]
    report["macs"] = analysis["macs"]
    macs = analysis["macs"]
    if macs["status"] == "COMPLETE":
        mac_pass = int(macs["total"]) <= selected_profile.max_macs
        report["gates"]["macs"] = {
            "status": "PASS" if mac_pass else "FAIL",
            "actual": macs["total"],
            "limit": selected_profile.max_macs,
            "unit": "MACs_per_inference",
        }
        if not mac_pass:
            report["errors"].append(
                {
                    "check": "macs",
                    "status": "FAIL",
                    "message": "MAC count exceeds the profile limit",
                    "details": {
                        "actual": macs["total"],
                        "limit": selected_profile.max_macs,
                    },
                }
            )
    else:
        mac_pass = False
        report["gates"]["macs"] = {
            "status": "UNRESOLVED",
            "actual": None,
            "limit": selected_profile.max_macs,
            "unit": "MACs_per_inference",
        }

    size_pass = len(data) <= selected_profile.max_model_bytes
    structure_pass = bool(analysis["static_structure_pass"])
    report["static_budget_pass"] = bool(size_pass and mac_pass and structure_pass)
    report["status"] = STATIC_PASS if report["static_budget_pass"] else STATIC_FAIL
    return report


__all__ = [
    "BudgetProfile",
    "CONTRACT_UNRESOLVED",
    "CONTRACT_DISCOVERY",
    "HOST_SCOPE",
    "RUNTIME_UNVERIFIED",
    "STATIC_FAIL",
    "STATIC_PASS",
    "check_model",
    "get_profile",
]
