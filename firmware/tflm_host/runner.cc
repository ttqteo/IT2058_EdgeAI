// Experimental host-only TFLite Micro runner.
//
// The model and I/O byte buffers are ordinary host storage.  Only the
// interpreter tensor arena is bounded by this runner; this is not an ESP32
// SRAM accounting or board-integration claim.

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <initializer_list>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>
#include <iostream>

#include "flatbuffers/flatbuffers.h"
#include "tensorflow/lite/micro/flatbuffer_utils.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {

constexpr std::size_t kMinArenaBytes = 8192;
constexpr std::size_t kMaxArenaBytes = 131072;
constexpr std::size_t kModelMaxBytes = 300000;
constexpr std::size_t kInputBytes = 1U * 298U * 40U * 1U;
constexpr std::size_t kOutputBytes = 1U * 192U;
constexpr char kSourceSha[] =
    "230c9d9531bd9b08d445b4f2ed0760756ffd5f78";
constexpr char kIdentifier[] = "TFL3";

struct alignas(16) ArenaWord {
  uint8_t bytes[16];
};

static_assert(alignof(ArenaWord) >= 16,
              "the host arena storage must provide TFLM's 16-byte alignment");

struct QuantizationInfo {
  float scale = 0.0F;
  int32_t zero_point = 0;
};

struct ModelInfo {
  const tflite::Model* model = nullptr;
  std::size_t model_bytes = 0;
  int32_t input_tensor_index = -1;
  int32_t output_tensor_index = -1;
  QuantizationInfo input_quantization;
  QuantizationInfo output_quantization;
  std::vector<std::string> operations;
  uint32_t schema_version = 0;
};

std::string JsonEscape(const std::string& value) {
  std::string escaped;
  escaped.reserve(value.size());
  for (const unsigned char character : value) {
    switch (character) {
      case '"':
        escaped += "\\\"";
        break;
      case '\\':
        escaped += "\\\\";
        break;
      case '\n':
        escaped += "\\n";
        break;
      case '\r':
        escaped += "\\r";
        break;
      case '\t':
        escaped += "\\t";
        break;
      default:
        if (character < 0x20U) {
          escaped += "?";
        } else {
          escaped.push_back(static_cast<char>(character));
        }
        break;
    }
  }
  return escaped;
}

void PrintFailure(const std::string& status, const std::string& message,
                  const std::optional<std::size_t>& arena_bytes) {
  std::cout << "{\"status\":\"" << JsonEscape(status)
            << "\",\"message\":\"" << JsonEscape(message)
            << "\",\"arena_capacity_bytes\":";
  if (arena_bytes.has_value()) {
    std::cout << arena_bytes.value();
  } else {
    std::cout << "null";
  }
  std::cout << ",\"arena_used_bytes\":null,\"tflm_source_sha\":\""
            << kSourceSha
            << "\",\"scope\":\"EXPERIMENTAL_HOST_ONLY\"}\n";
}

void PrintSuccess(const ModelInfo& info, std::size_t arena_bytes,
                  std::size_t arena_used_bytes) {
  std::cout << std::setprecision(std::numeric_limits<float>::max_digits10);
  std::cout << "{\"status\":\"PASS\",\"message\":\"TFLM Invoke completed\""
            << ",\"arena_capacity_bytes\":" << arena_bytes
            << ",\"arena_used_bytes\":" << arena_used_bytes
            << ",\"model_bytes\":" << info.model_bytes
            << ",\"input_bytes\":" << kInputBytes
            << ",\"output_bytes\":" << kOutputBytes
            << ",\"schema_version\":" << info.schema_version
            << ",\"tflm_source_sha\":\"" << kSourceSha << "\""
            << ",\"scope\":\"EXPERIMENTAL_HOST_ONLY\""
            << ",\"input\":{\"dtype\":\"int8\",\"shape\":[1,298,40,1]"
            << ",\"scale\":" << info.input_quantization.scale
            << ",\"zero_point\":" << info.input_quantization.zero_point
            << "}"
            << ",\"output\":{\"dtype\":\"int8\",\"shape\":[1,192]"
            << ",\"scale\":" << info.output_quantization.scale
            << ",\"zero_point\":" << info.output_quantization.zero_point
            << "},\"operations\":[";
  for (std::size_t index = 0; index < info.operations.size(); ++index) {
    if (index != 0) {
      std::cout << ',';
    }
    std::cout << "\"" << JsonEscape(info.operations[index]) << "\"";
  }
  std::cout << "]}\n";
}

bool ReadFile(const std::string& path, std::size_t max_bytes,
              std::vector<uint8_t>* bytes, std::string* error) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) {
    *error = "cannot open input file";
    return false;
  }
  const std::streampos end = stream.tellg();
  if (end < 0) {
    *error = "cannot determine input file length";
    return false;
  }
  const uintmax_t size = static_cast<uintmax_t>(end);
  if (size > max_bytes) {
    *error = "input file exceeds the bounded file-size limit";
    return false;
  }
  if (size > static_cast<uintmax_t>(std::numeric_limits<std::size_t>::max())) {
    *error = "input file length does not fit host size_t";
    return false;
  }
  bytes->assign(static_cast<std::size_t>(size), 0U);
  stream.seekg(0, std::ios::beg);
  if (size != 0U &&
      !stream.read(reinterpret_cast<char*>(bytes->data()),
                   static_cast<std::streamsize>(size))) {
    *error = "cannot read the complete input file";
    bytes->clear();
    return false;
  }
  return true;
}

bool ParseArenaSize(const std::string& text, std::size_t* arena_bytes,
                    std::string* error) {
  if (text.empty()) {
    *error = "arena size must be a decimal integer";
    return false;
  }
  uintmax_t parsed = 0;
  const char* begin = text.data();
  const char* end = begin + text.size();
  const std::from_chars_result result = std::from_chars(begin, end, parsed, 10);
  if (result.ec != std::errc() || result.ptr != end ||
      parsed > static_cast<uintmax_t>(std::numeric_limits<std::size_t>::max())) {
    *error = "arena size must be an unsigned decimal integer";
    return false;
  }
  *arena_bytes = static_cast<std::size_t>(parsed);
  if (*arena_bytes < kMinArenaBytes || *arena_bytes > kMaxArenaBytes) {
    *error = "arena size must be within the inclusive 8192..131072-byte host bound";
    return false;
  }
  return true;
}

bool IsExpectedShape(const flatbuffers::Vector<int32_t>* shape,
                     std::initializer_list<int32_t> expected) {
  if (shape == nullptr || shape->size() != expected.size()) {
    return false;
  }
  std::size_t index = 0;
  for (const int32_t dimension : expected) {
    if (shape->Get(index) != dimension) {
      return false;
    }
    ++index;
  }
  return true;
}

bool ReadQuantization(const tflite::Tensor* tensor, QuantizationInfo* result,
                      std::string* error) {
  if (tensor == nullptr || tensor->quantization() == nullptr) {
    *error = "I/O tensor has no quantization parameters";
    return false;
  }
  const tflite::QuantizationParameters* quantization = tensor->quantization();
  const flatbuffers::Vector<float>* scales = quantization->scale();
  const flatbuffers::Vector<int64_t>* zero_points = quantization->zero_point();
  if (scales == nullptr || zero_points == nullptr || scales->size() != 1U ||
      zero_points->size() != 1U) {
    *error = "I/O tensor must have exactly one per-tensor scale and zero point";
    return false;
  }
  const float scale = scales->Get(0);
  const int64_t zero_point = zero_points->Get(0);
  if (!std::isfinite(scale) || scale <= 0.0F || zero_point < -128 ||
      zero_point > 127) {
    *error = "I/O tensor quantization metadata is outside the supported finite range";
    return false;
  }
  result->scale = scale;
  result->zero_point = static_cast<int32_t>(zero_point);
  return true;
}

const char* BuiltinName(tflite::BuiltinOperator code) {
  switch (code) {
    case tflite::BuiltinOperator_CONV_2D:
      return "CONV_2D";
    case tflite::BuiltinOperator_DEPTHWISE_CONV_2D:
      return "DEPTHWISE_CONV_2D";
    case tflite::BuiltinOperator_AVERAGE_POOL_2D:
      return "AVERAGE_POOL_2D";
    case tflite::BuiltinOperator_RELU:
      return "RELU";
    case tflite::BuiltinOperator_RELU6:
      return "RELU6";
    case tflite::BuiltinOperator_RESHAPE:
      return "RESHAPE";
    case tflite::BuiltinOperator_FULLY_CONNECTED:
      return "FULLY_CONNECTED";
    default:
      return nullptr;
  }
}

bool SupportedVersion(tflite::BuiltinOperator code, int32_t version) {
  switch (code) {
    case tflite::BuiltinOperator_CONV_2D:
    case tflite::BuiltinOperator_DEPTHWISE_CONV_2D:
      return version >= 1 && version <= 3;
    case tflite::BuiltinOperator_AVERAGE_POOL_2D:
      return version >= 1 && version <= 2;
    case tflite::BuiltinOperator_RELU:
    case tflite::BuiltinOperator_RELU6:
      return version == 1;
    case tflite::BuiltinOperator_RESHAPE:
      return version >= 1 && version <= 2;
    case tflite::BuiltinOperator_FULLY_CONNECTED:
      return version >= 1 && version <= 4;
    default:
      return false;
  }
}

bool ValidTensorIndex(int32_t index, std::size_t tensor_count) {
  return index >= 0 && static_cast<std::size_t>(index) < tensor_count;
}

struct TensorMetadata {
  bool is_constant = false;
  std::size_t element_count = 0;
  std::size_t byte_count = 0;
};

bool IsSupportedTensorType(tflite::TensorType type) {
  return type == tflite::TensorType_INT8 || type == tflite::TensorType_INT32;
}

std::size_t BytesPerElement(tflite::TensorType type) {
  return type == tflite::TensorType_INT32 ? sizeof(int32_t) : sizeof(int8_t);
}

bool StaticShape(const tflite::Tensor* tensor, std::size_t* element_count,
                 std::string* error) {
  if (tensor == nullptr || tensor->shape() == nullptr ||
      tensor->shape()->size() == 0U) {
    *error = "every tensor must have a non-empty static shape";
    return false;
  }
  std::size_t product = 1U;
  for (std::size_t index = 0; index < tensor->shape()->size(); ++index) {
    const int32_t dimension = tensor->shape()->Get(index);
    if (dimension <= 0) {
      *error = "tensor shape dimensions must be positive and statically known";
      return false;
    }
    const std::size_t positive_dimension = static_cast<std::size_t>(dimension);
    if (product > std::numeric_limits<std::size_t>::max() /
                      positive_dimension) {
      *error = "tensor shape element count overflows host size_t";
      return false;
    }
    product *= positive_dimension;
  }
  *element_count = product;
  return true;
}

bool Float32Close(float left, float right) {
  if (!std::isfinite(left) || !std::isfinite(right)) {
    return false;
  }
  uint32_t left_bits = 0U;
  uint32_t right_bits = 0U;
  static_assert(sizeof(left_bits) == sizeof(left),
                "float32 quantization must have 32-bit storage");
  std::memcpy(&left_bits, &left, sizeof(left_bits));
  std::memcpy(&right_bits, &right, sizeof(right_bits));
  const uint32_t distance = left_bits >= right_bits
                                ? left_bits - right_bits
                                : right_bits - left_bits;
  return distance <= 1U;
}

bool ValidateTensorQuantization(const tflite::Tensor* tensor,
                                const TensorMetadata& metadata,
                                std::string* error) {
  const tflite::QuantizationParameters* quantization = tensor->quantization();
  if (quantization != nullptr &&
      quantization->details_type() != tflite::QuantizationDetails_NONE) {
    *error = "custom, blockwise, or multi-axis quantization details are unsupported";
    return false;
  }
  const flatbuffers::Vector<float>* scales =
      quantization == nullptr ? nullptr : quantization->scale();
  const flatbuffers::Vector<int64_t>* zero_points =
      quantization == nullptr ? nullptr : quantization->zero_point();
  const std::size_t scale_count = scales == nullptr ? 0U : scales->size();
  const std::size_t zero_point_count =
      zero_points == nullptr ? 0U : zero_points->size();
  if (scale_count == 0U && zero_point_count == 0U) {
    if (tensor->type() == tflite::TensorType_INT8) {
      *error = "every INT8 tensor requires finite positive quantization metadata";
      return false;
    }
    return true;
  }
  if (scale_count == 0U || scale_count != zero_point_count) {
    *error = "quantization scales and zero points must be non-empty and paired";
    return false;
  }
  for (std::size_t index = 0; index < scale_count; ++index) {
    const float scale = scales->Get(index);
    if (!std::isfinite(scale) || scale <= 0.0F) {
      *error = "quantization scales must be finite and positive";
      return false;
    }
  }
  if (scale_count > 1U) {
    const int32_t quantized_dimension = quantization->quantized_dimension();
    if (quantized_dimension < 0 ||
        static_cast<std::size_t>(quantized_dimension) >= tensor->shape()->size() ||
        scale_count != static_cast<std::size_t>(
                           tensor->shape()->Get(quantized_dimension))) {
      *error = "per-axis quantization dimension does not match tensor shape";
      return false;
    }
  }
  if (tensor->type() == tflite::TensorType_INT8) {
    for (std::size_t index = 0; index < zero_point_count; ++index) {
      const int64_t zero_point = zero_points->Get(index);
      if (zero_point < -128 || zero_point > 127) {
        *error = "INT8 zero points must fit the signed 8-bit range";
        return false;
      }
      if (metadata.is_constant && zero_point != 0) {
        *error = "constant INT8 weights must use symmetric zero point zero";
        return false;
      }
    }
    if (!metadata.is_constant && scale_count != 1U) {
      *error = "activation INT8 tensors must use per-tensor quantization";
      return false;
    }
  } else if (tensor->type() == tflite::TensorType_INT32) {
    for (std::size_t index = 0; index < zero_point_count; ++index) {
      const int64_t zero_point = zero_points->Get(index);
      if (zero_point < std::numeric_limits<int32_t>::min() ||
          zero_point > std::numeric_limits<int32_t>::max()) {
        *error = "INT32 zero points must fit the signed 32-bit range";
        return false;
      }
    }
  }
  return true;
}

bool ValidateWeightQuantization(const tflite::Tensor* tensor,
                                int32_t expected_axis,
                                std::size_t expected_channels,
                                std::size_t* scale_count,
                                std::string* error) {
  const tflite::QuantizationParameters* quantization = tensor->quantization();
  if (quantization == nullptr || quantization->scale() == nullptr ||
      quantization->zero_point() == nullptr ||
      quantization->scale()->size() == 0U ||
      quantization->scale()->size() != quantization->zero_point()->size()) {
    *error = "constant weight or bias lacks paired quantization metadata";
    return false;
  }
  const std::size_t count = quantization->scale()->size();
  if (count != 1U && count != expected_channels) {
    *error = "per-axis weight or bias scale count is not bounded by its output channels";
    return false;
  }
  if (count > 1U && quantization->quantized_dimension() != expected_axis) {
    *error = "per-axis weight or bias quantization uses an unexpected dimension";
    return false;
  }
  for (std::size_t index = 0; index < count; ++index) {
    if (quantization->zero_point()->Get(index) != 0) {
      *error = "integer weight and bias zero points must be zero";
      return false;
    }
  }
  *scale_count = count;
  return true;
}

bool ValidateBiasScaleRelation(const tflite::Tensor* input,
                               const tflite::Tensor* weights,
                               const tflite::Tensor* bias,
                               std::size_t weight_scale_count,
                               std::string* error) {
  const tflite::QuantizationParameters* input_quantization =
      input->quantization();
  const tflite::QuantizationParameters* weight_quantization =
      weights->quantization();
  const tflite::QuantizationParameters* bias_quantization =
      bias->quantization();
  if (input_quantization == nullptr || input_quantization->scale() == nullptr ||
      input_quantization->scale()->size() != 1U || weight_quantization == nullptr ||
      weight_quantization->scale() == nullptr ||
      bias_quantization == nullptr || bias_quantization->scale() == nullptr ||
      bias_quantization->scale()->size() != weight_scale_count) {
    *error = "bias scale count does not match input and weight quantization";
    return false;
  }
  const float input_scale = input_quantization->scale()->Get(0);
  for (std::size_t index = 0; index < weight_scale_count; ++index) {
    const float expected = input_scale * weight_quantization->scale()->Get(index);
    if (!Float32Close(expected, bias_quantization->scale()->Get(index))) {
      *error = "bias scales must equal input scale multiplied by weight scales";
      return false;
    }
  }
  return true;
}

bool RequireTensorType(const tflite::Tensor* tensor, tflite::TensorType type,
                       const char* role, std::string* error) {
  if (tensor == nullptr || tensor->type() != type) {
    *error = std::string(role) + " has an unsupported tensor dtype";
    return false;
  }
  return true;
}

bool RequireActivationQuantization(const tflite::Tensor* tensor,
                                   const char* role, std::string* error) {
  if (!RequireTensorType(tensor, tflite::TensorType_INT8, role, error)) {
    return false;
  }
  const tflite::QuantizationParameters* quantization = tensor->quantization();
  if (quantization == nullptr || quantization->scale() == nullptr ||
      quantization->zero_point() == nullptr ||
      quantization->scale()->size() != 1U ||
      quantization->zero_point()->size() != 1U) {
    *error = std::string(role) + " must use per-tensor INT8 quantization";
    return false;
  }
  return true;
}

bool CheckActivation(tflite::ActivationFunctionType activation,
                     std::string* error) {
  if (activation != tflite::ActivationFunctionType_NONE &&
      activation != tflite::ActivationFunctionType_RELU &&
      activation != tflite::ActivationFunctionType_RELU6) {
    *error = "fused activation is outside the bounded NONE/RELU/RELU6 set";
    return false;
  }
  return true;
}

bool SpatialOutput(int32_t input, int32_t kernel, int32_t stride,
                   tflite::Padding padding, int32_t dilation,
                   int32_t* output, std::string* error) {
  if (input <= 0 || kernel <= 0 || stride <= 0 || dilation <= 0 ||
      (padding != tflite::Padding_SAME && padding != tflite::Padding_VALID)) {
    *error = "spatial operator has an invalid positive or padding option";
    return false;
  }
  const int64_t effective_kernel =
      (static_cast<int64_t>(kernel) - 1) * dilation + 1;
  int64_t result = 0;
  if (padding == tflite::Padding_SAME) {
    result = (static_cast<int64_t>(input) + stride - 1) / stride;
  } else if (input >= effective_kernel) {
    result = (static_cast<int64_t>(input) - effective_kernel) / stride + 1;
  }
  if (result <= 0 || result > std::numeric_limits<int32_t>::max()) {
    *error = "spatial operator produces an invalid static output dimension";
    return false;
  }
  *output = static_cast<int32_t>(result);
  return true;
}

bool RequireOptionsNone(const tflite::Operator* op, std::string* error) {
  if (op->builtin_options_type() != tflite::BuiltinOptions_NONE ||
      op->builtin_options() != nullptr) {
    *error = "operator carries an unexpected builtin options table";
    return false;
  }
  return true;
}

bool ReadLittleEndianInt32(const uint8_t* data, std::size_t offset,
                           int32_t* value) {
  uint32_t bits = static_cast<uint32_t>(data[offset]) |
                  (static_cast<uint32_t>(data[offset + 1U]) << 8U) |
                  (static_cast<uint32_t>(data[offset + 2U]) << 16U) |
                  (static_cast<uint32_t>(data[offset + 3U]) << 24U);
  std::memcpy(value, &bits, sizeof(bits));
  return true;
}

bool ValidateConstantWeight(const tflite::Tensor* tensor,
                            const TensorMetadata& metadata,
                            const tflite::Buffer* buffer,
                            const char* role, std::string* error) {
  if (tensor == nullptr || !metadata.is_constant || buffer == nullptr ||
      buffer->data() == nullptr || buffer->data()->size() != metadata.byte_count) {
    *error = std::string(role) + " must be an exact inline constant buffer";
    return false;
  }
  return true;
}

bool ValidateConvolution(const tflite::Operator* op, bool depthwise,
                         const std::vector<const tflite::Tensor*>& tensors,
                         const std::vector<TensorMetadata>& metadata,
                         const std::vector<const tflite::Buffer*>& buffers,
                         std::string* error) {
  if (op->inputs() == nullptr || op->outputs() == nullptr ||
      op->inputs()->size() != 3U || op->outputs()->size() != 1U) {
    *error = "convolution requires exactly input, weights, bias, and output";
    return false;
  }
  for (std::size_t index = 0; index < 3U; ++index) {
    if (!ValidTensorIndex(op->inputs()->Get(index), tensors.size())) {
      *error = "convolution input arity contains a missing tensor";
      return false;
    }
  }
  if (!ValidTensorIndex(op->outputs()->Get(0), tensors.size())) {
    *error = "convolution output references an invalid tensor";
    return false;
  }
  const tflite::Tensor* input = tensors[op->inputs()->Get(0)];
  const tflite::Tensor* weights = tensors[op->inputs()->Get(1)];
  const tflite::Tensor* bias = tensors[op->inputs()->Get(2)];
  const tflite::Tensor* output = tensors[op->outputs()->Get(0)];
  if (!RequireActivationQuantization(input, "convolution input", error) ||
      !RequireTensorType(weights, tflite::TensorType_INT8,
                         "convolution weights", error) ||
      !RequireTensorType(bias, tflite::TensorType_INT32,
                         "convolution bias", error) ||
      !RequireActivationQuantization(output, "convolution output", error)) {
    return false;
  }
  if (!metadata[op->inputs()->Get(1)].is_constant ||
      !metadata[op->inputs()->Get(2)].is_constant ||
      !ValidateConstantWeight(weights, metadata[op->inputs()->Get(1)],
                              buffers[weights->buffer()], "convolution weights",
                              error) ||
      !ValidateConstantWeight(bias, metadata[op->inputs()->Get(2)],
                              buffers[bias->buffer()], "convolution bias", error)) {
    return false;
  }
  if (input->shape() == nullptr || weights->shape() == nullptr ||
      bias->shape() == nullptr || output->shape() == nullptr ||
      input->shape()->size() != 4U || weights->shape()->size() != 4U ||
      output->shape()->size() != 4U) {
    *error = "convolution tensors must be rank-4 static tensors";
    return false;
  }
  tflite::ActivationFunctionType activation =
      tflite::ActivationFunctionType_NONE;
  int32_t stride_h = 0;
  int32_t stride_w = 0;
  int32_t dilation_h = 1;
  int32_t dilation_w = 1;
  tflite::Padding padding = tflite::Padding_SAME;
  if (depthwise) {
    const auto* options = op->builtin_options_as_DepthwiseConv2DOptions();
    if (op->builtin_options_type() !=
            tflite::BuiltinOptions_DepthwiseConv2DOptions ||
        options == nullptr) {
      *error = "DEPTHWISE_CONV_2D options union does not match its operator";
      return false;
    }
    stride_h = options->stride_h();
    stride_w = options->stride_w();
    dilation_h = options->dilation_h_factor();
    dilation_w = options->dilation_w_factor();
    padding = options->padding();
    activation = options->fused_activation_function();
    const int32_t input_channels = input->shape()->Get(3);
    const int32_t depth_multiplier = options->depth_multiplier();
    if (depth_multiplier <= 0 ||
        static_cast<int64_t>(input_channels) * depth_multiplier >
            std::numeric_limits<int32_t>::max() ||
        weights->shape()->Get(0) != 1 ||
        weights->shape()->Get(3) != input_channels * depth_multiplier ||
        output->shape()->Get(3) != input_channels * depth_multiplier) {
      *error = "DEPTHWISE_CONV_2D channels do not match its bounded options";
      return false;
    }
  } else {
    const auto* options = op->builtin_options_as_Conv2DOptions();
    if (op->builtin_options_type() != tflite::BuiltinOptions_Conv2DOptions ||
        options == nullptr) {
      *error = "CONV_2D options union does not match its operator";
      return false;
    }
    stride_h = options->stride_h();
    stride_w = options->stride_w();
    dilation_h = options->dilation_h_factor();
    dilation_w = options->dilation_w_factor();
    padding = options->padding();
    activation = options->fused_activation_function();
    if (weights->shape()->Get(3) != input->shape()->Get(3) ||
        weights->shape()->Get(0) != output->shape()->Get(3)) {
      *error = "CONV_2D channels do not match its bounded tensor shapes";
      return false;
    }
  }
  if (!CheckActivation(activation, error) || stride_h <= 0 || stride_w <= 0 ||
      dilation_h <= 0 || dilation_w <= 0) {
    *error = "convolution strides, dilations, or activation are unsupported";
    return false;
  }
  int32_t computed_h = 0;
  int32_t computed_w = 0;
  if (!SpatialOutput(input->shape()->Get(1), weights->shape()->Get(1), stride_h,
                     padding, dilation_h, &computed_h, error) ||
      !SpatialOutput(input->shape()->Get(2), weights->shape()->Get(2), stride_w,
                     padding, dilation_w, &computed_w, error) ||
      output->shape()->Get(0) != input->shape()->Get(0) ||
      output->shape()->Get(1) != computed_h ||
      output->shape()->Get(2) != computed_w) {
    if (error->empty()) {
      *error = "convolution output shape does not follow its options";
    }
    return false;
  }
  const std::size_t output_channels =
      static_cast<std::size_t>(output->shape()->Get(3));
  std::size_t weight_scale_count = 0U;
  std::size_t bias_scale_count = 0U;
  const int32_t weight_axis = depthwise ? 3 : 0;
  if (!ValidateWeightQuantization(weights, weight_axis, output_channels,
                                  &weight_scale_count, error) ||
      !ValidateWeightQuantization(bias, 0, output_channels, &bias_scale_count,
                                  error) ||
      bias_scale_count != weight_scale_count ||
      !ValidateBiasScaleRelation(input, weights, bias, weight_scale_count,
                                 error) ||
      bias->shape()->size() != 1U ||
      static_cast<std::size_t>(bias->shape()->Get(0)) != output_channels) {
    if (error->empty()) {
      *error = "convolution bias shape or quantization relation is invalid";
    }
    return false;
  }
  return true;
}

bool ValidateAveragePool(const tflite::Operator* op,
                         const std::vector<const tflite::Tensor*>& tensors,
                         std::string* error) {
  if (op->inputs() == nullptr || op->outputs() == nullptr ||
      op->inputs()->size() != 1U || op->outputs()->size() != 1U ||
      !ValidTensorIndex(op->inputs()->Get(0), tensors.size()) ||
      !ValidTensorIndex(op->outputs()->Get(0), tensors.size())) {
    *error = "AVERAGE_POOL_2D requires one valid input and output";
    return false;
  }
  const auto* options = op->builtin_options_as_Pool2DOptions();
  if (op->builtin_options_type() != tflite::BuiltinOptions_Pool2DOptions ||
      options == nullptr) {
    *error = "AVERAGE_POOL_2D options union does not match its operator";
    return false;
  }
  const tflite::Tensor* input = tensors[op->inputs()->Get(0)];
  const tflite::Tensor* output = tensors[op->outputs()->Get(0)];
  if (!RequireActivationQuantization(input, "average-pool input", error) ||
      !RequireActivationQuantization(output, "average-pool output", error) ||
      input->shape() == nullptr || output->shape() == nullptr ||
      input->shape()->size() != 4U || output->shape()->size() != 4U ||
      input->shape()->Get(0) != output->shape()->Get(0) ||
      input->shape()->Get(3) != output->shape()->Get(3)) {
    *error = "AVERAGE_POOL_2D tensors must preserve batch and channels";
    return false;
  }
  const auto* input_quantization = input->quantization();
  const auto* output_quantization = output->quantization();
  if (!Float32Close(input_quantization->scale()->Get(0),
                    output_quantization->scale()->Get(0)) ||
      input_quantization->zero_point()->Get(0) !=
          output_quantization->zero_point()->Get(0)) {
    *error = "AVERAGE_POOL_2D must preserve activation quantization";
    return false;
  }
  int32_t output_h = 0;
  int32_t output_w = 0;
  if (!CheckActivation(options->fused_activation_function(), error) ||
      !SpatialOutput(input->shape()->Get(1), options->filter_height(),
                     options->stride_h(), options->padding(), 1, &output_h,
                     error) ||
      !SpatialOutput(input->shape()->Get(2), options->filter_width(),
                     options->stride_w(), options->padding(), 1, &output_w,
                     error) ||
      output->shape()->Get(1) != output_h || output->shape()->Get(2) != output_w ||
      options->filter_height() <= 0 || options->filter_width() <= 0 ||
      options->stride_h() <= 0 || options->stride_w() <= 0) {
    if (error->empty()) {
      *error = "AVERAGE_POOL_2D options or output shape is invalid";
    }
    return false;
  }
  return true;
}

bool ValidateRelu(const tflite::Operator* op,
                  const std::vector<const tflite::Tensor*>& tensors,
                  std::string* error) {
  if (op->inputs() == nullptr || op->outputs() == nullptr ||
      op->inputs()->size() != 1U || op->outputs()->size() != 1U ||
      !ValidTensorIndex(op->inputs()->Get(0), tensors.size()) ||
      !ValidTensorIndex(op->outputs()->Get(0), tensors.size()) ||
      !RequireOptionsNone(op, error)) {
    if (error->empty()) {
      *error = "activation requires one valid input and output";
    }
    return false;
  }
  const tflite::Tensor* input = tensors[op->inputs()->Get(0)];
  const tflite::Tensor* output = tensors[op->outputs()->Get(0)];
  if (!RequireActivationQuantization(input, "activation input", error) ||
      !RequireActivationQuantization(output, "activation output", error) ||
      input->shape() == nullptr || output->shape() == nullptr ||
      input->shape()->size() != output->shape()->size()) {
    return false;
  }
  for (std::size_t index = 0; index < input->shape()->size(); ++index) {
    if (input->shape()->Get(index) != output->shape()->Get(index)) {
      *error = "activation must preserve tensor shape";
      return false;
    }
  }
  return true;
}

bool ValidateReshape(const tflite::Operator* op,
                     const std::vector<const tflite::Tensor*>& tensors,
                     const std::vector<TensorMetadata>& metadata,
                     const std::vector<const tflite::Buffer*>& buffers,
                     std::string* error) {
  if (op->inputs() == nullptr || op->outputs() == nullptr ||
      op->inputs()->size() != 2U || op->outputs()->size() != 1U ||
      !ValidTensorIndex(op->inputs()->Get(0), tensors.size()) ||
      !ValidTensorIndex(op->inputs()->Get(1), tensors.size()) ||
      !ValidTensorIndex(op->outputs()->Get(0), tensors.size())) {
    *error = "RESHAPE requires two valid inputs and one output";
    return false;
  }
  const tflite::Tensor* input = tensors[op->inputs()->Get(0)];
  const tflite::Tensor* shape = tensors[op->inputs()->Get(1)];
  const tflite::Tensor* output = tensors[op->outputs()->Get(0)];
  if (!RequireActivationQuantization(input, "reshape input", error) ||
      !RequireTensorType(shape, tflite::TensorType_INT32, "reshape shape", error) ||
      !RequireActivationQuantization(output, "reshape output", error) ||
      !metadata[op->inputs()->Get(1)].is_constant ||
      !ValidateConstantWeight(shape, metadata[op->inputs()->Get(1)],
                              buffers[shape->buffer()], "reshape shape", error)) {
    return false;
  }
  if (!Float32Close(input->quantization()->scale()->Get(0),
                    output->quantization()->scale()->Get(0)) ||
      input->quantization()->zero_point()->Get(0) !=
          output->quantization()->zero_point()->Get(0)) {
    *error = "RESHAPE must preserve activation quantization";
    return false;
  }
  if (input->shape() == nullptr || output->shape() == nullptr ||
      shape->shape() == nullptr || shape->shape()->size() != 1U ||
      static_cast<std::size_t>(shape->shape()->Get(0)) != output->shape()->size()) {
    *error = "RESHAPE shape tensor rank does not match its output";
    return false;
  }
  const auto* raw = buffers[shape->buffer()]->data();
  if (raw == nullptr || raw->size() != metadata[op->inputs()->Get(1)].byte_count ||
      raw->size() % sizeof(int32_t) != 0U) {
    *error = "RESHAPE shape tensor buffer is not an exact INT32 array";
    return false;
  }
  for (std::size_t index = 0; index < output->shape()->size(); ++index) {
    int32_t shape_value = 0;
    ReadLittleEndianInt32(raw->Data(), index * sizeof(int32_t), &shape_value);
    if (shape_value != output->shape()->Get(index) || shape_value <= 0) {
      *error = "RESHAPE shape tensor does not exactly describe its output";
      return false;
    }
  }
  std::size_t input_elements = 0U;
  std::size_t output_elements = 0U;
  if (!StaticShape(input, &input_elements, error) ||
      !StaticShape(output, &output_elements, error) ||
      input_elements != output_elements) {
    *error = "RESHAPE must preserve the number of elements";
    return false;
  }
  if (op->builtin_options_type() == tflite::BuiltinOptions_NONE) {
    if (op->builtin_options() != nullptr) {
      *error = "RESHAPE NONE options unexpectedly carry a table";
      return false;
    }
  } else if (op->builtin_options_type() == tflite::BuiltinOptions_ReshapeOptions) {
    const auto* options = op->builtin_options_as_ReshapeOptions();
    if (options == nullptr) {
      *error = "RESHAPE options union is missing its table";
      return false;
    }
    const auto* new_shape = options->new_shape();
    if (new_shape != nullptr && new_shape->size() != 0U) {
      if (new_shape->size() != output->shape()->size()) {
        *error = "RESHAPE options new_shape disagrees with its output";
        return false;
      }
      for (std::size_t index = 0; index < new_shape->size(); ++index) {
        if (new_shape->Get(index) != output->shape()->Get(index)) {
          *error = "RESHAPE options new_shape disagrees with its output";
          return false;
        }
      }
    }
  } else {
    *error = "RESHAPE options union is outside the bounded NONE/ReshapeOptions set";
    return false;
  }
  return true;
}

bool ValidateFullyConnected(const tflite::Operator* op,
                            const std::vector<const tflite::Tensor*>& tensors,
                            const std::vector<TensorMetadata>& metadata,
                            const std::vector<const tflite::Buffer*>& buffers,
                            std::string* error) {
  if (op->inputs() == nullptr || op->outputs() == nullptr ||
      op->inputs()->size() != 3U || op->outputs()->size() != 1U ||
      !ValidTensorIndex(op->inputs()->Get(0), tensors.size()) ||
      !ValidTensorIndex(op->inputs()->Get(1), tensors.size()) ||
      !ValidTensorIndex(op->outputs()->Get(0), tensors.size())) {
    *error = "FULLY_CONNECTED requires input, weights, optional bias, and output";
    return false;
  }
  const int32_t bias_index = op->inputs()->Get(2);
  if (bias_index < -1 ||
      (bias_index >= 0 && !ValidTensorIndex(bias_index, tensors.size()))) {
    *error = "FULLY_CONNECTED bias sentinel is outside the supported -1/index set";
    return false;
  }
  const auto* options = op->builtin_options_as_FullyConnectedOptions();
  if (op->builtin_options_type() != tflite::BuiltinOptions_FullyConnectedOptions ||
      options == nullptr) {
    *error = "FULLY_CONNECTED options union does not match its operator";
    return false;
  }
  const tflite::Tensor* input = tensors[op->inputs()->Get(0)];
  const tflite::Tensor* weights = tensors[op->inputs()->Get(1)];
  const tflite::Tensor* output = tensors[op->outputs()->Get(0)];
  if (!RequireActivationQuantization(input, "fully-connected input", error) ||
      !RequireTensorType(weights, tflite::TensorType_INT8,
                         "fully-connected weights", error) ||
      !RequireActivationQuantization(output, "fully-connected output", error) ||
      !metadata[op->inputs()->Get(1)].is_constant ||
      !ValidateConstantWeight(weights, metadata[op->inputs()->Get(1)],
                              buffers[weights->buffer()],
                              "fully-connected weights", error)) {
    return false;
  }
  if (bias_index >= 0) {
    const tflite::Tensor* bias = tensors[bias_index];
    if (!RequireTensorType(bias, tflite::TensorType_INT32,
                           "fully-connected bias", error) ||
        !metadata[bias_index].is_constant ||
        !ValidateConstantWeight(bias, metadata[bias_index],
                                buffers[bias->buffer()],
                                "fully-connected bias", error)) {
      return false;
    }
  }
  // The schema option defaults to FLOAT32 and describes a quantized-bias
  // mode; it does not change the bias tensor's dtype.  This profile therefore
  // keeps the default option while requiring the actual bias tensor to be
  // INT32 above.
  if (options->weights_format() !=
          tflite::FullyConnectedOptionsWeightsFormat_DEFAULT ||
      options->keep_num_dims() || options->asymmetric_quantize_inputs() ||
      options->quantized_bias_type() != tflite::TensorType_FLOAT32 ||
      (options->quant_spec() != nullptr && options->quant_spec()->size() != 0U) ||
      !CheckActivation(options->fused_activation_function(), error)) {
    *error = "FULLY_CONNECTED options contain an unsupported quantized mode";
    return false;
  }
  // The pinned reference kernel derives batches from every output dimension
  // except the last and uses the last dimension as output_depth.  This host
  // profile deliberately accepts only its narrower, rank-2 batch-one case;
  // it does not claim support for the reference kernel's generic batching.
  if (input->shape() == nullptr || weights->shape() == nullptr ||
      output->shape() == nullptr || input->shape()->size() != 2U ||
      weights->shape()->size() != 2U || output->shape()->size() != 2U ||
      input->shape()->Get(0) != 1 || output->shape()->Get(0) != 1) {
    *error =
        "FULLY_CONNECTED host profile requires rank-2 input/output/weights "
        "with batch one";
    return false;
  }
  const int32_t input_depth = input->shape()->Get(1);
  const int32_t output_units = output->shape()->Get(1);
  if (input_depth <= 0 || output_units <= 0 ||
      weights->shape()->Get(0) != output_units ||
      weights->shape()->Get(1) != input_depth) {
    *error = "FULLY_CONNECTED weights do not match input and output shapes";
    return false;
  }
  const std::size_t output_channels = static_cast<std::size_t>(output_units);
  std::size_t weight_scale_count = 0U;
  if (!ValidateWeightQuantization(weights, 0, output_channels,
                                  &weight_scale_count, error)) {
    return false;
  }
  if (bias_index >= 0) {
    const tflite::Tensor* bias = tensors[bias_index];
    if (bias->shape() == nullptr || bias->shape()->size() != 1U ||
        bias->shape()->Get(0) != output_units) {
      *error = "FULLY_CONNECTED bias shape must equal output units";
      return false;
    }
    std::size_t bias_scale_count = 0U;
    if (!ValidateWeightQuantization(bias, 0, output_channels,
                                    &bias_scale_count, error) ||
        bias_scale_count != weight_scale_count ||
        !ValidateBiasScaleRelation(input, weights, bias, weight_scale_count,
                                   error)) {
      return false;
    }
  }
  return true;
}

bool PreflightModel(const std::vector<uint8_t>& model_bytes, ModelInfo* info,
                    std::string* error) {
  if (model_bytes.size() < sizeof(flatbuffers::uoffset_t) +
                              flatbuffers::kFileIdentifierLength) {
    *error = "model is shorter than the safe FlatBuffer identifier prefix";
    return false;
  }
  if (!tflite::ModelBufferHasIdentifier(model_bytes.data())) {
    *error = std::string("model identifier is not ") + kIdentifier;
    return false;
  }
  flatbuffers::Verifier verifier(model_bytes.data(), model_bytes.size());
  if (!tflite::VerifyModelBuffer(verifier)) {
    *error = "model FlatBuffer verification failed";
    return false;
  }
  const tflite::Model* model = tflite::GetModel(model_bytes.data());
  if (model == nullptr || model->version() != TFLITE_SCHEMA_VERSION) {
    *error = "model schema version is not the pinned TFLM schema version 3";
    return false;
  }
  const auto* operator_codes = model->operator_codes();
  const auto* subgraphs = model->subgraphs();
  const auto* buffers = model->buffers();
  if (operator_codes == nullptr || operator_codes->size() == 0U ||
      subgraphs == nullptr || subgraphs->size() != 1U || buffers == nullptr ||
      buffers->size() == 0U) {
    *error = "model must contain one subgraph and non-empty operator/buffer tables";
    return false;
  }
  if ((model->external_buffer_groups() != nullptr &&
       model->external_buffer_groups()->size() != 0U) ||
      (model->external_buffers() != nullptr &&
       model->external_buffers()->size() != 0U)) {
    *error = "external model buffers are not accepted by this host runner";
    return false;
  }
  for (std::size_t index = 0; index < buffers->size(); ++index) {
    const tflite::Buffer* buffer = buffers->Get(index);
    if (buffer == nullptr || buffer->offset() != 0U || buffer->size() != 0U) {
      *error = "model contains an external or otherwise unknown buffer range";
      return false;
    }
  }

  const tflite::SubGraph* subgraph = subgraphs->Get(0);
  if (subgraph == nullptr || subgraph->tensors() == nullptr ||
      subgraph->inputs() == nullptr || subgraph->outputs() == nullptr ||
      subgraph->operators() == nullptr || subgraph->operators()->size() == 0U ||
      subgraph->inputs()->size() != 1U || subgraph->outputs()->size() != 1U) {
    *error = "model subgraph does not have the required bounded graph vectors";
    return false;
  }
  const auto* tensors = subgraph->tensors();
  const auto* operators = subgraph->operators();
  std::vector<const tflite::Tensor*> tensor_refs;
  tensor_refs.reserve(tensors->size());
  std::vector<const tflite::Buffer*> buffer_refs;
  buffer_refs.reserve(buffers->size());
  for (std::size_t tensor_index = 0; tensor_index < tensors->size();
       ++tensor_index) {
    const tflite::Tensor* tensor = tensors->Get(tensor_index);
    if (tensor == nullptr || tensor->buffer() >= buffers->size() ||
        tensor->external_buffer() != 0U) {
      *error = "tensor references an invalid or external buffer";
      return false;
    }
    tensor_refs.push_back(tensor);
  }
  for (std::size_t buffer_index = 0; buffer_index < buffers->size();
       ++buffer_index) {
    const tflite::Buffer* buffer = buffers->Get(buffer_index);
    if (buffer == nullptr) {
      *error = "model contains a null buffer table entry";
      return false;
    }
    buffer_refs.push_back(buffer);
  }

  const int32_t input_tensor_index = subgraph->inputs()->Get(0);
  const int32_t output_tensor_index = subgraph->outputs()->Get(0);
  if (!ValidTensorIndex(input_tensor_index, tensors->size()) ||
      !ValidTensorIndex(output_tensor_index, tensors->size())) {
    *error = "graph I/O index is outside the verified tensor table";
    return false;
  }
  const tflite::Tensor* input_tensor = tensors->Get(input_tensor_index);
  const tflite::Tensor* output_tensor = tensors->Get(output_tensor_index);
  if (input_tensor == nullptr || output_tensor == nullptr ||
      !IsExpectedShape(input_tensor->shape(), {1, 298, 40, 1}) ||
      !IsExpectedShape(output_tensor->shape(), {1, 192})) {
    *error = "model I/O must have shapes [1,298,40,1] and [1,192]";
    return false;
  }

  std::vector<bool> graph_inputs(tensors->size(), false);
  graph_inputs[static_cast<std::size_t>(input_tensor_index)] = true;
  std::vector<std::size_t> producer_counts(tensors->size(), 0U);
  std::vector<std::string> operation_names;
  operation_names.reserve(operators->size());
  for (std::size_t operator_index = 0; operator_index < operators->size();
       ++operator_index) {
    const tflite::Operator* op = operators->Get(operator_index);
    if (op == nullptr || op->opcode_index() >= operator_codes->size() ||
        op->inputs() == nullptr || op->outputs() == nullptr) {
      *error = "operator has an invalid opcode or tensor vector";
      return false;
    }
    const tflite::OperatorCode* opcode = operator_codes->Get(op->opcode_index());
    if (opcode == nullptr) {
      *error = "operator references a null opcode table entry";
      return false;
    }
    const tflite::BuiltinOperator builtin =
        tflite::micro::GetBuiltinCode(opcode);
    const char* name = BuiltinName(builtin);
    if (name == nullptr || !SupportedVersion(builtin, opcode->version())) {
      *error = "operator code or version is outside the bounded resolver allowlist";
      return false;
    }
    if (op->builtin_options_2_type() != tflite::BuiltinOptions2_NONE ||
        op->builtin_options_2() != nullptr) {
      *error = "secondary builtin options are not supported by the bounded runner";
      return false;
    }
    if ((op->custom_options() != nullptr && op->custom_options()->size() != 0U) ||
        (op->mutating_variable_inputs() != nullptr &&
         op->mutating_variable_inputs()->size() != 0U) ||
        (op->intermediates() != nullptr && op->intermediates()->size() != 0U) ||
        op->large_custom_options_offset() != 0U ||
        op->large_custom_options_size() != 0U ||
        op->debug_metadata_index() != -1) {
      *error = "operator carries unsupported custom, variable, intermediate, or debug metadata";
      return false;
    }
    operation_names.emplace_back(name);
    for (std::size_t input_index = 0; input_index < op->inputs()->size();
         ++input_index) {
      const int32_t tensor_index = op->inputs()->Get(input_index);
      if (tensor_index != -1 && !ValidTensorIndex(tensor_index, tensors->size())) {
        *error = "operator input index is outside the verified tensor table";
        return false;
      }
    }
    for (std::size_t output_index = 0; output_index < op->outputs()->size();
         ++output_index) {
      const int32_t tensor_index = op->outputs()->Get(output_index);
      if (!ValidTensorIndex(tensor_index, tensors->size())) {
        *error = "operator output index is outside the verified tensor table";
        return false;
      }
      const std::size_t producer_index = static_cast<std::size_t>(tensor_index);
      if (++producer_counts[producer_index] != 1U) {
        *error = "a tensor has more than one bounded graph producer";
        return false;
      }
    }
  }
  if (producer_counts[static_cast<std::size_t>(input_tensor_index)] != 0U ||
      producer_counts[static_cast<std::size_t>(output_tensor_index)] != 1U) {
    *error = "bounded graph I/O must have one input source and one output producer";
    return false;
  }

  std::vector<TensorMetadata> tensor_metadata(tensors->size());
  for (std::size_t tensor_index = 0; tensor_index < tensors->size();
       ++tensor_index) {
    const tflite::Tensor* tensor = tensor_refs[tensor_index];
    TensorMetadata metadata;
    metadata.is_constant =
        !graph_inputs[tensor_index] && producer_counts[tensor_index] == 0U;
    if (!IsSupportedTensorType(tensor->type())) {
      *error = "every graph tensor must be INT8 or INT32 for the bounded profile";
      return false;
    }
    if (tensor->is_variable()) {
      *error = "variable tensors are not supported by the static host profile";
      return false;
    }
    if (tensor->sparsity() != nullptr) {
      *error = "sparse tensor metadata is not supported by the static host profile";
      return false;
    }
    if (tensor->variant_tensors() != nullptr &&
        tensor->variant_tensors()->size() != 0U) {
      *error = "variant tensor metadata is not supported by the static host profile";
      return false;
    }
    if (!tensor->has_rank()) {
      *error = "tensors without a known rank are not accepted by the bounded runner";
      return false;
    }
    if (tensor->shape() == nullptr) {
      *error = "tensor shape vector is missing";
      return false;
    }
    if (tensor->shape_signature() != nullptr) {
      if (tensor->shape_signature()->size() != tensor->shape()->size()) {
        *error = "tensor shape signature is not the same known rank as its shape";
        return false;
      }
      for (std::size_t dimension = 0;
           dimension < tensor->shape_signature()->size(); ++dimension) {
        if (tensor->shape_signature()->Get(dimension) !=
            tensor->shape()->Get(dimension)) {
          *error = "dynamic or ambiguous tensor shape signature is not accepted";
          return false;
        }
      }
    }
    if (!StaticShape(tensor, &metadata.element_count, error)) {
      return false;
    }
    if (tensor->type() == tflite::TensorType_INT32 && !metadata.is_constant) {
      *error = "only constant INT32 shape or bias metadata is supported";
      return false;
    }
    const std::size_t element_bytes = BytesPerElement(tensor->type());
    if (metadata.element_count >
        std::numeric_limits<std::size_t>::max() / element_bytes) {
      *error = "tensor byte count overflows host size_t";
      return false;
    }
    metadata.byte_count = metadata.element_count * element_bytes;
    if (!ValidateTensorQuantization(tensor, metadata, error)) {
      return false;
    }
    const tflite::Buffer* buffer = buffer_refs[tensor->buffer()];
    const std::size_t inline_bytes =
        buffer->data() == nullptr ? 0U : buffer->data()->size();
    if (metadata.is_constant) {
      if (!ValidateConstantWeight(tensor, metadata, buffer, "tensor", error)) {
        return false;
      }
    } else if (inline_bytes != 0U) {
      *error = "non-constant tensor has an unexpected inline buffer";
      return false;
    }
    tensor_metadata[tensor_index] = metadata;
  }

  if (input_tensor->type() != tflite::TensorType_INT8 ||
      output_tensor->type() != tflite::TensorType_INT8) {
    *error = "model I/O must be INT8 with shapes [1,298,40,1] and [1,192]";
    return false;
  }
  QuantizationInfo input_quantization;
  QuantizationInfo output_quantization;
  if (!ReadQuantization(input_tensor, &input_quantization, error) ||
      !ReadQuantization(output_tensor, &output_quantization, error)) {
    return false;
  }

  for (std::size_t operator_index = 0; operator_index < operators->size();
       ++operator_index) {
    const tflite::Operator* op = operators->Get(operator_index);
    const tflite::OperatorCode* opcode = operator_codes->Get(op->opcode_index());
    const tflite::BuiltinOperator builtin =
        tflite::micro::GetBuiltinCode(opcode);
    bool valid = false;
    switch (builtin) {
      case tflite::BuiltinOperator_CONV_2D:
        valid = ValidateConvolution(op, false, tensor_refs, tensor_metadata,
                                    buffer_refs, error);
        break;
      case tflite::BuiltinOperator_DEPTHWISE_CONV_2D:
        valid = ValidateConvolution(op, true, tensor_refs, tensor_metadata,
                                    buffer_refs, error);
        break;
      case tflite::BuiltinOperator_AVERAGE_POOL_2D:
        valid = ValidateAveragePool(op, tensor_refs, error);
        break;
      case tflite::BuiltinOperator_RELU:
      case tflite::BuiltinOperator_RELU6:
        valid = ValidateRelu(op, tensor_refs, error);
        break;
      case tflite::BuiltinOperator_RESHAPE:
        valid = ValidateReshape(op, tensor_refs, tensor_metadata, buffer_refs,
                                error);
        break;
      case tflite::BuiltinOperator_FULLY_CONNECTED:
        valid = ValidateFullyConnected(op, tensor_refs, tensor_metadata,
                                       buffer_refs, error);
        break;
      default:
        *error = "operator is outside the bounded semantic allowlist";
        valid = false;
        break;
    }
    if (!valid) {
      return false;
    }
  }

  info->model = model;
  info->model_bytes = model_bytes.size();
  info->input_tensor_index = input_tensor_index;
  info->output_tensor_index = output_tensor_index;
  info->input_quantization = input_quantization;
  info->output_quantization = output_quantization;
  info->operations = std::move(operation_names);
  info->schema_version = model->version();
  return true;
}

bool CheckRuntimeTensor(const TfLiteTensor* tensor,
                        const std::initializer_list<int>& expected_shape,
                        std::size_t expected_bytes, std::string* error) {
  if (tensor == nullptr || tensor->type != kTfLiteInt8 || tensor->data.int8 == nullptr ||
      tensor->bytes != expected_bytes || tensor->dims == nullptr ||
      tensor->dims->size != static_cast<int>(expected_shape.size())) {
    *error = "TFLM returned an unexpected INT8 tensor layout";
    return false;
  }
  std::size_t index = 0;
  for (const int expected_dimension : expected_shape) {
    if (tensor->dims->data[index] != expected_dimension) {
      *error = "TFLM returned an unexpected tensor shape";
      return false;
    }
    ++index;
  }
  if (!std::isfinite(tensor->params.scale) || tensor->params.scale <= 0.0F ||
      tensor->params.zero_point < -128 || tensor->params.zero_point > 127) {
    *error = "TFLM returned a non-finite or non-positive quantization scale";
    return false;
  }
  return true;
}

bool WriteAtomicOutput(const std::string& path, const std::vector<int8_t>& output,
                       std::string* error) {
  const std::string temporary_path = path + ".edgeai-tflm-tmp";
  std::remove(temporary_path.c_str());
  {
    std::ofstream stream(temporary_path, std::ios::binary | std::ios::trunc);
    if (!stream) {
      *error = "cannot create the temporary output file";
      return false;
    }
    stream.write(reinterpret_cast<const char*>(output.data()),
                 static_cast<std::streamsize>(output.size()));
    if (!stream) {
      stream.close();
      std::remove(temporary_path.c_str());
      *error = "cannot write the complete output file";
      return false;
    }
  }
  if (std::rename(temporary_path.c_str(), path.c_str()) != 0) {
    std::remove(temporary_path.c_str());
    *error = "cannot commit the output file";
    return false;
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 5) {
      PrintFailure("INVALID_ARGUMENT", "expected model, input, output, and arena_bytes", std::nullopt);
      return 2;
    }

    std::size_t arena_bytes = 0;
    std::string error;
    if (!ParseArenaSize(argv[4], &arena_bytes, &error)) {
      PrintFailure("REJECTED", error, std::nullopt);
      return 2;
    }

    std::vector<uint8_t> model_bytes;
    if (!ReadFile(argv[1], kModelMaxBytes, &model_bytes, &error)) {
      PrintFailure("REJECTED", "model: " + error, arena_bytes);
      return 3;
    }
    ModelInfo model_info;
    if (!PreflightModel(model_bytes, &model_info, &error)) {
      PrintFailure("REJECTED", error, arena_bytes);
      return 3;
    }

    std::vector<uint8_t> input_bytes;
    if (!ReadFile(argv[2], kInputBytes, &input_bytes, &error) ||
        input_bytes.size() != kInputBytes) {
      if (error.empty()) {
        error = "input must contain exactly 11920 bytes";
      }
      PrintFailure("REJECTED", "input: " + error, arena_bytes);
      return 3;
    }

    const std::size_t arena_words =
        (arena_bytes + sizeof(ArenaWord) - 1U) / sizeof(ArenaWord);
    std::vector<ArenaWord> arena_storage(arena_words);
    auto* arena = reinterpret_cast<uint8_t*>(arena_storage.data());

    tflite::MicroMutableOpResolver<7> resolver;
    if (resolver.AddConv2D() != kTfLiteOk ||
        resolver.AddDepthwiseConv2D() != kTfLiteOk ||
        resolver.AddAveragePool2D() != kTfLiteOk || resolver.AddRelu() != kTfLiteOk ||
        resolver.AddRelu6() != kTfLiteOk || resolver.AddReshape() != kTfLiteOk ||
        resolver.AddFullyConnected() != kTfLiteOk) {
      PrintFailure("REJECTED", "failed to initialize the bounded TFLM resolver", arena_bytes);
      return 4;
    }

    tflite::MicroInterpreter interpreter(model_info.model, resolver, arena,
                                         arena_bytes);
    if (interpreter.AllocateTensors() != kTfLiteOk) {
      PrintFailure("ALLOCATE_TENSORS_FAILED",
                   "TFLM AllocateTensors failed within the supplied arena",
                   arena_bytes);
      return 5;
    }

    TfLiteTensor* input_tensor = interpreter.input(0);
    TfLiteTensor* output_tensor = interpreter.output(0);
    if (!CheckRuntimeTensor(input_tensor, {1, 298, 40, 1}, kInputBytes, &error) ||
        !CheckRuntimeTensor(output_tensor, {1, 192}, kOutputBytes, &error)) {
      PrintFailure("REJECTED", error, arena_bytes);
      return 6;
    }
    std::memcpy(input_tensor->data.int8, input_bytes.data(), kInputBytes);
    if (interpreter.Invoke() != kTfLiteOk) {
      PrintFailure("INVOKE_FAILED", "TFLM Invoke failed", arena_bytes);
      return 7;
    }
    std::vector<int8_t> output_bytes(kOutputBytes);
    std::memcpy(output_bytes.data(), output_tensor->data.int8, kOutputBytes);
    const std::size_t arena_used_bytes = interpreter.arena_used_bytes();
    if (arena_used_bytes > arena_bytes) {
      PrintFailure("REJECTED", "TFLM reported arena usage beyond its capacity", arena_bytes);
      return 8;
    }
    if (!WriteAtomicOutput(argv[3], output_bytes, &error)) {
      PrintFailure("OUTPUT_FAILED", error, arena_bytes);
      return 9;
    }
    PrintSuccess(model_info, arena_bytes, arena_used_bytes);
    return 0;
  } catch (const std::exception& exception) {
    PrintFailure("EXCEPTION", exception.what(), std::nullopt);
    return 10;
  } catch (...) {
    PrintFailure("EXCEPTION", "unknown host exception", std::nullopt);
    return 10;
  }
}
