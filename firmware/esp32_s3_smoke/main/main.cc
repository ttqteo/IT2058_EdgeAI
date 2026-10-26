#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cstdarg>
#include <inttypes.h>
#include <new>

#include "sdkconfig.h"
#include "esp_chip_info.h"
#include "esp_attr.h"
#include "esp_heap_caps.h"
#include "esp_idf_version.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "driver/usb_serial_jtag.h"
#include "driver/usb_serial_jtag_vfs.h"
#include "freertos/FreeRTOS.h"
#include "nvs.h"
#include "nvs_flash.h"

#include "pcm_frontend.h"
#include "template_store.h"

#include "flatbuffers/flatbuffers.h"
#include "tensorflow/lite/micro/flatbuffer_utils.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

extern "C" const uint8_t _binary_encoder_int8_tflite_start[]
    asm("_binary_encoder_int8_tflite_start");
extern "C" const uint8_t _binary_encoder_int8_tflite_end[]
    asm("_binary_encoder_int8_tflite_end");

extern "C" const uint8_t _binary_real_train_000_input_int8_start[]
    asm("_binary_real_train_000_input_int8_start");
extern "C" const uint8_t _binary_real_train_000_input_int8_end[]
    asm("_binary_real_train_000_input_int8_end");
extern "C" const uint8_t _binary_real_train_000_expected_int8_start[]
    asm("_binary_real_train_000_expected_int8_start");
extern "C" const uint8_t _binary_real_train_000_expected_int8_end[]
    asm("_binary_real_train_000_expected_int8_end");
extern "C" const uint8_t _binary_real_train_001_input_int8_start[]
    asm("_binary_real_train_001_input_int8_start");
extern "C" const uint8_t _binary_real_train_001_input_int8_end[]
    asm("_binary_real_train_001_input_int8_end");
extern "C" const uint8_t _binary_real_train_001_expected_int8_start[]
    asm("_binary_real_train_001_expected_int8_start");
extern "C" const uint8_t _binary_real_train_001_expected_int8_end[]
    asm("_binary_real_train_001_expected_int8_end");
extern "C" const uint8_t _binary_real_train_002_input_int8_start[]
    asm("_binary_real_train_002_input_int8_start");
extern "C" const uint8_t _binary_real_train_002_input_int8_end[]
    asm("_binary_real_train_002_input_int8_end");
extern "C" const uint8_t _binary_real_train_002_expected_int8_start[]
    asm("_binary_real_train_002_expected_int8_start");
extern "C" const uint8_t _binary_real_train_002_expected_int8_end[]
    asm("_binary_real_train_002_expected_int8_end");
extern "C" const uint8_t _binary_real_train_003_input_int8_start[]
    asm("_binary_real_train_003_input_int8_start");
extern "C" const uint8_t _binary_real_train_003_input_int8_end[]
    asm("_binary_real_train_003_input_int8_end");
extern "C" const uint8_t _binary_real_train_003_expected_int8_start[]
    asm("_binary_real_train_003_expected_int8_start");
extern "C" const uint8_t _binary_real_train_003_expected_int8_end[]
    asm("_binary_real_train_003_expected_int8_end");
extern "C" const uint8_t _binary_real_train_004_input_int8_start[]
    asm("_binary_real_train_004_input_int8_start");
extern "C" const uint8_t _binary_real_train_004_input_int8_end[]
    asm("_binary_real_train_004_input_int8_end");
extern "C" const uint8_t _binary_real_train_004_expected_int8_start[]
    asm("_binary_real_train_004_expected_int8_start");
extern "C" const uint8_t _binary_real_train_004_expected_int8_end[]
    asm("_binary_real_train_004_expected_int8_end");

namespace {

constexpr char kTag[] = "encoder_smoke";
constexpr char kScope[] = "BOARD_ENCODER_SMOKE_ONLY";
#if defined(CONFIG_NN_ANSI_C)
constexpr char kRuntimeMode[] = "ESP_NN_ANSI_C_REFERENCE";
#elif defined(CONFIG_NN_OPTIMIZED)
constexpr char kRuntimeMode[] = "ESP_NN_OPTIMIZED";
#else
constexpr char kRuntimeMode[] = "ESP_NN_UNSELECTED";
#endif
constexpr char kModelSha256[] =
    "aa42a3140f375f3a2c93648893132b880dd3e9f4be4ca8f82cde09fc6c440727";
constexpr std::size_t kModelBytes = 154000U;
constexpr std::size_t kInputBytes = 1U * 298U * 40U * 1U;
constexpr std::size_t kOutputBytes = 1U * 192U;
constexpr std::size_t kArenaBytes = 131072U;
constexpr int kFixtureCount = 5;
constexpr int kMaxAbsDiffLsb = 1;
constexpr float kInputScale = 0.12695927917957306F;
constexpr int kInputZeroPoint = -12;
constexpr float kOutputScale = 0.01833495683968067F;
constexpr int kOutputZeroPoint = 6;
constexpr char kInputScaleText[] = "0.12695927917957306";
constexpr char kOutputScaleText[] = "0.01833495683968067";

constexpr char kProtocolMagic[] = "EDGEAI01";
constexpr std::size_t kProtocolMagicBytes = 8U;
constexpr std::size_t kRequestHeaderBytes = 20U;
constexpr uint8_t kOperationVersion = 1U;
constexpr uint8_t kOperationEncode = 0x01U;
constexpr uint8_t kOperationProvision = 0x10U;
constexpr uint8_t kOperationStatus = 0x11U;
constexpr uint8_t kOperationRecognize = 0x12U;
constexpr uint8_t kOperationAddAn = 0x13U;
constexpr uint8_t kOperationRecognizePcm = 0x20U;
constexpr uint8_t kOperationAppendPcm = 0x21U;
constexpr std::size_t kOperationHeaderBytes = 2U;
constexpr std::size_t kAppendPcmLabelBytes = edgeai_template::kLabelBytes;
constexpr std::size_t kEncodePayloadBytes =
    kOperationHeaderBytes + kInputBytes;
constexpr std::size_t kRecognizePayloadBytes =
    kOperationHeaderBytes + kInputBytes;
constexpr std::size_t kAddAnPayloadBytes =
    kOperationHeaderBytes + kInputBytes;
constexpr std::size_t kProvisionPayloadBytes =
    kOperationHeaderBytes + (2U * kInputBytes);
constexpr std::size_t kStatusPayloadBytes = kOperationHeaderBytes;
constexpr std::size_t kRecognizePcmPayloadBytes =
    kOperationHeaderBytes + edgeai_pcm_frontend::kPcmBytes;
constexpr std::size_t kAppendPcmPayloadBytes =
    kOperationHeaderBytes + kAppendPcmLabelBytes + edgeai_pcm_frontend::kPcmBytes;
constexpr std::size_t kLegacyMaxRequestPayloadBytes = kProvisionPayloadBytes;
constexpr std::size_t kMaxRequestPayloadBytes = kAppendPcmPayloadBytes;
constexpr int64_t kFrameTimeoutUs = 10LL * 1000LL * 1000LL;
constexpr uint32_t kUsbRxBufferBytes = 16384U;
constexpr uint32_t kUsbTxBufferBytes = 8192U;
constexpr std::size_t kResponseCapacity = 2048U;
constexpr uint32_t kTelemetryHeapCaps =
    MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT;

constexpr char kStatusReady[] = "READY";
constexpr char kStatusOk[] = "OK";
constexpr char kStatusError[] = "ERROR";
constexpr char kNvsNamespace[] = "edgeai_vm";
constexpr char kNvsKey[] = "templates";
constexpr char kPersistenceSource[] = "NVS_BOOT_READONLY";

static_assert(kOutputBytes == edgeai_template::kEmbeddingDim,
              "template dimension must match model output");
// Records/readback are heap-scoped. Boot parsing keeps the caller's output and
// ParseRecord's validation candidate simultaneously live on the main stack.
constexpr std::size_t kPersistenceStackWorkingSetBytes =
    2U * sizeof(edgeai_template::TemplateSet);
constexpr std::size_t kPersistenceStackSafetyMarginBytes = 8192U;
static_assert(CONFIG_ESP_MAIN_TASK_STACK_SIZE >=
                  kPersistenceStackWorkingSetBytes +
                      kPersistenceStackSafetyMarginBytes,
              "main task stack is too small for NVS template persistence");

enum class PersistenceState {
  kNotInitialized,
  kEmpty,
  kReady,
  kUnavailable,
  kCorrupt,
  kUncertain,
};

const char* PersistenceStateText(PersistenceState state) {
  switch (state) {
    case PersistenceState::kNotInitialized:
      return "NOT_INITIALIZED";
    case PersistenceState::kEmpty:
      return "EMPTY";
    case PersistenceState::kReady:
      return "READY";
    case PersistenceState::kUnavailable:
      return "UNAVAILABLE";
    case PersistenceState::kCorrupt:
      return "CORRUPT";
    case PersistenceState::kUncertain:
      return "PERSISTENCE_UNCERTAIN";
  }
  return "UNKNOWN";
}

DRAM_ATTR alignas(16) uint8_t g_tensor_arena[kArenaBytes];
DRAM_ATTR alignas(tflite::MicroInterpreter)
    uint8_t g_interpreter_storage[sizeof(tflite::MicroInterpreter)];
DRAM_ATTR uint8_t g_request_payload[kLegacyMaxRequestPayloadBytes];
char g_response[kResponseCapacity];
tflite::MicroMutableOpResolver<4> g_resolver;
tflite::MicroInterpreter* g_interpreter = nullptr;
std::size_t g_arena_used_bytes = 0U;
edgeai_template::TemplateVectors g_templates{};
edgeai_template::TemplateLabels g_template_labels{};
bool g_templates_loaded = false;
uint8_t g_template_count = 0U;
PersistenceState g_persistence_state = PersistenceState::kNotInitialized;
uint32_t g_template_record_crc32 = 0U;
std::size_t g_template_record_bytes = 0U;

void SetBuiltInTemplateLabels(edgeai_template::TemplateLabels* labels) {
  std::memset(labels, 0, sizeof(*labels));
  std::memcpy(labels->values[0], "Quang", sizeof("Quang"));
  std::memcpy(labels->values[1], "Tien", sizeof("Tien"));
  std::memcpy(labels->values[2], "An", sizeof("An"));
}

bool TemplatesReady() {
  return g_templates_loaded && g_persistence_state == PersistenceState::kReady &&
         g_template_count >= edgeai_template::kLegacyTemplateSlots &&
         g_template_count <= edgeai_template::kTemplateSlots;
}

struct HeapTelemetry {
  uint64_t uptime_us;
  uint32_t internal_heap_total_bytes;
  uint32_t internal_heap_free_bytes;
  uint32_t internal_heap_min_free_bytes;
  uint32_t internal_heap_largest_free_block_bytes;
};

HeapTelemetry CaptureHeapTelemetry() {
  const int64_t uptime_us = esp_timer_get_time();
  return {
      uptime_us < 0 ? 0U : static_cast<uint64_t>(uptime_us),
      static_cast<uint32_t>(heap_caps_get_total_size(kTelemetryHeapCaps)),
      static_cast<uint32_t>(heap_caps_get_free_size(kTelemetryHeapCaps)),
      static_cast<uint32_t>(
          heap_caps_get_minimum_free_size(kTelemetryHeapCaps)),
      static_cast<uint32_t>(
          heap_caps_get_largest_free_block(kTelemetryHeapCaps)),
  };
}

uint32_t CurrentInternalHeapFreeBytes() {
  return static_cast<uint32_t>(heap_caps_get_free_size(kTelemetryHeapCaps));
}

struct Fixture {
  const char* name;
  const uint8_t* input;
  const uint8_t* input_end;
  const uint8_t* expected;
  const uint8_t* expected_end;
};

bool NearlyEqual(float actual, float expected) {
  return std::isfinite(actual) && std::fabs(actual - expected) <= 1.0e-6F;
}

bool ExpectedShape(const flatbuffers::Vector<int32_t>* shape,
                   const int32_t* expected, std::size_t rank) {
  if (shape == nullptr || shape->size() != rank) {
    return false;
  }
  for (std::size_t index = 0; index < rank; ++index) {
    if (shape->Get(index) != expected[index]) {
      return false;
    }
  }
  return true;
}

bool ModelQuantization(const tflite::Tensor* tensor, float expected_scale,
                       int expected_zero_point, const char* label) {
  if (tensor == nullptr || tensor->quantization() == nullptr ||
      tensor->quantization()->scale() == nullptr ||
      tensor->quantization()->zero_point() == nullptr ||
      tensor->quantization()->scale()->size() != 1U ||
      tensor->quantization()->zero_point()->size() != 1U) {
    ESP_LOGE(kTag, "%s quantization metadata is missing or per-axis", label);
    return false;
  }
  const float scale = tensor->quantization()->scale()->Get(0);
  const int64_t zero_point = tensor->quantization()->zero_point()->Get(0);
  if (!NearlyEqual(scale, expected_scale) || zero_point != expected_zero_point) {
    ESP_LOGE(kTag,
             "%s quantization mismatch scale=%f zero_point=%" PRId64,
             label, static_cast<double>(scale), zero_point);
    return false;
  }
  ESP_LOGI(kTag, "%s quantization scale=%f zero_point=%" PRId64, label,
           static_cast<double>(scale), zero_point);
  return true;
}

bool CheckModel(const uint8_t* model_bytes, std::size_t model_size,
                const tflite::Model** model_out) {
  if (model_bytes == nullptr || model_size != kModelBytes) {
    ESP_LOGE(kTag, "model byte check failed expected=%u actual=%u",
             static_cast<unsigned>(kModelBytes),
             static_cast<unsigned>(model_size));
    return false;
  }
  ESP_LOGI(kTag, "model_sha256_expected=%s model_bytes=%u", kModelSha256,
           static_cast<unsigned>(model_size));
  if (!tflite::ModelBufferHasIdentifier(model_bytes)) {
    ESP_LOGE(kTag, "model FlatBuffer identifier check failed");
    return false;
  }
  flatbuffers::Verifier verifier(model_bytes, model_size);
  if (!tflite::VerifyModelBuffer(verifier)) {
    ESP_LOGE(kTag, "model FlatBuffer verification failed");
    return false;
  }
  const tflite::Model* model = tflite::GetModel(model_bytes);
  if (model == nullptr || model->version() != TFLITE_SCHEMA_VERSION ||
      model->subgraphs() == nullptr || model->subgraphs()->size() != 1U) {
    ESP_LOGE(kTag, "model schema/subgraph check failed version=%u",
             model == nullptr ? 0U : static_cast<unsigned>(model->version()));
    return false;
  }
  const tflite::SubGraph* subgraph = model->subgraphs()->Get(0);
  if (subgraph == nullptr || subgraph->tensors() == nullptr ||
      subgraph->inputs() == nullptr || subgraph->outputs() == nullptr ||
      subgraph->inputs()->size() != 1U ||
      subgraph->outputs()->size() != 1U) {
    ESP_LOGE(kTag, "model graph I/O vector check failed");
    return false;
  }
  const int32_t input_index = subgraph->inputs()->Get(0);
  const int32_t output_index = subgraph->outputs()->Get(0);
  const auto* tensors = subgraph->tensors();
  if (input_index < 0 || output_index < 0 ||
      static_cast<std::size_t>(input_index) >= tensors->size() ||
      static_cast<std::size_t>(output_index) >= tensors->size()) {
    ESP_LOGE(kTag, "model graph I/O indices are outside the tensor table");
    return false;
  }
  const tflite::Tensor* input = tensors->Get(input_index);
  const tflite::Tensor* output = tensors->Get(output_index);
  constexpr int32_t kInputShape[] = {1, 298, 40, 1};
  constexpr int32_t kOutputShape[] = {1, 192};
  if (input == nullptr || output == nullptr ||
      input->type() != tflite::TensorType_INT8 ||
      output->type() != tflite::TensorType_INT8 ||
      !ExpectedShape(input->shape(), kInputShape, 4U) ||
      !ExpectedShape(output->shape(), kOutputShape, 2U) ||
      !ModelQuantization(input, kInputScale, kInputZeroPoint, "model input") ||
      !ModelQuantization(output, kOutputScale, kOutputZeroPoint,
                         "model output")) {
    ESP_LOGE(kTag, "model I/O schema check failed");
    return false;
  }
  ESP_LOGI(kTag, "schema=%u I/O input=[1,298,40,1] int8 bytes=%u output=[1,192] int8 bytes=%u",
           static_cast<unsigned>(model->version()),
           static_cast<unsigned>(kInputBytes),
           static_cast<unsigned>(kOutputBytes));
  *model_out = model;
  return true;
}

bool CheckRuntimeTensor(const TfLiteTensor* tensor, const int* shape,
                        std::size_t rank, std::size_t expected_bytes,
                        float expected_scale, int expected_zero_point,
                        const char* label) {
  if (tensor == nullptr || tensor->type != kTfLiteInt8 || tensor->dims == nullptr ||
      tensor->dims->size != static_cast<int>(rank) ||
      tensor->bytes != expected_bytes || tensor->data.int8 == nullptr) {
    ESP_LOGE(kTag, "%s runtime tensor shape/type/bytes check failed", label);
    return false;
  }
  for (std::size_t index = 0; index < rank; ++index) {
    if (tensor->dims->data[index] != shape[index]) {
      ESP_LOGE(kTag, "%s runtime tensor dimension %u mismatch", label,
               static_cast<unsigned>(index));
      return false;
    }
  }
  if (!NearlyEqual(tensor->params.scale, expected_scale) ||
      tensor->params.zero_point != expected_zero_point) {
    ESP_LOGE(kTag, "%s runtime quantization mismatch scale=%f zero_point=%d",
             label, static_cast<double>(tensor->params.scale),
             tensor->params.zero_point);
    return false;
  }
  return true;
}

void LogRawOutput(const Fixture& fixture, const int8_t* output) {
  char line[3U * 16U + 1U] = {};
  for (std::size_t offset = 0; offset < kOutputBytes; offset += 16U) {
    std::size_t position = 0U;
    for (std::size_t index = 0; index < 16U; ++index) {
      const int written = std::snprintf(
          line + position, sizeof(line) - position, "%02x",
          static_cast<unsigned>(static_cast<uint8_t>(output[offset + index])));
      if (written < 0 || static_cast<std::size_t>(written) >= sizeof(line) - position) {
        line[0] = '\0';
        break;
      }
      position += static_cast<std::size_t>(written);
    }
    ESP_LOGI(kTag, "raw192bytes fixture=%s offset=%u data=%s", fixture.name,
             static_cast<unsigned>(offset), line);
  }
}

bool RunFixture(tflite::MicroInterpreter* interpreter,
                const Fixture& fixture) {
  if (static_cast<std::size_t>(fixture.input_end - fixture.input) !=
          kInputBytes ||
      static_cast<std::size_t>(fixture.expected_end - fixture.expected) !=
          kOutputBytes) {
    ESP_LOGE(kTag, "fixture=%s embedded byte lengths are invalid", fixture.name);
    return false;
  }
  TfLiteTensor* input = interpreter->input(0);
  TfLiteTensor* output = interpreter->output(0);
  if (input == nullptr || output == nullptr || input->data.int8 == nullptr ||
      output->data.int8 == nullptr) {
    ESP_LOGE(kTag, "fixture=%s runtime I/O buffers are unavailable", fixture.name);
    return false;
  }
  std::memcpy(input->data.int8, fixture.input, kInputBytes);
  const int64_t begin_us = esp_timer_get_time();
  const TfLiteStatus invoke_status = interpreter->Invoke();
  const int64_t elapsed_us = esp_timer_get_time() - begin_us;
  if (invoke_status != kTfLiteOk) {
    ESP_LOGE(kTag, "fixture=%s Invoke failed status=%d invoke_us=%" PRId64,
             fixture.name, static_cast<int>(invoke_status), elapsed_us);
    return false;
  }
  int maximum = 0;
  for (std::size_t index = 0; index < kOutputBytes; ++index) {
    const int actual = static_cast<int>(output->data.int8[index]);
    const int expected = static_cast<int>(
        reinterpret_cast<const int8_t*>(fixture.expected)[index]);
    const int difference = std::abs(actual - expected);
    if (difference > maximum) {
      maximum = difference;
    }
  }
  LogRawOutput(fixture, output->data.int8);
  const bool pass = maximum <= kMaxAbsDiffLsb;
  ESP_LOGI(kTag,
           "fixture=%s invoke_us=%" PRId64 " max_absdiff_lsb=%d cutoff=%d status=%s",
           fixture.name, elapsed_us, maximum, kMaxAbsDiffLsb,
           pass ? "PASS" : "FAIL");
  return pass;
}

bool InitializeRuntime(const tflite::Model* model) {
  if (model == nullptr || g_interpreter != nullptr) {
    return g_interpreter != nullptr;
  }
  if (g_resolver.AddConv2D() != kTfLiteOk ||
      g_resolver.AddAveragePool2D() != kTfLiteOk ||
      g_resolver.AddReshape() != kTfLiteOk ||
      g_resolver.AddFullyConnected() != kTfLiteOk) {
    ESP_LOGE(kTag, "operator resolver registration failed");
    return false;
  }
  tflite::MicroInterpreter* interpreter = new (g_interpreter_storage)
      tflite::MicroInterpreter(model, g_resolver, g_tensor_arena,
                               sizeof(g_tensor_arena));
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    ESP_LOGE(kTag, "AllocateTensors failed with arena_capacity=%u",
             static_cast<unsigned>(sizeof(g_tensor_arena)));
    return false;
  }
  g_interpreter = interpreter;
  g_arena_used_bytes = interpreter->arena_used_bytes();
  ESP_LOGI(kTag,
           "arena location_start=%p location_end=%p internal=DRAM capacity_bytes=%u "
           "used_bytes=%u aligned16=%s",
           static_cast<void*>(g_tensor_arena),
           static_cast<void*>(g_tensor_arena + sizeof(g_tensor_arena)),
           static_cast<unsigned>(sizeof(g_tensor_arena)),
           static_cast<unsigned>(g_arena_used_bytes),
           (reinterpret_cast<uintptr_t>(g_tensor_arena) % 16U) == 0U ? "yes"
                                                                    : "no");
  return true;
}

bool RunSmoke(tflite::MicroInterpreter* interpreter) {
  if (interpreter == nullptr) {
    return false;
  }
  constexpr int kInputShape[] = {1, 298, 40, 1};
  constexpr int kOutputShape[] = {1, 192};
  if (!CheckRuntimeTensor(interpreter->input(0), kInputShape, 4U, kInputBytes,
                          kInputScale, kInputZeroPoint, "runtime input") ||
      !CheckRuntimeTensor(interpreter->output(0), kOutputShape, 2U,
                          kOutputBytes, kOutputScale, kOutputZeroPoint,
                          "runtime output")) {
    return false;
  }
  ESP_LOGI(kTag,
           "reference=BUILTIN_REF_NO_DELEGATES max_absdiff_lsb_cutoff=%d "
           "comparison=desktop_expected_output",
           kMaxAbsDiffLsb);
  const Fixture fixtures[kFixtureCount] = {
      {"real_train_000", _binary_real_train_000_input_int8_start,
       _binary_real_train_000_input_int8_end,
       _binary_real_train_000_expected_int8_start,
       _binary_real_train_000_expected_int8_end},
      {"real_train_001", _binary_real_train_001_input_int8_start,
       _binary_real_train_001_input_int8_end,
       _binary_real_train_001_expected_int8_start,
       _binary_real_train_001_expected_int8_end},
      {"real_train_002", _binary_real_train_002_input_int8_start,
       _binary_real_train_002_input_int8_end,
       _binary_real_train_002_expected_int8_start,
       _binary_real_train_002_expected_int8_end},
      {"real_train_003", _binary_real_train_003_input_int8_start,
       _binary_real_train_003_input_int8_end,
       _binary_real_train_003_expected_int8_start,
       _binary_real_train_003_expected_int8_end},
      {"real_train_004", _binary_real_train_004_input_int8_start,
       _binary_real_train_004_input_int8_end,
       _binary_real_train_004_expected_int8_start,
       _binary_real_train_004_expected_int8_end},
  };
  bool all_pass = true;
  for (const Fixture& fixture : fixtures) {
    all_pass = RunFixture(interpreter, fixture) && all_pass;
  }
  return all_pass;
}

void MarkPersistenceUncertain(const char* operation, esp_err_t status) {
  // nvs_set_blob may report ESP_ERR_NVS_REMOVE_FAILED after writing a value,
  // and IDF permits implementations to write before nvs_commit. Do not infer
  // rollback from either error. Keep the previous RAM templates untouched and
  // fail closed until a reboot reloads the namespace.
  g_persistence_state = PersistenceState::kUncertain;
  ESP_LOGE(kTag,
           "persistence uncertain operation=%s status=0x%x; reboot required "
           "for reconciliation",
           operation, static_cast<unsigned>(status));
}

bool LoadPersistedTemplates() {
  g_templates_loaded = false;
  g_template_count = 0U;
  g_template_record_crc32 = 0U;
  g_template_record_bytes = 0U;
  const esp_err_t init_status = nvs_flash_init();
  if (init_status != ESP_OK) {
    // In particular, do not erase or reformat after NO_FREE_PAGES or any
    // other initialization error. The namespace is intentionally isolated.
    g_persistence_state = PersistenceState::kUnavailable;
    ESP_LOGE(kTag, "NVS init failed status=0x%x; no erase attempted",
             static_cast<unsigned>(init_status));
    return false;
  }

  nvs_handle_t handle = 0;
  const esp_err_t open_status =
      nvs_open(kNvsNamespace, NVS_READONLY, &handle);
  if (open_status == ESP_ERR_NVS_NOT_FOUND) {
    g_persistence_state = PersistenceState::kEmpty;
    ESP_LOGI(kTag, "template namespace=%s state=EMPTY", kNvsNamespace);
    return true;
  }
  if (open_status != ESP_OK) {
    g_persistence_state = PersistenceState::kUnavailable;
    ESP_LOGE(kTag, "NVS read-only open failed status=0x%x",
             static_cast<unsigned>(open_status));
    return false;
  }

  std::size_t record_length = 0U;
  const esp_err_t size_status =
      nvs_get_blob(handle, kNvsKey, nullptr, &record_length);
  if (size_status == ESP_ERR_NVS_NOT_FOUND) {
    nvs_close(handle);
    g_persistence_state = PersistenceState::kEmpty;
    ESP_LOGI(kTag, "template key=%s state=EMPTY", kNvsKey);
    return true;
  }
  if (size_status != ESP_OK ||
      (record_length != edgeai_template::kLegacyRecordBytes &&
       record_length != edgeai_template::kRecordBytes &&
       record_length != edgeai_template::kV3RecordBytes)) {
    nvs_close(handle);
    g_persistence_state = PersistenceState::kCorrupt;
    ESP_LOGE(kTag,
             "template record length invalid status=0x%x bytes=%u",
             static_cast<unsigned>(size_status),
             static_cast<unsigned>(record_length));
    return false;
  }

  uint8_t* record = static_cast<uint8_t*>(
      heap_caps_calloc(1U, edgeai_template::kMaxRecordBytes, kTelemetryHeapCaps));
  if (record == nullptr) {
    nvs_close(handle);
    g_persistence_state = PersistenceState::kUnavailable;
    ESP_LOGE(kTag, "template record heap allocation failed");
    return false;
  }
  std::size_t loaded_length = record_length;
  const esp_err_t read_status =
      nvs_get_blob(handle, kNvsKey, record, &loaded_length);
  nvs_close(handle);
  if (read_status != ESP_OK || loaded_length != record_length) {
    heap_caps_free(record);
    g_persistence_state = PersistenceState::kCorrupt;
    ESP_LOGE(kTag, "template record read failed status=0x%x bytes=%u",
             static_cast<unsigned>(read_status),
             static_cast<unsigned>(loaded_length));
    return false;
  }

  edgeai_template::TemplateSet loaded{};
  uint32_t record_crc32 = 0U;
  if (!edgeai_template::ParseRecord(record, loaded_length, kModelSha256,
                                    &loaded, &record_crc32)) {
    heap_caps_free(record);
    g_persistence_state = PersistenceState::kCorrupt;
    ESP_LOGE(kTag,
             "template record rejected: magic/version/model/labels/CRC/vector "
             "validation failed");
    return false;
  }
  g_templates = loaded.vectors;
  g_template_labels = loaded.labels;
  g_templates_loaded = true;
  g_template_count = loaded.count;
  g_template_record_crc32 = record_crc32;
  g_template_record_bytes = loaded_length;
  g_persistence_state = PersistenceState::kReady;
  heap_caps_free(record);
  ESP_LOGI(kTag, "template record loaded namespace=%s key=%s bytes=%u crc32=%" PRIu32,
           kNvsNamespace, kNvsKey, static_cast<unsigned>(loaded_length),
           record_crc32);
  return true;
}

bool PersistCandidate(const edgeai_template::TemplateVectors& candidate,
                      uint8_t candidate_count,
                      const edgeai_template::TemplateLabels* candidate_labels,
                      const char** error) {
  if (error != nullptr) {
    *error = "persistence_failed";
  }
  uint8_t* record = static_cast<uint8_t*>(heap_caps_calloc(
      1U, edgeai_template::kMaxRecordBytes, kTelemetryHeapCaps));
  if (record == nullptr) {
    if (error != nullptr) *error = "persistence_record_alloc_failed";
    return false;
  }
  const std::size_t record_bytes = candidate_count == edgeai_template::kLegacyTemplateSlots
      ? edgeai_template::kLegacyRecordBytes
      : candidate_count == edgeai_template::kV2TemplateSlots
          ? edgeai_template::kRecordBytes : edgeai_template::kV3RecordBytes;
  uint32_t candidate_crc32 = 0U;
  const bool record_ok = candidate_count == edgeai_template::kLegacyTemplateSlots
      ? edgeai_template::BuildLegacyRecord(candidate, kModelSha256, record,
                                            edgeai_template::kMaxRecordBytes, &candidate_crc32)
      : candidate_count == edgeai_template::kV2TemplateSlots &&
            edgeai_template::BuildRecord(candidate, kModelSha256, record,
                                         edgeai_template::kMaxRecordBytes, &candidate_crc32);
  const bool v3_record_ok = candidate_count > edgeai_template::kV2TemplateSlots &&
      candidate_count <= edgeai_template::kTemplateSlots && candidate_labels != nullptr &&
      edgeai_template::BuildV3Record(candidate, *candidate_labels, candidate_count,
                                    kModelSha256, record,
                                    edgeai_template::kMaxRecordBytes, &candidate_crc32);
  if (!record_ok && !v3_record_ok) {
    heap_caps_free(record);
    if (error != nullptr) {
      *error = "invalid_template_record";
    }
    return false;
  }

  nvs_handle_t write_handle = 0;
  const esp_err_t open_status =
      nvs_open(kNvsNamespace, NVS_READWRITE, &write_handle);
  if (open_status != ESP_OK) {
    heap_caps_free(record);
    MarkPersistenceUncertain("open_write", open_status);
    if (error != nullptr) {
      *error = "persistence_open_failed";
    }
    return false;
  }
  const esp_err_t set_status =
      nvs_set_blob(write_handle, kNvsKey, record, record_bytes);
  if (set_status != ESP_OK) {
    nvs_close(write_handle);
    heap_caps_free(record);
    MarkPersistenceUncertain("set_blob", set_status);
    if (error != nullptr) {
      *error = "persistence_set_failed_reboot_required";
    }
    return false;
  }
  const esp_err_t commit_status = nvs_commit(write_handle);
  nvs_close(write_handle);
  if (commit_status != ESP_OK) {
    heap_caps_free(record);
    MarkPersistenceUncertain("commit", commit_status);
    if (error != nullptr) {
      *error = "persistence_commit_uncertain_reboot_required";
    }
    return false;
  }

  // Reopen read-only so a successful commit is checked through the public
  // read path before the candidate replaces the prior RAM state.
  nvs_handle_t read_handle = 0;
  const esp_err_t read_open_status =
      nvs_open(kNvsNamespace, NVS_READONLY, &read_handle);
  if (read_open_status != ESP_OK) {
    heap_caps_free(record);
    MarkPersistenceUncertain("verify_open", read_open_status);
    if (error != nullptr) {
      *error = "persistence_verify_uncertain_reboot_required";
    }
    return false;
  }
  uint8_t* readback = static_cast<uint8_t*>(heap_caps_calloc(
      1U, edgeai_template::kMaxRecordBytes, kTelemetryHeapCaps));
  if (readback == nullptr) {
    nvs_close(read_handle);
    heap_caps_free(record);
    MarkPersistenceUncertain("verify_alloc", ESP_ERR_NO_MEM);
    if (error != nullptr) *error = "persistence_verify_uncertain_reboot_required";
    return false;
  }
  std::size_t readback_length = record_bytes;
  const esp_err_t read_status =
      nvs_get_blob(read_handle, kNvsKey, readback, &readback_length);
  nvs_close(read_handle);
  if (read_status != ESP_OK || readback_length != record_bytes ||
      std::memcmp(readback, record, record_bytes) != 0) {
    heap_caps_free(readback);
    heap_caps_free(record);
    MarkPersistenceUncertain("verify_readback",
                             read_status == ESP_OK ? ESP_FAIL : read_status);
    if (error != nullptr) {
      *error = "persistence_verify_uncertain_reboot_required";
    }
    return false;
  }

  // Only now publish the candidate. A failed write/commit/readback cannot
  // replace the previous in-RAM templates.
  g_templates = candidate;
  if (candidate_labels != nullptr) {
    g_template_labels = *candidate_labels;
  } else {
    SetBuiltInTemplateLabels(&g_template_labels);
  }
  g_templates_loaded = true;
  g_template_count = candidate_count;
  g_template_record_crc32 = candidate_crc32;
  g_template_record_bytes = record_bytes;
  g_persistence_state = PersistenceState::kReady;
  heap_caps_free(readback);
  heap_caps_free(record);
  if (error != nullptr) {
    *error = nullptr;
  }
  ESP_LOGI(kTag, "template record committed namespace=%s key=%s bytes=%u crc32=%" PRIu32,
           kNvsNamespace, kNvsKey,
           static_cast<unsigned>(record_bytes),
           candidate_crc32);
  return true;
}

uint32_t ReadLe32(const uint8_t* bytes) {
  return static_cast<uint32_t>(bytes[0]) |
         (static_cast<uint32_t>(bytes[1]) << 8U) |
         (static_cast<uint32_t>(bytes[2]) << 16U) |
         (static_cast<uint32_t>(bytes[3]) << 24U);
}

uint32_t Crc32Ieee(const uint8_t* bytes, std::size_t length) {
  uint32_t crc = 0xFFFFFFFFU;
  for (std::size_t index = 0; index < length; ++index) {
    crc ^= static_cast<uint32_t>(bytes[index]);
    for (int bit = 0; bit < 8; ++bit) {
      const uint32_t mask = 0U - (crc & 1U);
      crc = (crc >> 1U) ^ (0xEDB88320U & mask);
    }
  }
  return ~crc;
}

TickType_t ReadWaitTicks(int64_t deadline_us) {
  const int64_t remaining_us = deadline_us - esp_timer_get_time();
  if (remaining_us <= 0) {
    return 0;
  }
  uint64_t remaining_ms =
      (static_cast<uint64_t>(remaining_us) + 999U) / 1000U;
  if (remaining_ms > 100U) {
    remaining_ms = 100U;
  }
  TickType_t ticks = pdMS_TO_TICKS(static_cast<uint32_t>(remaining_ms));
  return ticks == 0 ? 1 : ticks;
}

bool ReadExact(uint8_t* destination, std::size_t length,
               int64_t deadline_us, std::size_t* bytes_read) {
  std::size_t offset = 0U;
  while (offset < length) {
    const TickType_t wait_ticks = ReadWaitTicks(deadline_us);
    if (wait_ticks == 0) {
      break;
    }
    const int received = usb_serial_jtag_read_bytes(
        destination + offset, static_cast<uint32_t>(length - offset),
        wait_ticks);
    if (received > 0) {
      offset += static_cast<std::size_t>(received);
    }
  }
  if (bytes_read != nullptr) {
    *bytes_read = offset;
  }
  return offset == length;
}

bool ReadMagic(uint8_t* header, int64_t deadline_us) {
  std::size_t matched = 0U;
  while (matched < kProtocolMagicBytes) {
    const TickType_t wait_ticks = ReadWaitTicks(deadline_us);
    if (wait_ticks == 0) {
      return false;
    }
    uint8_t byte = 0U;
    if (usb_serial_jtag_read_bytes(&byte, 1U, wait_ticks) <= 0) {
      continue;
    }
    if (byte == static_cast<uint8_t>(kProtocolMagic[matched])) {
      header[matched++] = byte;
      if (matched == kProtocolMagicBytes) {
        return true;
      }
    } else {
      matched = byte == static_cast<uint8_t>(kProtocolMagic[0]) ? 1U : 0U;
      if (matched == 1U) {
        header[0] = byte;
      }
    }
  }
  return false;
}

struct RequestHeader {
  uint32_t request_id = 0U;
  uint32_t payload_length = 0U;
  uint32_t crc32 = 0U;
};

struct RequestFrame {
  RequestHeader header;
  uint32_t payload_crc32 = 0U;
  uint8_t operation = 0U;
  std::size_t operation_payload_offset = 0U;
  bool legacy_encode = false;
  uint8_t* payload = nullptr;
  bool payload_heap_allocated = false;
  const char* error = nullptr;
};

enum class RequestReadResult {
  kNoFrame,
  kReady,
  kError,
};

void ReleaseRequestPayload(RequestFrame* frame) {
  if (frame != nullptr && frame->payload_heap_allocated &&
      frame->payload != nullptr) {
    heap_caps_free(frame->payload);
  }
  if (frame != nullptr) {
    frame->payload = nullptr;
    frame->payload_heap_allocated = false;
  }
}

class RequestPayloadGuard {
 public:
  explicit RequestPayloadGuard(RequestFrame* frame) : frame_(frame) {}
  ~RequestPayloadGuard() { ReleaseRequestPayload(frame_); }

  RequestPayloadGuard(const RequestPayloadGuard&) = delete;
  RequestPayloadGuard& operator=(const RequestPayloadGuard&) = delete;

 private:
  RequestFrame* frame_;
};

RequestReadResult ReadRequest(RequestFrame* frame) {
  uint8_t header[kRequestHeaderBytes] = {};
  if (!ReadMagic(header, esp_timer_get_time() + kFrameTimeoutUs)) {
    return RequestReadResult::kNoFrame;
  }
  const int64_t deadline_us = esp_timer_get_time() + kFrameTimeoutUs;

  std::size_t header_tail_bytes = 0U;
  if (!ReadExact(header + kProtocolMagicBytes,
                 kRequestHeaderBytes - kProtocolMagicBytes, deadline_us,
                 &header_tail_bytes)) {
    if (header_tail_bytes >= sizeof(uint32_t)) {
      frame->header.request_id = ReadLe32(header + kProtocolMagicBytes);
    }
    frame->error = "truncated_header";
    return RequestReadResult::kError;
  }

  frame->header.request_id = ReadLe32(header + 8U);
  frame->header.payload_length = ReadLe32(header + 12U);
  frame->header.crc32 = ReadLe32(header + 16U);
  if (frame->header.payload_length == 0U) {
    if (frame->header.crc32 != 0U) {
      frame->error = "invalid_hello_crc";
      return RequestReadResult::kError;
    }
    return RequestReadResult::kReady;
  }
  if (frame->header.payload_length > kMaxRequestPayloadBytes) {
    frame->error = "invalid_payload_length";
    return RequestReadResult::kError;
  }

  uint8_t* payload = g_request_payload;
  if (frame->header.payload_length > sizeof(g_request_payload)) {
    if (frame->header.payload_length != kRecognizePcmPayloadBytes &&
        frame->header.payload_length != kAppendPcmPayloadBytes) {
      frame->error = "invalid_payload_length";
      return RequestReadResult::kError;
    }
    payload = static_cast<uint8_t*>(heap_caps_malloc(
        frame->header.payload_length, kTelemetryHeapCaps));
    if (payload == nullptr) {
      frame->error = "large_payload_alloc_failed";
      return RequestReadResult::kError;
    }
    frame->payload = payload;
    frame->payload_heap_allocated = true;
  } else {
    frame->payload = g_request_payload;
  }

  std::size_t payload_bytes = 0U;
  if (!ReadExact(payload, frame->header.payload_length, deadline_us,
                 &payload_bytes)) {
    frame->error = "truncated_payload";
    return RequestReadResult::kError;
  }
  frame->payload_crc32 =
      frame->header.payload_length == kInputBytes
          ? Crc32Ieee(g_request_payload, kInputBytes)
          : Crc32Ieee(payload, frame->header.payload_length);
  if (frame->payload_crc32 != frame->header.crc32) {
    frame->error = "invalid_crc";
    return RequestReadResult::kError;
  }

  // The pre-v2 raw tensor frame remains accepted for the old fixture
  // diagnostics. New callers must use the explicit versioned operation header.
  if (frame->header.payload_length == kInputBytes) {
    frame->operation = kOperationEncode;
    frame->legacy_encode = true;
    frame->operation_payload_offset = 0U;
    return RequestReadResult::kReady;
  }
  if (frame->header.payload_length < kOperationHeaderBytes) {
    frame->error = "truncated_operation_header";
    return RequestReadResult::kError;
  }
  frame->operation = payload[0];
  frame->operation_payload_offset = kOperationHeaderBytes;
  if (payload[1] != kOperationVersion) {
    frame->error = "unsupported_operation_version";
    return RequestReadResult::kError;
  }
  const std::size_t payload_length = frame->header.payload_length;
  const bool length_ok =
      (frame->operation == kOperationEncode &&
       payload_length == kEncodePayloadBytes) ||
      (frame->operation == kOperationProvision &&
       payload_length == kProvisionPayloadBytes) ||
      (frame->operation == kOperationStatus &&
       payload_length == kStatusPayloadBytes) ||
      (frame->operation == kOperationRecognize &&
       payload_length == kRecognizePayloadBytes) ||
      (frame->operation == kOperationAddAn &&
       payload_length == kAddAnPayloadBytes) ||
      (frame->operation == kOperationRecognizePcm &&
       payload_length == kRecognizePcmPayloadBytes) ||
      (frame->operation == kOperationAppendPcm &&
       payload_length == kAppendPcmPayloadBytes);
  if (!length_ok) {
    frame->error = "invalid_operation_payload_length";
    return RequestReadResult::kError;
  }
  return RequestReadResult::kReady;
}

class ResponseWriter {
 public:
  ResponseWriter(char* buffer, std::size_t capacity)
      : buffer_(buffer), capacity_(capacity), length_(0U) {}

  bool Append(const char* text) {
    const std::size_t text_length = std::strlen(text);
    if (text_length >= capacity_ - length_) {
      return false;
    }
    std::memcpy(buffer_ + length_, text, text_length);
    length_ += text_length;
    return true;
  }

  bool AppendFormat(const char* format, ...) {
    if (length_ >= capacity_) {
      return false;
    }
    va_list arguments;
    va_start(arguments, format);
    const int written = std::vsnprintf(buffer_ + length_,
                                       capacity_ - length_, format, arguments);
    va_end(arguments);
    if (written < 0 || static_cast<std::size_t>(written) >= capacity_ - length_) {
      return false;
    }
    length_ += static_cast<std::size_t>(written);
    return true;
  }

  bool AppendQuoted(const char* text) {
    if (!Append("\"")) {
      return false;
    }
    for (const char* cursor = text; *cursor != '\0'; ++cursor) {
      if (*cursor == '"' || *cursor == '\\') {
        if (!Append("\\")) {
          return false;
        }
      }
      if (!AppendFormat("%c", *cursor)) {
        return false;
      }
    }
    return Append("\"");
  }

  bool AppendHex(const int8_t* bytes, std::size_t length) {
    constexpr char kHex[] = "0123456789abcdef";
    for (std::size_t index = 0; index < length; ++index) {
      const uint8_t value = static_cast<uint8_t>(bytes[index]);
      if (!AppendFormat("%c%c", kHex[value >> 4U], kHex[value & 0x0FU])) {
        return false;
      }
    }
    return true;
  }

  bool Finish() {
    return Append("}\n");
  }

  const char* data() const { return buffer_; }
  std::size_t length() const { return length_; }

 private:
  char* buffer_;
  std::size_t capacity_;
  std::size_t length_;
};

bool AppendTelemetry(ResponseWriter* response, const HeapTelemetry& telemetry,
                     bool include_query, uint32_t query_free_before_bytes,
                     int64_t recognition_us) {
  if (!response->Append(
          ",\"telemetry\":{\"version\":1,\"uptime_us\":") ||
      !response->AppendFormat("%" PRIu64, telemetry.uptime_us) ||
      !response->AppendFormat(
          ",\"internal_heap_total_bytes\":%" PRIu32
          ",\"internal_heap_free_bytes\":%" PRIu32
          ",\"internal_heap_min_free_bytes\":%" PRIu32
          ",\"internal_heap_largest_free_block_bytes\":%" PRIu32,
          telemetry.internal_heap_total_bytes,
          telemetry.internal_heap_free_bytes,
          telemetry.internal_heap_min_free_bytes,
          telemetry.internal_heap_largest_free_block_bytes)) {
    return false;
  }
  if (include_query &&
      (!response->AppendFormat(
           ",\"query_heap_free_before_bytes\":%" PRIu32
           ",\"query_heap_free_after_bytes\":%" PRIu32
           ",\"query_heap_delta_bytes\":%" PRId64
           ",\"recognition_us\":%" PRId64,
           query_free_before_bytes,
           telemetry.internal_heap_free_bytes,
           static_cast<int64_t>(query_free_before_bytes) -
               static_cast<int64_t>(telemetry.internal_heap_free_bytes),
           recognition_us))) {
    return false;
  }
  return response->Append("}");
}

bool SendResponse(ResponseWriter* response) {
  const int written = usb_serial_jtag_write_bytes(
      response->data(), response->length(), pdMS_TO_TICKS(2000));
  if (written != static_cast<int>(response->length())) {
    ESP_LOGE(kTag, "USB response write failed bytes=%d expected=%u", written,
             static_cast<unsigned>(response->length()));
    return false;
  }
  if (usb_serial_jtag_wait_tx_done(pdMS_TO_TICKS(2000)) != ESP_OK) {
    ESP_LOGE(kTag, "USB response TX timeout");
    return false;
  }
  return true;
}

bool BeginResponse(ResponseWriter* response, uint32_t request_id,
                   const char* status) {
  return response->Append("EDGEAI_RESPONSE {\"protocol_version\":1,\"request_id\":") &&
         response->AppendFormat("%" PRIu32, request_id) &&
         response->Append(",\"status\":") && response->AppendQuoted(status) &&
         response->Append(",\"model_sha256\":") &&
         response->AppendQuoted(kModelSha256) &&
         response->Append(",\"runtime_mode\":") &&
         response->AppendQuoted(kRuntimeMode) &&
         response->AppendFormat(
             ",\"input_bytes\":%u,\"output_bytes\":%u,\"input_scale\":%s,"
             "\"input_zero_point\":%d,\"output_scale\":%s,"
             "\"output_zero_point\":%d,\"arena_capacity_bytes\":%u,"
             "\"arena_used_bytes\":%u",
             static_cast<unsigned>(kInputBytes),
             static_cast<unsigned>(kOutputBytes), kInputScaleText,
             kInputZeroPoint, kOutputScaleText, kOutputZeroPoint,
             static_cast<unsigned>(kArenaBytes),
             static_cast<unsigned>(g_arena_used_bytes));
}

bool AppendOperation(ResponseWriter* response, const char* operation) {
  return response->Append(",\"operation\":") &&
         response->AppendQuoted(operation);
}

const char* OperationName(uint8_t operation) {
  switch (operation) {
    case kOperationEncode:
      return "ENCODE";
    case kOperationProvision:
      return "PROVISION";
    case kOperationStatus:
      return "STATUS";
    case kOperationRecognize:
      return "RECOGNIZE";
    case kOperationAddAn:
      return "ADD_AN";
    case kOperationRecognizePcm:
      return "RECOGNIZE_PCM";
    case kOperationAppendPcm:
      return "APPEND_PCM";
    default:
      return "FRAME";
  }
}

bool AppendPersistenceStatus(ResponseWriter* response) {
  const unsigned enrollment_count =
      g_persistence_state == PersistenceState::kReady && g_templates_loaded
          ? g_template_count
          : 0U;
  return response->Append(",\"persistence_state\":") &&
         response->AppendQuoted(PersistenceStateText(g_persistence_state)) &&
         response->Append(",\"persistence_source\":") &&
         response->AppendQuoted(kPersistenceSource) &&
         response->AppendFormat(
             ",\"enrollment_count\":%u,\"record_bytes\":%u"
             ",\"record_crc32\":%" PRIu32
             ",\"record_fingerprint\":\"crc32:%08" PRIx32 "\"",
             enrollment_count,
             static_cast<unsigned>(g_template_record_bytes),
             g_template_record_crc32, g_template_record_crc32);
}

bool AppendActiveLabels(ResponseWriter* response) {
  if (!response->Append(",\"labels\":[")) return false;
  const std::size_t count =
      g_persistence_state == PersistenceState::kReady && g_templates_loaded
          ? g_template_count : 0U;
  for (std::size_t slot = 0; slot < count; ++slot) {
    if ((slot != 0U && !response->Append(",")) ||
        !response->AppendQuoted(g_template_labels.values[slot])) return false;
  }
  return response->Append("]");
}

bool SendHelloResponse(uint32_t request_id) {
  ResponseWriter response(g_response, sizeof(g_response));
  if (!BeginResponse(&response, request_id, kStatusReady) ||
      !AppendOperation(&response, "HELLO") ||
      !response.AppendFormat(
          ",\"raw_pcm\":{\"operation\":%u,\"operation_version\":%u,"
          "\"sample_rate_hz\":%u,\"samples\":%u,\"bytes\":%u}",
          static_cast<unsigned>(kOperationRecognizePcm),
          static_cast<unsigned>(kOperationVersion),
          static_cast<unsigned>(edgeai_pcm_frontend::kSampleRateHz),
          static_cast<unsigned>(edgeai_pcm_frontend::kPcmSamples),
          static_cast<unsigned>(edgeai_pcm_frontend::kPcmBytes)) ||
      !response.AppendFormat(",\"append_pcm\":{\"operation\":%u,\"operation_version\":%u,\"label_bytes\":%u,\"sample_rate_hz\":%u,\"samples\":%u,\"bytes\":%u,\"max_templates\":%u}",
          static_cast<unsigned>(kOperationAppendPcm), static_cast<unsigned>(kOperationVersion),
          static_cast<unsigned>(kAppendPcmLabelBytes), static_cast<unsigned>(edgeai_pcm_frontend::kSampleRateHz),
          static_cast<unsigned>(edgeai_pcm_frontend::kPcmSamples), static_cast<unsigned>(edgeai_pcm_frontend::kPcmBytes),
          static_cast<unsigned>(edgeai_template::kTemplateSlots)) ||
      !response.Finish()) {
    ESP_LOGE(kTag, "HELLO response formatting failed");
    return false;
  }
  return SendResponse(&response);
}

bool SendErrorResponse(uint32_t request_id, const char* error,
                       uint8_t operation = 0U) {
  ResponseWriter response(g_response, sizeof(g_response));
  if (!BeginResponse(&response, request_id, kStatusError) ||
      !AppendOperation(&response, OperationName(operation)) ||
      !response.Append(",\"error\":") || !response.AppendQuoted(error) ||
      !response.Finish()) {
    ESP_LOGE(kTag, "ERROR response formatting failed");
    return false;
  }
  return SendResponse(&response);
}

bool InvokeRequest(const uint8_t* input_bytes, int64_t* invoke_us) {
  if (g_interpreter == nullptr || input_bytes == nullptr || invoke_us == nullptr) {
    return false;
  }
  TfLiteTensor* input = g_interpreter->input(0);
  TfLiteTensor* output = g_interpreter->output(0);
  if (input == nullptr || output == nullptr || input->data.int8 == nullptr ||
      output->data.int8 == nullptr) {
    return false;
  }
  if (reinterpret_cast<const uint8_t*>(input->data.int8) != input_bytes) {
    std::memcpy(input->data.int8, input_bytes, kInputBytes);
  }
  const int64_t begin_us = esp_timer_get_time();
  const TfLiteStatus invoke_status = g_interpreter->Invoke();
  *invoke_us = esp_timer_get_time() - begin_us;
  return invoke_status == kTfLiteOk;
}

bool SendInferenceResponse(uint32_t request_id, uint32_t input_crc32,
                           int64_t invoke_us, const char* operation) {
  if (g_interpreter == nullptr || g_interpreter->output(0) == nullptr ||
      g_interpreter->output(0)->data.int8 == nullptr) {
    return false;
  }
  ResponseWriter response(g_response, sizeof(g_response));
  if (!BeginResponse(&response, request_id, kStatusOk) ||
      !AppendOperation(&response, operation) ||
      !response.AppendFormat(",\"input_crc32\":%" PRIu32
                             ",\"output_hex\":\"",
                             input_crc32) ||
      !response.AppendHex(g_interpreter->output(0)->data.int8, kOutputBytes) ||
      !response.AppendFormat("\",\"invoke_us\":%" PRId64, invoke_us) ||
      !response.Finish()) {
    ESP_LOGE(kTag, "INFER response formatting failed");
    return false;
  }
  return SendResponse(&response);
}

bool SendStatusResponse(uint32_t request_id) {
  const HeapTelemetry telemetry = CaptureHeapTelemetry();
  ResponseWriter response(g_response, sizeof(g_response));
  if (!BeginResponse(&response, request_id, kStatusOk) ||
      !AppendOperation(&response, "STATUS") ||
      !AppendPersistenceStatus(&response) ||
      !AppendActiveLabels(&response) ||
      !AppendTelemetry(&response, telemetry, false, 0U, 0) ||
      !response.Finish()) {
    ESP_LOGE(kTag, "STATUS response formatting failed");
    return false;
  }
  return SendResponse(&response);
}

bool SendProvisionResponse(uint32_t request_id, uint32_t request_crc32) {
  ResponseWriter response(g_response, sizeof(g_response));
  if (!BeginResponse(&response, request_id, kStatusOk) ||
      !AppendOperation(&response, "PROVISION") ||
      !AppendPersistenceStatus(&response) ||
      !response.AppendFormat(",\"request_crc32\":%" PRIu32, request_crc32) ||
      !AppendActiveLabels(&response) ||
      !response.Finish()) {
    ESP_LOGE(kTag, "PROVISION response formatting failed");
    return false;
  }
  return SendResponse(&response);
}

bool SendRecognitionResponseInternal(uint32_t request_id,
                                     uint32_t input_crc32, int64_t invoke_us,
                                     uint32_t query_free_before_bytes,
                                     int64_t recognition_begin_us,
                                     const char* operation,
                                     bool include_pcm_metadata,
                                     int64_t frontend_us,
                                     uint32_t feature_crc32) {
  if (!g_templates_loaded ||
      g_persistence_state != PersistenceState::kReady ||
      g_template_count < edgeai_template::kLegacyTemplateSlots ||
      g_template_count > edgeai_template::kTemplateSlots ||
      g_interpreter == nullptr || g_interpreter->output(0) == nullptr ||
      g_interpreter->output(0)->data.int8 == nullptr) {
    return false;
  }
  float query[edgeai_template::kEmbeddingDim] = {};
  if (!edgeai_template::DequantizeAndNormalize(
          g_interpreter->output(0)->data.int8, kOutputBytes, kOutputScale,
          kOutputZeroPoint, query)) {
    return false;
  }
  float scores[edgeai_template::kTemplateSlots] = {};
  for (std::size_t slot = 0; slot < g_template_count; ++slot) {
    if (!edgeai_template::CosineSimilarity(
            query, g_templates.values[slot], edgeai_template::kEmbeddingDim,
            &scores[slot])) {
      return false;
    }
  }
  // Stable insertion sort: ties retain fixed slot order Quang, Tien, An.
  std::size_t order[edgeai_template::kTemplateSlots] = {0U, 1U, 2U, 3U, 4U};
  for (std::size_t left = 1U; left < g_template_count; ++left) {
    const std::size_t current = order[left];
    std::size_t position = left;
    while (position > 0U && scores[current] > scores[order[position - 1U]]) {
      order[position] = order[position - 1U];
      --position;
    }
    order[position] = current;
  }
  const int64_t recognition_us = esp_timer_get_time() - recognition_begin_us;
  const HeapTelemetry telemetry = CaptureHeapTelemetry();
  ResponseWriter response(g_response, sizeof(g_response));
  if (!BeginResponse(&response, request_id, kStatusOk) ||
      !AppendOperation(&response, operation) ||
      !response.AppendFormat(",\"input_crc32\":%" PRIu32
                             ",\"output_hex\":\"",
                             input_crc32) ||
      !response.AppendHex(g_interpreter->output(0)->data.int8, kOutputBytes) ||
      !response.AppendFormat("\",\"invoke_us\":%" PRId64, invoke_us)) {
    ESP_LOGE(kTag, "RECOGNIZE response formatting failed");
    return false;
  }
  if (include_pcm_metadata &&
      !response.AppendFormat(
          ",\"pcm_sample_rate_hz\":%u,\"pcm_samples\":%u,"
          "\"frontend_id\":\"LOGMEL_16K_400_160_512_40_CMN_V1\","
          "\"frontend_us\":%" PRId64 ",\"feature_crc32\":%" PRIu32,
          static_cast<unsigned>(edgeai_pcm_frontend::kSampleRateHz),
          static_cast<unsigned>(edgeai_pcm_frontend::kPcmSamples), frontend_us,
          feature_crc32)) {
    ESP_LOGE(kTag, "RECOGNIZE_PCM frontend metadata formatting failed");
    return false;
  }
  if (!AppendTelemetry(&response, telemetry, true, query_free_before_bytes,
                       recognition_us) ||
      !response.Append(",\"top1\":") ||
      !response.AppendQuoted(g_template_labels.values[order[0]]) ||
      !response.Append(",\"scores\":[")) {
    ESP_LOGE(kTag, "RECOGNIZE response formatting failed");
    return false;
  }
  for (std::size_t rank = 0; rank < g_template_count; ++rank) {
    if ((rank != 0U && !response.Append(",")) ||
        !response.Append("{\"label\":" ) ||
        !response.AppendQuoted(g_template_labels.values[order[rank]]) ||
        !response.AppendFormat(",\"cosine\":%.9f}", scores[order[rank]])) {
      ESP_LOGE(kTag, "RECOGNIZE score formatting failed");
      return false;
    }
  }
  if (!response.Append("]") || !response.Finish()) {
    ESP_LOGE(kTag, "RECOGNIZE response finish failed");
    return false;
  }
  return SendResponse(&response);
}

bool SendRecognitionResponse(uint32_t request_id, uint32_t input_crc32,
                             int64_t invoke_us,
                             uint32_t query_free_before_bytes,
                             int64_t recognition_begin_us) {
  return SendRecognitionResponseInternal(
      request_id, input_crc32, invoke_us, query_free_before_bytes,
      recognition_begin_us, "RECOGNIZE", false, 0, 0U);
}

bool SendPcmRecognitionResponse(uint32_t request_id, uint32_t input_crc32,
                                int64_t invoke_us,
                                uint32_t query_free_before_bytes,
                                int64_t recognition_begin_us,
                                int64_t frontend_us,
                                uint32_t feature_crc32) {
  return SendRecognitionResponseInternal(
      request_id, input_crc32, invoke_us, query_free_before_bytes,
      recognition_begin_us, "RECOGNIZE_PCM", true, frontend_us, feature_crc32);
}

bool RecognizePcmFromInput(uint32_t request_id, uint8_t* raw_pcm,
                           uint32_t input_crc32) {
  if (g_interpreter == nullptr || raw_pcm == nullptr) {
    SendErrorResponse(request_id, "pcm_frontend_unavailable",
                      kOperationRecognizePcm);
    return false;
  }
  TfLiteTensor* input = g_interpreter->input(0);
  if (input == nullptr || input->type != kTfLiteInt8 ||
      input->bytes != kInputBytes || input->data.int8 == nullptr ||
      !std::isfinite(input->params.scale) || input->params.scale <= 0.0F ||
      input->params.zero_point < -128 || input->params.zero_point > 127) {
    SendErrorResponse(request_id, "pcm_input_tensor_invalid",
                      kOperationRecognizePcm);
    return false;
  }

  const uint32_t query_free_before_bytes = CurrentInternalHeapFreeBytes();
  const int64_t recognition_begin_us = esp_timer_get_time();
  const int64_t frontend_begin_us = recognition_begin_us;
  void* workspace_memory = heap_caps_aligned_alloc(
      alignof(edgeai_pcm_frontend::Workspace),
      edgeai_pcm_frontend::kWorkspaceBytes, kTelemetryHeapCaps);
  if (workspace_memory == nullptr) {
    SendErrorResponse(request_id, "pcm_frontend_workspace_alloc_failed",
                      kOperationRecognizePcm);
    return false;
  }
  auto* workspace = static_cast<edgeai_pcm_frontend::Workspace*>(
      workspace_memory);
  const edgeai_pcm_frontend::Status frontend_status =
      edgeai_pcm_frontend::ComputeAndQuantizePcm16(
          raw_pcm, edgeai_pcm_frontend::kPcmBytes, workspace, input->data.int8,
          kInputBytes, input->params.scale, input->params.zero_point);
  heap_caps_free(workspace_memory);
  int64_t frontend_us = esp_timer_get_time() - frontend_begin_us;
  if (frontend_us <= 0) {
    frontend_us = 1;
  }
  if (frontend_status != edgeai_pcm_frontend::Status::kOk) {
    ESP_LOGE(kTag, "PCM frontend failed status=%s request_id=%" PRIu32,
             edgeai_pcm_frontend::StatusName(frontend_status), request_id);
    SendErrorResponse(request_id, "pcm_frontend_failed",
                      kOperationRecognizePcm);
    return false;
  }

  const int64_t feature_crc_begin_us = esp_timer_get_time();
  const uint32_t feature_crc32 =
      Crc32Ieee(reinterpret_cast<const uint8_t*>(input->data.int8),
                kInputBytes);
  const int64_t feature_crc_us = esp_timer_get_time() - feature_crc_begin_us;
  int64_t invoke_us = 0;
  if (!InvokeRequest(reinterpret_cast<const uint8_t*>(input->data.int8),
                     &invoke_us)) {
    ESP_LOGE(kTag, "RECOGNIZE_PCM Invoke failed request_id=%" PRIu32,
             request_id);
    SendErrorResponse(request_id, "invoke_failed", kOperationRecognizePcm);
    return false;
  }
  if (!SendPcmRecognitionResponse(
          request_id, input_crc32, invoke_us, query_free_before_bytes,
          recognition_begin_us + feature_crc_us, frontend_us, feature_crc32)) {
    ESP_LOGE(kTag, "RECOGNIZE_PCM response failed request_id=%" PRIu32,
             request_id);
    SendErrorResponse(request_id, "response_failed", kOperationRecognizePcm);
    return false;
  }
  return true;
}

const char* ProvisionStateError() {
  switch (g_persistence_state) {
    case PersistenceState::kEmpty:
      return nullptr;
    case PersistenceState::kReady:
      return "already_provisioned_no_overwrite";
    case PersistenceState::kUncertain:
      return "persistence_uncertain_reboot_required";
    case PersistenceState::kCorrupt:
      return "persistence_corrupt_operator_repair_required";
    case PersistenceState::kUnavailable:
      return "persistence_unavailable";
    case PersistenceState::kNotInitialized:
      return "persistence_not_initialized";
  }
  return "persistence_state_invalid";
}

bool ProvisionFromInputs(uint32_t request_id, uint32_t request_crc32) {
  const char* state_error = ProvisionStateError();
  if (state_error != nullptr) {
    SendErrorResponse(request_id, state_error, kOperationProvision);
    return false;
  }

  edgeai_template::TemplateVectors candidate{};
  const uint8_t* inputs = g_request_payload + kOperationHeaderBytes;
  for (std::size_t slot = 0; slot < edgeai_template::kLegacyTemplateSlots; ++slot) {
    int64_t invoke_us = 0;
    const uint8_t* input = inputs + slot * kInputBytes;
    if (!InvokeRequest(input, &invoke_us)) {
      ESP_LOGE(kTag, "PROVISION Invoke failed slot=%u request_id=%" PRIu32,
               static_cast<unsigned>(slot), request_id);
      SendErrorResponse(request_id, "provision_invoke_failed",
                        kOperationProvision);
      return false;
    }
    if (g_interpreter == nullptr || g_interpreter->output(0) == nullptr ||
        g_interpreter->output(0)->data.int8 == nullptr ||
        !edgeai_template::DequantizeAndNormalize(
            g_interpreter->output(0)->data.int8, kOutputBytes, kOutputScale,
            kOutputZeroPoint, candidate.values[slot])) {
      ESP_LOGE(kTag, "PROVISION invalid derived vector slot=%u",
               static_cast<unsigned>(slot));
      SendErrorResponse(request_id, "invalid_provision_vector",
                        kOperationProvision);
      return false;
    }
  }

  const char* persistence_error = nullptr;
  if (!PersistCandidate(candidate, edgeai_template::kLegacyTemplateSlots, nullptr,
                        &persistence_error)) {
    SendErrorResponse(request_id,
                      persistence_error == nullptr ? "persistence_failed"
                                                   : persistence_error,
                      kOperationProvision);
    return false;
  }
  if (!SendProvisionResponse(request_id, request_crc32)) {
    ESP_LOGE(kTag, "PROVISION response failed request_id=%" PRIu32,
             request_id);
    return false;
  }
  return true;
}

const char* AddAnStateError() {
  if (g_persistence_state == PersistenceState::kReady && g_templates_loaded) {
    return g_template_count == edgeai_template::kLegacyTemplateSlots
        ? nullptr : "an_already_enrolled_no_overwrite";
  }
  if (g_persistence_state == PersistenceState::kEmpty) return "templates_not_ready";
  if (g_persistence_state == PersistenceState::kUncertain) return "persistence_uncertain_reboot_required";
  if (g_persistence_state == PersistenceState::kCorrupt) return "persistence_corrupt_operator_repair_required";
  if (g_persistence_state == PersistenceState::kUnavailable) return "persistence_unavailable";
  return "persistence_not_initialized";
}

bool SendAddAnResponse(uint32_t request_id, uint32_t request_crc32) {
  ResponseWriter response(g_response, sizeof(g_response));
  if (!BeginResponse(&response, request_id, kStatusOk) ||
      !AppendOperation(&response, "ADD_AN") || !AppendPersistenceStatus(&response) ||
      !response.AppendFormat(",\"request_crc32\":%" PRIu32, request_crc32) ||
      !AppendActiveLabels(&response) || !response.Finish()) return false;
  return SendResponse(&response);
}

bool AddAnFromInput(uint32_t request_id, uint32_t request_crc32) {
  const char* state_error = AddAnStateError();
  if (state_error != nullptr) {
    SendErrorResponse(request_id, state_error, kOperationAddAn);
    return false;
  }
  edgeai_template::TemplateVectors candidate = g_templates;
  int64_t invoke_us = 0;
  const uint8_t* input = g_request_payload + kOperationHeaderBytes;
  if (!InvokeRequest(input, &invoke_us) || g_interpreter == nullptr ||
      g_interpreter->output(0) == nullptr || g_interpreter->output(0)->data.int8 == nullptr ||
      !edgeai_template::DequantizeAndNormalize(g_interpreter->output(0)->data.int8,
          kOutputBytes, kOutputScale, kOutputZeroPoint,
          candidate.values[edgeai_template::kLegacyTemplateSlots])) {
    SendErrorResponse(request_id, "invalid_add_an_vector", kOperationAddAn);
    return false;
  }
  const char* persistence_error = nullptr;
  if (!PersistCandidate(candidate, edgeai_template::kV2TemplateSlots, nullptr,
                        &persistence_error)) {
    SendErrorResponse(request_id, persistence_error == nullptr ? "persistence_failed" : persistence_error,
                      kOperationAddAn);
    return false;
  }
  return SendAddAnResponse(request_id, request_crc32);
}

bool SendAppendPcmResponse(uint32_t request_id, uint32_t request_crc32) {
  ResponseWriter response(g_response, sizeof(g_response));
  return BeginResponse(&response, request_id, kStatusOk) &&
      AppendOperation(&response, "APPEND_PCM") && AppendPersistenceStatus(&response) &&
      response.AppendFormat(",\"request_crc32\":%" PRIu32, request_crc32) &&
      AppendActiveLabels(&response) && response.Finish() && SendResponse(&response);
}

const char* AppendPcmStateError(const char label[edgeai_template::kLabelBytes]) {
  if (!TemplatesReady()) return g_persistence_state == PersistenceState::kUncertain
      ? "persistence_uncertain_reboot_required" : "templates_not_ready";
  if (g_template_count >= edgeai_template::kTemplateSlots) return "template_capacity_full";
  if (!edgeai_template::IsSafeLabel(label)) return "invalid_label";
  for (std::size_t slot = 0; slot < g_template_count; ++slot) {
    if (edgeai_template::LabelsEqual(label, g_template_labels.values[slot])) return "duplicate_label_no_overwrite";
  }
  return nullptr;
}

bool AppendPcmFromInput(uint32_t request_id, uint32_t request_crc32, uint8_t* body) {
  char label[edgeai_template::kLabelBytes] = {};
  std::memcpy(label, body, sizeof(label));
  const char* state_error = AppendPcmStateError(label);
  if (state_error != nullptr) { SendErrorResponse(request_id, state_error, kOperationAppendPcm); return false; }
  if (g_interpreter == nullptr || g_interpreter->input(0) == nullptr ||
      g_interpreter->input(0)->data.int8 == nullptr) {
    SendErrorResponse(request_id, "pcm_frontend_unavailable", kOperationAppendPcm); return false;
  }
  void* memory = heap_caps_aligned_alloc(alignof(edgeai_pcm_frontend::Workspace),
      edgeai_pcm_frontend::kWorkspaceBytes, kTelemetryHeapCaps);
  if (memory == nullptr) { SendErrorResponse(request_id, "pcm_frontend_workspace_alloc_failed", kOperationAppendPcm); return false; }
  TfLiteTensor* input = g_interpreter->input(0);
  const auto status = edgeai_pcm_frontend::ComputeAndQuantizePcm16(
      body + kAppendPcmLabelBytes, edgeai_pcm_frontend::kPcmBytes,
      static_cast<edgeai_pcm_frontend::Workspace*>(memory), input->data.int8,
      kInputBytes, input->params.scale, input->params.zero_point);
  heap_caps_free(memory);
  if (status != edgeai_pcm_frontend::Status::kOk || g_interpreter->Invoke() != kTfLiteOk ||
      g_interpreter->output(0) == nullptr || g_interpreter->output(0)->data.int8 == nullptr) {
    SendErrorResponse(request_id, "append_pcm_encoder_failed", kOperationAppendPcm); return false;
  }
  edgeai_template::TemplateVectors candidate = g_templates;
  edgeai_template::TemplateLabels candidate_labels = g_template_labels;
  if (!edgeai_template::CopySafeLabel(label, candidate_labels.values[g_template_count]) ||
      !edgeai_template::DequantizeAndNormalize(g_interpreter->output(0)->data.int8,
          kOutputBytes, kOutputScale, kOutputZeroPoint, candidate.values[g_template_count])) {
    SendErrorResponse(request_id, "invalid_append_pcm_vector", kOperationAppendPcm); return false;
  }
  const char* persistence_error = nullptr;
  if (!PersistCandidate(candidate, static_cast<uint8_t>(g_template_count + 1U),
                        &candidate_labels, &persistence_error)) {
    SendErrorResponse(request_id, persistence_error == nullptr ? "persistence_failed" : persistence_error,
                      kOperationAppendPcm); return false;
  }
  return SendAppendPcmResponse(request_id, request_crc32);
}

bool StartUsbService() {
  if (usb_serial_jtag_is_driver_installed()) {
    ESP_LOGE(kTag, "USB Serial/JTAG driver is already installed");
    return false;
  }
  usb_serial_jtag_driver_config_t config = {
      .tx_buffer_size = kUsbTxBufferBytes,
      .rx_buffer_size = kUsbRxBufferBytes,
  };
  const esp_err_t install_status = usb_serial_jtag_driver_install(&config);
  if (install_status != ESP_OK) {
    ESP_LOGE(kTag, "USB Serial/JTAG driver install failed status=0x%x",
             static_cast<unsigned>(install_status));
    return false;
  }
  usb_serial_jtag_vfs_use_driver();
  ESP_LOGI(kTag,
           "USB service ready protocol_version=1 operation_version=1 "
           "transport=NATIVE_USB_SERIAL_JTAG frame_timeout_sec=10 "
           "max_payload_bytes=%u",
           static_cast<unsigned>(kMaxRequestPayloadBytes));
  return true;
}

void ServeUsbRequests() {
  do {
    RequestFrame frame;
    const RequestReadResult result = ReadRequest(&frame);
    if (result == RequestReadResult::kNoFrame) {
      continue;
    }
    RequestPayloadGuard payload_guard(&frame);
    if (result == RequestReadResult::kError) {
      SendErrorResponse(frame.header.request_id,
                        frame.error == nullptr ? "invalid_frame" : frame.error,
                        frame.operation);
      continue;
    }
    if (frame.header.payload_length == 0U) {
      SendHelloResponse(frame.header.request_id);
      continue;
    }

    if (frame.operation == kOperationStatus) {
      SendStatusResponse(frame.header.request_id);
      continue;
    }

    if (frame.operation == kOperationProvision) {
      ProvisionFromInputs(frame.header.request_id, frame.payload_crc32);
      continue;
    }

    if (frame.operation == kOperationAddAn) {
      AddAnFromInput(frame.header.request_id, frame.payload_crc32);
      continue;
    }
    if (frame.operation == kOperationAppendPcm) {
      AppendPcmFromInput(frame.header.request_id, frame.payload_crc32,
                         frame.payload + frame.operation_payload_offset);
      continue;
    }

    if (frame.operation != kOperationEncode &&
        frame.operation != kOperationRecognize &&
        frame.operation != kOperationRecognizePcm) {
      SendErrorResponse(frame.header.request_id, "unsupported_operation",
                        frame.operation);
      continue;
    }

    uint8_t* input =
        frame.payload + frame.operation_payload_offset;
    if ((frame.operation == kOperationRecognize ||
         frame.operation == kOperationRecognizePcm) &&
        !TemplatesReady()) {
      SendErrorResponse(frame.header.request_id, "templates_not_ready",
                        frame.operation);
      continue;
    }
    if (frame.operation == kOperationRecognizePcm) {
      const uint32_t input_crc32 =
          Crc32Ieee(input, edgeai_pcm_frontend::kPcmBytes);
      RecognizePcmFromInput(frame.header.request_id, input, input_crc32);
      continue;
    }
    const uint32_t input_crc32 = Crc32Ieee(input, kInputBytes);
    uint32_t query_free_before_bytes = 0U;
    int64_t recognition_begin_us = 0;
    if (frame.operation == kOperationRecognize) {
      query_free_before_bytes = CurrentInternalHeapFreeBytes();
      recognition_begin_us = esp_timer_get_time();
    }
    int64_t invoke_us = 0;
    if (!InvokeRequest(input, &invoke_us)) {
      ESP_LOGE(kTag, "%s Invoke failed request_id=%" PRIu32,
               OperationName(frame.operation), frame.header.request_id);
      SendErrorResponse(frame.header.request_id, "invoke_failed",
                        frame.operation);
      continue;
    }
    bool response_ok = false;
    if (frame.operation == kOperationRecognize) {
      response_ok = SendRecognitionResponse(frame.header.request_id,
                                            input_crc32, invoke_us,
                                            query_free_before_bytes,
                                            recognition_begin_us);
    } else {
      response_ok = SendInferenceResponse(
          frame.header.request_id, input_crc32, invoke_us, "ENCODE");
    }
    if (!response_ok) {
      ESP_LOGE(kTag, "%s response failed request_id=%" PRIu32,
               OperationName(frame.operation), frame.header.request_id);
      SendErrorResponse(frame.header.request_id, "response_failed",
                        frame.operation);
    }
  } while (g_interpreter != nullptr);
}

}  // namespace

extern "C" void app_main(void) {
  esp_chip_info_t chip_info{};
  esp_chip_info(&chip_info);
  ESP_LOGI(kTag,
           "scope=%s runtime_mode=%s device=ESP32-S3 model=%d revision=%u "
           "cores=%u features=0x%" PRIx32 " idf=%s",
           kScope, kRuntimeMode, static_cast<int>(chip_info.model),
           chip_info.revision,
           static_cast<unsigned>(chip_info.cores), chip_info.features,
           esp_get_idf_version());
  if (chip_info.model != CHIP_ESP32S3) {
    ESP_LOGE(kTag, "device check failed: expected ESP32-S3");
    ESP_LOGE(kTag, "%s RESULT=FAIL", kScope);
    return;
  }

  const std::size_t model_size = static_cast<std::size_t>(
      _binary_encoder_int8_tflite_end - _binary_encoder_int8_tflite_start);
  const tflite::Model* model = nullptr;
  const bool model_ok = CheckModel(_binary_encoder_int8_tflite_start, model_size,
                                   &model);
  const bool runtime_ok = model_ok && InitializeRuntime(model);
  const bool smoke_ok = runtime_ok && RunSmoke(g_interpreter);
  ESP_LOGI(kTag, "%s RESULT=%s", kScope, smoke_ok ? "PASS" : "FAIL");
  if (!smoke_ok) {
    return;
  }
  LoadPersistedTemplates();
  if (!StartUsbService()) {
    ESP_LOGE(kTag, "%s USB service startup failed", kScope);
    return;
  }
  ServeUsbRequests();
}
