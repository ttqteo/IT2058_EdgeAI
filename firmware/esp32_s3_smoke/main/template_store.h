#pragma once

#include <cstddef>
#include <cstdint>

namespace edgeai_template {

// v1 and v2 are immutable on-disk contracts. v3 is append-only and carries
// up to five safe labels with the vector bits copied verbatim from v1/v2.
constexpr std::size_t kLegacyTemplateSlots = 2U;
constexpr std::size_t kV2TemplateSlots = 3U;
constexpr std::size_t kTemplateSlots = 5U;
constexpr std::size_t kEmbeddingDim = 192U;
constexpr std::size_t kModelSha256Bytes = 64U;
constexpr std::size_t kLabelBytes = 8U;
constexpr std::size_t kLegacyRecordBytes = 1640U;
constexpr std::size_t kRecordBytes = 2416U;  // v2, retained for callers.
constexpr std::size_t kV3RecordBytes = 3972U;
constexpr std::size_t kMaxRecordBytes = kV3RecordBytes;

constexpr std::size_t kRecordMagicOffset = 0U;
constexpr std::size_t kRecordVersionOffset = 8U;
constexpr std::size_t kRecordLengthOffset = 10U;
constexpr std::size_t kRecordModelShaOffset = 12U;
constexpr std::size_t kRecordSlotCountOffset = 76U;
constexpr std::size_t kRecordReservedOffset = 77U;
constexpr std::size_t kRecordEmbeddingDimOffset = 78U;
constexpr std::size_t kRecordLabelLengthOffset = 80U;
constexpr std::size_t kRecordReserved2Offset = 82U;
constexpr std::size_t kRecordLabelsOffset = 84U;
constexpr std::size_t kLegacyRecordTemplatesOffset = 100U;
constexpr std::size_t kLegacyRecordCrcOffset = 1636U;
constexpr std::size_t kRecordTemplatesOffset = 108U;
constexpr std::size_t kRecordCrcOffset = 2412U;
constexpr std::size_t kV3RecordLabelsOffset = 88U;
constexpr std::size_t kV3RecordTemplatesOffset = 128U;
constexpr std::size_t kV3RecordCrcOffset = 3968U;

constexpr uint16_t kLegacyRecordVersion = 1U;
constexpr uint16_t kRecordVersion = 2U;
constexpr uint16_t kV3RecordVersion = 3U;
constexpr uint8_t kLegacyRecordSlotCount = 2U;
constexpr uint8_t kRecordSlotCount = 3U;
constexpr uint16_t kRecordEmbeddingDim = 192U;
constexpr char kLegacyRecordMagic[] = "EAI-TPL1";
constexpr char kRecordMagic[] = "EAI-TPL2";
constexpr char kV3RecordMagic[] = "EAI-TPL3";
constexpr char kQuangLabel[] = "Quang";
constexpr char kTienLabel[] = "Tien";
constexpr char kAnLabel[] = "An";

struct TemplateVectors { float values[kTemplateSlots][kEmbeddingDim]; };
struct TemplateLabels { char values[kTemplateSlots][kLabelBytes]; };
struct TemplateSet {
  TemplateVectors vectors;
  TemplateLabels labels;
  uint8_t count;
};

uint32_t Crc32Ieee(const uint8_t* bytes, std::size_t length);
bool IsSafeLabel(const char* label);
bool CopySafeLabel(const char* label, char output[kLabelBytes]);
bool LabelsEqual(const char* left, const char* right);
bool DequantizeAndNormalize(const int8_t* quantized, std::size_t length,
                            float output_scale, int output_zero_point,
                            float* normalized);
bool CosineSimilarity(const float* left, const float* right,
                     std::size_t length, float* cosine);

bool BuildLegacyRecord(const TemplateVectors& vectors, const char* model_sha256,
                       uint8_t* record, std::size_t record_capacity,
                       uint32_t* record_crc32);
// v2 remains available for decoding/testing old three-template boards.
bool BuildRecord(const TemplateVectors& vectors, const char* model_sha256,
                 uint8_t* record, std::size_t record_capacity,
                 uint32_t* record_crc32);
bool BuildV3Record(const TemplateSet& templates, const char* model_sha256,
                   uint8_t* record, std::size_t record_capacity,
                   uint32_t* record_crc32);
bool BuildV3Record(const TemplateVectors& vectors, const TemplateLabels& labels,
                   uint8_t count, const char* model_sha256, uint8_t* record,
                   std::size_t record_capacity, uint32_t* record_crc32);

// Parses v1 (2), v2 (3), and v3 (3..5) without migration writes.
bool ParseRecord(const uint8_t* record, std::size_t record_length,
                 const char* expected_model_sha256, TemplateSet* out_templates,
                 uint32_t* record_crc32);

}  // namespace edgeai_template
