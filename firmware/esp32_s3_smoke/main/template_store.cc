#include "template_store.h"

#include <cmath>
#include <cstring>

namespace edgeai_template {
namespace {
constexpr uint8_t kQuang[kLabelBytes] = {'Q','u','a','n','g',0,0,0};
constexpr uint8_t kTien[kLabelBytes] = {'T','i','e','n',0,0,0,0};
constexpr uint8_t kAn[kLabelBytes] = {'A','n',0,0,0,0,0,0};
uint16_t ReadLe16(const uint8_t* b) { return static_cast<uint16_t>(b[0]) | static_cast<uint16_t>(static_cast<uint16_t>(b[1]) << 8U); }
uint32_t ReadLe32(const uint8_t* b) { return static_cast<uint32_t>(b[0]) | (static_cast<uint32_t>(b[1]) << 8U) | (static_cast<uint32_t>(b[2]) << 16U) | (static_cast<uint32_t>(b[3]) << 24U); }
void WriteLe16(uint8_t* b, uint16_t v) { b[0] = static_cast<uint8_t>(v); b[1] = static_cast<uint8_t>(v >> 8U); }
void WriteLe32(uint8_t* b, uint32_t v) { b[0] = static_cast<uint8_t>(v); b[1] = static_cast<uint8_t>(v >> 8U); b[2] = static_cast<uint8_t>(v >> 16U); b[3] = static_cast<uint8_t>(v >> 24U); }
bool IsModel(const char* m) { return m != nullptr && std::strlen(m) == kModelSha256Bytes; }
bool LabelChar(char c) { return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '-' || c == '_'; }
bool ValidLabel(const char* label) {
  if (label == nullptr) return false;
  std::size_t n = 0U;
  while (n < kLabelBytes && label[n] != '\0') { if (!LabelChar(label[n])) return false; ++n; }
  if (n == 0U || n >= kLabelBytes) return false;
  for (std::size_t i = n + 1U; i < kLabelBytes; ++i) if (label[i] != '\0') return false;
  return true;
}
void BuiltIns(TemplateLabels* labels) {
  std::memset(labels, 0, sizeof(*labels));
  std::memcpy(labels->values[0], kQuang, kLabelBytes);
  std::memcpy(labels->values[1], kTien, kLabelBytes);
  std::memcpy(labels->values[2], kAn, kLabelBytes);
}
bool ValidVectors(const TemplateVectors& v, std::size_t count) {
  if (count < kLegacyTemplateSlots || count > kTemplateSlots) return false;
  for (std::size_t s = 0; s < count; ++s) {
    double norm = 0.0;
    for (std::size_t i = 0; i < kEmbeddingDim; ++i) { if (!std::isfinite(v.values[s][i])) return false; norm += static_cast<double>(v.values[s][i]) * v.values[s][i]; }
    if (!std::isfinite(norm) || std::fabs(norm - 1.0) > 1.0e-3) return false;
  }
  return true;
}
bool ValidLabels(const TemplateLabels& labels, std::size_t count, bool builtins) {
  if (count < kLegacyTemplateSlots || count > kTemplateSlots) return false;
  const uint8_t* required[] = {kQuang, kTien, kAn};
  const std::size_t prefix = builtins ? (count < kV2TemplateSlots ? count : kV2TemplateSlots) : 0U;
  for (std::size_t s = 0; s < prefix; ++s) if (std::memcmp(labels.values[s], required[s], kLabelBytes) != 0) return false;
  for (std::size_t s = 0; s < count; ++s) {
    if (!ValidLabel(labels.values[s])) return false;
    for (std::size_t p = 0; p < s; ++p) if (std::memcmp(labels.values[s], labels.values[p], kLabelBytes) == 0) return false;
  }
  return true;
}
bool Header(const uint8_t* r, const char* magic, uint16_t version, uint16_t bytes, uint8_t count, const char* model) {
  return std::memcmp(r + kRecordMagicOffset, magic, 8U) == 0 && ReadLe16(r + kRecordVersionOffset) == version && ReadLe16(r + kRecordLengthOffset) == bytes && std::memcmp(r + kRecordModelShaOffset, model, kModelSha256Bytes) == 0 && r[kRecordSlotCountOffset] == count && r[kRecordReservedOffset] == 0U && ReadLe16(r + kRecordEmbeddingDimOffset) == kRecordEmbeddingDim;
}
bool Build(const TemplateVectors& v, const TemplateLabels& labels, std::size_t count, const char* magic, uint16_t version, std::size_t bytes, std::size_t vectors_at, std::size_t crc_at, const char* model, uint8_t* record, std::size_t capacity, uint32_t* crc) {
  if (record == nullptr || capacity < bytes || !IsModel(model) || !ValidVectors(v, count) || !ValidLabels(labels, count, true)) return false;
  std::memset(record, 0, bytes); std::memcpy(record, magic, 8U); WriteLe16(record + kRecordVersionOffset, version); WriteLe16(record + kRecordLengthOffset, static_cast<uint16_t>(bytes)); std::memcpy(record + kRecordModelShaOffset, model, kModelSha256Bytes); record[kRecordSlotCountOffset] = static_cast<uint8_t>(count); WriteLe16(record + kRecordEmbeddingDimOffset, kRecordEmbeddingDim);
  const std::size_t labels_at = version == kV3RecordVersion ? kV3RecordLabelsOffset : kRecordLabelsOffset;
  for (std::size_t s = 0; s < count; ++s) { record[kRecordLabelLengthOffset + s] = static_cast<uint8_t>(std::strlen(labels.values[s])); std::memcpy(record + labels_at + s * kLabelBytes, labels.values[s], kLabelBytes); }
  std::memcpy(record + vectors_at, v.values, count * kEmbeddingDim * sizeof(float)); const uint32_t value = Crc32Ieee(record, crc_at); WriteLe32(record + crc_at, value); if (crc != nullptr) *crc = value; return true;
}
}  // namespace

uint32_t Crc32Ieee(const uint8_t* bytes, std::size_t length) { if (bytes == nullptr && length != 0U) return 0U; uint32_t crc = 0xFFFFFFFFU; for (std::size_t i = 0; i < length; ++i) { crc ^= bytes[i]; for (int bit = 0; bit < 8; ++bit) { const uint32_t mask = 0U - (crc & 1U); crc = (crc >> 1U) ^ (0xEDB88320U & mask); } } return ~crc; }
bool IsSafeLabel(const char* label) { return ValidLabel(label); }
bool CopySafeLabel(const char* label, char output[kLabelBytes]) { if (output == nullptr || !ValidLabel(label)) return false; std::memcpy(output, label, kLabelBytes); return true; }
bool LabelsEqual(const char* left, const char* right) { return left != nullptr && right != nullptr && std::memcmp(left, right, kLabelBytes) == 0; }
bool DequantizeAndNormalize(const int8_t* q, std::size_t length, float scale, int zero, float* out) { if (q == nullptr || out == nullptr || length != kEmbeddingDim || !std::isfinite(scale) || scale <= 0.0F) return false; double norm = 0.0; for (std::size_t i = 0; i < length; ++i) { const float value = (static_cast<float>(q[i]) - static_cast<float>(zero)) * scale; if (!std::isfinite(value)) return false; out[i] = value; norm += static_cast<double>(value) * value; } if (!std::isfinite(norm) || norm <= 0.0) return false; norm = std::sqrt(norm); for (std::size_t i = 0; i < length; ++i) { out[i] = static_cast<float>(out[i] / norm); if (!std::isfinite(out[i])) return false; } return true; }
bool CosineSimilarity(const float* left, const float* right, std::size_t length, float* cosine) { if (left == nullptr || right == nullptr || cosine == nullptr || length == 0U) return false; double dot = 0.0, ln = 0.0, rn = 0.0; for (std::size_t i = 0; i < length; ++i) { if (!std::isfinite(left[i]) || !std::isfinite(right[i])) return false; dot += static_cast<double>(left[i]) * right[i]; ln += static_cast<double>(left[i]) * left[i]; rn += static_cast<double>(right[i]) * right[i]; } if (!std::isfinite(dot) || ln <= 0.0 || rn <= 0.0) return false; double value = dot / (std::sqrt(ln) * std::sqrt(rn)); if (!std::isfinite(value)) return false; if (value > 1.0) value = 1.0; if (value < -1.0) value = -1.0; *cosine = static_cast<float>(value); return true; }
bool BuildLegacyRecord(const TemplateVectors& v, const char* model, uint8_t* r, std::size_t cap, uint32_t* crc) { TemplateLabels labels{}; BuiltIns(&labels); return Build(v, labels, kLegacyTemplateSlots, kLegacyRecordMagic, kLegacyRecordVersion, kLegacyRecordBytes, kLegacyRecordTemplatesOffset, kLegacyRecordCrcOffset, model, r, cap, crc); }
bool BuildRecord(const TemplateVectors& v, const char* model, uint8_t* r, std::size_t cap, uint32_t* crc) { TemplateLabels labels{}; BuiltIns(&labels); return Build(v, labels, kV2TemplateSlots, kRecordMagic, kRecordVersion, kRecordBytes, kRecordTemplatesOffset, kRecordCrcOffset, model, r, cap, crc); }
bool BuildV3Record(const TemplateVectors& vectors, const TemplateLabels& labels,
                   uint8_t count, const char* model, uint8_t* r,
                   std::size_t cap, uint32_t* crc) {
  return count >= kV2TemplateSlots && count <= kTemplateSlots &&
      Build(vectors, labels, count, kV3RecordMagic, kV3RecordVersion,
            kV3RecordBytes, kV3RecordTemplatesOffset, kV3RecordCrcOffset,
            model, r, cap, crc);
}
bool BuildV3Record(const TemplateSet& set, const char* model, uint8_t* r, std::size_t cap, uint32_t* crc) {
  return BuildV3Record(set.vectors, set.labels, set.count, model, r, cap, crc);
}
bool ParseRecord(const uint8_t* r, std::size_t bytes, const char* model, TemplateSet* out, uint32_t* crc) {
  if (r == nullptr || out == nullptr || !IsModel(model)) return false;
  std::size_t count = 0U, vectors_at = 0U, crc_at = 0U;
  TemplateLabels labels{};
  if (bytes == kLegacyRecordBytes) { if (!Header(r, kLegacyRecordMagic, kLegacyRecordVersion, kLegacyRecordBytes, kLegacyRecordSlotCount, model) || ReadLe16(r + kRecordReserved2Offset) != 0U) return false; BuiltIns(&labels); count = kLegacyTemplateSlots; vectors_at = kLegacyRecordTemplatesOffset; crc_at = kLegacyRecordCrcOffset; for (std::size_t s = 0; s < count; ++s) if (r[kRecordLabelLengthOffset + s] != std::strlen(labels.values[s]) || std::memcmp(r + kRecordLabelsOffset + s * kLabelBytes, labels.values[s], kLabelBytes) != 0) return false; }
  else if (bytes == kRecordBytes) { if (!Header(r, kRecordMagic, kRecordVersion, kRecordBytes, kRecordSlotCount, model) || r[kRecordReserved2Offset + 1U] != 0U) return false; BuiltIns(&labels); count = kV2TemplateSlots; vectors_at = kRecordTemplatesOffset; crc_at = kRecordCrcOffset; for (std::size_t s = 0; s < count; ++s) if (r[kRecordLabelLengthOffset + s] != std::strlen(labels.values[s]) || std::memcmp(r + kRecordLabelsOffset + s * kLabelBytes, labels.values[s], kLabelBytes) != 0) return false; }
  else if (bytes == kV3RecordBytes) {
    const uint8_t c = r[kRecordSlotCountOffset];
    if (c < kV2TemplateSlots || c > kTemplateSlots ||
        !Header(r, kV3RecordMagic, kV3RecordVersion, kV3RecordBytes, c, model) ||
        r[85] != 0U || r[86] != 0U || r[87] != 0U) return false;
    count = c;
    vectors_at = kV3RecordTemplatesOffset;
    crc_at = kV3RecordCrcOffset;
    for (std::size_t s = 0; s < count; ++s) {
      const uint8_t n = r[kRecordLabelLengthOffset + s];
      if (n == 0U || n >= kLabelBytes ||
          r[kV3RecordLabelsOffset + s * kLabelBytes + n] != 0U) return false;
      std::memcpy(labels.values[s], r + kV3RecordLabelsOffset + s * kLabelBytes,
                  kLabelBytes);
      if (std::strlen(labels.values[s]) != n) return false;
    }
  }
  else return false;
  const uint32_t stored = ReadLe32(r + crc_at); if (stored != Crc32Ieee(r, crc_at) || !ValidLabels(labels, count, true)) return false; TemplateSet candidate{}; std::memcpy(candidate.vectors.values, r + vectors_at, count * kEmbeddingDim * sizeof(float)); candidate.labels = labels; candidate.count = static_cast<uint8_t>(count); if (!ValidVectors(candidate.vectors, count)) return false; *out = candidate; if (crc != nullptr) *crc = stored; return true;
}
}  // namespace edgeai_template
