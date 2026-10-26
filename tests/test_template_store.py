from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "firmware/esp32_s3_smoke/main"


@pytest.mark.skipif(shutil.which("g++") is None, reason="host C++ compiler unavailable")
def test_template_records_preserve_legacy_vectors_and_reject_tampering(tmp_path: Path) -> None:
    program = tmp_path / "template_store_check.cc"
    executable = tmp_path / "template_store_check"
    program.write_text(r'''
#include "template_store.h"
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>

int main() {
  constexpr const char kModel[] = "aa42a3140f375f3a2c93648893132b880dd3e9f4be4ca8f82cde09fc6c440727";
  edgeai_template::TemplateVectors vectors{};
  vectors.values[0][0] = 1.0f;
  vectors.values[1][1] = 1.0f;
  uint8_t legacy[edgeai_template::kLegacyRecordBytes] = {};
  uint32_t legacy_crc = 0;
  assert(edgeai_template::BuildLegacyRecord(vectors, kModel, legacy, sizeof(legacy), &legacy_crc));
  edgeai_template::TemplateSet loaded{};
  assert(edgeai_template::ParseRecord(legacy, sizeof(legacy), kModel, &loaded, nullptr));
  assert(loaded.count == 2 && loaded.vectors.values[0][0] == 1.0f && loaded.vectors.values[1][1] == 1.0f);
  const float quang_before = loaded.vectors.values[0][0];
  const float tien_before = loaded.vectors.values[1][1];
  loaded.vectors.values[2][2] = 1.0f;
  uint8_t v2[edgeai_template::kRecordBytes] = {};
  uint32_t v2_crc = 0;
  assert(edgeai_template::BuildRecord(loaded.vectors, kModel, v2, sizeof(v2), &v2_crc));
  edgeai_template::TemplateSet roundtrip{};
  assert(edgeai_template::ParseRecord(v2, sizeof(v2), kModel, &roundtrip, nullptr));
  assert(roundtrip.count == 3 && roundtrip.vectors.values[0][0] == quang_before &&
         roundtrip.vectors.values[1][1] == tien_before && roundtrip.vectors.values[2][2] == 1.0f);
  assert(std::memcmp(legacy + edgeai_template::kLegacyRecordTemplatesOffset,
                     v2 + edgeai_template::kRecordTemplatesOffset,
                     2 * edgeai_template::kEmbeddingDim * sizeof(float)) == 0);
  edgeai_template::TemplateSet appended{};
  appended = roundtrip;
  std::memcpy(appended.labels.values[3], "LS116", 6);
  appended.vectors.values[3][3] = 1.0f;
  appended.count = 4;
  uint8_t v3[edgeai_template::kV3RecordBytes] = {};
  assert(edgeai_template::BuildV3Record(appended, kModel, v3, sizeof(v3), nullptr));
  edgeai_template::TemplateSet v3_roundtrip{};
  assert(edgeai_template::ParseRecord(v3, sizeof(v3), kModel, &v3_roundtrip, nullptr));
  assert(v3_roundtrip.count == 4 && std::strcmp(v3_roundtrip.labels.values[3], "LS116") == 0);
  assert(std::memcmp(v2 + edgeai_template::kRecordTemplatesOffset,
                     v3 + edgeai_template::kV3RecordTemplatesOffset,
                     3 * edgeai_template::kEmbeddingDim * sizeof(float)) == 0);
  std::memcpy(appended.labels.values[4], "LS700", 6);
  appended.vectors.values[4][4] = 1.0f;
  appended.count = 5;
  assert(edgeai_template::BuildV3Record(appended.vectors, appended.labels,
      appended.count, kModel, v3, sizeof(v3), nullptr));
  assert(edgeai_template::ParseRecord(v3, sizeof(v3), kModel, &v3_roundtrip, nullptr));
  assert(v3_roundtrip.count == 5 && std::strcmp(v3_roundtrip.labels.values[4], "LS700") == 0);
  assert(std::memcmp(appended.vectors.values, v3_roundtrip.vectors.values,
                     sizeof(appended.vectors.values)) == 0);
  // Even a CRC-consistent record must reject contradictory label lengths.
  v3[edgeai_template::kRecordLabelLengthOffset + 4] = 6;
  const uint32_t bad_crc = edgeai_template::Crc32Ieee(v3, edgeai_template::kV3RecordCrcOffset);
  for (unsigned byte = 0; byte < 4; ++byte)
    v3[edgeai_template::kV3RecordCrcOffset + byte] = static_cast<uint8_t>(bad_crc >> (byte * 8));
  assert(!edgeai_template::ParseRecord(v3, sizeof(v3), kModel, &v3_roundtrip, nullptr));
  assert(edgeai_template::BuildV3Record(appended, kModel, v3, sizeof(v3), nullptr));
  v3[edgeai_template::kRecordLabelsOffset + 3 * edgeai_template::kLabelBytes] ^= 1;
  assert(!edgeai_template::ParseRecord(v3, sizeof(v3), kModel, &v3_roundtrip, nullptr));
  v2[edgeai_template::kRecordTemplatesOffset] ^= 1;
  assert(!edgeai_template::ParseRecord(v2, sizeof(v2), kModel, &roundtrip, nullptr));
  assert(!edgeai_template::ParseRecord(legacy, sizeof(legacy) - 1, kModel, &roundtrip, nullptr));
  assert(!edgeai_template::ParseRecord(legacy, sizeof(legacy), "0", &roundtrip, nullptr));
  edgeai_template::TemplateVectors zero{};
  assert(!edgeai_template::BuildRecord(zero, kModel, v2, sizeof(v2), nullptr));
  int8_t quantized[edgeai_template::kEmbeddingDim] = {};
  float normalized[edgeai_template::kEmbeddingDim] = {};
  assert(!edgeai_template::DequantizeAndNormalize(quantized, sizeof(quantized), 1.0f, 0, normalized));
  quantized[0] = 1;
  assert(edgeai_template::DequantizeAndNormalize(quantized, sizeof(quantized), 1.0f, 0, normalized));
  float cosine = 0.0f;
  assert(edgeai_template::CosineSimilarity(normalized, loaded.vectors.values[0], edgeai_template::kEmbeddingDim, &cosine));
  return 0;
}
''', encoding="utf-8")
    gxx = shutil.which("g++")
    assert gxx is not None
    subprocess.run([gxx, "-std=c++17", "-Wall", "-Wextra", "-Werror", f"-I{HELPER}",
                    str(HELPER / "template_store.cc"), str(program), "-o", str(executable)],
                   check=True, capture_output=True, text=True)
    subprocess.run([str(executable)], check=True, capture_output=True, text=True)


def test_firmware_append_is_explicit_atomic_and_fail_closed() -> None:
    source = (HELPER / "main.cc").read_text(encoding="utf-8")
    assert 'constexpr uint8_t kOperationAddAn = 0x13U' in source
    assert '"an_already_enrolled_no_overwrite"' in source
    assert "nvs_open(kNvsNamespace, NVS_READONLY" in source
    assert "nvs_set_blob(write_handle, kNvsKey, record, record_bytes)" in source
    assert "g_templates = candidate;" in source
    assert source.index("nvs_get_blob(read_handle") < source.index("g_templates = candidate;")
    assert "nvs_flash_erase" not in source
    assert "nvs_erase_all" not in source
    assert "kLegacyRecordBytes" in source and "kRecordBytes" in source
    assert "kOperationAppendPcm = 0x21U" in source
    assert "kAppendPcmPayloadBytes" in source
    assert "template_capacity_full" in source
    assert "duplicate_label_no_overwrite" in source
