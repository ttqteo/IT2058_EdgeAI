#include "pcm_frontend.h"

#include <cmath>
#include <cstdint>
#include <cstring>

#include "pcm_frontend_tables.h"

namespace edgeai_pcm_frontend {
namespace {

constexpr double kLogFloor = 1.0e-6;
constexpr double kPcmScale = 32768.0;

std::int16_t ReadLittleEndianPcm16(const std::uint8_t* bytes) {
  const std::uint16_t bits = static_cast<std::uint16_t>(bytes[0]) |
                             (static_cast<std::uint16_t>(bytes[1]) << 8U);
  const std::int32_t signed_value =
      bits < 0x8000U ? static_cast<std::int32_t>(bits)
                     : static_cast<std::int32_t>(bits) - 0x10000;
  return static_cast<std::int16_t>(signed_value);
}

void ForwardFft(Workspace* workspace) {
  for (std::size_t i = 1U; i < kFftLength; ++i) {
    const std::size_t j = kFftBitReverse[i];
    if (i < j) {
      const double real = workspace->fft_real[i];
      workspace->fft_real[i] = workspace->fft_real[j];
      workspace->fft_real[j] = real;
      const double imaginary = workspace->fft_imaginary[i];
      workspace->fft_imaginary[i] = workspace->fft_imaginary[j];
      workspace->fft_imaginary[j] = imaginary;
    }
  }

  for (std::size_t length = 2U; length <= kFftLength; length <<= 1U) {
    const std::size_t half_length = length >> 1U;
    const std::size_t twiddle_stride = kFftLength / length;
    for (std::size_t block = 0U; block < kFftLength; block += length) {
      for (std::size_t j = 0U; j < half_length; ++j) {
        const std::size_t even = block + j;
        const std::size_t odd = even + half_length;
        const FftTwiddle& twiddle = kFftTwiddles[j * twiddle_stride];
        const double odd_real = workspace->fft_real[odd] * twiddle.real -
                                workspace->fft_imaginary[odd] *
                                    twiddle.imaginary;
        const double odd_imaginary = workspace->fft_real[odd] *
                                         twiddle.imaginary +
                                     workspace->fft_imaginary[odd] *
                                         twiddle.real;
        const double even_real = workspace->fft_real[even];
        const double even_imaginary = workspace->fft_imaginary[even];
        workspace->fft_real[even] = even_real + odd_real;
        workspace->fft_imaginary[even] = even_imaginary + odd_imaginary;
        workspace->fft_real[odd] = even_real - odd_real;
        workspace->fft_imaginary[odd] = even_imaginary - odd_imaginary;
      }
    }
  }
}

bool ValidCommonArguments(const std::uint8_t* raw_pcm_le,
                          std::size_t raw_pcm_bytes,
                          const Workspace* workspace) {
  return raw_pcm_le != nullptr && raw_pcm_bytes == kPcmBytes &&
         workspace != nullptr;
}

void StoreFrameValue(std::uint8_t* raw_pcm_le, std::size_t frame,
                     std::size_t mel, float value) {
  const std::size_t byte_offset = frame * kHopLength * sizeof(std::int16_t) +
                                  mel * sizeof(float);
  std::memcpy(raw_pcm_le + byte_offset, &value, sizeof(value));
}

float LoadFrameValue(const std::uint8_t* raw_pcm_le, std::size_t frame,
                     std::size_t mel) {
  const std::size_t byte_offset = frame * kHopLength * sizeof(std::int16_t) +
                                  mel * sizeof(float);
  float value = 0.0F;
  std::memcpy(&value, raw_pcm_le + byte_offset, sizeof(value));
  return value;
}

Status ComputeFeaturesInPlace(std::uint8_t* raw_pcm_le,
                              Workspace* workspace) {
  if (!ValidCommonArguments(raw_pcm_le, kPcmBytes, workspace)) {
    return Status::kInvalidArgument;
  }
  std::memset(workspace->mel_sums, 0, sizeof(workspace->mel_sums));

  for (std::size_t frame = 0U; frame < kFrameCount; ++frame) {
    std::memset(workspace->fft_real, 0, sizeof(workspace->fft_real));
    std::memset(workspace->fft_imaginary, 0,
                sizeof(workspace->fft_imaginary));
    const std::size_t frame_start = frame * kHopLength;
    for (std::size_t sample = 0U; sample < kFrameLength; ++sample) {
      const std::size_t sample_offset =
          (frame_start + sample) * sizeof(std::int16_t);
      const std::int16_t pcm = ReadLittleEndianPcm16(raw_pcm_le + sample_offset);
      const double value = static_cast<double>(pcm) / kPcmScale;
      workspace->fft_real[sample] = value * kPeriodicHann[sample];
    }

    ForwardFft(workspace);
    for (std::size_t mel = 0U; mel < kMelBins; ++mel) {
      double weighted_sum = 0.0;
      const std::size_t first = kMelOffsets[mel];
      const std::size_t last = kMelOffsets[mel + 1U];
      for (std::size_t entry = first; entry < last; ++entry) {
        const std::size_t bin = kMelEntries[entry].bin;
        const double real = workspace->fft_real[bin];
        const double imaginary = workspace->fft_imaginary[bin];
        const double power = real * real + imaginary * imaginary;
        weighted_sum += power * static_cast<double>(kMelEntries[entry].weight);
      }
      if (weighted_sum < kLogFloor) {
        weighted_sum = kLogFloor;
      }
      const float log_value = static_cast<float>(std::log(weighted_sum));
      if (!std::isfinite(log_value)) {
        return Status::kNonFiniteFeature;
      }
      StoreFrameValue(raw_pcm_le, frame, mel, log_value);
      workspace->mel_sums[mel] += static_cast<double>(log_value);
    }
  }

  for (std::size_t mel = 0U; mel < kMelBins; ++mel) {
    const float mean = static_cast<float>(
        workspace->mel_sums[mel] / static_cast<double>(kFrameCount));
    for (std::size_t frame = 0U; frame < kFrameCount; ++frame) {
      const float log_value = LoadFrameValue(raw_pcm_le, frame, mel);
      const float centered = log_value - mean;
      if (!std::isfinite(centered)) {
        return Status::kNonFiniteFeature;
      }
      StoreFrameValue(raw_pcm_le, frame, mel, centered);
    }
  }
  return Status::kOk;
}

}  // namespace

const char* StatusName(Status status) {
  switch (status) {
    case Status::kOk:
      return "OK";
    case Status::kInvalidArgument:
      return "INVALID_ARGUMENT";
    case Status::kInvalidPcm:
      return "INVALID_PCM";
    case Status::kNonFiniteFeature:
      return "NON_FINITE_FEATURE";
  }
  return "UNKNOWN";
}

Status ComputeLogMelPcm16(std::uint8_t* raw_pcm_le,
                          std::size_t raw_pcm_bytes,
                          Workspace* workspace,
                          float* output,
                          std::size_t output_values) {
  if (!ValidCommonArguments(raw_pcm_le, raw_pcm_bytes, workspace) ||
      output == nullptr || output_values != kFeatureValues) {
    return Status::kInvalidArgument;
  }
  const Status status = ComputeFeaturesInPlace(raw_pcm_le, workspace);
  if (status != Status::kOk) {
    return status;
  }
  for (std::size_t frame = 0U; frame < kFrameCount; ++frame) {
    for (std::size_t mel = 0U; mel < kMelBins; ++mel) {
      output[frame * kMelBins + mel] = LoadFrameValue(raw_pcm_le, frame, mel);
    }
  }
  return Status::kOk;
}

Status ComputeAndQuantizePcm16(std::uint8_t* raw_pcm_le,
                               std::size_t raw_pcm_bytes,
                               Workspace* workspace,
                               std::int8_t* output,
                               std::size_t output_bytes,
                               float input_scale,
                               int input_zero_point) {
  if (!ValidCommonArguments(raw_pcm_le, raw_pcm_bytes, workspace) ||
      output == nullptr || output_bytes != kFeatureBytes ||
      !std::isfinite(input_scale) || input_scale <= 0.0F ||
      input_zero_point < -128 || input_zero_point > 127) {
    return Status::kInvalidArgument;
  }
  const Status status = ComputeFeaturesInPlace(raw_pcm_le, workspace);
  if (status != Status::kOk) {
    return status;
  }
  for (std::size_t frame = 0U; frame < kFrameCount; ++frame) {
    for (std::size_t mel = 0U; mel < kMelBins; ++mel) {
      const float centered = LoadFrameValue(raw_pcm_le, frame, mel);
      const float scaled = centered / input_scale;
      if (!std::isfinite(scaled)) {
        return Status::kNonFiniteFeature;
      }
      const int rounded = static_cast<int>(std::nearbyintf(scaled)) +
                          input_zero_point;
      const int saturated = rounded < -128 ? -128 : rounded > 127 ? 127 : rounded;
      output[frame * kMelBins + mel] = static_cast<std::int8_t>(saturated);
    }
  }
  return Status::kOk;
}

}  // namespace edgeai_pcm_frontend
