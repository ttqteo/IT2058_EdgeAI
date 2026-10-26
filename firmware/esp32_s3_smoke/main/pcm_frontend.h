#ifndef EDGEAI_PCM_FRONTEND_H
#define EDGEAI_PCM_FRONTEND_H

#include <cstddef>
#include <cstdint>

namespace edgeai_pcm_frontend {

constexpr std::size_t kSampleRateHz = 16000U;
constexpr std::size_t kPcmSamples = 48000U;
constexpr std::size_t kPcmBytes = kPcmSamples * sizeof(std::int16_t);
constexpr std::size_t kFrameLength = 400U;
constexpr std::size_t kHopLength = 160U;
constexpr std::size_t kFftLength = 512U;
constexpr std::size_t kFftBins = kFftLength / 2U + 1U;
constexpr std::size_t kMelBins = 40U;
constexpr std::size_t kFrameCount = 298U;
constexpr std::size_t kFeatureValues = kFrameCount * kMelBins;
constexpr std::size_t kFeatureBytes = kFeatureValues * sizeof(std::int8_t);

enum class Status {
  kOk = 0,
  kInvalidArgument,
  kInvalidPcm,
  kNonFiniteFeature,
};

/*
 * Bounded scratch for one raw-PCM request. The 96,000-byte request buffer is
 * separate and is also used as a streaming feature store: after each frame is
 * consumed, its first 160 samples (320 bytes) can hold that frame's 40 float
 * log-mel values without overwriting a later frame. The FFT and CMN sums use
 * double scratch; no [298,40] float matrix or large automatic array is
 * required on the board task stack.
 */
struct alignas(8) Workspace {
  double fft_real[kFftLength];
  double fft_imaginary[kFftLength];
  double mel_sums[kMelBins];
};

constexpr std::size_t kWorkspaceBytes = sizeof(Workspace);

static_assert(kPcmBytes == 96000U, "raw PCM wire size must remain 96000 bytes");
static_assert(kFeatureBytes == 11920U,
              "encoder input bytes must remain 11920 bytes");

const char* StatusName(Status status);

/*
 * Compute centered float32 log-mel values from little-endian PCM16.
 *
 * raw_pcm_le must be exactly kPcmBytes and remains caller-owned, but is
 * mutated after a frame is consumed. output receives exactly kFeatureValues
 * float32 values in [frame][mel] order. The arithmetic follows the locked
 * frontend: periodic Hann, 512-point unnormalized FFT, float32 HTK
 * coefficients, natural log with a 1e-6 floor, and per-mel CMN.
 */
Status ComputeLogMelPcm16(std::uint8_t* raw_pcm_le,
                          std::size_t raw_pcm_bytes,
                          Workspace* workspace,
                          float* output,
                          std::size_t output_values);

/*
 * Compute and quantize directly into the model's int8 input tensor. The
 * model-provided runtime scale and zero point are authoritative; rounding is
 * ties-to-even and the result saturates to int8.
 */
Status ComputeAndQuantizePcm16(std::uint8_t* raw_pcm_le,
                               std::size_t raw_pcm_bytes,
                               Workspace* workspace,
                               std::int8_t* output,
                               std::size_t output_bytes,
                               float input_scale,
                               int input_zero_point);

}  // namespace edgeai_pcm_frontend

#endif  // EDGEAI_PCM_FRONTEND_H
