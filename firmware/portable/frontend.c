#include "frontend.h"

#include <math.h>
#include <stddef.h>

static const double EDGEAI_PI = 3.141592653589793238462643383279502884;
static const double EDGEAI_LOG_FLOOR = 1.0e-6;

static void edgeai_fft_forward(double real[EDGEAI_FFT_LENGTH],
                               double imaginary[EDGEAI_FFT_LENGTH])
{
    size_t i;
    size_t j = 0u;

    /* In-place radix-2 bit reversal. */
    for (i = 1u; i < EDGEAI_FFT_LENGTH; ++i) {
        size_t bit = EDGEAI_FFT_LENGTH >> 1u;

        while ((j & bit) != 0u) {
            j ^= bit;
            bit >>= 1u;
        }
        j ^= bit;

        if (i < j) {
            double swap = real[i];
            real[i] = real[j];
            real[j] = swap;

            swap = imaginary[i];
            imaginary[i] = imaginary[j];
            imaginary[j] = swap;
        }
    }

    /* Forward transform, exp(-2*pi*i*k*n/N), with no normalization. */
    for (size_t length = 2u; length <= EDGEAI_FFT_LENGTH; length <<= 1u) {
        const size_t half_length = length >> 1u;
        const double length_as_double = (double)length;

        for (size_t block = 0u; block < EDGEAI_FFT_LENGTH; block += length) {
            for (j = 0u; j < half_length; ++j) {
                const double angle = -2.0 * EDGEAI_PI * (double)j /
                                     length_as_double;
                const double twiddle_real = cos(angle);
                const double twiddle_imaginary = sin(angle);
                const size_t even = block + j;
                const size_t odd = even + half_length;
                const double odd_real = real[odd] * twiddle_real -
                                        imaginary[odd] * twiddle_imaginary;
                const double odd_imaginary = real[odd] * twiddle_imaginary +
                                             imaginary[odd] * twiddle_real;
                const double even_real = real[even];
                const double even_imaginary = imaginary[even];

                real[even] = even_real + odd_real;
                imaginary[even] = even_imaginary + odd_imaginary;
                real[odd] = even_real - odd_real;
                imaginary[odd] = even_imaginary - odd_imaginary;
            }
        }
    }
}

static int edgeai_validate_inputs(const float *pcm,
                                  size_t pcm_count,
                                  const float *mel_filterbank,
                                  size_t mel_filterbank_count,
                                  const float *output,
                                  size_t output_count)
{
    if (pcm == NULL || mel_filterbank == NULL || output == NULL ||
        pcm_count != EDGEAI_PCM_SAMPLES ||
        mel_filterbank_count != EDGEAI_MEL_FILTERBANK_VALUES ||
        output_count != EDGEAI_OUTPUT_VALUES) {
        return EDGEAI_FRONTEND_INVALID_ARGUMENT;
    }

    for (size_t i = 0u; i < pcm_count; ++i) {
        if (!isfinite(pcm[i]) || pcm[i] < -1.0f || pcm[i] > 1.0f) {
            return EDGEAI_FRONTEND_INVALID_INPUT;
        }
    }

    for (size_t i = 0u; i < mel_filterbank_count; ++i) {
        if (!isfinite(mel_filterbank[i]) || mel_filterbank[i] < 0.0f ||
            mel_filterbank[i] > 1.0f) {
            return EDGEAI_FRONTEND_INVALID_INPUT;
        }
    }

    return EDGEAI_FRONTEND_OK;
}

int edgeai_frontend_log_mel(const float *pcm,
                            size_t pcm_count,
                            const float *mel_filterbank,
                            size_t mel_filterbank_count,
                            float *output,
                            size_t output_count)
{
    double window[EDGEAI_FRAME_LENGTH];
    double real[EDGEAI_FFT_LENGTH];
    double imaginary[EDGEAI_FFT_LENGTH];
    double power[EDGEAI_FFT_BINS];
    double mel_sums[EDGEAI_MEL_BINS] = {0.0};

    const int validation = edgeai_validate_inputs(pcm,
                                                  pcm_count,
                                                  mel_filterbank,
                                                  mel_filterbank_count,
                                                  output,
                                                  output_count);
    if (validation != EDGEAI_FRONTEND_OK) {
        return validation;
    }

    for (size_t n = 0u; n < EDGEAI_FRAME_LENGTH; ++n) {
        window[n] = 0.5 - 0.5 * cos(2.0 * EDGEAI_PI * (double)n /
                                   (double)EDGEAI_FRAME_LENGTH);
    }

    for (size_t frame = 0u; frame < EDGEAI_FRAME_COUNT; ++frame) {
        const size_t frame_start = frame * EDGEAI_HOP_LENGTH;

        for (size_t n = 0u; n < EDGEAI_FFT_LENGTH; ++n) {
            real[n] = 0.0;
            imaginary[n] = 0.0;
        }

        for (size_t n = 0u; n < EDGEAI_FRAME_LENGTH; ++n) {
            real[n] = (double)pcm[frame_start + n] * window[n];
        }

        edgeai_fft_forward(real, imaginary);

        for (size_t bin = 0u; bin < EDGEAI_FFT_BINS; ++bin) {
            power[bin] = real[bin] * real[bin] +
                         imaginary[bin] * imaginary[bin];
        }

        for (size_t mel = 0u; mel < EDGEAI_MEL_BINS; ++mel) {
            const size_t filter_offset = mel * EDGEAI_FFT_BINS;
            double weighted_sum = 0.0;

            for (size_t bin = 0u; bin < EDGEAI_FFT_BINS; ++bin) {
                weighted_sum += power[bin] *
                                (double)mel_filterbank[filter_offset + bin];
            }

            if (weighted_sum < EDGEAI_LOG_FLOOR) {
                weighted_sum = EDGEAI_LOG_FLOOR;
            }

            const double log_value = log(weighted_sum);

            output[frame * EDGEAI_MEL_BINS + mel] = (float)log_value;
            /* Center the values that will actually be exposed as float32. */
            mel_sums[mel] += (double)output[frame * EDGEAI_MEL_BINS + mel];
        }
    }

    for (size_t mel = 0u; mel < EDGEAI_MEL_BINS; ++mel) {
        const float mean = (float)(mel_sums[mel] /
                                   (double)EDGEAI_FRAME_COUNT);

        for (size_t frame = 0u; frame < EDGEAI_FRAME_COUNT; ++frame) {
            const size_t output_index = frame * EDGEAI_MEL_BINS + mel;
            output[output_index] -= mean;
        }
    }

    return EDGEAI_FRONTEND_OK;
}
