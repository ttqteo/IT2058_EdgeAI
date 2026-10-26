#ifndef EDGEAI_PORTABLE_FRONTEND_H
#define EDGEAI_PORTABLE_FRONTEND_H

#include <stddef.h>

/* EXPERIMENTAL_HOST_ONLY: these dimensions are the v0.1 host prototype. */
#define EDGEAI_SAMPLE_RATE_HZ 16000u
#define EDGEAI_PCM_SAMPLES 48000u
#define EDGEAI_FRAME_LENGTH 400u
#define EDGEAI_HOP_LENGTH 160u
#define EDGEAI_FFT_LENGTH 512u
#define EDGEAI_FFT_BINS 257u
#define EDGEAI_MEL_BINS 40u
#define EDGEAI_FRAME_COUNT 298u
#define EDGEAI_MEL_FILTERBANK_VALUES (EDGEAI_MEL_BINS * EDGEAI_FFT_BINS)
#define EDGEAI_OUTPUT_VALUES (EDGEAI_FRAME_COUNT * EDGEAI_MEL_BINS)

typedef enum edgeai_frontend_status {
    EDGEAI_FRONTEND_OK = 0,
    EDGEAI_FRONTEND_INVALID_ARGUMENT = -1,
    EDGEAI_FRONTEND_INVALID_INPUT = -2
} edgeai_frontend_status;

/*
 * Compute time-mean-normalized log-mel features for the host prototype.
 *
 * pcm is exactly EDGEAI_PCM_SAMPLES finite float32 samples in [-1, 1].
 * mel_filterbank is exactly EDGEAI_MEL_FILTERBANK_VALUES finite float32
 * weights in [0, 1], laid out as [mel_bin][fft_bin].
 * output is a caller-owned buffer of exactly EDGEAI_OUTPUT_VALUES float32
 * elements laid out as [frame][mel_bin].
 *
 * The implementation performs the FFT in double precision and allocates no
 * dynamic memory. Its automatic working arrays require approximately 14 KiB
 * (window, FFT real/imaginary, power, and mel accumulators). This is a
 * portable host arithmetic baseline, not an ESP-DSP/MCU memory claim.
 */
int edgeai_frontend_log_mel(const float *pcm,
                            size_t pcm_count,
                            const float *mel_filterbank,
                            size_t mel_filterbank_count,
                            float *output,
                            size_t output_count);

#endif /* EDGEAI_PORTABLE_FRONTEND_H */
