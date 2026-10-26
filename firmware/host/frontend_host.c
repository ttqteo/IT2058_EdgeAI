#include "frontend.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

_Static_assert(sizeof(float) == sizeof(uint32_t),
               "the host must use 32-bit float values");
_Static_assert(FLT_RADIX == 2 && FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128,
               "the host must use IEEE-754 binary32 float values");

static float decode_little_endian_float32(const unsigned char bytes[4])
{
    const uint32_t bits = (uint32_t)bytes[0] |
                          ((uint32_t)bytes[1] << 8u) |
                          ((uint32_t)bytes[2] << 16u) |
                          ((uint32_t)bytes[3] << 24u);
    float value;

    memcpy(&value, &bits, sizeof value);
    return value;
}

static int read_float32_file(const char *path,
                             float *values,
                             size_t value_count,
                             int unit_range)
{
    FILE *file;

    if (path == NULL || values == NULL) {
        return 0;
    }

    file = fopen(path, "rb");
    if (file == NULL) {
        return 0;
    }

    for (size_t i = 0u; i < value_count; ++i) {
        unsigned char bytes[4];
        float value;

        if (fread(bytes, sizeof bytes, 1u, file) != 1u) {
            (void)fclose(file);
            return 0;
        }

        value = decode_little_endian_float32(bytes);
        if (!isfinite(value) ||
            (unit_range && (value < -1.0f || value > 1.0f)) ||
            (!unit_range && (value < 0.0f || value > 1.0f))) {
            (void)fclose(file);
            return 0;
        }
        values[i] = value;
    }

    if (fgetc(file) != EOF || ferror(file)) {
        (void)fclose(file);
        return 0;
    }

    return fclose(file) == 0;
}

static int write_float32_file(const char *path,
                              const float *values,
                              size_t value_count)
{
    FILE *file;

    if (path == NULL || values == NULL) {
        return 0;
    }

    file = fopen(path, "wb");
    if (file == NULL) {
        return 0;
    }

    for (size_t i = 0u; i < value_count; ++i) {
        uint32_t bits;
        unsigned char bytes[4];

        if (!isfinite(values[i])) {
            (void)fclose(file);
            return 0;
        }

        memcpy(&bits, &values[i], sizeof bits);
        bytes[0] = (unsigned char)(bits & 0xffu);
        bytes[1] = (unsigned char)((bits >> 8u) & 0xffu);
        bytes[2] = (unsigned char)((bits >> 16u) & 0xffu);
        bytes[3] = (unsigned char)((bits >> 24u) & 0xffu);

        if (fwrite(bytes, sizeof bytes, 1u, file) != 1u) {
            (void)fclose(file);
            return 0;
        }
    }

    return fclose(file) == 0;
}

int main(int argc, char **argv)
{
    /* Host-only file buffers: 70,200 float32 values, about 281 kB static. */
    static float pcm[EDGEAI_PCM_SAMPLES];
    static float mel_filterbank[EDGEAI_MEL_FILTERBANK_VALUES];
    static float output[EDGEAI_OUTPUT_VALUES];
    int status;

    if (argc != 4) {
        (void)fprintf(stderr,
                      "usage: %s PCM_F32 MEL_F32 OUTPUT_F32\n",
                      argc > 0 && argv[0] != NULL ? argv[0] : "frontend_host");
        return 2;
    }

    if (!read_float32_file(argv[1], pcm, EDGEAI_PCM_SAMPLES, 1) ||
        !read_float32_file(argv[2],
                           mel_filterbank,
                           EDGEAI_MEL_FILTERBANK_VALUES,
                           0)) {
        (void)fprintf(stderr, "invalid PCM_F32 or MEL_F32 input\n");
        return 1;
    }

    status = edgeai_frontend_log_mel(pcm,
                                     EDGEAI_PCM_SAMPLES,
                                     mel_filterbank,
                                     EDGEAI_MEL_FILTERBANK_VALUES,
                                     output,
                                     EDGEAI_OUTPUT_VALUES);
    if (status != EDGEAI_FRONTEND_OK) {
        (void)fprintf(stderr, "frontend rejected input (status %d)\n", status);
        return 1;
    }

    if (!write_float32_file(argv[3], output, EDGEAI_OUTPUT_VALUES)) {
        (void)fprintf(stderr, "could not write OUTPUT_F32\n");
        return 1;
    }

    return 0;
}
