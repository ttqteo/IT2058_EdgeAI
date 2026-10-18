from __future__ import annotations

import wave

import numpy as np
import pytest

from edgeai.frontend import (
    FFT_LENGTH,
    FRAME_LENGTH,
    FRAME_STEP,
    NUM_FRAMES,
    NUM_FREQ_BINS,
    NUM_MELS,
    NUM_SAMPLES,
    SAMPLE_RATE,
    export_filterbank_header,
    load_wav,
    log_mel,
    mel_filterbank,
)


def test_locked_dimensions_and_filterbank_shape() -> None:
    assert SAMPLE_RATE == 16_000
    assert NUM_SAMPLES == 48_000
    assert NUM_FRAMES == (NUM_SAMPLES - FRAME_LENGTH) // FRAME_STEP + 1 == 298
    assert mel_filterbank().shape == (NUM_MELS, NUM_FREQ_BINS) == (40, 257)
    assert mel_filterbank().dtype == np.float32
    assert np.all(np.isfinite(mel_filterbank()))


def test_silence_is_analytically_zero_after_per_segment_centering() -> None:
    output = log_mel(np.zeros(NUM_SAMPLES, dtype=np.float32))
    assert output.shape == (NUM_FRAMES, NUM_MELS)
    assert output.dtype == np.float32
    np.testing.assert_allclose(output, np.zeros_like(output), atol=1e-6, rtol=0.0)


def test_log_mel_matches_independent_direct_dft() -> None:
    rng = np.random.default_rng(2058)
    pcm = rng.uniform(-0.8, 0.8, size=NUM_SAMPLES).astype(np.float32)

    frame_starts = np.arange(NUM_FRAMES, dtype=np.int64) * FRAME_STEP
    frames = np.zeros((NUM_FRAMES, FFT_LENGTH), dtype=np.float64)
    window = 0.5 - 0.5 * np.cos(
        2.0 * np.pi * np.arange(FRAME_LENGTH, dtype=np.float64) / FRAME_LENGTH
    )
    frames[:, :FRAME_LENGTH] = np.stack(
        [pcm[start : start + FRAME_LENGTH] for start in frame_starts]
    ) * window

    # Explicit forward DFT: no np.fft call, no normalization, no mirrored bins.
    sample_indices = np.arange(FFT_LENGTH, dtype=np.float64)
    bin_indices = np.arange(NUM_FREQ_BINS, dtype=np.float64)
    dft = np.exp(
        -2j * np.pi * bin_indices[:, None] * sample_indices[None, :] / FFT_LENGTH
    )
    spectrum = frames @ dft.T
    power = spectrum.real * spectrum.real + spectrum.imag * spectrum.imag
    mel_energy = power @ mel_filterbank().astype(np.float64).T
    expected = np.log(np.maximum(mel_energy, 1e-6))
    expected -= expected.mean(axis=0, keepdims=True)

    np.testing.assert_allclose(log_mel(pcm), expected, rtol=2e-5, atol=2e-5)


def test_log_mel_rejects_wrong_length_nonfinite_range_and_stereo_shape() -> None:
    with pytest.raises(ValueError, match="exactly 48000"):
        log_mel(np.zeros(NUM_SAMPLES - 1, dtype=np.float32))
    nonfinite = np.zeros(NUM_SAMPLES, dtype=np.float32)
    nonfinite[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        log_mel(nonfinite)
    out_of_range = np.zeros(NUM_SAMPLES, dtype=np.float32)
    out_of_range[0] = 1.01
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        log_mel(out_of_range)
    with pytest.raises(ValueError, match="one-dimensional"):
        log_mel(np.zeros((NUM_SAMPLES, 1), dtype=np.float32))


def _write_pcm16_wav(path, samples: np.ndarray, *, rate: int = SAMPLE_RATE) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(rate)
        wav_file.writeframes(np.asarray(samples, dtype="<i2").tobytes())


def test_wav_loader_reports_right_padding_and_tail_truncation(tmp_path) -> None:
    short_path = tmp_path / "short.wav"
    short_samples = np.array([0, 16_384, -16_384], dtype=np.int16)
    _write_pcm16_wav(short_path, short_samples)
    short_pcm, short_report = load_wav(short_path)
    assert short_pcm.shape == (NUM_SAMPLES,)
    assert short_pcm.dtype == np.float32
    np.testing.assert_allclose(short_pcm[:3], [0.0, 0.5, -0.5])
    assert short_report.adjustment == "pad_right"
    assert short_report.padded_samples == NUM_SAMPLES - 3
    assert short_report.truncated_samples == 0

    long_path = tmp_path / "long.wav"
    long_samples = np.arange(NUM_SAMPLES + 2, dtype=np.int16)
    _write_pcm16_wav(long_path, long_samples)
    long_pcm, long_report = load_wav(long_path)
    np.testing.assert_array_equal(long_pcm, long_samples[:NUM_SAMPLES].astype(np.float32) / 32768.0)
    assert long_report.adjustment == "truncate_tail"
    assert long_report.truncated_samples == 2
    assert long_report.padded_samples == 0


def test_wav_loader_rejects_non_16k(tmp_path) -> None:
    path = tmp_path / "wrong-rate.wav"
    _write_pcm16_wav(path, np.zeros(4, dtype=np.int16), rate=8_000)
    with pytest.raises(ValueError, match="sample rate"):
        load_wav(path)


def test_filterbank_header_export_is_explicitly_host_only(tmp_path) -> None:
    header_path = export_filterbank_header(tmp_path / "mel_filterbank.h")
    text = header_path.read_text(encoding="utf-8")
    assert "EXPERIMENTAL_HOST_ONLY" in text
    assert "static const float edgeai_mel_filterbank[40][257]" in text
    assert text.count("{") >= NUM_MELS + 1
