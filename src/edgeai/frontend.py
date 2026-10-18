"""NumPy reference frontend for the experimental host-only v0.1 contract.

This module intentionally implements only the bounded local reference.  It
does not claim parity with ESP-DSP, TFLite Micro, a USB transport, or a final
board runtime.  The constants and arithmetic are kept stable for an
independent C host peer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
import wave

import numpy as np

SAMPLE_RATE = 16_000
NUM_SAMPLES = 48_000
FRAME_LENGTH = 400
FRAME_STEP = 160
FFT_LENGTH = 512
NUM_FREQ_BINS = FFT_LENGTH // 2 + 1
NUM_FRAMES = (NUM_SAMPLES - FRAME_LENGTH) // FRAME_STEP + 1
NUM_MELS = 40
MIN_FREQ_HZ = 0.0
MAX_FREQ_HZ = SAMPLE_RATE / 2.0
LOG_FLOOR = 1e-6

EXPERIMENTAL_HOST_ONLY = "EXPERIMENTAL_HOST_ONLY"
CONTRACT_UNRESOLVED = "CONTRACT_UNRESOLVED"
CONTRACT_DISCOVERY = "BOOTSTRAP-CONTRACT-01"

_FLOAT32 = np.dtype(np.float32)


@dataclass(frozen=True)
class WavLoadReport:
    """Report describing a strict WAV decode and fixed-length adjustment."""

    path: str
    sample_rate: int
    channels: int
    sample_width_bytes: int
    input_samples: int
    output_samples: int
    adjustment: str
    padded_samples: int
    truncated_samples: int

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-friendly representation of the load report."""

        return asdict(self)


def _validate_pcm(pcm: object) -> np.ndarray:
    """Validate and return PCM as finite float64 values for arithmetic."""

    raw = np.asarray(pcm)
    if raw.ndim != 1:
        raise ValueError("PCM must be a one-dimensional mono sample array")
    if np.iscomplexobj(raw):
        raise ValueError("PCM must contain real-valued samples")
    try:
        values = raw.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("PCM must contain numeric samples") from exc
    if values.size != NUM_SAMPLES:
        raise ValueError(
            f"PCM must contain exactly {NUM_SAMPLES} samples; got {values.size}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("PCM must contain only finite samples")
    if np.any(values < -1.0) or np.any(values > 1.0):
        raise ValueError("PCM samples must be in the inclusive range [-1, 1]")
    return values


def _hz_to_mel(frequency_hz: np.ndarray | float) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(frequency_hz) / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (np.power(10.0, mel / 2595.0) - 1.0)


def _build_mel_filterbank() -> np.ndarray:
    """Build the unnormalized HTK triangular filterbank in float32."""

    mel_endpoints = np.linspace(
        _hz_to_mel(MIN_FREQ_HZ),
        _hz_to_mel(MAX_FREQ_HZ),
        NUM_MELS + 2,
        dtype=np.float64,
    )
    frequency_endpoints = _mel_to_hz(mel_endpoints)
    fft_frequencies = (
        np.arange(NUM_FREQ_BINS, dtype=np.float64) * SAMPLE_RATE / FFT_LENGTH
    )

    lower = frequency_endpoints[:-2, None]
    center = frequency_endpoints[1:-1, None]
    upper = frequency_endpoints[2:, None]
    frequencies = fft_frequencies[None, :]

    rising = (frequencies - lower) / (center - lower)
    falling = (upper - frequencies) / (upper - center)
    filters = np.maximum(0.0, np.minimum(rising, falling))
    return filters.astype(_FLOAT32)


_MEL_FILTERBANK = _build_mel_filterbank()
_PERIODIC_HANN = (
    0.5
    - 0.5
    * np.cos(
        2.0
        * np.pi
        * np.arange(FRAME_LENGTH, dtype=np.float64)
        / FRAME_LENGTH
    )
)


def mel_filterbank() -> np.ndarray:
    """Return a copy of the locked ``float32[40, 257]`` filterbank."""

    return _MEL_FILTERBANK.copy()


def log_mel(pcm: object) -> np.ndarray:
    """Compute the locked ``float32[298, 40]`` centered log-mel feature.

    ``pcm`` must be one-dimensional, contain exactly 48,000 finite samples,
    and have every value in ``[-1, 1]``.  Frames are left aligned, use a
    periodic Hann window, and are zero padded to 512 samples before the
    unnormalized forward real FFT.
    """

    values = _validate_pcm(pcm)
    frame_view = np.lib.stride_tricks.sliding_window_view(
        values, FRAME_LENGTH
    )[::FRAME_STEP]
    if frame_view.shape != (NUM_FRAMES, FRAME_LENGTH):
        raise AssertionError(
            f"unexpected frame shape {frame_view.shape}; expected "
            f"({NUM_FRAMES}, {FRAME_LENGTH})"
        )

    windowed = frame_view * _PERIODIC_HANN[None, :]
    frames = np.zeros((NUM_FRAMES, FFT_LENGTH), dtype=np.float64)
    frames[:, :FRAME_LENGTH] = windowed

    spectrum = np.fft.rfft(frames, n=FFT_LENGTH, axis=1)
    power = spectrum.real * spectrum.real + spectrum.imag * spectrum.imag
    energies = power @ _MEL_FILTERBANK.astype(np.float64).T
    log_energies = np.log(np.maximum(energies, LOG_FLOOR))
    centered = log_energies - np.mean(log_energies, axis=0, keepdims=True)
    return centered.astype(_FLOAT32)


def load_wav(path: str | Path) -> tuple[np.ndarray, WavLoadReport]:
    """Load a PCM16 mono 16 kHz WAV and pad/truncate to the host length.

    Format mismatches are rejected.  A shorter recording is zero padded on
    the right and a longer recording is truncated after the first 48,000
    samples.  The returned report makes that adjustment explicit.
    """

    wav_path = Path(path)
    try:
        wav_file = wave.open(str(wav_path), "rb")
    except (OSError, wave.Error) as exc:
        raise ValueError(f"could not open WAV file {wav_path}: {exc}") from exc

    with wav_file:
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        compression = wav_file.getcomptype()
        frame_count = wav_file.getnframes()

        if sample_rate != SAMPLE_RATE:
            raise ValueError(
                f"WAV sample rate must be {SAMPLE_RATE} Hz; got {sample_rate}"
            )
        if channels != 1:
            raise ValueError(f"WAV must be mono; got {channels} channels")
        if sample_width != 2 or compression != "NONE":
            raise ValueError(
                "WAV must be uncompressed PCM16; got "
                f"sample_width={sample_width}, compression={compression!r}"
            )

        raw = wav_file.readframes(frame_count)

    expected_bytes = frame_count * channels * sample_width
    if len(raw) != expected_bytes:
        raise ValueError(
            f"WAV payload is incomplete: expected {expected_bytes} bytes, "
            f"got {len(raw)}"
        )
    decoded = np.frombuffer(raw, dtype="<i2").astype(np.float32) / np.float32(32768.0)
    input_samples = int(decoded.size)

    if input_samples < NUM_SAMPLES:
        padded_samples = NUM_SAMPLES - input_samples
        pcm = np.pad(decoded, (0, padded_samples), mode="constant")
        adjustment = "pad_right"
        truncated_samples = 0
    elif input_samples > NUM_SAMPLES:
        truncated_samples = input_samples - NUM_SAMPLES
        pcm = decoded[:NUM_SAMPLES]
        adjustment = "truncate_tail"
        padded_samples = 0
    else:
        pcm = decoded
        adjustment = "none"
        padded_samples = 0
        truncated_samples = 0

    pcm = np.asarray(pcm, dtype=_FLOAT32)
    report = WavLoadReport(
        path=str(wav_path),
        sample_rate=sample_rate,
        channels=channels,
        sample_width_bytes=sample_width,
        input_samples=input_samples,
        output_samples=int(pcm.size),
        adjustment=adjustment,
        padded_samples=padded_samples,
        truncated_samples=truncated_samples,
    )
    return pcm, report


def _validate_c_identifier(symbol: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", symbol):
        raise ValueError(f"invalid C identifier: {symbol!r}")
    return symbol


def render_filterbank_header(symbol: str = "edgeai_mel_filterbank") -> str:
    """Render the host reference filterbank as a standalone C header."""

    symbol = _validate_c_identifier(symbol)
    guard = f"{symbol.upper()}_H"

    def c_float(value: float) -> str:
        rendered = f"{float(value):.9g}"
        if "." not in rendered and "e" not in rendered.lower():
            rendered += ".0"
        return rendered + "f"

    rows: list[str] = []
    for row in _MEL_FILTERBANK:
        values = ", ".join(c_float(float(value)) for value in row)
        rows.append(f"    {{ {values} }}")

    return "\n".join(
        [
            f"#ifndef {guard}",
            f"#define {guard}",
            "",
            "/* Generated from edgeai.frontend.mel_filterbank(). */",
            "/* EXPERIMENTAL_HOST_ONLY; not a final board contract. */",
            f"#define {symbol.upper()}_NUM_MELS {NUM_MELS}u",
            f"#define {symbol.upper()}_NUM_FREQ_BINS {NUM_FREQ_BINS}u",
            "",
            f"static const float {symbol}[{NUM_MELS}][{NUM_FREQ_BINS}] = {{",
            ",\n".join(rows),
            "};",
            "",
            f"#endif  /* {guard} */",
            "",
        ]
    )


def export_filterbank_header(
    path: str | Path, symbol: str = "edgeai_mel_filterbank"
) -> Path:
    """Write the reference filterbank header and return its path."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_filterbank_header(symbol), encoding="utf-8")
    return output_path


__all__ = [
    "CONTRACT_DISCOVERY",
    "CONTRACT_UNRESOLVED",
    "EXPERIMENTAL_HOST_ONLY",
    "FFT_LENGTH",
    "FRAME_LENGTH",
    "FRAME_STEP",
    "LOG_FLOOR",
    "MAX_FREQ_HZ",
    "MIN_FREQ_HZ",
    "NUM_FRAMES",
    "NUM_FREQ_BINS",
    "NUM_MELS",
    "NUM_SAMPLES",
    "SAMPLE_RATE",
    "WavLoadReport",
    "export_filterbank_header",
    "load_wav",
    "log_mel",
    "mel_filterbank",
    "render_filterbank_header",
]
