"""Host parity checks for the firmware-side raw PCM frontend.

The C++ runner is the board implementation compiled on the host.  The NumPy
oracle below is deliberately written independently of ``edgeai.frontend`` so
that this test can report implementation drift rather than reusing the same
call graph.  It does not establish ESP32 timing or board acceptance.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import wave

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "firmware" / "esp32_s3_smoke" / "main"
HOST_DIR = ROOT / "firmware" / "host"
HOST_BINARY = HOST_DIR / "build" / "frontend_host"
NUM_SAMPLES = 48_000
SAMPLE_RATE = 16_000
FRAME_LENGTH = 400
HOP_LENGTH = 160
FFT_LENGTH = 512
NUM_BINS = FFT_LENGTH // 2 + 1
NUM_MELS = 40
NUM_FRAMES = 298
INPUT_SCALE = np.float32(0.12695927917957306)
INPUT_ZERO_POINT = -12


def _python_filterbank() -> np.ndarray:
    """Build the independent float32 HTK filterbank used by both checks."""

    def hz_to_mel(hz: np.ndarray | float) -> np.ndarray:
        return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)

    def mel_to_hz(mel: np.ndarray) -> np.ndarray:
        return 700.0 * (np.power(10.0, mel / 2595.0) - 1.0)

    mel_points = np.linspace(
        hz_to_mel(0.0), hz_to_mel(SAMPLE_RATE / 2.0), NUM_MELS + 2,
        dtype=np.float64,
    )
    hz_points = mel_to_hz(mel_points)
    frequencies = np.arange(NUM_BINS, dtype=np.float64) * SAMPLE_RATE / FFT_LENGTH
    lower = hz_points[:-2, None]
    center = hz_points[1:-1, None]
    upper = hz_points[2:, None]
    bins = frequencies[None, :]
    rising = (bins - lower) / (center - lower)
    falling = (upper - bins) / (upper - center)
    return np.maximum(0.0, np.minimum(rising, falling)).astype(np.float32)


def _python_oracle(raw_pcm: bytes) -> np.ndarray:
    """Return centered float32 features from exact little-endian PCM16 bytes."""

    assert len(raw_pcm) == NUM_SAMPLES * 2
    pcm16 = np.frombuffer(raw_pcm, dtype="<i2")
    values = pcm16.astype(np.float32) / np.float32(32768.0)
    values = values.astype(np.float64)
    filterbank = _python_filterbank()

    frame_starts = np.arange(NUM_FRAMES, dtype=np.int64) * HOP_LENGTH
    frames = np.zeros((NUM_FRAMES, FFT_LENGTH), dtype=np.float64)
    window = 0.5 - 0.5 * np.cos(
        2.0 * np.pi * np.arange(FRAME_LENGTH, dtype=np.float64) / FRAME_LENGTH
    )
    frames[:, :FRAME_LENGTH] = (
        values[frame_starts[:, None] + np.arange(FRAME_LENGTH)] * window
    )
    spectrum = np.fft.rfft(frames, n=FFT_LENGTH, axis=1)
    power = spectrum.real * spectrum.real + spectrum.imag * spectrum.imag
    energies = power @ filterbank.astype(np.float64).T
    logs = np.log(np.maximum(energies, 1.0e-6))
    centered = logs - np.mean(logs, axis=0, keepdims=True)
    return centered.astype(np.float32)


def _pcm_bytes(samples: np.ndarray) -> bytes:
    values = np.asarray(samples, dtype=np.int16)
    assert values.shape == (NUM_SAMPLES,)
    return values.astype("<i2", copy=False).tobytes()


def _synthetic_pcm() -> dict[str, bytes]:
    axis = np.arange(NUM_SAMPLES, dtype=np.float64)
    rng = np.random.default_rng(2058)
    impulse = np.zeros(NUM_SAMPLES, dtype=np.int16)
    impulse[123] = 32767
    sine = np.rint(
        0.8 * 32767.0 * np.sin(2.0 * np.pi * 23.0 * axis / FFT_LENGTH)
    ).astype(np.int16)
    noise = np.rint(rng.uniform(-0.8, 0.8, NUM_SAMPLES) * 32767.0).astype(np.int16)
    chirp_phase = 2.0 * np.pi * (
        220.0 * axis / SAMPLE_RATE
        + 3600.0 * axis * axis / (2.0 * NUM_SAMPLES * SAMPLE_RATE)
    )
    chirp = np.rint(
        np.clip(1.1 * np.sin(chirp_phase), -1.0, 1.0) * 32767.0
    ).astype(np.int16)
    return {
        "silence": _pcm_bytes(np.zeros(NUM_SAMPLES, dtype=np.int16)),
        "impulse": _pcm_bytes(impulse),
        "bin_sine": _pcm_bytes(sine),
        "deterministic_noise": _pcm_bytes(noise),
        "chirp_clipping": _pcm_bytes(chirp),
    }


def _real_pcm() -> dict[str, bytes]:
    candidates = {
        "Quang_test": ROOT / "artifacts/voice-demo/20261019/prepared/Quang_test.wav",
        "Tien_test": ROOT / "artifacts/voice-demo/20261019/prepared/Tien_test.wav",
        "An": ROOT / "artifacts/voice-demo/20261020/an-prepared-001/An.wav",
    }
    result: dict[str, bytes] = {}
    for name, path in candidates.items():
        if not path.is_file():
            continue
        with wave.open(str(path), "rb") as handle:
            assert handle.getframerate() == SAMPLE_RATE
            assert handle.getnchannels() == 1
            assert handle.getsampwidth() == 2
            assert handle.getcomptype() == "NONE"
            raw = handle.readframes(NUM_SAMPLES)
        assert len(raw) == NUM_SAMPLES * 2
        result[name] = raw
    return result


@pytest.fixture(scope="session")
def pcm_runner(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("clang++")
    if compiler is None:
        pytest.skip("clang++ is required for the firmware frontend host check")
    root = tmp_path_factory.mktemp("pcm-frontend")
    source = root / "runner.cc"
    binary = root / "pcm_frontend_runner"
    source.write_text(
        r'''
#include <cstdint>
#include <cstdio>
#include <vector>

#include "pcm_frontend.h"

namespace {

bool ReadFile(const char* path, std::vector<std::uint8_t>* bytes) {
  FILE* file = std::fopen(path, "rb");
  if (file == nullptr || std::fseek(file, 0, SEEK_END) != 0) return false;
  const long length = std::ftell(file);
  if (length < 0 || std::fseek(file, 0, SEEK_SET) != 0) return false;
  bytes->resize(static_cast<std::size_t>(length));
  const std::size_t read = std::fread(bytes->data(), 1, bytes->size(), file);
  std::fclose(file);
  return read == bytes->size();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) return 2;
  std::vector<std::uint8_t> raw;
  if (!ReadFile(argv[1], &raw) ||
      raw.size() != edgeai_pcm_frontend::kPcmBytes) return 3;

  edgeai_pcm_frontend::Workspace workspace{};
  std::vector<float> features(edgeai_pcm_frontend::kFeatureValues);
  const auto feature_status = edgeai_pcm_frontend::ComputeLogMelPcm16(
      raw.data(), raw.size(), &workspace, features.data(), features.size());
  if (feature_status != edgeai_pcm_frontend::Status::kOk) return 4;
  FILE* feature_file = std::fopen(argv[2], "wb");
  if (feature_file == nullptr ||
      std::fwrite(features.data(), sizeof(float), features.size(), feature_file) !=
          features.size()) return 5;
  std::fclose(feature_file);

  if (!ReadFile(argv[1], &raw)) return 6;
  std::vector<std::int8_t> quantized(edgeai_pcm_frontend::kFeatureBytes);
  const auto quant_status = edgeai_pcm_frontend::ComputeAndQuantizePcm16(
      raw.data(), raw.size(), &workspace, quantized.data(), quantized.size(),
      0.12695927917957306F, -12);
  if (quant_status != edgeai_pcm_frontend::Status::kOk) return 7;
  FILE* quantized_file = std::fopen(argv[3], "wb");
  if (quantized_file == nullptr ||
      std::fwrite(quantized.data(), sizeof(std::int8_t), quantized.size(),
                  quantized_file) != quantized.size()) return 8;
  std::fclose(quantized_file);
  return 0;
}
''',
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic",
            "-I",
            str(MAIN),
            str(source),
            str(MAIN / "pcm_frontend.cc"),
            "-o",
            str(binary),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"firmware frontend host build failed\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    return binary


def _run_runner(binary: Path, tmp_path: Path, name: str, raw: bytes) -> tuple[np.ndarray, np.ndarray]:
    input_path = tmp_path / f"{name}.pcm"
    feature_path = tmp_path / f"{name}.features"
    quantized_path = tmp_path / f"{name}.quantized"
    input_path.write_bytes(raw)
    result = subprocess.run(
        [str(binary), str(input_path), str(feature_path), str(quantized_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"firmware frontend runner failed for {name}: {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    features = np.fromfile(feature_path, dtype="<f4")
    quantized = np.fromfile(quantized_path, dtype=np.int8)
    assert features.size == NUM_FRAMES * NUM_MELS
    assert quantized.size == NUM_FRAMES * NUM_MELS
    return features.reshape(NUM_FRAMES, NUM_MELS), quantized


@pytest.fixture(scope="session")
def portable_runner() -> Path:
    """Build the unchanged portable host peer for real-PCM regression."""

    result = subprocess.run(
        ["make", "-C", str(HOST_DIR), "CC=clang"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"portable host build failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert HOST_BINARY.is_file() and HOST_BINARY.stat().st_mode & 0o111
    return HOST_BINARY


def _run_portable_runner(
    binary: Path, tmp_path: Path, name: str, raw: bytes
) -> np.ndarray:
    pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / np.float32(32768.0)
    pcm_path = tmp_path / f"portable-{name}.pcm.f32"
    mel_path = tmp_path / f"portable-{name}.mel.f32"
    output_path = tmp_path / f"portable-{name}.features.f32"
    pcm_path.write_bytes(np.asarray(pcm, dtype="<f4").tobytes())
    mel_path.write_bytes(np.asarray(_python_filterbank(), dtype="<f4").tobytes())
    result = subprocess.run(
        [str(binary), str(pcm_path), str(mel_path), str(output_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"portable frontend runner failed for {name}: {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    output = np.fromfile(output_path, dtype="<f4")
    assert output.size == NUM_FRAMES * NUM_MELS
    return output.reshape(NUM_FRAMES, NUM_MELS)


@pytest.mark.parametrize("name", ["silence", "impulse", "bin_sine", "deterministic_noise", "chirp_clipping"])
def test_firmware_frontend_matches_independent_python_synthetic(
    pcm_runner: Path, name: str, tmp_path: Path
) -> None:
    raw = _synthetic_pcm()[name]
    c_features, c_quantized = _run_runner(pcm_runner, tmp_path, name, raw)
    python_features = _python_oracle(raw)
    difference = c_features.astype(np.float64) - python_features.astype(np.float64)
    max_error = float(np.max(np.abs(difference)))
    rms_error = float(np.sqrt(np.mean(difference * difference)))
    assert max_error <= 1.0e-4, f"{name}: max feature error={max_error}"
    assert rms_error <= 1.0e-5, f"{name}: RMS feature error={rms_error}"
    expected_quantized = np.clip(
        np.rint(python_features.astype(np.float32) / INPUT_SCALE)
        + INPUT_ZERO_POINT,
        -128,
        127,
    ).astype(np.int8).reshape(-1)
    np.testing.assert_array_equal(c_quantized, expected_quantized)


@pytest.mark.parametrize(
    "name", ["silence", "impulse", "bin_sine", "deterministic_noise", "chirp_clipping"]
)
def test_unchanged_portable_frontend_matches_independent_python_synthetic(
    portable_runner: Path, name: str, tmp_path: Path
) -> None:
    raw = _synthetic_pcm()[name]
    portable_features = _run_portable_runner(portable_runner, tmp_path, name, raw)
    python_features = _python_oracle(raw)
    difference = portable_features.astype(np.float64) - python_features.astype(np.float64)
    assert float(np.max(np.abs(difference))) <= 1.0e-4
    assert float(np.sqrt(np.mean(difference * difference))) <= 1.0e-5


@pytest.mark.parametrize("name", ["Quang_test", "Tien_test", "An"])
def test_firmware_frontend_matches_independent_python_real_pcm(
    pcm_runner: Path, name: str, tmp_path: Path
) -> None:
    raw = _real_pcm().get(name)
    if raw is None:
        pytest.skip(f"prepared real WAV for {name} is unavailable")
    c_features, c_quantized = _run_runner(pcm_runner, tmp_path, name, raw)
    python_features = _python_oracle(raw)
    difference = c_features.astype(np.float64) - python_features.astype(np.float64)
    max_error = float(np.max(np.abs(difference)))
    rms_error = float(np.sqrt(np.mean(difference * difference)))
    assert max_error <= 1.0e-4, f"{name}: max feature error={max_error}"
    assert rms_error <= 1.0e-5, f"{name}: RMS feature error={rms_error}"
    expected_quantized = np.clip(
        np.rint(python_features.astype(np.float32) / INPUT_SCALE)
        + INPUT_ZERO_POINT,
        -128,
        127,
    ).astype(np.int8).reshape(-1)
    np.testing.assert_array_equal(c_quantized, expected_quantized)


@pytest.mark.parametrize("name", ["Quang_test", "Tien_test", "An"])
def test_unchanged_portable_frontend_matches_independent_python_real_pcm(
    portable_runner: Path, name: str, tmp_path: Path
) -> None:
    raw = _real_pcm().get(name)
    if raw is None:
        pytest.skip(f"prepared real WAV for {name} is unavailable")
    portable_features = _run_portable_runner(portable_runner, tmp_path, name, raw)
    python_features = _python_oracle(raw)
    difference = portable_features.astype(np.float64) - python_features.astype(np.float64)
    assert float(np.max(np.abs(difference))) <= 1.0e-4
    assert float(np.sqrt(np.mean(difference * difference))) <= 1.0e-5


def test_frontend_contract_is_exact_and_workspace_is_bounded() -> None:
    header = (MAIN / "pcm_frontend.h").read_text(encoding="utf-8")
    source = (MAIN / "pcm_frontend.cc").read_text(encoding="utf-8")
    tables = (MAIN / "pcm_frontend_tables.h").read_text(encoding="utf-8")
    asset = (MAIN / "pcm_mel_filterbank.h").read_text(encoding="utf-8")
    assert "kPcmBytes = kPcmSamples * sizeof(std::int16_t)" in header
    assert "kFrameLength = 400U" in header
    assert "kHopLength = 160U" in header
    assert "kFftLength = 512U" in header
    assert "kFrameCount = 298U" in header
    assert "std::nearbyintf" in source
    assert "edgeai_mel_filterbank[40][257]" in asset
    assert "static const float" in asset


def test_optimized_frontend_uses_precomputed_sparse_tables() -> None:
    source = (MAIN / "pcm_frontend.cc").read_text(encoding="utf-8")
    tables = (MAIN / "pcm_frontend_tables.h").read_text(encoding="utf-8")
    assert '#include "pcm_frontend_tables.h"' in source
    assert "kPeriodicHann[400]" in tables
    assert "kFftTwiddles[256]" in tables
    assert "kFftBitReverse[512]" in tables
    assert "kMelOffsets[41]" in tables
    assert "kMelEntries[494]" in tables
    assert "std::cos" not in source
    assert "std::sin" not in source
    assert "workspace->power" not in source
    assert "edgeai_mel_filterbank" not in source
