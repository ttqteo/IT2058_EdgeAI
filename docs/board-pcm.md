# ESP32 raw PCM frontend

Status: firmware-side implementation frozen for review on 2026-10-20. This
document records host parity and the local ESP-IDF build; it is not a board
pass. No serial, flash, reset, or NVS operation was performed.

## Public contract

The existing framing remains unchanged: `EDGEAI01`, little-endian
`request_id`, little-endian `payload_length`, little-endian IEEE CRC32, then
the payload. Existing operations and plain `RECOGNIZE` retain their old
semantics.

The additive raw operation is:

```text
operation = 0x20 (RECOGNIZE_PCM)
operationVersion = 1
payload = [0x20, 0x01] + exactly 96000 bytes
          signed little-endian PCM16, mono, 16000 Hz, 48000 samples
```

The request CRC covers the complete 96,002-byte operation payload, as it does
for other versioned requests. A successful raw response reports
`input_crc32` over exactly the 96,000 PCM bytes and `feature_crc32` over the
computed 11,920-byte int8 tensor. It keeps `input_bytes=11920` and adds:

```json
{
  "operation": "RECOGNIZE_PCM",
  "pcm_sample_rate_hz": 16000,
  "pcm_samples": 48000,
  "frontend_id": "LOGMEL_16K_400_160_512_40_CMN_V1",
  "frontend_us": 123,
  "feature_crc32": 1234567890
}
```

`frontend_us` includes workspace allocation, DSP, quantization, and release.
`invoke_us` remains the encoder `Invoke()` duration only. The recognition
timer starts after raw-input CRC calculation, before frontend workspace
allocation, and ends after encoder invocation plus output dequantization,
normalization, and cosine ranking. Raw-input CRC runs before the timer; the
feature-CRC duration is measured and subtracted from the recognition interval.
Transport, CRC work, and response formatting are excluded.
The telemetry query-before snapshot is captured before frontend workspace
allocation; query-after is captured after that workspace is freed. The raw
request remains allocated until the response has been sent, so the reported
delta isolates the temporary frontend workspace rather than charging the
transport buffer twice.

HELLO advertises the same raw operation as `raw_pcm` with its version, rate,
sample count, and byte count.

## Frontend implementation

`pcm_frontend.cc` implements the locked path directly for mutable PCM16 input:

1. Decode explicit little-endian signed PCM16 and scale by 32768.
2. Apply periodic Hann over 400 samples, hop 160, and zero-pad to a 512-point
   unnormalized radix-2 FFT.
3. Compute the first 257 power bins and multiply by the generated float32 HTK
   triangular filterbank in `pcm_mel_filterbank.h`.
4. Apply `log(max(energy, 1e-6))`, mean each mel channel over 298 frames, and
   center as float32.
5. Quantize with the runtime tensor scale and zero point using ties-to-even
   rounding and int8 saturation.

The board path does not allocate a 298x40 float matrix. It consumes each frame
into `Workspace`, stores the 40 float log values in the already-consumed part
of the raw request, then writes the centered/quantized result directly to the
model tensor. The existing portable frontend was not changed.

## Resource envelope

| Item | Budget/result |
| --- | ---: |
| TFLM arena | 131,072 B (unchanged) |
| Legacy static request buffer | 23,842 B (unchanged) |
| Raw PCM request allocation | 96,002 B, exact versioned payload |
| Frontend workspace | 10,568 B (`Workspace`, heap allocated) |
| Raw + workspace logical peak | 106,570 B, plus allocator metadata |
| Response buffer | 2,048 B |
| Main task stack configuration | 16,384 B (unchanged) |
| Frontend large automatic arrays | none |

The frontend workspace is allocated from internal 8-bit heap and released before
the normal recognition response constructs telemetry. The encoder arena,
NVS/template record behavior, existing three-template `crc32:60a382e2`
fingerprint contract, and persistence stack assertion are unchanged.

The maximum synthetic `RECOGNIZE_PCM` JSON envelope used by
`tests/test_esp32_pcm.py` is below the 2,048-byte response capacity. This is a
formatting bound, not a live USB measurement.

## Host verification

The standalone host runner used the same C++ implementation that is compiled
into the board and an independent NumPy oracle. Quantized differences below
are element counts over all 11,920 values.

| Input | Max feature error | RMS error | Quantized differences |
| --- | ---: | ---: | ---: |
| silence | 6.395e-14 | 6.395e-14 | 0 |
| impulse | 9.537e-7 | 2.828e-7 | 0 |
| bin sine | 9.239e-7 | 2.425e-7 | 0 |
| deterministic noise | 4.768e-7 | 1.731e-7 | 0 |
| chirp/clipping | 9.537e-7 | 2.348e-7 | 0 |
| `Quang_test.wav`, first 48,000 samples | 9.537e-7 | 1.985e-7 | 0 |
| `Tien_test.wav`, first 48,000 samples | 9.537e-7 | 1.692e-7 | 0 |
| `An.wav`, first 48,000 samples | 9.686e-7 | 2.631e-7 | 0 |

The unchanged portable C host frontend produced the same max/RMS values for
all eight rows above; `tests/test_pcm_frontend.py` runs that comparison for
both synthetic and prepared real PCM as a separate regression path.

The prepared recordings are longer than the raw operation window. The host
check explicitly takes the first exact three seconds; the board operation
rejects any other raw length and performs no padding, cropping, or resampling.

Checks run:

- `./.venv/bin/pytest -q tests/test_pcm_frontend.py` — 17 passed.
- `./.venv/bin/pytest -q tests/test_esp32_pcm.py tests/test_pcm_frontend.py` — 22 passed.
- Existing firmware/protocol tests — 65 passed:
  `test_esp32_smoke.py`, `test_esp32_telemetry.py`,
  `test_demo_pcm_usb.py`, and `test_demo_voice_usb.py`.
- Existing host/portable frontend regression — 24 passed:
  `test_frontend.py` and `test_c_frontend.py`.
- Combined final firmware/frontend command — 111 passed.
- `bash scripts/esp32_idf.sh -C firmware/esp32_s3_smoke build` — passed.
- `bash scripts/esp32_idf.sh -C firmware/esp32_s3_smoke size` — passed.

At the initial firmware handback, no board-side operation had been performed.
The later bounded deployment check is recorded below; it is separate from
these host-only tests.

## Build and provenance pins

The pre-build artifacts were preserved before the first rebuild in
`artifacts/esp32-smoke/20261020/pcm-before-20261020-001/`.
Their copied hashes are: app `36e93554659c6602eba7a19eb9bdaf087af31b8a7bce30977b90aa8dd7862ef1`,
ELF `fd7c330c8ab10b069b443978dbd914082b6b1726e3aa5c2514655cf1d69e64a8`,
map `5d2f934fc0c261001c6039a4512895f45e115cb819bffa5fe59068e479ea3f4d`,
and sdkconfig `d530251dc71a63dba37afe81e2ac42a33c58f0fe22a08527d99030a8729e9056`.

| File | SHA-256 |
| --- | --- |
| `main/main.cc` | `a5670464fbf72e2a1c337c70515745af1b9001fae44c9c03a7e68bfab66fbefe` |
| `main/pcm_frontend.h` | `40cf5fc44dc79727c09407fbde20ff3c7c931e5f39b763e33ee349307f49acb3` |
| `main/pcm_frontend.cc` | `5af29d200ecc6328eb0377d069f0e13f95416b1ca497f20ab05cca400e0e866e` |
| `main/pcm_mel_filterbank.h` | `482f5abf97a01d8b2cc54b5bc8d15568b2dea87f7c5f57ce23f2dae2ba18f715` |
| `main/CMakeLists.txt` | `0aaaa9a7b9f1af432d9f0094b1c779cfbe148a0cddec6b071010f5d5b7d1da4d` |
| `build/esp32_s3_smoke.bin` | `e004d6f0b071c0b6af62fd91b4bce4831f3a3539f8495f62de0e254ddf4981f5` |
| `build/esp32_s3_smoke.elf` | `316527bbd4afcb458370efad71f99fc98d9fd67d718fa46fb2cd0001eda3af47` |
| `build/esp32_s3_smoke.map` | `6d760fbbad0826a19b25a705ba6d39ee0020c69849a8d43a93c03f229f6cb370` |
| `sdkconfig` | `d530251dc71a63dba37afe81e2ac42a33c58f0fe22a08527d99030a8729e9056` |

The final size report was: Flash Data 321,732 B; DIRAM 220,751/341,760 B
(64.59%, 121,009 B remaining); IRAM 16,384/16,384 B (the existing limit);
and total image content 756,045 B. The padded application binary is 756,160 B
(0xb89c0), with 0x47640 bytes free in the 1 MiB app partition.

Lead integration added focused timer and same-buffer-copy guards, and reran
the combined firmware, frontend, client, and dashboard checks: 128 passed. The
PCM path already writes into the encoder input, so `InvokeRequest` skips
`memcpy` when both pointers identify that same buffer. These remain host checks,
not a physical deployment result.

## Freeze and scope release

PCM firmware scope is frozen for lead/reviewer inspection. The changed source
is limited to the firmware main component, its coefficient asset, the two
focused tests, and this board contract note. Portable frontend, model,
template store, SDK configuration, client/UI, dashboard, slides, and Git
history were not changed by this task. Hardware acceptance and any future
microphone acquisition remain outside this freeze.

## Bounded physical check — 2026-10-21

The pinned app was written only at `0x10000`. Fresh security checks identified
the expected board with Secure Boot and Flash Encryption disabled. NVS and
partition readbacks were byte-identical before and after the write. Evidence:
`artifacts/esp32-smoke/20261020/pcm-deploy-002/`.

After incomplete RST/reopen captures and one failed HELLO attempt, the user
power-cycled the board. A POSIX probe then received HELLO/READY with the raw PCM
capability. The unchanged normal USB client subsequently completed
`pcm-check-002/report.json`:

- Five encoder fixtures: maximum signed INT8 differences `[1, 0, 0, 0, 1]` LSB.
- Two frozen PCM queries: Quang → Quang; Tien → Tien. Both feature CRCs matched
  the frozen host oracle tensors; both embeddings were byte-identical to the
  corresponding legacy tensor-path output.
- Board frontend: `7,659,134` / `7,656,455` µs. Encoder Invoke:
  `53,674` / `53,677` µs. Full board recognition:
  `7,715,062` / `7,712,394` µs. USB roundtrip:
  `8,101,228` / `8,099,008` µs. This reference frontend is slow; the approximately
  54 ms encoder time must not be presented as whole-pipeline latency.
- NVS remained READY with Quang/Tien/An and `crc32:60a382e2` before and after.
  STATUS after cleanup reported internal heap free `184,900` B. During PCM
  recognition it reported `86,592` B free with the raw request still allocated;
  the lifetime low-water mark was `76,020` B. Query heap delta `0` means no net
  change between the two recognition snapshots, not zero RAM usage or a
  per-query peak measurement.

The dashboard was restarted with the current PCM backend and a fresh
read-only board check completed. No microphone was recorded in this check.
Evidence establishes two fixed-query behavior only, not live-microphone
accuracy, direct I2S capture, unknown-speaker rejection, or startup reliability.

Independent cached review confirmed the raw/report bindings and numeric
results, but found nonfatal task-watchdog warnings during both PCM frontend
runs. The watchdog timeout is 5 s with panic disabled; the approximately 7.7 s
frontend starves the watched idle task long enough to warn. Both replies
completed. Lead accepts only the bounded experimental numeric/NVS evidence,
not warning-free operation or demo reliability. The warning must remain
visible in the retained USB logs; no watchdog setting was relaxed or disabled.
Scheduling/latency remediation and a fresh physical check remain future work.
