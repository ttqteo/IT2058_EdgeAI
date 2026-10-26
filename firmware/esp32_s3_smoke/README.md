# ESP32-S3 raw PCM speaker demo

The Mac captures three seconds of raw PCM16 mono at 16 kHz. The application
preserves the capture bytes and sends them over native USB. The ESP32 performs
log-mel/CMN, INT8 quantization, encoder inference, normalization and cosine ranking.
CoreAudio may convert or process acquisition audio; this is not calibrated
INMP441 acoustic equivalence. Capture completes before transfer; this is not
streaming I2S/DMA. The firmware does not implement a physical board microphone.

The runtime uses ESP-IDF 5.5.1 and locked esp-tflite-micro/esp-nn 1.4.0,
with ANSI-C kernels. Optimized ESP-NN was not accepted for this model.
It retains a 128 KiB internal tensor arena and a 16 KiB main-task stack.
Five frozen real-TRAIN encoder fixtures must pass at <=1 LSB before the USB
service becomes ready. This gate alone is not end-to-end accuracy evidence.

## USB and persistence

Frames are `EDGEAI01` + little-endian uint32 request ID, payload length and
IEEE CRC32(payload), followed by the payload. Versioned payloads begin with
opcode and version 1. Maximum payload is 96,010 bytes.

| Operation | Opcode | Payload bytes | Effect |
|---|---:|---:|---|
| HELLO | empty | 0 | Metadata/capabilities |
| ENCODE | 0x01 | 11,922 | Encoder diagnostic |
| PROVISION | 0x10 | 23,842 | Explicit initial Quang/Tien registration |
| STATUS | 0x11 | 2 | Read-only persistence/heap snapshot |
| RECOGNIZE | 0x12 | 11,922 | Legacy tensor query, bypasses frontend |
| ADD_AN | 0x13 | 11,922 | Explicit third-template append |
| RECOGNIZE_PCM | 0x20 | 96,002 | Raw PCM query, board frontend and ranking |
| APPEND_PCM | 0x21 | 96,010 | Explicit label + raw PCM registration |

Boot reads `edgeai_vm/templates` without migration or automatic provisioning.
It accepts legacy v1 (1,640 bytes), v2 (2,416 bytes), and v3 (3,972 bytes,
3–5 labels). Records bind the model, labels, normalized vectors and CRC.
Explicit writes commit and verify readback before publishing RAM state;
uncertain writes fail closed. There is no erase/reformat recovery command.
The dashboard only exposes STATUS and query operations.

Cosine scores are similarities, not probabilities. The highest-scoring
registered label is selected; unknown-speaker rejection is not implemented.

## Build prerequisites

The repository publishes source/configuration, not generated smoke inputs,
personal recordings, NVS backups, toolchains or compiled firmware. A fresh
checkout can run the dashboard against an already prepared compatible board.
Building firmware additionally requires the pinned generated model/fixture tree
and an ESP-IDF 5.5.1 installation. `scripts/prepare_esp32_smoke.py` can regenerate
that tree from the original FULL100 real-TRAIN cache and host runner; those
large local prerequisites are not included. The published model is also under
`models/full100-20261015/baseline_kd/encoder_int8.tflite`.

The generated directory must contain `encoder_int8.tflite` plus five
`real_train_000` through `real_train_004` input/expected `.int8` pairs and their
manifest. Never substitute arbitrary vectors to bypass the smoke gate.

With the existing repository-local toolchain and generated tree:

```sh
bash scripts/esp32_idf.sh -C firmware/esp32_s3_smoke build
```

With an independently installed IDF 5.5.1, activate its environment and use
`idf.py -C firmware/esp32_s3_smoke build`. This command compiles only.
The retained board factory slot is `[0x10000,0x110000)`; configuration is for
the measured DIO/16 MB board and is not a universal ESP32-S3 board preset.
The generated full flash bundle includes bootloader/partition writes. Existing
deployments used reviewed app-only writes at `0x10000` to preserve NVS;
this publication performs no device deployment.

See [dashboard](../../docs/demo-dashboard.md),
[PCM processing](../../docs/board-pcm.md),
[heap/timing semantics](../../docs/board-telemetry.md), and
[enrollment](../../docs/multi-enrollment.md).
