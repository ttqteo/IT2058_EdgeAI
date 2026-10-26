# Demo giọng nói qua Mac và ESP32-S3

## Thay đổi PCM ngày 20/10

`run --live` mới thu trực tiếp PCM16 mono 16 kHz, 3 giây rồi gửi âm thanh
thô bằng `RECOGNIZE_PCM=0x20`, phiên bản 1, payload 96.002 byte (header
operation 2 byte + PCM 96.000 byte). Board thực hiện frontend, CMN, INT8,
encoder và cosine. Ứng dụng Mac không resample/tính đặc trưng trong đường
live; định dạng acquisition vẫn do thiết bị/OS cung cấp. Mic nối board là
bước tiếp theo. Cần nạp và xác minh firmware PCM mới trước khi dùng.

Gửi lại một capture đúng định dạng, không crop/pad/chuyển đổi trên Mac:

```sh
artifacts/voice-demo-runtime/bin/python -B scripts/demo_voice_usb.py run-pcm \
  --wav <capture-3s-mono-pcm16-16k.wav> --output <new-output-directory>
```

Đường live hiện tại gửi PCM thô; ESP32 chạy log-mel/CMN, lượng tử hóa
INT8, encoder và cosine. Dashboard dùng `RawInputStream` để giữ nguyên byte
thu tại API; CoreAudio/driver có thể chuyển đổi hoặc xử lý ở nguồn thu.
Đây là đầu vào thay thế, chưa được hiệu chuẩn tương đương mic INMP441.

Các lệnh `run` không có `--live`, `provision`, `prepare-an` và `add-an` bên
dưới là đường tensor đối chứng/đăng ký cũ. Chúng dùng dữ liệu đã chuẩn bị
trên Mac và không chứng minh tiền xử lý trên board. Mô tả trước đây về
`--live` gửi tensor đã lượng tử hóa không còn áp dụng cho phiên bản này.

Board đọc từ 2 đến 5 mẫu tùy record v1/v2/v3. Nhận dạng không tự đăng ký
lại hoặc ghi NVS. Demo chọn nhãn có cosine cao nhất, chưa có ngưỡng từ chối
người lạ; không thể dùng kết quả này làm độ chính xác tổng quát.

## Dữ liệu cố định

- Enrollment: `data/Quang.m4a`, `data/Tien.m4a`.
- Query riêng: `data/Quang_test.m4a`, `data/Tien_test.m4a`; không bao giờ dùng
  query để enrollment.
- Prepared directory: `artifacts/voice-demo/20261019/prepared/`.
- Prepared manifest SHA-256:
  `1d73cad6e9697999f1cfa1ed0a672b12fd8fd0c533c34dab7b6afe622561baa7`.
- Model SHA-256:
  `aa42a3140f375f3a2c93648893132b880dd3e9f4be4ca8f82cde09fc6c440727`.
- An enrollment source: `data/An.m4a`, SHA-256
  `43d16865db83a96e154449530fc81eba53b8e2ace2ff640e5c0237aeff64fe8b`.
  Chuẩn bị An có manifest riêng và không dùng query/mượn manifest bốn file cũ.

The client verifies the pinned manifest and every source/WAV/INT8 input before
use. It does not rewrite personal recordings or silently choose a new
enrollment set.

## Explicit board lifecycle

Provision exactly once on an empty board. This operation sends the two frozen
enrollment tensors in fixed order (Quang, then Tien); the board runs its own
encoder and commits one model-bound NVS blob. It does not send Mac embeddings:

```sh
artifacts/voice-demo-runtime/bin/python -B scripts/demo_voice_usb.py provision \
  --output "artifacts/voice-demo/$(date +%Y%m%d-%H%M%S)-provision"
```

Read persistence status without changing flash:

```sh
artifacts/voice-demo-runtime/bin/python -B scripts/demo_voice_usb.py status \
  --output "artifacts/voice-demo/$(date +%Y%m%d-%H%M%S)-status"
```

Chỉ để append An vào board đang báo đúng `READY`, count 2, labels Quang/Tien:

```sh
artifacts/voice-demo-runtime/bin/python -B scripts/demo_voice_usb.py prepare-an \
  --output "artifacts/voice-demo/$(date +%Y%m%d-%H%M%S)-an-prepared"
artifacts/voice-demo-runtime/bin/python -B scripts/demo_voice_usb.py add-an \
  --prepared-an "artifacts/voice-demo/<an-prepared>" \
  --output "artifacts/voice-demo/$(date +%Y%m%d-%H%M%S)-add-an"
```

`add-an` sends one `[13 01] + An[11920]` request. It refuses an empty board,
unexpected/corrupt state, or a board that already has An; those cases make no write.

Legacy frozen-tensor runs never reprovision. They require status `READY`, run the five
frozen encoder smoke fixtures, then send only `RECOGNIZE` requests; top1 and
scores in `report.json` come from the board:

```sh
artifacts/voice-demo-runtime/bin/python -B scripts/demo_voice_usb.py run \
  --output "artifacts/voice-demo/$(date +%Y%m%d-%H%M%S)-usb"
```

An optional Mac-side oracle can be recorded for comparison. It is explicitly
reported as validation and never drives `top1`:

```sh
artifacts/voice-demo-runtime/bin/python -B scripts/demo_voice_usb.py run \
  --validate-mac --output "artifacts/voice-demo/$(date +%Y%m%d-%H%M%S)-usb-check"
```

For a raw Mac microphone query (board frontend, quantization, encoder and cosine):

```sh
artifacts/voice-demo-runtime/bin/python -B scripts/demo_voice_usb.py run --live \
  --output "artifacts/voice-demo/$(date +%Y%m%d-%H%M%S)-live"
```

The host backend remains an explicitly labeled offline oracle and is not a USB
fallback:

```sh
artifacts/voice-demo-runtime/bin/python -B scripts/demo_voice_usb.py run \
  --backend host --output "artifacts/voice-demo/$(date +%Y%m%d-%H%M%S)-host"
```

## Wire and persistence summary

The USB frame is `EDGEAI01`, request ID, payload length, and IEEE/zlib CRC-32;
the header is 20 bytes and the frame deadline is 10 seconds. The legacy maximum
payload is 23,842 bytes; `RECOGNIZE_PCM` uses 96,002 bytes
(`20 01` plus exactly 96,000 raw PCM bytes). `APPEND_PCM=0x21` uses
96,010 bytes (operation/version + 8-byte label + PCM), the current maximum.
Operation version 1 uses
`RECOGNIZE_PCM=0x20`, `ENCODE=0x01`, `PROVISION=0x10`,
`STATUS=0x11`, `RECOGNIZE=0x12`, and `ADD_AN=0x13`. `PROVISION` is exactly
`[10 01] + Quang[11920] + Tien[11920]`; `RECOGNIZE` is exactly
`[12 01] + query[11920]`; `ADD_AN` is `[13 01] + An[11920]`. Full bounds and response fields are in
`firmware/esp32_s3_smoke/README.md`.

The board loads CRC/model-bound v1 (1,640 bytes, two templates), v2
(2,416 bytes, three templates), or v3 (3,972 bytes, three to five templates)
from the dedicated `edgeai_vm/templates` namespace at boot, without automatic
migration. Records store normalized little-endian float32 vectors (192 values
per active label), labels, model metadata and an IEEE CRC-32; not audio files.
The first three labels in v2/v3 remain Quang/Tien/An. Explicit `APPEND_PCM`
computes a new vector on the board and preserves prior vectors before
committing/readback. NVS init never erases or
reformats. An ambiguous set/commit/readback reports
`PERSISTENCE_UNCERTAIN`, preserves the prior RAM state, and blocks live
recognition until reboot reconciliation. `STATUS` exposes the state, version-dependent
record size, and deterministic `record_fingerprint` (`crc32:<8 lowercase hex
digits>`); compare it before and after reboot to prove the same persisted
record was loaded without reprovisioning. Existing templates are not
silently overwritten, and there is no destructive clear command. A normal
`run --live` does not read the prepared enrollment/query files; it uses the
board-loaded record. The optional `--validate-mac` mode is the only live path
that additionally reads frozen enrollment inputs for an independent oracle.

Use only the approved native USB Serial/JTAG device (`VID/PID 303A:1001` and
the configured board serial). The USB bridge (`VID/PID 1A86:55D3`) is outside
this client contract. The Lead owns connection, flash, reset, and deployment;
no hardware PASS is implied by these source/build instructions.

## Kết quả lịch sử ngày 19/10/2026 — hai mẫu, tensor query

Đã nạp riêng app tại `0x10000`, đăng ký đúng hai mẫu và kiểm tra sau một lần
reset có kiểm soát. Đọc lại NVS xác nhận record 1.640 byte, fingerprint
`crc32:a00ff6a4`; hai vector khớp chính xác embedding đã chuẩn hóa từ encoder
trên board. Sau reset, board tự nạp lại record, không gửi lại enrollment hay
gọi `PROVISION`.

- `Quang_test.m4a` → Quang: cosine Quang `0.912962556`, Tien `0.662654161`.
- `Tien_test.m4a` → Tien: cosine Tien `0.799418211`, Quang `0.701422989`.
- Năm fixture encoder: sai lệch tối đa `[1, 0, 0, 0, 1]` LSB.
- Phép tính cosine độc lập từ vector đọc lại NVS lệch tối đa `1.55e-8`.

Bằng chứng: `artifacts/esp32-smoke/20261019/persist-match-001/deployment.json`
và `artifacts/voice-demo/20261019/persist-after-reboot-001/report.json`.
Trong lần kiểm tra lịch sử này board đã có hai mẫu; các lần mở rộng về sau
không thay đổi phạm vi số liệu ở trên. Khi board đã đăng ký, **không chạy
lại `provision`**. Dùng STATUS để đọc số mẫu hiện tại. Board lưu vector, không lưu file thu âm gốc. Kết quả này
chỉ xác nhận hai query thu sẵn và một lần reset; chưa kiểm tra thu mic trực
tiếp, rút nguồn, độ bền ghi NVS hay từ chối người lạ.
