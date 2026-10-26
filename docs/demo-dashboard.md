# Dashboard demo EdgeAI trên Mac

## Input profile raw PCM

Nút **Thu PCM · 3 giây** là một đầu vào thay thế cho microphone trên board,
không phải mô phỏng âm học INMP441. Dashboard mở `sounddevice.RawInputStream`
chỉ sau khi người dùng bấm nút, yêu cầu một kênh, `int16`, blocksize 480 và
16 kHz ở ranh giới API. Mặc định sounddevice tự chọn input hệ thống; có thể
dùng `--input-device 1` hoặc `--input-device "Tên thiết bị"` cho một lựa chọn
cục bộ khác. Chỉ số số được parse thành integer; dashboard không đổi global
device/default input.

`RawInputStream` cung cấp buffer bytes-like. Callback giữ nguyên bytes, kiểm tra
status, số frame và số byte của từng callback, rồi yêu cầu tổng đúng 48.000 mẫu
và 96.000 byte. Không crop, pad, resample, denoise, gain, AGC, noise
synthesis hay NumPy audio transformation. WAV chỉ là container local; payload
PCM trong WAV được gửi nguyên dạng bằng `RECOGNIZE_PCM` (0x20). Đây là luồng
**batch capture 3 giây → gửi PCM → board DSP/inference**, không phải streaming
I2S/DMA hay overlap capture/inference.

16 kHz là requested stream profile ở API; effective format được đọc từ stream
nếu backend báo và format lệch sẽ bị từ chối. Đây không phải tuyên bố native
samplerate của microphone. CoreAudio/driver có thể conversion và DSP/processing
ở acquisition boundary; dashboard ghi rõ khả năng đó nhưng không giả vờ tắt
macOS effects. `physical_mic=false` và `acoustic_equivalence=UNVERIFIED` trong
report chỉ ra rằng chưa có hiệu chuẩn để gọi dữ liệu tương đương INMP441.
PortAudio chỉ nhận request `dither_off`/`clip_off`; driver/device behavior vẫn
được ghi là unknown.

Luồng này cần firmware PCM mới và kiểm tra trên board trước khi demo.
Firmware cũ phải trả lỗi, không có đường fallback tính frontend trên Mac.
Các mẫu NVS hiện có được giữ nguyên. Nút **Đối chứng · 2 tensor cũ** tiếp tục
gửi tensor đã chuẩn bị trước đây, nên không chứng minh frontend trên board.
`frontend_us` đo frontend/quantization; `recognition_us` của lệnh PCM bao gồm
frontend + encoder + cosine, không thu âm/USB. Dữ liệu đo cũ vẫn thuộc lệnh cũ.

`scripts/demo_dashboard.py` là monitor local cho demo closed-set với 2–5 nhãn active
(board hiện đã đăng ký Quang/Tien/An/LS116/LS700).
Server dùng thư viện chuẩn của Python, chỉ bind vào `127.0.0.1:8765`, và gọi
client USB hiện có (`scripts/demo_voice_usb.py`) bằng backend thật mặc định.
Dashboard không có backend giả trong runtime, không có fallback host, không có
CDN và không chạy `PROVISION`, reset, erase hay flash.

Board phải được Lead chuẩn bị sẵn. Contract USB là native Serial/JTAG
`VID/PID 303A:1001`, serial `7C:0C:5F:5B:0E:00`; model và manifest được pin
theo các SHA trong client hiện có. Các thao tác `STATUS`/`RECOGNIZE` chỉ đọc
NVS. Dashboard không thay đổi enrollment.

## Chạy

Từ checkout mới, tạo runtime và cài các dependency cho dashboard:

```sh
python3.11 -m venv artifacts/voice-demo-runtime
artifacts/voice-demo-runtime/bin/python -m pip install -r requirements-voice-demo.txt
```

Cần board đã nạp firmware tương thích và có mẫu NVS hợp lệ. Clone GitHub
không khôi phục enrollment của board. Bản thu cá nhân, tensor query cũ,
smoke vectors sinh sẵn và các report local không được phát hành trong commit này.
Nút thu trực tiếp dùng board đã chuẩn bị; nút đối chứng cần bộ file local
được pin nên không có đủ dữ liệu trên checkout mới. Không tự đăng ký lại
hay dùng tensor khác để vượt kiểm tra hash.

Từ thư mục gốc repository:

```sh
artifacts/voice-demo-runtime/bin/python -B scripts/demo_dashboard.py
# Chọn input cục bộ theo tên hoặc index (tùy chọn):
artifacts/voice-demo-runtime/bin/python -B scripts/demo_dashboard.py --input-device 1
```

Mở bằng trình duyệt tại <http://127.0.0.1:8765/>. Trang ban đầu chỉ tải
HTML/CSS/JS, lấy CSRF token và đọc state cache; không mở serial, không gọi
`STATUS`, không gọi microphone. Server không tự mở trình duyệt.

## Trang hiệu năng riêng

Mở <http://127.0.0.1:8765/performance> song song với trang demo. Trang chỉ đọc
cache của cùng server, nên không tranh cổng serial hay tự mở mic. Nút **Đọc RAM
board** chủ động dùng action `check_board` có CSRF và khóa một job dùng chung.

RAM còn trống là heap `INTERNAL | 8BIT` do firmware đo, không phải phép trừ
512 KB SRAM của chip. Firmware cũ không gửi telemetry sẽ hiện **chưa có số đo**.
RAM trống, khối trống lớn nhất và low-water từ boot đều có thời điểm snapshot;
đây không phải giám sát RAM liên tục. Low-water tổng hợp từng heap không phải
RAM đỉnh riêng query.

Query gần nhất chỉ lấy từ job hoàn tất thành công (query cuối nếu chạy hai
file). Trang phân biệt `invoke_us` (encoder Invoke), `frontend_us` (frontend và
lượng tử hóa trên board, chỉ có với PCM), và `recognition_us` (xử lý trên
board: gồm frontend nếu là PCM, copy tensor, encoder, chuẩn hóa/cosine/top-1;
không gồm CRC, JSON hay USB). `usb_roundtrip_us` do Mac đo quanh lời gọi nhận
dạng: gồm truyền USB, xử lý board và chờ phản hồi, không gồm thu âm hay mở
kết nối. Không so thời gian lệnh tensor cũ với tổng xử lý PCM như cùng một phép đo.
Arena model là vùng tái sử dụng, tensor vào/ra đã nằm trong vùng đó. Heap
trước/sau và chênh lệch là hai snapshot; không tính stack hay cấp phát tạm đã
giải phóng, nên không được gọi là tổng RAM hay đỉnh RAM của query.

Query cũ vẫn ghi rõ ID/thời điểm khi job sau đang chạy hoặc gặp lỗi. Không
dùng kết quả lỗi hoặc dữ liệu giả thay cho số đo board.

Có thể double-click `launch-demo.command` trên macOS. Launcher chỉ khởi động
đúng runtime repository; vẫn mở URL bằng tay để người trình diễn chủ động.

Dừng server bằng `Ctrl-C` trong Terminal. Job đang chạy sẽ không được dùng làm
fallback; hãy chờ kết thúc hoặc dừng server rồi chạy lại sau.

## Ba nút trình diễn

- **Kiểm tra board**: mở đúng thiết bị, gọi `STATUS` read-only và hiển thị
  device thật, trạng thái NVS, số template/nhãn active, fingerprint, model hash, runtime
  `ESP_NN_ANSI_C_REFERENCE` và arena.
- **Đối chứng · 2 tensor cũ**: xác thực prepared manifest, chạy năm fixture parity,
  rồi gửi tensor đã chuẩn bị từ `Quang_test.m4a` và `Tien_test.m4a` tới board.
  Nút này bỏ qua frontend mới. Kết quả lấy từ
  `top1`/cosine của board; mỗi query có trường `correct` riêng. Đây là closed-set
  demo, không phải accuracy claim.
- **Thu PCM · 3 giây**: chỉ bắt đầu sau click. Recorder local mở
  `sounddevice.RawInputStream` và phát tín hiệu lifecycle khi stream đã mở rồi
  khi capture thật sự bắt đầu. UI giữ `MIC_PERMISSION`/`MIC_PREPARE` cho tới
  tín hiệu đó; chỉ sau đó mới hiện `MIC_RECORDING` và đếm ngược 3 giây, tiếp
  theo là `PCM_TRANSFER` và `BOARD_INFER`. File WAV chỉ nằm trong thư mục job
  local và không được upload. Live query không đọc enrollment hay prepared
  query trên Mac.

  Report ghi `profile_id`, input device/name/index, requested/effective stream
  format, sample count/bytes, SHA-256 và CRC32 của đúng PCM payload, cùng SHA
  của WAV container. Recorder được inject trong test được gắn
  `INJECTED_RECORDER_UNVERIFIED`; nó không phải bằng chứng microphone native.

Trong result card, timing được tách thành **board total `recognition_us`** từ
telemetry, **`frontend_us`** trên board (chỉ có với PCM), **encoder Invoke** và
**USB roundtrip** do Mac đo quanh request. Dashboard không suy ra ô thiếu từ ô
khác, không gọi chúng là end-to-end latency, và không tính thời gian chờ quyền
microphone vào một timer capture-to-result. Live query không có nhãn ground
truth nên `correct` là `null`, không suy diễn đúng/sai.

## Output và trạng thái lỗi

Mỗi lần click tạo một thư mục mới dưới:

```text
artifacts/demo-dashboard/<timestamp>-<random>/
```

`report.json` ghi contract, job, device evidence, board response đã rút gọn,
query result và lỗi nếu có. Không dùng tên file từ HTTP request để tạo path.
Dashboard chỉ cho phép một job phần cứng tại một thời điểm; các request thứ hai
bị từ chối trong khi job đầu đang giữ serial lock. Poll `/api/state` chỉ đọc
state trong RAM, không tự động poll USB. Nhãn `SNAPSHOT BOARD` kèm thời điểm
`Lần kiểm tra` là bằng chứng của lần action gần nhất, không phải live heartbeat.

Nếu không có board, sai hash, sai metadata, NVS chưa `READY`/không đủ hai
template, input lỗi hoặc protocol lỗi, UI hiển thị `OFFLINE`/`ERROR` và report
ERROR. Với USB native không khớp, UI nêu rõ `VID/PID 303A:1001`, serial pin và
hướng dẫn kết nối đúng board; dashboard không tự reset, erase hoặc flash. Không
tạo dữ liệu giả và không chuyển sang host backend. Report lỗi hoặc lỗi ghi
report đều không giữ lại result thành công.

HTTP server có các guard loopback/`Host`/`Origin`, CSRF token cho POST,
body JSON giới hạn 32 KiB, allowlist action, static route cố định và CSP
không cho CDN/CORS. Không dùng URL/path request để expose filesystem.

## Kiểm thử

```sh
.venv/bin/python -B -m pytest -p no:cacheprovider -q tests/test_demo_dashboard.py
```

Test dùng fake client chỉ qua dependency injection để kiểm tra state machine,
security guard, concurrency, report `correct=false`, error/no-fallback, output
scope, raw byte preservation, malformed callback/status/overshoot rejection,
capture provenance/hash binding, timing absence và pha thu PCM. Fake client và
injected recorder là dữ liệu synthetic, không phải bằng chứng cho board hay
microphone native. Browser rendering guard chạy riêng:

```sh
node tests/test_dashboard_web.cjs
```

Test hardware native USB và browser QA cần chạy riêng khi Lead kết nối board và
kiểm tra giao diện.
