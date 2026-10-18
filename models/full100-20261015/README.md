# FULL100 encoder artifacts

Hai encoder được train bằng LibriSpeech train-clean-100, export và kiểm tra
TFLM host. [Kết quả, protocol và giới hạn](../../docs/full100-results.md).
Chỉ dùng nghiên cứu/host prototype; chưa xác minh triển khai ESP32-S3.

| Thư mục | Objective | FP32 | INT8 |
|---|---|---:|---:|
| `baseline_supervised/` | AAM | 587.132 byte | 154.000 byte |
| `baseline_kd/` | AAM + cosine KD | 587.132 byte | 154.000 byte |

Mỗi thư mục giữ byte-identical `encoder_fp32.tflite`, `encoder_int8.tflite`,
`full_training_report.json`, `full_training_config.json`,
`full_training_freeze.json` và `history.json` từ lần chạy thật.
`evidence/` giữ báo cáo data audit, final audit độc lập, TFLM host và sidecar build.

## Giao diện encoder

- Input: log-mel NHWC `[1,298,40,1]`, cửa sổ PCM mono 16 kHz/48.000 sample;
  frontend tham chiếu trong `src/edgeai/frontend.py`.
- FP32 input/output float32; INT8 input/output int8. Lấy scale/zero-point
  từ metadata của **đúng artifact**, không dùng chung hằng số giữa model.
- Output `[1,192]` là embedding thô; dequantize khi cần, chuẩn hóa L2 và
  tạo template/cosine matching bên ngoài encoder như protocol đã công bố.
- Không có classifier 251 lớp, teacher, microphone driver hoặc firmware
  ứng dụng điểm danh trong FlatBuffer. Không gọi đây là nhận dạng tiếng nói.

## Kiểm tra và tái lập

Model/hash/bytes khai báo nằm trong `full_training_report.json`;
hash gói phát hành nằm trong `release_manifest.json`.
Các path tuyệt đối trong báo cáo được giữ nguyên như chứng cứ lịch sử trên
máy chạy; **không** trỏ tới file trên máy người tải. Để dùng model, lấy path
tương đối trong thư mục này; để train lại, theo
[runbook](../../docs/full100-run.md).

Gói không chứa dataset/WAV/features, teacher weights/embeddings,
initial/best classifier hay training checkpoint/optimizer, cached embedding
NPZ, runner binary, credentials, slide hoặc ảnh. Vì vậy gói phục vụ dùng lại
encoder/kiểm tra báo cáo và tái lập từ input gốc, không phải một checkpoint
bundle có thể tiếp tục đúng optimizer state. Raw report có thể hash-tham
chiếu các file cố ý không phân phối; chúng không bị giả định là hiện diện.

Host arena dùng 38.624 byte trong cap 131.072 byte; không phải tổng RAM
thiết bị. Chưa có board/field/latency/power hoặc independent-final-test claim.
Giới hạn provenance thủ công của `matching.py` và publish/resume boundary
được giữ rõ trong runbook.
