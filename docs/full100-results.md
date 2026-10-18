# FULL100: kết quả host sau phục hồi ngày 18/10/2026

## Dữ liệu và quy trình

Huấn luyện trên toàn bộ **LibriSpeech train-clean-100**: 251 người nói,
28.539 bản ghi, 100,5908796528 giờ âm thanh nguồn. Chia liên tiếp thành
135.438 cửa sổ 3 giây/16 kHz, không overlap, zero-pad phần đuôi. Không phải
toàn bộ LibriSpeech khoảng 1.000 giờ. Kiểm kê và replay độc lập nằm trong
[data audit](../models/full100-20261015/evidence/independent_data_audit.json).

Validation riêng từ dev-clean: 5 người không trùng train, 60 bản ghi;
3 enrollment và 9 query/người, tạo 45 target và 180 non-target trials.
Tập này đã được tái sử dụng và dùng chọn checkpoint; đây **không phải test
độc lập**. Không chạy calibration/test, không đo FAR mục tiêu trên dữ liệu mới.

Cùng encoder Conv8 → ReLU → AvgPool → Flatten → Dense192 và initialization;
head AAM 251 lớp chỉ dùng trong training. Supervised dùng AAM; KD dùng
AAM + cosine KD trọng số 1,0, teacher SpeechBrain ECAPA-TDNN cố định.
Seed 20261012; Adam 0,001; AAM margin 0,2, scale 30; tối đa 80 epoch,
patience 15. Chọn EER validation nhỏ nhất, giữ epoch sớm nhất khi hòa.
Mỗi epoch dùng đủ 135.438 hàng một lần, 6.772 batch, batch cuối 18 hàng.

## Kết quả validation

| Phương pháp | Epoch chạy / chọn | FP32 EER | INT8 EER | FP32 top-1 | INT8 top-1 |
|---|---:|---:|---:|---:|---:|
| Supervised AAM | 19 / 4 | 4,4444% | 4,4444% | 43/45 | 43/45 |
| KD + AAM | 20 / 5 | 2,7778% | 4,1667% | 45/45 | 45/45 |

Nguồn: báo cáo gốc [supervised](../models/full100-20261015/baseline_supervised/full_training_report.json)
và [KD](../models/full100-20261015/baseline_kd/full_training_report.json).
Top-1 là chẩn đoán closed-set trên năm người validation, không phải tỷ lệ
điểm danh thành công ngoài thực tế hoặc tiêu chí chọn checkpoint.

KD tốt hơn ở FP32 trên validation này; sau INT8, chênh lệch EER so với
supervised chỉ còn khoảng **0,28 điểm phần trăm**. Không có nhiều seed,
confidence interval hoặc test độc lập để kết luận ưu thế tổng quát.
Ngưỡng EER được tìm riêng cho mỗi artifact, không phải ngưỡng triển khai:

| Phương pháp | FP32 threshold | INT8 threshold |
|---|---:|---:|
| Supervised AAM | 0,9985509946740802 | 0,9985674580791044 |
| KD + AAM | 0,6196482862190023 | 0,613053801900957 |

[Audit độc lập](../models/full100-20261015/evidence/independent_final_audit.json)
dùng NumPy độc lập để tái dựng từ embedding cache, khớp EER/threshold/top-1
của cả bốn artifact. Audit còn đối soát ID/order, hash, checkpoint,
initialization và host bindings; không chạy lại training hoặc inference.
Kết luận `PASS_WITH_CAVEATS`, giữ nguyên các giới hạn bên dưới.

## Model và TFLM host

Hai model có cùng tài nguyên kiến trúc: FP32 **587.132 byte**, INT8
**154.000 byte** (khoảng 150,39 KiB), **360.480 Conv/Dense MACs** mỗi encoder
inference. MACs không gồm frontend FFT/log-mel, pooling, matching hay firmware.
Input INT8 `[1,298,40,1]`, output INT8 `[1,192]`; frontend ngoài encoder.

Cả hai [host supervised](../models/full100-20261015/evidence/host_supervised.json)
và [host KD](../models/full100-20261015/evidence/host_kd.json) đều `PASS`,
`TFLM_HOST_PASS`, build provenance `BUILD_BOUND`, TFLM commit
`230c9d9531bd9b08d445b4f2ed0760756ffd5f78`.
Mỗi model có ba fixture synthetic và năm cửa sổ train thật; cached parity
với desktop `BUILTIN_REF_NO_DELEGATES` tối đa **1 LSB**. Đây là kiểm tra số
học trên fixture đã chọn, không phải đo accuracy trên toàn corpus bằng TFLM.

Arena host đã dùng **38.624 byte**, trong cap cấp phát **131.072 byte**.
Không đồng nghĩa tổng SRAM của board, không bao gồm mọi buffer/frontend/app.
Đây là Google TFLM C++ chạy trên Mac, không phải giả lập ESP32 hay firmware
đã chạy trên thiết bị. Chưa đo latency, công suất, field test hoặc microphone.
Trạng thái: `EXPERIMENTAL_HOST_ONLY`, `UNVERIFIED_HARDWARE`.

## Phục hồi và provenance

KD phục hồi optimizer/model/history từ checkpoint epoch 13, chạy lại epoch
14 chưa hoàn tất; không đổi seed, hyperparameter, nguồn runtime hoặc dữ liệu.
Cả hai phương pháp đã early-stop theo cấu hình. Bản phát hành giữ nguyên
model và báo cáo gốc, không sửa các giá trị hay path trong báo cáo.

Hai giới hạn được chấp nhận, không bị che bởi kết quả hoàn tất:

- Publish/status crash boundary cần `MANUAL_RECONCILIATION_REQUIRED`, không
  tuyên bố tự động resume an toàn qua mọi thời điểm publish.
- `matching.py` thiếu trong danh sách hash tự động; pin quan sát ngày 18/10
  không chứng minh lịch sử file từ launch đầu. Chi tiết/hash tại
  [runbook](full100-run.md#giới-hạn-provenance-nguồn-đối-soát-thủ-công).

Dataset, teacher weights/embeddings, cache, checkpoint, embedding NPZ và
slide/media không nằm trong gói model công bố. Muốn train lại cần input
được xác minh và hai môi trường theo [runbook](full100-run.md).

