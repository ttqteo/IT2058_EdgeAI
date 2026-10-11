# IT2058 - Đồ án: Hệ thống điểm danh bằng giọng nói trên vi điều khiển ESP32-S3 dùng mô hình nén

Đồ án môn IT2058. Sinh viên đứng trước một thiết bị nhỏ đặt ở lớp, nói một câu ngắn ("Em là [họ tên], có mặt."), thiết bị nhận ra là ai, hiện tên lên màn hình và báo về sổ điểm danh trên laptop của giảng viên. Toàn bộ phần xử lý giọng nói chạy ngay trên vi điều khiển ESP32-S3, không gửi giọng nói lên mạng. Để nạp được vào vi điều khiển, nhóm nén một mô hình xác thực người nói cỡ lớn xuống khoảng 300KB INT8 và đo xem đổi lại mất bao nhiêu độ chính xác.

![Tình huống điểm danh](slides/diagrams/scenario.svg)

## Đề tài

Bài mẫu của môn điểm danh bằng khuôn mặt trên Jetson Nano. Nhóm làm cùng bài toán bằng giọng nói, trên ESP32-S3 (khoảng 200k, 512KB SRAM, 240MHz), thấp hơn Jetson cả một lớp thiết bị. Hệ thống có hai chế độ: nhận diện 1:N (chỉ cần nói, so với cả lớp), và xác thực 1:1 (giảng viên nhập MSSV trên laptop rồi sinh viên nói) làm dự phòng.

Teacher là ECAPA-TDNN pretrained từ SpeechBrain, khoảng 20 triệu tham số, chỉ chạy trên PC. Nhóm so hai cách đưa xuống ngân sách khoảng 300KB: chưng cất tri thức (KD) và train thẳng một model nhỏ. Cả hai viết bằng Keras, lượng tử hoá INT8 qua TFLite, model nào tốt hơn thì nạp vào ESP32-S3 chạy bằng TFLite Micro và ESP-NN.

Câu hỏi đo lường nhóm giữ lại: hệ thống quyết định bằng cách so điểm tương đồng với **một ngưỡng cố định**. Trên ESP32-S3, INT8 là bắt buộc. Ngưỡng chọn trong phòng lab có thể lệch khi lượng tử hoá từ FP32 sang INT8, và khi chuyển từ dữ liệu chuẩn sang giọng Việt qua micro rẻ trong phòng học thật. Nhóm đo mức lệch đó, và xem hiệu chỉnh lại bằng dữ liệu đăng ký bù được bao nhiêu.

Phạm vi, cách xử lý, bộ ràng buộc chung cho model và firmware, cách đánh giá xem [docs/bai-toan-diem-danh.md](docs/bai-toan-diem-danh.md). Đồ án không dùng cho mục đích an ninh và không chống giả mạo giọng nói. Mức dễ bị qua mặt bằng ghi âm được đo và báo cáo như một hạn chế.

## Dataset

VoxCeleb1 để huấn luyện, và bộ thử chuẩn VoxCeleb1-O để so với các bài đã công bố. MUSAN cho nhiễu nền. Dữ liệu thực tế tự thu từ 15–20 người tự nguyện có giấy đồng ý, không commit vào repo và xoá sau khi nộp.

## Phần cứng

ESP32-S3 N16R8 (16MB flash, 8MB PSRAM), micro INMP441 qua I2S, màn hình OLED 0,96", nút nhấn và buzzer, INA219 để đo năng lượng. Tổng khoảng 500k. Laptop của nhóm chạy app sổ điểm danh.

**Mốc chặn 20/10:** nếu chưa có board, chưa đọc được micro, hoặc chưa chạy được một model TFLite bất kỳ trên board thì chuyển sang Raspberry Pi Zero 2 W. Phần nghiên cứu giữ nguyên.

## Cách chạy

1. Dựng môi trường PC (Python, TensorFlow, SpeechBrain) và ESP-IDF, lắp phần cứng.
2. Tải VoxCeleb1 và MUSAN về `data/`.
3. Chạy teacher ECAPA-TDNN trên PC, tái lập EER trên VoxCeleb1-O, tính sẵn embedding teacher cho tập train.
4. Train hai student bằng Keras (KD và train thẳng), lượng tử hoá INT8 bằng TFLite converter.
5. Nạp model tốt hơn vào firmware ESP32-S3, chạy app sổ điểm danh trên laptop.
6. Đăng ký giọng, chạy thử nghiệm thực tế, gộp kết quả.

## Cấu trúc repo

```
docs/   bài toán điểm danh, kế hoạch, phân công, pipeline, đánh giá, phần cứng và môi trường, dàn ý báo cáo, tài liệu tham khảo
data/   nơi đặt dataset (không commit), kèm hướng dẫn tải
src/    mã nguồn PC: data/ chuẩn bị dữ liệu và log-mel chuẩn, compress/ KD và train thẳng, eval/ đo và gộp kết quả
firmware/  mã nguồn ESP-IDF chạy trên ESP32-S3
tests/  kiểm thử nhanh
```

## Nhóm thực hiện

| Vai | Họ tên | MSSV | Phụ trách |
|-----|--------|------|-----------|
| TV1 | Trần Tú Quang | 250201084 | Log-mel bản chuẩn, dữ liệu VoxCeleb1 và MUSAN, student KD, lượng tử hoá INT8, chọn và hiệu chỉnh ngưỡng |
| TV2 | Tô Huỳnh Minh Tiến | 250201095 | Firmware ESP32-S3: thu âm, log-mel bằng C, TFLite Micro, so khớp, OLED, giao tiếp với laptop, đo độ trễ và năng lượng |
| TV3 | Nguyễn Ấn | 250201042 | Teacher, student train thẳng, app sổ điểm danh trên laptop, khung tính metric, thử nghiệm thực tế, gộp kết quả |

## Tiến độ

Còn 4 tuần, từ 12/10 tới hạn nộp khoảng 08/11/2026. Lịch từng tuần xem [docs/bai-toan-diem-danh.md](docs/bai-toan-diem-danh.md#phân-công-và-lịch).

Các file `docs/ke-hoach.md`, `docs/phan-cong.md`, `docs/pipeline.md`, `docs/danh-gia.md`, `docs/phan-cung-moi-truong.md` và `docs/bao-cao-outline.md` viết cho hướng cũ (so sánh ba cách nén, không có ứng dụng điểm danh), đang chờ cập nhật.
