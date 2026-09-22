# IT2058 - Đồ án: So sánh ba phương pháp nén mô hình xác thực người nói trên vi điều khiển ESP32-S3

Đồ án môn IT2058. Nhóm lấy một mô hình xác thực người nói cỡ lớn đã được huấn luyện sẵn, nén xuống cỡ chạy được trên vi điều khiển ESP32-S3, rồi đo xem mỗi cách nén làm mất bao nhiêu độ chính xác và đổi lại được bao nhiêu về kích thước, độ trễ và điện năng.

## Đề tài

Xác thực người nói là bài toán open-set: hệ thống phải nhận ra người đã đăng ký và từ chối người lạ chưa từng thấy lúc huấn luyện. Các mô hình tốt hiện nay như ECAPA-TDNN có hàng chục triệu tham số, chạy trên máy chủ. Vi điều khiển chỉ có 512KB SRAM và 240MHz.

Nhóm so sánh ba đường đưa mô hình xuống cỡ vài trăm KB: chưng cất tri thức, tỉa kênh, và huấn luyện thẳng một mô hình nhỏ. Cả ba cùng được lượng tử hoá INT8, cùng nạp lên một thiết bị, cùng đo trên một bộ thử.

Điểm nhóm muốn làm rõ: literature về nén mô hình hầu hết báo cáo accuracy, trong khi bài toán open-set kết luận bằng cách so điểm tương đồng với **một ngưỡng cố định**. Nhiễu lượng tử hoá không chỉ làm giảm chất lượng, nó còn **dịch điểm vận hành**, khiến ngưỡng đã hiệu chuẩn trước khi nén không còn đúng sau khi nén. Đây là thứ nhóm đo.

Đồ án không tuyên bố xây dựng một hệ thống kiểm soát ra vào. Sản phẩm là một nghiên cứu đo lường kèm demo minh hoạ.

## Dataset

VoxCeleb1 cho huấn luyện và cho bộ thử chuẩn VoxCeleb1-O. MUSAN và RIR cho nhiễu nền và vang. ASVspoof 2017/2019 PA cho tấn công phát lại. Tất cả đều công khai và có baseline đã công bố để đối chiếu.

## Phần cứng

ESP32-S3 (16MB flash, 8MB PSRAM) với micro I2S INMP441. Đo điện năng bằng INA219. Chi tiết và dự trù kinh phí xem [docs/phan-cung-moi-truong.md](docs/phan-cung-moi-truong.md).

## Cách chạy

1. Dựng môi trường PC và ESP-IDF, lắp phần cứng.
2. Tải VoxCeleb1, MUSAN, ASVspoof PA về `data/`.
3. Chạy teacher ECAPA-TDNN trên PC, tái lập EER trên VoxCeleb1-O để có mốc trên.
4. Mỗi người dựng đường nén của mình theo [interface dùng chung](docs/pipeline.md).
5. Lượng tử hoá INT8, xuất TFLite, nạp lên ESP32-S3.
6. Đo trên thiết bị: EER, độ trễ, RAM, năng lượng. Gộp kết quả và so sánh.

## Cấu trúc repo

```
docs/   kế hoạch, phân công, pipeline, đánh giá, phần cứng và môi trường, dàn ý báo cáo, tài liệu tham khảo
data/   nơi đặt dataset (không commit), kèm hướng dẫn tải
src/    mã nguồn PC: data/ chuẩn bị dữ liệu, compress/ ba đường nén, eval/ đo và gộp kết quả
firmware/  mã nguồn ESP-IDF chạy trên thiết bị
tests/  kiểm thử nhanh
```

## Nhóm thực hiện

| Vai | Họ tên | MSSV | Đường nén phụ trách | Phần chung |
|-----|--------|------|---------------------|------------|
| TV1 | Trần Tú Quang | 250201084 | Chưng cất tri thức (KD) | Dữ liệu, bộ thử, giao thức nhiễu và replay |
| TV2 | Tô Huỳnh Minh Tiến | 250201095 | Tỉa kênh + tinh chỉnh | Pipeline nhúng trên ESP32-S3, đo trên thiết bị |
| TV3 | Nguyễn Ấn | 250201042 | Mô hình nhỏ train thẳng | Teacher, khung huấn luyện, tính metric và gộp kết quả |

Phân công chi tiết xem [docs/phan-cong.md](docs/phan-cong.md).

## Tiến độ

Làm trong 6 tuần. Mốc từng tuần và giả định về lịch xem [docs/ke-hoach.md](docs/ke-hoach.md).
