# Phần cứng và môi trường

## Dự trù phần cứng

| Linh kiện | Vai trò | Số lượng | Giá ước lượng |
|-----------|---------|----------|---------------|
| ESP32-S3 N16R8 (DevKitC-1 hoặc bản clone YD-ESP32-S3) | Thiết bị chính, 16MB flash 8MB PSRAM | 1 | ~250k |
| INMP441 | Micro I2S, mua dư một cái phòng hỏng khi hàn | 2 | ~50k/cái |
| INA219 | Đo dòng và điện áp để tính năng lượng | 1 | ~50k |
| Dây cắm, breadboard | | | ~50k |
| Cáp USB-C có truyền dữ liệu | Dùng cáp sẵn có nếu đúng loại | 0-1 | ~40k |
| **Tổng** | | | **~450k** |

Giá là ước lượng thị trường, cần kiểm tra lại khi mua.

Chỉ mua **một bộ**. Phần lớn công việc (huấn luyện, lượng tử hoá, tính EER bản FP32 và INT8) chạy trên PC. Việc đo trên thiết bị do TV2 làm tập trung cho cả ba model, nên một board là đủ. Chỉ mua thêm nếu board hỏng.

**Phải lấy đúng ESP32-S3, không phải ESP32 thường.** S3 có tập lệnh vector và thư viện ESP-NN tối ưu cho suy luận lượng tử hoá, nhanh hơn ESP32 đời cũ nhiều lần. ESP32 thường không có tăng tốc AI nào và sẽ không chạy nổi đồ án này.

Chọn bản có PSRAM (hậu tố R8) vì tensor arena có thể cần tràn sang PSRAM.

Loa để thử tấn công phát lại thì dùng loa điện thoại hoặc laptop có sẵn, không cần mua.

## Phương án dự phòng

Nếu đến cuối tuần 2 chưa thông được đường nhúng thì chuyển sang **Raspberry Pi Zero 2 W**, khoảng 800k-900k, làm bằng Python. Xem điểm quyết định trong [ke-hoach.md](ke-hoach.md).

## Môi trường PC

Cả nhóm dùng chung một môi trường cho số liệu tái lập được.

- python 3.10 trở lên
- torch và torchaudio cho huấn luyện
- speechbrain cho teacher ECAPA-TDNN pretrained
- numpy, scipy cho xử lý tín hiệu
- scikit-learn cho metric
- tensorflow hoặc onnxruntime cho bước lượng tử hoá và xuất TFLite
- matplotlib để vẽ đường DET và biểu đồ

Cố định phiên bản trong `requirements.txt` ngay khi bắt đầu code, tránh lệch kết quả giữa các máy.

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## GPU

Huấn luyện ba đường nén cần GPU. Chạy teacher để suy luận thì CPU cũng được nhưng chậm.

Nếu không ai có GPU thì dùng Google Colab, mỗi người một phiên, và lưu checkpoint ra Drive vì phiên Colab bị ngắt là mất. Nếu thời lượng GPU không đủ thì rút số người nói dùng để huấn luyện và ghi rõ đã rút bao nhiêu, nhưng giữ nguyên bộ thử.

Ghi cấu hình GPU vào báo cáo vì nó ảnh hưởng tính tái lập.

## Môi trường nhúng

ESP-IDF phiên bản ổn định mới nhất, kèm esp-dsp cho FFT tối ưu và esp-nn cho kernel lượng tử hoá. TFLite Micro qua component esp-tflite-micro của Espressif.

Ghi lại phiên bản ESP-IDF và toolchain, vì số đo độ trễ phụ thuộc trực tiếp vào chúng. Cả nhóm dùng đúng một phiên bản, khác phiên bản là số đo không đặt cạnh nhau được.

## Dữ liệu và dung lượng

VoxCeleb1 bản dev khoảng 30GB, chưa kể MUSAN, RIR và ASVspoof PA. Kiểm tra dung lượng ổ trước khi tải. Dữ liệu để ngoài repo, không commit, chỉ commit script tải và file cấu hình.

## Về dữ liệu giọng nói của người trong nhóm

Phần enroll dùng giọng của chính các thành viên. Ghi rõ trong báo cáo rằng dữ liệu này do thành viên tự nguyện cung cấp, chỉ dùng cho đồ án, và xoá sau khi nộp.

Không thu giọng người ngoài nhóm khi chưa có đồng ý rõ ràng. Không đẩy dữ liệu giọng lên dịch vụ công khai hay commit vào repo.

## Tái lập

Cố định seed ở một chỗ trong cấu hình. Ghi lại phiên bản thư viện thực tế bằng `pip freeze`. Lưu trial list, cấu hình nhiễu và checkpoint của cả ba model.
