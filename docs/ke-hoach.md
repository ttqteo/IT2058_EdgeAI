# Kế hoạch đồ án

## Mục tiêu

Nén một mô hình xác thực người nói xuống cỡ chạy được trên vi điều khiển, rồi đo mức suy giảm mà mỗi phương pháp nén gây ra. Sản phẩm của đồ án là bảng số liệu trả lời ba câu hỏi sau, không phải một hệ thống kiểm soát ra vào hoàn chỉnh:

- Với cùng ngân sách khoảng 300KB, ba đường nén (chưng cất, tỉa kênh, train thẳng mô hình nhỏ) cho EER khác nhau thế nào, và cách nào đáng công nhất.
- Lượng tử hoá INT8 làm **dịch điểm vận hành** của bài toán open-set bao nhiêu, tức ngưỡng đã hiệu chuẩn trước khi lượng tử hoá còn dùng được không, và hiệu chuẩn lại trên tập val có lấy lại được phần đã mất không.
- Dưới nhiễu nền và tấn công phát lại, ba đường nén suy giảm có giống nhau không, hay có đường nào bền hơn hẳn.

Câu thứ hai là phần nhóm muốn đóng góp. Literature về nén mô hình gần như chỉ báo cáo accuracy, mà accuracy thì bền với nhiễu lượng tử hoá vì `argmax` không đổi khi điểm số xê dịch chút ít. Bài toán open-set thì khác: kết luận bằng cách so điểm tương đồng với một ngưỡng cố định, nên điểm số xê dịch là kết quả đổi.

## Phạm vi

Làm trên VoxCeleb1 với bộ thử chuẩn VoxCeleb1-O, teacher là ECAPA-TDNN pretrained từ SpeechBrain. Ba đường nén về cùng một ngân sách tham số, cùng lượng tử hoá INT8, cùng nạp lên ESP32-S3, cùng đo trên một bộ thử.

Không huấn luyện mô hình lớn từ đầu, dùng teacher pretrained. Không làm speech recognition, chỉ làm speaker verification. Không tuyên bố đây là hệ thống kiểm soát ra vào đã được kiểm chứng; demo mở cửa cuối buổi bảo vệ chỉ là minh hoạ. Không thu giọng người ngoài nhóm khi chưa có đồng ý bằng văn bản.

## Pipeline tổng quát

```
VoxCeleb1 + MUSAN/RIR + ASVspoof PA
  -> chuẩn bị dữ liệu, trial list cố định, các mức nhiễu
  -> [teacher] ECAPA-TDNN trên PC, tái lập EER  => mốc trên
  -> ba đường nén, mỗi người một đường, cùng ngân sách tham số
  -> lượng tử hoá INT8, xuất TFLite            => áp như nhau cho cả ba
  -> nạp lên ESP32-S3, đo trên thiết bị
  -> gộp: EER, minDCF, kích thước, độ trễ, RAM, năng lượng
```

Chi tiết từng bước xem [pipeline.md](pipeline.md), cách đo xem [danh-gia.md](danh-gia.md).

## Mốc thời gian

**Giả định về lịch:** sáu tuần, bắt đầu 27/09/2026, hạn nộp khoảng 08/11/2026. Nếu lịch thật khác thì chỉ cần dịch các mốc, thứ tự công việc giữ nguyên.

Lưu ý: bốn tuần đầu chồng lấn với đồ án IT3003 (hạn 25/10). Nhóm nên dồn phần nặng về tư duy của đồ án này vào tuần 5-6, và ba tuần đầu chỉ làm những việc cơ học như tải dữ liệu, dựng môi trường, chạy lại teacher.

| Tuần | Việc chính | Đầu ra |
|------|------------|--------|
| 1 | Nhận phần cứng, dựng môi trường PC và ESP-IDF, tải dữ liệu | Môi trường chạy được, dữ liệu về máy, mic đọc được tín hiệu |
| 2 | Chốt trial list và giao thức, chạy teacher lấy mốc trên, **đưa một model rỗng chạy thông trên ESP32-S3** | Số teacher, giao thức cố định, đường nhúng đã thông |
| 3 | Mỗi người dựng và huấn luyện đường nén của mình | Ba model FP32 cùng ngân sách, có số EER trên PC |
| 4 | Lượng tử hoá INT8, xuất TFLite, nạp lên thiết bị | Ba model chạy trên ESP32-S3, có EER on-device |
| 5 | Chạy lưới nhiễu và replay, đo năng lượng, gộp kết quả | Bảng so sánh đầy đủ, đường DET, biểu đồ |
| 6 | Viết báo cáo, rà soát, chuẩn bị demo và nộp | Báo cáo hoàn chỉnh |

## Điểm quyết định cuối tuần 2

Đây là chốt chặn quan trọng nhất của đồ án. **Nếu cuối tuần 2 chưa đưa được một model bất kỳ chạy thông trên ESP32-S3** thì đừng cố nữa, chuyển sang Raspberry Pi Zero 2 W và làm bằng Python.

Toàn bộ phần nghiên cứu (ba đường nén, lượng tử hoá, dịch ngưỡng, nhiễu, replay) giữ nguyên không đổi, chỉ mất phần "ràng buộc khắc nghiệt". Đổi sớm thì mất một tuần, đổi ở tuần 4 thì mất cả đồ án.

Lý do đặt mốc này: phần khó của đồ án không nằm ở machine learning mà nằm ở lập trình nhúng C. Phải biết sớm mình có qua được cửa đó không.

## Sản phẩm nộp

Mã nguồn PC trong `src/` và firmware trong `firmware/`, chạy lại được từ đầu. Bảng kết quả và biểu đồ so sánh. Báo cáo theo dàn ý ở [bao-cao-outline.md](bao-cao-outline.md). README ghi rõ cách tái lập, kèm cấu hình phần cứng và phiên bản toolchain.

## Rủi ro

**Tải VoxCeleb1 nặng và link hay chết.** Bản dev khoảng 30GB. Tải sớm từ tuần 1, có bản mirror trên HuggingFace nếu link gốc hỏng. Nếu vẫn không đủ dung lượng thì rút số người nói dùng để huấn luyện và ghi rõ đã rút bao nhiêu, nhưng **giữ nguyên bộ thử VoxCeleb1-O** để số còn so được với literature.

**Model nhỏ cho EER kém.** Nhiều khả năng EER rơi vào khoảng 5-12% trong khi teacher khoảng 1%. Đây là **kết quả**, không phải thất bại. Cả đồ án được đóng khung để con số đó là thứ cần đo, không phải thứ cần giấu. Đừng ép cho ra số đẹp.

**Không đủ RAM trên thiết bị.** Ba giây audio 16kHz dạng float đã là 192KB, không được buffer cả câu. Bắt buộc làm streaming: tính đặc trưng theo từng khung rồi bỏ mẫu thô đi. Nếu vẫn tràn thì hạ số chiều mel hoặc rút ngắn cửa sổ và ghi rõ đã đánh đổi gì.

**Tỉa kênh làm hỏng mô hình.** ECAPA-TDNN có nhánh attention, tỉa ẩu là sập. Nếu TV2 vướng thì hạ mục tiêu tỉa xuống mức nhẹ hơn, hoặc chuyển sang tỉa theo nhóm kênh, và ghi lại đúng mức tỉa đã đạt.

**Nếu kết quả cho thấy không đường nén nào đủ tốt để dùng thật** thì vẫn báo cáo đúng như vậy kèm lý giải. Một kết luận "ngân sách 300KB chưa đủ cho xác thực người nói ở mức tin cậy vận hành, cần ít nhất X" là kết quả có giá trị, hợp với tinh thần đo lường của đồ án.
