# Pipeline chi tiết

Mô tả từng bước và khung mã nguồn, sao cho ai cầm repo cũng chạy lại được ra cùng số liệu.

## Sơ đồ

```
[1] Tải dữ liệu             -> data/
[2] Chuẩn bị audio          -> log-mel, đoạn đã cắt
[3] Chốt bộ thử             -> VoxCeleb1-O + biến thể nhiễu/vang/replay
[4] Teacher                 -> ECAPA-TDNN trên PC, EER mốc trên
[5] Ba đường nén            -> mỗi người một đường, cùng ngân sách
[6] Lượng tử hoá INT8       -> áp chung một quy trình cho cả ba
[7] Xuất và nạp thiết bị    -> TFLite -> ESP32-S3
[8] Đo trên thiết bị        -> EER, độ trễ, RAM, năng lượng
[9] Gộp và so sánh          -> bảng, đường DET, biểu đồ đánh đổi
```

## 1. Tải dữ liệu

VoxCeleb1 cần đăng ký, bản dev khoảng 30GB, tải sớm từ tuần 1. Nếu link gốc hỏng thì dùng mirror trên HuggingFace. MUSAN cho nhiễu nền, RIR cho vang, ASVspoof PA cho replay, cả ba đều tải thẳng được.

## 2. Chuẩn bị audio (`src/data/`)

Đọc audio về 16kHz đơn kênh, chuẩn hoá mức, cắt thành đoạn độ dài cố định. Tính log-mel với thông số cố định cho cả nhóm, ví dụ 40 dải mel, cửa sổ 25ms, bước nhảy 10ms.

Thông số log-mel phải giống hệt nhau giữa PC và thiết bị. Đây là chỗ dễ sai nhất và sai thì không phát hiện ra ngay: model vẫn chạy, embedding vẫn ra, chỉ là EER trên thiết bị tệ hơn trên PC mà không hiểu vì sao. Trước khi tin bất kỳ số nào đo trên thiết bị, phải kiểm tra log-mel tính trên ESP32-S3 khớp với log-mel tính trên PC cho cùng một file audio.

## 3. Chốt bộ thử

Dùng danh sách cặp VoxCeleb1-O gốc. Từ đó sinh ra các biến thể: trộn MUSAN ở vài mức SNR cố định, thêm vang RIR, và tập replay từ ASVspoof PA. Cố định seed lúc trộn nhiễu để ai chạy cũng ra đúng file đó.

Tách một tập val riêng để hiệu chuẩn ngưỡng, người nói trong val không được trùng với bộ thử.

## 4. Teacher

Nạp ECAPA-TDNN pretrained từ SpeechBrain, chạy trên bộ thử, tính EER. Số này phải khớp hợp lý với số đã công bố. Lệch nhiều nghĩa là bước 2 hoặc 3 có vấn đề, dừng lại sửa trước khi đi tiếp. Đây là phép kiểm tra tính đúng đắn của toàn bộ pipeline trước khi đi tiếp.

## 5. Ba đường nén (`src/compress/`)

Cả ba về cùng một kiến trúc student và cùng ngân sách tham số do TV3 chốt, nhắm khoảng 300KB sau khi lượng tử hoá INT8.

**KD (TV1)**: ép embedding của student bám embedding của teacher trên cùng đoạn audio, kết hợp với mất mát phân loại người nói. Cần cân bằng hai thành phần mất mát.

**Prune + FT (TV2)**: tỉa kênh trên teacher theo mức quan trọng, rồi tinh chỉnh lại. Tỉa từng mức một và đo lại sau mỗi mức, đừng tỉa thẳng xuống mục tiêu.

**Compact-scratch (TV3)**: huấn luyện thẳng kiến trúc student với hàm mất mát margin góc, không dùng teacher. Mốc dưới.

Cả ba lưu checkpoint để sinh lại kết quả.

## 6. Lượng tử hoá INT8

Lượng tử hoá sau huấn luyện với tập hiệu chuẩn lấy từ train, không lấy từ bộ thử. Quy trình do TV2 quy định và cả ba dùng y hệt, khác quy trình là hỏng so sánh.

Giữ lại cả bản FP32 lẫn bản INT8 của từng đường. Không giữ bản FP32 thì không tách được phần mất do lượng tử hoá, mà đó là kết luận chính của đồ án.

## 7. Xuất và nạp thiết bị (`firmware/`)

Xuất TFLite, nhúng vào firmware để trọng số nằm trong flash chứ không chiếm SRAM. Tensor arena cấp trong SRAM, phần nào tràn thì đẩy sang PSRAM và ghi rõ vì PSRAM chậm hơn nên ảnh hưởng độ trễ.

Đường tín hiệu trên thiết bị: I2S đọc từ INMP441, gom theo khung, tính log-mel dạng streaming bằng FFT của ESP-DSP, đẩy qua TFLite Micro, ra embedding, so cosine với mẫu đã đăng ký.

Bắt buộc làm dạng streaming. Ba giây audio 16kHz dạng float chiếm 192KB, vượt ngân sách SRAM, nên không thể đệm cả câu rồi mới xử lý.

## 8. Đo trên thiết bị

Đo EER on-device bằng cách phát lại các file của bộ thử qua loa vào mic, hoặc nạp thẳng mảng mẫu vào firmware để tách phần sai do micro ra khỏi phần sai do model. Nên làm cả hai và báo cáo riêng: nạp thẳng cho biết model mất gì, phát qua loa cho biết cả hệ thống mất gì.

Độ trễ và năng lượng đo bằng cách chạy lặp nhiều lần rồi báo cáo phân bố.

## 9. Gộp và so sánh (`src/eval/`)

Gom kết quả ba đường ở hai trạng thái và bốn điều kiện, xuất bảng và đường DET. Vẽ thêm biểu đồ đánh đổi, ví dụ EER theo kích thước model, hoặc EER theo năng lượng mỗi lần suy luận.

## Khung thư mục

```
src/
  data/       chuẩn bị audio, log-mel, trial list, trộn nhiễu
  compress/   ba đường nén: kd, prune, scratch, và kiến trúc student chung
  eval/       điểm cosine, EER, minDCF, DET, gộp kết quả
firmware/
  main/       I2S, log-mel streaming, TFLite Micro, so khớp
  models/     file TFLite đã nhúng
```

Các thư mục đang để trống, điền mã theo phân công ở [phan-cong.md](phan-cong.md).
