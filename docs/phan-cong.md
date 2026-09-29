# Phân công

Chia việc theo đường nén. Mỗi người phụ trách một cách đưa mô hình xuống ngân sách mục tiêu để so sánh, và kiêm thêm một phần chung của pipeline. Làm vậy để cả nhóm dùng chung một teacher, một bộ thử và một thiết bị, số liệu mới đặt cạnh nhau được.

| Vai | Họ tên | MSSV | Đường nén phụ trách | Phần chung phụ trách |
|-----|--------|------|---------------------|----------------------|
| TV1 | Trần Tú Quang | 250201084 | Chưng cất tri thức (KD) | Dữ liệu: VoxCeleb1, trial list, nhiễu MUSAN/RIR, replay ASVspoof |
| TV2 | Tô Huỳnh Minh Tiến | 250201095 | Tỉa kênh + tinh chỉnh | Pipeline nhúng ESP32-S3 và khung đo trên thiết bị |
| TV3 | Nguyễn Ấn | 250201042 | Mô hình nhỏ train thẳng | Teacher, kiến trúc student chung, tính metric và gộp kết quả |

## TV1 - Trần Tú Quang (250201084)

Lo chưng cất tri thức và phần dữ liệu chung.

**Phần chung, cần xong trong tuần 2 vì chặn đường cả nhóm.** Tải VoxCeleb1, dựng bước chuẩn bị dữ liệu trong `src/data/`: đọc audio, chuẩn hoá mức, cắt đoạn, tính log-mel. Chốt bộ thử là danh sách cặp VoxCeleb1-O gốc, không tự chế danh sách khác, để số của nhóm còn so được với các bài đã công bố. Dựng thêm các biến thể của bộ thử: trộn nhiễu MUSAN ở vài mức SNR cố định, thêm vang bằng RIR, và tập replay lấy từ ASVspoof PA. Các mức SNR và cách trộn phải cố định và ghi ra file cấu hình, không ai tự đổi giữa chừng.

**Đường nén của mình.** Chưng cất từ teacher ECAPA-TDNN xuống kiến trúc student mà TV3 chốt. Chưng cất ở mức embedding, tức ép embedding của student bám theo embedding của teacher trên cùng một đoạn audio, thay vì chỉ học nhãn người nói. Thử vài hệ số cân bằng giữa mất mát chưng cất và mất mát phân loại, ghi lại cái nào tốt hơn.

## TV2 - Tô Huỳnh Minh Tiến (250201095)

Lo tỉa kênh và toàn bộ đường nhúng. Đây là phần rủi ro nhất của đồ án nên được ưu tiên về thời gian.

**Phần chung, mốc quan trọng nhất là cuối tuần 2.** Dựng môi trường ESP-IDF, nối micro INMP441 qua I2S, và **đưa một model bất kỳ chạy thông từ đầu đến cuối trên ESP32-S3**, kể cả model trọng số ngẫu nhiên cũng được. Mục đích không phải ra kết quả đúng mà là chứng minh đường đi tồn tại: audio vào, đặc trưng tính được, TFLite Micro nạp được model, embedding ra được. Sau đó mới hoàn thiện phần tính log-mel dạng streaming bằng FFT tối ưu của ESP-DSP, trong ngân sách RAM cho phép.

Định nghĩa luôn quy ước để hai người kia cắm model vào: định dạng file TFLite, cách bố trí tensor vào ra, số chiều embedding, và script nạp model lên thiết bị. Viết khung đo trên thiết bị: độ trễ mỗi lần suy luận, RAM đỉnh, và năng lượng qua INA219.

**Đường nén của mình.** Tỉa kênh trên teacher rồi tinh chỉnh lại về ngân sách mục tiêu. ECAPA-TDNN có nhánh attention nên tỉa ẩu dễ làm sập mô hình, tỉa từ từ và đo lại sau mỗi mức. Nếu không đạt được ngân sách mà mô hình còn dùng được thì ghi lại mức tỉa tối đa thực tế đạt được, đó cũng là kết quả.

## TV3 - Nguyễn Ấn (250201042)

Lo mô hình nhỏ train thẳng và phần đánh giá.

**Phần chung.** Chạy teacher ECAPA-TDNN pretrained từ SpeechBrain trên PC và tái lập EER trên VoxCeleb1-O. Con số này là mốc trên của cả đồ án, phải khớp hợp lý với số đã công bố, lệch nhiều là dấu hiệu pipeline dữ liệu có vấn đề. Chốt kiến trúc student dùng chung cho cả ba đường nén, cùng số tham số và cùng số chiều embedding, để so sánh mới công bằng. Viết khung tính metric trong `src/eval/`: điểm cosine, EER, minDCF, đường DET, và script gộp kết quả từ ba đường thành một bảng.

**Đường nén của mình.** Huấn luyện thẳng kiến trúc student đó trên VoxCeleb1 với hàm mất mát margin góc, không dùng teacher. Đây là mốc dưới, trả lời câu hỏi chưng cất và tỉa kênh có thật sự hơn việc cứ train thẳng một mô hình nhỏ không.

## Phần cả nhóm cùng làm

Tuần 2 làm chung ba việc trước khi chia nhánh: chốt trial list và các mức nhiễu của TV1, lấy số teacher của TV3, và thông đường nhúng của TV2. Không ai bắt đầu đường nén của mình trước khi ba thứ này xong, vì thiếu một trong ba thì số làm ra không so được với ai.

Từ tuần 4, việc lượng tử hoá INT8 mỗi người tự làm cho đường của mình nhưng **dùng chung đúng một quy trình** do TV2 quy định, khác quy trình là hỏng so sánh.

Báo cáo chia theo phần đã làm, mỗi người viết phần đường nén của mình cộng phần chung mình phụ trách, một người ghép lại và thống nhất văn phong.

## Quy ước trong nhóm

Cả nhóm dùng chung một teacher, một kiến trúc student, một trial list và một bộ mức nhiễu, không ai tự đổi. Mọi thí nghiệm chạy với cùng seed ghi trong [danh-gia.md](danh-gia.md). Model xuất ra và kết quả lưu đúng cấu trúc thư mục để script gộp của TV3 chạy được.

Mỗi con số đưa vào báo cáo phải kèm được lệnh và cấu hình đã tạo ra nó. Chạy xong tới đâu ghi số liệu tới đó, đừng để dồn tới cuối.

Nhóm chỉ có một ESP32-S3 và TV2 giữ máy. TV1 và TV3 không cần cầm thiết bị: nộp file TFLite INT8 theo đúng quy ước của TV2, TV2 nạp và đo cả ba model trên cùng một board. Cách này còn giúp số đo công bằng hơn, vì cả ba model đều đo trên cùng một board, cùng firmware và cùng người đo. Muốn kiểm tra EER bản INT8 trước khi gửi thì mỗi người tự chạy trên PC bằng TFLite interpreter.
