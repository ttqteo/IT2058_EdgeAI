# Dàn ý báo cáo

Dàn ý theo cấu trúc một bài nghiên cứu, lúc viết chỉ việc điền số liệu và phân tích.

## 1. Giới thiệu

Nêu bài toán: xác thực người nói là bài toán open-set, mô hình tốt hiện nay cỡ hàng chục triệu tham số và chạy trên máy chủ, trong khi vi điều khiển chỉ có 512KB SRAM. Nêu ba lý do buộc phải xử lý tại thiết bị: dữ liệu sinh trắc không nên rời thiết bị, quyết định phải xong dưới một giây, và thiết bị phải hoạt động khi mất mạng.

Chốt câu hỏi nghiên cứu và đóng góp: so ba đường nén ở cùng ngân sách, và đo mức dịch điểm vận hành do lượng tử hoá gây ra, thứ mà literature về nén mô hình thường không báo cáo vì chỉ nhìn accuracy.

Nói rõ ngay ở đây rằng đồ án là nghiên cứu đo lường, không phải một hệ thống kiểm soát ra vào đã được kiểm chứng.

## 2. Tổng quan tài liệu

Xác thực người nói và các kiến trúc embedding, từ x-vector tới ECAPA-TDNN. Hàm mất mát margin góc. Nén mô hình cho thiết bị biên: chưng cất, tỉa kênh, lượng tử hoá. Suy luận trên vi điều khiển và các ràng buộc của nó. Tấn công phát lại và bộ dữ liệu ASVspoof.

Định vị đồ án: hai bài mẫu của môn đều triển khai trên Jetson Nano, tức lớp máy tính nhúng. Đồ án này xuống một bậc nữa, xuống lớp vi điều khiển, và đổi bài toán từ phân loại closed-set sang verification open-set. Danh sách đầy đủ ở [tai-lieu-tham-khao.md](tai-lieu-tham-khao.md).

## 3. Dữ liệu và giao thức

VoxCeleb1 và bộ thử VoxCeleb1-O. MUSAN, RIR và ASVspoof PA cho các điều kiện nhiễu, vang, replay. Cách tách tập val để hiệu chuẩn ngưỡng. Nhắc lại nguyên tắc người nói trong bộ thử không xuất hiện khi huấn luyện.

## 4. Phương pháp

Kiến trúc student dùng chung và ngân sách tham số. Ba đường nén kèm siêu tham số: chưng cất embedding, tỉa kênh và tinh chỉnh, huấn luyện thẳng. Quy trình lượng tử hoá INT8 dùng chung. Cách hiệu chuẩn ngưỡng.

Mô tả hệ thống nhúng: đường tín hiệu từ I2S tới embedding, cách tính log-mel dạng streaming, cách bố trí bộ nhớ giữa flash, SRAM và PSRAM.

## 5. Thí nghiệm và kết quả

Nhắc lại giao thức từ [danh-gia.md](danh-gia.md). Mốc trên của teacher. Bảng chính: ba đường nén × hai trạng thái FP32/INT8 × bốn điều kiện, cột EER và minDCF. Đường DET. Bảng chỉ số trên thiết bị: kích thước, độ trễ trung vị và phân vị 95, RAM đỉnh, µJ mỗi lần suy luận.

Bảng riêng cho độ dịch ngưỡng: ngưỡng FP32 áp cho INT8 thì FAR và FRR lệch bao nhiêu, hiệu chuẩn lại thì lấy về được bao nhiêu.

Biểu đồ đánh đổi: EER theo kích thước, EER theo năng lượng.

## 6. Thảo luận

Đường nén nào tốt nhất ở ngân sách này và vì sao. Chưng cất và tỉa kênh có đáng công hơn train thẳng không. Phần mất do nén kiến trúc so với phần mất do lượng tử hoá, cái nào lớn hơn.

Bàn kỹ về dịch ngưỡng: vì sao bài toán open-set nhạy với lượng tử hoá hơn bài toán phân loại, và điều đó có ý nghĩa gì với người triển khai hệ thống sinh trắc trên thiết bị biên.

Khoảng cách giữa EER đo bằng cách nạp mẫu thẳng và EER đo qua micro thật, tức micro và môi trường đóng góp bao nhiêu vào sai số.

Hạn chế: ngân sách tham số, quy mô dữ liệu huấn luyện đã rút, số người enroll ít, và việc đo trên một thiết bị duy nhất.

## 7. Kết luận

Trả lời gọn ba câu hỏi ở phần 1. Nếu ngân sách 300KB chưa đủ cho mức tin cậy vận hành thì nói thẳng và ước lượng cần bao nhiêu mới đủ. Nêu hướng làm tiếp nếu có thêm thời gian.

## 8. Phân công và đóng góp

Bảng ai làm gì, lấy từ [phan-cong.md](phan-cong.md).

## Phụ lục

Cách tái lập gồm lệnh chạy, cấu hình, seed, phiên bản thư viện, phiên bản ESP-IDF và sơ đồ nối dây.
