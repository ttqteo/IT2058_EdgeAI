# Giao thức đánh giá

Muốn so sánh được thì mọi thí nghiệm phải dùng chung một teacher, một kiến trúc student, một bộ thử và cùng seed, chỉ thay đổi cách nén.

## Nguyên tắc

Bộ thử là danh sách cặp VoxCeleb1-O gốc, giữ nguyên, không tự chế danh sách khác và không đụng vào. Dùng đúng danh sách gốc thì số của nhóm mới đặt cạnh được số đã công bố của ECAPA-TDNN và x-vector.

Người nói trong bộ thử không được xuất hiện trong tập huấn luyện. Đây là bài toán open-set, vi phạm điều này là toàn bộ số liệu mất nghĩa.

Tập val tách riêng để hiệu chuẩn ngưỡng, không lấy từ bộ thử. Ba đường nén dùng chung đúng một kiến trúc student và đúng một ngân sách tham số.

## Metric

Bài toán là verification chứ không phải phân loại, nên **không dùng accuracy làm metric chính**. Các metric dùng để kết luận:

- **EER**, điểm mà tỉ lệ chấp nhận nhầm và từ chối nhầm bằng nhau. Đây là metric chính.
- **minDCF**, chi phí quyết định nhỏ nhất, phản ánh điểm vận hành thiên về bảo mật.
- **Đường DET**, để thấy toàn bộ dải đánh đổi chứ không chỉ một điểm.
- **Độ dịch ngưỡng** sau khi lượng tử hoá, tức ngưỡng hiệu chuẩn trên model FP32 đem áp cho model INT8 thì FAR và FRR lệch bao nhiêu. Đây là metric riêng của đồ án này.

Bên cạnh đó là nhóm chỉ số trên thiết bị:

- Kích thước model, tính theo KB file TFLite.
- Độ trễ mỗi lần suy luận trên ESP32-S3, báo cáo cả trung vị và phân vị 95.
- RAM đỉnh, tách riêng phần tensor arena và phần đệm tín hiệu.
- Năng lượng mỗi lần suy luận, tính bằng µJ, đo qua INA219.

## Ma trận thí nghiệm

Chạy cùng một lưới cho mọi phương pháp:

| Phương pháp | Mô tả | Người |
|-------------|-------|-------|
| Teacher ECAPA-TDNN | FP32, chạy trên PC, mốc trên | Cả nhóm (tuần 2) |
| Compact-scratch | Kiến trúc student train thẳng, không teacher, mốc dưới | TV3 |
| KD | Chưng cất embedding từ teacher | TV1 |
| Prune + FT | Tỉa kênh từ teacher rồi tinh chỉnh | TV2 |

Mỗi đường chạy ở hai trạng thái, **FP32 và INT8**, để tách được phần mất do nén kiến trúc và phần mất do lượng tử hoá. Đây là điểm mấu chốt, gộp hai phần này lại là mất luôn kết luận chính của đồ án.

Mỗi trạng thái đo ở bốn điều kiện: sạch, nhiễu MUSAN ở vài mức SNR cố định, có vang RIR, và replay từ ASVspoof PA.

Đo trên thiết bị chỉ áp cho bản INT8, vì FP32 không nạp vừa. Bản FP32 đo trên PC để lấy mốc chất lượng.

## Hiệu chuẩn ngưỡng

Đây là phần dễ làm sai nhất nên tách riêng. Với mỗi model, lấy ngưỡng từ tập val chứ không phải từ bộ thử. Với bản INT8, làm hai lần:

- Lấy ngưỡng của bản FP32 đem áp thẳng cho bản INT8, ghi FAR và FRR bị lệch bao nhiêu.
- Hiệu chuẩn lại ngưỡng trên val bằng chính bản INT8, xem lấy lại được bao nhiêu.

Hiệu số giữa hai lần đó chính là câu trả lời cho câu hỏi nghiên cứu thứ hai.

## Tái lập

Cố định seed cho chia dữ liệu, cho huấn luyện và cho việc trộn nhiễu, khai báo ở một chỗ trong cấu hình thay vì rải trong code. Lưu trial list đã chốt, file cấu hình nhiễu, và checkpoint của cả ba model để chạy lại ra đúng số.

Ghi lại phiên bản ESP-IDF và toolchain, vì số đo độ trễ phụ thuộc vào chúng. Khi đo độ trễ và năng lượng thì chạy lặp nhiều lần rồi báo cáo phân bố, đừng lấy một lần đo.

Mỗi con số trong báo cáo phải chỉ ra được lệnh và cấu hình đã tạo ra nó.

## Đọc kết quả

So ba đường nén với nhau ở cùng ngân sách, xem cách nào cho EER thấp nhất. So từng đường với teacher để biết mất bao nhiêu. So với compact-scratch để biết chưng cất và tỉa kênh có đáng công hơn việc cứ train thẳng không.

So bản FP32 với bản INT8 của cùng một đường, tách phần mất do lượng tử hoá ra khỏi phần mất do nén kiến trúc. Rồi xem độ dịch ngưỡng, và hiệu chuẩn lại có bù được không.

Cuối cùng đặt các con số trên thiết bị cạnh chất lượng: nếu một đường EER tốt hơn 1% nhưng tốn gấp đôi năng lượng thì nói rõ đánh đổi đó, đừng chỉ xếp hạng theo EER.

Nếu không đường nén nào đạt mức dùng được trong thực tế thì vẫn báo cáo đúng như vậy kèm ước lượng cần ngân sách bao nhiêu mới đủ. Không ép số.
