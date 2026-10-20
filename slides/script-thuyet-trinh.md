# Kịch bản thuyết trình EdgeAI theo 14 slide

Cập nhật ngày 20/10/2026, dùng cùng [deck hiện tại](slide.html). Nguyễn Ấn nói toàn bộ và tự thao tác demo, đại diện nhóm. Không chia lời cho thành viên khác.

Thời lượng tập nói: khoảng 12 phút 55 giây, gồm khoảng 2 phút thao tác demo. Đây là dự kiến, cần bấm giờ khi tập. Chỉ đọc phần “Lời nói”. Thao tác, ghi chú và Q&A dành cho chuẩn bị.

Mạch trình bày: phát biểu bài toán và input/output, câu hỏi về KD và INT8, thiết kế so sánh, kết quả validation, kiểm chứng triển khai trên ESP32-S3, kết luận và hướng tiếp theo. Lưu mẫu sau reset là kiểm chứng kỹ thuật, không đặt thành câu hỏi nghiên cứu riêng.

## Slide 1 — Điểm danh bằng giọng nói trên ESP32-S3

Thời gian dự kiến: 20 giây.

### Lời nói

“Em chào Thầy và các bạn. Em là Nguyễn Ấn, đại diện nhóm trình bày đề tài điểm danh bằng giọng nói trên ESP32-S3.

Nhóm khảo sát ảnh hưởng của knowledge distillation và lượng tử hóa INT8 lên một model nhận diện người nói nhỏ, sau đó kiểm tra khả năng triển khai model trên thiết bị thật.”

## Slide 2 — Phát biểu bài toán

Thời gian dự kiến: 45 giây.

### Lời nói

“Bài toán của nhóm là: cho một đoạn giọng nói và các mẫu đã đăng ký, xác định người nói phù hợp nhất.

Đầu vào gồm query ba giây, chuẩn hóa về mono 16 kilohertz, cùng các mẫu đặc trưng có nhãn. Đầu ra của hệ thống là tên có cosine cao nhất và điểm tương đồng với từng mẫu.

Cần phân biệt với đầu ra của model: encoder nhận tensor INT8 kích thước 298 nhân 40 và trả embedding 192 chiều. Board chuẩn hóa embedding rồi so cosine để chọn tên. Mac đảm nhiệm thu âm và frontend.

Demo hiện chọn trong Quang, Tien và An. Nhóm chưa có ngưỡng từ chối người lạ, và điểm cosine không phải xác suất nhận đúng.”

## Slide 3 — Câu hỏi nghiên cứu

Thời gian dự kiến: 40 giây.

### Lời nói

“Câu hỏi chính của nhóm là: với cùng kiến trúc student và dữ liệu huấn luyện, học thêm từ teacher ảnh hưởng thế nào đến chất lượng xác minh người nói sau khi chuyển sang INT8, so với chỉ học từ nhãn?

Nhóm so sánh supervised AAM và KD cộng AAM. Mỗi nhánh có bản FP32 và INT8 để quan sát riêng tác động của lượng tử hóa.

Kết quả nghiên cứu được đọc trên validation. Phần demo ESP32-S3 bổ sung bằng chứng rằng encoder đã xuất có thể chạy và tham gia luồng nhận dạng thật. Việc lưu mẫu trong NVS là một phần của triển khai đó.”

## Slide 4 — Thiết kế thực nghiệm

Thời gian dự kiến: 75 giây.

### Lời nói

“Để phép đối chiếu tập trung vào mục tiêu huấn luyện, nhóm giữ cùng dữ liệu, frontend, kiến trúc baseline_conv8 và embedding 192 chiều.

Ở nhánh supervised, student học phân biệt nhãn người nói bằng AAM. Ở nhánh KD cộng AAM, student vừa học nhãn vừa học cách tạo embedding gần với embedding của teacher.

Công thức trên slide là mất mát AAM. Ý nghĩa chính là thêm một biên góc vào lớp đúng trong quá trình huấn luyện, để yêu cầu phân biệt lớp chặt hơn. Trong cấu hình này, biên m bằng 0,2 radian, hệ số s bằng 30 và có 251 lớp người nói.

Phần KD khuyến khích embedding student cùng hướng với teacher bằng mục tiêu cosine. Teacher chỉ chạy offline, không tham gia lúc demo.

Sau huấn luyện, nhóm bỏ head phân lớp và xuất encoder FP32 cùng encoder INT8. Vì kiến trúc không thay đổi giữa hai nhánh, nhóm có thể đối chiếu ảnh hưởng của cách học và lượng tử hóa mà không trộn thêm ảnh hưởng do đổi model.”

### Ghi chú

Không đọc công thức thành từng ký hiệu. Nếu Thầy hỏi sâu, giải thích AAM dùng nhãn, còn KD dùng embedding teacher. Đây là một phép đối chiếu thực nghiệm, chưa phải kết luận tổng quát hoặc đánh giá nhiều seed.

## Slide 5 — Dữ liệu và cách chia

Thời gian dự kiến: 35 giây.

### Lời nói

“Nhóm dùng LibriSpeech train-clean-100 để huấn luyện, gồm 251 người nói và 28.539 bản ghi.

Tập đánh giá có 5 người nói khác với tập huấn luyện. Với mỗi người, nhóm dùng 3 mẫu để đăng ký giọng nói và 9 mẫu còn lại để kiểm tra. Như vậy có tổng cộng 45 mẫu kiểm tra.

Các kết quả ở phần sau đều trên validation. Nhóm chưa đánh giá trên tập test độc lập.”

## Slide 6 — Teacher và student

Thời gian dự kiến: 50 giây.

### Lời nói

“Teacher nhóm dùng là ECAPA-TDNN của SpeechBrain, checkpoint spkrec-ecapa-voxceleb đã huấn luyện sẵn.

Student là baseline_conv8. Tên này là định danh kiến trúc trong code của nhóm, không có nghĩa model gồm tám lớp convolution. Encoder này dùng một lớp convolution tám kênh, pooling và lớp chiếu ra embedding 192 chiều.

Đầu vào là đặc trưng kích thước 298 nhân 40, một kênh. Đầu ra là vector biểu diễn giọng nói để so sánh với template.

Tên VoxCeleb thuộc checkpoint teacher. Student trong thí nghiệm này vẫn học trên LibriSpeech. Khi triển khai, board chỉ chạy encoder student INT8 của nhánh KD cộng AAM, không chứa teacher hay AAM head.”

## Slide 7 — Quantization và export

Thời gian dự kiến: 35 giây.

### Lời nói

“Nhóm chuyển encoder từ FP32 sang INT8 sau huấn luyện. Cả hai nhánh đều giảm từ 587.132 xuống 154.000 byte, nhỏ hơn khoảng 3,8 lần.

Khi demo, Mac gửi đặc trưng log-mel INT8 sang ESP32. Board chạy encoder, đổi embedding đầu ra về số thực rồi chuẩn hóa để lưu mẫu hoặc so cosine.

Kết quả full100 ở phần sau được đo trên máy tính với tập validation.”

## Slide 8 — Sơ đồ thử nghiệm

Thời gian dự kiến: 60 giây.

### Lời nói

“Sơ đồ này thể hiện phần nào chạy trên Mac, phần nào chạy trên ESP32-S3.

Khi đăng ký, Mac lấy audio và tính frontend, sau đó lượng tử hóa thành tensor INT8 rồi gửi qua USB. Board chạy encoder, chuẩn hóa embedding và lưu vector cùng nhãn vào NVS. Board lưu mẫu đặc trưng, không lưu file âm thanh gốc.

Khi query, Mac thu âm mới hoặc đọc file kiểm tra rồi thực hiện cùng frontend. Board nhận tensor, chạy encoder và so cosine với các template đã nạp từ NVS. Board chọn nhãn có điểm cao nhất, sau đó trả kết quả để Mac hiển thị.

Hiện có ba mẫu Quang, Tien và An. Query bình thường không ghi lại hay cập nhật các mẫu này.

Do chưa có microphone riêng cho board, frontend âm thanh vẫn ở Mac. Phần suy luận encoder và quyết định matching đều nằm trên ESP32-S3.”

### Thao tác

Chỉ hàng đăng ký, sau đó hàng query. Nhấn vào ranh giới Mac/board và điểm lưu template, không đọc mọi ô liên tục. Không thực hiện đăng ký lại trong demo.

## Slide 9 — Cách đánh giá

Thời gian dự kiến: 50 giây.

### Lời nói

“Nhóm dùng EER và top1 để nhìn hai khía cạnh của chất lượng.

EER là mức lỗi tại điểm tỷ lệ chấp nhận nhầm và từ chối nhầm bằng nhau hoặc được nội suy để cân bằng. EER càng thấp càng tốt. Trong protocol này có 45 cặp cùng người và 180 cặp khác người.

Top1 kiểm tra query có chọn đúng người trong năm người đã đăng ký ở validation hay không. Tổng cộng có 45 query.

Mỗi template validation được tạo từ ba embedding đăng ký. Đây là protocol của phần nghiên cứu, khác bộ ba template cá nhân dùng trong demo.

Top1 đúng hết vẫn có thể đi cùng EER khác không, vì xếp hạng người có điểm cao nhất và tách các cặp bằng một ngưỡng là hai yêu cầu khác nhau.”

## Slide 10 — Full100: kết quả validation

Thời gian dự kiến: 95 giây.

### Lời nói

“Với supervised AAM, cả FP32 và INT8 đạt EER khoảng 4,44 phần trăm, top1 đúng 43 trên 45 query.

Với KD cộng AAM, bản FP32 đạt EER khoảng 2,78 phần trăm, thấp nhất trong bảng. Sau chuyển sang INT8, EER tăng lên khoảng 4,17 phần trăm, còn top1 vẫn là 45 trên 45.

Nếu chỉ so hai bản INT8, KD có EER thấp hơn khoảng 0,28 điểm phần trăm và đúng thêm hai query. Hai bản có cùng kích thước 154.000 byte.

Điểm nhóm muốn nhấn mạnh là lượng tử hóa không tác động giống nhau lên mọi chỉ số. Nhánh KD vẫn giữ thứ hạng top1 đúng, nhưng EER tăng so với FP32. Vì vậy không thể chỉ nhìn 45 trên 45 rồi kết luận lượng tử hóa hoàn toàn không làm suy giảm chất lượng.

Đối với câu hỏi ban đầu, nhóm quan sát được lợi ích của KD ở bản INT8 trong phép đối chiếu này. Tuy nhiên, validation chỉ có năm người nói và chưa có đánh giá nhiều seed hay test độc lập. Chênh lệch hiện tại chưa đủ để khẳng định KD luôn tốt hơn hoặc có ý nghĩa thống kê.

Sau đây em chuyển sang phần kiểm chứng triển khai model đã xuất trên thiết bị.”

## Slide 11 — Kết quả demo trên ESP32-S3

Thời gian dự kiến: 170 giây, gồm khoảng 120 giây thao tác. Chỉ nói kết quả lượt chạy mới sau khi giao diện thật sự hoàn tất.

### Lời nói trước demo

“Bảng này là kết quả của hai file query riêng, so với ba mẫu Tien, Quang và An đang lưu trong board.

Với Quang_test, điểm cao nhất khoảng 0,913 thuộc Quang. Với Tien_test, điểm cao nhất khoảng 0,799 thuộc Tien. Mỗi hàng có đủ ba cosine, và board chọn người có điểm lớn nhất.

Đây là độ tương đồng giữa các vector, không phải xác suất hay phần trăm độ tin cậy. Hai file này cũng chưa đại diện cho độ chính xác tổng quát.

Encoder mất khoảng 53,5 mili giây mỗi mẫu trong lần đo trên slide. Tensor arena sử dụng 38.140 byte trong vùng cấp 131.072 byte. Tiếp theo em chạy lại trên dashboard và thử một query từ mic của Mac.”

### Thao tác demo

1. Mở http://127.0.0.1:8765/, bấm **Kiểm tra board**. Chỉ trạng thái sẵn sàng và ba mẫu Quang/Tien/An.
2. Bấm **Thử 2 file có sẵn**, chờ job hoàn tất. Chỉ tên query, nhãn board chọn và ba điểm cosine. Đây là hai query riêng, không phải dùng file đăng ký An để tự so lại chính nó.
3. Nếu đã thử mic trước buổi báo cáo, bấm **Mở mic · 3 giây**. Chỉ bắt đầu nói khi UI báo đang thu. Dùng câu tự nhiên khoảng ba giây, chẳng hạn “Hôm nay em trình bày kết quả thử nghiệm nhận dạng giọng nói trên thiết bị”.
4. Chờ kết quả mới. Chỉ nhãn board thực sự trả về, không đọc sẵn tên dự kiến và không gọi lượt live là đúng nếu chưa đối chiếu người nói.
5. Mở **Hiệu năng board** tại http://127.0.0.1:8765/performance. Chỉ thời gian encoder, nhận dạng trên board và heap còn trống của query gần nhất. Không bấm action khác khi job đang chạy.
6. Quay về slide 11 rồi chuyển slide 12. Không provision, thêm mẫu, reset hoặc nạp app trong lúc trình bày.

### Lời nói sau demo

Nếu hai file đều đúng: “Ở lần chạy vừa rồi, board chọn đúng nhãn của cả hai file kiểm tra. Encoder và cosine thực hiện trên ESP32, còn Mac thu hoặc đọc audio, làm frontend và hiển thị phản hồi.”

Với mic: “Đây là kết quả board chọn trong ba mẫu đã đăng ký cho query vừa thu. Hệ thống chưa có ngưỡng từ chối người lạ, nên hiện vẫn luôn chọn một trong ba tên.”

Với hiệu năng: “Các số này thuộc query vừa chạy. Thời gian encoder chỉ tính suy luận. Thời gian nhận dạng trên board tính thêm sao chép input, chuẩn hóa và matching. Thời gian USB do Mac đo riêng. Chúng chưa bao gồm toàn bộ thời gian thu âm và frontend.”

### Ghi chú khi đọc RAM

- Lần kiểm tra đã lưu: heap còn trống 185.412 byte, khoảng 181,1 KiB. Đây là snapshot của heap nội bộ 8-bit, không phải phép đo liên tục hay toàn bộ SRAM còn lại.
- Thời gian nhận dạng của hai query trong report dashboard khoảng 55,7–55,8 ms. Chỉ đọc số của job hiện trên giao diện khi demo, không coi số lịch sử là kết quả mới.
- Chênh lệch heap trước/sau bằng 0 byte không có nghĩa query không dùng RAM. Encoder tái sử dụng arena đã cấp, và phép đo này chưa đo peak RAM của query.
- Ba template đã có trong NVS và được đọc lại sau reset. Chỉ nhắc một câu nếu cần giải thích, không biến thành câu hỏi nghiên cứu hay diễn giải dài về cơ chế lưu flash.
- Nếu mic nhận nhầm hoặc nhiều lần đều chọn An, nói đúng hiện tượng. Không gọi đó là thành công nhận diện live. Giữ phần kiểm chứng hai file riêng với phần khảo sát mic.

### Nếu demo lỗi

“Lượt trực tiếp hiện chưa hoàn tất do lỗi kết nối hoặc xử lý, nên em chưa báo cáo kết quả mới. Bảng trên slide là kết quả của lần đo trước đã lưu báo cáo.”

Không dùng report cũ như kết quả trực tiếp, không chạy model trên Mac rồi gọi là board và không kéo dài phần khắc phục quá thời gian.

## Slide 12 — Kết luận

Thời gian dự kiến: 45 giây.

### Lời nói

“Quay lại câu hỏi nghiên cứu, KD cộng AAM INT8 đạt EER khoảng 4,17 phần trăm và top1 45 trên 45, so với 4,44 phần trăm và 43 trên 45 của supervised INT8 trên validation hiện tại. Đây là lợi ích quan sát được trong phép đối chiếu của nhóm, chưa phải khẳng định cho mọi dữ liệu.

Về triển khai, encoder 154.000 byte đã chạy trên ESP32-S3. Board giữ ba template trong NVS và thực hiện cosine matching. Hai file query riêng cho Quang và Tien cho kết quả đúng trong các lượt đã kiểm tra.

Phần nhóm thực hiện là phép so sánh và kiểm chứng triển khai. Nhóm chưa đề xuất thuật toán nhận diện người nói mới.”

## Slide 13 — Hướng phát triển tiếp theo

Thời gian dự kiến: 40 giây.

### Lời nói

“Bước tiếp theo của nghiên cứu là mở rộng đánh giá bằng query độc lập cho An, thêm người nói và điều kiện thu âm, đồng thời kiểm tra trên tập test riêng.

Với người chưa đăng ký, nhóm cần tập calibration để chọn ngưỡng cosine rồi đánh giá khả năng từ chối trên dữ liệu độc lập. Hiện hệ thống mới chọn top1 trong các mẫu có sẵn.

Về thiết bị, nhóm cần đo độ trễ toàn luồng từ thu âm đến hiển thị, kiểm tra nhiều lần tắt bật và bổ sung microphone cùng frontend trên board. Các phép đo encoder, matching và heap hiện có mới bao phủ từng phần của hệ thống.”

## Slide 14 — Cảm ơn Thầy và các bạn đã lắng nghe

Thời gian dự kiến: 15 giây.

### Lời nói

“Phần trình bày của nhóm em đến đây là kết thúc. Nhóm em cảm ơn Thầy và các bạn đã lắng nghe. Em xin nhận góp ý và trả lời câu hỏi của Thầy.”

## Chuẩn bị trước giờ demo

- Mở đúng deck 14 trang và dashboard. Giữ sẵn tab hiệu năng, không mở report chứa dữ liệu cá nhân trước lớp nếu không cần.
- Kết nối board bằng cáp data vào cổng native USB. Kiểm tra trạng thái và ba nhãn Quang/Tien/An, fingerprint `crc32:60a382e2`.
- Chạy thử hai file. Thử mic một lượt và nghe lại bản thu nếu kết quả bất thường. Chỉ có quyền mic không chứng minh audio đã thu đúng.
- Không mở chương trình serial thứ hai. Không thay model, thêm mẫu, provision, reset hay chỉnh ngưỡng sát giờ trình bày.
- Nếu live chưa ổn, giữ hai file cố định làm demo chính và nói rõ live đang khảo sát. Không hứa trước khả năng nhận đúng An từ giọng mới.
- Thuộc ranh giới xử lý: Mac thu âm/frontend/lượng tử hóa/hiển thị. ESP32 encoder/chuẩn hóa/cosine/top1. NVS giữ vector, không giữ WAV hoặc M4A.

## Q&A để chuẩn bị, không đọc nối vào bài

### Ý tưởng này đã có ai làm chưa? Nhóm thêm gì?

“Nhóm sử dụng speaker embedding, KD và lượng tử hóa đã có. Phần thực hiện là so sánh hai mục tiêu trên cùng student, dữ liệu và protocol, đọc tác động của INT8, rồi kiểm chứng model cùng matching trên ESP32-S3. Nhóm chưa nhận là đầu tiên hay đề xuất thuật toán mới.”

### Vì sao không chọn lưu mẫu sau reset làm câu hỏi nghiên cứu?

“Đó là tiêu chí kiểm chứng bản triển khai có giữ được template để vận hành. Câu hỏi nghiên cứu chính tập trung vào ảnh hưởng của KD và INT8 lên chất lượng model. Nhóm đưa kết quả lưu mẫu vào phần demo kỹ thuật.”

### Tại sao không dùng camera?

“Nhóm chọn giọng nói để khảo sát speaker embedding và mô hình nhỏ trên thiết bị hạn chế tài nguyên. Camera nằm ngoài phép so sánh này. Nhóm chưa có thí nghiệm để kết luận giọng nói tốt hơn khuôn mặt.”

### Full dataset có nghĩa đã đánh giá toàn bộ người nói không?

“Full chỉ việc dùng toàn bộ train-clean-100 để huấn luyện, gồm 251 người nói. Bảng kết quả vẫn là validation năm người nói tách khỏi train. Chưa có kết quả test độc lập.”

### Model nhỏ cụ thể là gì?

“baseline_conv8 là encoder convolution tám kênh, pooling và chiếu ra vector 192 chiều. Board dùng bản KD cộng AAM INT8, 154.000 byte. Teacher ECAPA-TDNN chỉ dùng offline lúc train.”

### Vì sao top1 đúng hết mà EER khác không?

“Top1 chỉ xét nhãn đứng đầu, còn EER xét tách cặp cùng người và khác người bằng ngưỡng. Xếp hạng vẫn đúng trong khi phân bố điểm còn chồng lấn. 45 trên 45 không đồng nghĩa độ chính xác tổng quát 100 phần trăm.”

### Người chưa đăng ký nói thì sao?

“Hiện board vẫn chọn một trong Quang, Tien và An có cosine cao nhất. Chưa có ngưỡng từ chối, chưa đánh giá open-set hoặc chống phát lại. Vì vậy đây chưa phải hệ thống điểm danh hoàn chỉnh để sử dụng thực tế.”

### Board lưu file mẫu hay vector?

“Board lưu ba vector float32 đã chuẩn hóa, mỗi vector 192 chiều, cùng nhãn và metadata trong record 2.416 byte. Các file âm thanh gốc không nằm trên board. Query không đăng ký lại mẫu.”

### Vì sao dùng ANSI C?

“Bản ESP-NN tối ưu ban đầu chạy được nhưng sai lệch output lớn. Với ANSI C, năm fixture từ train đều đạt sai lệch tối đa một LSB so với tham chiếu. Nhóm giữ cấu hình đúng số học cho demo, chưa xác định chắc operator gây lỗi ở bản tối ưu.”

### 53,5 ms và 0 byte nói lên điều gì?

“53,5 ms là thời gian encoder của các query đã đo, không tính thu âm, frontend hay USB. Heap delta 0 byte chỉ nói lượng heap trống trước và sau không đổi. Model vẫn dùng arena và bộ nhớ firmware, chưa có phép đo peak RAM theo query.”

## Phân bổ thời gian

| Slide | Giây dự kiến |
|---|---:|
| 1 | 20 |
| 2 | 45 |
| 3 | 40 |
| 4 | 75 |
| 5 | 35 |
| 6 | 50 |
| 7 | 35 |
| 8 | 60 |
| 9 | 50 |
| 10 | 95 |
| 11 | 170 |
| 12 | 45 |
| 13 | 40 |
| 14 | 15 |
| Tổng | 775 (12 phút 55 giây) |

Nếu chỉ có 10 phút, nói slide 2 trong 25 giây, rút khoảng 30 giây ở slide 4, 25 giây ở slide 10 và 120 giây ở slide 11 bằng cách chỉ chạy hai query cố định, bỏ lượt mic cùng phần đọc RAM chi tiết. Giữ input/output, câu hỏi nghiên cứu và giới hạn validation. Nếu có 15 phút, dùng phần dư cho một lượt mic có kiểm tra hoặc Q&A, không đọc thêm toàn bộ log.

## Nguồn để đối chiếu khi tập

- [Deck Markdown](slide.md) và [HTML](slide.html).
- [Báo cáo supervised](../models/full100-20261015/baseline_supervised/full_training_report.json) và [KD](../models/full100-20261015/baseline_kd/full_training_report.json).
- [Audit FULL100](../models/full100-20261015/evidence/independent_final_audit.json).
- Hai query so với ba template trên board, nguồn bảng slide 11: `artifacts/voice-demo/20261020/three-file-readonly-155417-874901/report.json` (bản local, không đưa lên GitHub). Chỉ hai dòng `INDEPENDENT_QUERY` thuộc bảng. Dòng An trong report là self-check file đăng ký, không phải query độc lập.
- Dashboard hai query và telemetry: `artifacts/demo-dashboard/20261020-084952-b4123338-d166a38a/report.json` (bản local).
- Tài liệu triển khai local: `docs/demo-mac-usb.md`, `docs/demo-dashboard.md`, `docs/board-telemetry.md`. Các file này không nằm trong gói xuất bản slide.

Khi thay thứ tự hoặc nội dung deck, đối chiếu lại cả 14 tiêu đề trước khi tập. Các số snapshot trong ghi chú là kết quả lịch sử có nguồn, không thay cho số của lượt demo mới.
