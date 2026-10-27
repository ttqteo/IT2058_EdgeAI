# Kịch bản thuyết trình theo 16 slide

Nguyễn Ấn trình bày toàn bộ và tự thao tác demo, đại diện nhóm. Bản ngày 27/10/2026, khớp thứ tự và tiêu đề của [slide hiện tại](slide.html). Dự kiến 12–15 phút, gồm demo; cần bấm giờ khi tập.

Các đoạn in nghiêng trong ngoặc vuông là hướng dẫn thao tác, không đọc. Phần chuẩn bị và trả lời câu hỏi nằm cuối tài liệu.

## Slide 1 · Điểm danh bằng giọng nói trên ESP32-S3

Em chào Thầy và các bạn. Em là Nguyễn Ấn, đại diện nhóm 9 trình bày đề tài điểm danh bằng giọng nói trên ESP32-S3.

Nhóm so sánh hai cách huấn luyện một encoder nhỏ: học từ nhãn người nói và học thêm từ một teacher. Sau đó, nhóm lượng tử hóa encoder sang INT8 và đo khả năng xử lý âm thanh trên board thật.

## Slide 2 · Bối cảnh và bài toán

Nhóm dùng bối cảnh điểm danh để đặt bài toán nhận diện người nói trong danh sách đã đăng ký.

Đầu vào là một đoạn PCM ba giây và danh sách mẫu có nhãn. Đầu ra là tên có cosine cao nhất cùng điểm của từng người. Một mẫu đăng ký là vector đặc trưng của giọng nói.

Trong phạm vi này, board luôn chọn một người trong danh sách. Nhóm chưa có ngưỡng từ chối người lạ, và điểm cosine không phải xác suất nhận đúng.

## Slide 3 · Mục tiêu và câu hỏi nghiên cứu

Nhóm khảo sát hai vấn đề.

Về chất lượng mô hình, KD thay đổi EER và Top-1 thế nào so với supervised, trước và sau INT8? Nhóm giữ cùng student, dữ liệu và quy trình đánh giá để đối chiếu hai cách huấn luyện.

Về thiết bị, nhóm muốn giảm thời gian cùng RAM tiền xử lý nhưng giữ nguyên đầu ra. Nhóm kiểm tra với bản tham chiếu, sau đó đo trên ESP32-S3.

## Slide 4 · Teacher và student

Teacher là checkpoint ECAPA-TDNN của SpeechBrain, đã huấn luyện sẵn trên VoxCeleb. Trong thí nghiệm này, nhóm huấn luyện student trên LibriSpeech.

Encoder nhận tensor INT8 kích thước 1 nhân 298 nhân 40 nhân 1 và trả embedding 192 chiều. Student có tên baseline_conv8. Số tám chỉ số kênh, không phải tám lớp convolution. Nhóm dùng một lớp Conv2D tám kênh với kernel 3 nhân 3, stride 2, tiếp theo là ReLU và average pooling. Sau flatten, nhóm đưa 760 giá trị qua lớp Dense để tạo embedding 192 chiều.

Nhóm giữ nguyên kiến trúc này cho hai cách huấn luyện. Board chạy encoder INT8 của nhánh KD cộng AAM.

*[Chỉ sơ đồ từ Conv2D đến Dense; không cần đọc lại toàn bộ kích thước tensor.]*

## Slide 5 · Huấn luyện với AAM và KD

Trong nhánh supervised, nhóm dùng AAM-Softmax để huấn luyện student. Nhóm chuẩn hóa embedding và vector trọng số của từng lớp người nói, sau đó tính cosine giữa chúng.

Với lớp đúng, nhóm thêm biên góc m bằng 0,2 radian trước khi tính logit, dùng hệ số s bằng 30 rồi tính Softmax và Cross-Entropy. Qua lan truyền ngược, nhóm cập nhật backbone cùng head phân lớp để học cách phân biệt người nói. Các vector lớp ở đây là trọng số học trong training; chúng khác mẫu đăng ký trong NVS.

Ở nhánh KD, nhóm giữ nguyên loss AAM và cộng thêm loss một trừ cosine giữa embedding student với embedding teacher, với hệ số lambda bằng một. Student vì thế vừa học nhãn, vừa học hướng biểu diễn từ teacher. Nhóm giữ cố định trọng số teacher.

Khi inference, nhóm bỏ AAM head và teacher, dùng encoder để tạo embedding rồi so cosine với các mẫu đăng ký.

*[Chỉ hai thành phần loss trong công thức. Không đọc lần lượt các ký hiệu.]*

## Slide 6 · Dữ liệu và quy trình đánh giá

Nhóm huấn luyện student trên LibriSpeech train-clean-100, gồm 251 người nói và 28.539 bản ghi gốc. Validation có năm người khác với train; mỗi người ba mẫu đăng ký và chín query. Nhóm tổng hợp ba embedding đăng ký thành một mẫu đại diện.

Nhóm dùng hai chỉ số. EER đo lỗi tại điểm tỷ lệ nhận nhầm bằng từ chối nhầm, từ 45 cặp cùng người và 180 cặp khác người. Top-1 là số query chọn đúng người trên tổng 45 query.

Người đúng có thể đứng đầu trong khi điểm cùng người và khác người vẫn chồng lấn, nên Top-1 đúng hoàn toàn chưa có nghĩa EER bằng không.

Nhóm dùng validation để chọn checkpoint và báo cáo kết quả, chưa có test độc lập. Năm người ở đây khác năm nhãn trong demo.

## Slide 7 · Full100: kết quả validation

Với supervised AAM, cả FP32 và INT8 đạt EER khoảng 4,44 phần trăm, đúng 43 trên 45 query.

Với KD cộng AAM, FP32 đạt EER khoảng 2,78 phần trăm. Sau INT8, EER tăng lên khoảng 4,17 phần trăm, trong khi Top-1 giữ 45 trên 45.

Nếu so hai bản INT8, KD đúng thêm hai query và có EER thấp hơn khoảng 0,28 điểm phần trăm. Hai bản có cùng kích thước encoder 154.000 byte.

Nhóm quan sát được lợi ích của KD trong phép đối chiếu này, đồng thời thấy lượng tử hóa làm EER của nhánh KD tăng dù Top-1 giữ nguyên. Vì vậy, nhóm không dùng riêng kết quả 45 trên 45 để kết luận INT8 giữ nguyên chất lượng.

Validation chỉ có năm người và nhóm đã dùng tập này để chọn checkpoint. Nhóm cần test độc lập cùng nhiều lần huấn luyện để đánh giá mức ổn định của chênh lệch.

## Slide 8 · Phần cứng

Nhóm dùng ESP32-S3 để tiền xử lý, chạy encoder và so khớp. Board có 512 KB SRAM nội bộ, 16 MB flash và 8 MB PSRAM.

Nhóm lưu firmware và model trong flash, cùng các mẫu đăng ký trong vùng NVS. Bản demo hiện dùng SRAM nội bộ và chưa bật PSRAM.

Nhóm chưa gắn INMP441. Mac thu PCM ba giây, mono 16 kilohertz, rồi gửi qua cổng native USB. Cách thu này cho phép nhóm kiểm tra luồng xử lý trên board trước khi bổ sung mic phần cứng. Nhóm chưa chứng minh chất lượng thu của mic Mac tương đương INMP441 và chưa đo điện năng.

## Slide 9 · Luồng xử lý PCM trên board

*[Chỉ đường PCM chung trước, rồi hai nhánh đăng ký và nhận diện.]*

Mac thu và gửi nguyên đoạn PCM qua USB. Trên ESP32-S3, nhóm tính log-mel, chuẩn hóa đặc trưng rồi lượng tử hóa INT8. Encoder chuyển tensor này thành embedding 192 chiều.

Khi đăng ký, board chuẩn hóa embedding và lưu vector cùng nhãn trong NVS. Hai mẫu công khai LS116 và LS700 dùng luồng PCM này; ba mẫu Quang, Tien và An đã đăng ký bằng luồng tensor trước đó. NVS giữ vector, không giữ file âm thanh.

Khi nhận diện, board chỉ đọc các mẫu NVS, tính cosine và trả tên cùng điểm về dashboard. Ứng dụng trên Mac không tính đặc trưng hay embedding. Nhóm đã huấn luyện và xuất model trước khi chạy demo.

## Slide 10 · Kích thước model và bộ nhớ

Encoder FP32 có kích thước 587.132 byte; bản INT8 nạp lên board là 154.000 byte. Đây là kích thước file model. Flash giữ firmware, model và mẫu NVS sau khi mất điện.

Khi suy luận, runtime dùng tensor arena trong SRAM. Nhóm cấp 131.072 byte và runtime báo dùng 38.140 byte trong vùng đó. Board còn cần RAM cho PCM, workspace, stack và các thành phần hệ thống, nên arena không đại diện cho tổng RAM sử dụng.

Bản ghi năm mẫu trong NVS chiếm 3.972 byte, chưa tính overhead lưu trữ. Snapshot STATUS khi chờ ghi nhận 183.192 byte heap trống trên tổng vùng đo 239.908 byte. Số đo này chưa phải RAM đỉnh của query.

## Slide 11 · Tối ưu tiền xử lý trên ESP32-S3

Ở bản đầu, board mất khoảng 7,66 giây để tiền xử lý một đoạn PCM ba giây. Riêng encoder chỉ mất khoảng 53,6 mili giây. Nhóm vì thế tập trung tối ưu frontend.

Nhóm tính sẵn cửa sổ Hann, hệ số FFT và thứ tự đảo bit. Với mel filterbank, nhóm chỉ duyệt 494 trọng số khác không thay cho việc duyệt cả ma trận có nhiều phần tử bằng không. Nhóm cũng bỏ buffer phổ công suất trung gian để giảm workspace.

Nhóm giữ phép tính double và kiểm tra đầu ra với bản tham chiếu. Trên tám đầu vào host đã kiểm tra, đặc trưng và tensor INT8 trùng từng byte. Với hai PCM đo trên board, embedding trùng đường tensor cũ.

Sau tối ưu, frontend còn khoảng 2,85 giây, giảm khoảng 62,7 phần trăm. Workspace giảm từ 10.568 xuống 8.512 byte, tiết kiệm 2.056 byte. Hai lượt đo trong bảng không xuất hiện cảnh báo watchdog; nhóm chưa đánh giá chạy liên tục.

Thời gian gửi PCM đến nhận kết quả qua USB còn khoảng 3,22 giây. Con số này chưa gồm ba giây thu âm.

## Slide 12 · Kết quả demo trên ESP32-S3

### Lời nói trước thao tác

Board đang lưu năm mẫu Quang, Tien, An, LS116 và LS700.

Trong bảng, hai query Quang và Tien dùng tensor cũ, bỏ qua frontend trên board. Bốn query LS116 và LS700 dùng PCM thô, với câu nói khác mẫu đăng ký. Cả bốn query PCM chọn đúng nhãn trong các lượt đã lưu.

Cosine cao nhất của LS116 khoảng 0,85 và 0,81; của LS700 khoảng 0,77 và 0,74. Đây là sáu ví dụ kiểm tra, chưa phải phép đo độ chính xác tổng quát.

Em sẽ kiểm tra board trên giao diện rồi thu một query mới. Mac chỉ thu PCM và truyền dữ liệu; ESP32-S3 xử lý và trả kết quả.

### Thao tác

*[Mở http://127.0.0.1:8765/. Bấm “Kiểm tra board”, chờ trạng thái sẵn sàng và đủ năm nhãn.]*

Board đã đọc năm mẫu từ NVS. Mỗi mẫu gồm tên và vector đặc trưng; trong lượt demo này em không đăng ký lại.

*[Nếu cần chạy đối chứng, bấm nút hai tensor cũ. Nói rõ thao tác này bỏ qua frontend. Chờ job kết thúc trước khi mở mic.]*

*[Bấm “Thu PCM · 3 giây”. Chờ UI báo đang thu rồi nói một câu khoảng ba giây. Có thể dùng: “Em đang thử nhận diện giọng nói với phần xử lý trên ESP32.” Chờ kết quả mới; không nói trước tên sẽ xuất hiện.]*

### Lời nói theo kết quả thật

Board trả về tên và năm điểm cosine cho đoạn vừa thu. Tên ở đầu là mẫu có độ tương đồng cao nhất. Em đối chiếu với người vừa nói để xem lượt này nhận đúng hay nhầm.

*[Chỉ tên thật trên màn hình. Nếu nhận nhầm, nói “Lượt này board chọn …, khác người vừa nói.” Không gọi kết quả là đúng chỉ vì giao diện đã hoàn tất.]*

*[Mở trang “Hiệu năng board”. Chỉ số liệu của query vừa chạy, không đọc số lịch sử thành số mới.]*

Ở trang này, em tách thời gian frontend khỏi encoder. Thời gian nhận diện trên board gồm cả tiền xử lý và so khớp. Thời gian USB tính từ gửi request đến nhận phản hồi, còn ba giây thu âm nằm ngoài khoảng đó.

RAM còn trống là snapshot của heap nội bộ. Chênh lệch heap trước và sau bằng không chỉ cho biết hai thời điểm đo có cùng lượng trống, chưa cho biết RAM đỉnh mà query sử dụng.

### Nếu lượt demo lỗi

Lượt demo này chưa hoàn tất nên em chưa báo cáo kết quả mới. Bảng trên slide là kết quả của các lượt đã lưu. Em xin tiếp tục phần kết luận.

*[Dành khoảng hai phút cho thao tác. Nếu mất kết nối hoặc thu âm lỗi, chuyển về slide; không nạp lại app, đăng ký mẫu hay dùng suy luận trên Mac để thay kết quả board.]*

## Slide 13 · Kết luận

Từ phép đối chiếu hiện tại, nhóm quan sát được lợi ích của KD so với supervised. Đồng thời, INT8 làm EER của nhánh KD tăng dù Top-1 giữ nguyên, nên cần xem cả hai chỉ số.

Trên ESP32-S3, nhóm đã chạy toàn bộ phần xử lý PCM và nhận diện. Phần tốn thời gian chủ yếu nằm ở frontend. Nhóm tối ưu phần này và kiểm tra đầu ra trước khi báo cáo mức giảm thời gian.

Phần nhóm thực hiện là thiết kế phép đối chiếu huấn luyện, kiểm chứng INT8 và đo chi phí triển khai trên board thật. Các phương pháp AAM, KD và tối ưu bằng bảng tính sẵn đã có trong nghiên cứu trước.

## Slide 14 · Giới hạn và bước tiếp theo

Kết quả còn bốn giới hạn cần kiểm chứng tiếp.

Validation nhỏ và đã dùng chọn checkpoint, nên nhóm cần test độc lập cùng nhiều lần huấn luyện. Với người lạ, nhóm cần tập riêng để hiệu chỉnh ngưỡng rồi đo lỗi chấp nhận và từ chối.

Phần thu âm hiện dùng mic Mac và gửi từng đoạn PCM. Bước tiếp theo là gắn mic I2S, đo lại chất lượng thu cùng độ trễ từ lúc bắt đầu nói đến kết quả. Nhóm cũng cần thử nhiễu và chạy liên tục để theo dõi RAM cùng độ ổn định.

## Slide 15 · Tài liệu tham khảo

Nhóm tham khảo ECAPA-TDNN và SpeechBrain cho teacher, AAM-Softmax cho mục tiêu phân lớp, cùng các công trình về KD, lượng tử hóa và TensorFlow Lite Micro.

Nhóm dùng LibriSpeech cho dữ liệu; hai người nói công khai bổ sung trong demo lấy từ dev-other, với thông tin nguồn và giấy phép đi kèm.

## Slide 16 · Cảm ơn Thầy và các bạn đã lắng nghe

Phần trình bày của em đến đây kết thúc. Nhóm em cảm ơn Thầy và các bạn đã lắng nghe. Em xin nhận góp ý và trả lời câu hỏi của Thầy.

---

## Chuẩn bị trước khi trình bày

- Mở đúng [deck 16 trang](slide.html), dashboard và trang hiệu năng. Kiểm tra cáp data ở cổng native USB; chỉ mở một chương trình sử dụng serial.
- Bấm kiểm tra board, xác nhận đủ Quang/Tien/An/LS116/LS700. Trạng thái đã lưu có fingerprint `crc32:90fb081d`; nếu khác, kiểm tra trước buổi báo cáo thay vì đăng ký lại giữa demo.
- Thu thử bằng đúng mic sẽ dùng, kiểm tra quyền mic của tiến trình thu trên Mac. Quyền mic của browser không chứng minh backend đã thu được PCM. Không mở mic trong lúc job khác còn chạy.
- Tập nói và bấm giờ. Thu ba giây rồi gửi USB là hai bước nối tiếp; với số đo khoảng 3,22 giây cho request/response, cần khoảng sáu giây trở lên từ lúc bắt đầu thu đến kết quả, chưa tính thao tác và các chi phí khác.
- Khi dùng hai tensor cũ, nói rõ chúng bỏ qua frontend. Muốn minh họa xử lý PCM, dùng lượt mic mới; bốn query PCM công khai trong slide là kết quả đã lưu, không mặc định có nút chạy chúng trên dashboard.
- Không đọc số liệu lịch sử như kết quả của query vừa chạy. Không đồng nhất 53,5 ms encoder với toàn bộ thời gian nhận diện.

## Câu trả lời ngắn khi Thầy hỏi

### Nhóm thêm gì so với những phương pháp đã có?

Nhóm dùng các phương pháp đã có để so sánh supervised AAM với KD cộng AAM trên cùng student, rồi đo tác động của INT8. Phần triển khai gồm frontend trên ESP32-S3, tối ưu bảng tính và mel thưa, kiểm tra đầu ra với bản tham chiếu, đo thời gian cùng bộ nhớ. Nhóm không nhận là đề xuất thuật toán nhận diện người nói mới.

### KD có thay AAM không?

Nhóm giữ AAM để student học nhãn, rồi cộng loss cosine với embedding teacher. Nhóm chỉ cập nhật student và head phân lớp của nó, giữ teacher cố định. Sau training, nhóm bỏ cả teacher lẫn head AAM.

### Nhóm tối ưu frontend bằng cách nào?

Nhóm tính sẵn Hann, hệ số FFT và đảo bit, chỉ duyệt trọng số mel khác không, đồng thời bỏ buffer không cần giữ. Nhóm giữ double vì thử float32 chưa đạt tiêu chí sai số. Trên tám đầu vào host đã kiểm tra, bản tối ưu trùng từng byte với bản tham chiếu; hai query PCM trên board cũng giữ embedding như đường tensor.

### Board có nhận âm thanh thô không, hay Mac vẫn tính đặc trưng?

Ở luồng mic hiện tại, ứng dụng Mac thu 48.000 mẫu PCM16 mono 16 kHz rồi gửi nguyên byte qua USB. Board tính log-mel, chuẩn hóa, INT8, encoder và cosine. Driver hoặc hệ điều hành có thể xử lý định dạng ở khâu thu; nhóm chưa đo để khẳng định mic Mac tương đương mic gắn board.

### Gắn mic lên board có còn mất khoảng ba giây xử lý không?

Nhóm chưa đo trên mic gắn board. Khoảng 2,9 giây hiện là thời gian xử lý một đoạn PCM có sẵn trong bộ nhớ. Với cách chạy batch hiện tại, còn cần thời gian thu. Muốn chồng lấp thu và xử lý, nhóm phải triển khai và kiểm tra luồng I2S cùng cơ chế xử lý theo frame; chưa thể suy ra độ trễ đó từ phép đo hiện tại.

### Vì sao 45/45 mà EER vẫn khác không?

Top-1 xét người có điểm cao nhất cho từng query. EER xét khả năng tách cặp cùng người và khác người bằng ngưỡng. Người đúng có thể đứng đầu dù hai phân bố điểm vẫn chồng lấn.

### Năm người trong validation có phải năm người trên board không?

Không. Validation dùng năm người từ dev-clean tách khỏi train, mỗi người ba mẫu đăng ký và chín query. Board demo lưu Quang/Tien/An cùng LS116/LS700. Nhóm báo cáo hai phần riêng vì dữ liệu và mục đích đánh giá khác nhau.

### Heap delta bằng không có nghĩa query không dùng RAM không?

Không. Encoder dùng arena đã cấp và frontend cần vùng làm việc. Heap delta là hiệu giữa hai snapshot trước và sau; vùng nhớ cấp rồi giải phóng trong khoảng đó không xuất hiện trong delta. Nhóm chưa dùng số này để kết luận RAM đỉnh.

### Một người lạ nói thì sao?

Board vẫn chọn tên có cosine cao nhất trong năm mẫu. Nhóm cần hiệu chỉnh ngưỡng trên tập riêng rồi đánh giá lỗi chấp nhận và từ chối trước khi triển khai chức năng người lạ.

### Tại sao không dùng camera?

Nhóm chọn giọng nói để khảo sát embedding, KD và INT8 trên ESP32-S3. Nhóm chưa làm thực nghiệm đối chiếu với camera nên chưa có căn cứ kết luận giọng nói tốt hơn nhận diện khuôn mặt.

---

## Nguồn đối chiếu dành cho người chuẩn bị

Phần này không đọc khi thuyết trình. Các số liệu là kết quả đã lưu; lần viết script này không chạy lại huấn luyện, board hay microphone.

| Nội dung | Nguồn |
|---|---|
| Thứ tự 16 trang và nội dung trình chiếu | [slide.md](slide.md) |
| EER, Top-1, AAM, KD và kích thước encoder | `artifacts/full100/20261015/baseline_supervised/full_training_report.json`, `artifacts/full100/20261015/baseline_kd/full_training_report.json` |
| Tối ưu frontend, workspace, parity và số đo trước/sau | [pcm-optimization.md](../docs/pcm-optimization.md) |
| Năm nhãn, NVS, arena và snapshot heap | `artifacts/voice-demo/20261023/enroll-status-after-reboot-001/report.json` |
| Bốn query PCM, hai tensor cũ và giới hạn bằng chứng | `evidence/enrollment/20261023/runtime-verification.json`, các report liên kết trong đó |
| Thu PCM thô và phạm vi DSP của ứng dụng | `scripts/demo_dashboard.py`, [demo-dashboard.md](../docs/demo-dashboard.md) |

Các artifact cá nhân có thể không đi kèm bản clone GitHub. Việc quan sát một lần khởi động và đọc lại NVS chưa chứng minh độ bền qua nhiều chu kỳ tắt bật. Log reset được review ghi `USB_UART_CHIP_RESET`, không tách riêng tác động của nút RST vật lý.
