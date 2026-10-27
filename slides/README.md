# Slide thuyết trình EdgeAI

Bản 16 trang sửa ngày 27/10/2026; Nguyễn Ấn trình bày toàn bộ. Giữ template, ảnh và bố cục hai trang đầu có ảnh; gộp các phần lặp về mục tiêu, kiến trúc và đánh giá. Trình bày theo mạch: bài toán → câu hỏi nghiên cứu → phương pháp → kết quả validation → triển khai và số đo board → demo → kết luận và giới hạn.

- [Mở bản HTML](slide.html), cần tải file hoặc clone repository để trình chiếu.
- [PDF](slide.pdf), dùng khi trình bày offline.
- [PowerPoint](slide.pptx).
- [Nguồn Marp](slide.md).
- [Script thuyết trình](script-thuyet-trinh.md), cập nhật ngày 27/10/2026 theo đủ 16 trang; Nguyễn Ấn trình bày toàn bộ.

Giữ `assets/UIT_logo.svg` cạnh bộ slide theo cấu trúc hiện tại. HTML dùng Google Fonts; khi không có mạng, dùng PDF để giữ bố cục đã xuất.

Để xuất lại, chạy trong thư mục `slides/` với Node.js và trình duyệt Chromium/Chrome đã cài:

```sh
npx --yes @marp-team/marp-cli@4.5.1 slide.md --html --allow-local-files -o slide.html
npx --yes @marp-team/marp-cli@4.5.1 slide.md --pdf --html --allow-local-files -o slide.pdf
npx --yes @marp-team/marp-cli@4.5.1 slide.md --pptx --html --allow-local-files -o slide.pptx
```

Bảng FULL100 lấy từ hai `full_training_report.json` trong `artifacts/full100/20261015/`: chỉ đánh giá validation, chưa chạy calibration/test. Số liệu tối ưu frontend nằm trong [pcm-optimization.md](../docs/pcm-optimization.md); kết quả năm mẫu đăng ký và sáu query cố định được tổng hợp trong `evidence/enrollment/20261023/runtime-verification.json`.

Luồng hiện tại: Mac thu PCM thô; ESP32-S3 tính log-mel, chuẩn hóa, INT8, encoder và cosine. Bốn query công khai gửi PCM; hai query Quang/Tien cũ gửi tensor và bỏ qua frontend. Thời gian encoder không phải thời gian toàn luồng. Các kết quả này chưa xác lập độ chính xác mic trực tiếp hoặc khả năng từ chối người lạ.

Log cá nhân, audio, NVS/flash backup, dataset và checkpoint không thuộc gói slide. Bản 22 trang và script trước khi gộp được giữ trong `privatebackup/20261027-164521-slides-deduplicate/`.
