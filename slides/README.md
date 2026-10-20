# Slide thuyết trình EdgeAI

Bản 14 trang được chọn ngày 20/10/2026; Nguyễn Ấn trình bày toàn bộ.

- [Mở bản HTML](slide.html), cần tải file hoặc clone repository để trình chiếu.
- [PDF](slide.pdf), dùng khi trình bày offline.
- [PowerPoint](slide.pptx).
- [Nguồn Marp](slide.md).
- [Script thuyết trình](script-thuyet-trinh.md), khoảng 13 phút gồm demo.

Giữ `assets/UIT_logo.svg` cạnh bộ slide theo cấu trúc hiện tại. HTML dùng Google Fonts; khi không có mạng, dùng PDF để giữ bố cục đã xuất.

Để xuất lại, chạy trong thư mục `slides/` với Node.js và trình duyệt Chromium/Chrome đã cài:

```sh
npx --yes @marp-team/marp-cli@4.5.1 slide.md --html --allow-local-files -o slide.html
npx --yes @marp-team/marp-cli@4.5.1 slide.md --pdf --html --allow-local-files -o slide.pdf
npx --yes @marp-team/marp-cli@4.5.1 slide.md --pptx --html --allow-local-files -o slide.pptx
```

Báo cáo FULL100 công khai nằm tại [models/full100-20261015](../models/full100-20261015/README.md). Log demo cá nhân, audio, NVS/flash backup, dataset và checkpoint không thuộc gói slide. Các số liệu board trong slide là kết quả lịch sử có phạm vi giới hạn, không phải số đo live hay độ chính xác tổng quát.
