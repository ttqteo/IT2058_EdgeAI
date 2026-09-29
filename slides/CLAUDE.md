# CLAUDE.md — Marp Slide Template (rules + vibe guide)

Quy ước khi tạo/sửa slide trong folder này (và mọi deck dùng template này).
File chính: **`slide.md`**. Sửa nội dung trong đó; giữ nguyên khối `style:` (theme).

## Công cụ & định dạng
- Slide viết bằng **Marp**, tỉ lệ **16:9** (mặc định) — không dùng 4:3.
- Render:
  ```bash
  npx -y @marp-team/marp-cli slide.md --html --allow-local-files -o slide.html
  npx -y @marp-team/marp-cli slide.md --pdf  --allow-local-files   # nộp PDF
  npx -y @marp-team/marp-cli slide.md --pptx --allow-local-files   # mở PowerPoint
  ```
  Luôn có `--allow-local-files` (nhúng logo/ảnh cục bộ). Cảnh báo "Insecure local file" là bình thường.
- Xem khi soạn: extension **Marp for VS Code** hoặc mở file `.html`.

## Theme (BẮT BUỘC giữ nhất quán)
- **Primary: navy `#1F3A68`**, accent xanh `#2a6df4`. Nền trắng (radial xanh nhạt góc trên-phải), chữ `#1d2b36`.
- **Font: Be Vietnam Pro** (nạp qua `@import` Google Fonts — cần mạng khi render/mở HTML).
- Bố cục Beamer "VSE":
  - Tiêu đề slide nội dung = `## ...` → **thanh header navy gradient** tràn viền, dính đỉnh.
  - **Logo UIT** góc phải mọi slide (chip trắng) qua directive `header:` — không chèn từng slide.
  - **Footer 4 ô gradient**: `Tên · Đề tài · Ngày · (số trang)`. Số trang dạng `current / total` tự động.
- Tên trường tiếng Anh: **University of Information Technology, VNU-HCM (UIT)**.
- **Không thêm box-shadow cho ảnh** (và tránh shadow trang trí không cần thiết). Ảnh để phẳng, sạch.

## Các loại slide (đặt directive ở đầu slide)
| Mục đích | Cú pháp |
|---|---|
| Slide nội dung | `## Tiêu đề` (thanh navy) |
| Slide bìa | `<!-- _class: lead cover -->` + `<div class="titlebox">` |
| **Slide chuyển mục (section)** | `<!-- _class: divider -->` (nền navy + số mờ lớn) |
| Slide cảm ơn | `<!-- _class: lead -->` + `<div class="thanks">` |
| Slide bảng/list dày | thêm `<!-- _class: tight -->` (thu nhỏ bảng & bullet) |

### Slide divider — copy nhanh
```html
<!-- _class: divider -->

<div class="dnum">1</div>
<div class="dbar"></div>

# Tên phần

<div class="dsub">ý 1 · ý 2 · ý 3</div>
<div class="dmeta">Phần 1</div>
```
Đổi footer theo section đang nói (tự giữ tới divider kế):
`<!-- footer: '<span>Tên</span><span>Phần 1 · ...</span><span>Ngày</span><span></span>' -->`

## Component có sẵn (class trong khối style)
- `.box` — hộp ý chính / định nghĩa / câu hỏi nghiên cứu (viền trái xanh).
- `.warn` — hộp lưu ý / cảnh báo (viền vàng).
- `.grid2` — 2 cột bằng nhau; `.cols` + `.col` — cột linh hoạt.
- `.flow` (+ `.step`, `.step.fill`, `.ar`) — sơ đồ các bước theo chiều dọc.
- `.chips` (+ `.chip`, `.chip.alt`, `.chip.hot`, `.sep`) — timeline pill ngang.
- `.pipeline` — khối monospace mô tả luồng xử lý.
- `.mono` (+ `.dim`) — minh hoạ ma trận/sơ đồ chữ nền tối.
- `.diagram` + `![h:300px](...)` — ảnh/sơ đồ căn giữa (ảnh để phẳng, không shadow).
- `.caption` / `.small` — chú thích nhỏ. `.center` — căn giữa. `.yes`/`.no` — xanh/đỏ.
- Công thức: KaTeX `$...$` / `$$...$$` (display tự bọc thành card).
- Khoảng cách dọc giữa các block (`.box .warn .pipeline .grid2 pre table`) đã set 16px — không cần thêm `<br>`.

## Sơ đồ Mermaid — QUAN TRỌNG
- **KHÔNG** nhúng Mermaid inline (`<div class="mermaid">` + `<script>`): VS Code preview chặn script, PDF bị **cắt chữ** và **thừa 1 trang trắng**.
- **LUÔN pre-render ra SVG** rồi nhúng ảnh:
  1. Viết nguồn trong `diagrams/*.mmd`
  2. Render: `npx -y @mermaid-js/mermaid-cli -i diagrams/x.mmd -o diagrams/x.svg -c diagrams/mermaid-config.json -b transparent`
  3. Nhúng: `![h:300px](diagrams/x.svg)` trong `<div class="diagram">`
- Config Mermaid phải đặt `flowchart.htmlLabels: false` (chữ thành SVG text thật, không clip) + theme navy.

## Hình ảnh
- Ảnh thật để trong `assets/`, nhúng `![h:300px](assets/figure.png)` trong `<div class="diagram">`. Không để placeholder ảnh vỡ.
- Ảnh nằm ngoài folder (vd kết quả): nhúng bằng đường dẫn tương đối, vẫn cần `--allow-local-files`.
- Ảnh để phẳng — **không** thêm box-shadow.

## Văn phong (rule người dùng)
- **Hạn chế em-dash (—).** Nhãn → dùng dấu hai chấm (`Label: mô tả`); trong câu → dấu phẩy hoặc middot `·`.
- **Hạn chế icon/emoji.** Blockquote/box đã có viền + nền nên không cần icon trang trí.
- Slide bìa theo thứ tự: hộp tiêu đề → tên thành viên → GVHD → môn `·` loại báo cáo → địa điểm, ngày.

## Vibe với Claude (cách nhờ AI sửa nhanh)
Nói ngắn gọn, dẫn tên slide/section; Claude tự dùng đúng class ở trên. Ví dụ prompt tốt:
- "Thêm slide chuyển mục `divider` cho Phần 3 — Methodology, sub gồm: data, model, eval."
- "Slide Results: đổi sang 2 cột `grid2`, trái là bảng số, phải là `![](assets/fig.png)` + caption."
- "Chèn `.warn` nhắc giả định mô hình ở slide Theoretical background."
- "Bảng đang tràn → thêm `_class: tight` cho slide đó."
- "Đổi footer giữa của section này thành tên phần đang nói."
- "Render lại HTML + PDF."

Mẹo: yêu cầu Claude **render lại sau khi sửa** để bắt lỗi tràn slide / ảnh vỡ sớm. Nếu số liệu/biểu đồ có sẵn trong repo, chỉ Claude tới file đó (vd `results/plots/x.png`, `results/x.csv`) và bảo "đối chiếu số trong slide với file thật" để tránh sai số.

## Cấu trúc folder
```
slides/
├── slide.md / .html / .pdf
├── assets/        (logo UIT, hình)
└── diagrams/      (*.mmd nguồn, *.svg đã render, mermaid-config.json)
```
