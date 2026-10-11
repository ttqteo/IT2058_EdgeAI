---
marp: true
title: "Hệ thống điểm danh bằng giọng nói trên vi điều khiển ESP32-S3 dùng mô hình nén"
author: "Trần Tú Quang · Tô Huỳnh Minh Tiến · Nguyễn Ấn"
paginate: true
html: true
math: katex
backgroundColor: "#ffffff"
color: "#1d2b36"
# ============================================================
#  MARP TEMPLATE — theme UIT navy (primary #1F3A68), gradient style
#  Quy ước dùng:
#   - Mỗi slide ngăn nhau bằng `---`
#   - Tiêu đề frame nội dung   ->  `## ...`  (ra THANH NAVY gradient tràn viền)
#   - Slide bìa / cảm ơn       ->  thêm `<!-- _class: lead -->`
#   - Slide chuyển mục (section)->  thêm `<!-- _class: divider -->` (nền navy + số mờ lớn)
#   - Footer (tên · đề tài · ngày) sửa ở dòng `footer:` cuối frontmatter
#   - Công thức: KaTeX  $...$ / $$...$$    |  Hình: ![w:600px](path)
#  Components có sẵn: .box .warn .grid2 .cols .flow .chips .mono .pipeline .pill .diagram
# ============================================================
style: |
  @import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:ital,wght@0,400;0,500;0,600;0,700;1,400&display=swap');
  :root {
    --navy:#1F3A68; --navy-deep:#16294d; --ink:#1d2b36;
    --accent:#2a6df4; --soft:#eef3fb; --line:#d4deee; --muted:#6b7a90;
  }
  section {
    font-family: "Be Vietnam Pro", "Segoe UI", system-ui, sans-serif;
    font-size: 24px;
    padding: 96px 70px 60px 70px;
    background:
      radial-gradient(1200px 380px at 88% -8%, #eef3fb 0%, rgba(238,243,251,0) 60%),
      #ffffff;
    color: var(--ink);
    display: flex; flex-direction: column;
    justify-content: flex-start !important; align-content: flex-start;
    letter-spacing:.1px;
  }
  /* --- THANH HEADER NAVY gradient tràn viền (dùng cho ## slide nội dung) --- */
  h2 {
    position: absolute; top: 0; left: 0; right: 0; margin: 0;
    background: linear-gradient(100deg, var(--navy-deep) 0%, var(--navy) 58%, #28508f 100%);
    color: #ffffff !important; font-size: 29px; font-weight: 600;
    padding: 18px 70px 16px 70px;
    box-shadow: 0 3px 14px rgba(15,29,56,.18);
  }
  h3 {
    color: var(--navy); font-size: 23px; font-weight: 700; margin: 2px 0 14px 0;
    padding-bottom: 7px; border-bottom: 2px solid var(--line); display: inline-block;
  }
  p { margin: 9px 0; }
  strong { color: var(--navy); font-weight: 700; }
  em { color: var(--accent); font-style: normal; font-weight: 600; }
  a { color: var(--navy); }
  code { background: var(--soft); color: var(--navy); padding: 1px 7px; border-radius: 5px; font-size: .94em; }
  /* --- bullet markers gradient vuông bo --- */
  ul { list-style: none; padding-left: 4px; margin: 8px 0; }
  ul li { position: relative; padding-left: 26px; margin: 12px 0; line-height: 1.45; }
  ul li::before {
    content: ""; position: absolute; left: 3px; top: .55em;
    width: 8px; height: 8px; border-radius: 3px;
    background: linear-gradient(135deg, var(--navy), var(--accent));
  }
  ol { padding-left: 22px; } ol li { margin: 11px 0; line-height: 1.45; }
  /* --- bảng bo góc + đổ bóng + zebra --- */
  table { display:table !important; font-size: 21px; border-collapse: collapse; margin: 10px 0; width: 100%;
    min-width:0; max-width:100%; box-sizing:border-box; table-layout:fixed;
    box-shadow: 0 2px 10px rgba(15,29,56,.07); border-radius: 10px; overflow: hidden; }
  thead th { background: var(--navy); color: #fff; font-weight: 600; }
  tbody tr:nth-child(even) td { background: #f6f9fe; }
  td, th { border: 1px solid var(--line); padding: 8px 14px; overflow-wrap:anywhere; }
  blockquote {
    border: none; border-left: 5px solid var(--navy);
    background: var(--soft); color: #20324f; padding: 12px 22px;
    border-radius: 0 12px 12px 0; margin: 12px 0;
  }
  /* --- công thức display dạng card --- */
  .katex-display {
    background: linear-gradient(180deg, #f7faff 0%, #eef3fb 100%);
    border: 1px solid var(--line); border-left: 5px solid var(--navy);
    border-radius: 12px; padding: 16px 22px; margin: 14px 0;
    box-shadow: 0 2px 12px rgba(15,29,56,.07);
  }
  .katex { font-size: 1.18em; }
  /* --- code block dạng card --- */
  pre { background: #f7faff; border: 1px solid var(--line);
    border-radius: 12px; padding: 14px 18px; box-shadow: 0 2px 12px rgba(15,29,56,.06); }
  pre code { background: none; color: #20324f; font-size: 19px; line-height: 1.6; }
  /* --- footer 4 ô gradient (Name | Title | Date | page) --- */
  footer { left:0; bottom:0; width:100%; box-sizing:border-box; display:flex; padding:0;
    height:26px; font-size:13px; color:#ffffff;
    background: linear-gradient(90deg,#0e1d38 0%,#16294d 30%,#1f3a68 62%,#2a4d86 100%); }
  footer span { flex:1; display:flex; align-items:center; justify-content:center;
    border-right:1px solid rgba(255,255,255,.28); }
  footer span:nth-child(4) { flex:0 0 64px; }   /* ô số trang hẹp hơn */
  footer span:last-child { border-right:none; }
  section::after { position:absolute; right:20px; bottom:5px; z-index:10; color:#ffffff;
    font-weight:600; font-size:13px;
    content: attr(data-marpit-pagination) " / " attr(data-marpit-pagination-total); }
  /* --- logo UIT góc phải, hiện trên MỌI slide (qua directive header:) --- */
  header { position:absolute; top:9px; right:16px; left:auto; margin:0; padding:0;
    background:none; box-shadow:none; z-index:40; }
  header img { height:52px; width:52px; object-fit:contain; display:block; background:#ffffff;
    border-radius:50%; padding:6px; box-sizing:border-box; box-shadow:0 1px 5px rgba(0,0,0,.22); }
  section.cover header img { background:none; box-shadow:none; padding:0; }
  /* --- slide bìa / chuyển mục dạng lead --- */
  section.lead { text-align:center; justify-content:center; }
  section.lead::before { content:""; position:absolute; top:0; left:0; right:0; height:8px;
    background: linear-gradient(90deg, var(--navy) 0%, var(--accent) 100%); }
  .titlebox { width:100%; box-sizing:border-box;
    background: linear-gradient(120deg, #16294d 0%, #1F3A68 60%, #2a558f 100%);
    border-radius:16px; padding:30px 44px; margin:10px 0 26px 0;
    box-shadow:0 10px 30px rgba(15,29,56,.22); text-align:center; }
  .titlebox h1 { background:none; border:none; color:#fff !important;
    font-size:42px; margin:0; padding:0; letter-spacing:.3px; }
  .titlebox h3 { color:#cfe0ff !important; font-weight:400; border:none; margin:10px 0 0 0; display:block; }
  section.lead h1 { color:var(--navy); font-size:42px; }
  section.lead h3 { color:var(--ink); font-weight:400; border:none; display:block; }
  .thanks h1 { background:none; border:none; box-shadow:none; color:var(--navy) !important;
    font-size:46px; font-weight:700; margin:40px 0 24px 0; padding:0; }
  /* --- slide chuyển mục (section divider): nền navy + số mờ lớn --- */
  section.divider {
    background-color:#16294d !important;
    background-image: linear-gradient(135deg,#0e1d38 0%,#16294d 45%,#1F3A68 100%) !important;
    color:#eaf1fc; justify-content:center !important; align-content:center;
    padding:96px 80px; overflow:hidden;
  }
  section.divider .dnum { position:absolute; top:14px; right:50px;
    font-size:260px; font-weight:800; line-height:1;
    color:rgba(255,255,255,.06); letter-spacing:-6px; z-index:0; pointer-events:none; }
  section.divider .dbar { width:64px; height:6px; border-radius:3px; position:relative; z-index:1;
    background:linear-gradient(90deg,var(--accent),#86b4ff); margin:0 0 20px 0; }
  section.divider h1 { color:#ffffff !important; background:none; border:none; box-shadow:none;
    font-size:48px; line-height:1.12; margin:0 0 16px 0; padding:0; position:relative; z-index:1; max-width:80%; }
  section.divider .dsub { color:#cfe0ff; font-size:23px; line-height:1.5; max-width:80%; position:relative; z-index:1; }
  section.divider .dmeta { color:#9db4d8; font-size:18px; margin-top:30px; position:relative; z-index:1; }
  /* --- components --- */
  .small { font-size:18px; color:var(--muted); }
  .caption { font-size:16px; color:#888; font-style:italic; }
  .box, .warn { background:#f7faff; border:1px solid var(--line); border-left:5px solid var(--navy);
    border-radius:0 12px 12px 0; padding:13px 22px; }
  .grid2 { display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); gap:20px; align-items:start; }
  .grid2 > *, .col { min-width:0; }
  .cols { display:flex; gap:30px; } .col { flex:1; }
  .center { text-align:center; }
  .yes { color:#15803d; font-weight:700; } .no { color:#b04a4a; font-weight:700; }
  /* flow dọc các bước */
  .flow { display:flex; flex-direction:column; align-items:center; gap:5px; margin:14px 0; }
  .flow .step { background:#fff; border:1.5px solid #cdd9ec; border-radius:10px;
    padding:9px 22px; font-weight:600; color:var(--navy); font-size:21px; box-shadow:0 2px 8px rgba(15,29,56,.06); }
  .flow .step.fill { background:var(--navy); color:#fff; border-color:var(--navy); }
  .flow .ar { color:var(--accent); font-size:17px; line-height:1; }
  /* chips timeline ngang */
  .chips { display:flex; flex-wrap:wrap; align-items:center; gap:7px; justify-content:center; margin:8px 0 4px; }
  .chip { background:var(--navy); color:#fff; border-radius:999px; padding:6px 15px; font-size:18px; font-weight:600; }
  .chip.alt { background:#eef3fb; color:var(--navy); border:1px solid #cdd9ec; }
  .chip.hot { background:linear-gradient(135deg,var(--navy),var(--accent)); }
  .sep { color:#9bb0cf; font-weight:700; }
  /* minh hoạ mono (ma trận/sơ đồ chữ) */
  .mono { background:#0f1f3d; color:#dbe7ff; border-radius:12px; padding:16px 22px;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size:19px; line-height:1.55; box-shadow:0 4px 16px rgba(15,29,56,.22); display:inline-block; }
  .mono .dim { color:#7f93bd; }
  /* pipeline / pill */
  .pipeline { background:#f7faff; border:1px solid var(--line); border-radius:12px;
    padding:12px 16px; box-shadow:0 2px 12px rgba(15,29,56,.06);
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size:20px; line-height:1.65; color:#20324f; }
  .pill { display:inline-block; border:1px solid #cdd9ec; background:#fff; color:var(--navy);
    border-radius:999px; padding:3px 12px; margin:3px 4px 3px 0; font-size:18px; }
  /* sơ đồ (ảnh SVG pre-render từ Mermaid) */
  .diagram { text-align:center; margin-top:12px; }
  .diagram img { max-height:300px; width:auto; }
  /* khoảng cách dọc giữa các block component (tránh dính nhau) */
  .box, .warn, .pipeline, .grid2, .cols, pre, table { margin-top:16px; margin-bottom:16px; }
  .mono { margin:8px 0; }
  /* tiện ích thu nhỏ bảng/list khi slide dày */
  section.tight table { font-size:18.5px; } section.tight li { font-size:22px; }
  section.hardware .cols { gap:20px; }
  section.hardware .cols table { font-size:17px; }
  section.hardware .cols table th:first-child { width:34%; }
  section.hardware .cols table th:last-child { width:66%; }
  section.hardware > table { font-size:16px; }
  section.results .results-layout { display:grid; grid-template-columns:minmax(0,1.35fr) minmax(0,1fr);
    gap:22px; margin:16px 0; align-items:stretch; }
  section.results .result-card { min-width:0; min-height:320px; box-sizing:border-box;
    background:#f7faff; border:1px solid var(--line); border-radius:10px; padding:16px 20px; }
  section.results .result-heading { color:var(--navy); font-weight:700; font-size:22px; margin:0 0 12px; }
  section.results .result-placeholder { height:238px; box-sizing:border-box; border:2px dashed #bdd0ea;
    border-radius:8px; background:#fff; display:flex; flex-direction:column; align-items:center;
    justify-content:center; text-align:center; gap:8px; color:var(--muted); font-size:18px; }
  section.results .result-placeholder strong { font-size:22px; }
  section.results .result-copy li { font-size:19px; line-height:1.35; margin:10px 0; }
  section.evaluation .evaluation-layout { display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr);
    gap:24px; margin:16px 0 12px; align-items:start; }
  section.evaluation .evaluation-title { color:var(--navy); font-size:21px; font-weight:700; margin:0 0 10px; }
  section.evaluation .evaluation-data li { font-size:19px; line-height:1.35; margin:9px 0; }
  section.evaluation .evaluation-figure { margin:0; }
  section.evaluation .evaluation-figure img { width:100%; height:210px; object-fit:cover;
    object-position:center; border-radius:8px; display:block; }
  section.evaluation .metrics-row { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:14px; }
  section.evaluation .metric { min-width:0; background:#f7faff; border:1px solid var(--line);
    border-left:4px solid var(--navy); border-radius:0 8px 8px 0; padding:10px 13px;
    font-size:18px; line-height:1.3; }
  /* luồng ngang vẽ bằng div (thay cho Mermaid) */
  .lane { background:#f7faff; border:1px solid var(--line); border-radius:12px; padding:10px 14px 12px; margin:10px 0; }
  .lane .lt { font-size:15px; font-weight:700; color:var(--navy); margin:0 0 8px 2px; }
  .hflow { display:flex; align-items:center; gap:8px; }
  .hflow .node { flex:1; background:#fff; border:1.5px solid #cdd9ec; border-radius:10px; padding:8px 10px;
    font-size:16px; line-height:1.3; text-align:center; color:var(--ink); }
  .hflow .node.fill { background:var(--navy); color:#fff; border-color:var(--navy); }
  .hflow .node.ask { border:2px dashed var(--accent); color:var(--navy); font-weight:600; }
  .hflow .ar { flex:0 0 auto; color:var(--accent); font-size:20px; font-weight:700; }
  .hflow .branch { flex:1.25; display:flex; flex-direction:column; gap:6px; }
  .hflow .branch .node { font-size:15px; text-align:left; }
  .hflow .tag { display:inline-block; font-size:12px; font-weight:700; color:#fff; background:var(--accent);
    border-radius:6px; padding:1px 6px; margin-right:6px; }
  .hflow .tag.no { background:#9aa7b8; }
  .vg { flex:1; text-align:center; }
  .vg .cells { display:flex; gap:3px; justify-content:center; }
  .vg .cells i { display:block; width:13px; height:30px; border-radius:2px; }
  .vg .cap { font-size:14px; color:var(--muted); margin-top:4px; }
  .pair { flex:1; text-align:center; }
  .pair span { display:inline-block; border:2px dashed var(--accent); border-radius:999px; padding:2px 14px;
    font-size:14px; font-weight:700; color:var(--navy); background:#fff; }
footer: '<span>Nhóm 9</span><span>Điểm danh bằng giọng nói trên ESP32-S3</span><span>10/2026</span><span></span>'
header: '<img src="assets/UIT_logo.svg" alt="UIT">'
---

<!-- _class: lead cover -->
<!-- _paginate: false -->

<div class="titlebox">

# Hệ thống điểm danh bằng giọng nói trên vi điều khiển ESP32-S3 dùng mô hình nén
### Voice-Based Attendance on the ESP32-S3 Microcontroller with Compressed Speaker Verification Models

</div>

**Trần Tú Quang** 250201084 · **Tô Huỳnh Minh Tiến** 250201095 · **Nguyễn Ấn** 250201042

GVHD: TS. Nguyễn Tất Bảo Thiện

IT2058 · Báo cáo tiến độ đồ án môn học

<span class="small">University of Information Technology, VNU-HCM (UIT) · TP.HCM, 10/2026</span>

---

## Nội dung

<div class="grid2">
<div>

1. **Bối cảnh và mục tiêu**
2. **Phạm vi bài toán**
3. **Pipeline**
4. **Triển khai và kết quả**
5. **Đánh giá hướng đi**

</div>
<div>

<img src="assets/photo-auditorium.jpg" style="width:100%;height:320px;object-fit:cover;border-radius:12px;display:block">

</div>
</div>

---

<!-- footer: '<span>Nhóm 9</span><span>Phần 1 · Bối cảnh và mục tiêu</span><span>10/2026</span><span></span>' -->
<!-- _class: tight -->

## Bối cảnh và ý tưởng

<div class="grid2">
<div>

<img src="assets/photo-raise-hand.jpg" style="width:100%;height:330px;object-fit:cover;border-radius:12px;display:block">

</div>
<div>

- **Hiện nay:** gọi tên từng bạn mất thời gian; ký tên trên giấy lại dễ bị điểm danh hộ. Bài mẫu dùng khuôn mặt, nhưng cần camera, ánh sáng tốt và Jetson Nano
- **Ý tưởng:** sinh viên nói một câu ngắn trước một thiết bị nhỏ, thiết bị nhận ra là ai và ghi vào sổ điểm danh
- **Thiết bị:** vi điều khiển ESP32-S3, rẻ hơn nhiều so với Jetson Nano, rẻ hơn cả Raspberry Pi, tốn ít điện
- **Xử lý tại chỗ:** giọng nói không rời thiết bị, không cần mạng, có kết quả ngay

</div>
</div>

---

<!-- _class: tight -->

## Mục tiêu

<div class="grid2">
<div>

<img src="assets/photo-microphone.jpg" style="width:100%;height:300px;object-fit:cover;border-radius:12px;display:block">

</div>
<div>

1. **Thiết bị điểm danh chạy được trên chip:** nói 2–3 giây, có kết quả trong khoảng 1 giây, ghi vào sổ điểm danh trên laptop
2. **Model nhận giọng đủ nhỏ để nạp vào chip mà vẫn nhận đúng:** thử hai cách tạo model nhỏ, chọn cách nhận đúng hơn
3. **Kiểm tra độ tin cậy khi dùng thật:** mức "đủ giống" chọn trên máy tính có còn đúng khi model bị nén và khi dùng trong lớp học thật không

</div>
</div>

<div class="warn">

**Khó khăn chính:** model nhận giọng tốt hiện nay nặng khoảng 80MB, trong khi chip chỉ có 512KB bộ nhớ. Phải làm model nhỏ hơn vài trăm lần.

</div>

---

<!-- footer: '<span>Nhóm 9</span><span>Phần 2 · Phạm vi</span><span>10/2026</span><span></span>' -->

## Bài toán: điểm danh bằng giọng nói

<div class="diagram">

![h:300px](diagrams/scenario.svg)

</div>

<div class="tight">

- **Đầu vào:** 2–3 giây giọng nói, câu gợi ý "Em tên là [họ tên], có mặt."
- **Đầu ra:** tên, MSSV, giờ điểm danh, hoặc "không nhận ra, mời nói lại"
- **Cách dùng:** mặc định chỉ cần nói, thiết bị tự so với cả lớp. Nếu không chắc, giảng viên nhập MSSV để thiết bị so riêng với người đó

</div>

---

<!-- _class: tight -->

## Phạm vi bài toán

<div class="grid2">
<div>

**Trong phạm vi**
- Lớp hoặc nhóm tối đa khoảng 40 người, một thiết bị đặt tại lớp, mỗi lượt một người nói
- Phòng học bình thường, có ồn nền vừa phải
- Nhận giọng hoàn toàn trên chip; sổ điểm danh trên laptop, xuất được file CSV
- Một model nhỏ khoảng 300KB, chọn từ hai cách tạo model ở phần sau

</div>
<div>

**Ngoài phạm vi**
- Chống giả mạo (ghi âm phát lại, giọng giả bằng AI): chỉ đo và báo cáo như hạn chế
- Nhiều người nói cùng lúc, nhận dạng nội dung lời nói
- Kết nối với hệ thống quản lý đào tạo của trường
- Dùng cho an ninh hay kiểm soát ra vào

</div>
</div>

<div class="warn">

**Nguyên tắc:** kết quả là gợi ý, giảng viên vẫn kiểm tra lại được. Thiết bị chỉ lưu mẫu giọng (192 số mỗi người), không lưu file ghi âm. Chỉ thu giọng người tự nguyện có giấy đồng ý.

</div>

---

<!-- footer: '<span>Nhóm 9</span><span>Phần 3 · Pipeline</span><span>10/2026</span><span></span>' -->
<!-- _class: tight -->

## Model lớn và model nhỏ

<div class="cols" style="align-items:center">
<div class="col" style="flex:1.6">
<div class="hflow">
<div class="node fill" style="flex:1.15"><b>Model lớn</b><br>ECAPA-TDNN · khoảng 80MB<br>rất chính xác, chỉ chạy trên máy tính</div>
<div class="ar">→</div>
<div class="vg"><div class="cells"><i style="background:#3d557e"></i><i style="background:#a1afc8"></i><i style="background:#6f82a3"></i><i style="background:#bfcade"></i><i style="background:#29436f"></i><i style="background:#8394b2"></i><i style="background:#5b7094"></i><i style="background:#c9d3e5"></i><i style="background:#8d9db9"></i><i style="background:#475e85"></i><i style="background:#abb8cf"></i><i style="background:#65799b"></i><i style="background:#798baa"></i><i style="background:#b5c1d7"></i><i style="background:#334c76"></i><i style="background:#97a6c0"></i></div><div class="cap">dấu vân giọng của model lớn</div></div>
</div>
<div class="hflow" style="margin:8px 0"><div style="flex:1.15"></div><div class="ar" style="visibility:hidden">→</div><div class="pair"><span>↕ so cho giống nhau</span></div></div>
<div class="hflow">
<div class="node" style="flex:1.15;border:2px solid #1F3A68"><b>Model nhỏ</b><br>khoảng 300KB, nhỏ hơn vài trăm lần<br>chạy được trên chip ESP32-S3</div>
<div class="ar">→</div>
<div class="vg"><div class="cells"><i style="background:#475e85"></i><i style="background:#97a6c0"></i><i style="background:#798baa"></i><i style="background:#b5c1d7"></i><i style="background:#334c76"></i><i style="background:#8d9db9"></i><i style="background:#51678d"></i><i style="background:#bfcade"></i><i style="background:#8394b2"></i><i style="background:#51678d"></i><i style="background:#a1afc8"></i><i style="background:#6f82a3"></i><i style="background:#8394b2"></i><i style="background:#abb8cf"></i><i style="background:#3d557e"></i><i style="background:#8d9db9"></i></div><div class="cap">dấu vân giọng của model nhỏ</div></div>
</div>
</div>
<div class="col">
<div class="box" style="margin:0 0 12px 0;font-size:18px"><b style="color:#1F3A68">Cách A · Học theo model lớn</b><br>model nhỏ được chỉnh để cho ra dấu vân giọng giống model lớn (mũi tên ↕)</div>
<div class="box" style="margin:0;font-size:18px"><b style="color:#1F3A68">Cách B · Tự học</b><br>không dùng model lớn, model nhỏ tự học từ dữ liệu có ghi "giọng này của ai"</div>
</div>
</div>

<span class="caption">Dấu vân giọng (embedding): dãy 192 số tóm tắt đặc điểm giọng. Hai câu của cùng một người cho hai dãy số gần giống nhau.</span>

- **Thử thách:** model nhỏ hơn vài trăm lần thì nhận giọng kém đi. Phải giữ độ chính xác cao nhất có thể trong 300KB, và chạy xong dưới 1 giây trên chip không có GPU
- **Nén INT8:** lưu mỗi con số bằng 1 byte thay vì 4 byte, model nhỏ thêm 4 lần và chạy nhanh hơn, nhưng điểm số bị xê dịch chút ít
- **Hai cách:** model lớn đã học từ giọng của hàng nghìn người, học theo nó có thể giúp model nhỏ nhận đúng hơn tự học. Nhóm đo cả hai, cách nào tốt hơn thì nạp vào chip

---

<!-- _class: tight -->

## Xử lý trên thiết bị

<div class="lane">
<div class="lt">Đăng ký · một lần đầu khoá</div>
<div class="hflow">
<div class="node">Nói 3–5 câu<br>mỗi câu ~3 giây</div><div class="ar">→</div>
<div class="node">Bỏ khoảng lặng<br>tính phổ âm thanh</div><div class="ar">→</div>
<div class="node fill">Model nhỏ<br>ra dấu vân giọng</div><div class="ar">→</div>
<div class="node">Lấy trung bình<br>các câu</div><div class="ar">→</div>
<div class="node">Lưu mẫu giọng<br>vào chip</div>
</div>
</div>

<div class="lane">
<div class="lt">Điểm danh · mỗi lượt</div>
<div class="hflow">
<div class="node">Nói<br>2–3 giây</div><div class="ar">→</div>
<div class="node">Bỏ khoảng lặng<br>tính phổ âm thanh</div><div class="ar">→</div>
<div class="node fill">Model nhỏ<br>ra dấu vân giọng</div><div class="ar">→</div>
<div class="node">So với mẫu giọng<br>cả lớp</div><div class="ar">→</div>
<div class="node ask">Nhận diện<br>chắc chắn?</div><div class="ar">→</div>
<div class="branch">
<div class="node"><span class="tag">CÓ</span>Hiện tên, gửi về laptop</div>
<div class="node"><span class="tag no">KHÔNG</span>Mời nói lại</div>
</div>
</div>
</div>

- **Đủ giống:** điểm giống nhau vượt **ngưỡng**
- **Tránh nhầm người:** nếu hai bạn có điểm gần nhau, thiết bị mời nói lại. Sau hai lần chưa rõ, giảng viên nhập MSSV để kiểm tra riêng

---

<!-- _class: tight -->

## Ngưỡng là gì, vì sao có thể lệch

<div class="diagram">

![h:270px](diagrams/threshold-shift.svg)

</div>

- **Ngưỡng:** điểm giống nhau từ mức này trở lên thì nhận là đúng người. Chọn sao cho gần như không nhận nhầm người khác (≤ 1%)
- **Vì sao lệch:** nén model, micro rẻ và phòng học thật đều làm điểm số xê dịch. Giữ ngưỡng cũ thì nhiều bạn bị từ chối nhầm hơn
- **Nhóm sẽ đo:** lệch bao nhiêu, và chỉnh lại ngưỡng bằng chính giọng đã đăng ký của lớp có khắc phục được không

---

<!-- footer: '<span>Nhóm 9</span><span>Phần 4 · Triển khai và kết quả</span><span>10/2026</span><span></span>' -->
<!-- _class: tight hardware -->

## Phần cứng

<div class="cols">
<div class="col" style="flex:1.35">

| Linh kiện | Vai trò |
|---|---|
| ESP32-S3 N16R8 | Thiết bị chính (16MB flash, 8MB PSRAM) |
| INMP441 | Micro thu âm |
| OLED 0,96" | Màn hình hiện tên |
| Nút, buzzer | Bắt đầu lượt nói, báo kết quả |
| INA219 | Đo điện năng tiêu thụ |

</div>
<div class="col" style="flex:0.65">

<img src="assets/photo-breadboard.jpg" style="width:100%;height:200px;object-fit:cover;border-radius:12px;display:block;margin-top:10px">

</div>
</div>

| So sánh | Jetson Nano (bài mẫu) | Raspberry Pi | ESP32-S3 (đồ án) |
|---|---|---|---|
| Loại thiết bị | Máy tính nhúng có GPU | Máy tính nhúng | Vi điều khiển |
| Chi phí | Cao | Trung bình | **Thấp nhất** |
| Điện năng | Cao | Trung bình | **Thấp, chạy pin được** |

---

## Kiến trúc phần mềm

<div class="cols">
<div class="col box">

**Máy tính · huấn luyện**

- Tính phổ âm thanh (bản chuẩn)
- Chạy model lớn
- Train hai model nhỏ
- Nén INT8, chọn ngưỡng

</div>
<div class="col box">

**ESP32-S3 · chương trình chip**

- Thu âm, bỏ khoảng lặng
- Tính phổ âm thanh
- Chạy model nhỏ, so với mẫu
- Hiện tên, gửi kết quả

</div>
<div class="col box">

**Laptop · sổ điểm danh**

- Nhận kết quả từ thiết bị
- Lưu sổ, chặn ghi trùng
- Đăng ký giọng, nhập MSSV
- Xuất file CSV

</div>
</div>

<div class="chips">
<span class="chip alt">Máy tính: file model nhỏ</span><span class="sep">→</span><span class="chip hot">nạp vào chip ESP32-S3</span><span class="sep">→</span><span class="chip alt">kết quả điểm danh</span><span class="sep">→</span><span class="chip">laptop</span>
</div>

---

<!-- _class: tight results -->

## Hiện trạng và kết quả

<div class="results-layout">
<div class="result-card">
<div class="result-heading">Biểu đồ kết quả</div>
<div class="result-placeholder">
<strong>Chờ số liệu thực nghiệm</strong>
<span>Vị trí dành cho hình kết quả sau khi đo</span>
</div>
</div>
<div class="result-card result-copy">
<div class="result-heading">Kết quả bằng chữ</div>
<ul>
<li>Chưa có số đo để kết luận</li>
<li>Sẽ đối chiếu với mục tiêu: model ≤ 300KB, xử lý ≤ 1 giây</li>
<li>Sẽ báo cáo tỉ lệ nhận nhầm, từ chối nhầm và điện năng</li>
</ul>
</div>
</div>


---

<!-- _class: evaluation -->

## Cách đánh giá

<div class="evaluation-layout">
<div class="evaluation-data">

<div class="evaluation-title">Dữ liệu dùng để thử</div>

- **VoxCeleb1:** huấn luyện và so kết quả với các nghiên cứu trước
- **MUSAN:** thêm tiếng ồn để thử độ bền
- **Giọng thu thực tế:** 15–20 người tự nguyện, hai ngày khác nhau, trong phòng yên tĩnh và phòng có tiếng ồn

</div>
<div>

<div class="evaluation-figure">
<img src="assets/experiment-setup-chibi.png" alt="Minh hoạ chibi bố trí thử nghiệm điểm danh bằng giọng nói">
</div>

</div>
</div>

<div class="evaluation-title">Nhóm sẽ kiểm tra</div>
<div class="metrics-row">
<div class="metric"><strong>Model</strong><br>Sai số trước và sau khi nén</div>
<div class="metric"><strong>Điểm danh</strong><br>Người quen, người lạ, ghi âm phát lại</div>
<div class="metric"><strong>Thiết bị</strong><br>Thời gian, bộ nhớ và điện năng</div>
</div>

<div class="box">

**Câu hỏi chính:** ngưỡng đặt trên máy tính còn phù hợp khi model được nén và dùng trong lớp học thật không? Nếu không, mẫu giọng đã đăng ký có giúp điều chỉnh lại được không?

</div>

---

<!-- footer: '<span>Nhóm 9</span><span>Phần 5 · Đánh giá hướng đi</span><span>10/2026</span><span></span>' -->
<!-- _class: tight -->

## Đánh giá hướng đi

<div class="grid2">
<div>

**Điểm mạnh**
- Bài toán thật, cùng bài toán với bài mẫu của môn nhưng trên thiết bị rẻ hơn nhiều
- Giọng nói không rời thiết bị, chạy không cần mạng
- Vẫn có phần nghiên cứu: so hai cách tạo model nhỏ, đo độ tin cậy khi dùng thật

</div>
<div>

**Hạn chế**
- Model nhỏ kém chính xác hơn model lớn, sẽ có lượt phải nói lại
- Không chống được điểm danh hộ bằng ghi âm
- Thử nghiệm thực tế ít người, chỉ đo trên một thiết bị

</div>
</div>

<div class="box">

**Rủi ro chính:** lập trình chip là phần khó nhất. Nếu không kịp, nhóm chuyển sang Raspberry Pi Zero 2 W mà không phải làm lại model, vì model nhỏ thiết kế cho chip cũng chạy được trên Pi.

</div>

---

<!-- footer: '<span>Nhóm 9</span><span>Điểm danh bằng giọng nói trên ESP32-S3</span><span>10/2026</span><span></span>' -->

## Tài liệu tham khảo

<div class="small">

1. Desplanques et al. *ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification.* Interspeech 2020.
2. Nagrani et al. *VoxCeleb: A Large-Scale Speaker Identification Dataset.* Interspeech 2017.
3. Snyder et al. *MUSAN: A Music, Speech, and Noise Corpus.* arXiv 2015.
4. Hinton et al. *Distilling the Knowledge in a Neural Network.* 2015.
5. Deng et al. *ArcFace: Additive Angular Margin Loss for Deep Face Recognition.* CVPR 2019.
6. Jacob et al. *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference.* CVPR 2018.
7. Ravanelli et al. *SpeechBrain: A General-Purpose Speech Toolkit.* 2021.
8. Nguyen-Tat, Bui, Ngo. *Automating attendance management in human resources.* IJIM Data Insights, 2024.
9. Nguyen-Tat, Ngo. *Lite-XNet: An efficient deep learning model for COVID-19 lung severity assessment on embedded systems.* CVIU, 2026.
10. David et al. *TensorFlow Lite Micro: Embedded Machine Learning for TinyML Systems.* MLSys 2021.

Một số ảnh minh hoạ lấy từ Unsplash.

</div>

---

<!-- _class: lead -->

<div class="thanks">

# Cảm ơn Thầy và các bạn đã lắng nghe

</div>

**Trần Tú Quang · Tô Huỳnh Minh Tiến · Nguyễn Ấn**

<span class="small">University of Information Technology, VNU-HCM (UIT)</span>
