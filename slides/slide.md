---
marp: true
title: "Điểm danh bằng giọng nói trên ESP32-S3: so sánh supervised và KD"
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
footer: '<span>Nhóm 9</span><span>Speaker verification trên ESP32-S3</span><span>10/2026</span><span></span>'
header: '<img src="assets/UIT_logo.svg" alt="UIT">'
---

<!-- _class: lead cover -->
<!-- _paginate: false -->

<div class="titlebox">

# Điểm danh bằng giọng nói trên ESP32-S3
### So sánh supervised và knowledge distillation

</div>

**Trần Tú Quang** 250201084 · **Tô Huỳnh Minh Tiến** 250201095 · **Nguyễn Ấn** 250201042

GVHD: TS. Nguyễn Tất Bảo Thiện

IT2058 · Báo cáo nghiên cứu và demo hệ thống

<span class="small">University of Information Technology, VNU-HCM (UIT) · TP.HCM, 10/2026</span>

---

<!-- _class: tight -->

## Bối cảnh và bài toán

<div class="grid2">
<div>

<img src="assets/photo-raise-hand.jpg" style="width:100%;height:330px;object-fit:cover;border-radius:12px;display:block">

</div>
<div>

- **Bối cảnh:** điểm danh bằng giọng nói trong danh sách đã đăng ký.
- **Đầu vào:** PCM16 mono 16 kHz, **3 giây** và mẫu đăng ký gồm nhãn + vector.
- **Đầu ra:** tên có cosine cao nhất cùng điểm của từng người.
- **Phạm vi:** chưa từ chối người lạ; cosine là độ tương đồng.

</div>
</div>
<!-- Giữ nguyên ảnh, kích thước và bố cục hai cột của slide bối cảnh. Gộp input/output từ slide 9 cũ; định nghĩa chỉ số chuyển sang phần đánh giá. -->

---

<!-- _class: tight -->

## Mục tiêu và câu hỏi nghiên cứu

<div class="grid2">
<div>

<img src="assets/photo-microphone.jpg" style="width:100%;height:300px;object-fit:cover;border-radius:12px;display:block">

</div>
<div>

1. **Chất lượng mô hình:** KD thay đổi EER và Top-1 thế nào so với supervised, trước và sau INT8?
2. **Chi phí trên board:** giảm được bao nhiêu thời gian và RAM tiền xử lý khi giữ nguyên đầu ra?

</div>
</div>

<div class="warn">

**Cách đối chiếu:** cùng student, dữ liệu và quy trình đánh giá. Đo tiền xử lý trên ESP32-S3 với bản tham chiếu.

</div>
<!-- Giữ nguyên ảnh, kích thước và bố cục slide mục tiêu. Gộp câu hỏi nghiên cứu từ slide 10 cũ; thông số bộ nhớ chỉ trình bày ở phần cứng. -->

---

## Teacher và student

<div class="grid2">
<div class="box">

<h3>Teacher</h3>

SpeechBrain · ECAPA-TDNN<br>
speechbrain/spkrec-ecapa-voxceleb

- Giữ cố định trọng số teacher
- Tạo embedding cho nhánh KD khi huấn luyện
- Không chạy trên ESP32-S3

</div>
<div class="box">

<h3>Student</h3>

<p style="font-size:22px;margin:2px 0 8px;"><strong>baseline_conv8</strong> <span style="font-size:18px;">(1 lớp convolution, 8 kênh)</span></p>
<p style="font-size:19px;margin:0 0 8px;">Input: [1, 298, 40, 1]</p>

<div class="flow" style="gap:2px;margin:4px 0 10px;" role="img" aria-label="Kiến trúc baseline: Conv2D tám kênh và ReLU, Average Pooling, Flatten 760, Dense 192 tạo embedding.">
<div class="step" style="width:100%;box-sizing:border-box;padding:5px 12px;font-size:20px;line-height:1.25;">Conv2D 8 kênh + ReLU<br><span style="font-size:16px;font-weight:400;">Kernel 3 × 3, stride 2 × 2</span></div>
<div class="ar" style="font-size:16px;line-height:1;">↓</div>
<div class="step" style="width:100%;box-sizing:border-box;padding:5px 12px;font-size:20px;line-height:1.25;">Average Pooling<br><span style="font-size:16px;font-weight:400;">Pool 8 × 4, stride 8 × 4</span></div>
<div class="ar" style="font-size:16px;line-height:1;">↓</div>
<div class="step" style="width:100%;box-sizing:border-box;padding:5px 12px;font-size:20px;line-height:1.25;">Flatten: 760 giá trị</div>
<div class="ar" style="font-size:16px;line-height:1;">↓</div>
<div class="step fill" style="width:100%;box-sizing:border-box;padding:5px 12px;font-size:20px;line-height:1.25;">Dense 192: embedding [1, 192]</div>
</div>

<p style="font-size:18px;margin:6px 0 0;">AAM head: <strong>chỉ dùng lúc train</strong>, không lên board.</p>
<!-- Architecture source: src/edgeai/training.py:410 build_student_model, reused by src/edgeai/full_training.py:929. Conv2D filters=8, kernel=(3,3), strides=(2,2), SAME; ReLU; AveragePooling2D pool/strides=(8,4), SAME; Flatten 19*5*8=760; Dense 192, linear activation. AAM classifier is external to this encoder. -->

</div>
</div>

<div class="caption">Teacher tạo embedding cho dữ liệu train. Chỉ student được xuất và nạp lên ESP32-S3.</div>

---

<!-- _class: tight -->
## Huấn luyện với AAM và KD

<div class="grid2" style="margin:8px 0;font-size:22px;">
<div class="box" style="padding:10px 18px;margin:0;">
<h3>Supervised AAM</h3>
<p>AAM dùng nhãn người nói để cập nhật student.</p>
</div>
<div class="box" style="padding:10px 18px;margin:0;">
<h3>KD + AAM</h3>
<p>Giữ loss AAM, thêm loss cosine giữa embedding student và teacher.</p>
</div>
</div>
<div style="margin:14px 0 6px;">
<p style="font-size:22px;margin:0;"><strong>Hàm mất mát AAM</strong> <span class="small">(chỉ dùng khi train)</span></p>
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block" style="font-size:31px;margin:12px 0;color:#1F3A68;" aria-label="L AAM bằng âm log của exp s cos theta y cộng m, chia cho exp s cos theta y cộng m cộng tổng exp s cos theta j với j khác y">
<mrow>
<msub><mi mathvariant="script">L</mi><mtext>AAM</mtext></msub><mo>=</mo><mo>−</mo><mi mathvariant="normal">log</mi><mspace width="0.12em"/>
<mfrac>
<msup><mi mathvariant="normal">e</mi><mrow><mi>s</mi><mspace width="0.12em"/><mi mathvariant="normal">cos</mi><mo>(</mo><msub><mi>θ</mi><mi>y</mi></msub><mo>+</mo><mi>m</mi><mo>)</mo></mrow></msup>
<mrow>
<msup><mi mathvariant="normal">e</mi><mrow><mi>s</mi><mspace width="0.12em"/><mi mathvariant="normal">cos</mi><mo>(</mo><msub><mi>θ</mi><mi>y</mi></msub><mo>+</mo><mi>m</mi><mo>)</mo></mrow></msup>
<mo>+</mo>
<munderover><mo>∑</mo><mrow><mi>j</mi><mo>=</mo><mn>1</mn><mo>,</mo><mi>j</mi><mo>≠</mo><mi>y</mi></mrow><mi>C</mi></munderover>
<msup><mi mathvariant="normal">e</mi><mrow><mi>s</mi><mspace width="0.12em"/><mi mathvariant="normal">cos</mi><msub><mi>θ</mi><mi>j</mi></msub></mrow></msup>
</mrow>
</mfrac>
</mrow>
</math>
<p style="font-size:18px;margin:6px 0;color:#6b7a90;">θ<sub>j</sub>: góc giữa embedding và vector lớp j, y: nhãn đúng, C = 251 lớp người nói.</p>
<p style="font-size:18px;margin:6px 0;color:#6b7a90;"><strong>m = 0,2 rad</strong>: biên góc, <strong>s = 30</strong>: hệ số nhân logit.</p>
</div>
<p style="font-size:22px;margin:10px 0;"><strong>KD:</strong> L = L<sub>AAM</sub> + λ · [1 − cos(z<sub>student</sub>, z<sub>teacher</sub>)], λ = 1.</p>
<div class="caption">Khi suy luận, bỏ AAM và teacher; chỉ dùng encoder và cosine với mẫu đăng ký.</div>
<!-- AAM reference: ArcFace, https://arxiv.org/abs/1801.07698. Standard per-sample form; implementation includes clamps and monotonic fallback. KD is the batch mean of 1-cosine; displayed formula is per sample. Sources: artifacts/full100/20261015/baseline_{supervised,kd}/full_training_report.json (aam_margin=.2, aam_scale=30, kd_weight=0/1), src/edgeai/comparative_training.py. -->

---

<!-- _class: tight -->

## Dữ liệu và quy trình đánh giá

<div class="grid2" style="gap:30px;margin-top:6px;">
<div>
<h3>Huấn luyện</h3>
<p>LibriSpeech <strong>train-clean-100</strong><br><strong>251 người nói · 28.539 bản ghi</strong></p>
</div>
<div>
<h3>Validation · tách người nói khỏi train</h3>
<p><strong>5 người · 60 mẫu</strong><br>Mỗi người: 3 đăng ký + 9 query</p>
</div>
</div>

<table style="margin-top:25px;">
<colgroup><col style="width:18%;"><col style="width:59%;"><col style="width:23%;"></colgroup>
<thead><tr><th>Chỉ số</th><th>Cách tính trên validation</th><th>Mục tiêu</th></tr></thead>
<tbody>
<tr><td><strong>EER</strong></td><td>Mức lỗi khi nhận nhầm bằng từ chối nhầm<br><span style="font-size:18px;">45 cặp cùng người + 180 cặp khác người</span></td><td>Thấp hơn</td></tr>
<tr><td><strong>Top-1</strong></td><td>Số query chọn đúng người / 45 query</td><td>Cao hơn</td></tr>
</tbody>
</table>

<p class="small" style="margin-top:20px;">Validation dùng để chọn checkpoint và báo cáo kết quả; chưa có test độc lập.</p>
<!-- Merges old slides 12/16 and evaluation side of 6. Enrollment template aggregates 3 embeddings per validation speaker; these 5 speakers differ from the board's demo labels. Sources: artifacts/full100/20261015/data/manifest.json, both full_training_report.json, independent_final_audit.json. -->

---

<!-- _class: tight -->

## Full100: kết quả validation

<table>
<thead>
<tr><th>Cách huấn luyện</th><th>Encoder</th><th>EER</th><th>Top-1</th></tr>
</thead>
<tbody>
<tr><td>Supervised AAM</td><td>FP32</td><td><strong>4.4444%</strong></td><td>43/45</td></tr>
<tr><td>Supervised AAM</td><td>INT8</td><td><strong>4.4444%</strong></td><td>43/45</td></tr>
<tr><td>KD + AAM</td><td>FP32</td><td><strong>2.7778%</strong></td><td>45/45</td></tr>
<tr><td>KD + AAM</td><td>INT8</td><td><strong>4.1667%</strong></td><td>45/45</td></tr>
</tbody>
</table>

<div class="box">

KD + AAM có EER thấp hơn ở FP32. Sau INT8, EER tăng từ **2.7778% lên 4.1667%**, trong khi Top-1 vẫn **45/45** trên validation.

</div>

<div class="caption">So sánh cùng student và protocol. Nguồn: hai báo cáo FULL100 và audit độc lập.</div>

---

<!-- _class: tight hardware -->

## Phần cứng

<div class="cols">
<div class="col" style="flex:1.35">

| Thiết bị đang dùng | Vai trò |
|---|---|
| ESP32-S3 | Tiền xử lý, encoder, cosine và lưu mẫu |
| Mic của Mac | Thu PCM 3 giây, mono 16 kHz |
| Native USB | Truyền PCM và kết quả |
| Dashboard trên Mac | Điều khiển demo, hiển thị số đo |

</div>
<div class="col" style="flex:0.65">

<img src="assets/photo-breadboard.jpg" style="width:100%;height:200px;object-fit:cover;border-radius:12px;display:block;margin-top:10px">
<p class="caption">Ảnh minh họa, không phải bộ demo đã lắp.</p>

</div>
</div>

**ESP32-S3:** 512 KB SRAM nội bộ, 16 MB Flash, 8 MB PSRAM.

<p class="small">Thực nghiệm dùng SRAM nội bộ, chưa bật PSRAM.</p>
<!-- Hardware sources: artifacts/esp32-smoke/20261018/preflash-record.json; artifacts/esp32-smoke/20261023/enrollment-deploy-001/security-before.log. Active sdkconfig disables CONFIG_SPIRAM. Flash is persistent storage, separate from SRAM and PSRAM. -->

<p class="small">Chưa gắn INMP441. Mic Mac là đầu vào tạm thời; chưa đo điện năng.</p>

---

<!-- _class: tight -->

## Luồng xử lý PCM trên board

<div class="lane" style="margin-top:8px;">
<div class="lt">Mac thu và gửi PCM thô qua USB</div>
<div class="hflow">
<div class="node">PCM 3 giây<br>48.000 mẫu</div><div class="ar">→</div>
<div class="node">ESP32-S3<br>Log-mel + chuẩn hóa</div><div class="ar">→</div>
<div class="node">Lượng tử hóa<br>INT8</div><div class="ar">→</div>
<div class="node fill">Encoder<br>embedding 192</div>
</div>
</div>

<div class="grid2" style="gap:30px;margin-top:26px;">
<div>
<h3>Đăng ký · thao tác riêng</h3>
<p>Chuẩn hóa embedding → lưu <strong>nhãn + vector</strong> trong NVS.</p>
<p class="small">LS116 và LS700 dùng luồng đăng ký PCM này. NVS không lưu file âm thanh.</p>
</div>
<div>
<h3>Nhận diện · mỗi query</h3>
<p>Cosine với mẫu NVS → chọn tên → gửi <strong>tên và điểm</strong> về Mac.</p>
<p class="small">Query chỉ đọc mẫu; dashboard hiển thị kết quả và số đo.</p>
</div>
</div>

<div class="caption" style="margin-top:20px;">Ứng dụng Mac không tính đặc trưng, embedding hay cosine. Huấn luyện và xuất model thực hiện trước khi demo.</div>
<!-- Merges old slides 5/8; original Quang/Tien/An enrollment used legacy tensor paths. Sources: main.cc RECOGNIZE_PCM/APPEND_PCM; demo_dashboard.py; evidence/enrollment/20261023/runtime-verification.json. Capture format handling by OS/device is outside application DSP scope. -->

---

## Kích thước model và bộ nhớ

<table style="margin-top:20px;">
<colgroup><col style="width:42%;"><col style="width:36%;"><col style="width:22%;"></colgroup>
<thead><tr><th>Thành phần</th><th>Kích thước</th><th>Trong demo</th></tr></thead>
<tbody>
<tr><td>Encoder FP32</td><td>587.132 byte</td><td>Không nạp</td></tr>
<tr><td>Encoder INT8 · đã nạp</td><td><strong>154.000 byte</strong></td><td>Flash</td></tr>
<tr><td>Tensor arena · dùng / cấp phát</td><td><strong>38.140 / 131.072 byte</strong></td><td>SRAM</td></tr>
<tr><td>NVS · 5 nhãn và vector</td><td><strong>3.972 byte</strong></td><td>Flash</td></tr>
</tbody>
</table>

<p style="font-size:23px;margin-top:26px;">Heap nội bộ khi chờ: <strong>183.192 byte trống / 239.908 byte tổng vùng đo</strong>.</p>
<p class="small">Khi xử lý còn cần buffer PCM, workspace frontend và stack. Arena không phải tổng RAM sử dụng.</p>
<div class="caption">Heap là snapshot STATUS, chưa phải RAM đỉnh. Bản ghi NVS chưa tính overhead lưu trữ.</div>
<!-- Model size: baseline_kd/full_training_report.json. Arena/NVS/heap: artifacts/board-audit/20261027-144805-live/report.json and artifacts/voice-demo/20261023/enroll-status-after-reboot-001/report.json. Workspace comparison appears only on optimization slide. FP32 is a comparison file, not the deployed firmware. -->

---

## Tối ưu tiền xử lý trên ESP32-S3

| Số đo | Trước tối ưu | Sau tối ưu |
|---|---:|---:|
| Frontend · Quang / Tien | 7,659 / 7,656 s | **2,854 / 2,851 s** |
| Nhận diện trên board | 7,715 / 7,712 s | **2,909 / 2,907 s** |
| Gửi PCM đến nhận kết quả qua USB | 8,101 / 8,099 s | **3,218 / 3,222 s** |
| Workspace của frontend | 10.568 byte | **8.512 byte** |

- Tính sẵn Hann, hệ số FFT và thứ tự đảo bit; chỉ duyệt **494 trọng số mel khác 0**.
- Giữ phép tính double. Trên hai query này, đầu ra encoder PCM trùng đường tensor cũ.

<p class="small">Frontend giảm khoảng 62,7%. Bản tối ưu không có cảnh báo watchdog trong hai lượt đo; chưa đánh giá chạy liên tục.</p>
<div class="caption">Hai PCM cố định; riêng encoder ≈53,6 ms. Thời gian USB chưa gồm 3 giây thu âm.</div>
<!-- Sources: docs/pcm-optimization.md; artifacts/esp32-smoke/20261020/pcm-deploy-002/pcm-check-002/report.json and 20261022/pcm-optimization-candidate-001/pcm-check-001/002. Host parity: evidence/esp32-smoke/20261022/pcm-optimization-local/parity-report.json, 8 synthetic/real inputs, exact features/int8 vs reference. No general latency or live-mic accuracy claim. -->

---

## Kết quả demo trên ESP32-S3

<p style="font-size:22px;">Năm mẫu trong NVS: <strong>Quang, Tien, An, LS116, LS700</strong>.</p>
<table style="font-size:21px;margin-top:16px;">
<colgroup><col style="width:30%;" /><col style="width:24%;" /><col style="width:23%;" /><col style="width:23%;" /></colgroup>
<thead>
<tr><th>Query</th><th>Gửi tới board</th><th>Board chọn</th><th>Cosine cao nhất</th></tr>
</thead>
<tbody>
<tr><td>Quang_test</td><td>Tensor cũ</td><td><strong>Quang</strong></td><td>0,9130</td></tr>
<tr><td>Tien_test</td><td>Tensor cũ</td><td><strong>Tien</strong></td><td>0,7994</td></tr>
<tr><td>LS116 · query 1 / 2</td><td>PCM thô</td><td><strong>LS116 / LS116</strong></td><td>0,8450 / 0,8094</td></tr>
<tr><td>LS700 · query 1 / 2</td><td>PCM thô</td><td><strong>LS700 / LS700</strong></td><td>0,7734 / 0,7375</td></tr>
</tbody>
</table>
<p class="small">Bốn query PCM dùng câu khác mẫu đăng ký, từ LibriSpeech dev-other. Board nhận diện trong danh sách năm người.</p>
<p class="small">Hai tensor cũ bỏ qua frontend. Sáu query cố định chưa đủ kết luận độ chính xác chung hay chất lượng mic trực tiếp.</p>
<div class="caption">Nguồn: bản ghi USB sau khởi động lại, 23/10/2026. Điểm đã làm tròn.</div>
<!-- Sources: artifacts/voice-demo/20261023/public-queries-after-reboot-001/report.json (four held-out PCM queries); artifacts/demo-dashboard/20261023-042246-c75ec747-69b40ba0/report.json (two legacy tensors). Labels/CRC persist across reboot, crc32:90fb081d. Public split/source/license bindings: evidence/public-demo/20261023/prepared-001/manifest.json. Six examples are not a benchmark and no held-out An query is asserted. -->

---

## Kết luận

<div style="margin-top:22px;">
<h3>Kết quả nghiên cứu</h3>
<p style="font-size:26px;line-height:1.5;">KD + AAM cho kết quả tốt hơn supervised trong phép so sánh hiện tại. INT8 làm EER của nhánh KD tăng dù Top-1 giữ nguyên.</p>
</div>

<div style="margin-top:28px;">
<h3>Kết quả triển khai</h3>
<p style="font-size:26px;line-height:1.5;">ESP32-S3 thực hiện toàn bộ xử lý PCM và nhận diện. Tối ưu frontend rút ngắn thời gian, giữ đầu ra trên các mẫu đã kiểm tra.</p>
</div>

<p class="small" style="margin-top:30px;">Phần nhóm thực hiện: đối chiếu cách huấn luyện, kiểm chứng INT8 và đo chi phí triển khai trên board thật.</p>
<!-- Synthesizes results without repeating tables. Method families AAM, KD, quantization and precomputation/sparse mel are established techniques; no new recognition algorithm is claimed. -->

---

## Giới hạn và bước tiếp theo

<table style="margin-top:24px;font-size:23px;">
<colgroup><col style="width:47%;"><col style="width:53%;"></colgroup>
<thead><tr><th>Giới hạn hiện tại</th><th>Bước kiểm chứng tiếp theo</th></tr></thead>
<tbody>
<tr><td>Validation nhỏ, đã dùng chọn checkpoint</td><td>Test độc lập, thêm người nói và lặp lại huấn luyện</td></tr>
<tr><td>Chưa từ chối người ngoài danh sách</td><td>Hiệu chỉnh ngưỡng trên tập riêng, đo lỗi chấp nhận / từ chối</td></tr>
<tr><td>Mic Mac, xử lý từng đoạn PCM</td><td>Gắn mic I2S, đo chất lượng thu và độ trễ toàn luồng</td></tr>
<tr><td>Các lượt đo ngắn, query cố định</td><td>Thử nhiễu, theo dõi RAM và độ ổn định khi chạy liên tục</td></tr>
</tbody>
</table>
<!-- Consolidates scope limitations and future work from old slides 6/19/20. Planned checks are not completed claims; INMP441 is a candidate I2S microphone, not attached hardware. -->

---

## Tài liệu tham khảo

<div class="small">

1. Desplanques et al. *ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification.* Interspeech 2020.
2. Nagrani et al. *VoxCeleb: A Large-Scale Speaker Identification Dataset.* Interspeech 2017.
3. Panayotov et al. *Librispeech: An ASR Corpus Based on Public Domain Audio Books.* ICASSP 2015.
4. Hinton et al. *Distilling the Knowledge in a Neural Network.* 2015.
5. Deng et al. *ArcFace: Additive Angular Margin Loss for Deep Face Recognition.* CVPR 2019.
6. Jacob et al. *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference.* CVPR 2018.
7. Ravanelli et al. *SpeechBrain: A General-Purpose Speech Toolkit.* 2021.
8. David et al. *TensorFlow Lite Micro: Embedded Machine Learning for TinyML Systems.* MLSys 2021.
9. Nguyen-Tat, Bui, Ngo. *Automating attendance management in human resources.* IJIM Data Insights, 2024.
10. Nguyen-Tat, Ngo. *Lite-XNet: An efficient deep learning model for COVID-19 lung severity assessment on embedded systems.* CVIU, 2026.

Ảnh minh họa: Unsplash. Dữ liệu demo LS116/LS700: LibriSpeech dev-other, CC BY 4.0.

</div>

---

<!-- _class: lead -->

<div class="thanks">

# Cảm ơn Thầy và các bạn đã lắng nghe

</div>

**Trần Tú Quang · Tô Huỳnh Minh Tiến · Nguyễn Ấn**

<span class="small">University of Information Technology, VNU-HCM (UIT)</span>
