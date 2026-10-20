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
### Đối chiếu supervised và knowledge distillation

</div>

**Trần Tú Quang** 250201084 · **Tô Huỳnh Minh Tiến** 250201095 · **Nguyễn Ấn** 250201042

GVHD: TS. Nguyễn Tất Bảo Thiện

IT2058 · Báo cáo nghiên cứu và demo hệ thống

<span class="small">University of Information Technology, VNU-HCM (UIT) · TP.HCM, 10/2026</span>

---

## Phát biểu bài toán

<p style="font-size:26px;margin:10px 0 26px;">Cho một đoạn giọng nói và các mẫu đăng ký, xác định người nói phù hợp nhất.</p>

<div class="grid2" style="gap:46px;font-size:24px;">
<div>
<h3>Đầu vào</h3>

- Audio query **3 giây**, chuẩn hóa về mono **16 kHz**
- Mẫu đã đăng ký: **nhãn + vector 192 chiều** cho mỗi người

</div>
<div>
<h3>Đầu ra</h3>

- **Tên người có cosine cao nhất** trong các mẫu đăng ký
- **Điểm cosine với từng mẫu**, thể hiện độ tương đồng

</div>
</div>

<div style="border-top:2px solid #d4deee;padding-top:18px;margin-top:22px;font-size:21px;line-height:1.5;">
<strong>I/O của encoder:</strong> tensor INT8 [1, 298, 40, 1], đầu ra embedding INT8 [1, 192].<br />
<strong>Phân chia xử lý:</strong> Mac thu âm và làm frontend. ESP32-S3 chạy encoder và cosine.
</div>

<p class="small" style="margin-top:18px;">Demo closed-set với Quang, Tien, An. Chưa có ngưỡng từ chối người lạ. Cosine không phải xác suất.</p>
<!-- Source: scripts/demo_voice_usb.py (INPUT_BYTES, frontend, active labels), firmware/esp32_s3_smoke/main/main.cc (RECOGNIZE) and main/template_store.cc. Audio is normalized to mono 16 kHz before frontend; microphone acquisition may use a different sample rate. NVS stores normalized float32 templates, not raw audio. The encoder emits 192 int8 values; board-side dequantization/normalization precedes cosine and top1. -->

---

## Câu hỏi nghiên cứu

<div class="box" style="font-size:23px">

**Câu hỏi:** Với cùng kiến trúc student và dữ liệu huấn luyện, KD ảnh hưởng thế nào đến chất lượng speaker verification sau INT8 so với supervised?

</div>

<div style="margin-top:28px;">
<h3>Các bản so sánh</h3>

- Cùng baseline_conv8, cùng embedding 192 chiều
- So sánh supervised AAM với KD + AAM
- Đo cùng hai trạng thái FP32 và INT8
</div>

<div class="caption">Full100 đánh giá trên validation. Demo board gồm nhận diện closed-set và đăng ký thêm mẫu.</div>

---

<!-- _class: tight -->
## Thiết kế thực nghiệm

<div class="grid2" style="margin:8px 0;font-size:22px;">
<div class="box" style="padding:10px 18px;margin:0;">
<h3>Supervised AAM</h3>
<p>Student học nhãn người nói bằng AAM.<br />Xuất encoder riêng ở FP32 và INT8.</p>
</div>
<div class="box" style="padding:10px 18px;margin:0;">
<h3>KD + AAM</h3>
<p>Student học nhãn và khớp embedding teacher.<br />Teacher chạy offline, không lên board.</p>
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
<div class="grid2" style="margin:8px 0 6px;font-size:20px;line-height:1.35;">
<div>
<h3 style="margin-bottom:6px;">Biến so sánh</h3>
<p style="margin:4px 0;">Supervised so với KD<br />FP32 so với INT8</p>
</div>
<div>
<h3 style="margin-bottom:6px;">Giữ cố định</h3>
<p style="margin:4px 0;">baseline_conv8, embedding 192 chiều<br />Dữ liệu, frontend và validation protocol</p>
</div>
</div>
<div class="caption">Nguồn: full_training_report.json của hai run và src/edgeai/comparative_training.py.</div>
<!-- AAM reference: ArcFace, https://arxiv.org/abs/1801.07698. Standard per-sample form. The implementation includes numerical clamps and the monotonic fallback. -->


---

## Dữ liệu và cách chia

<div class="grid2" style="margin-top:30px;">
<div class="box">

<h3>Huấn luyện</h3>

- LibriSpeech train-clean-100
- **251 người nói**
- **28.539 bản ghi**

</div>
<div class="box">

<h3>Đánh giá (validation)</h3>

- **5 người nói**, **60 mẫu**
- Mỗi người: **3 đăng ký + 9 kiểm tra**
- Tổng cộng **45 mẫu kiểm tra**

</div>
</div>

<p style="font-size:23px; line-height:1.6; margin-top:20px;"><strong>Hai tập không trùng người nói.</strong><br>Kết quả hiện tại chỉ trên validation, chưa có test độc lập.</p>

<!-- Nguồn: artifacts/full100/20261015/data/manifest.json và independent_final_audit.json. Validation: 5 người từ dev-clean, 3 enrollment + 9 query/người. -->


---

## Teacher và student

<div class="grid2">
<div class="box">

<h3>Teacher</h3>

SpeechBrain · ECAPA-TDNN  
speechbrain/spkrec-ecapa-voxceleb

- Snapshot frozen, chạy offline
- Chỉ cung cấp teacher embedding cho KD
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

<div class="caption">Teacher snapshot là nguồn hướng dẫn offline, không phải dữ liệu student độc lập và không được nạp lên board.</div>

---

## Quantization và export

<div style="width:90%; margin:30px auto 0;">

| Thành phần | Giá trị artifact |
|---|---|
| Encoder FP32 | 587.132 byte |
| Encoder INT8 | **154.000 byte** |
| Input tensor | INT8, **11.920 byte** |
| Output tensor | INT8, 192 giá trị |

<div class="caption">Nguồn: full_training_report.json của hai run và board report ngày 19/10/2026. Byte dùng đơn vị decimal.</div>

</div>


---

## Sơ đồ thử nghiệm

<div class="lane">
<div class="lt">Đăng ký mẫu</div>
<div class="hflow">
<div class="node">Mac audio<br>+ frontend</div><div class="ar">→</div>
<div class="node">quant INT8<br>11.920 B</div><div class="ar">→</div>
<div class="node fill">USB<br>ESP32-S3</div><div class="ar">→</div>
<div class="node">baseline_conv8<br>encoder only</div><div class="ar">→</div>
<div class="node">NVS<br>mẫu đã đăng ký</div>
</div>
</div>

<div class="lane">
<div class="lt">Query và hiển thị</div>
<div class="hflow">
<div class="node">Mac audio<br>+ frontend</div><div class="ar">→</div>
<div class="node">quant INT8<br>11.920 B</div><div class="ar">→</div>
<div class="node fill">ESP32-S3<br>encoder 192</div><div class="ar">→</div>
<div class="node">cosine + đọc mẫu NVS<br>top1 closed-set · không ghi</div><div class="ar">→</div>
<div class="node">Mac display</div>
</div>
</div>

<div class="caption">Board không có microphone. Đăng ký ghi mẫu vào NVS, query chỉ đọc để so cosine.</div>

---

## Cách đánh giá

<div style="margin-top:18px;">
<p style="font-size:34px;margin:0 0 8px;"><strong>5 người nói</strong></p>
<p style="font-size:25px;margin:0;">Mỗi người: <strong>3 mẫu đăng ký</strong> + <strong>9 mẫu kiểm tra</strong></p>
</div>
<div style="display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:50px;margin-top:42px;padding-top:28px;border-top:2px solid #d4deee;">
<div>
<p style="font-size:36px;margin:0 0 14px;"><strong>EER</strong></p>
<p style="margin:0 0 12px;">Lỗi nhận nhầm và từ chối nhầm</p>
<p style="font-size:21px;color:#6b7a90;margin:0;">45 cặp cùng người + 180 cặp khác người</p>
<p style="font-size:26px;margin:22px 0 0;"><strong>Càng thấp càng tốt</strong></p>
</div>
<div>
<p style="font-size:36px;margin:0 0 14px;"><strong>Top-1</strong></p>
<p style="margin:0 0 12px;">Tỷ lệ chọn đúng người trong 5 người</p>
<p style="font-size:21px;color:#6b7a90;margin:0;">Số mẫu đúng / 45 mẫu kiểm tra</p>
<p style="font-size:26px;margin:22px 0 0;"><strong>Càng cao càng tốt</strong></p>
</div>
</div>
<p class="small" style="margin-top:36px;">Chỉ báo cáo kết quả trên tập validation.</p>


---

<!-- _class: tight -->

## Full100: kết quả validation

<table>
<thead>
<tr><th>Mục tiêu train</th><th>Encoder</th><th>EER</th><th>top1</th></tr>
</thead>
<tbody>
<tr><td>Supervised AAM</td><td>FP32</td><td><strong>4.4444%</strong></td><td>43/45</td></tr>
<tr><td>Supervised AAM</td><td>INT8</td><td><strong>4.4444%</strong></td><td>43/45</td></tr>
<tr><td>KD + AAM</td><td>FP32</td><td><strong>2.7778%</strong></td><td>45/45</td></tr>
<tr><td>KD + AAM</td><td>INT8</td><td><strong>4.1667%</strong></td><td>45/45</td></tr>
</tbody>
</table>

<div class="box">

Trong bốn cấu hình, **KD + AAM INT8** có EER **4.1667%** và top1 **45/45** trên validation slice. Cả hai INT8 encoder đều có kích thước **154.000 byte**.

</div>

<div class="caption">Bảng dùng cùng 5 speaker, 3 enrollment và 9 query mỗi speaker. Nguồn: frozen reports + independent final audit.</div>

---

## Kết quả demo trên ESP32-S3

<table style="font-size:24px;margin-top:30px;">
<colgroup><col style="width:32%;" /><col style="width:17%;" /><col style="width:17%;" /><col style="width:17%;" /><col style="width:17%;" /></colgroup>
<thead>
<tr><th rowspan="2">File kiểm tra</th><th colspan="3">Cosine với mẫu đăng ký</th><th rowspan="2">Board chọn</th></tr>
<tr><th>Tien.m4a</th><th>Quang.m4a</th><th>An.m4a</th></tr>
</thead>
<tbody>
<tr><td>Quang_test.m4a</td><td style="text-align:center;">0.6627</td><td style="text-align:center;"><strong>0.9130</strong></td><td style="text-align:center;">0.4884</td><td style="text-align:center;"><strong>Quang</strong></td></tr>
<tr><td>Tien_test.m4a</td><td style="text-align:center;"><strong>0.7994</strong></td><td style="text-align:center;">0.7014</td><td style="text-align:center;">0.4298</td><td style="text-align:center;"><strong>Tien</strong></td></tr>
</tbody>
</table>
<p class="small" style="margin-top:18px;">Board chọn người có điểm cosine cao nhất.</p>
<div class="grid2" style="border-top:2px solid #d4deee;padding-top:18px;margin-top:28px;font-size:22px;">
<p>Encoder: <strong>khoảng 53,5 ms/mẫu</strong></p>
<p>Tensor arena: <strong>38.140 / 131.072 byte</strong></p>
</div>
<div class="caption">Nguồn: đo trực tiếp trên ESP32-S3 (20/10). Điểm cosine đã làm tròn.</div>
<!-- Board evidence: artifacts/voice-demo/20261020/three-file-readonly-155417-874901/report.json. This table uses only the independent Quang_test.m4a and Tien_test.m4a query rows and all three cosine scores against the active Tien, Quang and An enrollment templates. The An enrollment self-check is excluded. Highest-score values are bold; Board top1 is read from the response. Frozen encoder parity passed; NVS fingerprint stayed crc32:60a382e2 before/after. Runtime is encoder-only; approximately 53.5 ms is the median of the two displayed queries. -->

---

## Kết luận

<div style="margin-top:24px;font-size:25px;line-height:1.5;">
<p style="margin:0 0 30px;"><strong>Chất lượng:</strong> KD + AAM INT8 đạt EER <strong>4.1667%</strong>, Top-1 <strong>45/45</strong> trên validation.</p>
<p style="margin:0 0 30px;"><strong>Triển khai:</strong> encoder INT8 <strong>154.000 byte</strong> chạy trên ESP32-S3, khoảng <strong>53,5 ms/mẫu</strong>.</p>
<p style="margin:0 0 30px;"><strong>Demo:</strong> nhận diện đúng hai file Quang/Tien và đăng ký thêm <strong>An</strong> vào NVS.</p>
</div>
<p class="small" style="margin-top:18px;">Kết quả hiện giới hạn ở validation và các mẫu demo đã kiểm tra.</p>
<!-- Reproducibility sources retained from the former provenance slide: artifacts/full100/20261015/data/manifest.json, baseline_supervised/full_training_report.json, baseline_kd/full_training_report.json, independent_final_audit.json (PASS_WITH_CAVEATS). Board: docs/demo-mac-usb.md; artifacts/voice-demo/20261019/persist-after-reboot-001/report.json; persist-oracle-001/report.json; artifacts/voice-demo/20261020/add-an-001/report.json. The NVS reset evidence applies to the original Quang/Tien record; no three-template reboot result is asserted here. -->

---

## Hướng phát triển tiếp theo

<div style="margin-top:22px;">
<div style="margin-bottom:26px;">
<h3 style="font-size:25px;">Kiểm thử độc lập</h3>
<p style="font-size:24px;">Thu query riêng cho An, thêm người nói và điều kiện thu âm.</p>
</div>
<div style="margin-bottom:26px;">
<h3 style="font-size:25px;">Nhận diện người lạ</h3>
<p style="font-size:24px;">Hiệu chỉnh ngưỡng cosine trên tập calibration riêng.</p>
</div>
<div>
<h3 style="font-size:25px;">Kiểm tra độ ổn định</h3>
<p style="font-size:24px;">Đo độ trễ toàn luồng và thử NVS qua nhiều lần tắt/bật.</p>
</div>
</div>
<!-- Proposed follow-up work derived from current limits: no independent test/calibration, no independent An query, no open-set evaluation, encoder-only timing, and limited NVS reset/durability evidence. These are development directions, not completed results. -->

---

<!-- _class: lead -->

<div class="thanks">

# Cảm ơn Thầy và các bạn đã lắng nghe

</div>

**Trần Tú Quang · Tô Huỳnh Minh Tiến · Nguyễn Ấn**

<span class="small">University of Information Technology, VNU-HCM (UIT)</span>
