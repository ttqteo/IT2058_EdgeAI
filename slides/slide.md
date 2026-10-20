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

## Câu hỏi nghiên cứu

<div class="box" style="font-size:23px">

**Câu hỏi:** (1) KD giữ chất lượng speaker verification sau INT8 thế nào so với supervised? (2) Sau reset, hệ thống có đọc lại đúng các template từ NVS không?

</div>

<div class="grid2">
<div>

<h3>Đối chiếu có kiểm soát</h3>

- Cùng baseline_conv8, cùng embedding 192 chiều
- So sánh supervised AAM với KD + AAM
- Đo cùng hai trạng thái FP32 và INT8

</div>
<div>

<h3>Đóng góp trong phạm vi</h3>

- Reproducible full-corpus training trên LibriSpeech train-clean-100
- Validation metrics có audit độc lập
- Demo thật trên ESP32-S3 với NVS persistence

</div>
</div>

<div class="caption">Hai nguồn bằng chứng giữ riêng phạm vi: full100 là validation-only, board demo là closed-set hai người.</div>

---

<!-- _class: tight -->
## Thiết kế thực nghiệm

<div class="pipeline center">Cùng dữ liệu · cùng frontend · cùng baseline_conv8 · embedding 192</div>

<div class="grid2">
<div class="box">

<h3>Supervised AAM</h3>

Student học theo nhãn speaker. AAM chỉ dùng lúc train; xuất encoder riêng để đo FP32/INT8.

</div>
<div class="box">

<h3>KD + AAM</h3>

Student học nhãn speaker và khớp embedding teacher. Teacher chạy offline, không lên board.

</div>
</div>

<div class="grid2">
<div>

<h3>Biến so sánh</h3>

- Objective: supervised so với KD
- Export: FP32 so với INT8

</div>
<div>

<h3>Giữ cố định</h3>

- baseline_conv8 và embedding 192 chiều
- Dữ liệu, frontend và validation protocol

</div>
</div>

<div class="caption">Nguồn: artifacts/full100/20261015/{baseline_supervised,baseline_kd}/full_training_report.json.</div>

---

## Dữ liệu và cách chia

<div class="grid2">
<div class="box">

<h3>Full train</h3>

- LibriSpeech train-clean-100
- **251 speaker**, **28.539 utterance**
- **135.438** cửa sổ liên tiếp; tail ngắn cuối zero-pad, không đồng nghĩa tất cả là 3 giây audio gốc
- Dùng cho cả hai run full training

</div>
<div class="box">

<h3>Validation</h3>

- **60 dòng**, **5 speaker** tách khỏi train
- Mỗi speaker: **3 enrollment + 9 query**
- **45 target trials + 180 non-target trials**
- Tổng cộng **45 query** cho top1

</div>
</div>

<div class="warn">

**Phạm vi:** đây là validation-only diagnostic. Artifact ghi rõ calibration và test chưa chạy, nên không gọi đây là test độc lập.

</div>

<div class="caption">Nguồn: artifacts/full100/20261015/data/manifest.json và independent_final_audit.json.</div>

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

baseline_conv8

- Input: [1, 298, 40, 1]
- Output: embedding [1, 192]
- AAM head chỉ dùng lúc train

</div>
</div>

<div class="mono" style="display:block;text-align:center">audio window → frontend → [1, 298, 40, 1] → encoder → [1, 192]</div>

<div class="caption">Teacher snapshot là nguồn hướng dẫn offline, không phải dữ liệu student độc lập và không được nạp lên board.</div>

---

## Quantization và export

<div class="grid2">
<div>

| Thành phần | Giá trị artifact |
|---|---|
| Encoder FP32 | 587.132 byte |
| Encoder INT8 | **154.000 byte** |
| Input tensor | INT8, **11.920 byte** |
| Output tensor | INT8, 192 giá trị |

</div>
<div class="box">

<h3>Contract triển khai</h3>

- Hai run đều xuất encoder INT8 154.000 byte
- Board nhận tensor đã lượng tử hóa từ Mac
- Board trả embedding INT8, sau đó chuẩn hóa thành float32 để lưu template
- Kết quả full100 vẫn được đọc theo scope host validation

</div>
</div>

<div class="caption">Nguồn: full_training_report.json của hai run và board report ngày 19/10/2026. Byte dùng đơn vị decimal.</div>

---

## Mapping Mac ↔ ESP32-S3

<div class="lane">
<div class="lt">Đăng ký, một lần</div>
<div class="hflow">
<div class="node">Mac audio<br>+ frontend</div><div class="ar">→</div>
<div class="node">quant INT8<br>11.920 B</div><div class="ar">→</div>
<div class="node fill">USB<br>ESP32-S3</div><div class="ar">→</div>
<div class="node">baseline_conv8<br>encoder only</div><div class="ar">→</div>
<div class="node">NVS<br>2 template</div>
</div>
</div>

<div class="lane">
<div class="lt">Query và hiển thị</div>
<div class="hflow">
<div class="node">Mac audio<br>+ frontend</div><div class="ar">→</div>
<div class="node">quant INT8<br>11.920 B</div><div class="ar">→</div>
<div class="node fill">ESP32-S3<br>encoder 192</div><div class="ar">→</div>
<div class="node">cosine + đọc 2 template NVS<br>top1 closed-set · không ghi</div><div class="ar">→</div>
<div class="node">Mac display</div>
</div>
</div>

<div class="caption">Board không có microphone. KD + AAM chỉ ở train; enrollment ghi NVS một lần, query đọc hai template để matching.</div>

---

## Metrics và protocol

<div class="grid2">
<div class="box">

<h3>Full100 validation</h3>

- EER trên 45 target và 180 non-target trials
- top1 trên 45 query
- Enrollment prototype: trung bình 3 embedding mỗi speaker
- Chỉ báo cáo validation-only

</div>
<div class="box">

<h3>Board evidence</h3>

- 2 query cố định: Quang_test, Tien_test
- 5 ANSI C fixtures, max LSB **[1, 0, 0, 0, 1]**
- Arena dùng **38.140 / 131.072 byte**
- Encoder-only invoke khoảng **53,5 ms**

</div>
</div>

<div class="warn">

**Không gộp hai scope:** EER/top1 của full100 không phải accuracy tổng quát của board. Board report chỉ chứng minh đường chạy closed-set và persistence đã quan sát.

</div>

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

## Đọc kết quả: tác động của INT8

<div class="grid2">
<div class="box">

<h3>Supervised AAM</h3>

FP32 → INT8  
EER **4.4444% → 4.4444%**  
top1 **43/45 → 43/45**

</div>
<div class="box">

<h3>KD + AAM</h3>

FP32 → INT8  
EER **2.7778% → 4.1667%**  
top1 **45/45 → 45/45**

</div>
</div>

<div class="warn">

Trong validation slice này, KD + AAM tốt hơn supervised ở cả FP32 và INT8. INT8 làm EER của KD tăng, nhưng top1 không đổi trên 45 query.

</div>

<div class="caption">Đây là kết luận của phép đối chiếu có kiểm soát, không phải tuyên bố về speaker population khác.</div>

---

## Demo board: hai query closed-set

<table>
<thead>
<tr><th>Query</th><th>Board top1</th><th>Cosine đúng</th><th>Cosine còn lại</th></tr>
</thead>
<tbody>
<tr><td>Quang_test.m4a</td><td><strong>Quang</strong></td><td>0.912962556</td><td>Tien · 0.662654161</td></tr>
<tr><td>Tien_test.m4a</td><td><strong>Tien</strong></td><td>0.799418211</td><td>Quang · 0.701422989</td></tr>
</tbody>
</table>

<div class="grid2">
<div class="box">

**Board runtime**  
ESP_NN_ANSI_C_REFERENCE  
encoder-only invoke: **53.422–53.562 ms**

</div>
<div class="box">

**Memory evidence**  
arena dùng: **38.140 / 131.072 byte**  
output: 192 byte INT8

</div>
</div>

<div class="caption">Nguồn: artifacts/voice-demo/20261019/persist-after-reboot-001/report.json. Hai query cố định, không phải benchmark độ chính xác tổng quát.</div>

---

## Demo board: persistence sau reset

<div class="grid2">
<div>

| Trạng thái | Quan sát |
|---|---|
| Boot source | NVS_BOOT_READONLY |
| Persistence | READY |
| Record | **1.640 byte** |
| Fingerprint | crc32:a00ff6a4 |
| Labels | Quang, Tien |

</div>
<div class="box">

<h3>Record đã đọc lại</h3>

- Hai template normalized float32
- Mỗi template có 192 giá trị
- Board tự nạp record sau reset
- Demo run không gửi lại enrollment

</div>
</div>

<div class="warn">

Cosine tính độc lập từ vector đọc lại NVS lệch tối đa **1.55e-8**. Điều này xác nhận record và matching của hai template demo khớp nhau sau reset.

</div>

<div class="caption">Nguồn: docs/demo-mac-usb.md, persist-after-reboot-001/report.json, persist-oracle-001/report.json.</div>

---

## Giới hạn kết luận

<div class="grid2">
<div>

<h3>Đã có bằng chứng</h3>

- Hai objective được train trên cùng full corpus
- Validation metrics được independent audit đối chiếu
- INT8 encoder chạy qua USB trên ESP32-S3
- NVS record được đọc lại sau một lần reset có kiểm soát

</div>
<div>

<h3>Chưa thể kết luận</h3>

- Chưa có independent test hoặc calibration
- Chưa đo open-set và dữ liệu hiện trường
- Board demo chỉ có hai người và hai query cố định
- Chưa đo độ bền dài hạn của record flash

</div>
</div>

<div class="box">

**Cách đọc an toàn:** full100 trả lời câu hỏi so sánh model trên validation slice. Board evidence trả lời câu hỏi tích hợp encoder, cosine và NVS trong closed set.

</div>

---

## Tái lập và provenance

<div class="grid2">
<div class="box">

<h3>Full100</h3>

data/manifest.json  
baseline_supervised/full_training_report.json  
baseline_kd/full_training_report.json  
independent_final_audit.json

<span class="small">Cùng manifest, cùng validation split, audit PASS_WITH_CAVEATS.</span>

</div>
<div class="box">

<h3>Board demo</h3>

docs/demo-mac-usb.md  
persist-after-reboot-001/report.json  
persist-oracle-001/report.json

<span class="small">USB report lấy top1 và cosine từ board, Mac oracle chỉ để đối chiếu.</span>

</div>
</div>

<div class="caption">Các pointer đều thuộc artifact local của run 15–19/10/2026. Không dùng thêm dữ liệu, inference, training hay hardware action trong lần sửa slide này.</div>

---

## Kết luận

<div class="box" style="font-size:23px">

1. **KD + AAM INT8** đạt EER **4.1667%**, top1 **45/45** trên validation slice, trong khi supervised INT8 đạt **4.4444%**, top1 **43/45**.

2. Pipeline query thực tế đã nối **Mac frontend → USB 11.920 B → ESP32-S3 encoder → đọc hai template từ NVS → cosine closed-set top1 → Mac display**.

3. Enrollment ghi hai normalized float32 template vào NVS một lần; sau reset, board tự đọc lại mà không cần gửi lại enrollment. Kết luận hiện giới hạn ở validation slice và demo hai người.

</div>

<div class="caption">Đóng góp chính: controlled comparison, full-corpus reproducibility, quantization evidence và actual-board persistence.</div>

---

<!-- _class: lead -->

<div class="thanks">

# Cảm ơn Thầy và các bạn đã lắng nghe

</div>

**Trần Tú Quang · Tô Huỳnh Minh Tiến · Nguyễn Ấn**

<span class="small">University of Information Technology, VNU-HCM (UIT)</span>
