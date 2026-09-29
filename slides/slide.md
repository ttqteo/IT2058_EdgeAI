---
marp: true
title: "Triển khai xác thực người nói trên vi điều khiển ESP32-S3 bằng nén mô hình"
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
  table { font-size: 21px; border-collapse: collapse; margin: 10px 0; width: 100%;
    box-shadow: 0 2px 10px rgba(15,29,56,.07); border-radius: 10px; overflow: hidden; }
  thead th { background: var(--navy); color: #fff; font-weight: 600; }
  tbody tr:nth-child(even) td { background: #f6f9fe; }
  td, th { border: 1px solid var(--line); padding: 8px 14px; }
  blockquote {
    border: none; border-left: 5px solid var(--accent);
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
  .box { background:#f7faff; border:1px solid var(--line); border-left:5px solid var(--accent);
    border-radius:0 12px 12px 0; padding:13px 22px; box-shadow:0 2px 12px rgba(15,29,56,.06); }
  .warn { background:#fff8ec; border:1px solid #f3dca6; border-left:5px solid #e0a51e;
    border-radius:0 12px 12px 0; padding:13px 22px; }
  .grid2 { display:grid; grid-template-columns:1fr 1fr; gap:20px; align-items:start; }
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
  .tight table { font-size:18.5px; } .tight li { font-size:22px; }
footer: '<span>Nhóm: Quang · Tiến · Ấn</span><span>Xác thực người nói trên ESP32-S3</span><span>29/09/2026</span><span></span>'
header: '<img src="assets/UIT_logo.svg" alt="UIT">'
---

<!-- _class: lead cover -->
<!-- _paginate: false -->

<div class="titlebox">

# Triển khai xác thực người nói trên vi điều khiển ESP32-S3 bằng nén mô hình
### Speaker Verification on the ESP32-S3 Microcontroller via Model Compression

</div>

**Trần Tú Quang** 250201084 · **Tô Huỳnh Minh Tiến** 250201095 · **Nguyễn Ấn** 250201042

GVHD: ........................

IT2058 · Báo cáo tiến độ đồ án môn học

<span class="small">University of Information Technology, VNU-HCM (UIT) · TP.HCM, 29/09/2026</span>

---

## Giới thiệu đề tài

- **Bối cảnh:** các ứng dụng Edge AI hiện thường chạy trên máy tính nhúng như Jetson Nano, ví dụ điểm danh bằng khuôn mặt [8] và Lite-XNet [9]. Nhóm chọn làm ở lớp thấp hơn là **vi điều khiển ESP32-S3**, giá khoảng 250k, chỉ có **512KB SRAM** và CPU **240MHz**.
- **Bài toán:** xác thực người nói, tức xác định *ai đang nói* và từ chối người lạ. Bài toán này hợp để làm trên thiết bị biên vì dữ liệu giọng nói là dữ liệu sinh trắc không nên gửi lên server, kết quả cần có dưới 1 giây, và hệ thống phải chạy được cả khi mất mạng.
- **Thách thức:** mô hình tốt hiện nay là ECAPA-TDNN [1], khoảng **20 triệu tham số**, không thể nạp vào vi điều khiển. Nhóm phải nén mô hình xuống còn khoảng **300KB** bằng các kỹ thuật của môn học: chưng cất tri thức [4], tỉa kênh [5] và lượng tử hoá INT8 [6].
- **Điểm cần chú ý:** hệ thống quyết định bằng cách **so điểm tương đồng với một ngưỡng**. Sau khi lượng tử hoá, điểm số có thể bị lệch, nên ngưỡng chọn cho bản FP32 có thể không còn đúng với bản INT8. Nhóm sẽ đo và hiệu chuẩn lại ngưỡng.

---

## Mục tiêu đồ án

- **Mục tiêu 1. Triển khai trên ESP32-S3:** nén mô hình xác thực người nói xuống khoảng **300KB INT8** và chạy real-time với micro I2S, trong đó log-mel được tính dạng streaming để không tràn SRAM.
- **Mục tiêu 2. So sánh ba cách nén cùng ngân sách:** chưng cất embedding từ teacher, tỉa kênh rồi tinh chỉnh, và train thẳng mô hình nhỏ làm mốc dưới. Cả ba dùng chung kiến trúc student, quy trình INT8 và board đo.
- **Mục tiêu 3. Đánh giá trên thiết bị:** đo độ chính xác, độ trễ, RAM và điện năng của từng mô hình trên board thật, xem lượng tử hoá INT8 làm giảm chất lượng bao nhiêu và hiệu chuẩn lại ngưỡng để bù.

<div class="warn">

**Phạm vi:** sản phẩm gồm một hệ thống demo chạy trên ESP32-S3 và bảng đánh giá, chưa phải hệ thống kiểm soát ra vào để dùng thật.

</div>

---

## Hướng thực hiện

<div class="diagram">

![h:200px](diagrams/pipeline.svg)

</div>

<div class="tight">

- **Teacher:** ECAPA-TDNN pretrained (SpeechBrain), chạy trên PC, EER khoảng 1% trên VoxCeleb1-O. Đây là mốc trên.
- **KD:** ép embedding của student bám theo teacher, kết hợp với loss phân loại người nói.
- **Tỉa kênh + FT:** tỉa dần kênh của teacher, đo lại sau mỗi mức rồi tinh chỉnh.
- **Train thẳng:** train student từ đầu bằng AAM-softmax, không dùng teacher. Đây là mốc dưới.
- **INT8 chung:** lượng tử hoá sau huấn luyện, giữ lại cả bản FP32 và INT8 của mỗi đường để tách được hai loại mất mát.

</div>

---

## Hệ thống trên thiết bị

<div class="diagram">

![h:130px](diagrams/device.svg)

</div>

<div class="grid2">
<div>

**Ràng buộc và cách xử lý**
- 3 giây audio 16kHz dạng float đã chiếm **192KB**, nên phải xử lý streaming theo từng khung
- Trọng số để trong flash, tensor arena trong SRAM, chỉ tràn sang PSRAM khi cần
- FFT dùng ESP-DSP, kernel INT8 dùng ESP-NN

</div>
<div>

**Phần cứng (~450k)**
- ESP32-S3 N16R8 (16MB flash, 8MB PSRAM) · 250k
- 2 micro INMP441 (I2S) · 100k
- INA219 đo dòng để tính µJ mỗi lần suy luận · 50k
- Breadboard, dây cắm · 50k

</div>
</div>

---

## Dữ liệu và đánh giá

<div class="grid2">
<div>

**Dữ liệu (đều công khai)**
- **VoxCeleb1** [2]: dùng để train, bộ thử là **VoxCeleb1-O** gốc nên so được với số đã công bố
- **MUSAN, RIR:** nhiễu nền ở nhiều mức SNR cố định, vang phòng
- **ASVspoof 2019 PA** [3]: tấn công phát lại

</div>
<div>

**Độ đo**
- **EER** (chính), **minDCF**, đường **DET**
- **Độ dịch ngưỡng:** FAR và FRR lệch bao nhiêu khi dùng ngưỡng FP32 cho model INT8
- **Trên thiết bị:** kích thước, độ trễ p50/p95, RAM đỉnh, µJ mỗi lần suy luận

</div>
</div>

<div class="box">

**Ma trận thí nghiệm:** 3 cách nén × 2 dạng (FP32, INT8) × 4 điều kiện (sạch, nhiễu, vang, replay). Dùng chung seed và trial list, người nói trong bộ thử không có trong tập train, ngưỡng lấy từ tập val.

</div>

---

## Kết quả dự kiến

1. **Hệ thống chạy real-time trên ESP32-S3:** xác thực người nói từ micro I2S với model INT8 khoảng 300KB, kèm số đo độ trễ, RAM và năng lượng.
2. **Bảng so sánh ba cách nén:** EER, minDCF, kích thước, độ trễ và năng lượng, ở cả dạng FP32 và INT8, kèm đường DET.
3. **Ảnh hưởng của lượng tử hoá INT8:** mức lệch FAR và FRR, và phần bù được khi hiệu chuẩn lại.
4. **Đánh giá độ bền:** mức suy giảm dưới nhiễu nền, vang phòng và tấn công phát lại cho từng cách nén.
5. **Demo trực tiếp:** người đã đăng ký được chấp nhận, người lạ và bản ghi âm phát lại bị từ chối.

---

## Tài liệu tham khảo

<div class="small">

1. Desplanques et al. *ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification.* Interspeech 2020.
2. Nagrani et al. *VoxCeleb: A Large-Scale Speaker Identification Dataset.* Interspeech 2017.
3. Todisco et al. *ASVspoof 2019: Future Horizons in Spoofed and Fake Audio Detection.* Interspeech 2019.
4. Hinton et al. *Distilling the Knowledge in a Neural Network.* 2015.
5. Li et al. *Pruning Filters for Efficient ConvNets.* ICLR 2017.
6. Jacob et al. *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference.* CVPR 2018.
7. David et al. *TensorFlow Lite Micro: Embedded Machine Learning for TinyML Systems.* MLSys 2021.
8. Nguyen-Tat, Bui, Ngo. *Automating attendance management in human resources.* IJIM Data Insights, 2024.
9. Nguyen-Tat, Ngo. *Lite-XNet: An efficient deep learning model for COVID-19 lung severity assessment on embedded systems.* CVIU, 2026.

</div>

---

<!-- _class: lead -->

<div class="thanks">

# Cảm ơn thầy/cô và các bạn đã lắng nghe

</div>

**Trần Tú Quang · Tô Huỳnh Minh Tiến · Nguyễn Ấn**

<span class="small">University of Information Technology, VNU-HCM (UIT)</span>
