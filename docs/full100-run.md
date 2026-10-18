# Full train-clean-100 — host-first run, 15/10/2026

## Phạm vi đã khóa

Yêu cầu: huấn luyện trên đầy đủ train-clean-100, chạy TensorFlow Lite Micro
trên host trước; tích hợp board là bước riêng. Không phải train toàn bộ
LibriSpeech khoảng 1.000 giờ. Không thay đổi artifact pilot trước đây.

Nguồn chính thức: [OpenSLR SLR12](https://www.openslr.org/12/), CC BY 4.0.
Archive local:
`artifacts/datasets/librispeech-train-clean-100/download/train-clean-100.tar.gz`.
MD5 chính thức: `2a93770f6d5c6c964bc36631d331a522`.

Kiểm kê độc lập của Lead ngày 15/10, đọc header bằng libsndfile sau MD5 gate,
không dùng validator chuẩn bị dữ liệu mới:

| Đại lượng | Giá trị |
|---|---:|
| Archive members | 29.966 |
| Bản ghi FLAC | 28.539 |
| Người nói train | 251 |
| Sample âm thanh nguồn, 16 kHz | 5.794.034.668 |
| Giờ âm thanh nguồn | 100,5908796528 |
| Cửa sổ train 48.000 sample | 135.438 |
| Sample padding tổng cộng | 706.989.332 |

Mỗi bản ghi được chia liên tiếp, không overlap, thành cửa sổ 3 giây; cửa sổ
cuối được zero-pad bên phải. Mỗi sample nguồn được phủ đúng một lần, không
chỉ lấy một crop đầu file. Padding không được tính thành giờ âm thanh nguồn.
ID cửa sổ phải gắn với ID bản ghi nguồn và offset sample.

Validation giữ nguyên 60 hàng / 5 người của pilot dev-clean: theo ID đã sort,
3 utterance enrollment và 9 query mỗi người. Không trộn 20 speaker train cũ
của dev-clean vào train mới. Train và validation phải speaker-disjoint.
Validation này đã được dùng nhiều lần và nhỏ; không phải independent final
test. Không tải, suy luận hay chọn cấu hình theo calibration/test trong run này.

## Hai phương pháp

- Encoder baseline Conv8 → ReLU → AvgPool → Flatten → Dense192,
  cùng initialization và classifier 251 lớp chỉ dùng lúc train.
- `baseline_supervised`: AAM-softmax CE; không mở hoặc hash teacher artifact.
- `baseline_kd`: AAM-softmax CE + 1,0 × cosine KD; teacher cố định/stop-gradient.
- Teacher: SpeechBrain `speechbrain/spkrec-ecapa-voxceleb`, snapshot
  `0f99f2d0ebe89ac095bcc5903c4dd8f72b367286`, chỉ dùng offline với hash đã khóa.
- Seed 20261012; Adam 0,001; AAM margin 0,2, scale 30, easy-margin false;
  tối đa 80 epoch, patience 15, chọn EER validation nhỏ nhất, giữ epoch sớm
  nhất khi hòa. Không đổi tham số sau khi xem kết quả validation của run này.
- Sampler phải dùng đủ mọi hàng train đúng một lần mỗi epoch, không ép mỗi
  speaker về 20 hàng như pilot. Ghi rõ sampling/weighting thực tế trong report.

## Bộ nhớ và kiểm chứng

Máy hiện có Apple M2 Pro, 16 GiB RAM. PCM int16 payload khoảng 13,00 GB và
features float32 khoảng 6,46 GB; phải dùng mmap/batch, không nạp cả PCM và
features vào RAM. Chỉ materialize batch được dùng. Teacher embedding float64
L2-norm check có tolerance 1e-5; finite/nonzero và ID/order binding bắt buộc.

Checkpoint dài hạn phải kèm optimizer, encoder, classifier, epoch/history,
best checkpoint, cấu hình và hash nguồn/dữ liệu. Resume không được thay
nguồn/cấu hình hoặc ghi đè kết quả đã hoàn tất. Dữ liệu chuẩn bị chỉ được
publish sau khi counts, coverage, hashes và nguồn bất biến qua kiểm tra.

Sau freeze encoder: xuất FP32 và full INT8, representative data chỉ train;
check static budget, dynamic/fixed batch-1 parity; chạy TFLM host thật và
đối chiếu builtin-reference desktop không delegate. TFLM upstream commit:
`230c9d9531bd9b08d445b4f2ed0760756ffd5f78`.

Runner hiện có `artifacts/tflm-host/edgeai_tflm_host` đã qua kiểm tra sidecar
`BUILD_BOUND` ngày 15/10. Đây là kiểm tra runtime, không xác nhận model mới
trước khi model mới thực sự được train/export/invoke.

Tensor arena cap 131.072 byte là giới hạn arena host, không phải tổng SRAM
board. Model-size gate 300.000 byte, Conv/Dense MAC gate 50 triệu. Không suy
ra board latency, công suất hoặc deployment suitability từ kết quả host.

Google mô tả workflow train → convert → C++ inference trong
[LiteRT for Microcontrollers](https://developers.google.com/edge/litert/microcontrollers/overview).
Board integration và final test độc lập vẫn là các bước sau.

## Probe thời gian (train-only, không phải metric chất lượng)

Batch 16 / CPU 4 threads có cliff: 22,14 giây cho 16 cửa sổ. Không dùng
ngoại suy này cho batch khác. Probe tiếp theo với batch 4 trên 64 cửa sổ
train thực, sau warm-up: 1,53968625 giây, khoảng 24,06 ms/cửa sổ; mỗi batch
0,0905–0,1049 giây. Khóa teacher batch 4 / CPU 4 threads. Ngoại suy khoảng
0,91 giờ inference thuần; I/O, hash, nhiệt độ và tải máy có thể làm lâu hơn.
Không dùng timing này làm latency board.

## Trạng thái lần launch đầu (ghi chép lịch sử)

Pipeline thật đã được launch lúc **21:44:45 ngày 15/10/2026 (Asia/Ho_Chi_Minh)**
trong terminal Paseo `92e7d172-8655-47c8-a8f6-1a788446785d`, dùng `caffeinate -i`
để giữ máy hoạt động trong thời gian process chạy. Pha chuẩn bị đã hoàn tất
135.438 cửa sổ. `independent_data_audit.json` ghi
`INDEPENDENT_DATA_AUDIT_PASS`: counts/coverage/padding khớp oracle, không có
feature row toàn số 0, năm PCM/frontend replays khớp bit.

Tại lần kiểm tra **21:51**, supervised AAM đã thực sự xử lý **256/6.772 batch
của epoch 1**, tức 5.120 hàng train. Chưa có epoch hoàn tất, model export hay
kết quả TFLM mới. Teacher/KD chưa chạy. Kết quả thực thi tiếp theo phải đọc từ
báo cáo/job status, không từ mốc tiến độ đã chụp này.

## Điều phối và theo dõi

Entrypoint: `scripts/run_full100.py`. `--dry-run` chỉ in lệnh, không train hay
ghi artifact. Chạy thật dùng thư mục mới `artifacts/full100/20261015/`; không
ghi đè pilot. Trình tự cố định:

1. Chuẩn bị toàn bộ cửa sổ vào mmap.
2. Audit độc lập bằng raw tar/libsndfile: coverage mọi source sample, ID,
   tổng frame/padding, hash, replay PCM và frontend ở năm cửa sổ.
3. Huấn luyện supervised AAM trong process TensorFlow mới.
4. TFLM host: synthetic fixtures và năm cửa sổ train thật, đối chiếu desktop
   builtin-reference không delegate, sai lệch tối đa 1 LSB.
5. Trích teacher theo shard có resume trong runtime PyTorch riêng.
6. Huấn luyện KD+AAM trong process TensorFlow mới; cùng gate TFLM host.

`run_status.json` ghi pha đang chạy, command, log, lỗi và hash output hoàn tất.
Log riêng từng lần thử nằm trong `logs/`; không ghi đè log cũ. `--resume` chỉ
tiếp tục khi config/source/input và output đã hoàn tất vẫn khớp. Checkpoint
training giữ trạng thái optimizer theo epoch; teacher giữ các shard hợp lệ.
Không được đổi source/hyperparameter để tiếp tục cùng run.

### Giới hạn resume đã được review độc lập

Review `FULL100-REVIEW-03` không thấy mismatch chặn lần launch đầu, nhưng
phát hiện khe hở giữa việc child publish thư mục kết quả cuối và parent ghi
pha `COMPLETE`. Nếu process bị ngắt đúng khoảng này, kết quả cuối có thể đã
tồn tại trong khi `run_status.json` vẫn ghi pha đang chạy; `--resume` sẽ từ
chối output đã tồn tại. **Không tuyên bố resume tự động an toàn tại ranh giới
publish này.** Lead chấp nhận đối soát thủ công nếu tình huống đó xảy ra:

1. Xác nhận không còn writer/process của run; giữ nguyên mọi output/log và
   sao lưu status trước khi phục hồi. Không xóa output hoặc train lại đè lên.
2. Kiểm tra độc lập report/config/freeze, toàn bộ hash/size checkpoint và
   artifact, cùng source/input bindings so với run đã khóa. Có thư mục cuối
   không tự nó chứng minh pha thành công.
3. Chỉ khi tất cả binding và báo cáo hoàn tất hợp lệ mới ghi nhận phục hồi
   pha kèm bằng chứng và tiếp tục; nếu chưa đủ thì giữ trạng thái gián đoạn
   để điều tra. Không sửa source/config hoặc bỏ qua gate để ép resume.

Đây là giới hạn phục hồi đã biết, không phải lỗi đã xuất hiện ở lần chạy này.

Trước khi launch thật phải hoàn tất review các module mới và kiểm tra giao
diện CLI. Các test điều phối dùng subprocess giả không chứng minh training
hoặc chất lượng trên dữ liệu thật.

### Kiểm tra tích hợp ngày 15/10

- `.venv/bin/python -B -m pytest -q`: **257 passed, 1 skipped**.
- Module chuẩn bị mới đọc toàn bộ archive thật thành công: 29.966 members,
  28.539 FLAC, 251 người nói, 5.794.034.668 sample nguồn. SHA256 archive:
  `d4ddd1d5a6ab303066f14971d768ee43278a5f2a0aa43dc716b0e64ecbbbf6e2`.
- Loader validation thật: 60 hàng `[60,298,40,1]`; kiểm tra WAV hash và
  containment cho đúng 60 hàng này, không suy luận các split còn lại.
- TensorFlow process mới cấu hình luồng thành công; teacher snapshot pin
  và sidecar runner `BUILD_BOUND`, không có lỗi hash.
- Sửa ánh xạ embedding validation theo thứ tự ID sorted; regression đảo
  hàng giữ nguyên metric. Checkpoint giữ EER float64 để không đổi lựa chọn
  earliest-tie do làm tròn sang float32. Thêm test KD trainer synthetic.

Lệnh launch trong terminal lâu dài (không thay đổi thiết lập nguồn điện toàn máy):

```sh
caffeinate -i .venv/bin/python -B -u scripts/run_full100.py
```

Theo dõi `artifacts/full100/20261015/run_status.json`, các log pha trong
`logs/`, và `baseline_supervised.stage/progress.json` hoặc
`baseline_kd.stage/progress.json` trong lúc train. Cập nhật batch mỗi 256
batch; mỗi epoch dùng 6.772 batch (batch cuối 18 hàng). Trích teacher ghi
tiến độ theo shard 256 hàng. Các chỉ số trong test synthetic không phải
kết quả full-corpus; chỉ report sau chạy thật mới là kết quả của run này.

Heartbeat Paseo `f6e2d6d7` của lần chạy đầu đã được gỡ sau khi xác nhận job
bị gián đoạn ngày 16/10. Không tự đổi cấu hình, retry mù hoặc
xóa output khi lỗi. `caffeinate -i` không bảo đảm job sống qua reboot, tắt
máy hoặc đóng nắp laptop; cần giữ máy hoạt động và đủ dung lượng đĩa.

## Phục hồi ngày 18/10/2026

Job cũ mất process/terminal dù status còn `RUNNING`. Log dừng ở epoch 14,
batch 2.304/6.772; checkpoint bền vững gần nhất là epoch 13. Không tìm được
bằng chứng xác định nguyên nhân process bị dừng. Không xóa output hay
checkpoint, không đổi hyperparameter hoặc nguồn runtime.

Sau yêu cầu tiếp tục của người dùng, đã kiểm tra SHA256 của 16 file nguồn,
history và toàn bộ latest/best checkpoint, rồi chạy lệnh:

```sh
caffeinate -i .venv/bin/python -B -u scripts/run_full100.py --resume
```

Runner kiểm tra lại source/input và output của từng pha hoàn tất trước khi
tiếp tục KD. Resume khôi phục encoder, classifier, optimizer và trạng thái
early stopping từ epoch 13; epoch 14 chưa hoàn tất được chạy lại toàn bộ.
Log mới là `logs/kd.002.log`; log cũ được giữ nguyên.

### Giới hạn provenance nguồn: đối soát thủ công

Danh sách hash tự động của runner và trainer không bao phủ toàn bộ import
closure. Cụ thể, `src/edgeai/matching.py` được import bởi `evaluation.py`
và `model.py` nhưng không có trong `SOURCE_NAMES` hoặc source bindings của
checkpoint. SHA256 quan sát khi review bản phát hành ngày 18/10/2026 là:

```text
26aee62f32a13394629215a302cf650d2d1d54eaa6edf1f2dc550de6d0371b8e  src/edgeai/matching.py
```

Bản nguồn ở checkout chạy và worktree phát hành khớp hash này tại thời điểm
review; file phải được giữ nguyên và đối chiếu lại trước khi công bố.
Đây là pin quan sát bổ sung, không chứng minh ngược rằng file không thay
đổi từ lần launch đầu. Vì vậy không tuyên bố resume có provenance tự động
đầy đủ cho mọi dependency nguồn. Giữ nguyên runtime đang chạy; chấp nhận
giới hạn đối soát thủ công này thay vì sửa danh sách hash giữa thí nghiệm.

## Dựng môi trường và tái lập

Các lệnh dưới đây là hướng dẫn dựng máy mới, không phải lệnh đã được chạy
lại trên dữ liệu trong lần phục hồi. Hai môi trường phải tách biệt:

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-full100.txt
.venv/bin/python -m pip install -e .
python3.11 -m venv .venv-teacher
.venv-teacher/bin/python -m pip install -r requirements-teacher.txt
```

`requirements-macos-py311-lock.txt` ghi môi trường TensorFlow macOS arm64 đã
quan sát, không phải lock đa nền tảng. Runtime teacher có SoundFile 0.14.0;
runtime chuẩn bị dữ liệu có SoundFile 0.13.1. Resume còn yêu cầu version
runtime khớp chính xác cấu hình lưu trong checkpoint.

Cần tự chuẩn bị các input không commit vào Git:

- Archive train-clean-100 đúng MD5 ở trên.
- Bộ validation pilot dev-clean, tạo bằng `python -m edgeai.pilot_data
  --archive PATH_TO_DEV_CLEAN --output artifacts/datasets/librispeech-dev-clean/prepared`;
  lệnh tạo đầy đủ split nhưng FULL100 chỉ chạy inference/đánh giá 60 hàng validation.
- Snapshot teacher đúng revision và hash trong `src/edgeai/teacher.py`.
  Không tự thay snapshot bằng revision mới; loader mặc định offline.
- TFLM host runner và sidecar build. `sh scripts/build_tflm_host.sh` dựng từ
  source pin; trên máy mới bước này có thể tải vendor/toolchain, cần chủ
  động cho phép mạng. Không cần build lại để phục hồi job đang chạy.

`scripts/run_full100.py --dry-run` in toàn bộ lệnh. Muốn chạy mới phải chọn
output chưa tồn tại bằng `--output PATH`. Không trỏ một lần train mới vào
thư mục đã hoàn tất. `--resume` không cho phép đổi source/config/dữ liệu.

Các model công bố là encoder-only; classifier 251 lớp và teacher không
được đưa vào thiết bị. Kết quả host không thay thế kiểm chứng firmware,
microphone, RAM tổng, tốc độ hoặc năng lượng trên ESP32-S3.

## Kết thúc run và kiểm chứng ngày 18/10/2026

`run_status.json` đã là `COMPLETE`, cả bảy pha hoàn tất. Supervised dừng
sớm sau 19 epoch/chọn epoch 4; KD sau 20 epoch/chọn epoch 5. Cả hai report
là `FULL_TRAINING_COMPLETE`; host reports `PASS`, `TFLM_HOST_PASS`,
`BUILD_BOUND`, mỗi model có năm fixture train thật sai lệch không quá 1 LSB.

Lead đã đọc và hash lại toàn bộ 16 source binding, sáu input binding và
output binding của cả bảy pha, gồm PCM/features đầy đủ; tất cả khớp trạng
thái đã hoàn tất. Audit độc lập riêng dùng cached embedding/NumPy kiểm tra
metric, ID/order, checkpoint, source/artifact và host binding, không chạy
lại inference; không rehash các mảng train lớn trong audit bounded đó.

Kết quả chi tiết và gói encoder:
[full100-results.md](full100-results.md),
[models/full100-20261015](../models/full100-20261015/README.md).
Không có kết quả calibration/test hay xác nhận board. Hai caveat thủ công
ở trên vẫn có hiệu lực, không bị nâng cấp thành bảo đảm tự động đầy đủ.
