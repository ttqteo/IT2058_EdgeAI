# Tài liệu tham khảo

Danh sách khởi đầu, mỗi người bổ sung thêm cho phần mình phụ trách. Đánh số để các tài liệu khác trong `docs/` trích dẫn được.

## Xác thực người nói

[1] Snyder et al., *X-Vectors: Robust DNN Embeddings for Speaker Recognition*, ICASSP 2018. Kiến trúc embedding nền tảng, làm mốc lịch sử.

[2] Desplanques et al., *ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification*, Interspeech 2020. Teacher của đồ án.

[3] Deng et al., *ArcFace: Additive Angular Margin Loss for Deep Face Recognition*, CVPR 2019. Hàm mất mát margin góc, gốc từ nhận dạng khuôn mặt nhưng nay là chuẩn cho cả speaker verification.

[4] Ravanelli et al., *SpeechBrain: A General-Purpose Speech Toolkit*, 2021. Nguồn lấy teacher pretrained.

## Dữ liệu

[5] Nagrani et al., *VoxCeleb: A Large-Scale Speaker Identification Dataset*, Interspeech 2017. Dataset chính và bộ thử VoxCeleb1-O.

[6] Chung et al., *VoxCeleb2: Deep Speaker Recognition*, Interspeech 2018. Bản lớn hơn, dùng nếu cần mở rộng.

[7] Snyder et al., *MUSAN: A Music, Speech, and Noise Corpus*, 2015. Nguồn nhiễu nền.

[8] Ko et al., *A Study on Data Augmentation of Reverberant Speech for Robust Speech Recognition*, ICASSP 2017. Nguồn đáp ứng xung phòng cho phần vang.

## Tấn công phát lại

[9] Kinnunen et al., *The ASVspoof 2017 Challenge: Assessing the Limits of Replay Spoofing Attack Detection*, Interspeech 2017.

[10] Todisco et al., *ASVspoof 2019: Future Horizons in Spoofed and Fake Audio Detection*, Interspeech 2019. Phần PA là tấn công phát lại, dùng cho điều kiện replay của đồ án.

## Nén mô hình

[11] Hinton et al., *Distilling the Knowledge in a Neural Network*, 2015. Gốc của chưng cất tri thức, phần TV1 phụ trách.

[12] Li et al., *Pruning Filters for Efficient ConvNets*, ICLR 2017. Tỉa kênh có cấu trúc, phần TV2 phụ trách.

[13] Jacob et al., *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference*, CVPR 2018. Nền tảng của lượng tử hoá INT8.

[14] Howard et al., *MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications*, 2017. Tích chập tách chiều sâu, dùng khi thiết kế kiến trúc student.

## Suy luận trên thiết bị biên

[15] David et al., *TensorFlow Lite Micro: Embedded Machine Learning for TinyML Systems*, MLSys 2021. Runtime dùng trên ESP32-S3.

[16] Banbury et al., *MLPerf Tiny Benchmark*, 2021. Cách chuẩn hoá việc đo độ trễ và năng lượng trên vi điều khiển, tham khảo cho phần đo của nhóm.

## Hai bài mẫu của môn

[17] Nguyen-Tat et al., *Automating attendance management in human resources: A design science approach using computer vision and facial recognition*, IJIM Data Insights, 2024. Triển khai trên Jetson Nano, dùng để định vị đồ án ở lớp phần cứng thấp hơn.

[18] Nguyen-Tat và Ngo, *Lite-XNet: An efficient deep learning model for COVID-19 lung severity assessment on embedded systems*, CVIU, 2026. Khuôn mẫu về cách thiết kế mô hình nhẹ rồi so với các kiến trúc nặng và triển khai lên thiết bị.
