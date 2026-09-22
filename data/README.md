# Dữ liệu

Không commit dữ liệu vào repo. Thư mục này chỉ chứa hướng dẫn tải và file cấu hình.

| Bộ | Dùng để | Ghi chú |
|----|---------|---------|
| VoxCeleb1 | Huấn luyện và bộ thử VoxCeleb1-O | Cần đăng ký, bản dev ~30GB, có mirror trên HuggingFace nếu link gốc hỏng |
| MUSAN | Nhiễu nền | Tải thẳng |
| RIR | Vang phòng | Tải thẳng |
| ASVspoof 2017/2019 PA | Tấn công phát lại | Cần đăng ký |

Bộ thử dùng **danh sách cặp VoxCeleb1-O gốc**, không tự chế danh sách khác, để số của nhóm còn so được với các bài đã công bố.

Kiểm tra dung lượng ổ trước khi tải. TV1 phụ trách phần này, cần xong trong tuần 2 vì chặn đường cả nhóm.

Giọng enroll của các thành viên để trong `data/enroll/`, không commit, xoá sau khi nộp.
