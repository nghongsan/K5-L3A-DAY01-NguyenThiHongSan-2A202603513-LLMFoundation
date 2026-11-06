# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:** 2A202603513
- **Họ và tên:** Nguyễn Thị Hồng San
- **SEED** (in ở ô đầu notebook): 7264

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☐ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☐ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☐ |

## 2. Kết quả chính
Phương án	F1 (test)	F1 (SUB 40)	p95 ms/tin	USD/1k tin
1. BoW + LogReg	0.895	0.824	0.13	0
2. BoW + MLP	0.926	0.919	0.44	0
3a. Encoder + kNN	0.919	0.976	78.2	0
3b. Encoder + LogReg	0.954	0.974	37.5	0
3c. Encoder + MLP head	0.954	0.974	42.7	0
4. LLM	—	—	—	—

LLM không có số liệu vì tài khoản OpenAI miễn phí chỉ cho 50 request/ngày (lỗi 429).

## 3. Dự đoán và thực tế
Dự đoán	Đoán	Thực tế	Vì sao lệch
ml_lr	lr = 50 thì loss dao động liên tục	Loss vọt lên khoảng 2.6 rồi tụt về 0.006	Bài toán lồi, dữ liệu dễ tách nên vẫn hội tụ; loss thấp chỉ là loss trên tập train
dl_relu	Bỏ ReLU thì F1 thấp hơn MLP	F1 = 0.927, ngang MLP (0.926)	Với BoW, mô hình tuyến tính đã đủ tốt; lớp phi tuyến không giúp thêm
ml_f1	F1 khoảng 0.85	F1 = 0.895	Spam trong dữ liệu này phụ thuộc mạnh vào từ khoá nên BoW bắt tốt hơn em nghĩ

## 4. Phân tích 50 case khó

*Model nào tụt nhiều nhất, ở nhóm case nào? Nêu 2 case cụ thể và cơ chế khiến model sai.*

## 5. Thử thách cá nhân

*Đề theo SEED của bạn, cách làm, kết quả, giải thích.*

## 6. Phương án cho sản phẩm thật
Yêu cầu: 10 triệu SMS/ngày (trung bình khoảng 116 tin/giây), p95 < 50 ms/tin.

Đạt SLA độ trễ: 1, 2, 3b, 3c. Không đạt: 3a kNN (p95 78 ms) và LLM (mỗi lần gọi khoảng 1 giây trở lên).
Em chọn 3b (Encoder MiniLM + LogReg):
F1 test 0.954, cao nhất trong nhóm đạt SLA.
p95 37.5 ms < 50 ms.
Không mất phí API.
Đánh đổi chấp nhận:
Chậm hơn BoW khoảng 300 lần.
Trên CPU Colab chỉ xử lý khoảng 33 tin/s, nên cần khoảng 4–8 máy CPU (tính cả giờ cao điểm), hoặc GPU và batching.
Nếu hạ tầng rất hạn chế, phương án 2 (BoW + MLP: F1 0.926, p95 0.44 ms) là lựa chọn dự phòng rẻ.
LLM không phù hợp làm lớp chính:
Theo tính tay, 1 triệu tin tốn $10.5/ngày. Với 10 triệu tin là khoảng $105/ngày, tức khoảng $3.150/tháng.
Độ trễ hàng giây, vượt SLA rất xa.
Còn phụ thuộc vào hạn mức của nhà cung cấp. Em đã gặp trực tiếp khi tài khoản bị chặn ở 50 request/ngày.
Thiết kế lai đề xuất:
3b xử lý mọi tin.
Chỉ những tin "lưỡng lự" (p ∈ [0.3, 0.7], ước tính vài %) được chuyển sang LLM xử lý bất đồng bộ.
Nhờ vậy giữ được độ trễ thấp, còn chi phí LLM giảm hàng chục lần.

## 7. AI log và điều đã học

Phần	Em hỏi AI gì	AI trả lời / viết gì	Em đã sửa / kiểm chứng gì
Đọc đề	Nghiên cứu repo và notebook lab	Tóm tắt cấu trúc bài, luật lab và các lưu ý khi chạy Colab	Đối chiếu với README, RULES
Phần 0	Câu 0.1 (data leakage), Tính tay #1	Giải thích tin trùng làm điểm test lạc quan; accuracy luôn đoán ham = 4516/5169 ≈ 0.87	Tự kiểm tra 94 + 309 = 403 tin trùng; tính lại trên giấy
Phần 1	TODO 1.1–1.4 (được coach cho phép dùng AI)	Code vectorize, sigmoid, predict_proba, train_logreg, metrics kèm giải thích từng dòng	Chạy ô kiểm tra; tự trả lời câu hỏi vấn đáp (đổi = thành +=, vì sao cần if word in vocab)
Phần 1	Gợi ý dự đoán ml_f1, ml_lr; Giải thích 1, Câu 1.B, Bắt lỗi 1	Lập luận về dữ liệu lệch lớp, learning rate; bản nháp câu trả lời	Yêu cầu viết ngắn gọn, chọn tin "cal;l" từ chính output của em
Phần 2	TODO 2.1–2.2; dự đoán; Câu 2.A, Giải thích 2	Code MLP và vòng lặp train 5 dòng; giải thích overfit, zero_grad	Chạy kiểm tra; xoá một dòng idx = ... bị lặp; đối chiếu biểu đồ val loss (thấp nhất ở epoch 9)
Phần 3	TODO 3.1–3.3; dự đoán; Câu hỏi 3, Bắt lỗi 2, Checkpoint B	Code cosine, top-k (nhân ma trận), kNN; giải thích theo bảng kết quả	Chạy kiểm tra; xác nhận 3b = 3c ở cả 4 chỉ số là kết quả thật, không phải lỗi
Phần 4	TODO 4.1–4.4, dự đoán, Bắt lỗi 3; xử lý lỗi 429	Code call_openai, prompt, parse_label, llm_classify_batch; chẩn đoán lỗi hết credit, rồi lỗi hạn mức 50 request/ngày	Kiểm tra bằng API giả lập; ô Say hello chạy thật được; không đưa key vào code
Phần 5	Dự đoán tradeoff, hard_eval; Câu hỏi 5; báo cáo	Phân tích Pareto từ bảng của em; tính 1 tin sai trên SUB ≈ 2.5 điểm F1; khung báo cáo	Đối chiếu với output "Trên đường Pareto" của notebook; tự viết phần thử thách và stress test
