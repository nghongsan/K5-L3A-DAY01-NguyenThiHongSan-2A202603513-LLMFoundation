# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603513
- **Họ và tên:** Nguyễn Thị Hồng San
- **SEED** (in ở ô đầu notebook): 7264
- **Thử thách cá nhân:** Đề #5 (`challenge = 4` trong `lab1_report.json`)
- **Bộ kiểm tra tự động:** 15/15 ✅ (TODO 1.1 → 4.4 và 5 phép tính tay)

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

Dữ liệu sau khi bỏ trùng: 5.169 tin (653 spam, 4.516 ham), đã loại 403 tin trùng.
Chia dữ liệu: train 3.619, val 775, test 775 (99 spam), SUB 40 tin (20 spam + 20 ham).

## 2. Kết quả chính

| Phương án | Precision | Recall | F1 (test) | F1 (SUB 40) | Train (s) | p50 ms/tin | p95 ms/tin | USD/1k tin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1. BoW + LogReg (tự code) | 0.988 | 0.818 | 0.895 | 0.824 | 8.8 | 0.05 | 0.50 | 0 |
| 2. BoW + MLP | 0.957 | 0.889 | 0.921 | 0.919 | 14.9 | 0.30 | 4.36 | 0 |
| 3a. Encoder MiniLM + kNN | 0.866 | 0.980 | 0.919 | 0.976 | 120.7 | 44.73 | 60.67 | 0 |
| 3b. Encoder MiniLM + LogReg | 0.969 | 0.939 | **0.954** | 0.974 | 121.5 | 24.90 | 36.10 | 0 |
| 3c. Encoder MiniLM + MLP head | 0.969 | 0.939 | **0.954** | 0.974 | 121.6 | 24.75 | 34.34 | 0 |
| 4a. LLM nhỏ zero-shot (`gpt-6-luna`) | — | — | — | 0.000* | 0 | 697 | 1.152 | 0.0065 |
| 4b. LLM lớn zero-shot (`gpt-6.1-terra`) | — | — | — | 0.000* | 0 | 861 | 1.073 | 0.0649 |
| 4c. LLM nhỏ few-shot | — | — | — | 0.000* | 0 | 714 | 1.019 | 0.0206 |

\* Cả 3 hàng LLM có `n_invalid = 40/40`: model trả về chuỗi rỗng, `parse_label` trả `None`, và tin được tính là ham. F1 = 0 vì vậy là **lỗi pipeline**, không phản ánh chất lượng model. Ô "Say hello" (đặt `max_tokens` = 20) vẫn chạy được, nên key và kết nối không có vấn đề. Em xác định nguyên nhân là giới hạn 3 token khi phân loại không đủ để model trả về nhãn. Độ trễ và chi phí vẫn đo được vì các lần gọi API vẫn thành công và vẫn bị tính tiền.

**Nhận xét nhanh**

- Embedding (3b/3c) cho F1 test cao nhất (0.954), hơn BoW + LogReg 0.059.
- Thêm hidden layer giúp trên BoW (0.895 → 0.921), nhưng không giúp trên embedding (3b = 3c ở cả 4 chỉ số).
- Đường Pareto (1 − F1 trên SUB ↔ p95): **1 → 2 → 3c → 3a**. 3b bị 3c lấn át vì cùng lỗi 2.6% nhưng chậm hơn khoảng 2 ms; mức chênh này chỉ là nhiễu đo. Cả 3 LLM bị mọi phương án khác lấn át.
- Đạt SLA p95 < 50 ms: 1, 2, 3b, 3c. Không đạt: 3a (60.7 ms) và mọi LLM (trên 1 giây).
- SUB chỉ có 40 tin, nên 1 tin sai làm F1 đổi khoảng 2.5 điểm. 3a, 3b, 3c đều chỉ sai 1 tin, vì vậy không xếp hạng được giữa ba phương án này.

## 3. Dự đoán và thực tế

Cả 10 dự đoán đều được ghi **trước** khi chạy thí nghiệm (có dấu thời gian trong `lab1_report.json`).

| Key | Thời điểm | Dự đoán | Thực tế | Khớp? / Vì sao lệch |
| --- | --- | --- | --- | --- |
| `ml_f1` | 18:59:30 | F1 khoảng 0.85, precision > recall | F1 = 0.895; P = 0.988, R = 0.818 | Khớp xu hướng. F1 cao hơn đoán vì spam trong dữ liệu này phụ thuộc mạnh vào từ khoá (`call`, `free`, `txt`, `reply`…), nên BoW bắt tốt hơn em nghĩ |
| `ml_lr` | 18:59:40 | lr = 0.001: loss gần như nằm ngang quanh 0.69; lr = 50: loss nhảy lên xuống | lr = 0.001: 0.69 → 0.63. lr = 50: vọt lên khoảng 2.6 rồi tụt về 0.006 | Khớp một nửa. Bài toán lồi và dữ liệu train dễ tách, nên dù bước quá lớn trọng số vẫn bị đẩy rất xa và mô hình quá tự tin. Loss thấp ở đây chỉ là loss **trên train** |
| `dl_curve` | 19:00:09 | Val loss giảm rồi tăng (overfit); F1 MLP ≈ LogReg | Val loss thấp nhất ở epoch 9 rồi tăng; F1 = 0.921 so với 0.895 | Khớp |
| `dl_relu` | 19:00:24 | Bỏ ReLU thì mạng tương đương LogReg, F1 ≈ 0.9, thấp hơn hoặc bằng MLP | F1 = 0.927, nhỉnh hơn MLP (0.921) | Gần khớp. Với input BoW, mô hình tuyến tính đã đủ; chênh 0.006 là nhiễu |
| `emb_search` | 19:03:31 | Kết quả gần nhất là spam | 6/6 tin gần nhất đều là spam (cosine 0.53–0.56) | Khớp. Query không có `free/win/prize` nhưng gần nghĩa với "holiday / cash award" |
| `emb_tricky` | 19:04:49 | Encoder + LogReg đúng nhiều hơn BoW | Encoder + LogReg 5/6, BoW 4/6 | Khớp |
| `llm_models` | 19:05:11 | Model lớn hơn F1 khoảng 0.02–0.05, đắt gấp ~10 lần, không thắng rõ 3b | F1 = 0 cả hai (invalid 40/40); chi phí $0.00260 so với $0.00026 (gấp 10 lần) | Phần chi phí khớp. Phần F1 sai hoàn toàn vì lỗi cấu hình token |
| `llm_fewshot` | 19:06:18 | F1 tăng nhẹ; chi phí mỗi tin tăng 2.5–3.5 lần | F1 = 0 (invalid 40/40); chi phí $0.0206 so với $0.0065 /1k tin, gấp **3.2 lần** | Phần chi phí khớp. Token input tăng theo 4 ví dụ, còn output vẫn chỉ 1 từ |
| `tradeoff` | 19:06:50 | Pareto: 1, 2, 3c, 3a; 3b bị lấn át sát nút; 3a và LLM vượt SLA | Đúng như dự đoán | Khớp |
| `hard_eval` | 19:06:51 | BoW 50–60%, Embedding 60–70%, LLM 42% nếu chưa sửa lỗi invalid | BoW 54% / 52%; Embedding 68% / 66% / 66%; LLM 42% | Khớp |

## 4. Phân tích 50 case khó

**F1 trên tập test thường (SUB) so với F1 trên 50 case khó**

| Model | F1 SUB | F1 50 case khó | Độ tụt | Accuracy 50 case khó |
| --- | ---: | ---: | ---: | ---: |
| 1. BoW + LogReg | 0.824 | 0.378 | −0.45 | 54% |
| 2. BoW + MLP | 0.919 | 0.368 | **−0.55** | 52% |
| 3a. Emb + kNN | 0.976 | 0.724 | −0.25 | 68% |
| 3b. Emb + LogReg | 0.974 | 0.667 | −0.31 | 66% |
| 3c. Emb + MLP head | 0.974 | 0.653 | −0.32 | 66% |
| 4a / 4d. LLM (lỗi invalid) | — | 0.000 | — | 42% (= 21 tin ham / 50) |

**% đoán đúng theo nhóm**

| Nhóm | BoW+LR | BoW+MLP | Emb+kNN | Emb+LR | Emb+MLP |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tưởng spam nhưng không phải | 100% | 100% | 70% | 70% | 80% |
| Thao túng / lừa đảo | 12% | 38% | 100% | 38% | 38% |
| Spam giọng tự nhiên | 20% | **0%** | 40% | 40% | 40% |
| Né từ khoá | 17% | **0%** | 83% | 83% | 67% |
| Prompt injection | 67% | 83% | 83% | 100% | 100% |
| Nói về spam | 100% | 100% | 25% | 75% | 75% |
| Tin giao dịch hợp lệ | 83% | 67% | 67% | 83% | 83% |

**Model tụt nhiều nhất: BoW + MLP**, F1 giảm từ 0.919 xuống 0.368. Tập test thường "quá lạc quan" vì nó cùng phân phối với tập train: spam trong test vẫn dùng lại những từ khoá và đầu số model đã học (`free`, `txt`, `09061…`). Model đạt điểm cao nhờ nhớ mẫu, chứ không phải vì hiểu thế nào là spam.

**Case 1 — #30 "F R E E  e n t r y  into our weekly draw, t x t WIN to 80086" (Né từ khoá, spam).**
BoW + LogReg và BoW + MLP đều đoán ham. `tokenize` tách chuỗi thành các chữ cái đơn `f, r, e, e, t, x, t…`, nên những token này không trùng với `free` hay `txt` trong vocab. Các từ khoá spam mạnh nhất biến mất khỏi vector, chỉ còn các từ trung tính. Nhóm Né từ khoá vì vậy chỉ đạt 17% / 0% với BoW. Embedding làm tốt hơn (83%) vì tokenizer sub-word của MiniLM vẫn giữ được một phần nghĩa của chuỗi bị biến dạng.

**Case 2 — #14 "Hi, I'm working from home and making about £300 a day, happy to show you how…" (Spam giọng tự nhiên).**
Đây là một trong 5 case mà **tất cả** model đều sai (#12, #13, #14, #16, #19 đều thuộc nhóm Spam giọng tự nhiên). Tin không có từ khoá spam, nên BoW không có tín hiệu nào. Về nghĩa, câu cũng thật sự giống tin của một người quen kể chuyện. Embedding đặt nó gần vùng ham vì dữ liệu train gần như chỉ có spam kiểu trúng thưởng hoặc nhắn mã số, không có kiểu lừa việc làm.

**Điểm ngược chiều.** Embedding + kNN lại yếu ở nhóm *Nói về spam* (25%): tin cảnh báo một vụ lừa đảo có nghĩa rất gần với chính tin lừa đảo đó, nên các tin hàng xóm gần nhất bỏ phiếu spam. BoW làm đúng 100% nhóm này vì tin cảnh báo có nhiều từ kéo về ham (`i`, `my`, `me`).

**LLM và prompt injection.** Lần chạy này chưa kết luận được: 4a và 4d (prompt phòng thủ) cho kết quả giống hệt nhau vì cả hai đều trả về chuỗi rỗng. Con số 17% ở nhóm Prompt injection chỉ là case #40, tin ham duy nhất trong nhóm. Đề xuất phòng thủ ngoài prompt: chuẩn hoá đầu vào, dùng rule gắn cờ các mẫu như `ignore previous instructions`, `label =`, `[SYSTEM`, và kết hợp LLM với 3b để LLM không phải lớp kiểm tra duy nhất.

**Bonus — case tự nghĩ (`MY_CASES`).** Em đã viết 6 case gốc:

| # | Tin nhắn (rút gọn) | Nhãn | Nhóm | Vì sao khó |
| --- | --- | --- | --- | --- |
| 1 | My nan just got a text saying she won a free holiday and must call 09061 now, I blocked the number… | ham | Nói về spam | Chứa đủ `won`, `free`, `call`, đầu số 09061, nhưng là kể lại |
| 2 | Warning from IT: emails saying 'claim your bonus, click here' are phishing… | ham | Nói về spam | Trích nguyên văn câu lừa đảo trong lời cảnh báo |
| 3 | WINNER!! Our team won the cup final, free drinks at the pub tonight, call me… | ham | Tưởng spam | `WINNER`, `free`, `call` trong tin bạn bè |
| 4 | Hey, it's Tom from your old school… selling tickets for the hotel deal, can I send you the booking link? | spam | Spam giọng tự nhiên | Giả làm bạn cũ, không có từ khoá |
| 5 | Hi love… this supplement helped me lose 5kg, happy to send you a discount code | spam | Spam giọng tự nhiên | Giọng thân mật, bán hàng kín đáo |
| 6 | Hi dad, this is my work phone, my card got blocked. Could you transfer £150… | spam | Thao túng | Mạo danh người thân, tạo áp lực tiền bạc |

Lưu ý: ô `MY_CASES` được chạy **sau** ô "Chạy mọi model trên 50 case khó", nên bảng kết quả trong notebook vẫn chỉ có 50 case gốc. Dựa trên kết quả theo nhóm ở trên, em dự đoán case 4–5 sẽ lừa được mọi model (giống #14, #16), case 6 lừa được BoW (nhóm Thao túng chỉ đạt 12–38%), và case 1–2 lừa được Emb + kNN (nhóm Nói về spam chỉ 25%).

## 5. Thử thách cá nhân

**Đề #5 (Embedding):** viết 4 tin nhắn của riêng mình (2 spam né từ khoá, 2 ham chứa `free`/`win`), so dự đoán của BoW với Embedding + LogReg và giải thích từng tin.

**Cách làm:** dùng lại `w, b` (Phần 1) và `w_e, b_e, mu, sd` (3b), in xác suất của cả hai model, cùng 4 từ có |trọng số| lớn nhất mà BoW dựa vào.

| Tin | Thật | BoW (p) | Enc + LR (p) | Từ BoW dựa vào nhiều nhất |
| --- | --- | --- | --- | --- |
| "Hello, we have a nice job for you at home. You can get good money every week…" | spam | ham ✗ (0.12) | ham ✗ (0.01) | your +0.81, me −0.62, for +0.59, send +0.43 |
| "Your bank card is stopped today. Please open this link and write your card number…" | spam | ham ✗ (0.33) | ham ✗ (0.13) | to +0.85, your +0.81, it −0.40, please +0.35 |
| "Are you free this evening? Let's go to the park and play football" | ham | ham ✓ (0.25) | ham ✓ (0.14) | free +0.91, to +0.85, this +0.32, go −0.13 |
| "I hope our team can win the game tomorrow, see you at school" | ham | ham ✓ (0.01) | ham ✓ (0.00) | i −1.43, our +0.38, win +0.37, at −0.28 |

**Kết quả:** BoW đúng 2/4, Encoder + LogReg đúng 2/4. Cả hai đều để lọt 2 tin spam và đoán đúng 2 tin ham.

**Giải thích:**

- **Tin 1 (lừa việc làm):** không có từ khoá spam. BoW chỉ dựa vào các từ chức năng như `your`, `for`, nên xác suất thấp. Embedding còn tự tin hơn (0.01) vì spam trong dữ liệu train gần như toàn kiểu trúng thưởng hoặc nhắn mã số, còn lừa việc làm viết bằng câu đơn giản thì rất giống tin bình thường.
- **Tin 2 (giả ngân hàng):** BoW cho 0.33, gần ngưỡng nhưng vẫn đoán ham; Embedding cho 0.13. Kiểu lừa đảo dẫn link ngân hàng gần như không có trong SMS Spam Collection, nên cả hai model đều chưa từng học mẫu này.
- **Tin 3 (ham có `free`):** `free` có trọng số spam lớn (+0.91), nhưng phần còn lại của câu kéo xác suất xuống 0.25, nên BoW vẫn đúng.
- **Tin 4 (ham có `win`):** `win` chỉ +0.37, trong khi `i` có trọng số −1.43 kéo mạnh về ham. Tin có ngôi "tôi" thường là tin người thật.

**Kết luận:** BoW quyết định phần lớn dựa vào các từ không mang nghĩa (`your`, `to`, `i`). Embedding hiểu nghĩa câu, nhưng chỉ nhận ra những **kiểu** spam đã gặp khi train. Kết quả này khớp với nhóm "Spam giọng tự nhiên" trong stress test: muốn bắt được các kiểu lừa đảo mới thì phải bổ sung dữ liệu, chứ không chỉ đổi cách biểu diễn.

## 6. Phương án cho sản phẩm thật

Yêu cầu: 10 triệu SMS/ngày (trung bình khoảng 116 tin/giây), p95 < 50 ms/tin.

**Đạt SLA độ trễ:** 1 (0.50 ms), 2 (4.36 ms), 3b (36.1 ms), 3c (34.3 ms).
**Không đạt:** 3a kNN (60.7 ms) và LLM (p95 từ 1.019 đến 1.152 ms).

**Em chọn 3c (Encoder MiniLM + MLP head), dùng 3b làm phương án tương đương:**

- F1 test 0.954, cao nhất trong nhóm đạt SLA; trên 50 case khó đạt 66%, hơn BoW khoảng 12 điểm.
- p95 34.3 ms < 50 ms, và không mất phí API.
- Đánh đổi chấp nhận: chậm hơn BoW khoảng 70 lần ở p95. Trên CPU Colab chỉ xử lý khoảng 38 tin/s cả lô, nên cần ít nhất 116 / 38 ≈ **4 máy CPU** cho tải trung bình (nhiều hơn khi cao điểm), hoặc dùng GPU và batching.
- Phương án dự phòng khi hạ tầng hạn chế: 2 (BoW + MLP), F1 0.921, p95 4.4 ms. Tuy vậy, phương án này yếu rõ với spam né từ khoá (0%).

**LLM không phù hợp làm lớp chính:**

- Chi phí (theo số đo `usd_per_1k` của em): LLM nhỏ $0.0065/1k tin × 10 triệu ≈ **$65/ngày ≈ $1.950/tháng**. LLM lớn ≈ $649/ngày ≈ **$19.500/tháng**. Theo bài tính tay (100 token input, 1 token output), 1 triệu tin tốn $10.50/ngày, vậy 10 triệu tin ≈ $105/ngày ≈ $3.150/tháng.
- Độ trễ trên 1 giây, vượt SLA khoảng 20 lần.
- Còn phụ thuộc vào cấu hình và nhà cung cấp: lần chạy của em cho thấy chỉ cần sai giới hạn token là 100% câu trả lời hỏng mà vẫn bị tính tiền.

**Thiết kế lai (router):**

1. Lớp rule: chặn link hoặc đầu số đã biết là xấu, và gắn cờ các mẫu prompt injection.
2. 3c chấm điểm mọi tin: p < 0.3 cho qua, p > 0.7 chặn.
3. Chỉ những tin lưỡng lự (p ∈ [0.3, 0.7]) mới gửi sang LLM xử lý **bất đồng bộ**.

Nếu tin lưỡng lự dưới 5%, p95 của luồng chính vẫn dưới 50 ms và chi phí LLM chỉ khoảng 5% × $1.950 ≈ **$100/tháng**. Tin dễ được xử lý nhanh và rẻ, tin khó mới dùng đến LLM. Có nhiều lớp kiểm tra cũng giúp hệ thống khó bị prompt injection qua mặt hơn so với dùng một LLM duy nhất.

## 7. AI log và điều đã học

### AI log

| Phần | Em hỏi AI gì | AI trả lời / viết gì | Em đã sửa / kiểm chứng gì |
| --- | --- | --- | --- |
| Phần 0 — Khám phá dữ liệu | Giải thích data leakage và vì sao phải bỏ trùng trước khi chia train/test | Giải thích việc dữ liệu test lộ vào train làm điểm đánh giá cao hơn thực tế | Đối chiếu số liệu thật: 403 tin trùng, sau khi bỏ còn 5.169 tin; tự tính baseline 4516/5169 ≈ 0.87 trên giấy |
| Phần 1 — Machine Learning | Giải thích gradient descent, sigmoid, cách cập nhật trọng số khi mô hình đoán sai | Giải thích cách tính error và gradient, ý nghĩa của learning rate | Tự viết code bằng NumPy, chạy các ô kiểm tra, đối chiếu loss và F1 với dự đoán; tự chọn tin "cal;l" từ output của mình cho Giải thích 1 |
| Phần 2 — Deep Learning | Giải thích vai trò của ReLU, backprop và overfitting trong MLP | Giải thích quy trình train, vai trò của `zero_grad()`, `backward()`, `step()` | Tự viết bằng PyTorch; đọc biểu đồ train/val loss và xác định epoch 9 có val loss thấp nhất |
| Phần 3 — Embedding | Kiểm tra đúng/sai các TODO; giải thích cosine, kNN, BoW so với embedding | Góp ý code `cosine_similarity`, `top_k_similar`, `knn_predict`; giải thích biểu diễn nghĩa bằng vector | Chạy kiểm tra; xác nhận 3b = 3c ở cả 4 chỉ số là kết quả thật, không phải lỗi |
| Phần 4 — LLM (vùng được dùng AI) | Sửa lỗi API key; viết TODO 4.1–4.4 (gọi API, prompt zero/few-shot, `parse_label`) | Chẩn đoán lỗi 401; viết `call_openai`, prompt, `parse_label`, `llm_classify_batch` | Rà soát code, chạy kiểm tra với API giả lập; không đưa key vào code; tự phát hiện 40/40 phản hồi invalid do giới hạn token |
| Phần 5 — Tổng hợp | So sánh mô hình theo F1, độ trễ, chi phí, SLA; cách đọc đường Pareto | Giải thích các đánh đổi và soạn bản nháp câu trả lời | Đối chiếu với bảng và output "Trên đường Pareto" của notebook; tự làm thử thách #5 (không dùng AI) |

### Điều đã học

1. **Điều em dự đoán sai nhiều nhất là LLM.** Em đoán F1 khoảng 0.9, nhưng thực tế là 0 vì cả 120 lần gọi đều trả về chuỗi rỗng mà vẫn bị tính tiền. Em học được rằng cấu hình (`max_tokens`, `reasoning_effort`) và việc kiểm tra `n_invalid` quan trọng không kém việc chọn model. Không được tin một con số F1 khi chưa xem output thô.
2. **Điểm test cao chưa chắc là model hiểu bài toán.** Embedding đạt 0.97 F1 trên SUB nhưng chỉ còn 0.65–0.72 trên 50 case khó. Tập test cùng phân phối với train chỉ đo được khả năng nhớ mẫu.
3. **Biểu diễn quan trọng hơn độ sâu của mạng.** Thêm hidden layer giúp BoW tăng 0.026 F1, còn đổi sang embedding giúp tăng 0.059 và bền hơn nhiều trước tin né từ khoá.
4. **Phải chọn phương án bằng số đo.** Với SLA 50 ms và 10 triệu tin/ngày, phương án hợp lý là encoder nhỏ chạy tại chỗ, kết hợp LLM chỉ cho phần nhỏ tin khó. Không phải "model lớn nhất".
