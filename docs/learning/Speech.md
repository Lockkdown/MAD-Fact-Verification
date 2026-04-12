# Kịch Bản Thuyết Trình: 2.4 CƠ CHẾ DEBATE

> **Hướng dẫn:** Phần này giải thích quy trình cốt lõi của hệ thống Multi-Agent Debate (MAD). Cần nói mạch lạc, nhấn mạnh vào luồng xử lý từ lúc nhận đầu vào cho đến khi ra kết quả. Slide được chia làm 4 khối, hãy chỉ tay / dùng laser pointer vào từng khối khi nói đến.

---

## 1. Mở đầu (Chuyển ý vào slide)
"Dạ vâng, tiếp theo xin mời thầy cô và các bạn cùng nhìn lên sơ đồ **Cơ chế Debate** (phần 2.4)

## 2. Vòng 1: Phân tích độc lập (Khối đầu tiên bên trái)
**Vòng 1**. Các agent sẽ được cung cấp claim và evidence, vòng này thì bọn em thiết kế là 1 vòng phân tích độc lập, có nghĩa là các agent này không biết ý kiến của nhau mà phải tự phân tích độc lập.  
Quá trình này gồm 2 bước:
- **Bước 1:** agent sẽ **tách claim đó ra thành 2 đến 5 ý nhỏ**. Sau đó, nó đối chiếu từng ý nhỏ với bằng chứng (evidence) và gắn nhãn là: *Hợp lí*, *Thiếu thông tin*, hoặc *Mâu thuẫn*.
- **Bước 2:** Dựa vào các ý nhỏ này, AI mới đưa ra quyết định tổng thể:
  - Nếu **tất cả các ý đều hợp lí** với bằng chứng, claim được đánh giá là **HỖ TRỢ**.
  - Chỉ cần **có 1 ý mâu thuẫn**, toàn bộ claim bị đánh giá là **BÁC BỎ**.
  - Và trong trường hợp nếu không có mâu thuẫn nhưng lại **thiếu thông tin** để xác nhận, kết quả sẽ là **CHƯA ĐỦ THÔNG TIN (NEI)**."

## 3. Vòng 2 đến k: Tranh luận chéo (Khối thứ 2)
"Sau khi có nhận định ban đầu.
Mỗi agent sẽ được xem lập luận của nhau. Lúc này, agent phải đối chiếu trích dẫn của mình với trích dẫn  của người khác, chỉ ra điểm đồng thuận hoặc bất đồng dựa trên evidence, và quyết định giữ hay đổi. 

## 4. Đánh giá của Judge và Early Stopping (Khối thứ 3)
Hệ thống (cụ thể là module Orchestrator và Judge) sẽ kiểm tra xem mức độ đồng thuận của hội đồng đến đâu:
- **Nếu đã đồng thuận (Early Stopping):** Hệ thống sẽ *dừng lại ngay lập tức* để tiết kiệm tài nguyên. Lúc này, Judge (thẩm phán) sẽ làm bước cuối là xác minh lại xem sự đồng thuận đó và đưa ra kết luận cuối cùng.
- **Nếu vẫn bất đồng:** Judge sẽ đánh giá chất lượng của các lập luận, và yêu cầu các Debaters **tiếp tục** quay lại tranh luận (vòng lặp mũi tên gạch đứt) cho đến khi đạt mức đồng thuận hoặc chạm đến số vòng tối đa (k)."

## 5. Kết quả cuối cùng (Khối ngoài cùng bên phải)
"Cuối cùng, sau khi quá trình tranh luận kết thúc (do đạt đồng thuận hoặc hết số vòng quy định), Judge sẽ tổng hợp lại toàn bộ diễn biến và đưa ra **Kết quả cuối cùng**, thuộc 1 trong 3 nhãn: **HỖ TRỢ**, **BÁC BỎ**, hoặc **CHƯA ĐỦ THÔNG TIN**.

Cơ chế này đảm bảo mọi quyết định đưa ra đều được 'suy nghĩ' kỹ lưỡng, đa chiều và có bằng chứng rõ ràng, khắc phục được nhược điểm ảo giác (hallucination) của các mô hình ngôn ngữ đơn lẻ. Xin cảm ơn thầy cô và các bạn."

---

# Kịch Bản Thuyết Trình: PHẦN 3 - KẾT QUẢ THỰC NGHIỆM

## 1. Slide 3: KẾT QUẢ THỰC NGHIỆM (Slide Mục lục)
"Khép lại phần kiến trúc hệ thống, tiếp theo xin mời thầy cô và các bạn bước sang **Phần 3: Kết quả thực nghiệm**. 
Trong phần này, nhóm sẽ lần lượt đi qua 3 nội dung chính: 
Thứ nhất là đánh giá các mô hình ngôn ngữ nhỏ (PLM) để chọn ra mô hình làm cổng định tuyến. Thứ hai là bài toán tìm ngưỡng tự tin (threshold) tối ưu nhất cho cơ chế Hybrid. Và cuối cùng là bức tranh kết quả tổng hợp của toàn bộ dự án."

## 2. Slide 3.1: KẾT QUẢ CỦA CÁC PLM
"Đi vào chi tiết đầu tiên ở Slide 3.1, đây là kết quả huấn luyện độc lập của 4 mô hình ngôn ngữ phổ biến hiện nay: ViBERT, mBERT, XLM-R và PhoBERT. 

Như thầy cô có thể thấy trên bảng, **PhoBERT** đã thể hiện sự vượt trội hoàn toàn với điểm Macro F1 đạt **85.08%**, bỏ xa các mô hình còn lại. Đặc biệt, PhoBERT giữ được sự đồng đều ở cả 3 nhãn, kể cả nhãn khó dự đoán nhất là nhãn Refute (Bác bỏ) cũng đạt trên 80%. 
Nhờ năng lực xuất sắc này, nhóm đã quyết định chọn **PhoBERT làm mô hình 'người gác cổng' (Routing Gate)** cho cơ chế Hybrid Debate tiếp theo."

## 3. Slide 3.2: TÌM NGƯỠNG CHẠY THRESHOLD (Biểu đồ)
"Sang slide 3.2, nhóm đối mặt với một bài toán cốt lõi của kiến trúc Hybrid: *Làm sao để biết mẫu nào nên cho qua (skip), mẫu nào phải đưa vào hội đồng debate?* Nhóm đã quyết định quét thử nghiệm (sweep) các mức ngưỡng tự tin (threshold) từ 0.50 đến 0.95.

Trên biểu đồ này:
- **Đường màu cam nét đứt** là tỉ lệ DSR (Debate Skip Rate) — tức là phần trăm số câu hỏi được hệ thống giải quyết nhanh mà không cần debate.
- **Đường màu xanh liền** là chất lượng của hệ thống — đo bằng Macro F1.

Mọi người có thể thấy một sự đánh đổi (trade-off) cực kỳ rõ ràng: Khi chúng ta tăng ngưỡng lên (đi về bên phải), hệ thống trở nên khắt khe hơn, ít mẫu được bỏ qua hơn (đường cam cắm đầu đi xuống), nhưng bù lại điểm F1 tăng dần và **chạm đỉnh tại vị trí t = 0.85** (đường chấm thẳng đứng). Nếu tiếp tục tăng ngưỡng quá cao, đưa quá nhiều mẫu dễ vào debate một cách không cần thiết, nhiễu thông tin xuất hiện và điểm F1 lại có xu hướng đi ngang hoặc giảm."

## 4. Slide 3.2: TÌM NGƯỠNG THRESHOLD CHẠY HYBRID (Bảng số liệu)
"Để làm rõ hơn cho biểu đồ vừa rồi, bảng số liệu này bóc tách chi tiết kết quả quét ngưỡng trên nhiều cấu hình debate khác nhau (từ n2k3 đến n4k5). 

Tại hàng được bôi đậm ứng với ngưỡng **τ = 0.85**, thầy cô có thể thấy các điểm số đồng loạt đạt mức cao nhất hoặc tiệm cận cao nhất ở hầu hết mọi cấu hình (ví dụ n3k3 đạt 87.63%, n4k3 đạt 87.77%). 
Điều tuyệt vời nhất là tại ngưỡng t=0.85 này, tỉ lệ bỏ qua debate (DSR) vẫn giữ được ở mức rất cao: **87.00%**. 

Điều này mang ý nghĩa thực tiễn cực kỳ lớn: Hệ thống **chỉ cần đưa 13% số lượng mẫu "khó" nhất vào hội đồng LLM để debate**, 87% còn lại được xử lý nhanh bởi PhoBERT. Đây chính là điểm cân bằng hoàn hảo nhất giữa Chất lượng (F1 cao nhất) và Chi phí/Tốc độ (DSR rất lớn). Do đó, nhóm đã chốt **t* = 0.85** làm ngưỡng tiêu chuẩn cho toàn bộ mô hình Hybrid thực nghiệm cuối cùng."

## 5. Slide 3.3: KẾT QUẢ TỔNG HỢP
"Và đây là bảng kết quả quan trọng nhất của toàn bộ nghiên cứu — nơi chúng ta sẽ nhìn thấy rõ lý do tại sao kiến trúc Hybrid lại là giải pháp tối ưu nhất. 

Trước khi đi vào các con số, xin phép được giới thiệu nhanh hai chỉ số cực kỳ quan trọng ở 2 cột cuối cùng:
1. **AC (Average Agent Calls):** Là số lần gọi API trung bình trên mỗi câu hỏi. AC càng thấp nghĩa là hệ thống càng tiết kiệm chi phí và chạy càng nhanh.
2. **DSR (Debate Skip Rate):** Là tỉ lệ phần trăm số câu hỏi được xử lý ngay lập tức bởi 'người gác cổng' PhoBERT mà không cần gọi đến hội đồng Debate. DSR càng cao, AC càng giảm.

Bảng được chia làm 3 phần: Baselines (các mô hình đơn lẻ), Full-Debate (bắt buộc tranh luận 100%), và Hybrid (tranh luận có định tuyến).

*Đầu tiên, nhìn vào phần Baselines:*
Các mô hình đơn lẻ (như GPT-4o mini, Llama-3.3, hay thậm chí DeepSeek-V3) chỉ dùng 1 lần gọi (AC = 1) và đạt F1 loanh quanh 84-87%. Mặc dù rất tốt, nhưng chúng dễ mắc lỗi ở các câu khẳng định phức tạp (ảo giác).

*Thứ hai, nhìn vào phần Full-Debate:*
Khi ghép các mô hình thành hội đồng tranh luận (ví dụ N=4, k=3), F1 có thể duy trì tốt ở mức 86.74%, nhưng **chi phí thì đội lên khủng khiếp**: AC lên tới 6.58. Tức là trung bình mỗi câu kiểm chứng tốn gần 7 lần gọi API, cực kỳ tốn kém và chậm chạp nếu triển khai thực tế.

*Cuối cùng, phần Hybrid (với ngưỡng t=0.85):*
Hãy nhìn vào dòng cuối cùng **N=4, k=5**, đây là 'ngôi sao' của toàn bộ hệ thống. 
Cấu hình Hybrid đạt mức **Macro F1 đỉnh nhất toàn bảng: 87.71%**, đánh bại tất cả các phương pháp Full-Debate và hầu hết các Baseline. Điểm nhấn lớn nhất là nó làm được điều này với **AC chỉ là 0.94** — tức là chi phí gọi API trung bình chưa tới 1 lần, **tiết kiệm gấp 7 lần** so với Full-Debate (AC=7.21).

Lý do cho sự kỳ diệu này nằm ở cột DSR: **88.53%** mẫu dễ đã được PhoBERT giải quyết hoàn hảo (với AC=0), hệ thống chỉ dồn sức gọi hội đồng chuyên gia cho ~11% mẫu thật sự khó. 

**Kết luận:** Thông qua bảng này, nhóm đã chứng minh thành công giả thuyết ban đầu: Cơ chế Hybrid Debate không chỉ **vượt trội về mặt chất lượng (Accuracy/F1 cao hơn)** mà còn **giải quyết triệt để bài toán chi phí thực tiễn (AC < 1)**."