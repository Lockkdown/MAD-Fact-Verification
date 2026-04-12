**TO-DO:** 

1. Độ đo AC là gì? cách tính? minh họa luôn bảng kết quả
2. Độ đo DSR là gì? cách tính? minh họa luôn bảng kết quả
3. Confident của ML? nguyên lý sinh ra confident?
4. Cơ chế debate? nguyên lý hoạt động? 
5. Cơ chế tìm threshold trên dev? nguyên lý hoạt động? 
6. Tìm ví dụ mẫu chạy demo.


**WRITING:**

---

## 1. AC — Average Agent Calls

### Định nghĩa
AC là **số lần gọi LLM API trung bình trên mỗi mẫu test**. Nó đo lường **hiệu quả tính toán** của hệ thống — AC càng thấp, hệ thống càng tiết kiệm chi phí API.

### Tại sao sinh ra?
Trong MAD, mỗi vòng debate gọi N debater, cộng thêm 1 lần gọi judge ở cuối. Nếu hệ thống chạy hết k vòng mà không có early stopping, chi phí rất cao. AC giúp đo xem trung bình thực tế tốn bao nhiêu call — phản ánh hiệu quả của cơ chế **early stopping** (dừng sớm khi debaters đồng thuận) và **routing gate** (bỏ qua debate hoàn toàn ở chế độ Hybrid).

### Công thức

**Full Debate (không có routing gate):**
```
AC = avg_rounds_used × N + 1
```
- `avg_rounds_used`: số vòng debate trung bình thực tế (≤ k, vì có early stopping)
- `N`: số debater
- `+1`: judge LUÔN được gọi, bất kể kết quả debate

**Hybrid Debate (có routing gate):**
```
AC = judge_called_rate × (avg_rounds_per_debate × N + 1)
   = (1 - DSR) × (avg_rounds_per_debate × N + 1)
```
- Mẫu bị routing gate chặn lại (high confidence) → 0 agent call
- Chỉ mẫu vào debate mới tốn call

### Minh họa từ kết quả thực tế

**Ví dụ 1 — `full/n3k3` (N=3, k=3):**

| Thông số | Giá trị |
|---|---|
| `avg_rounds_used` | 1.3 |
| N (số debater) | 3 |
| `judge_called_rate` | 1.0 (luôn gọi) |
| **AC tính = 1.3 × 3 + 1** | **= 4.9 ≈ 4.91** ✓ |

> Tại sao rounds_used chỉ là 1.3 dù k=3? Vì early stopping hoạt động rất tốt: **80.44%** mẫu đã đồng thuận ngay vòng 1, chỉ ~6% buộc phải đi hết 3 vòng.

**Cách tính `avg_rounds_used` từ `unanimous_rate` — `full/n3k3`:**

Với early stopping, mỗi mẫu dừng tại vòng đầu tiên debaters đồng thuận. Nếu không bao giờ đồng thuận → chạy hết k vòng. Do đó:

```
avg_rounds_used = Σ (round_i × tỉ_lệ_dừng_ở_round_i) + k × tỉ_lệ_không_bao_giờ_đồng_thuận
```

| Trường hợp | Tỉ lệ | Rounds dùng | Đóng góp |
|---|---|---|---|
| Đồng thuận ở vòng 1 | 0.8044 | 1 | 1 × 0.8044 = 0.8044 |
| Đồng thuận ở vòng 2 | 0.0871 | 2 | 2 × 0.0871 = 0.1742 |
| Đồng thuận ở vòng 3 | 0.0484 | 3 | 3 × 0.0484 = 0.1452 |
| Không bao giờ (→ k=3) | 0.0601 | 3 | 3 × 0.0601 = 0.1803 |
| **Tổng** | **1.0** | | **= 1.3041 ≈ 1.30** ✓ |

**Ví dụ 2 — `hybrid/n4k5` (N=4, k=5):**

| Thông số | Giá trị |
|---|---|
| DSR | 0.8853 (88.53% bỏ qua debate) |
| `judge_called_rate` | 0.1147 (chỉ 11.47% vào debate) |
| `avg_rounds_per_debate` | 1.81 (trung bình rounds khi đã vào debate) |
| N (số debater) | 4 |
| **AC tính = 0.1147 × (1.81 × 4 + 1)** | **= 0.1147 × 8.24 = 0.945 ≈ 0.94** ✓ |

**Cách tính `avg_rounds_per_debate` — `hybrid/n4k5`:**

Ở Hybrid, `unanimous_rate` tính trên **toàn bộ 1,447 mẫu** (kể cả mẫu skip). Mẫu skip không vào debate → chúng đóng góp vào bucket `never`. Cần quy về tỉ lệ **trong số 11.47% mẫu thực sự debate**:

| Vòng dừng | Tỉ lệ (trên tổng) | Tỉ lệ (trong debate) | Rounds | Đóng góp |
|---|---|---|---|---|
| Vòng 1 | 0.0705 | 0.0705/0.1147 = 0.615 | 1 | 0.615 |
| Vòng 2 | 0.0221 | 0.0221/0.1147 = 0.193 | 2 | 0.386 |
| Vòng 3 | 0.0069 | 0.0069/0.1147 = 0.060 | 3 | 0.180 |
| Vòng 4 | 0.0041 | 0.0041/0.1147 = 0.036 | 4 | 0.143 |
| Vòng 5 | 0.0014 | 0.0014/0.1147 = 0.012 | 5 | 0.061 |
| Không bao giờ (→ k=5) | 0.895 − 0.8853 = 0.0097 | 0.0097/0.1147 = 0.085 | 5 | 0.425 |
| **Tổng** | | **≈ 1.0** | | **= 1.810 ≈ 1.81** ✓ |

> `avg_rounds_used` (tổng thể) = `(1 − DSR) × avg_rounds_per_debate` = 0.1147 × 1.81 = **0.21** ✓ — mẫu skip đóng góp 0 rounds.

> So sánh: Full N=3k3 tốn **4.91 calls/mẫu**, Hybrid N=4k5 chỉ tốn **0.94 calls/mẫu** — tiết kiệm hơn **5×** nhờ routing gate.

---

## 2. DSR — Debate Skip Rate

### Định nghĩa
DSR là **tỉ lệ phần trăm mẫu được routing gate chuyển thẳng sang fast path** (dùng kết quả PLM trực tiếp, bỏ qua toàn bộ debate). **Chỉ áp dụng cho chế độ Hybrid**, Full Debate không có DSR.

### Tại sao sinh ra?
Ý tưởng cốt lõi của Hybrid MAD là: **không phải mẫu nào cũng cần tranh luận**. Nếu PLM (PhoBERT) đã rất tự tin (confidence ≥ threshold t*), kết quả của nó đã đủ tốt → tiết kiệm chi phí API bằng cách bỏ qua debate. DSR đo lường tỉ lệ tiết kiệm được này.

### Công thức
```
DSR = số mẫu PLM confidence ≥ t* / tổng số mẫu test
    = 1 - judge_called_rate
```

Trong code, một mẫu được "skip" khi:
```python
max(softmax(plm_logits)) >= threshold  →  fast path (DSR += 1)
```

### Minh họa từ kết quả thực tế

**Ví dụ 1 — `full/n3k3`:**

| Thông số | Giá trị |
|---|---|
| DSR | `null` |
| Lý do | Full Debate không có routing gate — **100% mẫu vào debate** |

**Ví dụ 2 — `hybrid/n4k5` (threshold t* = 0.85):**

| Thông số | Giá trị |
|---|---|
| Tổng mẫu test | 1,447 |
| `judge_called_rate` | 0.1147 |
| DSR = 1 − 0.1147 | **= 0.8853 = 88.53%** ✓ |
| Mẫu skip debate | 1,447 × 0.8853 ≈ **1,281 mẫu** |
| Mẫu vào debate | 1,447 × 0.1147 ≈ **166 mẫu** |

> DSR = 88.53% có nghĩa là gần **9 trong 10 mẫu** được PhoBERT xử lý trực tiếp mà không cần gọi đến LLM debate council. Đây là lý do Hybrid đạt AC chỉ 0.94 thay vì ~8+ nếu tất cả đều debate.

### Lưu ý: DSR cao không phải lúc nào cũng tốt

DSR cao → tiết kiệm chi phí, nhưng nếu routing gate sai quá nhiều, chất lượng giảm:
- `routing_fp_rate = 0.5361` → 53.61% mẫu skip nhưng thực ra PLM dự đoán sai (lãng phí cơ hội sửa sai qua debate)
- `routing_fn_rate = 0.1069` → 10.69% mẫu vào debate nhưng PLM đã đúng (tốn API không cần thiết)

→ Đây là **trade-off giữa hiệu quả (DSR cao) và chất lượng (F1)** — điều chỉnh bằng threshold t*.

