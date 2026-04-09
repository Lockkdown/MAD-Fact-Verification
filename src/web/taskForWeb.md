Hãy đọc kỹ @CODEBASE_CONTEXT.md trước khi làm bất cứ điều gì. Đó là nguồn sự thật duy nhất về tên, hằng số, và kiến trúc dự án.

---

## 🎯 Mục tiêu

Tạo một web demo tương tác cho hệ thống ViMAD (Vietnamese Multi-Agent Debate) tại thư mục `web/` trong project root.

---

## 📁 Cấu trúc thư mục cần tạo

```
web/
├── backend/
│   ├── main.py              # FastAPI app entry point
│   ├── routers/
│   │   └── predict.py       # POST /api/predict (SSE streaming)
│   ├── services/
│   │   ├── plm_service.py   # Singleton loader cho tất cả PLM models
│   │   └── debate_service.py # Mock + real debate orchestration
│   ├── models/
│   │   └── schemas.py       # Pydantic request/response schemas
│   ├── mock/
│   │   └── mock_responses.py # Mock data để test không cần API key
│   └── requirements.txt
└── frontend/
    ├── src/
    │   ├── App.jsx
    │   ├── components/
    │   │   ├── InputSection.jsx      # Claim + Evidence textarea
    │   │   ├── ConfigSelector.jsx    # Radio group chọn config
    │   │   ├── AnalyzeButton.jsx     # Nút phân tích + loading state
    │   │   ├── ResultPanel.jsx       # Container kết quả
    │   │   ├── PLMResult.jsx         # Kết quả cho PLM mode
    │   │   ├── AgentCard.jsx         # Card cho mỗi debater/judge
    │   │   ├── ThinkingAnimation.jsx # Animated thinking indicator
    │   │   └── VerdictBadge.jsx      # Badge Support/Refute/NEI
    │   ├── hooks/
    │   │   └── usePredict.js         # Custom hook xử lý SSE stream
    │   ├── constants/
    │   │   └── agents.js             # Agent metadata (màu, icon, tên)
    │   └── main.jsx
    ├── package.json
    ├── vite.config.js        # Proxy /api → localhost:8000
    └── index.html
```

---

## 🖥️ Backend — FastAPI

### `backend/main.py`

- Dùng `FastAPI` với `lifespan` context manager
- Khi startup: gọi `plm_service.load để load 1 PLM model PhoBERT vào memory (singleton)
- Mount CORS cho `http://localhost:5173`
- Include router từ `routers/predict.py`
- Endpoint `GET /api/health` trả về `{"status": "ok", "models_loaded": [...]}`

### `backend/services/plm_service.py`

### `backend/services/plm_service.py`
- Class `PLMService` singleton, load duy nhất `phobert` (`vinai/phobert-base`) khi startup
- ⚠️ Bắt buộc dùng `pyvi.ViTokenizer.tokenize()` trước khi tokenize
- Checkpoint: `reports/models_plm/phobert/checkpoints/best_model.pt`
- Nếu checkpoint không tồn tại → mock prediction, log warning
- `predict(claim, evidence)` → `{"label": "Support"|"Refute"|"NEI", "confidence": float, "probabilities": {...}}`
- ⚠️ PhoBERT bắt buộc dùng `pyvi.ViTokenizer.tokenize()` để segment text trước khi tokenize
- Checkpoint path pattern: `reports/models_plm/{model_key}/checkpoints/best_model.pt` (relative to project root)
- Nếu checkpoint không tồn tại → fallback sang mock prediction, log warning
- `predict(model_key, claim, evidence)` → `{"label": "Support"|"Refute"|"NEI", "confidence": float, "probabilities": {"Support": f, "Refute": f, "NEI": f}}`

### `backend/routers/predict.py`

- `POST /api/predict` nhận body:
```json
{
  "claim": "string",
  "evidence": "string", 
  "config": "phobert|xlmr|mbert|vibert|full_n2k3|full_n2k5|full_n3k3|full_n3k5|full_n4k3|full_n4k5|hybrid_n2k3|hybrid_n2k5|hybrid_n3k3|hybrid_n3k5|hybrid_n4k3|hybrid_n4k5",
  "use_mock": false
}
```
- Response: `StreamingResponse` với `media_type="text/event-stream"`
- SSE event format (mỗi event là 1 dòng `data: {json}\n\n`):

```
# PLM mode — chỉ 2 events
data: {"type": "plm_start", "model": "phobert"}
data: {"type": "plm_result", "label": "Support", "confidence": 0.923, "probabilities": {...}}

# Debate mode — sequence of events
data: {"type": "round_start", "round": 1, "total_rounds": 3}
data: {"type": "agent_thinking", "agent_id": "mistral", "round": 1}
data: {"type": "agent_result", "agent_id": "mistral", "round": 1, "verdict": "Support", "reasoning": "...full reasoning text..."}
data: {"type": "agent_thinking", "agent_id": "gpt4o_mini", "round": 1}
data: {"type": "agent_result", "agent_id": "gpt4o_mini", "round": 1, "verdict": "Refute", "reasoning": "..."}
# ... các debaters còn lại
data: {"type": "round_end", "round": 1, "is_unanimous": false}
data: {"type": "round_start", "round": 2, "total_rounds": 3}
# ... tiếp tục
data: {"type": "judge_thinking"}
data: {"type": "judge_result", "verdict": "Support", "reasoning": "...judge reasoning...", "rounds_used": 2}
data: {"type": "final", "label": "Support", "rounds_used": 2, "total_agent_calls": 7}
data: {"type": "done"}

# Hybrid mode — thêm PLM routing step ở đầu
data: {"type": "routing", "plm_model": "phobert", "confidence": 0.923, "threshold": 0.85, "routed_to": "fast_path"|"debate"}
# Nếu fast_path → chỉ emit routing + final
# Nếu debate → emit routing rồi tiếp tục debate flow như trên
```

### `backend/mock/mock_responses.py`

- Hàm `generate_mock_debate(claim, evidence, config)` → async generator yield SSE events
- Mỗi `agent_thinking` event: sleep 0.8–1.2s (random) để giả lập latency
- Mỗi `agent_result`: reasoning text dài ~150 words, có đề cập đến claim/evidence thực sự được pass vào
- Judge thinking: sleep 1.5s
- Reasoning text phải realistic (tiếng Anh, format fact-checking analysis)

---

## 🎨 Frontend — ReactJS + TailwindCSS

### Design System

- **Màu nền:** `#F8F9FA` (trắng xám nhẹ)
- **Card:** `#FFFFFF`, border `1px solid #E5E7EB`, border-radius `12px`
- **Font:** System UI / Inter
- **Không dùng:** gradient, glassmorphism, shadow quá đậm
- **Accent:** `#2563EB` (xanh dương) cho nút chính

### Agent Identity — `frontend/src/constants/agents.js`

```js
export const AGENTS = {
  mistral: {
    id: "mistral",
    name: "Mistral Small 4",
    shortName: "Mistral",
    ecosystem: "EU",
    color: "#7C3AED",        // tím
    bgColor: "#F5F3FF",
    borderColor: "#DDD6FE",
    icon: "⚡",
  },
  gpt4o_mini: {
    id: "gpt4o_mini", 
    name: "GPT-4o mini",
    shortName: "GPT-4o mini",
    ecosystem: "US",
    color: "#059669",        // xanh lá
    bgColor: "#ECFDF5",
    borderColor: "#A7F3D0",
    icon: "✦",
  },
  qwen: {
    id: "qwen",
    name: "Qwen-2.5-72B",
    shortName: "Qwen",
    ecosystem: "CN",
    color: "#D97706",        // cam
    bgColor: "#FFFBEB",
    borderColor: "#FDE68A",
    icon: "◈",
  },
  llama: {
    id: "llama",
    name: "Llama-3.3-70B",
    shortName: "Llama",
    ecosystem: "US",
    color: "#2563EB",        // xanh dương
    bgColor: "#EFF6FF",
    borderColor: "#BFDBFE",
    icon: "◎",
  },
  judge: {
    id: "judge",
    name: "DeepSeek-V3",
    shortName: "Judge",
    ecosystem: "CN",
    color: "#DC2626",        // đỏ
    bgColor: "#FEF2F2",
    borderColor: "#FECACA",
    icon: "⚖",
  },
};

// Map config → debaters list theo N
export const CONFIG_AGENTS = {
  n2: ["mistral", "gpt4o_mini"],
  n3: ["mistral", "gpt4o_mini", "qwen"],
  n4: ["mistral", "gpt4o_mini", "qwen", "llama"],
};
```

### `ConfigSelector.jsx`

3 nhóm radio/tab:

**Nhóm 1 — PLM Standalone:**
| Label hiển thị | config value |
|---|---|
| PhoBERT ★ | `phobert` |

Chú thích nhỏ dưới nhóm: `"★ PhoBERT là Routing Gate (M*) — F1: 85.08%"`

**Nhóm 2 — Full Debate:**
Grid 2×3: N2K3, N2K5, N3K3, N3K5, N4K3, N4K5
Mỗi option hiển thị: label lớn `N=2, k=3` + label nhỏ màu xám `"2 debaters · 3 rounds"`

**Nhóm 3 — Hybrid Debate:**
Grid 2×3: tương tự Full Debate
Thêm badge nhỏ `"t*=0.85"` ở góc trên phải mỗi option

Default selected: `phobert`

### `AgentCard.jsx`

```
┌─────────────────────────────────────────┐
│ ⚡ Mistral Small 4          [EU]  Round 1│
│─────────────────────────────────────────│
│ Verdict: [Support ✓]                    │
│                                         │
│ ▼ Reasoning  (click to expand)          │
│   "The evidence clearly states that..." │
└─────────────────────────────────────────┘
```

- Border-left `4px solid {agent.color}`
- Header background: `{agent.bgColor}`
- Verdict badge: xanh lá cho Support, đỏ cho Refute, vàng cho NEI
- Reasoning: collapsed mặc định, expand khi click, max-height với overflow-y scroll
- Khi đang thinking (trước khi có result): hiện `ThinkingAnimation` thay cho content

### `ThinkingAnimation.jsx`

- 3 chấm nhảy lên xuống lần lượt (bounce animation CSS)
- Text bên cạnh: `"Đang phân tích..."` cho debater, `"Đang tổng hợp phán quyết..."` cho judge
- Màu text: màu của agent tương ứng

### `VerdictBadge.jsx`

```
Support  → bg:#DCFCE7 text:#166534 border:#BBF7D0  icon: ✓
Refute   → bg:#FEE2E2 text:#991B1B border:#FECACA  icon: ✗  
NEI      → bg:#FEF3C7 text:#92400E border:#FDE68A  icon: ?
```

Kích thước lớn cho Final Verdict, nhỏ cho từng agent card.

### `usePredict.js` — Custom Hook

```js
// Quản lý SSE connection
// State: { status, events, rounds, finalVerdict, error }
// status: 'idle' | 'loading' | 'streaming' | 'done' | 'error'
// Expose: { state, startPredict(claim, evidence, config), cancel() }
// Dùng AbortController để cancel khi user bấm lại nút
// Parse SSE events → update state theo từng event type
```

### Layout tổng thể `App.jsx`

```
[Header: "ViMAD Demo — Vietnamese Fact Checking"]

[Card: Input]
  Claim:    [textarea, 3 rows, placeholder: "Nhập luận điểm cần kiểm tra..."]
  Evidence: [textarea, 5 rows, placeholder: "Nhập bằng chứng liên quan..."]

[Card: Cấu hình]
  [ConfigSelector]
  [Toggle: "Dùng Mock Data (không cần API key)"] ← default ON

[Button: "🔍 Phân tích" → disabled khi đang stream, text đổi thành "⏹ Dừng lại"]

[Card: Kết quả — chỉ hiện sau khi bấm phân tích]
  PLM mode → PLMResult (spinner → confidence bar → verdict)
  Debate mode → theo thứ tự rounds:
    [Round 1 header]
    [AgentCard × N — render từng cái khi event đến]
    [Round 2 header] (nếu có)
    ...
    [Divider: "⚖ Phán quyết cuối cùng"]
    [AgentCard judge — nổi bật hơn, border đậm hơn]
    [VerdictBadge lớn — Final Verdict]
    [Stats row: Rounds used · Agent calls · Config]
```

---

## ⚙️ Vite Config — `vite.config.js`

```js
// proxy: { '/api': 'http://localhost:8000' }
```

---

## 📦 Dependencies

**Backend `requirements.txt`:**
```
fastapi
uvicorn[standard]
transformers
torch
pyvi
pydantic
httpx
```

**Frontend `package.json`:**
```json
"dependencies": {
  "react": "^18",
  "react-dom": "^18"
},
"devDependencies": {
  "vite": "^5",
  "@vitejs/plugin-react": "^4",
  "tailwindcss": "^3",
  "autoprefixer": "^10",
  "postcss": "^8"
}
```

---

## ✅ Ràng buộc bắt buộc

1. **Label strings PHẢI đúng casing:** `Support`, `Refute`, `NEI` — không dùng `SUPPORTS`, `supports`, `not enough info`
2. **PhoBERT PHẢI dùng PyVi** để segment text trước khi tokenize — xem CODEBASE_CONTEXT.md
3. **PLM load một lần duy nhất** khi server startup, không re-load mỗi request
4. **Mock mode mặc định ON** — nếu checkpoint không tồn tại hoặc toggle mock=true → dùng mock
5. **AbortController** trong frontend — cancel stream khi user bấm "Dừng lại"
6. **Không hardcode model strings** trong frontend logic — dùng `AGENTS` constant
7. **Judge card phải render SAU khi tất cả rounds kết thúc**, không render song song với debaters

---

## 🚀 README tối thiểu

Tạo `web/README.md` với hướng dẫn:
```
# Khởi động backend
cd web/backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Khởi động frontend (terminal mới)
cd web/frontend
npm install
npm run dev
# → http://localhost:5173
```