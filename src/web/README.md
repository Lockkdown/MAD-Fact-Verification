# ViMAD Web Demo

Giao diện demo tương tác cho hệ thống **ViMAD** (Vietnamese Multi-Agent Debate) — hỗ trợ PLM standalone và debate mode với SSE streaming.

## Khởi động Backend

```bash
cd src/web/backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

API sẽ chạy tại `http://localhost:8000`  
Health check: `GET http://localhost:8000/api/health`

## Khởi động Frontend

```bash
# Terminal mới
cd src/web/frontend
npm install
npm run dev
# → http://localhost:5173
```

## Mock Mode

Toggle **"Dùng Mock Data"** mặc định bật — không cần API key hay model checkpoint.  
Tắt mock để dùng PhoBERT thật (cần checkpoint tại `reports/models_plm/phobert/checkpoints/best_model.pt`).

## Cấu hình hỗ trợ

| Group | Options |
|---|---|
| PLM Standalone | `phobert` |
| Full Debate | `full_n2k3`, `full_n2k5`, `full_n3k3`, `full_n3k5`, `full_n4k3`, `full_n4k5` |
| Hybrid Debate | `hybrid_n2k3`, `hybrid_n2k5`, `hybrid_n3k3`, `hybrid_n3k5`, `hybrid_n4k3`, `hybrid_n4k5` |
