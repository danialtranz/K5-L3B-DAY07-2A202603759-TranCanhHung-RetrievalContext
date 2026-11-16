# Day 07 · Web chat (React + Vite)

Agentic RAG chat tuân thủ starter Day 7:

- Backend gọi `agentic.run_agent` + `OpenAICompatController` + retriever Day 6/7
- Agent có thể **search nhiều bước** (`max_searches`, mặc định 4)
- FE nhận **một phản hồi cuối** (không stream từng token)
- Cite IEEE `[1]`, `[2]` trên FE (map từ chunk-id của agent)

## Dev

Terminal 1 — API:

```powershell
cd starter
python app.py
```

Terminal 2 — FE:

```powershell
cd starter\web
npm install
npm run dev
```

Mở http://127.0.0.1:5173

## API

- `POST /api/chat` — JSON final answer (dùng bởi FE)
- `POST /api/chat/stream` — SSE chỉ emit `done` (payload cuối), không có `token`
- Default retrieval method: `hybrid_wide`
