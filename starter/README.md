# Starter Day 7

| File | Công dụng |
|---|---|
| `agentic.py` | Khung agent ví dụ với số lượt search, top-k và context budget có thể cấu hình |
| `baseline.py` | TF-IDF lexical baseline, chạy offline |
| `convert_day6.py` | Chuyển một số trường JSON từ Day 6 |
| `export_day6.py` | Export chunk strategy C từ Day 6 `lab.db` + copy chroma |
| `build_benchmark.py` | Convert + align gold + thêm 4 câu abstain + `cases.json` |
| `run_retrieval_bench.py` | Benchmark tfidf / dense / hybrid / hybrid_filter / hybrid_wide |
| `run_agent_bench.py` | Agent loop trên cases (LLM hoặc extractive) |
| `llm_controller.py` | OpenAI/OpenRouter tool-calling controller |
| `extractive_controller.py` | Offline fallback khi thiếu/expired API key |
| `retrieval/` | MiniLM + Chroma C + TF-IDF + Hybrid RRF |
| `agent_metrics.py` | Tổng hợp nhãn đánh giá do bạn tự chấm |

## 0. Solution runbook (bài nộp)

```bash
pip install -r requirements.txt
# 1) Export Day 6 (cần sibling repo Day 6 hoặc --day6-data)
python starter/export_day6.py --output starter/data/raw --copy-runtime --runtime-dir starter/data
python starter/build_benchmark.py --raw-chunks starter/data/raw/day6_chunks_raw.json --raw-questions starter/data/raw/day6_questions_raw.json --output starter/data/converted

# 2) Retrieval suite
python starter/run_retrieval_bench.py --chunks starter/data/converted/chunks.json --questions starter/data/converted/questions.json --split dev --output starter/results/retrieval-dev
python starter/run_retrieval_bench.py --chunks starter/data/converted/chunks.json --questions starter/data/converted/questions.json --split test --methods tfidf,hybrid_wide,hybrid_filter --output starter/results/retrieval-test

# 3) Agent với model thật (starter/.env: OPENAI_API_KEY + CHAT_MODEL; reasoning models cần REASONING_EFFORT=none)
python starter/run_agent_bench.py --cases starter/data/converted/cases.json --chunks starter/data/converted/chunks.json --questions starter/data/converted/questions.json --split test --retriever hybrid_wide --budget 5000 --top-k 5 --controller llm --output starter/results/agent-test-hybrid_wide
python starter/run_agent_bench.py --cases starter/data/converted/cases.json --chunks starter/data/converted/chunks.json --questions starter/data/converted/questions.json --split dev --retriever hybrid_wide --budget 5000 --top-k 5 --controller llm --output starter/results/agent-dev-hybrid_wide
python starter/agent_metrics.py --cases starter/data/converted/cases.json --traces starter/results/agent-test-hybrid_wide/traces.json --reviews starter/results/agent-test-hybrid_wide/reviews.json --split test --output starter/results/agent-test-hybrid_wide/summary_metrics.json
```

Cấu hình đã chọn: **hybrid_wide** + **LLM controller**. Báo cáo: `report/individual/2A202603759_TranCanhHung.md`.

### Web UI (agentic + IEEE cites)

```powershell
# Terminal 1 — API
cd starter
python app.py
# Terminal 2 — FE
cd starter\web
npm install
npm run dev
```

Mở http://127.0.0.1:5173 — `POST /api/chat` gọi `run_agent` (multi-search), FE nhận **câu trả lời cuối**; cite chip `[n]` → nguồn/PDF. Chi tiết: [web/README.md](web/README.md).

Starter gốc vẫn là khung; phần trên là đường chạy solution đã nối Day 6.

## 1. Kết nối agent

Trong `agentic.py`, hiện thực:

- `Controller.decide`: gọi model thật và trả `{action: "search", query: "..."}` hoặc `{action: "answer", answer: "... [chunk-id]", citations: ["chunk-id"]}`.
- `Retriever.search`: tạo query embedding tương thích Day 6, truy vấn vector store thật và trả chunk theo [định dạng dữ liệu](../data/README.md).

Tự viết chương trình nối hai phần với `run_agent`, hoặc thay thế bằng kiến trúc nhiều tool. Tham số `max_searches`, `top_k` và `budget` chỉ là mặc định của ví dụ, không phải giới hạn của đề. Xem `tests/test_agentic.py` để hiểu cách gọi; kết nối trong test là mock. Khung code kiểm tra citation ID có trong context, nhưng bạn phải tự đọc xem nguồn có hỗ trợ câu trả lời. Báo lỗi kết nối, không thay bằng kết quả giả.

## 2. Chạy TF-IDF baseline

Chạy từ thư mục gốc với Python 3.11 trở lên:

```bash
python3 -m unittest discover -s tests -v
python3 starter/baseline.py --chunks data/example_chunks.json --questions data/example_questions.json --split dev --output starter/results/example-dev
```

Kết quả có `summary.json` và `traces.json`. Dữ liệu mẫu chỉ có một câu, không đủ để nộp bài. TF-IDF là lexical retrieval, không dùng filter và không thay dense retrieval.

## 3. Dùng dữ liệu Day 6

Xuất chunk và câu hỏi thành hai JSON array. Starter Day 6 không quy định tên bảng hay cách xuất; bạn cần tự thực hiện. Nếu chunk có `chunk_id`, `document_id`, `source`, `page`, `text`, `metadata` và câu hỏi có `gold_source_id`, `gold_page`, `evidence_quote`, chạy:

```bash
python3 starter/convert_day6.py --chunks <day6_chunks.json> --questions <day6_questions.json> --output starter/data/converted
python3 starter/baseline.py --chunks starter/data/converted/chunks.json --questions starter/data/converted/questions.json --split dev --output starter/results/dev-tfidf
```

Thay phần trong `< >` bằng đường dẫn thật. Script chỉ đổi tên trường, không tự truy cập database hay đoán evidence. Nếu cấu trúc khác, tự viết adapter. Đối chiếu lại nguồn, trang và evidence.

`--split dev` chỉ chấm dev. Sau khi chốt cấu hình, dùng `--split test` và chọn thư mục kết quả khác. Nếu test Day 6 từng dùng để chỉnh hệ thống, tạo test mới theo [GUIDE.md](../GUIDE.md). Baseline mẫu dùng evidence nằm trong một chunk và context từ top-3; cần mở rộng hoặc tự đánh giá nếu dùng multi-hop, context từ nhiều lượt hay schema khác. Khớp nguyên văn bằng code không thay thế đánh giá thủ công.

## 4. Tổng hợp đánh giá agent

Sau khi chạy agent thật và tự chấm, chuẩn bị ba JSON array:

- `cases.json`: mỗi câu có `id`, `split`, `answerable` (`true` nếu có đủ evidence).
- `traces.json`: mỗi kết quả có `query_id`, `status` (`answered`, `abstain`, `error`), `events`, `citations`. Thêm `query_id` vào kết quả `run_agent`; lỗi kết nối ghi riêng với `status: "error"`. Script đếm tool use qua event `search`: nếu dùng nhiều tool, chuẩn hóa mỗi lượt retrieval về event này và ghi tên tool trong trường khác, hoặc tự viết metric tương đương.
- `reviews.json`: mỗi nhãn có `query_id`, `citation_support` là danh sách đúng/sai ứng với từng citation. Câu có đáp án cần `answer_grade` 0–2 theo [BENCHMARK.md](../BENCHMARK.md); nếu bị từ chối, chấm 0. Câu thiếu evidence cần `abstention_correct` (`true`/`false`) do người chấm xác nhận. Lần chạy lỗi không cần nhãn.

```bash
python3 starter/agent_metrics.py --cases <cases.json> --traces <traces.json> --reviews <reviews.json> --split test --output starter/results/agent-test/summary.json
```

Thay đường dẫn trong `< >`. Script tổng hợp nhãn bạn đã chấm, không tự đánh giá nội dung. Mẫu số bằng 0 nghĩa là không áp dụng. Lỗi kết nối được báo riêng, không tính vào chất lượng; giữ trace và nhãn để đối chiếu.

## Lưu ý

TF-IDF dùng cách tách từ đơn giản. `baseline.py` dùng mặc định context budget 800 ký tự, gồm cả nguồn; chunk không vừa bị bỏ. Bạn có thể chọn cách khác. File kết quả cùng tên sẽ bị ghi đè. `starter/data/` và `starter/results/` không được Git theo dõi mặc định; nếu được phép chia sẻ, dùng `git add -f <đường-dẫn>`, hoặc ghi cách tái tạo. Không nộp dữ liệu mật và secret.

Khung agent ưu tiên giữ chunk từ lần search trước. Khi chunk đầu chiếm gần hết budget, lần search sau có thể không thêm được evidence; hãy kiểm tra trace và thử cách chọn context phù hợp trên dev. Citation ID hợp lệ hoặc verifier trả “đúng” không đảm bảo câu trả lời được hỗ trợ; cần chấm thủ công.
