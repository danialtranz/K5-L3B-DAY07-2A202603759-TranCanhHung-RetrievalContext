# Báo cáo Day 7

**MSSV:** 2A202603759 · **Họ tên:** Trần Cảnh Hưng · **Lớp:** L3B

## Dữ liệu và benchmark

- Nguồn, quyền sử dụng, số trang và số chunk: Wiley DDBMS Chapter 5 (Rahimi & Haug, 2010); PDF sách có bản quyền — repo không phân phối PDF; dùng bản local như Day 6. Strategy **C**: **242 chunks** (heading+paragraph pack + section metadata). Export từ `lab.db` qua `starter/export_day6.py` + `starter/build_benchmark.py` (căn evidence hyphen/whitespace).
- Vector store, embedding model, phiên bản và số chiều: Chroma persistent cosine (`starter/data/chroma`, collection `chunks_C`); `sentence-transformers/all-MiniLM-L6-v2`, **384-d**.
- Agent model, cách chạy và phân chia dev/test: **`OpenAICompatController`** với model `gpt-6-luna` (OpenAI Chat Completions + tools, `REASONING_EFFORT=none` vì model reasoning không cho tools nếu để effort mặc định). Retriever đã chọn: `hybrid_wide`. Split: **20 answerable + 2 abstain (dev)** / **5 + 2 (test)**.
- Cách xác định câu có/thiếu evidence và tiêu chí chấm: có đáp án khi evidence nằm trong chunk đúng nguồn/trang; abstain (abs01–abs04) ngoài corpus. Retrieval: Hit@1 / Evidence Hit@3 / Context Evidence Rate. Agent: grounded 0–2, citation_support, abstention_correct (đọc thủ công).

## Retrieval và thử nghiệm

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| Baseline | TF-IDF lexical, top-k=3, budget 800 | Lexical baseline offline | dev | 9/20 (0.45) | 16/20 (0.80) | 3/20 (0.15) | median ~0.5 ms / $0 |
| Dense | Chroma C + MiniLM, top-k=3, budget 800 | Dense Day 6 | dev | 9/20 (0.45) | 16/20 (0.80) | 2/20 (0.10) | ~13 ms / $0 |
| Hybrid | RRF dense+TF-IDF, cand=8, top-k=3, budget 800 | Fusion xếp hạng | dev | 11/20 (0.55) | 18/20 (0.90) | 3/20 (0.15) | ~13 ms / $0 |
| Cải thiện 1 | Hybrid + section filter user (`hybrid_filter`) | Filter giảm nhiễu trên câu có section | dev | 12/20 (0.60) | 18/20 (0.90) | 3/20 (0.15) | ~14 ms / $0 |
| Cải thiện 2 | Hybrid cand=12, budget 1200 (`hybrid_wide`) | Budget lớn hơn giữ evidence trong context | dev | 11/20 (0.55) | 17/20 (0.85) | **11/20 (0.55)** | ~13 ms / $0 |
| Cấu hình đã chọn | hybrid_wide (agent: top-k=5, budget 2000) | Ưu tiên Context Evidence Rate cho grounded answer | test | 3/5 (0.60) | 5/5 (1.00) | 2/5 (0.40) | ~15 ms / $0 |
| Baseline | TF-IDF | So sánh test | test | 2/5 (0.40) | 4/5 (0.80) | 0/5 (0.00) | ~0.7 ms / $0 |

Hai lỗi/dev insight và lý do chọn:

1. **Budget 800 cắt mất evidence dù Hit@3 đúng** — dense/hybrid đạt Hit@3 cao nhưng Context Evidence Rate chỉ 0.10–0.15 vì chunk dài. Thử `hybrid_wide` (budget 1200) nâng context hit lên 0.55 trên dev → chọn cho agent.
2. **q24 (test) false abstain với LLM** — gold exclusive/shared modes có trong hit `00122` ở lần search cuối, nhưng `pack_context` giữ chunk dài `00130` trước nên model không thấy evidence và abstain sau `max_searches=4`. Giả thuyết tiếp: dedupe/re-rank theo keyword overlap trước khi pack, hoặc tăng số chunk trong budget.

Chọn **hybrid_wide** thay vì hybrid_filter vì agent cần context đủ evidence hơn là Hit@1 tối đa.

## Agent và đánh giá

- Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi: tools `search` + `final_answer`; `max_searches=4`; context qua `pack_context`; lỗi provider ghi `status=error`. Dev LLM: 13 answered / 9 abstain (gồm 2 abstain đúng abs01–02 + vài false abstain trên câu có đáp án). Test LLM: 4 answered / 3 abstain (trong đó 1 false abstain q24).

| Cấu hình / split | Tool use | Số lượt gọi | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| hybrid_wide + gpt-6-luna / test | 7/7 | 1–4 search/câu (median ~6.3s) | 0:1, 1:0, 2:4 · mean **1.6**/2 (N=5) | 4/4 | 4/4 | 2/2 | 1/5 |

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| q21 | có đáp án | 1× search | phantoms + row locking predicate `[C_..._00090]` | grade 2; citation support |
| q22 | có đáp án | 1× search | TO theo timestamp/age `[C_..._00096]` | grade 2; citation support |
| q23 | có đáp án | 1× search | EP/VP/CP `[C_..._00113]` | grade 2 |
| q24 | có đáp án | 4× search rồi abstain | từ chối | grade 0; false abstention |
| q25 | có đáp án | 1× search | centralized 2PL / CLM `[C_..._00151]` | grade 2; citation support |
| abs03 | thiếu evidence | 4× search rồi abstain | từ chối | abstention_correct true |
| abs04 | thiếu evidence | 4× search rồi abstain | từ chối | abstention_correct true |

- Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí:
  - Retrieval: `starter/results/retrieval-dev/`, `starter/results/retrieval-test/`
  - Agent LLM: `starter/results/agent-test-hybrid_wide/`, `starter/results/agent-dev-hybrid_wide/`
  - Metrics: `starter/results/agent-test-hybrid_wide/summary_metrics.json`
  - Lệnh: `starter/README.md` §0 với `--controller llm`
  - Embedding local $0; chi phí chat API theo usage OpenAI (không log USD chi tiết trong run này). Median latency agent test ~6.3s/câu.
- Kết quả trước/sau, ví dụ lỗi, đánh đổi và giới hạn: tăng budget cải thiện context hit retrieval; LLM paraphrase tốt hơn extractive trên q21–q23/q25 nhưng vẫn false abstain q24 do context packing. Giới hạn: PDF không nộp; test 5 câu có đáp án; reasoning model cần `REASONING_EFFORT=none` khi dùng tools.

## Thử thách nâng cao (nếu thực hiện)

Không làm trong submission này.
