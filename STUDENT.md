# Student solution notes — Day 7

- MSSV / name: 2A202603759 / Trần Cảnh Hưng
- Day 6 sibling: `../K5-L3B-Day06-TranCanhHung-2A202603759-DataIngestionEngineering`
- Selected retrieval: `hybrid_wide` (RRF dense+TF-IDF, candidate_k=12, larger context budget)
- Embedding: `sentence-transformers/all-MiniLM-L6-v2` (384-d), Chroma collection `chunks_C`
- Agent controller: `OpenAICompatController` (`gpt-6-luna`, `REASONING_EFFORT=none`) on hybrid_wide; test metrics in `starter/results/agent-test-hybrid_wide/summary_metrics.json`
- Report: `report/individual/2A202603759_TranCanhHung.md`
