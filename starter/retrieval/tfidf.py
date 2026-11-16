"""TF-IDF lexical retriever returning Day 7 chunk dicts."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Allow importing starter/baseline.py when run as script from repo root.
_STARTER = Path(__file__).resolve().parents[1]
if str(_STARTER) not in sys.path:
    sys.path.insert(0, str(_STARTER))

from baseline import TfidfIndex  # noqa: E402


class TfidfRetriever:
    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self.chunks = chunks
        self.index = TfidfIndex(chunks)

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        # Lexical baseline ignores metadata filters (documented Day 7 starter behavior).
        _ = filters
        results = self.index.search(query, top_k=top_k)
        out = []
        for c in results:
            out.append(
                {
                    'id': c['id'],
                    'text': c['text'],
                    'source_id': c['source_id'],
                    'source': c['source'],
                    'page': c['page'],
                    'metadata': c.get('metadata') or {},
                    'score': float(c.get('score') or 0.0),
                }
            )
        return out
