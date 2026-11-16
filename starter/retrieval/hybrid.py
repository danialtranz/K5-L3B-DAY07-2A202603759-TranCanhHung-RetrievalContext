"""Hybrid retrieval via Reciprocal Rank Fusion (dense + TF-IDF)."""
from __future__ import annotations

from typing import Any, Protocol


class _Searcher(Protocol):
    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        ...


def rrf_fuse(
    ranked_lists: list[list[dict[str, Any]]],
    *,
    top_k: int,
    k: int = 60,
) -> list[dict[str, Any]]:
    scores: dict[str, float] = {}
    best: dict[str, dict[str, Any]] = {}
    for results in ranked_lists:
        for rank, hit in enumerate(results, start=1):
            cid = hit['id']
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            prev = best.get(cid)
            if prev is None or float(hit.get('score') or 0) > float(prev.get('score') or 0):
                best[cid] = hit
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
    out = []
    for cid, fused in ordered:
        item = dict(best[cid])
        item['score'] = fused
        out.append(item)
    return out


class HybridRetriever:
    def __init__(
        self,
        dense: _Searcher,
        lexical: _Searcher,
        *,
        candidate_k: int = 8,
        rrf_k: int = 60,
    ) -> None:
        self.dense = dense
        self.lexical = lexical
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        # Dense can apply metadata filters; lexical ignores them but still contributes.
        dense_hits = self.dense.search(query, filters=filters, top_k=self.candidate_k)
        lexical_hits = self.lexical.search(query, filters=None, top_k=self.candidate_k)
        return rrf_fuse([dense_hits, lexical_hits], top_k=top_k, k=self.rrf_k)
