"""Read-only Chroma search over Day 6 persistent collections."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

from .embeddings import MiniLMEmbedder


class DenseChromaRetriever:
    """Dense retrieval on Day 6 collection chunks_{STRATEGY}."""

    def __init__(
        self,
        persist_dir: str | Path,
        embedder: MiniLMEmbedder,
        strategy: str = 'C',
    ) -> None:
        self.persist_dir = Path(persist_dir)
        self.embedder = embedder
        self.strategy = strategy.upper()
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_collection(name=f'chunks_{self.strategy}')

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        if top_k < 1:
            raise ValueError('top_k must be >= 1')
        vector = self.embedder.embed_query(query)
        where = None
        if filters:
            allowed: dict[str, Any] = {}
            for key in ('section', 'chunk_type', 'document_id', 'page'):
                if key in filters and filters[key] not in (None, ''):
                    allowed[key] = filters[key]
            if len(allowed) == 1:
                where = allowed
            elif len(allowed) > 1:
                where = {'$and': [{k: v} for k, v in allowed.items()]}

        result = self.collection.query(
            query_embeddings=[vector],
            n_results=top_k,
            where=where,
            include=['documents', 'metadatas', 'distances'],
        )
        ids = (result.get('ids') or [[]])[0]
        docs = (result.get('documents') or [[]])[0]
        metas = (result.get('metadatas') or [[]])[0]
        dists = (result.get('distances') or [[]])[0]

        hits: list[dict[str, Any]] = []
        for cid, doc, meta, dist in zip(ids, docs, metas, dists):
            meta = meta or {}
            score = 1.0 - float(dist) if dist is not None else 0.0
            hits.append(
                {
                    'id': cid,
                    'text': doc or '',
                    'source_id': str(meta.get('document_id') or ''),
                    'source': str(meta.get('source') or ''),
                    'page': int(meta.get('page') or 0),
                    'metadata': {
                        'strategy_id': str(meta.get('strategy_id') or self.strategy),
                        'chunk_type': str(meta.get('chunk_type') or 'text'),
                        'section': str(meta.get('section') or ''),
                        'figure_id': str(meta.get('figure_id') or ''),
                        'table_id': str(meta.get('table_id') or ''),
                    },
                    'score': score,
                }
            )
        return hits
