"""Local SentenceTransformer embeddings (compatible with Day 6 MiniLM index)."""
from __future__ import annotations

from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
DEFAULT_DIMENSIONS = 384


class MiniLMEmbedder:
    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name or DEFAULT_MODEL
        self.model = SentenceTransformer(self.model_name)
        dim_fn = getattr(self.model, 'get_embedding_dimension', None) or getattr(
            self.model, 'get_sentence_embedding_dimension', None
        )
        dim = dim_fn() if callable(dim_fn) else None
        self.dimensions = int(dim) if dim else DEFAULT_DIMENSIONS

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 32,
            convert_to_numpy=True,
        )
        return [v.tolist() for v in vectors]

    def embed_query(self, question: str) -> list[float]:
        vector = self.model.encode(
            [question],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]
        return vector.tolist()
