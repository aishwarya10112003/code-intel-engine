"""
build_pipeline() — assembles a RagPipeline for a named configuration.

This one function is the reason evaluation is easy: we can build "dense", "hybrid", or
"hybrid_rerank" pipelines and score them against the same golden test set to PROVE each
upgrade helps. Each phase below simply adds a new config here.

  dense         → vector search only                       (Phase 2/3)
  hybrid        → vector + keyword (BM25) fused with RRF    (Phase 5)
  hybrid_rerank → hybrid, then a cross-encoder reranker     (Phase 6)
"""
from __future__ import annotations

from src.embeddings import Embedder
from src.store import VectorStore

from .pipeline import RagPipeline


def build_pipeline(
    config: str = "dense",
    chunks_path: str = "chunks.json",
    top_k: int = 5,
) -> RagPipeline:
    embedder = Embedder()
    store = VectorStore()

    from src.retrieval.dense import DenseRetriever

    dense = DenseRetriever(embedder, store)

    if config == "dense":
        return RagPipeline(dense, top_k=top_k)

    if config == "hybrid":
        from src.retrieval.base import load_corpus
        from src.retrieval.bm25 import BM25Retriever
        from src.retrieval.hybrid import HybridRetriever

        bm25 = BM25Retriever(load_corpus(chunks_path))
        return RagPipeline(HybridRetriever(dense, bm25), top_k=top_k)

    if config == "hybrid_rerank":
        from src.retrieval.base import load_corpus
        from src.retrieval.bm25 import BM25Retriever
        from src.retrieval.hybrid import HybridRetriever
        from src.retrieval.rerank import CrossEncoderReranker

        bm25 = BM25Retriever(load_corpus(chunks_path))
        hybrid = HybridRetriever(dense, bm25)
        # Retrieve a bigger candidate pool (candidate_k) so the reranker can pick the best.
        return RagPipeline(hybrid, reranker=CrossEncoderReranker(), top_k=top_k, candidate_k=15)

    raise ValueError(f"Unknown config: {config!r} (use dense | hybrid | hybrid_rerank)")
