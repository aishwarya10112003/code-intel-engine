"""
The Embedder — turns text into vectors (lists of numbers) that capture *meaning*.

WHY vectors?
  A computer can't compare the *meaning* of two sentences directly. So we convert each
  piece of text into a list of numbers (an "embedding") using a neural network trained
  so that texts with similar meaning end up as vectors that are close together. Then
  "find similar text" becomes "find nearby vectors" — pure math.

MODEL:
  We use `BAAI/bge-small-en-v1.5`, a small, fast, open-source embedding model that runs
  locally (no API, no cost). It outputs 384-dimensional vectors.

ONE IMPORTANT DETAIL (asymmetric embedding):
  bge models work best when you tell them whether a text is a *query* (a question) or a
  *document* (a passage to be searched). Queries get a short instruction prefix;
  documents don't. Using the wrong mode quietly hurts retrieval quality — so we expose
  two methods: `embed_documents` and `embed_query`.
"""
from __future__ import annotations

from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Recommended by the bge authors: prepend this to a QUERY (not to documents).
_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class Embedder:
    def __init__(self, model_name: str = MODEL_NAME) -> None:
        # Loads the model once (downloads it the first time, then caches on disk).
        self.model = SentenceTransformer(model_name)

    @property
    def dimension(self) -> int:
        """How many numbers are in each vector (384 for bge-small)."""
        # Method was renamed across sentence-transformers versions; support both.
        get_dim = getattr(self.model, "get_embedding_dimension", None) or \
            self.model.get_sentence_embedding_dimension
        return get_dim()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed passages/chunks to be stored and searched over."""
        vectors = self.model.encode(
            texts,
            # normalize → every vector has length 1, so cosine similarity = dot product.
            # This makes "how similar are these two vectors?" a fast, clean comparison.
            normalize_embeddings=True,
            batch_size=64,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed a user's question — with the query instruction prefix."""
        vector = self.model.encode(
            _QUERY_INSTRUCTION + text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vector.tolist()
