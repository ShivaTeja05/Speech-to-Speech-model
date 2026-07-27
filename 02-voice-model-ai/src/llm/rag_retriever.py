"""
Retrieval-Augmented Generation (RAG) for the Voice AI pipeline.

Ported from the Demo-THIT prototype (`models/embeddings.py`) and adapted to
reuse voice-model-ai's existing embedding path: instead of pulling in
sentence-transformers, it embeds documents and queries with the same local
Ollama `embed_text` (nomic-embed-text) already used by the semantic cache.

Grounds the LLM in a small knowledge base (FAQs, doctor/department info) so
answers stay factual instead of hallucinated.

Backends:
  - FAISS (if installed) for the vector index
  - NumPy cosine similarity fallback otherwise
"""

import json
import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class RAGRetriever:
    """Embeds a document set once, then answers nearest-neighbour queries."""

    def __init__(self, embed_fn=None):
        # embed_fn(text)->List[float]; defaults to Ollama embed_text (lazy import
        # so the module can be unit-tested with a stub embedder).
        if embed_fn is None:
            from .ollama_client import embed_text
            embed_fn = embed_text
        self._embed = embed_fn
        self.documents: List[Dict[str, Any]] = []
        self._matrix: Optional[np.ndarray] = None
        self._faiss_index = None
        self._ready = False

    # -- indexing ----------------------------------------------------------
    def add_documents(self, documents: List[Dict[str, Any]], text_field: str = "text"):
        for doc in documents:
            if text_field in doc and doc[text_field]:
                d = dict(doc)
                d["_text_field"] = text_field
                self.documents.append(d)

    def build_index(self) -> bool:
        if not self.documents:
            logger.warning("RAG: no documents to index")
            return False
        try:
            texts = [d[d["_text_field"]] for d in self.documents]
            vecs = np.array([self._embed(t) for t in texts], dtype="float32")
            if vecs.ndim != 2 or vecs.shape[0] != len(texts):
                logger.error("RAG: embedding produced unexpected shape %s", vecs.shape)
                return False
            self._matrix = vecs
            try:
                import faiss  # optional
                index = faiss.IndexFlatL2(vecs.shape[1])
                index.add(vecs)
                self._faiss_index = index
                logger.info("RAG: FAISS index built (%d docs, dim %d)", *vecs.shape)
            except Exception:
                self._faiss_index = None
                logger.info("RAG: using NumPy cosine fallback (%d docs)", vecs.shape[0])
            self._ready = True
            return True
        except Exception as e:
            logger.error("RAG: build_index failed: %s", e)
            return False

    def is_ready(self) -> bool:
        return self._ready

    # -- search ------------------------------------------------------------
    def search(self, query: str, top_k: int = 3, language: str = "en") -> List[Dict[str, Any]]:
        if not self._ready or self._matrix is None:
            return []
        try:
            q = np.array(self._embed(query), dtype="float32")
        except Exception as e:
            logger.error("RAG: query embedding failed: %s", e)
            return []

        k = min(top_k, len(self.documents))
        if self._faiss_index is not None:
            distances, indices = self._faiss_index.search(q.reshape(1, -1), k)
            results = []
            for rank, idx in enumerate(indices[0]):
                if 0 <= idx < len(self.documents):
                    doc = dict(self.documents[idx])
                    doc["_score"] = float(1.0 / (1.0 + distances[0][rank]))
                    results.append(doc)
        else:
            results = self._cosine_search(q, k)

        return self._rerank_by_language(results, language)[:top_k]

    def _cosine_search(self, q: np.ndarray, k: int) -> List[Dict[str, Any]]:
        from numpy.linalg import norm
        denom = norm(self._matrix, axis=1) * (norm(q) or 1e-9)
        sims = np.dot(self._matrix, q) / np.where(denom == 0, 1e-9, denom)
        top = np.argsort(sims)[-k:][::-1]
        out = []
        for idx in top:
            doc = dict(self.documents[int(idx)])
            doc["_score"] = float(sims[int(idx)])
            out.append(doc)
        return out

    @staticmethod
    def _rerank_by_language(results: List[Dict[str, Any]], language: str) -> List[Dict[str, Any]]:
        if not language:
            return results
        preferred = [d for d in results if d.get("language") == language]
        other = [d for d in results if d.get("language") != language]
        return preferred + other

    def build_context_block(self, query: str, top_k: int = 3, language: str = "en") -> str:
        """Return a formatted grounding block to inject into the system prompt."""
        hits = self.search(query, top_k=top_k, language=language)
        if not hits:
            return ""
        lines = ["\n\n## KNOWLEDGE BASE (ground your answer in this; do not invent facts)"]
        for i, doc in enumerate(hits, 1):
            text = doc.get(doc.get("_text_field", "text"), "")
            lines.append(f"{i}. {text}")
        return "\n".join(lines)


def load_documents_from_json(filepath: str) -> List[Dict[str, Any]]:
    """Load a knowledge-base JSON file: a list of {text, language?, ...} dicts."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "documents" in data:
            data = data["documents"]
        return data if isinstance(data, list) else []
    except Exception as e:
        logger.error("RAG: failed to load %s: %s", filepath, e)
        return []
