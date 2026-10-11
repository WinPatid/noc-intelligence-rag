"""
NOC Intelligence — Sentence Transformers Embedding Subsystem
Provides a clean, provider-independent embedding interface using local
sentence-transformers with bilingual Thai-English semantic support.
Zero external cloud or network calls.
"""

import os
import math
from typing import List, Optional

# Suppress HuggingFace / tokenizers parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEFAULT_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


class SentenceTransformerEmbedder:
    """
    Local embedding generator wrapping sentence-transformers.
    Features:
    - Configurable model name
    - Lazy model loading on first inference
    - L2 normalization for robust cosine similarity
    - Graceful empty/whitespace text handling
    - Dynamic dimension discovery (never hard-coded)
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model_name = model_name
        self._model = None
        self._dimension = None

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                # Load with local cache support
                self._model = SentenceTransformer(self.model_name)
                # Discover dimension dynamically
                dummy_emb = self._model.encode("test", convert_to_numpy=True)
                self._dimension = int(dummy_emb.shape[0])
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load embedding model '{self.model_name}': {e}"
                )

    def get_dimension(self) -> int:
        """Returns the dynamic vector dimension of the loaded model."""
        self._load_model()
        return self._dimension

    def _normalize_vector(self, vec: List[float]) -> List[float]:
        """L2-normalizes a float vector."""
        norm = math.sqrt(sum(x * x for x in vec))
        if norm < 1e-12:
            return vec
        return [x / norm for x in vec]

    def embed_query(self, text: str) -> List[float]:
        """
        Embeds a single query string.
        Returns a normalized vector of length get_dimension().
        Handles empty or whitespace text gracefully by returning a zero-vector.
        """
        self._load_model()
        cleaned = text.strip() if text else ""
        if not cleaned:
            return [0.0] * self._dimension

        raw_vec = self._model.encode(cleaned, convert_to_numpy=True).tolist()
        return self._normalize_vector(raw_vec)

    def embed_documents(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        """
        Embeds a batch of document strings.
        Returns a list of normalized vectors.
        """
        self._load_model()
        if not texts:
            return []

        # Sanitize empty items within batch
        processed_texts = [t.strip() if (t and t.strip()) else " " for t in texts]

        raw_vecs = self._model.encode(
            processed_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )

        normalized = []
        for v in raw_vecs:
            normalized.append(self._normalize_vector(v.tolist()))
        return normalized

    def __call__(self, input: List[str]) -> List[List[float]]:
        """ChromaDB EmbeddingFunction protocol compatibility."""
        return self.embed_documents(input)


_default_embedder_instance = None


def get_default_embedder() -> SentenceTransformerEmbedder:
    """Returns a singleton instance of the default embedder."""
    global _default_embedder_instance
    if _default_embedder_instance is None:
        _default_embedder_instance = SentenceTransformerEmbedder()
    return _default_embedder_instance


if __name__ == "__main__":
    print(f"[*] Testing Embedder ({DEFAULT_MODEL_NAME})...")
    embedder = get_default_embedder()
    dim = embedder.get_dimension()
    print(f"  [+] Discovered Embedding Dimension: {dim}")

    vec_th = embedder.embed_query("สายไฟเบอร์ขาดที่เชียงใหม่")
    vec_en = embedder.embed_query("Fiber cut in Chiang Mai")
    print(f"  [+] Thai vector length: {len(vec_th)}")
    print(f"  [+] English vector length: {len(vec_en)}")

    # Compute dot product (cosine similarity on unit vectors)
    dot = sum(a * b for a, b in zip(vec_th, vec_en))
    print(f"  [+] Cross-lingual similarity (Thai vs English): {dot:.4f}")
