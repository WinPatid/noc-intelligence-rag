"""
NOC Intelligence — Semantic Retrieval Engine
Provides top-k dense semantic search over indexed work orders and trouble tickets.
Supports validated metadata filters and threshold filtering.
Explicitly distinguishes between Cosine Distance (ChromaDB native) and Cosine Similarity.
"""

import os
os.environ["ANONYMIZED_TELEMETRY"] = "false"

from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings

from .embedder import get_default_embedder, SentenceTransformerEmbedder
from .indexer import DEFAULT_CHROMA_DIR


class SemanticRetriever:
    """
    Retrieval client for ChromaDB vector collections.
    
    SCORING CONVENTION:
    - Distance (ChromaDB native): Cosine distance D in [0, 2].
      D = 0.0 means identical angle; D = 1.0 means orthogonal.
    - Similarity Score: S = round(1.0 - D, 4).
      S = 1.0 means perfect semantic match; S = 0.0 means orthogonal.
      Threshold filtering uses S: results with S < threshold are excluded.
    """

    def __init__(
        self,
        chroma_dir: str = DEFAULT_CHROMA_DIR,
        embedder: Optional[SentenceTransformerEmbedder] = None
    ):
        self.chroma_dir = chroma_dir
        self.embedder = embedder or get_default_embedder()
        self.client = chromadb.PersistentClient(
            path=self.chroma_dir,
            settings=Settings(anonymized_telemetry=False)
        )

    def _get_collection(self, name: str):
        try:
            return self.client.get_collection(name=name)
        except Exception:
            return None

    def _parse_chroma_results(
        self,
        raw_results: Dict[str, Any],
        threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        if not raw_results or not raw_results.get("documents") or not raw_results["documents"][0]:
            return []

        docs = raw_results["documents"][0]
        metas = raw_results["metadatas"][0]
        distances = raw_results["distances"][0]
        ids = raw_results["ids"][0]

        parsed = []
        for doc_id, doc, meta, dist in zip(ids, docs, metas, distances):
            similarity_score = round(1.0 - dist, 4)

            # Apply threshold filter on similarity score if requested
            if threshold is not None and similarity_score < threshold:
                continue

            parsed.append({
                "id": doc_id,
                "document": doc,
                "metadata": meta,
                "cosine_distance": round(dist, 4),
                "similarity_score": similarity_score
            })

        # Sort by similarity score descending (distance ascending)
        parsed.sort(key=lambda x: x["similarity_score"], reverse=True)
        return parsed

    def query_work_orders(
        self,
        query: str,
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
        threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves engineering work orders by semantic symptom or technical description.
        Returns empty list if query is empty or collection does not exist.
        """
        cleaned = query.strip() if query else ""
        if not cleaned:
            return []

        col = self._get_collection("work_orders")
        if col is None or col.count() == 0:
            return []

        query_vec = self.embedder.embed_query(cleaned)
        kwargs = {
            "query_embeddings": [query_vec],
            "n_results": min(top_k, col.count()),
            "include": ["documents", "metadatas", "distances"]
        }
        if where:
            kwargs["where"] = where

        try:
            results = col.query(**kwargs)
            return self._parse_chroma_results(results, threshold=threshold)
        except Exception:
            # Fallback if where filter causes Chroma error
            return []

    def query_trouble_tickets(
        self,
        query: str,
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
        threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves trouble tickets by reported symptom or incident description.
        """
        cleaned = query.strip() if query else ""
        if not cleaned:
            return []

        col = self._get_collection("trouble_tickets")
        if col is None or col.count() == 0:
            return []

        query_vec = self.embedder.embed_query(cleaned)
        kwargs = {
            "query_embeddings": [query_vec],
            "n_results": min(top_k, col.count()),
            "include": ["documents", "metadatas", "distances"]
        }
        if where:
            kwargs["where"] = where

        try:
            results = col.query(**kwargs)
            return self._parse_chroma_results(results, threshold=threshold)
        except Exception:
            return []

    def query_all(
        self,
        query: str,
        top_k: int = 5,
        filter_customer_id: Optional[int] = None,
        filter_circuit_ref: Optional[str] = None,
        threshold: Optional[float] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Unified multi-collection retrieval querying both work orders and trouble tickets.
        Applies optional customer or circuit scoping filters where available.
        """
        ticket_where = None
        if filter_customer_id and filter_circuit_ref:
            ticket_where = {
                "$and": [
                    {"customer_id": filter_customer_id},
                    {"circuit_ref": filter_circuit_ref}
                ]
            }
        elif filter_customer_id:
            ticket_where = {"customer_id": filter_customer_id}
        elif filter_circuit_ref:
            ticket_where = {"circuit_ref": filter_circuit_ref}

        return {
            "work_orders": self.query_work_orders(query, top_k=top_k, threshold=threshold),
            "trouble_tickets": self.query_trouble_tickets(
                query, top_k=top_k, where=ticket_where, threshold=threshold
            )
        }
