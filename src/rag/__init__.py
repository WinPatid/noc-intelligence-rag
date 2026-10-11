"""
NOC Intelligence — Semantic Vector Store Subsystem (ChromaDB + sentence-transformers)
"""
from .embedder import SentenceTransformerEmbedder, get_default_embedder
from .indexer import SemanticIndexer, build_or_update_index, DEFAULT_CHROMA_DIR
from .retriever import SemanticRetriever

__all__ = [
    "SentenceTransformerEmbedder",
    "get_default_embedder",
    "SemanticIndexer",
    "build_or_update_index",
    "DEFAULT_CHROMA_DIR",
    "SemanticRetriever"
]
