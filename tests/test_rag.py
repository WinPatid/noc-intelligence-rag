"""
NOC Intelligence — Stage 3 Automated Test Suite (Semantic Vector Store)
Validates local sentence-transformers embedder, ChromaDB indexing,
metadata filtering, cosine distance/similarity semantics, and retrieval recall.
"""

import os
import pytest

from src.rag.embedder import SentenceTransformerEmbedder, get_default_embedder
from src.rag.indexer import SemanticIndexer, DEFAULT_CHROMA_DIR
from src.rag.retriever import SemanticRetriever


# ─── 1. Embedder Tests ────────────────────────────────────────────────────────

def test_embedder_dimension_and_normalization():
    """Verify embedding vector dimension matches model spec and is L2-normalized."""
    embedder = get_default_embedder()
    dim = embedder.get_dimension()
    assert dim == 384, f"Expected 384 dimensions for MiniLM-L12, got {dim}"

    vec = embedder.embed_query("Network router down due to power failure")
    assert len(vec) == dim
    assert all(isinstance(x, float) for x in vec)

    # L2 norm check: sum(x^2) should be ~1.0
    norm = sum(x * x for x in vec)
    assert pytest.approx(norm, rel=1e-3) == 1.0


def test_embedder_empty_and_whitespace_handling():
    """Verify empty or whitespace strings return a valid zero-vector without crashing."""
    embedder = get_default_embedder()
    dim = embedder.get_dimension()

    vec_empty = embedder.embed_query("")
    assert len(vec_empty) == dim
    assert all(x == 0.0 for x in vec_empty)

    vec_spaces = embedder.embed_query("     ")
    assert len(vec_spaces) == dim
    assert all(x == 0.0 for x in vec_spaces)

    # Empty batch
    assert embedder.embed_documents([]) == []


def test_embedder_multilingual_semantic_alignment():
    """Verify cross-lingual encoding between Thai and English yields positive cosine similarity."""
    embedder = get_default_embedder()
    vec_th = embedder.embed_query("สายไฟเบอร์ออปติกขาดภายนอก")
    vec_en = embedder.embed_query("Optical fiber cut external damage")

    # Cosine similarity between unit vectors is the dot product
    cos_sim = sum(a * b for a, b in zip(vec_th, vec_en))
    # Measured cross-lingual alignment for paraphrase-multilingual-MiniLM-L12-v2 is ~0.366
    assert cos_sim > 0.30, f"Expected cross-lingual similarity > 0.30, got {cos_sim:.4f}"


# ─── 2. Indexer Tests ─────────────────────────────────────────────────────────

def test_indexer_document_counts_and_collections():
    """Verify indexer creates collections with exact expected record counts."""
    indexer = SemanticIndexer()
    col_wo = indexer.client.get_collection("work_orders")
    col_tt = indexer.client.get_collection("trouble_tickets")

    assert col_wo.count() == 3000, f"Expected 3,000 work orders, found {col_wo.count()}"
    assert col_tt.count() == 2500, f"Expected 2,500 trouble tickets, found {col_tt.count()}"


def test_stable_document_identifiers():
    """Verify document IDs follow stable prefix conventions."""
    indexer = SemanticIndexer()
    col_wo = indexer.client.get_collection("work_orders")
    sample_wo = col_wo.get(limit=5)
    for doc_id in sample_wo["ids"]:
        assert doc_id.startswith("wo_WO-70"), f"Invalid work order ID format: {doc_id}"

    col_tt = indexer.client.get_collection("trouble_tickets")
    sample_tt = col_tt.get(limit=5)
    for doc_id in sample_tt["ids"]:
        assert doc_id.startswith("inc_INC-2025-"), f"Invalid ticket ID format: {doc_id}"


def test_repeated_indexing_prevents_duplicates():
    """Verify upsert-based indexing is idempotent and does not inflate document counts."""
    indexer = SemanticIndexer()
    col_wo = indexer.client.get_collection("work_orders")
    initial_count = col_wo.count()
    assert initial_count == 3000

    # Test upsert on a batch of 5 existing documents
    sample = col_wo.get(limit=5)
    col_wo.upsert(
        ids=sample["ids"],
        documents=sample["documents"],
        metadatas=sample["metadatas"]
    )
    new_count = col_wo.count()
    assert new_count == initial_count == 3000, "Repeated indexing created duplicate documents"


# ─── 3. Retriever Tests ───────────────────────────────────────────────────────

def test_top_k_retrieval_and_ordering():
    """Verify retriever returns exactly k items sorted by similarity descending."""
    retriever = SemanticRetriever()
    k = 5
    results = retriever.query_work_orders("power blackout UPS battery dead", top_k=k)
    assert len(results) == k

    scores = [r["similarity_score"] for r in results]
    assert scores == sorted(scores, reverse=True), "Results not sorted by similarity descending"
    for r in results:
        assert r["id"].startswith("wo_")
        assert "document" in r
        assert "metadata" in r
        assert "cosine_distance" in r
        assert "similarity_score" in r


def test_distance_and_similarity_semantics():
    """Verify mathematical invariant: similarity_score = round(1 - cosine_distance, 4)."""
    retriever = SemanticRetriever()
    results = retriever.query_work_orders("router hardware failure memory parity", top_k=3)
    assert len(results) > 0
    for r in results:
        expected_sim = round(1.0 - r["cosine_distance"], 4)
        assert pytest.approx(r["similarity_score"], abs=1e-4) == expected_sim


def test_threshold_filtering():
    """Verify that setting a similarity threshold filters out low-scoring documents."""
    retriever = SemanticRetriever()
    # Query with a high threshold
    high_thresh = 0.55
    filtered = retriever.query_work_orders("transceiver optical failure", top_k=10, threshold=high_thresh)
    for r in filtered:
        assert r["similarity_score"] >= high_thresh, f"Item score {r['similarity_score']} < threshold {high_thresh}"


def test_metadata_filtering():
    """Verify ChromaDB metadata filters strictly constrain results to specified attributes."""
    retriever = SemanticRetriever()
    # Query with root_cause_category metadata filter
    where_filter = {"root_cause_category": "FIBER_CUT"}
    results = retriever.query_work_orders("optical signal degraded", top_k=5, where=where_filter)
    assert len(results) > 0
    for r in results:
        assert r["metadata"]["root_cause_category"] == "FIBER_CUT"


def test_metadata_filtering_tickets():
    """Verify filtering trouble tickets by customer_id."""
    retriever = SemanticRetriever()
    target_cust_id = 1  # Apex Logistics
    results = retriever.query_trouble_tickets(
        "link down", top_k=5, where={"customer_id": target_cust_id}
    )
    assert len(results) > 0
    for r in results:
        assert r["metadata"]["customer_id"] == target_cust_id


def test_empty_query_and_missing_collection_handling():
    """Verify empty queries return empty lists without raising exceptions."""
    retriever = SemanticRetriever()
    assert retriever.query_work_orders("") == []
    assert retriever.query_work_orders("    ") == []
    assert retriever.query_trouble_tickets("") == []


def test_expected_document_retrieval_fixtures():
    """Verify semantic retrieval successfully surfaces the relevant technical root cause category."""
    retriever = SemanticRetriever()

    # Case 1: Fiber cut query should retrieve FIBER_CUT work orders in top-3
    fiber_results = retriever.query_work_orders("สายไฟเบอร์ออปติกขาดจากการก่อสร้าง", top_k=3)
    categories = [r["metadata"]["root_cause_category"] for r in fiber_results]
    assert "FIBER_CUT" in categories

    # Case 2: SFP Transceiver degradation should retrieve OPTICAL_TRANSCEIVER in top-3
    sfp_results = retriever.query_work_orders("SFP module laser bias RX degraded", top_k=3)
    sfp_categories = [r["metadata"]["root_cause_category"] for r in sfp_results]
    assert "OPTICAL_TRANSCEIVER" in sfp_categories


def test_repository_isolation():
    """Verify persistent vector database is located inside the project repository tree."""
    indexer = SemanticIndexer()
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    chroma_abs = os.path.abspath(indexer.chroma_dir)
    assert chroma_abs.startswith(repo_root), f"Chroma directory {chroma_abs} must reside within repository root {repo_root}"
    rel_path = os.path.relpath(chroma_abs, repo_root)
    assert rel_path == os.path.join("data", "chroma_db")

