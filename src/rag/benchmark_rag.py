"""
NOC Intelligence — Stage 3 Semantic Retrieval & Embedder Benchmark
Measures embedding inference latency, vector query response times,
cross-lingual semantic precision, and similarity score distributions.
"""

import os
import sys

# Suppress HuggingFace and ChromaDB telemetry messages
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "false"

import time
import statistics
from typing import List, Dict, Any

from src.rag.embedder import get_default_embedder
from src.rag.retriever import SemanticRetriever
from src.rag.indexer import DEFAULT_CHROMA_DIR


def get_dir_size_mb(path: str) -> float:
    """Calculates directory size in megabytes."""
    total_bytes = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if not os.path.islink(fp):
                total_bytes += os.path.getsize(fp)
    return total_bytes / (1024 * 1024)


def benchmark_embedding_latency(embedder, iterations: int = 50):
    print("=" * 75)
    print("1. EMBEDDING INFERENCE LATENCY BENCHMARK")
    print("=" * 75)

    test_queries = [
        "สายไฟเบอร์ออปติกขาดภายนอกอาคาร",
        "Optical fiber cut external construction damage",
        "ไฟฟ้าดับ PEA blackout node down",
        "SFP+ transceiver module faulty replace hardware",
        "BGP routing flap session drop high packet loss"
    ]

    # Warmup
    for q in test_queries:
        embedder.embed_query(q)

    latencies_ms = []
    for _ in range(iterations):
        for q in test_queries:
            t0 = time.perf_counter_ns()
            embedder.embed_query(q)
            latencies_ms.append((time.perf_counter_ns() - t0) / 1_000_000)

    latencies_ms.sort()
    mean_lat = statistics.mean(latencies_ms)
    median_lat = statistics.median(latencies_ms)
    p95_lat = latencies_ms[int(len(latencies_ms) * 0.95)]
    min_lat = min(latencies_ms)
    max_lat = max(latencies_ms)

    print(f"  Model Name           : {embedder.model_name}")
    print(f"  Vector Dimension     : {embedder.get_dimension()} dimensions")
    print(f"  Total Inferences     : {len(latencies_ms)} single query calls")
    print(f"  Latency (Min)        : {min_lat:6.2f} ms")
    print(f"  Latency (Mean)       : {mean_lat:6.2f} ms")
    print(f"  Latency (Median/p50) : {median_lat:6.2f} ms")
    print(f"  Latency (p95)        : {p95_lat:6.2f} ms")
    print(f"  Latency (Max)        : {max_lat:6.2f} ms")

    # Batch latency test
    batch_16 = test_queries * 3 + [test_queries[0]]  # 16 items
    batch_t0 = time.perf_counter()
    embedder.embed_documents(batch_16, batch_size=16)
    batch_16_time = (time.perf_counter() - batch_t0) * 1000
    print(f"  Batch 16 Throughput  : {batch_16_time:6.2f} ms total ({batch_16_time/16:5.2f} ms/doc)")


def benchmark_retrieval_latency(retriever: SemanticRetriever, iterations: int = 30):
    print("\n" + "=" * 75)
    print("2. VECTOR SEARCH END-TO-END LATENCY BENCHMARK")
    print("=" * 75)

    scenarios = [
        ("query_work_orders (Thai)", "สายเคเบิลใยแก้วนำแสงขาดจากการขุดถนน", "wo"),
        ("query_work_orders (English)", "Optical fiber cable cut due to civil road works", "wo"),
        ("query_trouble_tickets (Thai)", "ไฟดับ หม้อแปลงระเบิด ไฟฟ้าส่วนภูมิภาค", "tt"),
        ("query_trouble_tickets (English)", "Commercial electrical power outage substation", "tt"),
        ("query_all (Unified)", "SFP optical transceiver module error", "all"),
    ]

    for label, query, target in scenarios:
        # Warmup
        if target == "wo":
            retriever.query_work_orders(query, top_k=5)
        elif target == "tt":
            retriever.query_trouble_tickets(query, top_k=5)
        else:
            retriever.query_all(query, top_k=5)

        lats = []
        for _ in range(iterations):
            t0 = time.perf_counter_ns()
            if target == "wo":
                retriever.query_work_orders(query, top_k=5)
            elif target == "tt":
                retriever.query_trouble_tickets(query, top_k=5)
            else:
                retriever.query_all(query, top_k=5)
            lats.append((time.perf_counter_ns() - t0) / 1_000_000)

        mean_l = statistics.mean(lats)
        p95_l = sorted(lats)[int(len(lats) * 0.95)]
        print(f"  ⚡ {label:<38} : Mean {mean_l:6.2f} ms | p95 {p95_l:6.2f} ms")


def evaluate_semantic_scenarios(retriever: SemanticRetriever):
    print("\n" + "=" * 75)
    print("3. REPRESENTATIVE SEMANTIC RETRIEVAL QUALITY EVALUATION")
    print("=" * 75)

    test_cases = [
        {
            "scenario": "Fiber Optic Cable Cut",
            "expected_category": "FIBER_CUT",
            "queries": [
                ("TH", "สายเคเบิลใยแก้วนำแสงขาดเนื่องจากการก่อสร้างขุดถนน"),
                ("EN", "Optical fiber cable cut due to road excavation work")
            ]
        },
        {
            "scenario": "Commercial Power Outage / Blackout",
            "expected_category": "COMMERCIAL_POWER",
            "queries": [
                ("TH", "ไฟฟ้าดับเป็นวงกว้าง การไฟฟ้าส่วนภูมิภาค PEA ดับไฟ แบตเตอรี่ UPS หมด"),
                ("EN", "Regional commercial power blackout building backup UPS batteries depleted")
            ]
        },
        {
            "scenario": "Optical Transceiver Failure",
            "expected_category": "OPTICAL_TRANSCEIVER",
            "queries": [
                ("TH", "SFP 10G transceiver error RX loss of signal เปลี่ยนโมดูลหัวรับส่งแสง"),
                ("EN", "SFP+ 10G optical transceiver module faulty low RX power laser degradation")
            ]
        },
        {
            "scenario": "Router Configuration & Policy Error",
            "expected_category": "CONFIGURATION_ERROR",
            "queries": [
                ("TH", "การกำหนดค่าคอนฟิกผิดพลาด MTU mismatch ปรับแก้ค่า route-policy filter"),
                ("EN", "Mismatched MTU configuration and route-policy filter dropping Jumbo frames")
            ]
        },
        {
            "scenario": "Hardware Route Processor Failure",
            "expected_category": "HARDWARE_FAILURE",
            "queries": [
                ("TH", "การ์ดประมวลผลชำรุด memory parity fault สลับการ์ดควบคุมสำรอง microcode"),
                ("EN", "Main route processor board memory parity fault device reload hot-swapped backup RP")
            ]
        },
        {
            "scenario": "Environmental Water Seepage",
            "expected_category": "ENVIRONMENTAL",
            "queries": [
                ("TH", "น้ำฝนซึมเข้าตู้พักสายใต้ดิน คอนเน็กเตอร์ชื้น สูบน้ำระบายความชื้น ซีลกันน้ำ"),
                ("EN", "Water seepage into underground splice enclosure drained manhole heat-shrink resealed")
            ]
        }
    ]

    for tc in test_cases:
        print(f"\n[Scenario: {tc['scenario']}] Expected: {tc['expected_category']}")
        for lang, query in tc["queries"]:
            results = retriever.query_work_orders(query, top_k=5)
            if not results:
                print(f"  [{lang}] Query: \"{query}\" -> NO RESULTS RETURNED")
                continue

            top_hit = results[0]
            top_meta = top_hit["metadata"]
            top_cat = top_meta.get("root_cause_category")
            top_score = top_hit["similarity_score"]
            top_dist = top_hit["cosine_distance"]

            cat_matches = [
                r for r in results
                if r["metadata"].get("root_cause_category") == tc["expected_category"]
            ]
            top5_hit_rate = len(cat_matches) / len(results) * 100.0

            match_icon = "✅" if top_cat == tc["expected_category"] else "❌"
            print(
                f"  [{lang}] {match_icon} Top-1 Match: {top_cat} (score: {top_score:.4f}, dist: {top_dist:.4f}) | "
                f"Top-5 Recall: {len(cat_matches)}/{len(results)} ({top5_hit_rate:.0f}%)"
            )
            print(f"       WO ID: {top_meta.get('work_order_id')} | Assigned: {top_meta.get('assigned_team')}")


def inspect_storage_footprint(retriever: SemanticRetriever):
    print("\n" + "=" * 75)
    print("4. CHROMADB VECTOR STORE STORAGE & INTEGRITY METRICS")
    print("=" * 75)

    wo_col = retriever._get_collection("work_orders")
    tt_col = retriever._get_collection("trouble_tickets")

    wo_count = wo_col.count() if wo_col else 0
    tt_count = tt_col.count() if tt_col else 0
    dir_size_mb = get_dir_size_mb(DEFAULT_CHROMA_DIR)

    print(f"  Storage Directory       : {DEFAULT_CHROMA_DIR}")
    print(f"  Disk Space Utilized     : {dir_size_mb:.2f} MB")
    print(f"  'work_orders' Collection: {wo_count:,} indexed records")
    print(f"  'trouble_tickets' Col.  : {tt_count:,} indexed records")
    print(f"  Total Indexed Vectors   : {wo_count + tt_count:,} vectors")
    print(f"  HNSW Metric Space       : Cosine (L2-normalized vectors)")
    print(f"  Embedder Model Name     : {retriever.embedder.model_name}")
    print(f"  Embedding Dimension     : {retriever.embedder.get_dimension()} dimensions")


def run_all_benchmarks():
    embedder = get_default_embedder()
    retriever = SemanticRetriever(embedder=embedder)

    benchmark_embedding_latency(embedder)
    benchmark_retrieval_latency(retriever)
    evaluate_semantic_scenarios(retriever)
    inspect_storage_footprint(retriever)


if __name__ == "__main__":
    run_all_benchmarks()
