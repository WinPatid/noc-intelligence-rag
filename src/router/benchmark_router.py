"""
NOC Intelligence — Stage 4 Deterministic Router Benchmark & Evaluator
Evaluates route classification accuracy, entity extraction precision/recall,
per-route breakdown, clarification behavior, and dispatch latencies.
"""

import os
import json
import time
from typing import Dict, Any, List

from src.router.router import DeterministicRouter
from src.router.models import RouteType, IntentType

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_FIXTURES_PATH = os.path.join(ROOT_DIR, "data", "routing_evaluation_fixtures.json")


def run_routing_evaluation(fixtures_path: str = DEFAULT_FIXTURES_PATH):
    print("=" * 75)
    print("STAGE 4 DETERMINISTIC ROUTER EVALUATION BENCHMARK")
    print("=" * 75)

    with open(fixtures_path, "r", encoding="utf-8") as f:
        fixtures = json.load(f)

    test_cases = fixtures["test_cases"]
    total = len(test_cases)
    print(f"[*] Loaded {total} benchmark test cases from {fixtures_path}")

    router = DeterministicRouter()

    correct_route = 0
    correct_intent = 0
    correct_entities = 0
    entity_eval_count = 0

    false_sql_count = 0
    false_hybrid_count = 0

    route_stats = {
        rt.value: {"tp": 0, "fp": 0, "fn": 0, "total_expected": 0}
        for rt in RouteType
    }

    latencies_us = []

    print("\n" + "-" * 75)
    print(f"{'Case ID':<12} | {'Lang':<4} | {'Expected Route':<16} | {'Actual Route':<16} | {'Match'}")
    print("-" * 75)

    for tc in test_cases:
        cid = tc["case_id"]
        lang = tc["language"]
        query = tc["query"]
        exp_route = tc["expected_route"]
        exp_intent = tc["expected_intent"]

        route_stats[exp_route]["total_expected"] += 1

        t0 = time.perf_counter_ns()
        decision = router.classify(query)
        lat_us = (time.perf_counter_ns() - t0) / 1000.0
        latencies_us.append(lat_us)

        act_route = decision.route_type.value
        act_intent = decision.normalized_intent.value

        # Check Route Match
        is_route_match = (act_route == exp_route)
        if is_route_match:
            correct_route += 1
            route_stats[act_route]["tp"] += 1
        else:
            route_stats[exp_route]["fn"] += 1
            route_stats[act_route]["fp"] += 1
            if act_route == "SQL_LOOKUP":
                false_sql_count += 1
            if act_route == "HYBRID_SEARCH":
                false_hybrid_count += 1

        # Check Intent Match
        if act_intent == exp_intent:
            correct_intent += 1

        # Check Entity Extraction if specified in fixture
        ent_matched = True
        has_entity_check = False
        ents = decision.extracted_entities

        if "expected_ticket_id" in tc:
            has_entity_check = True
            if tc["expected_ticket_id"] not in ents.ticket_ids:
                ent_matched = False
        if "expected_wo_id" in tc:
            has_entity_check = True
            if tc["expected_wo_id"] not in ents.work_order_ids:
                ent_matched = False
        if "expected_circuit_ref" in tc:
            has_entity_check = True
            if tc["expected_circuit_ref"] not in ents.circuit_refs:
                ent_matched = False
        if "expected_customer" in tc:
            has_entity_check = True
            matched_cname = (ents.matched_customer or {}).get("company_name_en")
            if tc["expected_customer"].lower() not in (matched_cname or "").lower():
                ent_matched = False
        if "expected_status" in tc:
            has_entity_check = True
            if ents.status_filter != tc["expected_status"]:
                ent_matched = False

        if has_entity_check:
            entity_eval_count += 1
            if ent_matched:
                correct_entities += 1

        status_icon = "✅" if (is_route_match and (act_intent == exp_intent)) else "❌"
        print(f"{cid:<12} | {lang:<4} | {exp_route:<16} | {act_route:<16} | {status_icon} {act_intent}")

    # Summary Metrics
    route_acc = (correct_route / total) * 100.0
    intent_acc = (correct_intent / total) * 100.0
    entity_acc = (correct_entities / entity_eval_count * 100.0) if entity_eval_count > 0 else 100.0
    avg_lat_us = sum(latencies_us) / len(latencies_us)

    print("\n" + "=" * 75)
    print("METRICS SUMMARY")
    print("=" * 75)
    print(f"  Total Evaluated Cases       : {total}")
    print(f"  Route Classification Acc.   : {correct_route}/{total} ({route_acc:.2f}%)")
    print(f"  Intent Classification Acc.  : {correct_intent}/{total} ({intent_acc:.2f}%)")
    print(f"  Entity Extraction Acc.      : {correct_entities}/{entity_eval_count} ({entity_acc:.2f}%)")
    print(f"  False SQL Routing Cases     : {false_sql_count}")
    print(f"  False Hybrid Routing Cases  : {false_hybrid_count}")
    print(f"  Mean Classification Latency : {avg_lat_us:.2f} µs ({avg_lat_us/1000.0:.3f} ms)")

    print("\n" + "-" * 75)
    print(f"{'Route Type':<18} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support'}")
    print("-" * 75)

    for rt, stats in route_stats.items():
        tp = stats["tp"]
        fp = stats["fp"]
        fn = stats["fn"]
        supp = stats["total_expected"]

        prec = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
        rec = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        print(f"{rt:<18} | {prec:8.2f}% | {rec:8.2f}% | {f1:8.2f}% | {supp}")

    # Dispatch Verification Test
    print("\n" + "=" * 75)
    print("SAMPLE END-TO-END DISPATCH VERIFICATION")
    print("=" * 75)

    sample_queries = [
        ("SQL Direct Ticket", "Ticket INC-2025-00001 สถานะเป็นอะไร"),
        ("SQL Bridge Traversal", "ตั๋ว INC-2025-00042 เชื่อมโยงกับใบงานช่างใบไหนบ้าง"),
        ("Semantic Resolution", "สายเคเบิลใยแก้วนำแสงขาดเนื่องจากการก่อสร้างขุดถนน ซ่อมแซมอย่างไร"),
        ("Hybrid Customer Query", "ค้นหาปัญหา fiber degradation ของบริษัท Apex Logistics"),
        ("Clarification Query", "เน็ตช้ามาก ใช้งานไม่ได้ ช่วยดูให้หน่อยครับ"),
        ("Unsupported Query", "SELECT * FROM trouble_tickets; DROP TABLE customers; --")
    ]

    for label, q in sample_queries:
        res = router.dispatch(q, top_k=3)
        print(f"\n[{label}] Query: \"{q}\"")
        print(f"  -> Route: {res.decision.route_type.value} | Intent: {res.decision.normalized_intent.value}")
        print(f"  -> Success: {res.success} | Latency: {res.execution_time_ms:.2f} ms")
        if res.decision.clarification_required:
            print(f"  -> Clarification Prompt: {res.decision.clarification_question}")
        elif res.retrieval_results:
            results_preview = str(res.retrieval_results)[:120] + "..."
            print(f"  -> Results: {results_preview}")


if __name__ == "__main__":
    run_routing_evaluation()
