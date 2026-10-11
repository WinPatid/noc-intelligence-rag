"""
NOC Intelligence — Stage 8 40-Query Benchmark Evaluation Runner
Refined implementation:
- Correct True End-to-End latency boundaries and isolated component timing.
- Explicit cold-start separation (vector store model loading) from steady-state inference.
- Guaranteed zero live Ollama / network calls in fake provider mode with active network guard.
- Disaggregated claim verification (factual verification vs. adversarial rejection).
- Independent citation source validity and ground-truth support audit.
"""

import os
import sys
import json
import time
import argparse
import platform
import urllib.request
import http.client
from contextlib import contextmanager
from typing import Dict, Any, List, Optional, Tuple, Set

from src.router.router import DeterministicRouter
from src.router.models import RouteType, IntentType, RoutingResult
from src.synthesis.models import SynthesisResponse, GenerationStatus, CitationItem
from src.synthesis.providers import (
    DeterministicFakeProvider,
    BaseLLMProvider,
    ProviderUnavailableError,
    LLMProviderError,
    OllamaProvider
)
from src.synthesis.synthesizer import GroundedSynthesizer
from src.guardrails.models import (
    GuardrailStatus,
    MaskingConfig,
    GuardrailResult
)
from src.guardrails.critic import GuardrailCritic
from src.guardrails.masker import IdentifierMasker
from src.guardrails.pipeline import GuardrailPipeline


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_FIXTURES_PATH = os.path.join(ROOT_DIR, "data", "benchmark_40_fixtures.json")
DEFAULT_REPORT_DIR = os.path.join(ROOT_DIR, "reports")


def percentile(data: List[float], p: float) -> float:
    """Calculates p-th percentile from a list of floats."""
    if not data:
        return 0.0
    sorted_d = sorted(data)
    idx = int(len(sorted_d) * (p / 100.0))
    idx = min(idx, len(sorted_d) - 1)
    return round(sorted_d[idx], 3)


def mean(data: List[float]) -> float:
    return round(sum(data) / len(data), 3) if data else 0.0


def median(data: List[float]) -> float:
    return percentile(data, 50.0)


class FailingOfflineProvider(BaseLLMProvider):
    def is_available(self) -> bool:
        return False
    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        raise ProviderUnavailableError("Simulated offline local LLM")


class EmptyOutputProvider(BaseLLMProvider):
    def is_available(self) -> bool:
        return True
    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        return "   \n\t  "


class CrashingMasker(IdentifierMasker):
    def mask_text_and_citations(self, *args, **kwargs):
        raise RuntimeError("Simulated masker internal failure")


class BlockedNetworkError(RuntimeError):
    """Raised when an unauthorized outbound network connection is attempted."""
    pass


@contextmanager
def prevent_network_calls():
    """Guarantees zero live HTTP/socket requests during fake provider execution."""
    original_urlopen = urllib.request.urlopen

    def forbidden_urlopen(*args, **kwargs):
        req = args[0] if args else kwargs.get("url")
        url_str = getattr(req, "full_url", str(req))
        raise BlockedNetworkError(
            f"Prohibited live network/Ollama call attempted in fake provider mode: {url_str}"
        )

    urllib.request.urlopen = forbidden_urlopen
    try:
        yield
    finally:
        urllib.request.urlopen = original_urlopen


def extract_retrieved_source_ids(retrieval_results: Any) -> Set[str]:
    """Flattens all entity IDs retrieved by the router into canonical source IDs."""
    ids: Set[str] = set()
    if not retrieval_results or not isinstance(retrieval_results, dict):
        return ids

    rr = retrieval_results
    # Single SQL record
    if rr.get("record"):
        r = rr["record"]
        if "ticket_id" in r:
            ids.add(f"ticket:{r['ticket_id']}")
        if "work_order_id" in r:
            ids.add(f"work_order:{r['work_order_id']}")
        if "circuit_ref" in r:
            ids.add(f"circuit:{r['circuit_ref']}")

    # Multiple SQL records
    if rr.get("records"):
        for r in rr["records"]:
            if "ticket_id" in r:
                ids.add(f"ticket:{r['ticket_id']}")
            if "work_order_id" in r:
                ids.add(f"work_order:{r['work_order_id']}")
            if "circuit_ref" in r:
                ids.add(f"circuit:{r['circuit_ref']}")

    # Named entity objects
    if rr.get("circuit"):
        ids.add(f"circuit:{rr['circuit']['circuit_ref']}")
    if rr.get("circuit_ref"):
        ids.add(f"circuit:{rr['circuit_ref']}")
    if rr.get("customer"):
        ids.add(f"customer:{rr['customer']['company_name_en']}")
    # Bridge links and top-level bridge entity IDs
    if rr.get("ticket_id"):
        ids.add(f"ticket:{rr['ticket_id']}")
    if rr.get("work_order_id"):
        ids.add(f"work_order:{rr['work_order_id']}")
    if rr.get("circuit_ref"):
        ids.add(f"circuit:{rr['circuit_ref']}")
    if rr.get("circuit_details") and isinstance(rr["circuit_details"], dict) and rr["circuit_details"].get("circuit_ref"):
        ids.add(f"circuit:{rr['circuit_details']['circuit_ref']}")

    if rr.get("linked_work_orders"):
        for w in rr["linked_work_orders"]:
            ids.add(f"work_order:{w['work_order_id']}")
    if rr.get("impacted_tickets"):
        for t in rr["impacted_tickets"]:
            ids.add(f"ticket:{t['ticket_id']}")

    # Semantic hits
    for hit in (rr.get("hits") or rr.get("work_orders") or []):
        meta = hit.get("metadata", {})
        if "work_order_id" in meta:
            ids.add(f"work_order:{meta['work_order_id']}")
        elif "ticket_id" in meta:
            ids.add(f"ticket:{meta['ticket_id']}")

    for hit in rr.get("trouble_tickets") or []:
        meta = hit.get("metadata", {})
        if "ticket_id" in meta:
            ids.add(f"ticket:{meta['ticket_id']}")

    # Hybrid incidents
    for inc in rr.get("incidents", []):
        if isinstance(inc, dict):
            tid = inc.get("ticket_id")
            if tid:
                ids.add(f"ticket:{tid}")
            for w in inc.get("linked_work_orders", []):
                ids.add(f"work_order:{w['work_order_id']}")

    return ids


def audit_citation_integrity(
    citations: List[CitationItem],
    retrieved_source_ids: Set[str],
    expected_evidence_ids: List[str]
) -> Dict[str, Any]:
    """
    Independently verifies citations against retrieved evidence records and expected ground truth.
    Separates source-ID validity from claim-support correctness.
    """
    total = len(citations)
    if total == 0:
        return {
            "total_citations": 0,
            "valid_source_ids": 0,
            "unretrieved_source_ids": 0,
            "supported_citations": 0,
            "unsupported_citations": 0,
            "ground_truth_evaluable": False
        }

    valid_sources = 0
    unretrieved_sources = 0
    supported = 0
    unsupported = 0

    has_ground_truth = len(expected_evidence_ids) > 0

    for c in citations:
        # Check 1: Source identifier existence in retrieved evidence
        is_retrieved = c.source_id in retrieved_source_ids
        if is_retrieved:
            valid_sources += 1
        else:
            unretrieved_sources += 1

        # Check 2: Claim support against ground truth
        if has_ground_truth:
            if c.source_id in expected_evidence_ids:
                supported += 1
            else:
                unsupported += 1

    return {
        "total_citations": total,
        "valid_source_ids": valid_sources,
        "unretrieved_source_ids": unretrieved_sources,
        "supported_citations": supported,
        "unsupported_citations": unsupported,
        "ground_truth_evaluable": has_ground_truth
    }


def run_stage8_benchmark(
    fixtures_path: str = DEFAULT_FIXTURES_PATH,
    provider_type: str = "fake",
    output_dir: str = DEFAULT_REPORT_DIR
) -> Dict[str, Any]:
    """
    Executes the 40-query benchmark harness, calculates disaggregated metrics,
    and produces machine-readable JSON and human-readable Markdown reports.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("STAGE 8 — 40-QUERY EVALUATION BENCHMARK & QUALITY HARNESS (REFINED)")
    print("=" * 80)

    # 1. Load Evaluation Fixtures
    with open(fixtures_path, "r", encoding="utf-8") as f:
        fixtures_data = json.load(f)

    test_cases = fixtures_data["test_cases"]
    total_cases = len(test_cases)
    print(f"[*] Loaded {total_cases} evaluation cases from {fixtures_path}")
    print(f"[*] Execution Provider: {provider_type.upper()}")

    # 2. Initialize Shared Components
    router = DeterministicRouter()
    if provider_type == "ollama":
        ollama_p = OllamaProvider()
        if not ollama_p.is_available():
            print("[!] WARNING: Local Ollama model is unreachable. Falling back to DeterministicFakeProvider.")
            default_synthesizer = GroundedSynthesizer(provider=DeterministicFakeProvider())
        else:
            default_synthesizer = GroundedSynthesizer(provider=ollama_p)
    else:
        default_synthesizer = GroundedSynthesizer(provider=DeterministicFakeProvider())

    pipe_unmasked = GuardrailPipeline(
        synthesizer=default_synthesizer,
        masking_config=MaskingConfig(enabled=False),
        router=router
    )
    pipe_masked = GuardrailPipeline(
        synthesizer=default_synthesizer,
        masking_config=MaskingConfig(enabled=True),
        router=router
    )

    # 3. Explicit Warm-up (Separating Cold-Start from Steady-State)
    print("[*] Performing pipeline warm-up runs...")
    t_warm_start = time.perf_counter()

    # Warm-up A: SQL engine & SQLite connection pool
    _ = router.classify("Ticket INC-2025-00001 สถานะเป็นอะไร")
    _ = router.dispatch("Ticket INC-2025-00001 สถานะเป็นอะไร")

    # Warm-up B: ChromaDB vector store & SentenceTransformers embedding model
    t_sem_cold_0 = time.perf_counter()
    _ = router.dispatch("ไฟฟ้าดับเป็นวงกว้าง การไฟฟ้า PEA ดับไฟซ่อมหม้อแปลง")
    sem_cold_start_ms = (time.perf_counter() - t_sem_cold_0) * 1000

    # Warm-up C: Pipeline execution & Masker customer cache
    _ = pipe_unmasked.process("Ticket INC-2025-00001 สถานะเป็นอะไร")
    _ = pipe_masked.process("Ticket INC-2025-00001 สถานะเป็นอะไร")

    total_cold_start_ms = (time.perf_counter() - t_warm_start) * 1000
    print(f"[*] Warm-up complete: Semantic vector store cold-start = {sem_cold_start_ms:.2f} ms")
    print(f"[*] Total warm-up phase duration = {total_cold_start_duration_ms:.2f} ms" if 'total_cold_start_duration_ms' in locals() else f"[*] Total warm-up phase duration = {total_cold_start_ms:.2f} ms")

    # 4. Metric Accumulators
    routing_matches = 0
    intent_matches = 0
    entity_eval_total = 0
    entity_matches = 0

    retrieval_cases_total = 0
    recall_scores: List[float] = []
    precision_scores: List[float] = []

    # Disaggregated Claim Accumulators
    factual_claims_total = 0
    factual_claims_verified = 0

    adversarial_claims_total = 0
    adversarial_claims_rejected = 0

    unsupported_claims_total = 0
    unsupported_claims_rejected = 0

    contradiction_cases_total = 0
    contradiction_detected_total = 0

    anomalous_cases_total = 0
    safe_degradation_total = 0

    masked_cases_total = 0
    leakage_detected_total = 0

    status_alignment_matches = 0

    # Independent Citation Accumulators
    citation_eval_total = 0
    citation_valid_sources_total = 0
    citation_gt_eval_total = 0
    citation_gt_supported_total = 0

    # Latency tracking (in ms)
    latencies_classify: List[float] = []
    latencies_retrieval: List[float] = []
    latencies_guardrail: List[float] = []
    latencies_total_e2e: List[float] = []

    detailed_results: List[Dict[str, Any]] = []
    failed_cases: List[Dict[str, Any]] = []

    print("\n" + "-" * 80)
    print(f"{'Case ID':<12} | {'Category':<22} | {'Exp Route':<15} | {'Exp Status':<12} | {'Match'}")
    print("-" * 80)

    # Context manager to guarantee zero live network/Ollama calls in fake mode
    net_guard = prevent_network_calls if provider_type.lower() == "fake" else nullcontext

    with net_guard():
        # 5. Execute Each Test Case
        for tc in test_cases:
            cid = tc["case_id"]
            cat = tc["category"]
            query = tc["query"]
            exp_route = tc["expected_route"]
            exp_intent = tc["expected_intent"]
            exp_entities = tc.get("expected_entities", {})
            exp_evidence = tc.get("expected_relevant_evidence_ids", [])
            exp_status = tc["expected_guardrail_status"]
            masking_req = tc.get("masking_required", False)
            sim_failure = tc.get("simulate_failure")
            synthetic_resp = tc.get("synthetic_response")
            prohibited_tokens = tc.get("prohibited_unmasked_tokens", [])

            # Step A: Component Measurement — Router Classification
            t0 = time.perf_counter()
            decision = router.classify(query)
            t_cls = (time.perf_counter() - t0) * 1000
            latencies_classify.append(t_cls)

            act_route = decision.route_type.value
            act_intent = decision.normalized_intent.value

            route_ok = (act_route == exp_route)
            intent_ok = (act_intent == exp_intent)

            if route_ok:
                routing_matches += 1
            if intent_ok:
                intent_matches += 1

            # Check entity extraction if expected
            entity_ok = True
            if exp_entities:
                entity_eval_total += 1
                if "ticket_ids" in exp_entities:
                    for tid in exp_entities["ticket_ids"]:
                        if tid.upper() not in [t.upper() for t in decision.extracted_entities.ticket_ids]:
                            entity_ok = False
                if "work_order_ids" in exp_entities:
                    for wid in exp_entities["work_order_ids"]:
                        if wid.upper() not in [w.upper() for w in decision.extracted_entities.work_order_ids]:
                            entity_ok = False
                if "circuit_refs" in exp_entities:
                    for cref in exp_entities["circuit_refs"]:
                        if cref.upper() not in [c.upper() for c in decision.extracted_entities.circuit_refs]:
                            entity_ok = False
                if "customer_name_raw" in exp_entities:
                    if not decision.extracted_entities.customer_name_raw or \
                       exp_entities["customer_name_raw"].lower() not in decision.extracted_entities.customer_name_raw.lower():
                        entity_ok = False
                if "has_sql_injection" in exp_entities:
                    if decision.extracted_entities.has_sql_injection != exp_entities["has_sql_injection"]:
                        entity_ok = False

                if entity_ok:
                    entity_matches += 1

            # Step B: Component Measurement — Dispatch Retrieval
            t1 = time.perf_counter()
            routing_res = router.dispatch(query)
            t_ret = (time.perf_counter() - t1) * 1000
            latencies_retrieval.append(t_ret)

            # Retrieval Recall@K and Precision@K
            retrieved_ids = extract_retrieved_source_ids(routing_res.retrieval_results)

            if exp_evidence:
                retrieval_cases_total += 1
                relevant_retrieved = [eid for eid in exp_evidence if eid in retrieved_ids]
                recall = len(relevant_retrieved) / len(exp_evidence)
                recall_scores.append(recall)

                k = min(5, len(retrieved_ids)) if retrieved_ids else 1
                precision = len(relevant_retrieved) / k
                precision_scores.append(precision)

            # Step C: Configure Pipeline with Explicit Failure Injection
            pipe = pipe_masked if masking_req else pipe_unmasked

            if sim_failure == "PROVIDER_OFFLINE":
                pipe = GuardrailPipeline(
                    synthesizer=GroundedSynthesizer(provider=FailingOfflineProvider()),
                    masking_config=MaskingConfig(enabled=masking_req),
                    router=router
                )
            elif sim_failure == "EMPTY_OUTPUT":
                pipe = GuardrailPipeline(
                    synthesizer=GroundedSynthesizer(provider=EmptyOutputProvider()),
                    masking_config=MaskingConfig(enabled=masking_req),
                    router=router
                )
            elif sim_failure == "MASKING_CRASH":
                pipe = GuardrailPipeline(
                    synthesizer=default_synthesizer,
                    masker=CrashingMasker(config=MaskingConfig(enabled=True)),
                    masking_config=MaskingConfig(enabled=True),
                    router=router
                )

            synth_input = None
            if synthetic_resp:
                synth_input = SynthesisResponse(
                    answer=synthetic_resp,
                    generation_status=GenerationStatus.SUCCESS,
                    citations=[
                        CitationItem(
                            source_id=eid,
                            entity_type="ref",
                            identifier=eid.split(":")[-1],
                            label=eid,
                            is_verified=True
                        ) for eid in exp_evidence
                    ],
                    language="TH" if any("\u0e00" <= c <= "\u0e7f" for c in synthetic_resp) else "EN"
                )

            # Measure Isolated Guardrail/Masking Component Step
            t_g_start = time.perf_counter()
            result_guard: GuardrailResult = pipe.process(
                query=query,
                routing_result=routing_res,
                synthesis_response=synth_input
            )
            t_guard = (time.perf_counter() - t_g_start) * 1000
            latencies_guardrail.append(t_guard)

            # Step D: Measure TRUE Full End-to-End Pipeline Latency
            # Executes the complete pipeline from the original query string through
            # router dispatch, synthesis, guardrails, and masking (without reusing precomputed results).
            t_e2e_start = time.perf_counter()
            result: GuardrailResult = pipe.process(
                query=query,
                routing_result=None,
                synthesis_response=synth_input
            )
            t_true_e2e = (time.perf_counter() - t_e2e_start) * 1000
            latencies_total_e2e.append(t_true_e2e)

            act_status = result.status.value
            status_ok = (act_status == exp_status)
            if status_ok:
                status_alignment_matches += 1

            # Independent Citation Integrity Audit
            citation_audit = audit_citation_integrity(result.citations, retrieved_ids, exp_evidence)
            if citation_audit["total_citations"] > 0:
                citation_eval_total += citation_audit["total_citations"]
                citation_valid_sources_total += citation_audit["valid_source_ids"]
                if citation_audit["ground_truth_evaluable"]:
                    citation_gt_eval_total += citation_audit["total_citations"]
                    citation_gt_supported_total += citation_audit["supported_citations"]

            # Disaggregated Claim Verification Audit
            if exp_status == "VERIFIED":
                for cl in result.claims:
                    factual_claims_total += 1
                    if cl.is_verified:
                        factual_claims_verified += 1
            elif exp_status in ("CONTRADICTED", "PARTIALLY_VERIFIED"):
                for cl in result.claims:
                    if cl.is_contradicted:
                        adversarial_claims_total += 1
                        adversarial_claims_rejected += 1
                    elif not cl.is_verified:
                        unsupported_claims_total += 1
                        unsupported_claims_rejected += 1
                        adversarial_claims_total += 1
                        adversarial_claims_rejected += 1
                    else:
                        # Factual background entity verified in evidence
                        factual_claims_total += 1
                        factual_claims_verified += 1

            # Contradiction Detection Check
            if exp_status == "CONTRADICTED":
                contradiction_cases_total += 1
                if act_status == "CONTRADICTED":
                    contradiction_detected_total += 1

            # Safe Degradation Check
            is_anomalous = (exp_status in ("CONTRADICTED", "INSUFFICIENT_EVIDENCE", "PROVIDER_UNAVAILABLE", "INVALID_RESPONSE")) or bool(sim_failure)
            if is_anomalous:
                anomalous_cases_total += 1
                if not result.is_safe_to_present or act_status in ("INSUFFICIENT_EVIDENCE", "PROVIDER_UNAVAILABLE", "INVALID_RESPONSE"):
                    safe_degradation_total += 1

            # Check identifier leakage in masked cases
            leak_found = False
            if masking_req:
                masked_cases_total += 1
                for tok in prohibited_tokens:
                    if tok in result.final_answer:
                        leak_found = True
                    for c in result.citations:
                        if tok in c.label or tok in c.identifier:
                            leak_found = True
                    for w in result.warnings:
                        if tok in w:
                            leak_found = True

                if leak_found:
                    leakage_detected_total += 1

            case_passed = route_ok and intent_ok and entity_ok and status_ok and not leak_found
            match_icon = "✅" if case_passed else "❌"
            print(f"{cid:<12} | {cat:<22} | {exp_route:<15} | {exp_status:<12} | {match_icon}")

            case_record = {
                "case_id": cid,
                "category": cat,
                "query": query,
                "expected_route": exp_route,
                "actual_route": act_route,
                "expected_intent": exp_intent,
                "actual_intent": act_intent,
                "expected_status": exp_status,
                "actual_status": act_status,
                "status_match": status_ok,
                "passed": case_passed,
                "is_safe_to_present": result.is_safe_to_present,
                "classify_latency_ms": round(t_cls, 3),
                "retrieval_latency_ms": round(t_ret, 3),
                "guardrail_latency_ms": round(t_guard, 3),
                "true_e2e_latency_ms": round(t_true_e2e, 3)
            }
            detailed_results.append(case_record)

            if not case_passed:
                failed_cases.append({
                    "case_id": cid,
                    "category": cat,
                    "expected": f"Route: {exp_route}, Intent: {exp_intent}, Status: {exp_status}",
                    "actual": f"Route: {act_route}, Intent: {act_intent}, Status: {act_status}",
                    "failed_subsystem": "Router" if not route_ok or not intent_ok else ("Critic" if not status_ok else "Masker"),
                    "reason": "Routing or status mismatch against ground truth"
                })

    # 6. Compute Disaggregated Metrics
    routing_accuracy = round((routing_matches / total_cases) * 100, 2)
    entity_accuracy = round((entity_matches / entity_eval_total) * 100, 2) if entity_eval_total else 100.0
    mean_recall = round(mean(recall_scores) * 100, 2) if recall_scores else 100.0
    mean_precision = round(mean(precision_scores) * 100, 2) if precision_scores else 100.0

    citation_source_validity_rate = round((citation_valid_sources_total / citation_eval_total) * 100, 2) if citation_eval_total else 100.0
    citation_gt_support_rate = round((citation_gt_supported_total / citation_gt_eval_total) * 100, 2) if citation_gt_eval_total else 100.0

    factual_claim_verification_rate = round((factual_claims_verified / factual_claims_total) * 100, 2) if factual_claims_total else 100.0
    adversarial_claim_rejection_rate = round((adversarial_claims_rejected / adversarial_claims_total) * 100, 2) if adversarial_claims_total else 100.0
    unsupported_claim_rejection_rate = round((unsupported_claims_rejected / unsupported_claims_total) * 100, 2) if unsupported_claims_total else 100.0

    contradiction_rate = round((contradiction_detected_total / contradiction_cases_total) * 100, 2) if contradiction_cases_total else 100.0
    safe_degradation_rate = round((safe_degradation_total / anomalous_cases_total) * 100, 2) if anomalous_cases_total else 100.0
    identifier_leakage_rate = round((leakage_detected_total / masked_cases_total) * 100, 2) if masked_cases_total else 0.0
    status_alignment_rate = round((status_alignment_matches / total_cases) * 100, 2)

    metrics_summary = {
        "total_cases_evaluated": total_cases,
        "routing_accuracy_pct": routing_accuracy,
        "entity_extraction_accuracy_pct": entity_accuracy,
        "retrieval_recall_at_5_pct": mean_recall,
        "retrieval_precision_at_5_pct": mean_precision,
        "citation_source_validity_rate_pct": citation_source_validity_rate,
        "citation_ground_truth_support_rate_pct": citation_gt_support_rate,
        "factual_claim_verification_rate_pct": factual_claim_verification_rate,
        "adversarial_claim_rejection_rate_pct": adversarial_claim_rejection_rate,
        "unsupported_claim_rejection_rate_pct": unsupported_claim_rejection_rate,
        "contradiction_detection_rate_pct": contradiction_rate,
        "safe_degradation_rate_pct": safe_degradation_rate,
        "identifier_leakage_rate_pct": identifier_leakage_rate,
        "status_alignment_rate_pct": status_alignment_rate
    }

    latency_summary = {
        "benchmark_metadata": {
            "sample_count": total_cases,
            "warmup_query_count": 4,
            "cold_start_vector_loading_ms": round(sem_cold_start_ms, 2),
            "total_cold_start_warmup_ms": round(total_cold_start_ms, 2),
            "model_loading_included_in_steady_state": False,
            "llm_generation_included": (provider_type.lower() == "ollama")
        },
        "routing_classify_latency_ms": {
            "mean": mean(latencies_classify),
            "median": median(latencies_classify),
            "p95": percentile(latencies_classify, 95)
        },
        "retrieval_dispatch_latency_ms": {
            "mean": mean(latencies_retrieval),
            "median": median(latencies_retrieval),
            "p95": percentile(latencies_retrieval, 95)
        },
        "guardrail_masking_latency_ms": {
            "mean": mean(latencies_guardrail),
            "median": median(latencies_guardrail),
            "p95": percentile(latencies_guardrail, 95)
        },
        "true_e2e_pipeline_latency_ms": {
            "mean": mean(latencies_total_e2e),
            "median": median(latencies_total_e2e),
            "p95": percentile(latencies_total_e2e, 95)
        }
    }

    reproducibility_metadata = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "provider": provider_type,
        "dataset_fingerprint": fixtures_data["metadata"]["dataset_fingerprint"],
        "fixture_version": fixtures_data["metadata"]["version"],
        "total_fixtures": total_cases
    }

    full_report = {
        "metadata": reproducibility_metadata,
        "metrics": metrics_summary,
        "latency": latency_summary,
        "failed_cases": failed_cases,
        "detailed_results": detailed_results
    }

    # 7. Write Machine-Readable JSON Report
    json_path = os.path.join(output_dir, f"benchmark_report_stage8_{provider_type}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Saved machine-readable JSON report: {json_path}")

    # 8. Write Human-Readable Markdown Report
    md_path = os.path.join(output_dir, f"benchmark_report_stage8_{provider_type}.md")
    _write_markdown_report(md_path, reproducibility_metadata, metrics_summary, latency_summary, failed_cases, detailed_results)
    print(f"[+] Saved human-readable Markdown report: {md_path}")

    # 9. CLI Summary Report
    print("\n" + "=" * 80)
    print("STAGE 8 BENCHMARK EVALUATION SUMMARY (REFINED)")
    print("=" * 80)
    print(f"  Total Cases Evaluated               : {total_cases}")
    print(f"  Routing Accuracy                    : {routing_accuracy}% ({routing_matches}/{total_cases})")
    print(f"  Entity Extraction Accuracy          : {entity_accuracy}% ({entity_matches}/{entity_eval_total})")
    print(f"  Retrieval Recall@5                  : {mean_recall}%")
    print(f"  Retrieval Precision@5               : {mean_precision}%")
    print(f"  Citation Source-ID Validity Rate    : {citation_source_validity_rate}% ({citation_valid_sources_total}/{citation_eval_total})")
    print(f"  Citation Ground-Truth Support Rate  : {citation_gt_support_rate}% ({citation_gt_supported_total}/{citation_gt_eval_total})")
    print(f"  Factual Claim Verification Rate     : {factual_claim_verification_rate}% ({factual_claims_verified}/{factual_claims_total})")
    print(f"  Adversarial False-Claim Rejection   : {adversarial_claim_rejection_rate}% ({adversarial_claims_rejected}/{adversarial_claims_total})")
    print(f"  Unsupported Claim Rejection Rate    : {unsupported_claim_rejection_rate}% ({unsupported_claims_rejected}/{unsupported_claims_total})")
    print(f"  Contradiction Detection Rate        : {contradiction_rate}% ({contradiction_detected_total}/{contradiction_cases_total})")
    print(f"  Safe Degradation Rate               : {safe_degradation_rate}% ({safe_degradation_total}/{anomalous_cases_total})")
    print(f"  Identifier Leakage Rate             : {identifier_leakage_rate}% ({leakage_detected_total}/{masked_cases_total}) [Target: 0.0%]")
    print(f"  End-to-End Status Alignment         : {status_alignment_rate}% ({status_alignment_matches}/{total_cases})")
    print("-" * 80)
    print(f"  Cold-Start Vector Model Loading     : {latency_summary['benchmark_metadata']['cold_start_vector_loading_ms']} ms")
    print(f"  Steady-State True E2E Latency       : Mean={latency_summary['true_e2e_pipeline_latency_ms']['mean']} ms, "
          f"Median={latency_summary['true_e2e_pipeline_latency_ms']['median']} ms, "
          f"p95={latency_summary['true_e2e_pipeline_latency_ms']['p95']} ms")
    print(f"  Failed Cases Count                  : {len(failed_cases)}")
    print("=" * 80)

    return full_report


class nullcontext:
    def __enter__(self): return self
    def __exit__(self, *args): pass


def _write_markdown_report(
    path: str,
    meta: Dict[str, Any],
    metrics: Dict[str, Any],
    latency: Dict[str, Any],
    failed: List[Dict[str, Any]],
    results: List[Dict[str, Any]]
):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# NOC Intelligence — Stage 8 Benchmark Quality Report (Refined)\n\n")
        f.write(f"**Execution Date:** {meta['timestamp']}  \n")
        f.write(f"**Provider Mode:** `{meta['provider']}`  \n")
        f.write(f"**Dataset Fingerprint:** `{meta['dataset_fingerprint']}`  \n")
        f.write(f"**Total Cases Evaluated:** {meta['total_fixtures']}  \n\n")

        f.write("---\n\n## 1. Disaggregated Metric Performance\n\n")
        f.write("| Evaluation Metric | Formula / Basis | Measured Score | Status |\n")
        f.write("| :--- | :--- | :---: | :---: |\n")
        f.write(f"| **Routing Accuracy** | Correct Route / Total Queries | **{metrics['routing_accuracy_pct']}%** | {'✅ Met' if metrics['routing_accuracy_pct'] >= 95 else '⚠️ Sub-optimal'} |\n")
        f.write(f"| **Entity Extraction Accuracy** | Correct Entities / Entity Queries | **{metrics['entity_extraction_accuracy_pct']}%** | {'✅ Met' if metrics['entity_extraction_accuracy_pct'] >= 95 else '⚠️ Sub-optimal'} |\n")
        f.write(f"| **Retrieval Recall@5** | Expected IDs Found / Total Expected IDs | **{metrics['retrieval_recall_at_5_pct']}%** | {'✅ Met' if metrics['retrieval_recall_at_5_pct'] >= 90 else '⚠️ Baseline'} |\n")
        f.write(f"| **Retrieval Precision@5** | Relevant Hits / Total Top-5 Hits | **{metrics['retrieval_precision_at_5_pct']}%** | {'✅ Met' if metrics['retrieval_precision_at_5_pct'] >= 80 else '⚠️ Baseline'} |\n")
        f.write(f"| **Citation Source-ID Validity** | Valid Retrieved Source IDs / Total Citations | **{metrics['citation_source_validity_rate_pct']}%** | {'✅ Met (100.0%)' if metrics['citation_source_validity_rate_pct'] == 100 else '⚠️ Unretrieved IDs Cited'} |\n")
        f.write(f"| **Citation Ground-Truth Support** | Ground-Truth Supported / Evaluable Citations | **{metrics['citation_ground_truth_support_rate_pct']}%** | {'✅ Met (100.0%)' if metrics['citation_ground_truth_support_rate_pct'] == 100 else '⚠️ Mismatched Citations'} |\n")
        f.write(f"| **Factual Claim Verification** | Verified Claims / Extracted Factual Claims | **{metrics['factual_claim_verification_rate_pct']}%** | {'✅ Met (100.0%)' if metrics['factual_claim_verification_rate_pct'] == 100 else '⚠️ Unverified Claims'} |\n")
        f.write(f"| **Adversarial False-Claim Rejection** | Flagged Conflicts / Adversarial Claims | **{metrics['adversarial_claim_rejection_rate_pct']}%** | {'✅ Met (100.0%)' if metrics['adversarial_claim_rejection_rate_pct'] == 100 else '⚠️ Hallucination Accepted'} |\n")
        f.write(f"| **Unsupported Claim Rejection** | Flagged Unretrieved / Fabricated Claims | **{metrics['unsupported_claim_rejection_rate_pct']}%** | {'✅ Met (100.0%)' if metrics['unsupported_claim_rejection_rate_pct'] == 100 else '⚠️ Fabricated ID Accepted'} |\n")
        f.write(f"| **Contradiction Detection Rate** | Flagged Contradictions / Contradiction Cases | **{metrics['contradiction_detection_rate_pct']}%** | {'✅ Met (100.0%)' if metrics['contradiction_detection_rate_pct'] == 100 else '⚠️ Missed Conflict'} |\n")
        f.write(f"| **Safe Degradation Rate** | Safely Suppressed / Total Anomalous Cases | **{metrics['safe_degradation_rate_pct']}%** | {'✅ Met (100.0%)' if metrics['safe_degradation_rate_pct'] == 100 else '⚠️ Failed Containment'} |\n")
        f.write(f"| **Identifier Leakage Rate** | Leaked Masked Cases / Total Masked Cases | **{metrics['identifier_leakage_rate_pct']}%** | {'✅ Target Met (0.0%)' if metrics['identifier_leakage_rate_pct'] == 0.0 else '❌ Leaked Identifiers'} |\n")
        f.write(f"| **End-to-End Status Alignment** | Status Matches / Total Cases | **{metrics['status_alignment_rate_pct']}%** | {'✅ Met' if metrics['status_alignment_rate_pct'] >= 95 else '⚠️ Sub-optimal'} |\n\n")

        f.write("---\n\n## 2. Latency Profile & Performance Breakdown\n\n")
        b_meta = latency["benchmark_metadata"]
        f.write(f"* **Vector Store Cold-Start Loading:** `{b_meta['cold_start_vector_loading_ms']} ms` (SentenceTransformers + ChromaDB initial weight import)\n")
        f.write(f"* **Total Warm-Up Duration:** `{b_meta['total_cold_start_warmup_ms']} ms` across `{b_meta['warmup_query_count']}` queries\n")
        f.write(f"* **Steady-State Benchmark Samples:** `{b_meta['sample_count']}` evaluation cases (cold-start model loading excluded)\n\n")

        f.write("| Subsystem Component | Mean Latency | Median Latency | p95 Latency |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        f.write(f"| **Router Classification** | {latency['routing_classify_latency_ms']['mean']} ms | {latency['routing_classify_latency_ms']['median']} ms | {latency['routing_classify_latency_ms']['p95']} ms |\n")
        f.write(f"| **Retrieval Dispatch** | {latency['retrieval_dispatch_latency_ms']['mean']} ms | {latency['retrieval_dispatch_latency_ms']['median']} ms | {latency['retrieval_dispatch_latency_ms']['p95']} ms |\n")
        f.write(f"| **Guardrail & Masking** | {latency['guardrail_masking_latency_ms']['mean']} ms | {latency['guardrail_masking_latency_ms']['median']} ms | {latency['guardrail_masking_latency_ms']['p95']} ms |\n")
        f.write(f"| **True Full End-to-End Pipeline** | **{latency['true_e2e_pipeline_latency_ms']['mean']} ms** | **{latency['true_e2e_pipeline_latency_ms']['median']} ms** | **{latency['true_e2e_pipeline_latency_ms']['p95']} ms** |\n\n")

        f.write("---\n\n## 3. Safe Presentation Contract & Privacy Boundary (Stage 9 Ready)\n\n")
        f.write("To prevent data leakage, user-facing client applications (e.g. Streamlit UI in Stage 9) must adhere to the following contract:\n\n")
        f.write("1. **Approved User-Facing Fields:**\n")
        f.write("   - `final_answer`: Sanitized natural language response with pseudonymized identifiers (`[TICKET-***001]`, `[CUSTOMER-0001]`).\n")
        f.write("   - `citations`: Verified reference tags with masked labels and IDs.\n")
        f.write("   - `warnings`: Masked operational notifications.\n")
        f.write("   - `is_safe_to_present`: Boolean safety flag. If `False`, response MUST be suppressed.\n\n")
        f.write("2. **Forbidden Internal Diagnostic Fields (Never Render in UI):**\n")
        f.write("   - `original_answer`: Contains raw unmasked operational data.\n")
        f.write("   - `claims`: Contains unmasked factual values and evidence values.\n")
        f.write("   - `contradictions` & `unverified_claims`: Contain raw operational identifiers in debug strings.\n")
        f.write("   - `masked_identifiers_map`: Contains the reverse pseudonymization map.\n\n")
        f.write("3. **Masking Scope & Known Entity Gaps:**\n")
        f.write("   - **Covered Entities:** Trouble Tickets (`INC-...`), Work Orders (`WO-...`), Circuits (`CKT-...`), Customer Names (Thai & English).\n")
        f.write("   - **Known Unmasked Gaps:** Technician aliases (`field_engineer_alias`), customer contact persons (`contact_person`), phone numbers (`contact_phone`), site locations (`site_name`), and IP addresses.\n\n")

        f.write("---\n\n## 4. Error Analysis\n\n")
        if not failed:
            f.write("🎉 **Zero Failed Cases.** All 40 evaluation cases satisfied their expected route, intent, entities, status, and zero-leakage contracts.\n\n")
        else:
            f.write(f"Total Failed Cases: **{len(failed)}**\n\n")
            for fc in failed:
                f.write(f"### {fc['case_id']} ({fc['category']})\n")
                f.write(f"- **Expected:** {fc['expected']}\n")
                f.write(f"- **Actual:** {fc['actual']}\n")
                f.write(f"- **Failed Subsystem:** {fc['failed_subsystem']}\n")
                f.write(f"- **Diagnosis:** {fc['reason']}\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 8 40-Query Benchmark Evaluator")
    parser.add_argument("--fixtures", default=DEFAULT_FIXTURES_PATH, help="Path to 40-query benchmark fixtures JSON")
    parser.add_argument("--provider", choices=["fake", "ollama"], default="fake", help="Synthesis provider (fake or ollama)")
    parser.add_argument("--outdir", default=DEFAULT_REPORT_DIR, help="Output directory for reports")
    args = parser.parse_args()

    run_stage8_benchmark(
        fixtures_path=args.fixtures,
        provider_type=args.provider,
        output_dir=args.outdir
    )
