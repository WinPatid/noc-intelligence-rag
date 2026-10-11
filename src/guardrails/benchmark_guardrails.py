"""
NOC Intelligence — Stage 6 Guardrail Critic & Masking Benchmark Evaluator
Measures contradiction detection, unsupported-claim detection, identifier leakage,
citation integrity, and safe degradation across synthetic evaluation fixtures.
"""

import os
import json
import time
from typing import Dict, Any, List

from src.router.router import DeterministicRouter
from src.synthesis.models import SynthesisResponse, GenerationStatus, CitationItem
from src.guardrails.pipeline import GuardrailPipeline
from src.guardrails.models import GuardrailStatus, MaskingConfig

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_FIXTURES = os.path.join(ROOT_DIR, "data", "guardrails_evaluation_fixtures.json")


def run_guardrails_evaluation(fixtures_path: str = DEFAULT_FIXTURES):
    print("=" * 75)
    print("STAGE 6 ACTIVE GUARDRAIL CRITIC & IDENTIFIER MASKING EVALUATION")
    print("=" * 75)

    with open(fixtures_path, "r", encoding="utf-8") as f:
        fixtures = json.load(f)

    test_cases = fixtures["test_cases"]
    total = len(test_cases)
    print(f"[*] Loaded {total} benchmark evaluation cases from {fixtures_path}")

    router = DeterministicRouter()

    # Pipelines: standard and masking-enabled
    pipe_standard = GuardrailPipeline(masking_config=MaskingConfig(enabled=False))
    pipe_masked = GuardrailPipeline(masking_config=MaskingConfig(enabled=True))

    # Metric accumulators
    contradiction_total = 0
    contradiction_detected = 0

    unsupported_total = 0
    unsupported_detected = 0

    masking_cases_total = 0
    identifier_leaked_count = 0

    citation_cases_total = 0
    citation_integrity_count = 0

    anomalous_total = 0
    safe_degraded_count = 0

    latencies_us = []

    print("\n" + "-" * 75)
    print(f"{'Case ID':<13} | {'Category':<26} | {'Exp Status':<14} | {'Act Status':<14} | {'Match'}")
    print("-" * 75)

    for tc in test_cases:
        cid = tc["case_id"]
        cat = tc["category"]
        query = tc["query"]
        synth_ans = tc["synthetic_response"]
        exp_status = tc["expected_guardrail_status"]
        test_mask = tc.get("test_masking", False)

        # 1. Route query to fetch actual database / vector evidence
        routing_res = router.dispatch(query)

        # Construct synthetic response object
        gen_status = GenerationStatus.SUCCESS
        if exp_status == "INSUFFICIENT_EVIDENCE":
            gen_status = GenerationStatus.INSUFFICIENT_EVIDENCE
        elif exp_status == "PROVIDER_UNAVAILABLE":
            gen_status = GenerationStatus.PROVIDER_UNAVAILABLE

        # Extract citation references if present
        mock_citations = []
        if "[SOURCE:" in synth_ans:
            import re
            tags = re.findall(r"\[SOURCE:\s*([^\]]+)\]", synth_ans)
            for t in tags:
                parts = t.split(":", 1)
                etype = parts[0]
                ident = parts[1] if len(parts) > 1 else t
                mock_citations.append(CitationItem(source_id=t, entity_type=etype, identifier=ident, label=t))

        synth_res = SynthesisResponse(
            answer=synth_ans,
            language="TH" if any("\u0e00" <= c <= "\u0e7f" for c in synth_ans) else "EN",
            citations=mock_citations,
            evidence_used=[c.source_id for c in mock_citations],
            insufficient_evidence=(exp_status == "INSUFFICIENT_EVIDENCE"),
            provider="fake",
            model="deterministic-fake",
            generation_status=gen_status
        )

        pipeline = pipe_masked if test_mask else pipe_standard

        t0 = time.perf_counter_ns()
        result = pipeline.process(
            query=query,
            routing_result=routing_res,
            synthesis_response=synth_res
        )
        lat_us = (time.perf_counter_ns() - t0) / 1000.0
        latencies_us.append(lat_us)

        act_status = result.status.value
        is_status_match = (act_status == exp_status)

        # Evaluate Contradiction Detection
        if exp_status == "CONTRADICTED":
            contradiction_total += 1
            if result.status == GuardrailStatus.CONTRADICTED:
                contradiction_detected += 1

        # Evaluate Unsupported Claim / Fabricated ID Detection
        if exp_status == "PARTIALLY_VERIFIED" or "fabricated" in cat:
            unsupported_total += 1
            if result.unverified_claims or result.status == GuardrailStatus.PARTIALLY_VERIFIED:
                unsupported_detected += 1

        # Evaluate Identifier Leakage Rate
        if test_mask:
            masking_cases_total += 1
            forbidden = tc.get("forbidden_unmasked", [])
            leaked = any(raw_id in result.final_answer for raw_id in forbidden)
            if leaked:
                identifier_leaked_count += 1

        # Evaluate Citation Integrity Rate
        if mock_citations:
            citation_cases_total += 1
            # Check all citations in result are verified
            if all(c.is_verified for c in result.citations):
                citation_integrity_count += 1

        # Evaluate Safe Degradation Rate
        if exp_status in ["CONTRADICTED", "INSUFFICIENT_EVIDENCE", "PROVIDER_UNAVAILABLE"]:
            anomalous_total += 1
            if result.status in [GuardrailStatus.CONTRADICTED, GuardrailStatus.INSUFFICIENT_EVIDENCE, GuardrailStatus.PROVIDER_UNAVAILABLE]:
                if not result.is_safe_to_present or result.status != GuardrailStatus.CONTRADICTED:
                    safe_degraded_count += 1

        icon = "✅" if is_status_match else "❌"
        print(f"{cid:<13} | {cat:<26} | {exp_status:<14} | {act_status:<14} | {icon}")

    # Compute Explicit Independent Rates
    contra_rate = (contradiction_detected / contradiction_total * 100.0) if contradiction_total > 0 else 0.0
    unsupp_rate = (unsupported_detected / unsupported_total * 100.0) if unsupported_total > 0 else 0.0
    leakage_rate = (identifier_leaked_count / masking_cases_total * 100.0) if masking_cases_total > 0 else 0.0
    cit_rate = (citation_integrity_count / citation_cases_total * 100.0) if citation_cases_total > 0 else 0.0
    degrade_rate = (safe_degraded_count / anomalous_total * 100.0) if anomalous_total > 0 else 0.0
    mean_lat_us = sum(latencies_us) / len(latencies_us)

    print("\n" + "=" * 75)
    print("STAGE 6 EVALUATION METRICS REPORT (Disaggregated)")
    print("=" * 75)
    print(f"  Total Cases Evaluated            : {total}")
    print(f"  Contradiction Detection Rate     : {contradiction_detected}/{contradiction_total} ({contra_rate:.1f}%)")
    print(f"  Unsupported-Claim Detection Rate : {unsupported_detected}/{unsupported_total} ({unsupp_rate:.1f}%)")
    print(f"  Identifier Leakage Rate          : {identifier_leaked_count}/{masking_cases_total} ({leakage_rate:.1f}%) [Zero Leakage = Target]")
    print(f"  Citation Integrity Rate          : {citation_integrity_count}/{citation_cases_total} ({cit_rate:.1f}%)")
    print(f"  Safe Degradation Rate            : {safe_degraded_count}/{anomalous_total} ({degrade_rate:.1f}%)")
    print(f"  Mean Guardrail Latency           : {mean_lat_us:.2f} µs ({mean_lat_us/1000.0:.3f} ms)")


if __name__ == "__main__":
    run_guardrails_evaluation()
