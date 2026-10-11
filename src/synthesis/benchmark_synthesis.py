"""
NOC Intelligence — Stage 5 Synthesis Grounding & Citation Evaluator
Evaluates grounded synthesis responses, citation validity, fact preservation,
and language consistency across synthetic NOC test cases.
"""

import os
import json
import time
from typing import Optional

from src.router.router import DeterministicRouter
from src.synthesis.synthesizer import GroundedSynthesizer
from src.synthesis.providers import get_provider
from src.synthesis.models import ProviderConfig

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_FIXTURES = os.path.join(ROOT_DIR, "data", "synthesis_evaluation_fixtures.json")


def run_synthesis_evaluation(
    provider_name: str = "fake",
    fixtures_path: str = DEFAULT_FIXTURES
):
    print("=" * 75)
    print(f"STAGE 5 GROUNDED SYNTHESIS EVALUATION (Provider: {provider_name.upper()})")
    print("=" * 75)

    with open(fixtures_path, "r", encoding="utf-8") as f:
        fixtures = json.load(f)

    test_cases = fixtures["test_cases"]
    total = len(test_cases)
    print(f"[*] Loaded {total} evaluation test cases from {fixtures_path}")

    # Initialize router and synthesizer
    router = DeterministicRouter()
    provider = get_provider(provider_name, ProviderConfig(provider_name=provider_name))
    synthesizer = GroundedSynthesizer(provider=provider)

    passed_status = 0
    passed_citations = 0
    passed_facts = 0
    zero_unverified_citations = 0
    latencies_ms = []

    print("\n" + "-" * 75)
    print(f"{'Case ID':<13} | {'Lang':<4} | {'Expected Status':<16} | {'Actual Status':<16} | {'Status'}")
    print("-" * 75)

    for tc in test_cases:
        cid = tc["case_id"]
        lang = tc["language"]
        query = tc["query"]
        exp_status = tc["expected_status"]
        exp_cits = tc.get("expected_citations", [])
        exp_facts = tc.get("expected_fact_keywords", [])

        # Step 1: Route & Dispatch retrieval
        routing_res = router.dispatch(query)

        # Step 2: Synthesize grounded response
        t0 = time.perf_counter()
        synth_res = synthesizer.synthesize(query, routing_res)
        lat_ms = (time.perf_counter() - t0) * 1000
        latencies_ms.append(lat_ms)

        act_status = synth_res.generation_status.value
        status_match = (act_status == exp_status)
        if status_match:
            passed_status += 1

        # Check citation validity
        actual_cit_ids = {c.source_id for c in synth_res.citations}
        cit_valid = True
        if exp_cits:
            for ec in exp_cits:
                if ec not in actual_cit_ids:
                    cit_valid = False
                    break
        if cit_valid:
            passed_citations += 1

        # Check for unverified/hallucinated citations in warnings
        has_unverified = any("unverified citation" in w.lower() for w in synth_res.warnings)
        if not has_unverified:
            zero_unverified_citations += 1

        # Check expected factual keywords
        ans_text = synth_res.answer.lower()
        facts_ok = True
        if exp_facts:
            # At least one key fact or matching term present
            facts_ok = any(f.lower() in ans_text for f in exp_facts)
        if facts_ok:
            passed_facts += 1

        all_ok = status_match and cit_valid and (not has_unverified)
        icon = "✅" if all_ok else "⚠️"
        print(f"{cid:<13} | {lang:<4} | {exp_status:<16} | {act_status:<16} | {icon}")

    status_acc = (passed_status / total) * 100.0
    cit_acc = (passed_citations / total) * 100.0
    unverified_free_acc = (zero_unverified_citations / total) * 100.0
    fact_acc = (passed_facts / total) * 100.0
    avg_lat = sum(latencies_ms) / len(latencies_ms)

    print("\n" + "=" * 75)
    print("EVALUATION METRICS SUMMARY")
    print("=" * 75)
    print(f"  Total Cases Evaluated          : {total}")
    print(f"  Status Contract Alignment     : {passed_status}/{total} ({status_acc:.1f}%)")
    print(f"  Citation Grounding Alignment  : {passed_citations}/{total} ({cit_acc:.1f}%)")
    print(f"  Zero Unverified Citations Rate: {zero_unverified_citations}/{total} ({unverified_free_acc:.1f}%)")
    print(f"  Factual Keyword Retention     : {passed_facts}/{total} ({fact_acc:.1f}%)")
    print(f"  Average Synthesis Latency      : {avg_lat:.2f} ms")


if __name__ == "__main__":
    # Test with deterministic fake provider first
    run_synthesis_evaluation(provider_name="fake")

    # If local Ollama is available, also run with local Ollama
    ollama_prov = get_provider("ollama")
    if ollama_prov.is_available():
        print("\n\n" + "#" * 75)
        print("LOCAL OLLAMA INTEGRATION RUN (Model: typhoon2:8b)")
        print("#" * 75)
        run_synthesis_evaluation(provider_name="ollama")
    else:
        print("\n[!] Local Ollama not active; integration run skipped.")
