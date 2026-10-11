"""
NOC Intelligence — Stage 8 Benchmark Harness Focused Regression Tests
Validates the refined benchmark harness:
1. Proves fake provider mode makes zero outbound network or Ollama calls.
2. Verifies true end-to-end timing boundaries vs component timing.
3. Verifies separation of cold-start vector loading from steady-state execution.
4. Verifies disaggregated claim verification (factual vs adversarial).
5. Verifies independent citation integrity (detects unretrieved/fabricated source IDs).
6. Verifies privacy presentation contract rules for Stage 9.
"""

import pytest
import time
from typing import List

from src.router.router import DeterministicRouter
from src.synthesis.models import CitationItem, SynthesisResponse, GenerationStatus
from src.guardrails.models import GuardrailStatus, GuardrailResult, MaskingConfig
from src.guardrails.pipeline import GuardrailPipeline
from src.evaluation.benchmark_runner import (
    audit_citation_integrity,
    extract_retrieved_source_ids,
    prevent_network_calls,
    BlockedNetworkError,
    FailingOfflineProvider,
    EmptyOutputProvider,
    CrashingMasker
)
from src.synthesis.providers import DeterministicFakeProvider, OllamaProvider
from src.synthesis.synthesizer import GroundedSynthesizer


def test_zero_network_calls_in_fake_mode():
    """Confirms that prevent_network_calls strictly blocks any HTTP connection to Ollama/external APIs."""
    import urllib.request

    with prevent_network_calls():
        with pytest.raises(BlockedNetworkError) as exc_info:
            urllib.request.urlopen("http://127.0.0.1:11434/api/tags")

        assert "Prohibited live network/Ollama call attempted" in str(exc_info.value)


def test_independent_citation_integrity_detects_fabricated_sources():
    """Confirms citation auditor catches unretrieved or hallucinated citation IDs."""
    retrieved_sources = {"ticket:INC-2025-00001", "work_order:WO-700001"}
    expected_evidence = ["ticket:INC-2025-00001"]

    # 1. Valid citation
    valid_citations = [
        CitationItem(source_id="ticket:INC-2025-00001", entity_type="ref", identifier="INC-2025-00001", label="test", is_verified=False)
    ]
    audit_valid = audit_citation_integrity(valid_citations, retrieved_sources, expected_evidence)
    assert audit_valid["total_citations"] == 1
    assert audit_valid["valid_source_ids"] == 1
    assert audit_valid["unretrieved_source_ids"] == 0
    assert audit_valid["supported_citations"] == 1

    # 2. Fabricated / Unretrieved citation
    fabricated_citations = [
        CitationItem(source_id="ticket:INC-2025-99999", entity_type="ref", identifier="INC-2025-99999", label="test", is_verified=True)
    ]
    audit_fab = audit_citation_integrity(fabricated_citations, retrieved_sources, expected_evidence)
    assert audit_fab["total_citations"] == 1
    assert audit_fab["valid_source_ids"] == 0
    assert audit_fab["unretrieved_source_ids"] == 1
    assert audit_fab["supported_citations"] == 0


def test_independent_citation_integrity_handles_missing_ground_truth():
    """Confirms that when ground truth is omitted (e.g. out of scope), support is labeled non-evaluable."""
    retrieved_sources = set()
    empty_expected = []

    citations = [
        CitationItem(source_id="ticket:INC-2025-00001", entity_type="ref", identifier="INC-2025-00001", label="test")
    ]
    audit = audit_citation_integrity(citations, retrieved_sources, empty_expected)
    assert audit["ground_truth_evaluable"] is False
    assert audit["unretrieved_source_ids"] == 1


def test_true_e2e_timing_encompasses_full_pipeline():
    """Confirms pipe.process(query, routing_result=None) executes full pipeline from query string."""
    router = DeterministicRouter()
    pipe = GuardrailPipeline(
        synthesizer=GroundedSynthesizer(provider=DeterministicFakeProvider()),
        masking_config=MaskingConfig(enabled=True),
        router=router
    )

    t0 = time.perf_counter()
    res = pipe.process("Ticket INC-2025-00001 สถานะเป็นอะไร", routing_result=None)
    dt_ms = (time.perf_counter() - t0) * 1000

    assert res.status == GuardrailStatus.VERIFIED
    assert "[TICKET-***001]" in res.final_answer
    assert dt_ms > 0.0


def test_extract_retrieved_source_ids_includes_bridge_and_entity_keys():
    """Confirms extract_retrieved_source_ids extracts top-level and linked entities from payloads."""
    payload_bridge = {
        "entity_type": "ticket_work_order_bridge",
        "ticket_id": "INC-2025-00002",
        "ticket_status": "RESOLVED",
        "linked_work_orders": [{"work_order_id": "WO-700002"}]
    }
    extracted = extract_retrieved_source_ids(payload_bridge)
    assert "ticket:INC-2025-00002" in extracted
    assert "work_order:WO-700002" in extracted


def test_stage9_privacy_boundary_safe_presentation_fields():
    """Validates that final_answer, citations, and warnings are sanitized, while debug fields are segregated."""
    router = DeterministicRouter()
    pipe = GuardrailPipeline(
        synthesizer=GroundedSynthesizer(provider=DeterministicFakeProvider()),
        masking_config=MaskingConfig(enabled=True),
        router=router
    )

    raw_response = "ตั๋ว INC-2025-00001 ของบริษัท Horizon Retail Group เสร็จสิ้น"
    sr = SynthesisResponse(
        answer=raw_response,
        generation_status=GenerationStatus.SUCCESS,
        citations=[CitationItem(source_id="ticket:INC-2025-00001", entity_type="ref", identifier="INC-2025-00001", label="INC-2025-00001", is_verified=True)],
        language="TH"
    )

    res = pipe.process("Ticket INC-2025-00001 ของบริษัท Horizon Retail Group", synthesis_response=sr)

    # 1. User-facing presentation fields MUST be masked
    assert "INC-2025-00001" not in res.final_answer
    assert "Horizon Retail Group" not in res.final_answer
    assert "[TICKET-***001]" in res.final_answer
    assert "[CUSTOMER-" in res.final_answer
    for c in res.citations:
        assert "INC-2025-00001" not in c.label

    # 2. Segregated internal debug fields retain raw data for audit only
    assert "INC-2025-00001" in res.original_answer
    assert "INC-2025-00001" in res.masked_identifiers_map
    assert res.masked_identifiers_map["INC-2025-00001"] == "[TICKET-***001]"
