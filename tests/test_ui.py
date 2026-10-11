"""
NOC Intelligence — Stage 9 Streamlit UI & Presentation Adapter Test Suite
Verifies:
1. Safe presentation adapter allowlisting
2. Unsafe result suppression (Fail-closed on is_safe_to_present=False)
3. Missing or invalid safety flag fails closed
4. Raw operational identifiers are never exposed in user-facing fields
5. Internal diagnostic fields (original_answer, claims, etc.) never reach UI layer
6. Masking failure does not reveal original unmasked answer
7. Provider failure produces safe user-facing explanation
8. Fake provider mode performs zero network / Ollama requests
9. Representative SQL query path through UI service
10. Representative semantic query path through UI service
11. Representative hybrid query path through UI service
12. Clarification and unsupported query behaviors
13. Streamlit AppTest headless execution smoke tests
"""

import pytest
from typing import Dict, Any, List

from src.guardrails.models import (
    GuardrailStatus,
    GuardrailResult,
    MaskingConfig,
    ClaimVerificationItem,
    ClaimType
)
from src.synthesis.models import CitationItem
from src.ui.presentation_adapter import (
    PresentationAdapter,
    SafePresentationView,
    SafeCitationView,
    SAFE_FALLBACK_TH,
    SAFE_FALLBACK_EN,
    FORBIDDEN_FIELDS,
)
from src.ui.pipeline_service import (
    execute_safe_query,
    get_pipeline,
    check_ollama_availability
)
from src.evaluation.benchmark_runner import (
    prevent_network_calls,
    BlockedNetworkError
)
from streamlit.testing.v1 import AppTest


# ═══════════════════════════════════════════════════════════════════════════════
# 1. ADAPTER ALLOWLISTING & FIELD BOUNDARY TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_presentation_adapter_allowlist():
    """Confirms only allowlisted fields exist on SafePresentationView."""
    res = GuardrailResult(
        status=GuardrailStatus.VERIFIED,
        original_answer="Internal raw response with INC-2025-00001",
        final_answer="Sanitized response for [TICKET-***001]",
        claims=[
            ClaimVerificationItem(
                claim_type=ClaimType.STATUS,
                claimed_value="RESOLVED",
                evidence_value="RESOLVED",
                is_verified=True,
                is_contradicted=False
            )
        ],
        contradictions=[],
        unverified_claims=[],
        masked_identifiers_map={"INC-2025-00001": "[TICKET-***001]"},
        citations=[
            CitationItem(
                source_id="ticket:[TICKET-***001]",
                entity_type="trouble_ticket",
                identifier="[TICKET-***001]",
                label="Ticket status",
                is_verified=True
            )
        ],
        is_safe_to_present=True,
        warnings=["Test warning"],
        execution_time_ms=12.4
    )

    view = PresentationAdapter.adapt(res, route_type="SQL_LOOKUP", intent="TICKET_STATUS")

    assert isinstance(view, SafePresentationView)
    assert view.is_safe_to_present is True
    assert view.final_answer == "Sanitized response for [TICKET-***001]"
    assert len(view.citations) == 1
    assert view.citations[0].identifier == "[TICKET-***001]"
    assert view.route_type == "SQL_LOOKUP"
    assert view.intent == "TICKET_STATUS"
    assert view.status == "VERIFIED"

    # Verify dictionary export contains ONLY allowlisted keys
    view_dict = view.to_dict()
    for forbidden in FORBIDDEN_FIELDS:
        assert forbidden not in view_dict, f"Forbidden field '{forbidden}' found in to_dict()!"


def test_internal_diagnostic_fields_never_reach_ui():
    """Confirms accessing forbidden diagnostic fields raises AttributeError."""
    res = GuardrailResult(
        status=GuardrailStatus.VERIFIED,
        original_answer="Internal raw response",
        final_answer="Sanitized answer",
        is_safe_to_present=True
    )
    view = PresentationAdapter.adapt(res)

    for forbidden in FORBIDDEN_FIELDS:
        with pytest.raises(AttributeError) as exc_info:
            getattr(view, forbidden)
        assert f"Access to internal diagnostic field '{forbidden}' is strictly prohibited" in str(exc_info.value)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. FAIL-CLOSED SAFETY GATE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_unsafe_result_suppression():
    """Confirms when is_safe_to_present=False, the response is replaced with a safe fallback."""
    res = GuardrailResult(
        status=GuardrailStatus.CONTRADICTED,
        original_answer="False statement: Ticket INC-2025-00001 is OPEN",
        final_answer="False statement: Ticket [TICKET-***001] is OPEN",
        citations=[
            CitationItem(source_id="ticket:1", entity_type="ticket", identifier="1", label="test")
        ],
        is_safe_to_present=False,
        execution_time_ms=5.0
    )

    view = PresentationAdapter.adapt(res, language_hint="EN")

    assert view.is_safe_to_present is False
    assert view.final_answer == SAFE_FALLBACK_EN
    # Citations must be completely stripped on suppressed outputs
    assert view.citations == []
    assert view.status == "CONTRADICTED"


def test_missing_safety_flag_fails_closed():
    """Confirms None or non-boolean safety flag strictly fails closed."""
    class MalformedResult:
        final_answer = "Potentially unsafe response"
        is_safe_to_present = None  # Missing / non-boolean
        status = "UNKNOWN"
        citations = []
        warnings = []
        execution_time_ms = 1.0

    view = PresentationAdapter.adapt(MalformedResult(), language_hint="TH")

    assert view.is_safe_to_present is False
    assert view.final_answer == SAFE_FALLBACK_TH
    assert view.citations == []


def test_raw_identifiers_not_exposed_in_rendered_fields():
    """Confirms defense-in-depth catches unmasked operational IDs and fails closed."""
    res = GuardrailResult(
        status=GuardrailStatus.VERIFIED,
        original_answer="Raw INC-2025-00001 and WO-700165",
        # Leaked unmasked raw ticket in final_answer despite is_safe_to_present=True
        final_answer="Unmasked leak of INC-2025-00001 in response text",
        is_safe_to_present=True
    )

    # When masking is enabled, this violation MUST be caught and suppressed
    view = PresentationAdapter.adapt(res, language_hint="EN", masking_enabled=True)

    assert view.is_safe_to_present is False
    assert "Unmasked operational identifier detected" in view.final_answer
    assert view.status == "MASKING_VIOLATION"


def test_masking_failure_does_not_reveal_original_answer():
    """Confirms failure during masking in pipeline suppresses output safely."""
    class FailingPipelineStub:
        def process(self, q, routing_result=None):
            return GuardrailResult(
                status=GuardrailStatus.INVALID_RESPONSE,
                original_answer="Sensitive operational ticket INC-2025-00099",
                final_answer="⚠️ Masking failed. Suppressed for data safety.",
                is_safe_to_present=False,
                warnings=["Masking error: Exception"]
            )

    view = execute_safe_query(
        "Ticket INC-2025-00099 status",
        pipeline=FailingPipelineStub(),
        masking_enabled=True
    )

    assert view.is_safe_to_present is False
    assert "Sensitive operational ticket" not in view.final_answer
    assert "INC-2025-00099" not in view.final_answer
    assert len(view.citations) == 0


def test_provider_failure_produces_safe_message():
    """Confirms provider exception produces safe degradation without unmasked leakage."""
    class CrashingPipelineStub:
        def process(self, q, routing_result=None):
            raise RuntimeError("Database connection timeout or internal failure")

    view = execute_safe_query(
        "What is the status of INC-2025-00001?",
        pipeline=CrashingPipelineStub()
    )

    assert view.is_safe_to_present is False
    assert view.status == "EXECUTION_ERROR"
    # Never leak internal stack trace in final_answer
    assert "RuntimeError" not in view.final_answer
    assert "Database connection timeout" not in view.final_answer


# ═══════════════════════════════════════════════════════════════════════════════
# 3. ZERO NETWORK CALLS & REPRESENTATIVE PATH TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_fake_provider_mode_performs_no_network_calls():
    """Verifies pipeline with fake provider makes zero outbound network or Ollama socket calls."""
    with prevent_network_calls():
        view = execute_safe_query(
            "Ticket INC-2025-00001 สถานะเป็นอะไร",
            provider_type="fake",
            masking_enabled=True
        )

        assert view.is_safe_to_present is True
        assert view.status == "VERIFIED"
        assert "[TICKET-***001]" in view.final_answer


def test_representative_sql_query_path():
    """Verifies end-to-end SQL query path through UI service."""
    view = execute_safe_query(
        "Ticket INC-2025-00001 สถานะเป็นอะไร",
        provider_type="fake",
        masking_enabled=True
    )

    assert view.is_safe_to_present is True
    assert view.status == "VERIFIED"
    assert view.route_type == "SQL_LOOKUP"
    assert view.intent == "TICKET_STATUS"
    assert "[TICKET-***001]" in view.final_answer
    assert any("[TICKET-***001]" in c.identifier for c in view.citations)
    assert view.execution_time_ms > 0.0


def test_representative_work_order_and_circuit_sql_paths():
    """Verifies work order detail and circuit lookup through UI service."""
    # Work order
    view_wo = execute_safe_query(
        "Show resolution for work order WO-700165",
        provider_type="fake",
        masking_enabled=True
    )
    assert view_wo.is_safe_to_present is True
    assert view_wo.route_type == "SQL_LOOKUP"
    assert view_wo.intent == "WORK_ORDER_DETAIL"
    assert "[WO-***165]" in view_wo.final_answer

    # Circuit detail
    view_ckt = execute_safe_query(
        "วงจร CKT-880001 เป็นของบริษัทอะไร",
        provider_type="fake",
        masking_enabled=True
    )
    assert view_ckt.is_safe_to_present is True
    assert view_ckt.route_type == "SQL_LOOKUP"
    assert view_ckt.intent == "CIRCUIT_DETAIL"
    assert "[CIRCUIT-***001]" in view_ckt.final_answer


def test_representative_semantic_query_path():
    """Verifies semantic troubleshooting retrieval through UI service."""
    view = execute_safe_query(
        "ไฟฟ้าดับเป็นวงกว้าง การไฟฟ้า PEA ดับไฟซ่อมหม้อแปลง",
        provider_type="fake",
        masking_enabled=True
    )

    assert view.is_safe_to_present is True
    assert view.status == "VERIFIED"
    assert view.route_type == "SEMANTIC_SEARCH"
    assert view.intent == "SEMANTIC_SYMPTOM"
    assert len(view.citations) > 0


def test_representative_hybrid_query_path():
    """Verifies hybrid SQL-scoped vector search through UI service."""
    view = execute_safe_query(
        "ค้นหาปัญหา fiber degradation ของบริษัท Apex Logistics",
        provider_type="fake",
        masking_enabled=True
    )

    assert view.is_safe_to_present is True
    assert view.status == "VERIFIED"
    assert view.route_type == "HYBRID_SEARCH"
    assert view.intent == "HYBRID_CUSTOMER_INCIDENTS"
    assert len(view.citations) > 0


def test_clarification_and_unsupported_query_behavior():
    """Verifies clarification requests and unsupported queries handle gracefully."""
    # Empty query
    view_empty = execute_safe_query("", provider_type="fake")
    assert view_empty.is_safe_to_present is True
    assert view_empty.status == "EMPTY_QUERY"

    # Ambiguous query (missing ID)
    view_clarify = execute_safe_query("ตั๋วสถานะเป็นอะไร", provider_type="fake")
    assert view_clarify.is_safe_to_present is True
    assert view_clarify.route_type == "CLARIFICATION"
    assert "กรุณาระบุหมายเลขตั๋ว" in view_clarify.final_answer
    assert view_clarify.citations == []

    # Unsupported query (out of scope)
    view_unsupported = execute_safe_query("พยากรณ์อากาศที่กรุงเทพวันนี้ฝนจะตกไหม", provider_type="fake")
    assert view_unsupported.is_safe_to_present is True
    assert view_unsupported.route_type == "UNSUPPORTED"
    assert "ระบบ NOC Intelligence รองรับเฉพาะข้อมูลปฏิบัติการโทรคมนาคม" in view_unsupported.final_answer
    assert view_unsupported.citations == []


# ═══════════════════════════════════════════════════════════════════════════════
# 4. STREAMLIT CHATBOT APPTEST TESTS (HEADLESS EXECUTION)
# ═══════════════════════════════════════════════════════════════════════════════

def test_chatbot_free_text_submission():
    """
    Verifies free-text query submission via chat_input:
    1. Pipeline processes query
    2. User and assistant messages appended to session state
    3. Status badge and masked citation rendered without uncaught exceptions
    """
    at = AppTest.from_file("src/ui/app.py", default_timeout=30)
    at.run()

    assert len(at.exception) == 0
    assert len(at.chat_input) == 1
    assert len(at.session_state["messages"]) == 0

    # Submit free-text question
    at.chat_input[0].set_value("Ticket INC-2025-00001 สถานะเป็นอะไร").run()

    assert len(at.exception) == 0
    assert len(at.session_state["messages"]) == 2
    assert at.session_state["messages"][0]["role"] == "user"
    assert at.session_state["messages"][1]["role"] == "assistant"
    assert len(at.chat_message) == 2

    # Verify rendered content
    rendered_text = " ".join([m.value for m in at.markdown])
    assert "VERIFIED EVIDENCE" in rendered_text
    assert "[TICKET-***001]" in rendered_text


def test_chatbot_multi_turn_history():
    """
    Verifies conversation history across sequential multi-turn questions:
    Both turns remain visible in session state and chat message list.
    """
    at = AppTest.from_file("src/ui/app.py", default_timeout=30)
    at.run()

    # Turn 1
    at.chat_input[0].set_value("Ticket INC-2025-00001 สถานะเป็นอะไร").run()
    assert len(at.session_state["messages"]) == 2

    # Turn 2
    at.chat_input[0].set_value("Show resolution for work order WO-700165").run()
    assert len(at.session_state["messages"]) == 4
    assert len(at.chat_message) == 4

    # Ensure messages from both turns are present in rendered markdown
    rendered_text = " ".join([m.value for m in at.markdown])
    assert "[TICKET-***001]" in rendered_text
    assert "[WO-***165]" in rendered_text


def test_chatbot_clear_conversation():
    """
    Verifies the Clear Conversation action resets chat history.
    """
    at = AppTest.from_file("src/ui/app.py", default_timeout=30)
    at.run()

    # Submit a query first
    at.chat_input[0].set_value("Ticket INC-2025-00001 สถานะเป็นอะไร").run()
    assert len(at.session_state["messages"]) == 2

    # Click Clear Conversation button
    clear_btn = [b for b in at.button if b.key == "clear_chat_button"][0]
    clear_btn.click().run()

    assert len(at.exception) == 0
    assert len(at.session_state["messages"]) == 0
    assert len(at.chat_message) == 0


def test_chatbot_starter_prompt_submission():
    """
    Verifies clicking an example starter prompt submits the question directly.
    """
    at = AppTest.from_file("src/ui/app.py", default_timeout=30)
    at.run()

    starter_btn = [b for b in at.button if b.key == "starter_btn_0"][0]
    starter_btn.click().run()

    assert len(at.exception) == 0
    assert len(at.session_state["messages"]) == 2
    assert at.session_state["messages"][0]["content"] == "Ticket INC-2025-00001 สถานะเป็นอะไร"
    assert len(at.chat_message) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# 7. FACTUAL EVIDENCE CONTENT & CONVERSATIONAL REGRESSION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_ui_factual_ticket_status():
    """Verifies that ticket inquiry returns factual status and severity in final answer."""
    # Unmasked view
    view = execute_safe_query("Ticket INC-2025-00001 สถานะเป็นอะไร", masking_enabled=False)
    assert view.is_safe_to_present is True
    assert "RESOLVED" in view.final_answer
    assert "P1-Critical" in view.final_answer
    assert "INC-2025-00001" in view.final_answer

    # Masked view
    view_m = execute_safe_query("Ticket INC-2025-00001 สถานะเป็นอะไร", masking_enabled=True)
    assert view_m.is_safe_to_present is True
    assert "[TICKET-***001]" in view_m.final_answer
    assert "INC-2025-00001" not in view_m.final_answer
    assert "RESOLVED" in view_m.final_answer


def test_ui_factual_work_order_resolution():
    """Verifies that work order inquiry returns factual cause and resolution in final answer."""
    view = execute_safe_query("Show resolution for work order WO-700165", masking_enabled=False)
    assert view.is_safe_to_present is True
    assert "COMPLETED" in view.final_answer
    assert "MTU" in view.final_answer
    assert "WO-700165" in view.final_answer

    # Masked view
    view_m = execute_safe_query("Show resolution for work order WO-700165", masking_enabled=True)
    assert view_m.is_safe_to_present is True
    assert "[WO-***165]" in view_m.final_answer
    assert "WO-700165" not in view_m.final_answer
    assert "MTU" in view_m.final_answer


def test_ui_factual_linked_work_order_bridge():
    """Verifies that bridge traversal returns linked work orders and statuses."""
    view = execute_safe_query("ตั๋ว INC-2025-00042 เชื่อมโยงกับใบงานช่างใบไหนบ้าง", masking_enabled=False)
    assert view.is_safe_to_present is True
    assert "INC-2025-00042" in view.final_answer
    assert "WO-700042" in view.final_answer
    assert len(view.citations) >= 1

    # Masked view
    view_m = execute_safe_query("ตั๋ว INC-2025-00042 เชื่อมโยงกับใบงานช่างใบไหนบ้าง", masking_enabled=True)
    assert view_m.is_safe_to_present is True
    assert "[TICKET-***042]" in view_m.final_answer
    assert "[WO-***042]" in view_m.final_answer


def test_ui_factual_circuit_details():
    """Verifies that circuit inquiry returns customer and site details."""
    view = execute_safe_query("วงจร CKT-880001 เป็นของบริษัทอะไร และอยู่ที่สาขาไหน", masking_enabled=False)
    assert view.is_safe_to_present is True
    assert "CKT-880001" in view.final_answer
    assert "Apex Logistics" in view.final_answer


def test_ui_language_consistency_english_query():
    """Verifies that an English query returns an English response even with Thai DB entities."""
    view = execute_safe_query("What is the status of ticket INC-2025-00001?", masking_enabled=False)
    assert view.is_safe_to_present is True
    assert "According to verified system records" in view.final_answer
    assert "RESOLVED" in view.final_answer
    assert "P1-Critical" in view.final_answer


def test_ui_language_consistency_thai_query():
    """Verifies that a Thai query returns a Thai response."""
    view = execute_safe_query("Ticket INC-2025-00001 สถานะเป็นอะไร", masking_enabled=False)
    assert view.is_safe_to_present is True
    assert "จากข้อมูลในระบบ" in view.final_answer
    assert "สถานะปัจจุบัน" in view.final_answer


def test_ui_conversational_greeting_english():
    """Verifies that English greetings route to clarification with capabilities guide."""
    view = execute_safe_query("Hello! How can you help me?")
    assert view.is_safe_to_present is True
    assert view.route_type == "CLARIFICATION"
    assert view.intent == "CLARIFICATION_GREETING"
    assert "NOC Intelligence" in view.final_answer
    assert "trouble ticket statuses" in view.final_answer.lower()


def test_ui_conversational_greeting_thai():
    """Verifies that Thai greetings route to clarification with capabilities guide in Thai."""
    view = execute_safe_query("สวัสดีครับ ช่วยอะไรได้บ้าง")
    assert view.is_safe_to_present is True
    assert view.route_type == "CLARIFICATION"
    assert view.intent == "CLARIFICATION_GREETING"
    assert "สวัสดีครับ" in view.final_answer
    assert "NOC Intelligence" in view.final_answer


def test_ui_unsupported_out_of_scope_query():
    """Verifies that truly out-of-scope non-greeting queries are rejected safely."""
    view = execute_safe_query("พยากรณ์อากาศที่กรุงเทพวันนี้ฝนจะตกไหม")
    assert view.is_safe_to_present is True
    assert view.route_type == "UNSUPPORTED"
    assert "UNSUPPORTED" in view.intent
    assert "NOC Intelligence" in view.final_answer


def test_ui_offline_network_guard_verification():
    """Verifies fake provider pipeline strictly performs zero external network requests."""
    test_queries = [
        "Ticket INC-2025-00001 สถานะเป็นอะไร",
        "Show resolution for work order WO-700165",
        "Hello! What can you do?",
        "สวัสดีครับ ช่วยอะไรได้บ้าง",
        "พยากรณ์อากาศที่กรุงเทพวันนี้ฝนจะตกไหม"
    ]
    with prevent_network_calls():
        for q in test_queries:
            view = execute_safe_query(q, provider_type="fake")
            assert view.is_safe_to_present is True


