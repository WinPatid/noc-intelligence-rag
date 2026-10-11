"""
NOC Intelligence — Stage 4 Router Automated Tests
Verifies deterministic classification, entity extraction, SQL routing,
semantic routing, hybrid retrieval, clarification prompts, and injection defense.
"""

import pytest

from src.router.router import DeterministicRouter
from src.router.models import RouteType, IntentType


@pytest.fixture(scope="module")
def router():
    return DeterministicRouter()


# ─── 1. SQL Route Tests ────────────────────────────────────────────────────────

def test_route_sql_direct_ticket_status(router):
    """Direct ticket ID routes to SQL_LOOKUP / TICKET_STATUS."""
    query = "Ticket INC-2025-00001 สถานะเป็นอะไร"
    decision = router.classify(query)
    assert decision.route_type == RouteType.SQL_LOOKUP
    assert decision.normalized_intent == IntentType.TICKET_STATUS
    assert "INC-2025-00001" in decision.extracted_entities.ticket_ids
    assert decision.confidence_score == 1.0

    # Test dispatch
    res = router.dispatch(query)
    assert res.success is True
    assert res.retrieval_results["found"] is True
    assert res.retrieval_results["record"]["ticket_id"] == "INC-2025-00001"


def test_route_sql_direct_work_order(router):
    """Direct work order ID routes to SQL_LOOKUP / WORK_ORDER_DETAIL."""
    query = "Work Order WO-700001 สาเหตุและอุปกรณ์ที่มีปัญหาคืออะไร"
    decision = router.classify(query)
    assert decision.route_type == RouteType.SQL_LOOKUP
    assert decision.normalized_intent == IntentType.WORK_ORDER_DETAIL
    assert "WO-700001" in decision.extracted_entities.work_order_ids

    res = router.dispatch(query)
    assert res.success is True
    assert res.retrieval_results["found"] is True
    assert res.retrieval_results["record"]["work_order_id"] == "WO-700001"


def test_route_sql_bridge_ticket_to_wo(router):
    """Ticket ID with bridge keywords routes to SQL_LOOKUP / BRIDGE_TICKET_TO_WO."""
    query = "ตั๋ว INC-2025-00042 เชื่อมโยงกับใบงานช่างใบไหนบ้าง"
    decision = router.classify(query)
    assert decision.route_type == RouteType.SQL_LOOKUP
    assert decision.normalized_intent == IntentType.BRIDGE_TICKET_TO_WO

    res = router.dispatch(query)
    assert res.success is True
    assert res.retrieval_results["ticket_found"] is True
    assert res.retrieval_results["linked_work_order_count"] >= 1


def test_route_sql_reverse_bridge_wo_to_tickets(router):
    """Work order ID with affected tickets keywords routes to SQL_LOOKUP / BRIDGE_WO_TO_TICKETS."""
    query = "ใบงาน WO-700001 กระทบกับตั๋วลูกค้าใบไหนบ้าง"
    decision = router.classify(query)
    assert decision.route_type == RouteType.SQL_LOOKUP
    assert decision.normalized_intent == IntentType.BRIDGE_WO_TO_TICKETS

    res = router.dispatch(query)
    assert res.success is True
    assert res.retrieval_results["work_order_found"] is True
    assert res.retrieval_results["impacted_ticket_count"] >= 1


def test_route_sql_circuit_detail(router):
    """Direct circuit reference routes to SQL_LOOKUP / CIRCUIT_DETAIL."""
    query = "วงจร CKT-880001 เป็นของบริษัทอะไร และอยู่ที่สาขาไหน"
    decision = router.classify(query)
    assert decision.route_type == RouteType.SQL_LOOKUP
    assert decision.normalized_intent == IntentType.CIRCUIT_DETAIL
    assert "CKT-880001" in decision.extracted_entities.circuit_refs

    res = router.dispatch(query)
    assert res.success is True
    assert res.retrieval_results["found"] is True
    assert res.retrieval_results["record"]["circuit_ref"] == "CKT-880001"
    assert res.retrieval_results["record"]["company_name_en"] == "Apex Logistics"


def test_route_sql_customer_ticket_count(router):
    """Customer ticket counting query routes to SQL_LOOKUP / CUSTOMER_TICKET_COUNT."""
    query = "Apex Logistics มีตั๋วแจ้งปัญหาทั้งหมดกี่รายการ"
    decision = router.classify(query)
    assert decision.route_type == RouteType.SQL_LOOKUP
    assert decision.normalized_intent == IntentType.CUSTOMER_TICKET_COUNT

    res = router.dispatch(query)
    assert res.success is True
    assert res.retrieval_results["counts"]["found"] is True
    assert res.retrieval_results["counts"]["total_tickets"] > 0


def test_route_sql_customer_circuits(router):
    """Customer circuit inventory query routes to SQL_LOOKUP / CUSTOMER_CIRCUITS."""
    query = "List all active leased line circuits for Nova Cloud Solutions"
    decision = router.classify(query)
    assert decision.route_type == RouteType.SQL_LOOKUP
    assert decision.normalized_intent == IntentType.CUSTOMER_CIRCUITS

    res = router.dispatch(query)
    assert res.success is True
    assert res.retrieval_results["circuit_count"] == 6


# ─── 2. Semantic Route Tests ───────────────────────────────────────────────────

def test_route_semantic_symptoms(router):
    """Descriptive symptom query without customer routes to SEMANTIC_SEARCH / SEMANTIC_SYMPTOM."""
    query = "ไฟฟ้าดับเป็นวงกว้าง การไฟฟ้า PEA ดับไฟซ่อมหม้อแปลง"
    decision = router.classify(query)
    assert decision.route_type == RouteType.SEMANTIC_SEARCH
    assert decision.normalized_intent == IntentType.SEMANTIC_SYMPTOM
    assert len(decision.extracted_entities.semantic_terms) > 0


def test_route_semantic_resolution(router):
    """Technical repair/resolution question routes to SEMANTIC_SEARCH / SEMANTIC_RESOLUTION."""
    query = "How did engineers resolve core router BGP route-policy flap?"
    decision = router.classify(query)
    assert decision.route_type == RouteType.SEMANTIC_SEARCH
    assert decision.normalized_intent == IntentType.SEMANTIC_RESOLUTION


# ─── 3. Hybrid Route Tests ─────────────────────────────────────────────────────

def test_route_hybrid_customer_incidents(router):
    """Customer-scoped query with semantic symptom terms routes to HYBRID_SEARCH."""
    query = "ค้นหาปัญหา fiber degradation ของบริษัท Apex Logistics"
    decision = router.classify(query)
    assert decision.route_type == RouteType.HYBRID_SEARCH
    assert decision.normalized_intent == IntentType.HYBRID_CUSTOMER_INCIDENTS
    assert decision.extracted_entities.matched_customer is not None

    res = router.dispatch(query, top_k=3)
    assert res.success is True
    assert res.retrieval_results["hybrid_type"] == "customer_scoped"
    assert res.retrieval_results["customer_id"] == 1
    assert "incidents" in res.retrieval_results


def test_route_hybrid_circuit_incidents(router):
    """Circuit-scoped query with semantic terms routes to HYBRID_SEARCH."""
    query = "มีประวัติสายไฟเบอร์ขาดหรือแสงตกสำหรับวงจร CKT-880001 หรือไม่"
    decision = router.classify(query)
    assert decision.route_type == RouteType.HYBRID_SEARCH
    assert decision.normalized_intent == IntentType.HYBRID_CIRCUIT_INCIDENTS
    assert "CKT-880001" in decision.extracted_entities.circuit_refs


# ─── 4. Clarification Route Tests ─────────────────────────────────────────────

def test_clarification_empty_query(router):
    """Empty and whitespace queries trigger CLARIFICATION."""
    for q in ["", "   ", "\t\n"]:
        decision = router.classify(q)
        assert decision.route_type == RouteType.CLARIFICATION
        assert decision.clarification_required is True
        assert decision.clarification_question is not None


def test_clarification_status_missing_id(router):
    """Asking status without ticket or work order ID triggers CLARIFICATION."""
    queries = [
        "ตั๋วสถานะเป็นอะไร",
        "What is the status of my ticket please?",
        "งานนี้เสร็จหรือยัง"
    ]
    for q in queries:
        decision = router.classify(q)
        assert decision.route_type == RouteType.CLARIFICATION
        assert decision.normalized_intent == IntentType.CLARIFICATION_MISSING_ID
        assert decision.clarification_required is True


def test_clarification_vague_operational_complaint(router):
    """Vague complaints without company or circuit reference trigger CLARIFICATION."""
    queries = [
        "เน็ตช้ามาก ใช้งานไม่ได้ ช่วยดูให้หน่อยครับ",
        "Internet is extremely slow and connection is down, please check urgently"
    ]
    for q in queries:
        decision = router.classify(q)
        assert decision.route_type == RouteType.CLARIFICATION
        assert decision.normalized_intent == IntentType.CLARIFICATION_AMBIGUOUS_QUERY
        assert decision.clarification_required is True


def test_clarification_count_missing_customer(router):
    """Counting tickets without customer name triggers CLARIFICATION."""
    query = "ขอทราบจำนวนตั๋วที่มีปัญหาหน่อยครับ"
    decision = router.classify(query)
    assert decision.route_type == RouteType.CLARIFICATION
    assert decision.normalized_intent == IntentType.CLARIFICATION_MISSING_CUSTOMER
    assert decision.clarification_required is True


# ─── 5. Unsupported & Safety Defense Tests ────────────────────────────────────

def test_unsupported_out_of_scope(router):
    """Out-of-scope queries route to UNSUPPORTED."""
    queries = [
        "พยากรณ์อากาศที่กรุงเทพวันนี้ฝนจะตกไหม",
        "Who won the UEFA Champions League?"
    ]
    for q in queries:
        decision = router.classify(q)
        assert decision.route_type == RouteType.UNSUPPORTED
        assert decision.normalized_intent == IntentType.UNSUPPORTED_OUT_OF_SCOPE


def test_unsupported_sql_injection_defense(router):
    """Malicious SQL injection attempts are intercepted and routed to UNSUPPORTED."""
    injection_queries = [
        "SELECT * FROM trouble_tickets WHERE 1=1; DROP TABLE customers; --",
        "INC-2025-00001' UNION SELECT * FROM users --",
        "WO-700001; EXEC xp_cmdshell('dir');"
    ]
    for q in injection_queries:
        decision = router.classify(q)
        assert decision.route_type == RouteType.UNSUPPORTED
        assert decision.normalized_intent == IntentType.UNSUPPORTED_MALICIOUS_SQL

        # Verify dispatch rejects execution
        res = router.dispatch(q)
        assert res.success is False
        assert res.retrieval_results is None


# ─── 6. Determinism & Isolation Tests ─────────────────────────────────────────

def test_router_determinism(router):
    """Repeated classification of identical queries produces 100% identical outputs."""
    query = "Ticket INC-2025-00001 สถานะเป็นอะไร"
    first = router.classify(query)
    for _ in range(50):
        repeat = router.classify(query)
        assert repeat.route_type == first.route_type
        assert repeat.normalized_intent == first.normalized_intent
        assert repeat.confidence_score == first.confidence_score
        assert repeat.routing_reason == first.routing_reason


def test_safe_error_containment(router):
    """Non-existent IDs return clean structured results without exceptions."""
    res_ticket = router.dispatch("Ticket INC-2025-99999 สถานะเป็นอะไร")
    assert res_ticket.success is True
    assert res_ticket.retrieval_results["found"] is False

    res_wo = router.dispatch("Work Order WO-709999 สาเหตุคืออะไร")
    assert res_wo.success is True
    assert res_wo.retrieval_results["found"] is False
