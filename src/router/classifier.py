"""
NOC Intelligence — Stage 4 Deterministic Intent Classifier
Implements rule-based intent classification and routing supervisor logic.
Strict precedence ordering ensures 100% deterministic routing decisions.
"""

from typing import Optional

from src.router.models import (
    RouteType,
    IntentType,
    ExtractedEntities,
    RoutingDecision
)
from src.router.entity_extractor import EntityExtractor
from src.router.patterns import RE_VAGUE_COMPLAINT, RE_STATUS_INQUIRY, RE_GREETING


class DeterministicClassifier:
    """
    Deterministic rule-based classifier for classifying incoming NOC user queries.
    Zero probabilistic / random behaviors. Identical input guarantees identical output.
    """

    def __init__(self, extractor: Optional[EntityExtractor] = None, db_path: Optional[str] = None):
        self.extractor = extractor or EntityExtractor(db_path=db_path)

    def classify(self, query: str) -> RoutingDecision:
        cleaned = query.strip() if query else ""

        # ── 1. Empty / Whitespace Query ───────────────────────────────────────
        if not cleaned:
            return RoutingDecision(
                route_type=RouteType.CLARIFICATION,
                normalized_intent=IntentType.CLARIFICATION_AMBIGUOUS_QUERY,
                extracted_entities=ExtractedEntities(),
                confidence_score=1.0,
                routing_reason="Empty or whitespace-only query received.",
                clarification_required=True,
                clarification_question="กรุณาระบุคำถาม หมายเลขตั๋ว (INC-2025-XXXXX) หรือใบงาน (WO-70XXXX) ที่ต้องการตรวจสอบครับ"
            )

        # Extract entities
        entities = self.extractor.extract(cleaned)

        # ── 2. Malicious SQL Injection Defense ─────────────────────────────────
        if entities.has_sql_injection:
            return RoutingDecision(
                route_type=RouteType.UNSUPPORTED,
                normalized_intent=IntentType.UNSUPPORTED_MALICIOUS_SQL,
                extracted_entities=entities,
                confidence_score=1.0,
                routing_reason="Query contains dangerous SQL manipulation tokens or command injection patterns.",
                clarification_required=False,
                clarification_question=None
            )

        # ── 3. Explicit Ticket Identifier (INC-2025-XXXXX) ─────────────────────
        if entities.ticket_ids:
            tid = entities.ticket_ids[0]
            if entities.is_bridge_query:
                return RoutingDecision(
                    route_type=RouteType.SQL_LOOKUP,
                    normalized_intent=IntentType.BRIDGE_TICKET_TO_WO,
                    extracted_entities=entities,
                    confidence_score=1.0,
                    routing_reason=f"Direct bridge traversal: retrieving linked work orders for ticket {tid}.",
                    clarification_required=False
                )
            return RoutingDecision(
                route_type=RouteType.SQL_LOOKUP,
                normalized_intent=IntentType.TICKET_STATUS,
                extracted_entities=entities,
                confidence_score=1.0,
                routing_reason=f"Direct SQL lookup: retrieving status and details for ticket {tid}.",
                clarification_required=False
            )

        # ── 4. Explicit Work Order Identifier (WO-70XXXX) ──────────────────────
        if entities.work_order_ids:
            wo_id = entities.work_order_ids[0]
            if entities.is_reverse_bridge:
                return RoutingDecision(
                    route_type=RouteType.SQL_LOOKUP,
                    normalized_intent=IntentType.BRIDGE_WO_TO_TICKETS,
                    extracted_entities=entities,
                    confidence_score=1.0,
                    routing_reason=f"Reverse bridge traversal: retrieving affected customer tickets for work order {wo_id}.",
                    clarification_required=False
                )
            return RoutingDecision(
                route_type=RouteType.SQL_LOOKUP,
                normalized_intent=IntentType.WORK_ORDER_DETAIL,
                extracted_entities=entities,
                confidence_score=1.0,
                routing_reason=f"Direct SQL lookup: retrieving technical cause and resolution for work order {wo_id}.",
                clarification_required=False
            )

        # ── 5. Explicit Circuit Identifier (CKT-88XXXX) ────────────────────────
        if entities.circuit_refs:
            ckt = entities.circuit_refs[0]
            if entities.is_semantic_query:
                return RoutingDecision(
                    route_type=RouteType.HYBRID_SEARCH,
                    normalized_intent=IntentType.HYBRID_CIRCUIT_INCIDENTS,
                    extracted_entities=entities,
                    confidence_score=0.95,
                    routing_reason=f"Hybrid search: combining circuit {ckt} filter with semantic symptoms: {', '.join(entities.semantic_terms)}.",
                    clarification_required=False
                )
            return RoutingDecision(
                route_type=RouteType.SQL_LOOKUP,
                normalized_intent=IntentType.CIRCUIT_DETAIL,
                extracted_entities=entities,
                confidence_score=1.0,
                routing_reason=f"Direct SQL lookup: retrieving circuit {ckt} configuration, customer, and site location.",
                clarification_required=False
            )

        # ── 6. Matched Customer Entity ─────────────────────────────────────────
        if entities.matched_customer:
            cust_name = entities.customer_name_raw or entities.matched_customer.get("company_name_en")

            if entities.is_semantic_query:
                return RoutingDecision(
                    route_type=RouteType.HYBRID_SEARCH,
                    normalized_intent=IntentType.HYBRID_CUSTOMER_INCIDENTS,
                    extracted_entities=entities,
                    confidence_score=0.95,
                    routing_reason=f"Hybrid search: customer {cust_name} scoped with semantic symptoms: {', '.join(entities.semantic_terms)}.",
                    clarification_required=False
                )
            elif entities.is_inventory_query:
                return RoutingDecision(
                    route_type=RouteType.SQL_LOOKUP,
                    normalized_intent=IntentType.CUSTOMER_CIRCUITS,
                    extracted_entities=entities,
                    confidence_score=1.0,
                    routing_reason=f"SQL lookup: listing circuits registered to customer {cust_name}.",
                    clarification_required=False
                )
            else:
                # Default for customer inquiry is ticket count / overview
                return RoutingDecision(
                    route_type=RouteType.SQL_LOOKUP,
                    normalized_intent=IntentType.CUSTOMER_TICKET_COUNT,
                    extracted_entities=entities,
                    confidence_score=1.0,
                    routing_reason=f"SQL aggregation: counting tickets for customer {cust_name} (status filter: {entities.status_filter or 'ALL'}).",
                    clarification_required=False
                )

        # ── 7. Semantic Query (No customer/ticket scope) ───────────────────────
        if entities.is_semantic_query:
            # Check if asking specifically about resolution actions
            is_resolution = any(
                term in cleaned.lower()
                for term in [
                    "แก้ไขอย่างไร", "วิธีแก้", "action taken", "remediation", "เปลี่ยนอุปกรณ์",
                    "ขั้นตอน", "ซ่อมแซม", "resolve", "resolved", "solution", "fix",
                    "mitigation", "mitigate"
                ]
            )
            intent = IntentType.SEMANTIC_RESOLUTION if is_resolution else IntentType.SEMANTIC_SYMPTOM
            return RoutingDecision(
                route_type=RouteType.SEMANTIC_SEARCH,
                normalized_intent=intent,
                extracted_entities=entities,
                confidence_score=0.90,
                routing_reason=f"Dense semantic vector search for operational symptom terms: {', '.join(entities.semantic_terms)}.",
                clarification_required=False
            )

        # ── 8. Clarification Required ──────────────────────────────────────────
        if entities.is_clarification_trigger:
            if RE_VAGUE_COMPLAINT.search(cleaned):
                return RoutingDecision(
                    route_type=RouteType.CLARIFICATION,
                    normalized_intent=IntentType.CLARIFICATION_AMBIGUOUS_QUERY,
                    extracted_entities=entities,
                    confidence_score=1.0,
                    routing_reason="Vague operational symptom reported without customer identity or circuit SID.",
                    clarification_required=True,
                    clarification_question="กรุณาระบุชื่อบริษัทหรือรหัสวงจร (CKT-88XXXX) เพื่อให้ศูนย์ NOC ตรวจสอบระบบได้ถูกต้องครับ"
                )
            elif RE_STATUS_INQUIRY.search(cleaned):
                return RoutingDecision(
                    route_type=RouteType.CLARIFICATION,
                    normalized_intent=IntentType.CLARIFICATION_MISSING_ID,
                    extracted_entities=entities,
                    confidence_score=1.0,
                    routing_reason="Status inquiry missing ticket ID or work order ID.",
                    clarification_required=True,
                    clarification_question="กรุณาระบุหมายเลขตั๋วปัญหา (INC-2025-XXXXX) หรือใบงาน (WO-70XXXX) ที่ต้องการตรวจสอบสถานะครับ"
                )
            elif entities.is_count_query:
                return RoutingDecision(
                    route_type=RouteType.CLARIFICATION,
                    normalized_intent=IntentType.CLARIFICATION_MISSING_CUSTOMER,
                    extracted_entities=entities,
                    confidence_score=1.0,
                    routing_reason="Ticket count requested without specifying customer.",
                    clarification_required=True,
                    clarification_question="กรุณาระบุชื่อบริษัทหรือรหัสลูกค้าที่ต้องการนับจำนวนตั๋วครับ"
                )

        # ── 8b. Conversational Greeting or Capability Inquiries ────────────────
        if RE_GREETING.search(cleaned):
            is_th = any("\u0e00" <= c <= "\u0e7f" for c in cleaned)
            return RoutingDecision(
                route_type=RouteType.CLARIFICATION,
                normalized_intent=IntentType.CLARIFICATION_GREETING,
                extracted_entities=entities,
                confidence_score=1.0,
                routing_reason="Conversational greeting or capability inquiry.",
                clarification_required=True,
                clarification_question=(
                    "สวัสดีครับ ผมคือผู้ช่วยอัจฉริยะศูนย์ปฏิบัติการเครือข่าย (NOC Intelligence) คุณสามารถสอบถามข้อมูลสถานะตั๋วปัญหา (INC-2025-XXXXX), รายละเอียดการซ่อมใบงานช่าง (WO-70XXXX), ข้อมูลวงจร (CKT-88XXXX), ยอดตั๋วลูกค้า หรือค้นหาสาเหตุและอาการขัดข้องของระบบเครือข่ายได้ครับ"
                    if is_th else
                    "Hello! I am NOC Intelligence, an AI-powered Network Operations Center assistant. You can ask me about trouble ticket statuses (INC-2025-XXXXX), work order resolutions (WO-70XXXX), leased circuits (CKT-88XXXX), customer ticket counts, or technical network symptoms."
                )
            )

        # ── 9. Fallback / Out of Domain ────────────────────────────────────────
        return RoutingDecision(
            route_type=RouteType.UNSUPPORTED,
            normalized_intent=IntentType.UNSUPPORTED_OUT_OF_SCOPE,
            extracted_entities=entities,
            confidence_score=1.0,
            routing_reason="Query is out of scope for the NOC Intelligence telecommunications operations domain.",
            clarification_required=False,
            clarification_question=None
        )
