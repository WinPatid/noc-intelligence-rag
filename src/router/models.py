"""
NOC Intelligence — Stage 4 Router Contracts & Data Models
Defines typed models, route enums, intent types, and routing results.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


class RouteType(str, Enum):
    """The high-level routing destination selected by the supervisor."""
    SQL_LOOKUP = "SQL_LOOKUP"
    SEMANTIC_SEARCH = "SEMANTIC_SEARCH"
    HYBRID_SEARCH = "HYBRID_SEARCH"
    CLARIFICATION = "CLARIFICATION"
    UNSUPPORTED = "UNSUPPORTED"


class IntentType(str, Enum):
    """Normalized specific intent categorized by deterministic rules."""
    # SQL Intents
    TICKET_STATUS = "TICKET_STATUS"
    WORK_ORDER_DETAIL = "WORK_ORDER_DETAIL"
    BRIDGE_TICKET_TO_WO = "BRIDGE_TICKET_TO_WO"
    BRIDGE_WO_TO_TICKETS = "BRIDGE_WO_TO_TICKETS"
    CIRCUIT_DETAIL = "CIRCUIT_DETAIL"
    CUSTOMER_TICKET_COUNT = "CUSTOMER_TICKET_COUNT"
    CUSTOMER_CIRCUITS = "CUSTOMER_CIRCUITS"
    TICKET_HISTORY = "TICKET_HISTORY"

    # Semantic Intents
    SEMANTIC_SYMPTOM = "SEMANTIC_SYMPTOM"
    SEMANTIC_RESOLUTION = "SEMANTIC_RESOLUTION"
    SEMANTIC_GENERAL = "SEMANTIC_GENERAL"

    # Hybrid Intents
    HYBRID_CUSTOMER_INCIDENTS = "HYBRID_CUSTOMER_INCIDENTS"
    HYBRID_CIRCUIT_INCIDENTS = "HYBRID_CIRCUIT_INCIDENTS"

    # Clarification Intents
    CLARIFICATION_MISSING_ID = "CLARIFICATION_MISSING_ID"
    CLARIFICATION_MISSING_CUSTOMER = "CLARIFICATION_MISSING_CUSTOMER"
    CLARIFICATION_AMBIGUOUS_QUERY = "CLARIFICATION_AMBIGUOUS_QUERY"
    CLARIFICATION_GREETING = "CLARIFICATION_GREETING"

    # Unsupported Intents
    UNSUPPORTED_OUT_OF_SCOPE = "UNSUPPORTED_OUT_OF_SCOPE"
    UNSUPPORTED_MALICIOUS_SQL = "UNSUPPORTED_MALICIOUS_SQL"


@dataclass
class ExtractedEntities:
    """Entities deterministically extracted from the query."""
    ticket_ids: List[str] = field(default_factory=list)
    work_order_ids: List[str] = field(default_factory=list)
    circuit_refs: List[str] = field(default_factory=list)
    customer_codes: List[str] = field(default_factory=list)
    customer_name_raw: Optional[str] = None
    matched_customer: Optional[Dict[str, Any]] = None
    status_filter: Optional[str] = None
    semantic_terms: List[str] = field(default_factory=list)
    is_bridge_query: bool = False
    is_reverse_bridge: bool = False
    is_count_query: bool = False
    is_inventory_query: bool = False
    is_semantic_query: bool = False
    is_clarification_trigger: bool = False
    has_sql_injection: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticket_ids": self.ticket_ids,
            "work_order_ids": self.work_order_ids,
            "circuit_refs": self.circuit_refs,
            "customer_codes": self.customer_codes,
            "customer_name_raw": self.customer_name_raw,
            "matched_customer": self.matched_customer,
            "status_filter": self.status_filter,
            "semantic_terms": self.semantic_terms,
            "is_bridge_query": self.is_bridge_query,
            "is_reverse_bridge": self.is_reverse_bridge,
            "is_count_query": self.is_count_query,
            "is_inventory_query": self.is_inventory_query,
            "is_semantic_query": self.is_semantic_query,
            "is_clarification_trigger": self.is_clarification_trigger,
            "has_sql_injection": self.has_sql_injection
        }


@dataclass
class RoutingDecision:
    """The classification decision output by DeterministicClassifier."""
    route_type: RouteType
    normalized_intent: IntentType
    extracted_entities: ExtractedEntities
    confidence_score: float
    routing_reason: str
    clarification_required: bool = False
    clarification_question: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "route_type": self.route_type.value,
            "normalized_intent": self.normalized_intent.value,
            "extracted_entities": self.extracted_entities.to_dict(),
            "confidence_score": self.confidence_score,
            "routing_reason": self.routing_reason,
            "clarification_required": self.clarification_required,
            "clarification_question": self.clarification_question
        }


@dataclass
class RoutingResult:
    """The final result after dispatching the query to its respective engine."""
    decision: RoutingDecision
    retrieval_results: Optional[Any] = None
    success: bool = True
    error_message: Optional[str] = None
    execution_time_ms: float = 0.0

    @property
    def route_type(self) -> RouteType:
        return self.decision.route_type

    @property
    def normalized_intent(self) -> IntentType:
        return self.decision.normalized_intent

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision.to_dict(),
            "retrieval_results": self.retrieval_results,
            "success": self.success,
            "error_message": self.error_message,
            "execution_time_ms": round(self.execution_time_ms, 2)
        }
