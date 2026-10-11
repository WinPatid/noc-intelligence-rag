"""
NOC Intelligence — Stage 6 Guardrail Data Models & Contracts
Defines typed models for claim verification, contradiction reports,
masking configurations, and final guardrail results.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from src.synthesis.models import SynthesisResponse, CitationItem


class GuardrailStatus(str, Enum):
    """Integrity status determined by the Guardrail Critic."""
    VERIFIED = "VERIFIED"
    PARTIALLY_VERIFIED = "PARTIALLY_VERIFIED"
    CONTRADICTED = "CONTRADICTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"
    INVALID_RESPONSE = "INVALID_RESPONSE"


class ClaimType(str, Enum):
    """Category of factual claim extracted from generated response."""
    STATUS = "STATUS"
    SEVERITY = "SEVERITY"
    ROOT_CAUSE = "ROOT_CAUSE"
    CUSTOMER_NAME = "CUSTOMER_NAME"
    CIRCUIT_REF = "CIRCUIT_REF"
    TICKET_ID = "TICKET_ID"
    WORK_ORDER_ID = "WORK_ORDER_ID"
    LINKED_ENTITY = "LINKED_ENTITY"
    DATE = "DATE"
    NUMERIC = "NUMERIC"


@dataclass
class ClaimVerificationItem:
    """Detailed verification record for a specific factual claim."""
    claim_type: ClaimType
    claimed_value: str
    evidence_value: Optional[str]
    is_verified: bool
    is_contradicted: bool
    source_id: Optional[str] = None
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_type": self.claim_type.value,
            "claimed_value": self.claimed_value,
            "evidence_value": self.evidence_value,
            "is_verified": self.is_verified,
            "is_contradicted": self.is_contradicted,
            "source_id": self.source_id,
            "notes": self.notes
        }


@dataclass
class MaskingConfig:
    """Configuration for deterministic operational identifier masking."""
    enabled: bool = False
    mask_tickets: bool = True
    mask_work_orders: bool = True
    mask_circuits: bool = True
    mask_customers: bool = True
    preserve_citation_mapping: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "mask_tickets": self.mask_tickets,
            "mask_work_orders": self.mask_work_orders,
            "mask_circuits": self.mask_circuits,
            "mask_customers": self.mask_customers,
            "preserve_citation_mapping": self.preserve_citation_mapping
        }


@dataclass
class GuardrailResult:
    """The final response contract produced after guardrail critique and masking."""
    status: GuardrailStatus
    original_answer: str
    final_answer: str
    claims: List[ClaimVerificationItem] = field(default_factory=list)
    contradictions: List[str] = field(default_factory=list)
    unverified_claims: List[str] = field(default_factory=list)
    masked_identifiers_map: Dict[str, str] = field(default_factory=dict)
    citations: List[CitationItem] = field(default_factory=list)
    is_safe_to_present: bool = True
    warnings: List[str] = field(default_factory=list)
    execution_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "original_answer": self.original_answer,
            "final_answer": self.final_answer,
            "claims": [c.to_dict() for c in self.claims],
            "contradictions": self.contradictions,
            "unverified_claims": self.unverified_claims,
            "masked_identifiers_map": self.masked_identifiers_map,
            "citations": [c.to_dict() for c in self.citations],
            "is_safe_to_present": self.is_safe_to_present,
            "warnings": self.warnings,
            "execution_time_ms": round(self.execution_time_ms, 2)
        }
