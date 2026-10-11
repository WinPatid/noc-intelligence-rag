"""
NOC Intelligence — Stage 6 Contradiction Detection Rules
Deterministic verification rules detecting material factual conflicts
between generated claims and structured database evidence.
"""

from typing import Optional, List, Dict, Any


def check_status_contradiction(
    claimed_status: str,
    evidence_status: str
) -> Optional[str]:
    """Detects mutually exclusive status claims."""
    c_norm = claimed_status.strip().upper()
    e_norm = evidence_status.strip().upper()

    # Terminal vs Active statuses
    active_states = {"OPEN", "IN_PROGRESS", "กำลังดำเนินการ", "เปิด"}
    resolved_states = {"RESOLVED", "COMPLETED", "CLOSED", "แก้ไขแล้ว", "เสร็จสิ้น", "ปิดงาน"}

    if e_norm in resolved_states and any(a in c_norm for a in active_states):
        return f"Status contradiction: claim asserts active state '{claimed_status}', but evidence confirms resolved/closed '{evidence_status}'."

    if e_norm in active_states and any(r in c_norm for r in resolved_states):
        return f"Status contradiction: claim asserts resolved state '{claimed_status}', but evidence confirms active/in-progress '{evidence_status}'."

    if c_norm != e_norm and c_norm in {"OPEN", "RESOLVED", "CLOSED", "IN_PROGRESS"}:
        return f"Status mismatch: claimed '{claimed_status}' differs from verified record '{evidence_status}'."

    return None


def check_severity_contradiction(
    claimed_severity: str,
    evidence_severity: str
) -> Optional[str]:
    """Detects mismatched incident severity."""
    c_norm = claimed_severity.strip().upper()
    e_norm = evidence_severity.strip().upper()

    if "P1" in e_norm and ("P2" in c_norm or "P3" in c_norm):
        return f"Severity contradiction: claim asserts '{claimed_severity}', but verified record is critical '{evidence_severity}'."

    if "P3" in e_norm and ("P1" in c_norm or "P2" in c_norm):
        return f"Severity contradiction: claim asserts '{claimed_severity}', but verified record is minor '{evidence_severity}'."

    return None


def check_root_cause_contradiction(
    claimed_cause: str,
    evidence_cause: str
) -> Optional[str]:
    """Detects mismatched root cause category assertions."""
    c_norm = claimed_cause.strip().upper()
    e_norm = evidence_cause.strip().upper()

    canonical_causes = [
        "FIBER_CUT", "COMMERCIAL_POWER", "OPTICAL_TRANSCEIVER",
        "CONFIGURATION_ERROR", "HARDWARE_FAILURE", "ENVIRONMENTAL", "CONGESTION"
    ]

    for cause in canonical_causes:
        if cause in e_norm and cause not in c_norm:
            # Check if claim asserted a DIFFERENT known category
            other_causes = [other for other in canonical_causes if other != cause]
            for other in other_causes:
                if other in c_norm:
                    return f"Root cause contradiction: claim asserts '{other}', but verified record is '{cause}'."

    return None


def check_linkage_contradiction(
    claimed_wo_id: str,
    linked_wo_ids: List[str]
) -> Optional[str]:
    """Detects hallucinated or incorrect work order bridge links."""
    c_id = claimed_wo_id.strip().upper()
    known_links = [w.strip().upper() for w in linked_wo_ids]

    if known_links and c_id not in known_links:
        return f"Bridge linkage contradiction: work order '{c_id}' is not linked to this ticket in database records (valid links: {known_links})."

    return None


def check_customer_association_contradiction(
    claimed_customer: str,
    evidence_customer: str
) -> Optional[str]:
    """Detects assigning an incident to the wrong corporate account."""
    c_norm = claimed_customer.strip().lower()
    e_norm = evidence_customer.strip().lower()

    if c_norm and e_norm and c_norm not in e_norm and e_norm not in c_norm:
        return f"Customer contradiction: claim associates incident with '{claimed_customer}', but evidence belongs to '{evidence_customer}'."

    return None
