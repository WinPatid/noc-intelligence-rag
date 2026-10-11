"""
NOC Intelligence — Stage 9 Presentation Adapter & Safe Presentation Boundary
Enforces strict allowlisting between internal pipeline structures and client UI.

CONTRACT:
- Allowed user-facing fields:
  * final_answer (only when is_safe_to_present is strictly True)
  * citations (only sanitized CitationItem -> SafeCitationView)
  * warnings (sanitized warning strings)
  * is_safe_to_present (boolean safety gate)
  * route_type (optional human-readable route string)
  * intent (optional human-readable intent string)
  * execution_time_ms (actual single-query execution duration)
  * status (high-level verification status)

- Forbidden internal fields (NEVER accessible, NEVER exposed):
  * original_answer
  * claims
  * contradictions
  * unverified_claims
  * masked_identifiers_map
  * raw SQL records / row factories
  * raw ChromaDB metadata containing unmasked operational identifiers
  * raw LLM provider payloads

FAIL-CLOSED BEHAVIOR:
If is_safe_to_present is missing (None), non-boolean, or False:
- final_answer is completely suppressed and replaced with a safe fallback message.
- citations are emptied.
- is_safe_to_present is strictly set to False.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import re

# Standard safe fallback strings (bilingual)
SAFE_FALLBACK_TH = (
    "⚠️ [ระบบความปลอดภัยของข้อมูล] ขออภัยครับ ระบบไม่สามารถแสดงผลคำตอบได้ "
    "เนื่องจากผลลัพธ์ไม่ผ่านเกณฑ์การตรวจสอบความถูกต้องของข้อมูล (Safe Guardrail Suppression)"
)
SAFE_FALLBACK_EN = (
    "⚠️ [Data Safety Guardrail] The response could not be presented because "
    "it did not pass integrity verification standards (Safe Guardrail Suppression)."
)

SAFE_ERROR_TH = "เกิดข้อผิดพลาดในการประมวลผลคำตอบ กรุณาลองใหม่อีกครั้งครับ"
SAFE_ERROR_EN = "An error occurred while processing the request. Please try again."

FORBIDDEN_FIELDS = frozenset({
    "original_answer",
    "claims",
    "contradictions",
    "unverified_claims",
    "masked_identifiers_map",
    "raw_rows",
    "evidence_used",
    "prompt",
    "messages",
})

# Operational regexes for defense-in-depth sanitization
RE_RAW_TICKET = re.compile(r"\bINC-2025-\d{5}\b")
RE_RAW_WO = re.compile(r"\bWO-70\d{4}\b")
RE_RAW_CKT = re.compile(r"\bCKT-88\d{4}\b")


@dataclass(frozen=True)
class SafeCitationView:
    """Safe, allowlisted view of a retrieved evidence citation."""
    source_id: str
    identifier: str
    label: str
    entity_type: str
    is_verified: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "identifier": self.identifier,
            "label": self.label,
            "entity_type": self.entity_type,
            "is_verified": self.is_verified,
        }


@dataclass(frozen=True)
class SafePresentationView:
    """
    Allowlist-only presentation model.
    Internal diagnostic fields (original_answer, claims, contradictions, etc.)
    do not exist on this object by design.
    """
    final_answer: str
    citations: List[SafeCitationView]
    warnings: List[str]
    is_safe_to_present: bool
    status: str
    execution_time_ms: float
    route_type: Optional[str] = None
    intent: Optional[str] = None
    language: str = "TH"
    masking_enabled: bool = True

    def __getattr__(self, name: str) -> Any:
        if name in FORBIDDEN_FIELDS:
            raise AttributeError(
                f"Access to internal diagnostic field '{name}' is strictly prohibited "
                f"by the Safe Presentation Contract."
            )
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def to_dict(self) -> Dict[str, Any]:
        """Returns strictly allowlisted dictionary representation."""
        return {
            "final_answer": self.final_answer,
            "citations": [c.to_dict() for c in self.citations],
            "warnings": list(self.warnings),
            "is_safe_to_present": self.is_safe_to_present,
            "status": self.status,
            "execution_time_ms": round(self.execution_time_ms, 2),
            "route_type": self.route_type,
            "intent": self.intent,
            "language": self.language,
            "masking_enabled": self.masking_enabled,
        }

    def get_safe_diagnostics(self) -> Dict[str, Any]:
        """
        Returns only safe, non-sensitive telemetry metadata for UI display.
        No raw identifiers, records, or internal dictionaries are included.
        """
        return {
            "Route Type": self.route_type or "UNKNOWN",
            "Intent": self.intent or "UNKNOWN",
            "Language": self.language,
            "Verification Status": self.status,
            "Safe to Present": "Yes" if self.is_safe_to_present else "No (Suppressed)",
            "Execution Latency": f"{round(self.execution_time_ms, 2)} ms",
            "Verified Citations Count": len(self.citations),
            "Identifier Masking Active": "Yes" if self.masking_enabled else "Disabled",
        }


class PresentationAdapter:
    """
    Adapter boundary between GuardrailPipeline / GuardrailResult and client presentation layer.
    Guarantees that unverified or unsafe results fail closed, and internal diagnostic
    fields cannot leak to the UI.
    """

    @classmethod
    def adapt(
        cls,
        result: Any,
        route_type: Optional[str] = None,
        intent: Optional[str] = None,
        language_hint: Optional[str] = None,
        masking_enabled: bool = True,
    ) -> SafePresentationView:
        """
        Transforms a pipeline result (GuardrailResult or compatible object)
        into a SafePresentationView.

        Fails closed on missing or False safety flags.
        """
        # Determine language for safe fallback messaging
        is_thai = True
        if language_hint:
            is_thai = (language_hint.upper() == "TH")
        elif hasattr(result, "final_answer") and isinstance(result.final_answer, str):
            is_thai = any("\u0e00" <= c <= "\u0e7f" for c in result.final_answer)

        # 1. Validate safety flag (Strict Fail-Closed)
        is_safe = getattr(result, "is_safe_to_present", None)
        if not isinstance(is_safe, bool) or not is_safe:
            # FAILED CLOSED: Suppress answer and strip citations
            status_str = getattr(result, "status", "UNSAFE_SUPPRESSED")
            if hasattr(status_str, "value"):
                status_str = status_str.value
            elif not isinstance(status_str, str):
                status_str = "UNSAFE_SUPPRESSED"

            raw_time = getattr(result, "execution_time_ms", 0.0)
            exec_time = float(raw_time) if isinstance(raw_time, (int, float)) else 0.0

            # Collect only safe, sanitized warnings
            sanitized_warnings = cls._sanitize_warnings(
                getattr(result, "warnings", []), masking_enabled=masking_enabled
            )

            fallback_msg = SAFE_FALLBACK_TH if is_thai else SAFE_FALLBACK_EN

            return SafePresentationView(
                final_answer=fallback_msg,
                citations=[],  # Never leak citations on suppressed answers
                warnings=sanitized_warnings,
                is_safe_to_present=False,
                status=str(status_str),
                execution_time_ms=exec_time,
                route_type=route_type,
                intent=intent,
                language="TH" if is_thai else "EN",
                masking_enabled=masking_enabled,
            )

        # 2. Approved presentation path (is_safe_to_present == True)
        final_answer = getattr(result, "final_answer", "")
        if not isinstance(final_answer, str):
            final_answer = SAFE_ERROR_TH if is_thai else SAFE_ERROR_EN

        status_val = getattr(result, "status", "VERIFIED")
        if hasattr(status_val, "value"):
            status_str = status_val.value
        else:
            status_str = str(status_val)

        raw_time = getattr(result, "execution_time_ms", 0.0)
        exec_time = float(raw_time) if isinstance(raw_time, (int, float)) else 0.0

        # Build sanitized citation views
        raw_citations = getattr(result, "citations", [])
        safe_citations: List[SafeCitationView] = []
        if isinstance(raw_citations, list):
            for c in raw_citations:
                safe_c = cls._adapt_citation(c)
                if safe_c is not None:
                    safe_citations.append(safe_c)

        # Sanitize warnings
        sanitized_warnings = cls._sanitize_warnings(
            getattr(result, "warnings", []), masking_enabled=masking_enabled
        )

        # Defense-in-depth: If masking is enabled, check if raw unmasked identifiers leak into final_answer
        if masking_enabled:
            # Masking contract checks
            if RE_RAW_TICKET.search(final_answer) or RE_RAW_WO.search(final_answer):
                # Leaked unmasked operational ID: fail closed immediately!
                return SafePresentationView(
                    final_answer=(
                        "⚠️ [ระบบความปลอดภัยของข้อมูล] ตรวจพบข้อมูลระบุตัวตนที่ไม่ได้ถูกปิดบัง "
                        "ระบบจึงระงับการแสดงผลเพื่อความปลอดภัย (Fail-Closed on Unmasked ID)"
                        if is_thai else
                        "⚠️ [Data Safety Guardrail] Unmasked operational identifier detected. "
                        "Response suppressed to prevent data leakage (Fail-Closed on Unmasked ID)."
                    ),
                    citations=[],
                    warnings=["Masking violation detected: raw identifier leaked into final answer."],
                    is_safe_to_present=False,
                    status="MASKING_VIOLATION",
                    execution_time_ms=exec_time,
                    route_type=route_type,
                    intent=intent,
                    language="TH" if is_thai else "EN",
                    masking_enabled=masking_enabled,
                )

        return SafePresentationView(
            final_answer=final_answer,
            citations=safe_citations,
            warnings=sanitized_warnings,
            is_safe_to_present=True,
            status=status_str,
            execution_time_ms=exec_time,
            route_type=route_type,
            intent=intent,
            language="TH" if is_thai else "EN",
            masking_enabled=masking_enabled,
        )

    @classmethod
    def _adapt_citation(cls, raw_c: Any) -> Optional[SafeCitationView]:
        """Converts raw citation object to SafeCitationView with allowlisted fields."""
        if not raw_c:
            return None

        source_id = getattr(raw_c, "source_id", "")
        identifier = getattr(raw_c, "identifier", "")
        label = getattr(raw_c, "label", "")
        entity_type = getattr(raw_c, "entity_type", "reference")
        is_verified = getattr(raw_c, "is_verified", False)

        return SafeCitationView(
            source_id=str(source_id),
            identifier=str(identifier),
            label=str(label),
            entity_type=str(entity_type),
            is_verified=bool(is_verified),
        )

    @classmethod
    def _sanitize_warnings(cls, warnings: Any, masking_enabled: bool = True) -> List[str]:
        """Strips raw internal references from warning strings."""
        if not isinstance(warnings, (list, tuple)):
            return []

        sanitized: List[str] = []
        for w in warnings:
            if not isinstance(w, str) or not w.strip():
                continue
            cleaned = w.strip()
            # If masking enabled, sanitize any residual raw IDs in warning texts
            if masking_enabled:
                cleaned = RE_RAW_TICKET.sub("[TICKET-***]", cleaned)
                cleaned = RE_RAW_WO.sub("[WO-***]", cleaned)
                cleaned = RE_RAW_CKT.sub("[CIRCUIT-***]", cleaned)
            sanitized.append(cleaned)
        return sanitized
