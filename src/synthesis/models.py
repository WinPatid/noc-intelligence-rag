"""
NOC Intelligence — Stage 5 Grounded Synthesis Contracts & Data Models
Defines typed models for LLM provider configs, citation items, and grounded responses.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


class GenerationStatus(str, Enum):
    """Execution status of the synthesis process."""
    SUCCESS = "SUCCESS"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    CLARIFICATION = "CLARIFICATION"
    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"
    ERROR = "ERROR"


@dataclass
class CitationItem:
    """Stable citation reference pointing to a verified retrieved evidence source."""
    source_id: str          # e.g., "ticket:INC-2025-00001" or "work_order:WO-700001"
    entity_type: str        # "trouble_ticket", "work_order", "circuit", "customer", "activity"
    identifier: str         # "INC-2025-00001", "WO-700001", "CKT-880001"
    label: str              # Human-readable title/summary
    is_verified: bool = True# True if mapped to actual retrieved evidence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "entity_type": self.entity_type,
            "identifier": self.identifier,
            "label": self.label,
            "is_verified": self.is_verified
        }


@dataclass
class ProviderConfig:
    """Configuration settings for LLM generation."""
    provider_name: str = "ollama"       # "ollama" or "fake"
    model_name: str = "typhoon2:8b"     # Default local model
    temperature: float = 0.0            # 0.0 for deterministic factual generation
    max_tokens: int = 1024
    timeout_seconds: float = 30.0
    endpoint_url: str = "http://127.0.0.1:11434"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider_name": self.provider_name,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout_seconds": self.timeout_seconds,
            "endpoint_url": self.endpoint_url
        }


@dataclass
class SynthesisResponse:
    """The complete response contract returned by the Grounded Synthesizer."""
    answer: str
    language: str                       # "TH" or "EN"
    citations: List[CitationItem] = field(default_factory=list)
    evidence_used: List[str] = field(default_factory=list)
    insufficient_evidence: bool = False
    provider: str = "ollama"
    model: str = "typhoon2:8b"
    generation_status: GenerationStatus = GenerationStatus.SUCCESS
    warnings: List[str] = field(default_factory=list)
    execution_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "language": self.language,
            "citations": [c.to_dict() for c in self.citations],
            "evidence_used": self.evidence_used,
            "insufficient_evidence": self.insufficient_evidence,
            "provider": self.provider,
            "model": self.model,
            "generation_status": self.generation_status.value,
            "warnings": self.warnings,
            "execution_time_ms": round(self.execution_time_ms, 2)
        }
