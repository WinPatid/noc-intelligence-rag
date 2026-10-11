"""
NOC Intelligence — Stage 5 Grounded LLM Synthesis Subsystem
"""

from .models import (
    GenerationStatus,
    CitationItem,
    ProviderConfig,
    SynthesisResponse
)
from .providers import (
    BaseLLMProvider,
    LLMProviderError,
    ProviderUnavailableError,
    DeterministicFakeProvider,
    OllamaProvider,
    get_provider
)
from .context_builder import EvidenceContextBuilder, sanitize_text
from .synthesizer import GroundedSynthesizer

__all__ = [
    "GenerationStatus",
    "CitationItem",
    "ProviderConfig",
    "SynthesisResponse",
    "BaseLLMProvider",
    "LLMProviderError",
    "ProviderUnavailableError",
    "DeterministicFakeProvider",
    "OllamaProvider",
    "get_provider",
    "EvidenceContextBuilder",
    "sanitize_text",
    "GroundedSynthesizer"
]
