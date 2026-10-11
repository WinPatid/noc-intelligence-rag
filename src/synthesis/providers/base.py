"""
NOC Intelligence — Base LLM Provider Interface
Abstract base class defining the provider-independent generation contract.
"""

from abc import ABC, abstractmethod
from typing import Optional

from src.synthesis.models import ProviderConfig


class LLMProviderError(Exception):
    """Base exception for provider generation failures."""
    pass


class ProviderUnavailableError(LLMProviderError):
    """Raised when the requested provider service (e.g. local Ollama) is unreachable."""
    pass


class BaseLLMProvider(ABC):
    """
    Abstract interface for LLM text generation.
    Decouples synthesis logic from specific model runtimes.
    """

    def __init__(self, config: Optional[ProviderConfig] = None):
        self.config = config or ProviderConfig()

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        override_config: Optional[ProviderConfig] = None
    ) -> str:
        """Generates text completion synchronously."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Checks whether the provider service and model are reachable."""
        pass
