"""
NOC Intelligence — LLM Providers Module
"""

from typing import Optional

from src.synthesis.models import ProviderConfig
from .base import BaseLLMProvider, LLMProviderError, ProviderUnavailableError
from .fake import DeterministicFakeProvider
from .ollama import OllamaProvider


def get_provider(
    provider_name: str = "ollama",
    config: Optional[ProviderConfig] = None
) -> BaseLLMProvider:
    """Factory creating configured LLM provider instance."""
    cfg = config or ProviderConfig(provider_name=provider_name)
    if provider_name.lower() == "fake":
        return DeterministicFakeProvider(config=cfg)
    elif provider_name.lower() == "ollama":
        return OllamaProvider(config=cfg)
    else:
        raise ValueError(
            f"Unsupported or unauthorized provider '{provider_name}'. Supported: 'ollama', 'fake'."
        )


__all__ = [
    "BaseLLMProvider",
    "LLMProviderError",
    "ProviderUnavailableError",
    "DeterministicFakeProvider",
    "OllamaProvider",
    "get_provider"
]
