"""
NOC Intelligence — Local Ollama LLM Provider
Connects to a local Ollama daemon (default: http://127.0.0.1:11434).
Zero external cloud or network requests.
"""

import json
import urllib.request
import urllib.error
from typing import Optional

from src.synthesis.models import ProviderConfig
from src.synthesis.providers.base import (
    BaseLLMProvider,
    ProviderUnavailableError,
    LLMProviderError
)


class OllamaProvider(BaseLLMProvider):
    """
    Communicates with local Ollama instance via HTTP API.
    Guarantees no outbound internet/cloud requests.
    """

    def __init__(self, config: Optional[ProviderConfig] = None):
        cfg = config or ProviderConfig(
            provider_name="ollama",
            model_name="typhoon2:8b",
            temperature=0.0,
            endpoint_url="http://127.0.0.1:11434"
        )
        super().__init__(cfg)

    def is_available(self) -> bool:
        """Verifies that local Ollama is active and the configured model is installed."""
        url = f"{self.config.endpoint_url}/api/tags"
        req = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=2.5) as resp:
                if resp.status != 200:
                    return False
                data = json.loads(resp.read().decode("utf-8"))
                models = [m.get("name") for m in data.get("models", [])]
                # Check exact or prefix match (e.g., "typhoon2:8b" in ["typhoon2:8b", "typhoon2:latest"])
                target = self.config.model_name
                return any(target in m or m in target for m in models)
        except Exception:
            return False

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        override_config: Optional[ProviderConfig] = None
    ) -> str:
        cfg = override_config or self.config
        url = f"{cfg.endpoint_url}/api/generate"

        payload = {
            "model": cfg.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": cfg.temperature,
                "num_predict": cfg.max_tokens
            }
        }
        if system_prompt:
            payload["system"] = system_prompt

        req_data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=req_data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=cfg.timeout_seconds) as resp:
                if resp.status != 200:
                    raise LLMProviderError(f"Ollama returned HTTP error code {resp.status}")
                raw_body = resp.read().decode("utf-8")
                result_json = json.loads(raw_body)
                return result_json.get("response", "").strip()

        except urllib.error.URLError as e:
            raise ProviderUnavailableError(
                f"Failed to connect to local Ollama server at {cfg.endpoint_url}: {e.reason}"
            )
        except TimeoutError:
            raise ProviderUnavailableError(
                f"Ollama generation timed out after {cfg.timeout_seconds} seconds."
            )
        except Exception as e:
            raise LLMProviderError(f"Ollama generation failure: {str(e)}")
