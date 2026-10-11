"""
NOC Intelligence — Stage 9 Pipeline Service Layer
Coordinates deterministic router execution, guardrail pipeline processing,
real-time execution timing, and presentation adaptation for Streamlit UI.
"""

import time
import logging
from typing import Optional, Dict, Any, Tuple

from src.router.router import DeterministicRouter
from src.router.models import RoutingResult
from src.synthesis.models import ProviderConfig
from src.synthesis.providers import (
    DeterministicFakeProvider,
    OllamaProvider,
    BaseLLMProvider,
    ProviderUnavailableError
)
from src.synthesis.synthesizer import GroundedSynthesizer
from src.guardrails.models import MaskingConfig, GuardrailResult, GuardrailStatus
from src.guardrails.pipeline import GuardrailPipeline
from src.ui.presentation_adapter import (
    PresentationAdapter,
    SafePresentationView,
    SAFE_FALLBACK_TH,
    SAFE_FALLBACK_EN,
    SAFE_ERROR_TH,
    SAFE_ERROR_EN,
)

logger = logging.getLogger("noc_intelligence.ui.pipeline_service")

# Global singleton cache for pipeline components to avoid re-instantiation overhead
_SHARED_ROUTER: Optional[DeterministicRouter] = None
_PIPELINES: Dict[Tuple[str, bool], GuardrailPipeline] = {}


def get_shared_router() -> DeterministicRouter:
    """Returns singleton DeterministicRouter instance."""
    global _SHARED_ROUTER
    if _SHARED_ROUTER is None:
        _SHARED_ROUTER = DeterministicRouter()
    return _SHARED_ROUTER


def check_ollama_availability() -> Dict[str, Any]:
    """
    Safely checks if local Ollama daemon is reachable.
    Only called when user explicitly selects Local Ollama mode.
    """
    provider = OllamaProvider()
    is_avail = provider.is_available()
    if is_avail:
        return {
            "available": True,
            "message": f"Local Ollama online ({provider.config.model_name} ready at {provider.config.endpoint_url})."
        }
    return {
        "available": False,
        "message": (
            f"Ollama daemon not reachable at {provider.config.endpoint_url} with model "
            f"'{provider.config.model_name}'. Switch to Fake Provider for offline testing."
        )
    }


def get_pipeline(provider_type: str = "fake", masking_enabled: bool = True) -> GuardrailPipeline:
    """
    Returns cached or new GuardrailPipeline for the given provider and masking settings.
    """
    global _PIPELINES
    key = (provider_type.lower(), masking_enabled)
    if key in _PIPELINES:
        return _PIPELINES[key]

    router = get_shared_router()

    if provider_type.lower() == "ollama":
        provider: BaseLLMProvider = OllamaProvider()
    else:
        # DeterministicFakeProvider is strictly offline and makes zero network requests
        provider = DeterministicFakeProvider()

    synthesizer = GroundedSynthesizer(provider=provider)
    masking_cfg = MaskingConfig(enabled=masking_enabled)

    pipeline = GuardrailPipeline(
        synthesizer=synthesizer,
        masking_config=masking_cfg,
        router=router
    )
    _PIPELINES[key] = pipeline
    return pipeline


def execute_safe_query(
    query: str,
    provider_type: str = "fake",
    masking_enabled: bool = True,
    pipeline: Optional[GuardrailPipeline] = None
) -> SafePresentationView:
    """
    Executes a user query through the full architecture and adapts output into
    a strictly allowlisted SafePresentationView.

    Guarantees:
    - Real wall-clock timing measurement of the actual query execution.
    - Router classification metadata (route_type, intent) extracted cleanly.
    - Catches all exceptions and fails closed without leaking stack traces or unmasked IDs.
    """
    t0 = time.perf_counter()
    clean_query = (query or "").strip()
    is_thai = any("\u0e00" <= c <= "\u0e7f" for c in clean_query)

    if not clean_query:
        return SafePresentationView(
            final_answer=(
                "กรุณากรอกคำถามหรือข้อความที่ต้องการค้นหาครับ เช่น สถานะตั๋ว หรือประวัติการซ่อมบำรุง"
                if is_thai else
                "Please enter a question or operational lookup query."
            ),
            citations=[],
            warnings=[],
            is_safe_to_present=True,
            status="EMPTY_QUERY",
            execution_time_ms=0.0,
            route_type="CLARIFICATION",
            intent="CLARIFICATION_EMPTY_QUERY",
            language="TH" if is_thai else "EN",
            masking_enabled=masking_enabled,
        )

    try:
        pipe = pipeline or get_pipeline(provider_type=provider_type, masking_enabled=masking_enabled)
        
        # Route query first to capture explicit routing metadata
        router = pipe.router or get_shared_router()
        routing_res: RoutingResult = router.dispatch(clean_query)
        route_str = routing_res.route_type.value if hasattr(routing_res.route_type, "value") else str(routing_res.route_type)
        intent_str = routing_res.normalized_intent.value if hasattr(routing_res.normalized_intent, "value") else str(routing_res.normalized_intent)

        # Execute full pipeline with precomputed routing result
        guardrail_res: GuardrailResult = pipe.process(clean_query, routing_result=routing_res)

        total_ms = (time.perf_counter() - t0) * 1000

        # Transform through strict PresentationAdapter
        view = PresentationAdapter.adapt(
            result=guardrail_res,
            route_type=route_str,
            intent=intent_str,
            language_hint="TH" if is_thai else "EN",
            masking_enabled=masking_enabled
        )

        # Update with real end-to-end wall-clock latency
        return SafePresentationView(
            final_answer=view.final_answer,
            citations=view.citations,
            warnings=view.warnings,
            is_safe_to_present=view.is_safe_to_present,
            status=view.status,
            execution_time_ms=total_ms,
            route_type=view.route_type,
            intent=view.intent,
            language=view.language,
            masking_enabled=view.masking_enabled
        )

    except ProviderUnavailableError as e:
        logger.warning(f"Synthesis provider unavailable: {e}")
        total_ms = (time.perf_counter() - t0) * 1000
        msg = (
            f"⚠️ ไม่สามารถเชื่อมต่อกับ LLM Provider ได้ ({provider_type}) กรุณาตรวจสอบสถานะ service "
            f"หรือเปลี่ยนเป็น Fake Provider สำหรับการทดสอบแบบ Offline"
            if is_thai else
            f"⚠️ LLM provider '{provider_type}' is unavailable. Please verify service status "
            f"or switch to Fake Provider for offline testing."
        )
        return SafePresentationView(
            final_answer=msg,
            citations=[],
            warnings=[f"Provider unavailable: {provider_type}"],
            is_safe_to_present=False,
            status="PROVIDER_UNAVAILABLE",
            execution_time_ms=total_ms,
            route_type=None,
            intent=None,
            language="TH" if is_thai else "EN",
            masking_enabled=masking_enabled
        )

    except Exception as e:
        logger.error(f"Unexpected error in pipeline execution: {e}", exc_info=True)
        total_ms = (time.perf_counter() - t0) * 1000
        # Fail closed: never leak internal stack trace or raw operational data
        fallback_msg = SAFE_ERROR_TH if is_thai else SAFE_ERROR_EN
        return SafePresentationView(
            final_answer=fallback_msg,
            citations=[],
            warnings=["Internal execution error safely contained."],
            is_safe_to_present=False,
            status="EXECUTION_ERROR",
            execution_time_ms=total_ms,
            route_type=None,
            intent=None,
            language="TH" if is_thai else "EN",
            masking_enabled=masking_enabled
        )
