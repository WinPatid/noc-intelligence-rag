"""
NOC Intelligence — Stage 6 Guardrail Pipeline
Combines Grounded Synthesis, Independent Critic Verification, Contradiction Rejection,
and Operational Identifier Masking into a unified, secure response processor.
"""

import time
from typing import Optional, Dict, Any, List

from src.router.models import RoutingResult
from src.router.router import DeterministicRouter
from src.synthesis.models import SynthesisResponse, GenerationStatus, CitationItem
from src.synthesis.synthesizer import GroundedSynthesizer
from src.guardrails.models import (
    GuardrailStatus,
    GuardrailResult,
    MaskingConfig
)
from src.guardrails.critic import GuardrailCritic
from src.guardrails.masker import IdentifierMasker


class GuardrailPipeline:
    """
    Independent guardrail processor executed between LLM synthesis and user presentation.
    Coordinates router dispatch, synthesis, claim-level critique, safe degradation,
    and operational identifier masking.
    """

    def __init__(
        self,
        critic: Optional[GuardrailCritic] = None,
        masker: Optional[IdentifierMasker] = None,
        synthesizer: Optional[GroundedSynthesizer] = None,
        masking_config: Optional[MaskingConfig] = None,
        router: Optional[DeterministicRouter] = None,
        db_path: Optional[str] = None
    ):
        self.db_path = db_path
        self.critic = critic or GuardrailCritic(db_path=db_path)
        self.masker = masker or IdentifierMasker(config=masking_config, db_path=db_path)
        self.synthesizer = synthesizer or GroundedSynthesizer()
        self.router = router or DeterministicRouter(db_path=db_path)

    def process(
        self,
        query: str,
        routing_result: Optional[RoutingResult] = None,
        synthesis_response: Optional[SynthesisResponse] = None
    ) -> GuardrailResult:
        """
        Runs the full verification, contradiction critique, and masking pipeline.
        If routing_result is omitted, automatically executes self.router.dispatch(query).
        """
        t0 = time.perf_counter()
        is_thai = any("\u0e00" <= c <= "\u0e7f" for c in query)

        # Step 0: Route and dispatch query if routing_result is not pre-computed
        if routing_result is None:
            try:
                routing_result = self.router.dispatch(query)
            except Exception as e:
                return GuardrailResult(
                    status=GuardrailStatus.INVALID_RESPONSE,
                    original_answer="",
                    final_answer=(
                        "เกิดข้อผิดพลาดในการประมวลผลคำค้นหาครับ"
                        if is_thai else
                        "An error occurred during query dispatch."
                    ),
                    is_safe_to_present=False,
                    warnings=[f"Router dispatch error: {type(e).__name__}"],
                    execution_time_ms=(time.perf_counter() - t0) * 1000
                )

        # Step 1: Synthesize response if not already provided
        synth_res = synthesis_response
        if synth_res is None:
            try:
                synth_res = self.synthesizer.synthesize(query, routing_result)
            except Exception as e:
                return GuardrailResult(
                    status=GuardrailStatus.INVALID_RESPONSE,
                    original_answer="",
                    final_answer=(
                        "เกิดข้อผิดพลาดในการสร้างคำตอบครับ"
                        if is_thai else
                        "An error occurred during response synthesis."
                    ),
                    is_safe_to_present=False,
                    warnings=[f"Synthesizer error: {type(e).__name__}"],
                    execution_time_ms=(time.perf_counter() - t0) * 1000
                )

        raw_answer = synth_res.answer
        citations = synth_res.citations
        is_thai = (synth_res.language == "TH")

        # Step 2: Critic claim-level verification and contradiction check
        try:
            status, claims, contradictions, unverified = self.critic.critique(
                generated_answer=raw_answer,
                routing_result=routing_result,
                synthesis_response=synth_res
            )
        except Exception as e:
            status = GuardrailStatus.INVALID_RESPONSE
            claims = []
            contradictions = [f"Guardrail critic error: {type(e).__name__}"]
            unverified = []

        # Step 3: Handle Material Contradiction & Invalid Response (Safe Degradation)
        is_safe = True
        warnings = list(synth_res.warnings) + contradictions + unverified
        final_answer = raw_answer

        if status == GuardrailStatus.CONTRADICTED:
            is_safe = False
            # Suppress false contradicted claims
            if is_thai:
                final_answer = (
                    "⚠️ [ระบบความปลอดภัยของข้อมูล] ตรวจพบข้อขัดแย้งของข้อเท็จจริงระหว่างคำตอบที่สร้างขึ้น"
                    "และบันทึกฐานข้อมูลจริงของศูนย์ NOC (สถานะหรือสาเหตุไม่ตรงกับหลักฐาน)"
                    " ระบบจึงขอระงับการแสดงผลข้อความที่ไม่ถูกต้องเพื่อป้องกันความเข้าใจผิดครับ"
                )
            else:
                final_answer = (
                    "⚠️ [Data Safety Guardrail] A factual contradiction was detected between the generated response "
                    "and verified database records. To prevent misinformation, the unverified response has been suppressed."
                )

        elif status == GuardrailStatus.INVALID_RESPONSE:
            is_safe = False
            final_answer = (
                "เกิดข้อผิดพลาดในการสร้างคำตอบครับ" if is_thai else "An error occurred during response generation."
            )

        # Step 4: Apply Operational Identifier Masking if enabled and safe
        masked_map: Dict[str, str] = {}
        if is_safe and self.masker.config.enabled:
            try:
                final_answer, citations, masked_map = self.masker.mask_text_and_citations(
                    text=final_answer,
                    citations=citations
                )
                # Mask user-facing warnings to prevent identifier leakage
                masked_warnings = []
                for w in warnings:
                    mw, _, _ = self.masker.mask_text_and_citations(w, [])
                    masked_warnings.append(mw)
                warnings = masked_warnings
            except Exception as e:
                # Masking failure: fail closed to prevent leaking raw identifiers!
                is_safe = False
                status = GuardrailStatus.INVALID_RESPONSE
                final_answer = (
                    "⚠️ [ระบบความปลอดภัยของข้อมูล] ไม่สามารถปกปิดข้อมูลระบุตัวตนได้ จึงระงับการแสดงผลเพื่อความปลอดภัย"
                    if is_thai else
                    "⚠️ [Data Safety Guardrail] Identifier masking failed. Response suppressed to prevent identifier exposure."
                )
                citations = []
                masked_map = {}
                warnings.append(f"Masking error: {type(e).__name__}")

        return GuardrailResult(
            status=status,
            original_answer=raw_answer,
            final_answer=final_answer,
            claims=claims,
            contradictions=contradictions,
            unverified_claims=unverified,
            masked_identifiers_map=masked_map,
            citations=citations,
            is_safe_to_present=is_safe,
            warnings=warnings,
            execution_time_ms=(time.perf_counter() - t0) * 1000
        )
