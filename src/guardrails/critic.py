"""
NOC Intelligence — Stage 6 Guardrail Critic
Performs independent claim-level verification and contradiction checks
between generated responses and structured retrieval evidence.
"""

import re
from typing import List, Dict, Any, Tuple, Optional

from src.router.patterns import (
    RE_TICKET_ID,
    RE_WORK_ORDER_ID,
    RE_CIRCUIT_REF
)
from src.router.models import RoutingResult, RouteType, IntentType
from src.synthesis.models import SynthesisResponse, GenerationStatus
from src.guardrails.models import (
    GuardrailStatus,
    ClaimType,
    ClaimVerificationItem
)
from src.db.sql_engine import get_all_customers
from src.guardrails.contradiction import (
    check_status_contradiction,
    check_severity_contradiction,
    check_root_cause_contradiction,
    check_linkage_contradiction,
    check_customer_association_contradiction
)


class GuardrailCritic:
    """
    Independent critic evaluating generated text against actual retrieved operational facts.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path
        self._customers_cache = []
        self._load_customers()

    def _load_customers(self):
        try:
            self._customers_cache = get_all_customers(self.db_path)
        except Exception:
            self._customers_cache = []

    def critique(
        self,
        generated_answer: str,
        routing_result: RoutingResult,
        synthesis_response: SynthesisResponse
    ) -> Tuple[GuardrailStatus, List[ClaimVerificationItem], List[str], List[str]]:
        """
        Critiques generated answer.
        Returns (status, claims_list, contradictions_list, unverified_claims_list).
        """
        claims: List[ClaimVerificationItem] = []
        contradictions: List[str] = []
        unverified: List[str] = []

        # ── 1. Pass-through for non-generative / degraded statuses ───────────
        if synthesis_response.generation_status == GenerationStatus.INSUFFICIENT_EVIDENCE:
            return GuardrailStatus.INSUFFICIENT_EVIDENCE, [], [], ["Insufficient evidence to support factual claims."]

        if synthesis_response.generation_status == GenerationStatus.PROVIDER_UNAVAILABLE:
            return GuardrailStatus.PROVIDER_UNAVAILABLE, [], [], ["LLM provider offline; synthesis bypassed."]

        if routing_result.route_type in (RouteType.CLARIFICATION, RouteType.UNSUPPORTED):
            return GuardrailStatus.VERIFIED, [], [], []

        if synthesis_response.generation_status == GenerationStatus.ERROR:
            return GuardrailStatus.INVALID_RESPONSE, [], ["Synthesis generation error."], []

        if synthesis_response.generation_status == GenerationStatus.CLARIFICATION:
            return GuardrailStatus.VERIFIED, [], [], []

        if not generated_answer or not generated_answer.strip():
            return GuardrailStatus.INVALID_RESPONSE, [], ["Response is empty."], []

        # ── 2. Collect Ground-Truth Reference Records from Evidence ───────────
        evidence_records = self._extract_evidence_records(routing_result)

        # ── 3. Verify Mentioned Identifiers Against Evidence ─────────────────
        found_tickets = [t.upper() for t in RE_TICKET_ID.findall(generated_answer)]
        found_wos = [w.upper() for w in RE_WORK_ORDER_ID.findall(generated_answer)]
        found_ckts = [c.upper() for c in RE_CIRCUIT_REF.findall(generated_answer)]

        for tid in found_tickets:
            is_present = any(tid == rec.get("ticket_id") for rec in evidence_records.get("tickets", []))
            claims.append(ClaimVerificationItem(
                claim_type=ClaimType.TICKET_ID,
                claimed_value=tid,
                evidence_value=tid if is_present else None,
                is_verified=is_present,
                is_contradicted=not is_present,
                source_id=f"ticket:{tid}",
                notes="Verified in evidence" if is_present else "Fabricated/unretrieved ticket ID mentioned in text"
            ))
            if not is_present:
                unverified.append(f"Answer generated unretrieved ticket identifier '{tid}'.")

        for wo_id in found_wos:
            is_present = any(wo_id == rec.get("work_order_id") for rec in evidence_records.get("work_orders", []))
            claims.append(ClaimVerificationItem(
                claim_type=ClaimType.WORK_ORDER_ID,
                claimed_value=wo_id,
                evidence_value=wo_id if is_present else None,
                is_verified=is_present,
                is_contradicted=not is_present,
                source_id=f"work_order:{wo_id}",
                notes="Verified in evidence" if is_present else "Fabricated/unretrieved work order ID mentioned in text"
            ))
            if not is_present:
                unverified.append(f"Answer generated unretrieved work order identifier '{wo_id}'.")

        for ckt in found_ckts:
            is_present = any(ckt == rec.get("circuit_ref") for rec in evidence_records.get("circuits", []))
            claims.append(ClaimVerificationItem(
                claim_type=ClaimType.CIRCUIT_REF,
                claimed_value=ckt,
                evidence_value=ckt if is_present else None,
                is_verified=is_present,
                is_contradicted=not is_present,
                source_id=f"circuit:{ckt}",
                notes="Verified in evidence" if is_present else "Fabricated/unretrieved circuit identifier mentioned in text"
            ))
            if not is_present:
                unverified.append(f"Answer generated unretrieved circuit identifier '{ckt}'.")

        # ── 4. Verify Status & Severity Claims Against Structured Evidence ──
        for t_rec in evidence_records.get("tickets", []):
            tid = t_rec.get("ticket_id")
            if tid and tid in found_tickets:
                ev_status = t_rec.get("status")
                if ev_status:
                    # Check if text asserts conflicting status
                    contra = check_status_contradiction(generated_answer, ev_status)
                    if contra:
                        contradictions.append(f"[{tid}] {contra}")
                        claims.append(ClaimVerificationItem(
                            claim_type=ClaimType.STATUS,
                            claimed_value="Contradictory Status",
                            evidence_value=ev_status,
                            is_verified=False,
                            is_contradicted=True,
                            source_id=f"ticket:{tid}",
                            notes=contra
                        ))
                    else:
                        claims.append(ClaimVerificationItem(
                            claim_type=ClaimType.STATUS,
                            claimed_value=ev_status,
                            evidence_value=ev_status,
                            is_verified=True,
                            is_contradicted=False,
                            source_id=f"ticket:{tid}",
                            notes=f"Ticket status matches '{ev_status}'"
                        ))

                ev_sev = t_rec.get("severity")
                if ev_sev:
                    contra_sev = check_severity_contradiction(generated_answer, ev_sev)
                    if contra_sev:
                        contradictions.append(f"[{tid}] {contra_sev}")
                        claims.append(ClaimVerificationItem(
                            claim_type=ClaimType.SEVERITY,
                            claimed_value="Contradictory Severity",
                            evidence_value=ev_sev,
                            is_verified=False,
                            is_contradicted=True,
                            source_id=f"ticket:{tid}",
                            notes=contra_sev
                        ))

        # ── 5. Verify Work Order Cause & Linkages ─────────────────────────────
        for wo_rec in evidence_records.get("work_orders", []):
            wo_id = wo_rec.get("work_order_id")
            if wo_id and wo_id in found_wos:
                ev_cause = wo_rec.get("root_cause_category")
                if ev_cause:
                    contra_cause = check_root_cause_contradiction(generated_answer, ev_cause)
                    if contra_cause:
                        contradictions.append(f"[{wo_id}] {contra_cause}")
                        claims.append(ClaimVerificationItem(
                            claim_type=ClaimType.ROOT_CAUSE,
                            claimed_value="Contradictory Root Cause",
                            evidence_value=ev_cause,
                            is_verified=False,
                            is_contradicted=True,
                            source_id=f"work_order:{wo_id}",
                            notes=contra_cause
                        ))

        # Check bridge link claims
        if evidence_records.get("valid_linked_wo_ids") is not None and found_wos:
            valid_links = evidence_records["valid_linked_wo_ids"]
            for wo_id in found_wos:
                link_contra = check_linkage_contradiction(wo_id, valid_links)
                if link_contra:
                    contradictions.append(link_contra)
                    claims.append(ClaimVerificationItem(
                        claim_type=ClaimType.LINKED_ENTITY,
                        claimed_value=wo_id,
                        evidence_value=str(valid_links),
                        is_verified=False,
                        is_contradicted=True,
                        notes=link_contra
                    ))

        # ── 6. Verify Customer Association Claims ─────────────────────────────
        evidence_cust_names = set()
        for c_rec in evidence_records.get("customers", []):
            if c_rec.get("company_name_en"):
                evidence_cust_names.add(c_rec["company_name_en"].strip().lower())
            if c_rec.get("company_name_th"):
                evidence_cust_names.add(c_rec["company_name_th"].strip().lower())
        for t_rec in evidence_records.get("tickets", []):
            if t_rec.get("company_name_en"):
                evidence_cust_names.add(t_rec["company_name_en"].strip().lower())
            if t_rec.get("company_name_th"):
                evidence_cust_names.add(t_rec["company_name_th"].strip().lower())

        if evidence_cust_names and self._customers_cache:
            primary_evidence_name = next(iter(evidence_cust_names))
            gen_lower = generated_answer.lower()
            for cust in self._customers_cache:
                c_en = cust.get("company_name_en") or ""
                c_th = cust.get("company_name_th") or ""

                en_in_text = bool(c_en and len(c_en) >= 3 and c_en.lower() in gen_lower)
                th_in_text = bool(c_th and len(c_th) >= 3 and c_th.lower() in gen_lower)

                if en_in_text or th_in_text:
                    claimed_name = c_en if en_in_text else c_th
                    is_verified_cust = (
                        (c_en and c_en.lower() in evidence_cust_names) or
                        (c_th and c_th.lower() in evidence_cust_names)
                    )
                    if not is_verified_cust:
                        cust_contra = check_customer_association_contradiction(
                            claimed_customer=claimed_name,
                            evidence_customer=primary_evidence_name
                        )
                        if cust_contra:
                            contradictions.append(cust_contra)
                            claims.append(ClaimVerificationItem(
                                claim_type=ClaimType.CUSTOMER_NAME,
                                claimed_value=claimed_name,
                                evidence_value=primary_evidence_name,
                                is_verified=False,
                                is_contradicted=True,
                                source_id=f"ticket:{found_tickets[0]}" if found_tickets else None,
                                notes=cust_contra
                            ))
                    else:
                        claims.append(ClaimVerificationItem(
                            claim_type=ClaimType.CUSTOMER_NAME,
                            claimed_value=claimed_name,
                            evidence_value=claimed_name,
                            is_verified=True,
                            is_contradicted=False,
                            source_id=f"ticket:{found_tickets[0]}" if found_tickets else None,
                            notes=f"Customer verified in evidence: {claimed_name}"
                        ))

        # ── 7. Determine Guardrail Status ─────────────────────────────────────
        if contradictions:
            status = GuardrailStatus.CONTRADICTED
        elif unverified:
            status = GuardrailStatus.PARTIALLY_VERIFIED
        else:
            status = GuardrailStatus.VERIFIED

        return status, claims, contradictions, unverified

    def _extract_evidence_records(self, routing_res: RoutingResult) -> Dict[str, Any]:
        """Flattens retrieved structured payloads into entity dictionaries."""
        records: Dict[str, Any] = {
            "tickets": [],
            "work_orders": [],
            "circuits": [],
            "customers": [],
            "valid_linked_wo_ids": None
        }
        res = routing_res.retrieval_results
        if not res or not isinstance(res, dict):
            return records

        entity_type = res.get("entity_type")
        if entity_type == "trouble_ticket" and res.get("record"):
            records["tickets"].append(res["record"])
            if res["record"].get("circuit_ref"):
                records["circuits"].append({"circuit_ref": res["record"]["circuit_ref"]})
            if res["record"].get("company_name_en") or res["record"].get("company_name_th"):
                records["customers"].append(res["record"])

        elif entity_type == "work_order" and res.get("record"):
            records["work_orders"].append(res["record"])

        elif entity_type == "ticket_work_order_bridge":
            tid = res.get("ticket_id")
            if tid:
                records["tickets"].append({"ticket_id": tid, "status": res.get("ticket_status")})
            wos = res.get("linked_work_orders", [])
            records["work_orders"].extend(wos)
            records["valid_linked_wo_ids"] = [w["work_order_id"] for w in wos]

        elif entity_type == "work_order_ticket_bridge":
            wo_id = res.get("work_order_id")
            if wo_id:
                records["work_orders"].append({"work_order_id": wo_id, "status": res.get("work_order_status")})
            records["tickets"].extend(res.get("impacted_tickets", []))

        elif entity_type == "circuit" and res.get("record"):
            records["circuits"].append(res["record"])
            if res["record"].get("circuit_ref"):
                records["circuits"].append({"circuit_ref": res["record"]["circuit_ref"]})
            if res["record"].get("company_name_en") or res["record"].get("company_name_th"):
                records["customers"].append(res["record"])

        # Top-level circuit or customer payloads (from SQL or Hybrid)
        if res.get("circuit"):
            records["circuits"].append(res["circuit"])
            if res["circuit"].get("circuit_ref"):
                records["circuits"].append({"circuit_ref": res["circuit"]["circuit_ref"]})
        if res.get("circuit_ref"):
            records["circuits"].append({"circuit_ref": res["circuit_ref"]})
        if res.get("circuit_details"):
            records["circuits"].append(res["circuit_details"])
            if res["circuit_details"].get("circuit_ref"):
                records["circuits"].append({"circuit_ref": res["circuit_details"]["circuit_ref"]})

        if res.get("customer"):
            records["customers"].append(res["customer"])
        if res.get("company_name_en") or res.get("customer_id"):
            records["customers"].append({
                "customer_id": res.get("customer_id"),
                "company_name_en": res.get("company_name_en")
            })

        # Semantic hits
        for hit in (res.get("hits") or res.get("work_orders") or []):
            if isinstance(hit, dict) and hit.get("metadata"):
                meta = hit["metadata"]
                if "work_order_id" in meta:
                    records["work_orders"].append(meta)
                if "ticket_id" in meta:
                    records["tickets"].append(meta)
                if "circuit_ref" in meta:
                    records["circuits"].append({"circuit_ref": meta["circuit_ref"]})

        for hit in (res.get("trouble_tickets") or []):
            if isinstance(hit, dict) and hit.get("metadata"):
                meta = hit["metadata"]
                records["tickets"].append(meta)
                if "circuit_ref" in meta:
                    records["circuits"].append({"circuit_ref": meta["circuit_ref"]})

        # Hybrid incidents
        for inc in res.get("incidents", []):
            if isinstance(inc, dict):
                tid = inc.get("ticket_id")
                t_meta = inc.get("ticket_metadata") or {}
                t_rec = dict(t_meta)
                if tid:
                    t_rec["ticket_id"] = tid
                records["tickets"].append(t_rec)
                if t_rec.get("circuit_ref"):
                    records["circuits"].append({"circuit_ref": t_rec["circuit_ref"]})
                if t_rec.get("company_name_en") or t_rec.get("company_name_th"):
                    records["customers"].append(t_rec)
                records["work_orders"].extend(inc.get("linked_work_orders", []))

        return records
