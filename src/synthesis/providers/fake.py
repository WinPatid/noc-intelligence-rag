"""
NOC Intelligence — Deterministic Fake LLM Provider
Provides 100% reproducible, zero-latency text generation for unit tests.
Zero external runtime dependencies.
Synthesizes factual natural language answers directly from <evidence_context>.
"""

import re
from typing import Optional, Dict, List

from src.synthesis.models import ProviderConfig
from src.synthesis.providers.base import BaseLLMProvider, ProviderUnavailableError, LLMProviderError


class DeterministicFakeProvider(BaseLLMProvider):
    """
    Fake LLM provider for fast, deterministic automated unit tests.
    Generates predictable responses grounded directly in retrieved operational evidence.
    """

    def __init__(
        self,
        config: Optional[ProviderConfig] = None,
        custom_responses: Optional[Dict[str, str]] = None,
        simulate_unavailable: bool = False,
        simulate_error: bool = False
    ):
        super().__init__(config or ProviderConfig(provider_name="fake", model_name="deterministic-fake"))
        self.custom_responses = custom_responses or {}
        self.simulate_unavailable = simulate_unavailable
        self.simulate_error = simulate_error

    def is_available(self) -> bool:
        return not self.simulate_unavailable

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        override_config: Optional[ProviderConfig] = None
    ) -> str:
        if self.simulate_unavailable:
            raise ProviderUnavailableError("Simulated provider unavailability.")

        if self.simulate_error:
            raise LLMProviderError("Simulated LLM generation error.")

        # Check explicit custom responses first
        for key, resp in self.custom_responses.items():
            if key in prompt:
                return resp

        # Extract <evidence_context> block if present
        ctx_match = re.search(r"<evidence_context>(.*?)</evidence_context>", prompt, re.DOTALL)
        evidence_text = ctx_match.group(1).strip() if ctx_match else prompt

        # Detect language specifically from the user question, avoiding Thai customer names in context
        q_match = re.search(r"Question:\s*(.*?)(?:\n\nAnswer:|$)", prompt, re.DOTALL)
        question_text = q_match.group(1).strip() if q_match else prompt
        is_thai = any("\u0e00" <= c <= "\u0e7f" for c in question_text)

        # Check if prompt contains citations in <evidence_context>
        source_tags = re.findall(r"\[SOURCE:\s*([^\]]+)\]", prompt)

        if not source_tags or "was not found" in evidence_text.lower() or "not found in the database" in evidence_text.lower():
            if is_thai:
                return "ไม่พบข้อมูลหลักฐานในระบบสำหรับรายการที่สอบถาม"
            return "No verified operational evidence found in the system for this inquiry."

        primary_source = source_tags[0]

        # ── 1. Bridge: Ticket to Work Orders ──────────────────────────────────
        if "total linked work orders:" in evidence_text.lower() or "[linked work order" in evidence_text.lower():
            tid_m = re.search(r"Ticket ID:\s*([A-Za-z0-9\-]+)", evidence_text)
            t_status_m = re.search(r"Ticket ID:\s*[^\n|]+\|\s*Status:\s*([A-Za-z0-9_\-]+)", evidence_text)
            count_m = re.search(r"Total Linked Work Orders:\s*(\d+)", evidence_text)

            tid = tid_m.group(1) if tid_m else "N/A"
            t_status = t_status_m.group(1) if t_status_m else "UNKNOWN"
            count = count_m.group(1) if count_m else "0"

            # Parse each linked work order
            wo_blocks = re.findall(
                r"\[SOURCE:\s*(work_order:[^\]]+)\].*?Work Order ID:\s*([A-Za-z0-9\-]+).*?Status:\s*([A-Za-z0-9_\-]+).*?Root Cause:\s*([^\n\r]+).*?Specific Cause:\s*([^\n\r]+).*?Action Taken:\s*([^\n\r]+).*?Assigned Team:\s*([^\s|]+)",
                evidence_text,
                re.DOTALL
            )

            if is_thai:
                lines = [f"จากข้อมูลในระบบ [SOURCE: ticket:{tid}] ตั๋วปัญหา {tid} (สถานะ: {t_status}) เชื่อมโยงกับใบงานช่าง {count} รายการ:"]
                for src, wo_id, status, rc, sc, act, team in wo_blocks:
                    lines.append(f"- ใบงาน {wo_id} [SOURCE: {src}] (สถานะ: {status}): สาเหตุหลัก {rc} ({sc}) การแก้ไข: {act} มอบหมายทีม {team}")
                return "\n".join(lines)
            else:
                lines = [f"According to verified system records [SOURCE: ticket:{tid}], ticket {tid} (status: {t_status}) is linked to {count} work order(s):"]
                for src, wo_id, status, rc, sc, act, team in wo_blocks:
                    lines.append(f"- Work order {wo_id} [SOURCE: {src}] (status: {status}): Root cause {rc} ({sc}). Action taken: {act}. Assigned team: {team}.")
                return "\n".join(lines)

        # ── 2. Reverse Bridge: Work Order to Tickets ──────────────────────────
        if "total impacted customer tickets:" in evidence_text.lower() or "[affected ticket" in evidence_text.lower():
            wo_m = re.search(r"Work Order ID:\s*([A-Za-z0-9\-]+)", evidence_text)
            wo_status_m = re.search(r"Work Order ID:\s*[^\n|]+\|\s*Status:\s*([A-Za-z0-9_\-]+)", evidence_text)
            count_m = re.search(r"Total Impacted Customer Tickets:\s*(\d+)", evidence_text)

            wo_id = wo_m.group(1) if wo_m else "N/A"
            wo_status = wo_status_m.group(1) if wo_status_m else "UNKNOWN"
            count = count_m.group(1) if count_m else "0"

            t_blocks = re.findall(
                r"\[SOURCE:\s*(ticket:[^\]]+)\].*?Ticket ID:\s*([A-Za-z0-9\-]+)\s*\|\s*Customer:\s*([^\n\r|]+).*?Circuit:\s*([^\n\r|]+)\s*\|\s*Status:\s*([A-Za-z0-9_\-]+).*?Severity:\s*([^\n\r|]+).*?Symptom:\s*([^\n\r]+)",
                evidence_text,
                re.DOTALL
            )

            if is_thai:
                lines = [f"จากข้อมูลในระบบ [SOURCE: work_order:{wo_id}] ใบงานช่าง {wo_id} (สถานะ: {wo_status}) ส่งผลกระทบต่อตั๋วปัญหา {count} รายการ:"]
                for src, tid, cust, ckt, status, sev, sym in t_blocks:
                    lines.append(f"- ตั๋ว {tid} [SOURCE: {src}] (ลูกค้า: {cust.strip()}, สถานะ: {status}, ระดับ: {sev.strip()}): {sym.strip()}")
                return "\n".join(lines)
            else:
                lines = [f"According to verified system records [SOURCE: work_order:{wo_id}], work order {wo_id} (status: {wo_status}) impacted {count} customer ticket(s):"]
                for src, tid, cust, ckt, status, sev, sym in t_blocks:
                    lines.append(f"- Ticket {tid} [SOURCE: {src}] (Customer: {cust.strip()}, Status: {status}, Severity: {sev.strip()}): {sym.strip()}.")
                return "\n".join(lines)

        # ── 3. Single Trouble Ticket Lookup ───────────────────────────────────
        if "ticket id:" in evidence_text.lower() and "[database fact" in evidence_text.lower():
            tid_m = re.search(r"Ticket ID:\s*([A-Za-z0-9\-]+)", evidence_text)
            cust_m = re.search(r"Customer:\s*([^\n\r]+)", evidence_text)
            ckt_m = re.search(r"Circuit:\s*([^\n\r]+)", evidence_text)
            status_m = re.search(r"Status:\s*([A-Za-z0-9_\-]+)", evidence_text)
            sev_m = re.search(r"Severity:\s*([^\n\r]+)", evidence_text)
            sym_m = re.search(r"Reported Symptom:\s*([^\n\r]+)", evidence_text)

            tid = tid_m.group(1) if tid_m else "N/A"
            cust = cust_m.group(1).strip() if cust_m else "N/A"
            ckt = ckt_m.group(1).strip() if ckt_m else "N/A"
            status = status_m.group(1).strip() if status_m else "UNKNOWN"
            sev = sev_m.group(1).strip() if sev_m else "N/A"
            sym = sym_m.group(1).strip() if sym_m else "N/A"

            if is_thai:
                return (
                    f"จากข้อมูลในระบบ [SOURCE: ticket:{tid}] ตั๋วปัญหา {tid} "
                    f"ของลูกค้า {cust} มีสถานะปัจจุบันเป็น {status} "
                    f"ระดับความรุนแรง {sev} บนวงจร {ckt} "
                    f"โดยมีอาการที่ได้รับแจ้งคือ: {sym}"
                )
            else:
                return (
                    f"According to verified system records [SOURCE: ticket:{tid}], ticket {tid} "
                    f"for customer {cust} is currently in {status} status "
                    f"with severity {sev} on circuit {ckt}. "
                    f"Reported symptom: {sym}."
                )

        # ── 4. Single Work Order Lookup ───────────────────────────────────────
        if "work order id:" in evidence_text.lower() and "[database fact" in evidence_text.lower():
            wo_m = re.search(r"Work Order ID:\s*([A-Za-z0-9\-]+)", evidence_text)
            status_m = re.search(r"Status:\s*([A-Za-z0-9_\-]+)", evidence_text)
            rc_m = re.search(r"Root Cause Category:\s*([^\n\r]+)", evidence_text)
            sc_m = re.search(r"Specific Cause:\s*([^\n\r]+)", evidence_text)
            act_m = re.search(r"Action Taken:\s*([^\n\r]+)", evidence_text)
            team_m = re.search(r"Assigned Team:\s*([^\n\r]+)", evidence_text)

            wo_id = wo_m.group(1) if wo_m else "N/A"
            status = status_m.group(1).strip() if status_m else "UNKNOWN"
            rc = rc_m.group(1).strip() if rc_m else "N/A"
            sc = sc_m.group(1).strip() if sc_m else "N/A"
            act = act_m.group(1).strip() if act_m else "N/A"
            team = team_m.group(1).strip() if team_m else "N/A"

            if is_thai:
                return (
                    f"จากข้อมูลในระบบ [SOURCE: work_order:{wo_id}] ใบงานช่าง {wo_id} "
                    f"มีสถานะ {status} สาเหตุหลักคือ {rc} โดยมีสาเหตุเฉพาะคือ {sc} "
                    f"การแก้ไขที่ดำเนินการ: {act} มอบหมายทีม {team}"
                )
            else:
                return (
                    f"According to verified system records [SOURCE: work_order:{wo_id}], work order {wo_id} "
                    f"is in {status} status with root cause category {rc}. "
                    f"Specific cause: {sc}. Action taken: {act}. "
                    f"Assigned team: {team}."
                )

        # ── 5. Circuit Detail Lookup ──────────────────────────────────────────
        if "circuit ref:" in evidence_text.lower():
            ckt_m = re.search(r"Circuit Ref:\s*([A-Za-z0-9\-]+)", evidence_text)
            cust_m = re.search(r"Customer:\s*([^\n\r]+)", evidence_text)
            site_m = re.search(r"Site Name:\s*([^\n\r]+)", evidence_text)
            bw_m = re.search(r"Bandwidth:\s*([^\n\r]+)", evidence_text)
            status_m = re.search(r"Circuit Status:\s*([^\n\r]+)", evidence_text)
            iface_m = re.search(r"Interface:\s*([^\n\r]+)", evidence_text)

            ckt = ckt_m.group(1) if ckt_m else "N/A"
            cust = cust_m.group(1).strip() if cust_m else "N/A"
            site = site_m.group(1).strip() if site_m else "N/A"
            bw = bw_m.group(1).strip() if bw_m else "N/A"
            status = status_m.group(1).strip() if status_m else "UNKNOWN"
            iface = iface_m.group(1).strip() if iface_m else "N/A"

            if is_thai:
                return (
                    f"จากข้อมูลในระบบ [SOURCE: circuit:{ckt}] วงจร {ckt} "
                    f"ของลูกค้า {cust} ณ ไซต์ {site} มีสถานะ {status} "
                    f"ความเร็ว {bw} (อินเทอร์เฟซ: {iface})"
                )
            else:
                return (
                    f"According to verified system records [SOURCE: circuit:{ckt}], circuit {ckt} "
                    f"for customer {cust} at site {site} has status {status} "
                    f"with bandwidth {bw} (Interface: {iface})."
                )

        # ── 6. Customer Ticket Count ──────────────────────────────────────────
        if "total tickets:" in evidence_text.lower():
            cust_m = re.search(r"Customer:\s*([^\n\r]+)", evidence_text)
            total_m = re.search(r"Total Tickets:\s*(\d+)", evidence_text)
            bd_m = re.search(r"Status Breakdown:\s*([^\n\r]+)", evidence_text)
            filt_m = re.search(r"Applied Status Filter:\s*([^\n\r]+)", evidence_text)

            cust = cust_m.group(1).strip() if cust_m else "Customer"
            total = total_m.group(1) if total_m else "0"
            bd = bd_m.group(1).strip() if bd_m else "N/A"
            filt = filt_m.group(1).strip() if filt_m else "None"

            if is_thai:
                return (
                    f"จากข้อมูลในระบบ [SOURCE: {primary_source}] ลูกค้า {cust} "
                    f"มีตั๋วปัญหาบันทึกในระบบทั้งหมด {total} รายการ "
                    f"สรุปตามสถานะ: {bd} (เงื่อนไขตัวกรอง: {filt})"
                )
            else:
                return (
                    f"According to verified system records [SOURCE: {primary_source}], customer {cust} "
                    f"has a total of {total} trouble ticket(s) recorded in the database. "
                    f"Status breakdown: {bd} (Applied filter: {filt})."
                )

        # ── 7. Customer Circuits Inventory ────────────────────────────────────
        if "total leased circuits:" in evidence_text.lower():
            cust_m = re.search(r"Customer:\s*([^\n\r]+)", evidence_text)
            total_m = re.search(r"Total Leased Circuits:\s*(\d+)", evidence_text)
            circ_lines = [line.strip() for line in evidence_text.splitlines() if line.strip().startswith("- CKT-")]

            cust = cust_m.group(1).strip() if cust_m else "Customer"
            total = total_m.group(1) if total_m else str(len(circ_lines))

            if is_thai:
                return (
                    f"จากข้อมูลในระบบ [SOURCE: {primary_source}] ลูกค้า {cust} มีวงจรสื่อสารทั้งหมด {total} วงจร:\n"
                    + "\n".join(circ_lines)
                )
            else:
                return (
                    f"According to verified system records [SOURCE: {primary_source}], customer {cust} has {total} leased circuit(s) registered in the repository:\n"
                    + "\n".join(circ_lines)
                )

        # ── 8. Semantic Search & Hybrid Hits ──────────────────────────────────
        hits = re.split(r"(?=\[SOURCE:\s*[^\]]+\])", evidence_text)
        parsed_hits: List[str] = []

        for h in hits:
            h_strip = h.strip()
            if not h_strip:
                continue
            src_m = re.search(r"\[SOURCE:\s*([^\]]+)\]", h_strip)
            if not src_m:
                continue
            src_tag = src_m.group(1)

            # Check if work order hit
            wo_m = re.search(r"Work Order ID:\s*([A-Za-z0-9\-]+)", h_strip)
            rc_m = re.search(r"Root Cause:\s*([^\n\r]+)", h_strip)
            doc_m = re.search(r"(?:Resolution Document|Incident Summary):\s*(.*?)(?=\n\n|$)", h_strip, re.DOTALL)

            tid_m = re.search(r"Ticket ID:\s*([A-Za-z0-9\-]+)", h_strip)
            status_m = re.search(r"Status:\s*([A-Za-z0-9_\-]+)", h_strip)

            if wo_m:
                wo_id = wo_m.group(1)
                rc = rc_m.group(1).strip() if rc_m else "Operational Issue"
                snippet = doc_m.group(1).strip()[:180] if doc_m else "Record details logged."
                if is_thai:
                    parsed_hits.append(f"- ใบงาน {wo_id} [SOURCE: {src_tag}] สาเหตุ: {rc} — {snippet}")
                else:
                    parsed_hits.append(f"- Work order {wo_id} [SOURCE: {src_tag}] (Root cause: {rc}): {snippet}")
            elif tid_m:
                tid = tid_m.group(1)
                status = status_m.group(1) if status_m else "RESOLVED"
                snippet = doc_m.group(1).strip()[:180] if doc_m else "Record details logged."
                if is_thai:
                    parsed_hits.append(f"- ตั๋วปัญหา {tid} [SOURCE: {src_tag}] สถานะ: {status} — {snippet}")
                else:
                    parsed_hits.append(f"- Ticket {tid} [SOURCE: {src_tag}] (Status: {status}): {snippet}")

        if parsed_hits:
            if is_thai:
                return (
                    "จากข้อมูลการตรวจสอบในระบบ พบหลักฐานปฏิบัติการที่เกี่ยวข้องดังนี้:\n"
                    + "\n".join(parsed_hits)
                )
            else:
                return (
                    "According to verified system records, the operational investigation retrieved the following matching evidence:\n"
                    + "\n".join(parsed_hits)
                )

        # ── 9. Fallback if source tags exist but no template matched ──────────
        if is_thai:
            return f"จากข้อมูลในระบบ พบรายละเอียดตามบันทึก [SOURCE: {primary_source}] ปัญหาได้รับการตรวจสอบและยืนยันตามหลักฐาน"
        return f"According to verified system records [SOURCE: {primary_source}], the operational record has been verified in the repository."
