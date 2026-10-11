"""
NOC Intelligence — Stage 6 Deterministic Identifier Masking Engine
Applies consistent, deterministic pseudonyms and masks to operational identifiers
and customer corporate names without modifying underlying databases.
"""

import re
from typing import Dict, Tuple, List, Optional

from src.router.patterns import (
    RE_TICKET_ID,
    RE_WORK_ORDER_ID,
    RE_CIRCUIT_REF
)
from src.synthesis.models import CitationItem
from src.guardrails.models import MaskingConfig
from src.db.sql_engine import get_all_customers


# Deterministic customer pseudonym tokens
CUSTOMER_PSEUDONYMS = [
    "CUSTOMER-ALPHA", "CUSTOMER-BETA", "CUSTOMER-GAMMA", "CUSTOMER-DELTA",
    "CUSTOMER-EPSILON", "CUSTOMER-ZETA", "CUSTOMER-ETA", "CUSTOMER-THETA",
    "CUSTOMER-IOTA", "CUSTOMER-KAPPA", "CUSTOMER-LAMBDA", "CUSTOMER-MU",
    "CUSTOMER-NU", "CUSTOMER-XI", "CUSTOMER-OMICRON", "CUSTOMER-PI",
    "CUSTOMER-RHO", "CUSTOMER-SIGMA", "CUSTOMER-TAU", "CUSTOMER-UPSILON"
]


class IdentifierMasker:
    """
    Deterministic masking engine for sensitive synthetic telecom entities.
    Guarantees referential consistency within each query session.
    """

    def __init__(self, config: Optional[MaskingConfig] = None, db_path: Optional[str] = None):
        self.config = config or MaskingConfig()
        self.db_path = db_path
        self._customers_cache = []
        self._load_customers()

    def _load_customers(self):
        try:
            self._customers_cache = get_all_customers(self.db_path)
            # Sort by longest name descending to avoid substring conflicts
            self._customers_cache.sort(
                key=lambda c: max(
                    len(c.get("company_name_en") or ""),
                    len(c.get("company_name_th") or "")
                ),
                reverse=True
            )
        except Exception:
            self._customers_cache = []

    def mask_text_and_citations(
        self,
        text: str,
        citations: List[CitationItem]
    ) -> Tuple[str, List[CitationItem], Dict[str, str]]:
        """
        Applies configured masking rules to text and citations.
        Returns (masked_text, masked_citations, identifier_map).
        """
        if not self.config.enabled or not text:
            return text, citations, {}

        masked_text = text
        id_map: Dict[str, str] = {}

        # ── 1. Mask Ticket Identifiers (INC-2025-XXXXX) ────────────────────────
        if self.config.mask_tickets:
            for match in RE_TICKET_ID.finditer(text):
                exact_id = match.group(0)
                canonical_id = exact_id.upper()
                last_digits = canonical_id.split("-")[-1][-3:]
                pseudo = f"[TICKET-***{last_digits}]"
                id_map[canonical_id] = pseudo
                if exact_id != canonical_id:
                    id_map[exact_id] = pseudo

        # ── 2. Mask Work Order Identifiers (WO-70XXXX) ─────────────────────────
        if self.config.mask_work_orders:
            for match in RE_WORK_ORDER_ID.finditer(text):
                exact_id = match.group(0)
                canonical_id = exact_id.upper()
                last_digits = canonical_id.split("-")[-1][-3:]
                pseudo = f"[WO-***{last_digits}]"
                id_map[canonical_id] = pseudo
                if exact_id != canonical_id:
                    id_map[exact_id] = pseudo

        # ── 3. Mask Circuit Identifiers (CKT-88XXXX) ───────────────────────────
        if self.config.mask_circuits:
            for match in RE_CIRCUIT_REF.finditer(text):
                exact_id = match.group(0)
                canonical_id = exact_id.upper()
                last_digits = canonical_id.split("-")[-1][-3:]
                pseudo = f"[CIRCUIT-***{last_digits}]"
                id_map[canonical_id] = pseudo
                if exact_id != canonical_id:
                    id_map[exact_id] = pseudo

        # ── 4. Mask Customer Names (English & Thai) ───────────────────────────
        if self.config.mask_customers and self._customers_cache:
            for idx, c in enumerate(self._customers_cache):
                name_en = c.get("company_name_en")
                name_th = c.get("company_name_th")
                short_name = c.get("company_short_name")
                pseudo = f"[{CUSTOMER_PSEUDONYMS[idx % len(CUSTOMER_PSEUDONYMS)]}]"

                if name_en and re.search(re.escape(name_en), masked_text, re.IGNORECASE):
                    id_map[name_en] = pseudo
                if name_th and name_th in masked_text:
                    id_map[name_th] = pseudo
                if short_name and len(short_name) >= 4 and re.search(re.escape(short_name), masked_text, re.IGNORECASE):
                    id_map[short_name] = pseudo

        # Apply replacements to text (sorted by length descending to prevent partial overlaps)
        for raw, masked in sorted(id_map.items(), key=lambda x: len(x[0]), reverse=True):
            pattern = re.compile(re.escape(raw), re.IGNORECASE)
            masked_text = pattern.sub(masked, masked_text)

        # ── 5. Mask Citations ──────────────────────────────────────────────────
        masked_citations: List[CitationItem] = []
        for cit in citations:
            new_label = cit.label
            for raw, masked in sorted(id_map.items(), key=lambda x: len(x[0]), reverse=True):
                pattern = re.compile(re.escape(raw), re.IGNORECASE)
                new_label = pattern.sub(masked, new_label)

            # Preserve internal source_id for backend traceability, mask user label
            masked_id = id_map.get(cit.identifier.upper(), id_map.get(cit.identifier, cit.identifier))
            masked_citations.append(
                CitationItem(
                    source_id=cit.source_id,
                    entity_type=cit.entity_type,
                    identifier=masked_id,
                    label=new_label,
                    is_verified=cit.is_verified
                )
            )

        return masked_text, masked_citations, id_map

    def unmask_text(self, text: str, id_map: Dict[str, str]) -> str:
        """Helper to reverse masking for authorized internal inspection."""
        unmasked = text
        for raw, masked in id_map.items():
            unmasked = unmasked.replace(masked, raw)
        return unmasked
