"""
NOC Intelligence — Stage 4 Deterministic Entity Extractor
Extracts and validates identifiers, customer entities, status filters,
and intent cues from raw text input.
"""

from typing import List, Dict, Any, Optional

from src.db.sql_engine import get_all_customers
from src.router.models import ExtractedEntities
from src.router.patterns import (
    RE_TICKET_ID,
    RE_WORK_ORDER_ID,
    RE_CIRCUIT_REF,
    RE_CUSTOMER_CODE,
    RE_SQL_INJECTION,
    RE_BRIDGE_TICKET_TO_WO,
    RE_BRIDGE_WO_TO_TICKETS,
    COUNT_KEYWORDS,
    CIRCUIT_INVENTORY_KEYWORDS,
    STATUS_KEYWORDS_MAP,
    SEMANTIC_SYMPTOM_KEYWORDS,
    RE_VAGUE_COMPLAINT,
    RE_STATUS_INQUIRY
)


class EntityExtractor:
    """
    Deterministic extractor for NOC domain entities.
    Caches synthetic customer references from SQLite on startup.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path
        self._customers_cache: List[Dict[str, Any]] = []
        self._load_customers_cache()

    def _load_customers_cache(self):
        try:
            self._customers_cache = get_all_customers(self.db_path)
            # Sort customers by length of company_name_en descending to prioritize exact long matches
            self._customers_cache.sort(
                key=lambda c: max(
                    len(c.get("company_name_en") or ""),
                    len(c.get("company_name_th") or ""),
                    len(c.get("company_short_name") or "")
                ),
                reverse=True
            )
        except Exception:
            # Fallback if DB is not initialized yet in isolated unit test
            self._customers_cache = []

    def extract(self, query: str) -> ExtractedEntities:
        cleaned = query.strip() if query else ""
        if not cleaned:
            return ExtractedEntities()

        # 1. Regex Identifier Extraction
        ticket_ids = [t.upper() for t in RE_TICKET_ID.findall(cleaned)]
        work_order_ids = [w.upper() for w in RE_WORK_ORDER_ID.findall(cleaned)]
        circuit_refs = [c.upper() for c in RE_CIRCUIT_REF.findall(cleaned)]
        customer_codes = [cc.upper() for cc in RE_CUSTOMER_CODE.findall(cleaned)]

        # 2. SQL Injection Detection
        has_sql_injection = bool(RE_SQL_INJECTION.search(cleaned))

        # 3. Customer Entity Matching
        matched_cust = None
        matched_name_raw = None
        lower_query = cleaned.lower()

        # Check customer codes first if present
        if customer_codes:
            c_code = customer_codes[0]
            for c in self._customers_cache:
                if (c.get("customer_code") or "").upper() == c_code:
                    matched_cust = c
                    matched_name_raw = c_code
                    break

        # Check customer names in cache
        if not matched_cust:
            for c in self._customers_cache:
                name_en = (c.get("company_name_en") or "").lower()
                name_th = (c.get("company_name_th") or "").lower()
                short_name = (c.get("company_short_name") or "").lower()

                if name_en and name_en in lower_query:
                    matched_cust = c
                    matched_name_raw = c.get("company_name_en")
                    break
                if name_th and name_th in lower_query:
                    matched_cust = c
                    matched_name_raw = c.get("company_name_th")
                    break
                # Only match short name if it's at least 4 characters to avoid false hits
                if short_name and len(short_name) >= 4 and short_name in lower_query:
                    matched_cust = c
                    matched_name_raw = c.get("company_short_name")
                    break

        # 4. Status Filter Matching
        status_filter = None
        for keyword, norm_status in STATUS_KEYWORDS_MAP.items():
            if keyword.lower() in lower_query:
                status_filter = norm_status
                break

        # 5. Intent Flags
        is_bridge = bool(RE_BRIDGE_TICKET_TO_WO.search(cleaned))
        is_reverse_bridge = bool(RE_BRIDGE_WO_TO_TICKETS.search(cleaned))
        is_count = any(kw.lower() in lower_query for kw in COUNT_KEYWORDS)
        is_inventory = any(kw.lower() in lower_query for kw in CIRCUIT_INVENTORY_KEYWORDS)

        # 6. Semantic Terms Matching
        semantic_terms = []
        for term in SEMANTIC_SYMPTOM_KEYWORDS:
            if term.lower() in lower_query:
                semantic_terms.append(term)
        is_semantic = len(semantic_terms) > 0

        # 7. Clarification / Vague Cue Matching
        has_vague_cue = bool(RE_VAGUE_COMPLAINT.search(cleaned))
        has_status_cue = bool(RE_STATUS_INQUIRY.search(cleaned))

        # Clarification trigger if asking about status without any ID, or reporting vague complaints, or counting without customer
        is_clarification = (
            has_vague_cue or
            (has_status_cue and not ticket_ids and not work_order_ids) or
            (is_count and not matched_cust)
        )

        return ExtractedEntities(
            ticket_ids=ticket_ids,
            work_order_ids=work_order_ids,
            circuit_refs=circuit_refs,
            customer_codes=customer_codes,
            customer_name_raw=matched_name_raw,
            matched_customer=matched_cust,
            status_filter=status_filter,
            semantic_terms=semantic_terms,
            is_bridge_query=is_bridge,
            is_reverse_bridge=is_reverse_bridge,
            is_count_query=is_count,
            is_inventory_query=is_inventory,
            is_semantic_query=is_semantic,
            is_clarification_trigger=is_clarification,
            has_sql_injection=has_sql_injection
        )
