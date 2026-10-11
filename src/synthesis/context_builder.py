"""
NOC Intelligence — Stage 5 Evidence Context Builder
Transforms structured retrieval results into sanitized, deduplicated,
and source-tagged prompt contexts for grounded LLM synthesis.
"""

import re
from typing import Dict, Any, List, Tuple, Optional

from src.router.models import RoutingResult, RouteType, IntentType
from src.synthesis.models import CitationItem


# Escapes instruction tokens that could attempt prompt injection from retrieved text
INJECTION_SANITIZE_PATTERN = re.compile(
    r"(ignore\s+(all\s+)?(previous|prior)\s+instructions|system\s*:\s*|human\s*:\s*|<script>|</script>)",
    re.IGNORECASE
)


def sanitize_text(text: str) -> str:
    """Neutralizes potential prompt injection phrases in untrusted retrieved text."""
    if not text:
        return ""
    return INJECTION_SANITIZE_PATTERN.sub("[REDACTED_PROMPT_DIRECTIVE]", text).strip()


class EvidenceContextBuilder:
    """
    Constructs an inert, structured, citation-tagged evidence context block
    from router and dispatcher retrieval results.
    """

    def __init__(self, max_chars: int = 4000):
        self.max_chars = max_chars

    def build_context(
        self,
        routing_result: RoutingResult
    ) -> Tuple[str, Dict[str, CitationItem], bool]:
        """
        Builds the grounded evidence context.
        Returns:
          (context_block: str, valid_citations_map: Dict[str, CitationItem], is_empty: bool)
        """
        results = routing_result.retrieval_results
        route_type = routing_result.decision.route_type
        intent = routing_result.decision.normalized_intent

        if not results or not routing_result.success:
            return ("", {}, True)

        seen_sources: Dict[str, CitationItem] = {}
        chunks: List[str] = []

        # ── 1. SQL Route Handling ─────────────────────────────────────────────
        if route_type == RouteType.SQL_LOOKUP:
            self._format_sql_results(results, intent, chunks, seen_sources)

        # ── 2. Semantic Route Handling ────────────────────────────────────────
        elif route_type == RouteType.SEMANTIC_SEARCH:
            self._format_semantic_results(results, intent, chunks, seen_sources)

        # ── 3. Hybrid Route Handling ──────────────────────────────────────────
        elif route_type == RouteType.HYBRID_SEARCH:
            self._format_hybrid_results(results, intent, chunks, seen_sources)

        if not chunks:
            return ("", {}, True)

        # Assemble evidence context
        assembled = "\n\n".join(chunks)
        if len(assembled) > self.max_chars:
            assembled = assembled[:self.max_chars] + "\n... [Remaining evidence truncated for context limits]"

        context_block = f"<evidence_context>\n{assembled}\n</evidence_context>"
        return (context_block, seen_sources, False)

    def _format_sql_results(
        self,
        results: Dict[str, Any],
        intent: IntentType,
        chunks: List[str],
        seen_sources: Dict[str, CitationItem]
    ):
        entity_type = results.get("entity_type")

        # A. Trouble Ticket Lookup
        if entity_type == "trouble_ticket":
            if not results.get("found") or not results.get("record"):
                chunks.append(f"[DATABASE FACT (Exact SQL Record)]\nTicket ID {results.get('identifier')} was not found in the database.")
                return

            rec = results["record"]
            tid = rec["ticket_id"]
            source_id = f"ticket:{tid}"
            seen_sources[source_id] = CitationItem(
                source_id=source_id,
                entity_type="trouble_ticket",
                identifier=tid,
                label=f"Ticket {tid} ({rec.get('status')})"
            )
            chunks.append(
                f"[SOURCE: {source_id}] [DATABASE FACT (Exact SQL Record)]\n"
                f"Ticket ID: {tid}\n"
                f"Customer: {sanitize_text(rec.get('company_name_en', ''))} ({sanitize_text(rec.get('company_name_th', ''))})\n"
                f"Circuit: {rec.get('circuit_ref')} - {rec.get('site_name')} ({rec.get('region')})\n"
                f"Status: {rec.get('status')}\n"
                f"Severity: {rec.get('severity')}\n"
                f"Symptom Category: {rec.get('symptom_category')}\n"
                f"Reported Symptom: {sanitize_text(rec.get('reported_symptom', ''))}\n"
                f"Contact Person: {rec.get('contact_person')}\n"
                f"Date Opened: {rec.get('dt_opened')}\n"
                f"Date Closed: {rec.get('dt_closed') or 'Still Open'}"
            )

        # B. Work Order Lookup
        elif entity_type == "work_order":
            if not results.get("found") or not results.get("record"):
                chunks.append(f"[DATABASE FACT (Exact SQL Record)]\nWork Order ID {results.get('identifier')} was not found in the database.")
                return

            rec = results["record"]
            wo_id = rec["work_order_id"]
            source_id = f"work_order:{wo_id}"
            seen_sources[source_id] = CitationItem(
                source_id=source_id,
                entity_type="work_order",
                identifier=wo_id,
                label=f"Work Order {wo_id} ({rec.get('status')})"
            )
            chunks.append(
                f"[SOURCE: {source_id}] [DATABASE FACT (Exact SQL Record)]\n"
                f"Work Order ID: {wo_id}\n"
                f"Status: {rec.get('status')}\n"
                f"Root Cause Category: {rec.get('root_cause_category')} ({rec.get('root_cause_category_th')})\n"
                f"Specific Cause: {sanitize_text(rec.get('specific_cause', ''))}\n"
                f"Action Taken: {sanitize_text(rec.get('action_taken', ''))}\n"
                f"Equipment: {rec.get('equipment_type')} ({rec.get('equipment_model')})\n"
                f"Assigned Team: {rec.get('assigned_team')}\n"
                f"Engineer Alias: {rec.get('field_engineer_alias')}\n"
                f"Date Dispatched: {rec.get('dt_dispatched')}\n"
                f"Date Restored: {rec.get('dt_restored') or 'Pending'}\n"
                f"Date Closed: {rec.get('dt_closed') or 'Pending'}"
            )

        # C. Bridge: Ticket to Work Orders
        elif entity_type == "ticket_work_order_bridge":
            tid = results.get("ticket_id")
            source_id = f"ticket:{tid}"
            seen_sources[source_id] = CitationItem(
                source_id=source_id,
                entity_type="trouble_ticket",
                identifier=tid,
                label=f"Ticket {tid} Bridge Overview"
            )
            linked_wos = results.get("linked_work_orders", [])
            header = (
                f"[SOURCE: {source_id}] [DATABASE FACT (Exact SQL Record)]\n"
                f"Ticket ID: {tid} | Status: {results.get('ticket_status')}\n"
                f"Total Linked Work Orders: {results.get('linked_work_order_count')}"
            )
            chunks.append(header)

            for wo in linked_wos:
                wo_id = wo["work_order_id"]
                wo_src = f"work_order:{wo_id}"
                seen_sources[wo_src] = CitationItem(
                    source_id=wo_src,
                    entity_type="work_order",
                    identifier=wo_id,
                    label=f"Linked Work Order {wo_id}"
                )
                chunks.append(
                    f"[SOURCE: {wo_src}] [LINKED WORK ORDER (Traversed Bridge)]\n"
                    f"Work Order ID: {wo_id} (Relationship: {wo.get('relationship_type')})\n"
                    f"Status: {wo.get('status')}\n"
                    f"Root Cause: {wo.get('root_cause_category')} ({wo.get('root_cause_category_th')})\n"
                    f"Specific Cause: {sanitize_text(wo.get('specific_cause', ''))}\n"
                    f"Action Taken: {sanitize_text(wo.get('action_taken', ''))}\n"
                    f"Assigned Team: {wo.get('assigned_team')} | Linked At: {wo.get('dt_linked')}"
                )

        # D. Bridge: Work Order to Tickets
        elif entity_type == "work_order_ticket_bridge":
            wo_id = results.get("work_order_id")
            wo_src = f"work_order:{wo_id}"
            seen_sources[wo_src] = CitationItem(
                source_id=wo_src,
                entity_type="work_order",
                identifier=wo_id,
                label=f"Work Order {wo_id} Impact Overview"
            )
            impacted = results.get("impacted_tickets", [])
            header = (
                f"[SOURCE: {wo_src}] [DATABASE FACT (Exact SQL Record)]\n"
                f"Work Order ID: {wo_id} | Status: {results.get('work_order_status')}\n"
                f"Total Impacted Customer Tickets: {results.get('impacted_ticket_count')}"
            )
            chunks.append(header)

            for t in impacted:
                tid = t["ticket_id"]
                t_src = f"ticket:{tid}"
                seen_sources[t_src] = CitationItem(
                    source_id=t_src,
                    entity_type="trouble_ticket",
                    identifier=tid,
                    label=f"Impacted Ticket {tid}"
                )
                chunks.append(
                    f"[SOURCE: {t_src}] [AFFECTED TICKET (Reverse Bridge)]\n"
                    f"Ticket ID: {tid} | Customer: {t.get('company_name_en')}\n"
                    f"Circuit: {t.get('circuit_ref')} | Status: {t.get('status')}\n"
                    f"Severity: {t.get('severity')}\n"
                    f"Symptom: {sanitize_text(t.get('reported_symptom', ''))}"
                )

        # E. Circuit Detail
        elif entity_type == "circuit":
            if not results.get("found") or not results.get("record"):
                chunks.append(f"[DATABASE FACT (Exact SQL Record)]\nCircuit {results.get('identifier')} was not found.")
                return

            rec = results["record"]
            ckt = rec["circuit_ref"]
            source_id = f"circuit:{ckt}"
            seen_sources[source_id] = CitationItem(
                source_id=source_id,
                entity_type="circuit",
                identifier=ckt,
                label=f"Circuit {ckt} ({rec.get('site_name')})"
            )
            chunks.append(
                f"[SOURCE: {source_id}] [DATABASE FACT (Exact SQL Record)]\n"
                f"Circuit Ref: {ckt}\n"
                f"Customer: {rec.get('company_name_en')} ({rec.get('company_name_th')})\n"
                f"Site Name: {rec.get('site_name')} ({rec.get('region')})\n"
                f"Bandwidth: {rec.get('bandwidth_mbps')} Mbps\n"
                f"Interface: {rec.get('interface_type')} | Gateway POP: {rec.get('gateway_pop')}\n"
                f"Circuit Status: {rec.get('status')}"
            )

        # F. Customer Ticket Count
        elif entity_type == "customer_ticket_count":
            c_data = results.get("counts", {})
            if not c_data.get("found"):
                chunks.append(f"[DATABASE FACT (Exact SQL Record)]\nCustomer '{results.get('company_name')}' was not found.")
                return

            c_name = c_data.get("company_name_en")
            source_id = f"customer:{c_name}"
            seen_sources[source_id] = CitationItem(
                source_id=source_id,
                entity_type="customer",
                identifier=c_name,
                label=f"Customer Ticket Aggregate: {c_name}"
            )
            chunks.append(
                f"[SOURCE: {source_id}] [DATABASE FACT (Exact SQL Record)]\n"
                f"Customer: {c_name} ({c_data.get('company_name_th')})\n"
                f"Total Tickets: {c_data.get('total_tickets')}\n"
                f"Status Breakdown: {c_data.get('breakdown')}\n"
                f"Applied Status Filter: {c_data.get('status_filter') or 'None (All Statuses)'}"
            )

        # G. Customer Circuits Inventory
        elif entity_type == "customer_circuits":
            c_name = results.get("company_name")
            circuits = results.get("circuits", [])
            source_id = f"customer:{c_name}"
            seen_sources[source_id] = CitationItem(
                source_id=source_id,
                entity_type="customer",
                identifier=c_name,
                label=f"Circuit Inventory: {c_name}"
            )
            circ_lines = [
                f"- {c['circuit_ref']}: {c['site_name']} ({c['region']}) | {c['bandwidth_mbps']} Mbps | {c['status']}"
                for c in circuits
            ]
            chunks.append(
                f"[SOURCE: {source_id}] [DATABASE FACT (Exact SQL Record)]\n"
                f"Customer: {c_name}\n"
                f"Total Leased Circuits: {len(circuits)}\n" +
                "\n".join(circ_lines)
            )

    def _format_semantic_results(
        self,
        results: Dict[str, Any],
        intent: IntentType,
        chunks: List[str],
        seen_sources: Dict[str, CitationItem]
    ):
        hits = results.get("hits") or []
        if not hits and results.get("collection") == "unified":
            hits = (results.get("work_orders") or []) + (results.get("trouble_tickets") or [])

        for idx, hit in enumerate(hits):
            meta = hit.get("metadata", {})
            doc = sanitize_text(hit.get("document", ""))
            score = hit.get("similarity_score", 0.0)

            # Detect source ID
            if "work_order_id" in meta:
                wo_id = meta["work_order_id"]
                source_id = f"work_order:{wo_id}"
                seen_sources[source_id] = CitationItem(
                    source_id=source_id,
                    entity_type="work_order",
                    identifier=wo_id,
                    label=f"Work Order {wo_id} ({meta.get('root_cause_category')})"
                )
            elif "ticket_id" in meta:
                tid = meta["ticket_id"]
                source_id = f"ticket:{tid}"
                seen_sources[source_id] = CitationItem(
                    source_id=source_id,
                    entity_type="trouble_ticket",
                    identifier=tid,
                    label=f"Ticket {tid} ({meta.get('company_name_en')})"
                )
            else:
                source_id = f"record:{idx+1}"

            chunks.append(
                f"[SOURCE: {source_id}] [SIMILAR OPERATIONAL RECORD (Semantic Match, Similarity Score: {score:.4f})]\n"
                f"{doc}"
            )

    def _format_hybrid_results(
        self,
        results: Dict[str, Any],
        intent: IntentType,
        chunks: List[str],
        seen_sources: Dict[str, CitationItem]
    ):
        hybrid_type = results.get("hybrid_type")

        if hybrid_type == "customer_scoped":
            c_name = results.get("company_name_en")
            c_src = f"customer:{c_name}"
            seen_sources[c_src] = CitationItem(
                source_id=c_src,
                entity_type="customer",
                identifier=c_name,
                label=f"Customer Scope: {c_name}"
            )
            chunks.append(
                f"[SOURCE: {c_src}] [STRUCTURED SCOPE (SQL Filter)]\n"
                f"Customer: {c_name} (ID: {results.get('customer_id')})\n"
                f"Total Matched Incidents: {results.get('matched_incidents_count')}"
            )

            for inc in results.get("incidents", []):
                tid = inc["ticket_id"]
                t_src = f"ticket:{tid}"
                score = inc["similarity_score"]
                seen_sources[t_src] = CitationItem(
                    source_id=t_src,
                    entity_type="trouble_ticket",
                    identifier=tid,
                    label=f"Customer Incident {tid}"
                )
                chunks.append(
                    f"[SOURCE: {t_src}] [MATCHED CUSTOMER TICKET (Semantic Score: {score:.4f})]\n"
                    f"{sanitize_text(inc.get('ticket_document', ''))}"
                )

                for wo in inc.get("linked_work_orders", []):
                    wo_id = wo["work_order_id"]
                    wo_src = f"work_order:{wo_id}"
                    seen_sources[wo_src] = CitationItem(
                        source_id=wo_src,
                        entity_type="work_order",
                        identifier=wo_id,
                        label=f"Linked Work Order {wo_id}"
                    )
                    chunks.append(
                        f"[SOURCE: {wo_src}] [LINKED FIELD WORK ORDER (Traversed Bridge)]\n"
                        f"Work Order: {wo_id} | Root Cause: {wo.get('root_cause_category')}\n"
                        f"Specific Cause: {sanitize_text(wo.get('specific_cause', ''))}\n"
                        f"Action Taken: {sanitize_text(wo.get('action_taken', ''))}\n"
                        f"Status: {wo.get('status')} | Team: {wo.get('assigned_team')}"
                    )

        elif hybrid_type == "circuit_scoped":
            ckt = results.get("circuit_ref")
            ckt_src = f"circuit:{ckt}"
            seen_sources[ckt_src] = CitationItem(
                source_id=ckt_src,
                entity_type="circuit",
                identifier=ckt,
                label=f"Circuit Scope: {ckt}"
            )
            chunks.append(
                f"[SOURCE: {ckt_src}] [STRUCTURED SCOPE (SQL Filter)]\n"
                f"Circuit Reference: {ckt}\n"
                f"Circuit Details: {results.get('circuit_details')}\n"
                f"Total Matched Incidents: {results.get('matched_incidents_count')}"
            )

            for inc in results.get("incidents", []):
                tid = inc["ticket_id"]
                t_src = f"ticket:{tid}"
                score = inc["similarity_score"]
                seen_sources[t_src] = CitationItem(
                    source_id=t_src,
                    entity_type="trouble_ticket",
                    identifier=tid,
                    label=f"Circuit Incident {tid}"
                )
                chunks.append(
                    f"[SOURCE: {t_src}] [MATCHED CIRCUIT TICKET (Semantic Score: {score:.4f})]\n"
                    f"{sanitize_text(inc.get('ticket_document', ''))}"
                )
