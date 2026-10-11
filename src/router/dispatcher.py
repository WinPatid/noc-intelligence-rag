"""
NOC Intelligence — Stage 4 Routing Dispatcher
Dispatches classified decisions to the existing Stage 2 SQL engine or Stage 3
Semantic Retriever, and combines results for Hybrid retrieval.
Enforces safe exception containment (no leaked raw stack traces).
"""

import time
from typing import Optional, Dict, Any, List

from src.db.sql_engine import (
    query_ticket_status,
    query_work_order_detail,
    query_ticket_work_orders,
    query_work_order_tickets,
    query_circuit_detail,
    count_customer_tickets,
    query_customer_circuits,
    query_ticket_history
)
from src.rag.retriever import SemanticRetriever
from src.router.models import (
    RouteType,
    IntentType,
    RoutingDecision,
    RoutingResult
)


class RoutingDispatcher:
    """
    Executes retrieval according to the deterministic routing decision.
    """

    def __init__(
        self,
        retriever: Optional[SemanticRetriever] = None,
        db_path: Optional[str] = None
    ):
        self.retriever = retriever  # Lazy-loaded on first semantic/hybrid call if None
        self.db_path = db_path

    def _get_retriever(self) -> SemanticRetriever:
        if self.retriever is None:
            self.retriever = SemanticRetriever()
        return self.retriever

    def dispatch(
        self,
        decision: RoutingDecision,
        raw_query: str,
        top_k: int = 5,
        threshold: Optional[float] = None
    ) -> RoutingResult:
        t0 = time.perf_counter()

        try:
            # ── 1. CLARIFICATION ──────────────────────────────────────────────
            if decision.route_type == RouteType.CLARIFICATION:
                return RoutingResult(
                    decision=decision,
                    retrieval_results=None,
                    success=True,
                    error_message=None,
                    execution_time_ms=(time.perf_counter() - t0) * 1000
                )

            # ── 2. UNSUPPORTED ────────────────────────────────────────────────
            if decision.route_type == RouteType.UNSUPPORTED:
                return RoutingResult(
                    decision=decision,
                    retrieval_results=None,
                    success=False,
                    error_message=decision.routing_reason,
                    execution_time_ms=(time.perf_counter() - t0) * 1000
                )

            # ── 3. SQL LOOKUP ─────────────────────────────────────────────────
            if decision.route_type == RouteType.SQL_LOOKUP:
                sql_results = self._dispatch_sql(decision)
                return RoutingResult(
                    decision=decision,
                    retrieval_results=sql_results,
                    success=True,
                    error_message=None,
                    execution_time_ms=(time.perf_counter() - t0) * 1000
                )

            # ── 4. SEMANTIC SEARCH ────────────────────────────────────────────
            if decision.route_type == RouteType.SEMANTIC_SEARCH:
                sem_results = self._dispatch_semantic(decision, raw_query, top_k, threshold)
                return RoutingResult(
                    decision=decision,
                    retrieval_results=sem_results,
                    success=True,
                    error_message=None,
                    execution_time_ms=(time.perf_counter() - t0) * 1000
                )

            # ── 5. HYBRID SEARCH ──────────────────────────────────────────────
            if decision.route_type == RouteType.HYBRID_SEARCH:
                hybrid_results = self._dispatch_hybrid(decision, raw_query, top_k, threshold)
                return RoutingResult(
                    decision=decision,
                    retrieval_results=hybrid_results,
                    success=True,
                    error_message=None,
                    execution_time_ms=(time.perf_counter() - t0) * 1000
                )

            # Unhandled fallback
            return RoutingResult(
                decision=decision,
                retrieval_results=None,
                success=False,
                error_message=f"Unhandled route type: {decision.route_type}",
                execution_time_ms=(time.perf_counter() - t0) * 1000
            )

        except Exception as e:
            # Safe exception containment: never expose raw stack traces
            return RoutingResult(
                decision=decision,
                retrieval_results=None,
                success=False,
                error_message=f"Execution error during retrieval: {str(e)}",
                execution_time_ms=(time.perf_counter() - t0) * 1000
            )

    def _dispatch_sql(self, decision: RoutingDecision) -> Dict[str, Any]:
        intent = decision.normalized_intent
        entities = decision.extracted_entities

        if intent == IntentType.TICKET_STATUS:
            tid = entities.ticket_ids[0]
            record = query_ticket_status(tid, db_path=self.db_path)
            return {
                "entity_type": "trouble_ticket",
                "identifier": tid,
                "found": record is not None,
                "record": record
            }

        elif intent == IntentType.BRIDGE_TICKET_TO_WO:
            tid = entities.ticket_ids[0]
            ticket = query_ticket_status(tid, db_path=self.db_path)
            linked_wos = query_ticket_work_orders(tid, db_path=self.db_path)
            return {
                "entity_type": "ticket_work_order_bridge",
                "ticket_id": tid,
                "ticket_found": ticket is not None,
                "ticket_status": ticket.get("status") if ticket else None,
                "linked_work_order_count": len(linked_wos),
                "linked_work_orders": linked_wos
            }

        elif intent == IntentType.WORK_ORDER_DETAIL:
            wo_id = entities.work_order_ids[0]
            record = query_work_order_detail(wo_id, db_path=self.db_path)
            return {
                "entity_type": "work_order",
                "identifier": wo_id,
                "found": record is not None,
                "record": record
            }

        elif intent == IntentType.BRIDGE_WO_TO_TICKETS:
            wo_id = entities.work_order_ids[0]
            wo_record = query_work_order_detail(wo_id, db_path=self.db_path)
            impacted_tickets = query_work_order_tickets(wo_id, db_path=self.db_path)
            return {
                "entity_type": "work_order_ticket_bridge",
                "work_order_id": wo_id,
                "work_order_found": wo_record is not None,
                "work_order_status": wo_record.get("status") if wo_record else None,
                "impacted_ticket_count": len(impacted_tickets),
                "impacted_tickets": impacted_tickets
            }

        elif intent == IntentType.CIRCUIT_DETAIL:
            ckt = entities.circuit_refs[0]
            record = query_circuit_detail(ckt, db_path=self.db_path)
            return {
                "entity_type": "circuit",
                "identifier": ckt,
                "found": record is not None,
                "record": record
            }

        elif intent == IntentType.CUSTOMER_TICKET_COUNT:
            cust = entities.matched_customer
            comp_name = cust.get("company_name_en") if cust else (entities.customer_name_raw or "")
            counts = count_customer_tickets(comp_name, optional_status=entities.status_filter, db_path=self.db_path)
            return {
                "entity_type": "customer_ticket_count",
                "company_name": comp_name,
                "counts": counts
            }

        elif intent == IntentType.CUSTOMER_CIRCUITS:
            cust = entities.matched_customer
            comp_name = cust.get("company_name_en") if cust else (entities.customer_name_raw or "")
            circuits = query_customer_circuits(comp_name, db_path=self.db_path)
            return {
                "entity_type": "customer_circuits",
                "company_name": comp_name,
                "circuit_count": len(circuits),
                "circuits": circuits
            }

        return {"error": f"Unknown SQL intent: {intent}"}

    def _dispatch_semantic(
        self,
        decision: RoutingDecision,
        raw_query: str,
        top_k: int,
        threshold: Optional[float]
    ) -> Dict[str, Any]:
        retriever = self._get_retriever()
        intent = decision.normalized_intent

        if intent == IntentType.SEMANTIC_RESOLUTION:
            hits = retriever.query_work_orders(raw_query, top_k=top_k, threshold=threshold)
            return {
                "collection": "work_orders",
                "total_hits": len(hits),
                "hits": hits
            }
        elif intent == IntentType.SEMANTIC_SYMPTOM:
            hits = retriever.query_trouble_tickets(raw_query, top_k=top_k, threshold=threshold)
            return {
                "collection": "trouble_tickets",
                "total_hits": len(hits),
                "hits": hits
            }
        else:
            multi_hits = retriever.query_all(raw_query, top_k=top_k, threshold=threshold)
            return {
                "collection": "unified",
                "work_orders": multi_hits["work_orders"],
                "trouble_tickets": multi_hits["trouble_tickets"]
            }

    def _dispatch_hybrid(
        self,
        decision: RoutingDecision,
        raw_query: str,
        top_k: int,
        threshold: Optional[float]
    ) -> Dict[str, Any]:
        """
        Executes hybrid retrieval:
        1. Uses SQL entity constraints (customer_id or circuit_ref) to scope the search.
        2. Retrieves dense semantic vector matches in ChromaDB.
        3. Traverses SQLite bridge to find related work orders and activities.
        4. Deduplicates and ranks results by similarity score.
        """
        retriever = self._get_retriever()
        entities = decision.extracted_entities
        intent = decision.normalized_intent

        # Customer-scoped Hybrid Search
        if intent == IntentType.HYBRID_CUSTOMER_INCIDENTS and entities.matched_customer:
            cust = entities.matched_customer
            cust_id = cust["customer_id"]
            cust_name = cust["company_name_en"]

            # Filter ChromaDB trouble_tickets by customer_id metadata
            ticket_hits = retriever.query_trouble_tickets(
                raw_query,
                top_k=top_k,
                where={"customer_id": cust_id},
                threshold=threshold
            )

            # Traverse SQLite bridge for each retrieved ticket to enrich with work order context
            enriched_incidents = []
            seen_ticket_ids = set()

            for hit in ticket_hits:
                tid = hit["metadata"].get("ticket_id")
                if tid in seen_ticket_ids:
                    continue
                seen_ticket_ids.add(tid)

                linked_wos = query_ticket_work_orders(tid, db_path=self.db_path)
                enriched_incidents.append({
                    "ticket_id": tid,
                    "similarity_score": hit["similarity_score"],
                    "cosine_distance": hit["cosine_distance"],
                    "ticket_document": hit["document"],
                    "ticket_metadata": hit["metadata"],
                    "linked_work_orders": linked_wos
                })

            # Sort by similarity score descending
            enriched_incidents.sort(key=lambda x: x["similarity_score"], reverse=True)

            return {
                "hybrid_type": "customer_scoped",
                "customer_id": cust_id,
                "company_name_en": cust_name,
                "matched_incidents_count": len(enriched_incidents),
                "incidents": enriched_incidents
            }

        # Circuit-scoped Hybrid Search
        elif intent == IntentType.HYBRID_CIRCUIT_INCIDENTS and entities.circuit_refs:
            ckt_ref = entities.circuit_refs[0]
            ckt_info = query_circuit_detail(ckt_ref, db_path=self.db_path)

            ticket_hits = retriever.query_trouble_tickets(
                raw_query,
                top_k=top_k,
                where={"circuit_ref": ckt_ref},
                threshold=threshold
            )

            enriched_incidents = []
            for hit in ticket_hits:
                tid = hit["metadata"].get("ticket_id")
                linked_wos = query_ticket_work_orders(tid, db_path=self.db_path)
                enriched_incidents.append({
                    "ticket_id": tid,
                    "similarity_score": hit["similarity_score"],
                    "cosine_distance": hit["cosine_distance"],
                    "ticket_document": hit["document"],
                    "ticket_metadata": hit["metadata"],
                    "linked_work_orders": linked_wos
                })

            enriched_incidents.sort(key=lambda x: x["similarity_score"], reverse=True)

            return {
                "hybrid_type": "circuit_scoped",
                "circuit_ref": ckt_ref,
                "circuit_details": ckt_info,
                "matched_incidents_count": len(enriched_incidents),
                "incidents": enriched_incidents
            }

        # Fallback to standard semantic
        return self._dispatch_semantic(decision, raw_query, top_k, threshold)
