"""
NOC Intelligence — Stage 4 Deterministic Router Facade
Unified supervisor interface providing deterministic classification and retrieval dispatch.
"""

from typing import Optional

from src.router.models import (
    RouteType,
    IntentType,
    ExtractedEntities,
    RoutingDecision,
    RoutingResult
)
from src.router.entity_extractor import EntityExtractor
from src.router.classifier import DeterministicClassifier
from src.router.dispatcher import RoutingDispatcher
from src.rag.retriever import SemanticRetriever


class DeterministicRouter:
    """
    Main supervisor interface for routing NOC operational questions.
    Combines rule-based classification, SQL parameterization, ChromaDB dense retrieval,
    and hybrid bridge correlation.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        retriever: Optional[SemanticRetriever] = None
    ):
        self.db_path = db_path
        self.extractor = EntityExtractor(db_path=db_path)
        self.classifier = DeterministicClassifier(extractor=self.extractor)
        self.dispatcher = RoutingDispatcher(retriever=retriever, db_path=db_path)

    def classify(self, query: str) -> RoutingDecision:
        """Deterministically classifies user question without performing retrieval."""
        return self.classifier.classify(query)

    def dispatch(
        self,
        query: str,
        top_k: int = 5,
        threshold: Optional[float] = None
    ) -> RoutingResult:
        """Classifies and dispatches retrieval to the appropriate subsystem."""
        decision = self.classify(query)
        return self.dispatcher.dispatch(
            decision=decision,
            raw_query=query,
            top_k=top_k,
            threshold=threshold
        )

    def route_and_dispatch(
        self,
        query: str,
        top_k: int = 5,
        threshold: Optional[float] = None
    ) -> RoutingResult:
        """Alias for dispatch."""
        return self.dispatch(query, top_k=top_k, threshold=threshold)
