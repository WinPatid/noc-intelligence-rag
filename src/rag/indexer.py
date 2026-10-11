"""
NOC Intelligence — ChromaDB Semantic Indexing Subsystem
Serializes structured relational records from SQLite into semantic text chunks
and indexes them into ChromaDB with deterministic IDs and rich metadata.
"""

import os
os.environ["ANONYMIZED_TELEMETRY"] = "false"
import time
from typing import Dict, Any, Optional

import chromadb
from chromadb.config import Settings

from ..db.schema import get_db_connection, DEFAULT_DB_PATH
from .embedder import get_default_embedder, SentenceTransformerEmbedder

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_CHROMA_DIR = os.path.join(ROOT_DIR, "data", "chroma_db")


class SemanticIndexer:
    """
    Manages indexing of operational knowledge into ChromaDB.
    Indexed Collections:
    1. 'work_orders': 3,000 engineering resolution chunks
    2. 'trouble_tickets': 2,500 customer incident symptom chunks
    """

    def __init__(
        self,
        chroma_dir: str = DEFAULT_CHROMA_DIR,
        embedder: Optional[SentenceTransformerEmbedder] = None
    ):
        self.chroma_dir = chroma_dir
        os.makedirs(self.chroma_dir, exist_ok=True)
        self.embedder = embedder or get_default_embedder()
        self.client = chromadb.PersistentClient(
            path=self.chroma_dir,
            settings=Settings(anonymized_telemetry=False)
        )

    def _get_or_create_collection(self, name: str, reset: bool = False):
        if reset:
            try:
                self.client.delete_collection(name=name)
            except Exception:
                pass
        return self.client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"}
        )

    def index_work_orders(
        self,
        db_path: str = None,
        batch_size: int = 250,
        reset: bool = False
    ) -> Dict[str, Any]:
        """
        Indexes engineering work orders from SQLite.
        Document Format:
          WorkOrder_ID, Root Cause, Specific Cause, Action Taken, Equipment, Team, Status
        Metadata:
          work_order_id, root_cause_category, equipment_type, status, assigned_team
        """
        collection = self._get_or_create_collection("work_orders", reset=reset)

        sql = """
            SELECT
                work_order_id,
                status,
                root_cause_category,
                root_cause_category_th,
                specific_cause,
                action_taken,
                equipment_type,
                equipment_model,
                assigned_team,
                field_engineer_alias
            FROM work_orders
            ORDER BY work_order_id ASC;
        """

        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(sql)
            rows = cursor.fetchall()

        total = len(rows)
        print(f"[*] Indexing {total} work orders into ChromaDB...")

        batch_docs, batch_ids, batch_metas = [], [], []
        t0 = time.perf_counter()

        for idx, r in enumerate(rows):
            wo_id = r["work_order_id"]
            doc_text = (
                f"WorkOrder_ID: {wo_id}\n"
                f"Root Cause: {r['root_cause_category']} ({r['root_cause_category_th']})\n"
                f"Specific Cause: {r['specific_cause']}\n"
                f"Action Taken: {r['action_taken']}\n"
                f"Equipment: {r['equipment_type']} ({r['equipment_model']})\n"
                f"Assigned Team: {r['assigned_team']}\n"
                f"Status: {r['status']}"
            )
            metadata = {
                "work_order_id": str(wo_id),
                "root_cause_category": str(r["root_cause_category"]),
                "equipment_type": str(r["equipment_type"]),
                "status": str(r["status"]),
                "assigned_team": str(r["assigned_team"])
            }

            batch_docs.append(doc_text)
            batch_ids.append(f"wo_{wo_id}")
            batch_metas.append(metadata)

            if len(batch_docs) >= batch_size:
                vecs = self.embedder.embed_documents(batch_docs)
                collection.upsert(
                    documents=batch_docs,
                    embeddings=vecs,
                    ids=batch_ids,
                    metadatas=batch_metas
                )
                batch_docs, batch_ids, batch_metas = [], [], []

        if batch_docs:
            vecs = self.embedder.embed_documents(batch_docs)
            collection.upsert(
                documents=batch_docs,
                embeddings=vecs,
                ids=batch_ids,
                metadatas=batch_metas
            )

        duration = time.perf_counter() - t0
        count = collection.count()
        print(f"  [+] Work orders indexed: {count} documents in {duration:.2f}s")
        return {"collection": "work_orders", "count": count, "duration_seconds": duration}

    def index_trouble_tickets(
        self,
        db_path: str = None,
        batch_size: int = 250,
        reset: bool = False
    ) -> Dict[str, Any]:
        """
        Indexes customer trouble tickets joined with customer and circuit info.
        Document Format:
          Ticket_ID, Customer, Circuit, Site, Severity, Status, Category, Reported Symptom
        Metadata:
          ticket_id, customer_id, company_name_en, circuit_ref, severity, status, symptom_category
        """
        collection = self._get_or_create_collection("trouble_tickets", reset=reset)

        sql = """
            SELECT
                t.ticket_id,
                t.customer_id,
                c.company_name_en,
                c.company_name_th,
                t.circuit_ref,
                cir.site_name,
                cir.region,
                t.severity,
                t.status,
                t.symptom_category,
                t.reported_symptom
            FROM trouble_tickets t
            JOIN customers c ON t.customer_id = c.customer_id
            JOIN circuits cir ON t.circuit_id = cir.circuit_id
            ORDER BY t.ticket_id ASC;
        """

        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(sql)
            rows = cursor.fetchall()

        total = len(rows)
        print(f"[*] Indexing {total} trouble tickets into ChromaDB...")

        batch_docs, batch_ids, batch_metas = [], [], []
        t0 = time.perf_counter()

        for idx, r in enumerate(rows):
            tid = r["ticket_id"]
            doc_text = (
                f"Ticket_ID: {tid}\n"
                f"Customer: {r['company_name_en']} ({r['company_name_th']})\n"
                f"Circuit: {r['circuit_ref']} - {r['site_name']} ({r['region']})\n"
                f"Severity: {r['severity']}\n"
                f"Status: {r['status']}\n"
                f"Category: {r['symptom_category']}\n"
                f"Reported Symptom: {r['reported_symptom']}"
            )
            metadata = {
                "ticket_id": str(tid),
                "customer_id": int(r["customer_id"]),
                "company_name_en": str(r["company_name_en"]),
                "circuit_ref": str(r["circuit_ref"]),
                "severity": str(r["severity"]),
                "status": str(r["status"]),
                "symptom_category": str(r["symptom_category"])
            }

            batch_docs.append(doc_text)
            batch_ids.append(f"inc_{tid}")
            batch_metas.append(metadata)

            if len(batch_docs) >= batch_size:
                vecs = self.embedder.embed_documents(batch_docs)
                collection.upsert(
                    documents=batch_docs,
                    embeddings=vecs,
                    ids=batch_ids,
                    metadatas=batch_metas
                )
                batch_docs, batch_ids, batch_metas = [], [], []

        if batch_docs:
            vecs = self.embedder.embed_documents(batch_docs)
            collection.upsert(
                documents=batch_docs,
                embeddings=vecs,
                ids=batch_ids,
                metadatas=batch_metas
            )

        duration = time.perf_counter() - t0
        count = collection.count()
        print(f"  [+] Trouble tickets indexed: {count} documents in {duration:.2f}s")
        return {"collection": "trouble_tickets", "count": count, "duration_seconds": duration}

    def build_all(self, db_path: str = None, reset: bool = True) -> Dict[str, Any]:
        """Builds both collections and returns summary metrics."""
        print("=" * 70)
        print("BUILDING CHROMADB SEMANTIC VECTOR INDEX")
        print("=" * 70)
        res_wo = self.index_work_orders(db_path=db_path, reset=reset)
        res_tt = self.index_trouble_tickets(db_path=db_path, reset=reset)
        total_docs = res_wo["count"] + res_tt["count"]
        total_time = res_wo["duration_seconds"] + res_tt["duration_seconds"]
        print(f"✅ Total indexed documents: {total_docs} across 2 collections in {total_time:.2f}s")
        return {
            "work_orders": res_wo,
            "trouble_tickets": res_tt,
            "total_documents": total_docs,
            "total_duration_seconds": total_time
        }


def build_or_update_index(db_path: str = None, reset: bool = False) -> Dict[str, Any]:
    indexer = SemanticIndexer()
    return indexer.build_all(db_path=db_path, reset=reset)


if __name__ == "__main__":
    indexer = SemanticIndexer()
    indexer.build_all(reset=True)
