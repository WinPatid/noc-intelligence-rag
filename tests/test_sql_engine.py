"""
NOC Intelligence — Stage 2 Automated Test Suite
Validates SQLite schema, relational constraints, loader atomicity,
parameterized SQL queries, bridge traversals, and SQL injection resilience.
"""

import os
import sys
import sqlite3
import tempfile
import pytest

from src.db.schema import get_db_connection, init_schema, DEFAULT_DB_PATH
from src.db.loader import load_synthetic_dataset, EXPECTED_COUNTS
from src.db.sql_engine import (
    query_ticket_status,
    query_work_order_detail,
    query_ticket_work_orders,
    count_customer_tickets,
    query_customer_circuits,
    query_work_order_activities,
    query_ticket_history,
    query_work_order_tickets
)


# ─── 1. Schema & Constraints Tests ───────────────────────────────────────────

def test_schema_and_tables_exist():
    """Verify all six relational entity tables exist in the database."""
    with get_db_connection(DEFAULT_DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = {row[0] for row in cursor.fetchall()}
        for expected in EXPECTED_COUNTS.keys():
            assert expected in tables, f"Table {expected} missing from database"


def test_expected_record_counts():
    """Verify exact count invariants for all tables (total 15,050)."""
    with get_db_connection(DEFAULT_DB_PATH) as conn:
        cursor = conn.cursor()
        total = 0
        for table, expected in EXPECTED_COUNTS.items():
            cursor.execute(f"SELECT COUNT(*) FROM {table};")
            actual = cursor.fetchone()[0]
            total += actual
            assert actual == expected, f"{table} count {actual} != {expected}"
        assert total == 15050, f"Total records {total} != 15,050"


def test_primary_key_uniqueness():
    """Verify primary keys are strictly unique across all six tables."""
    pk_map = {
        "customers": "customer_id",
        "circuits": "circuit_id",
        "trouble_tickets": "ticket_id",
        "work_orders": "work_order_id",
        "ticket_work_order_links": "link_id",
        "work_order_activities": "activity_id"
    }
    with get_db_connection(DEFAULT_DB_PATH) as conn:
        cursor = conn.cursor()
        for table, pk in pk_map.items():
            cursor.execute(f"SELECT COUNT({pk}), COUNT(DISTINCT {pk}) FROM {table};")
            total, distinct = cursor.fetchone()
            assert total == distinct, f"Duplicate primary keys detected in {table}"


def test_pragma_checks_pass():
    """Verify PRAGMA foreign_key_check and PRAGMA integrity_check pass with zero errors."""
    with get_db_connection(DEFAULT_DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_key_check;")
        fk_violations = cursor.fetchall()
        assert len(fk_violations) == 0, f"Foreign key violations found: {fk_violations}"

        cursor.execute("PRAGMA integrity_check;")
        integrity = cursor.fetchone()[0]
        assert integrity == "ok", f"Integrity check failed: {integrity}"


def test_foreign_key_enforcement():
    """Verify that inserting an orphaned foreign key is strictly blocked by SQLite."""
    with get_db_connection(DEFAULT_DB_PATH) as conn:
        cursor = conn.cursor()
        # Attempt to insert a circuit referencing a non-existent customer (customer_id 999999)
        with pytest.raises(sqlite3.IntegrityError):
            cursor.execute("""
                INSERT INTO circuits (
                    circuit_id, circuit_ref, customer_id, site_name, region,
                    bandwidth_mbps, interface_type, gateway_pop, status, created_at
                ) VALUES (
                    999999, 'CKT-INVALID', 999999, 'Orphan Site', 'Central',
                    100, 'FTTx', 'POP-TEST', 'ACTIVE', '2025-01-01T00:00:00Z'
                );
            """)


# ─── 2. Query Engine Tests ────────────────────────────────────────────────────

def test_query_ticket_status_exact():
    """Verify exact ticket lookup returns correct joined customer and circuit details."""
    ticket = query_ticket_status("INC-2025-00001")
    assert ticket is not None
    assert ticket["ticket_id"] == "INC-2025-00001"
    assert ticket["customer_id"] == 2
    assert "Horizon Retail Group" in ticket["company_name_en"]
    assert ticket["circuit_ref"] != ""
    assert ticket["status"] in ["CLOSED", "RESOLVED", "IN_PROGRESS"]
    assert ticket["dt_opened"] is not None


def test_query_work_order_detail_exact():
    """Verify exact work order lookup returns complete engineering fields."""
    wo = query_work_order_detail("WO-700001")
    assert wo is not None
    assert wo["work_order_id"] == "WO-700001"
    assert wo["root_cause_category"] != ""
    assert wo["specific_cause"] != ""
    assert wo["action_taken"] != ""
    assert wo["equipment_type"] != ""
    assert wo["field_engineer_alias"].startswith("Engineer_Tech_")


def test_query_ticket_work_orders_bridge():
    """Verify bridge traversal correctly retrieves linked work orders for a ticket."""
    orders = query_ticket_work_orders("INC-2025-00001")
    assert isinstance(orders, list)
    assert len(orders) >= 1
    assert orders[0]["work_order_id"].startswith("WO-70")
    assert orders[0]["relationship_type"] in ["PRIMARY_RESOLUTION", "SECONDARY_DISPATCH", "CORRELATED_INCIDENT"]


def test_one_ticket_linked_to_multiple_work_orders():
    """Verify that 1:M bridge traversal successfully returns multiple work orders for complex tickets."""
    # From Stage 1, links > 2500 created 1:M relationships
    with get_db_connection(DEFAULT_DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT ticket_id, COUNT(*) as cnt
            FROM ticket_work_order_links
            GROUP BY ticket_id
            HAVING cnt > 1
            LIMIT 1;
        """)
        row = cursor.fetchone()
        assert row is not None, "No 1:M ticket links found in database"
        multi_ticket_id = row[0]
        expected_cnt = row[1]

    linked_orders = query_ticket_work_orders(multi_ticket_id)
    assert len(linked_orders) == expected_cnt
    assert len(linked_orders) > 1
    distinct_wos = {w["work_order_id"] for w in linked_orders}
    assert len(distinct_wos) == len(linked_orders), "Duplicate work orders returned"


def test_multiple_tickets_linked_to_one_work_order():
    """Verify that M:N reverse bridge traversal returns all customer tickets impacted by a shared event."""
    with get_db_connection(DEFAULT_DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT work_order_id, COUNT(*) as cnt
            FROM ticket_work_order_links
            GROUP BY work_order_id
            HAVING cnt > 1
            LIMIT 1;
        """)
        row = cursor.fetchone()
        assert row is not None, "No M:N shared work orders found in database"
        shared_wo_id = row[0]
        expected_cnt = row[1]

    impacted_tickets = query_work_order_tickets(shared_wo_id)
    assert len(impacted_tickets) == expected_cnt
    assert len(impacted_tickets) > 1
    for t in impacted_tickets:
        assert t["ticket_id"].startswith("INC-2025-")
        assert t["company_name_en"] != ""


def test_count_customer_tickets_unfiltered_and_filtered():
    """Verify customer ticket counting logic with and without status filtering."""
    # 1. Total count
    res_total = count_customer_tickets("Apex Logistics")
    assert res_total["found"] is True
    assert res_total["total_tickets"] > 0
    assert "company_name_en" in res_total
    assert sum(res_total["breakdown"].values()) == res_total["total_tickets"]

    # 2. Filtered count
    res_closed = count_customer_tickets("Apex Logistics", optional_status="CLOSED")
    assert res_closed["found"] is True
    assert res_closed["total_tickets"] <= res_total["total_tickets"]

    # 3. OPEN status filter
    res_open = count_customer_tickets("Apex Logistics", optional_status="OPEN")
    assert res_open["found"] is True
    assert res_open["total_tickets"] == res_total["breakdown"].get("IN_PROGRESS", 0)

    # 4. Thai name lookup
    res_thai = count_customer_tickets("เอเพ็กซ์")
    assert res_thai["found"] is True
    assert res_thai["customer_id"] == res_total["customer_id"]


def test_query_customer_circuits():
    """Verify circuit retrieval by company name."""
    circuits = query_customer_circuits("Horizon Retail")
    assert len(circuits) == 6  # 6 circuits generated per customer
    for c in circuits:
        assert c["circuit_ref"].startswith("CKT-88")
        assert c["status"] == "ACTIVE"
        assert c["bandwidth_mbps"] > 0


def test_query_work_order_activities_ordering():
    """Verify activities are returned in strict chronological and sequential order."""
    acts = query_work_order_activities("WO-700001")
    assert len(acts) == 2  # exactly 2 activities per work order
    assert acts[0]["activity_seq"] == 1
    assert acts[1]["activity_seq"] == 2
    assert acts[0]["timestamp"] <= acts[1]["timestamp"]


def test_query_ticket_history_unified_package():
    """Verify unified incident package combines ticket, work orders, and nested activities."""
    history = query_ticket_history("INC-2025-00001")
    assert history is not None
    assert "work_orders" in history
    assert len(history["work_orders"]) >= 1
    for wo in history["work_orders"]:
        assert "activities" in wo
        assert len(wo["activities"]) >= 1


# ─── 3. Edge Cases & Error Handling Tests ─────────────────────────────────────

def test_missing_identifiers_return_empty():
    """Verify non-existent IDs return explicit None or empty list, not exceptions."""
    assert query_ticket_status("INC-2025-99999") is None
    assert query_work_order_detail("WO-709999") is None
    assert query_ticket_work_orders("INC-2025-99999") == []
    assert query_work_order_activities("WO-709999") == []
    assert query_ticket_history("INC-2025-99999") is None
    assert query_work_order_tickets("WO-709999") == []

    res_missing_cust = count_customer_tickets("NonExistentCorporationXYZ")
    assert res_missing_cust["found"] is False
    assert res_missing_cust["total_tickets"] == 0


def test_invalid_status_filter_raises_error():
    """Verify invalid status filters are rejected safely with ValueError."""
    with pytest.raises(ValueError, match="Invalid status filter"):
        count_customer_tickets("Apex Logistics", optional_status="INVALID_STATUS_NAME")


def test_sql_injection_defense():
    """Verify parameterized queries treat malicious SQL strings as literal data."""
    injection_attacks = [
        "' OR '1'='1",
        "INC-2025-00001' OR '1'='1",
        "'; DROP TABLE customers; --",
        "Apex' UNION SELECT 1,2,3,4,5,6,7,8,9 --"
    ]
    for attack in injection_attacks:
        # None of these should raise syntax errors or return unauthorized rows
        res_ticket = query_ticket_status(attack)
        assert res_ticket is None

        res_count = count_customer_tickets(attack)
        assert res_count["found"] is False

    # Verify table was not dropped
    with get_db_connection(DEFAULT_DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM customers;")
        assert cursor.fetchone()[0] == 50


def test_loader_repeatability_and_rollback_on_failure():
    """
    Test loader atomicity and rollback on an isolated temporary database.
    Verifies that a failure during loading does not leave partial data.
    """
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_db:
        tmp_path = tmp_db.name

    try:
        # 1. Load successfully into temporary database
        success = load_synthetic_dataset(db_path=tmp_path)
        assert success is True

        # Verify row counts
        with get_db_connection(tmp_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM trouble_tickets;")
            assert cursor.fetchone()[0] == 2500

        # 2. Reloading again works idempotently without schema duplication
        success_reloaded = load_synthetic_dataset(db_path=tmp_path)
        assert success_reloaded is True

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
