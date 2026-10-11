"""
NOC Intelligence — Relational Data Loader
Loads the Stage 1 synthetic dataset into SQLite while strictly preserving
relational foreign keys, primary keys, and indexes. Uses atomic transactions.
"""

import os
import sys
import json
import sqlite3
from typing import Dict, Any

from .schema import get_db_connection, init_schema, DEFAULT_DB_PATH

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SYNTHETIC_DATA_DIR = os.path.join(ROOT_DIR, "data", "synthetic")

EXPECTED_COUNTS = {
    "customers": 50,
    "circuits": 300,
    "trouble_tickets": 2500,
    "work_orders": 3000,
    "ticket_work_order_links": 3200,
    "work_order_activities": 6000
}


def load_json_table(table_name: str) -> list[Dict[str, Any]]:
    path = os.path.join(SYNTHETIC_DATA_DIR, f"{table_name}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing synthetic dataset file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_synthetic_dataset(db_path: str = None) -> bool:
    """
    Loads all six synthetic tables into SQLite in dependency order:
    1. customers
    2. circuits
    3. trouble_tickets
    4. work_orders
    5. ticket_work_order_links
    6. work_order_activities

    Uses an explicit transaction with rollback on error.
    """
    path = db_path or DEFAULT_DB_PATH
    print(f"[*] Loading synthetic dataset into SQLite database: {path}")

    conn = get_db_connection(path)
    try:
        init_schema(conn)

        # Atomic transaction
        with conn:
            cursor = conn.cursor()

            # Clean existing records in reverse dependency order for safe repeatability
            print("  [*] Resetting existing records in reverse dependency order...")
            cursor.execute("DELETE FROM work_order_activities;")
            cursor.execute("DELETE FROM ticket_work_order_links;")
            cursor.execute("DELETE FROM trouble_tickets;")
            cursor.execute("DELETE FROM work_orders;")
            cursor.execute("DELETE FROM circuits;")
            cursor.execute("DELETE FROM customers;")

            # 1. customers (50)
            customers = load_json_table("customers")
            cursor.executemany("""
                INSERT INTO customers (
                    customer_id, customer_code, company_name_en, company_name_th,
                    company_short_name, industry_segment, sla_tier, account_status, created_at
                ) VALUES (
                    :customer_id, :customer_code, :company_name_en, :company_name_th,
                    :company_short_name, :industry_segment, :sla_tier, :account_status, :created_at
                );
            """, customers)
            print(f"  [+] Loaded {len(customers)} customers")

            # 2. circuits (300)
            circuits = load_json_table("circuits")
            cursor.executemany("""
                INSERT INTO circuits (
                    circuit_id, circuit_ref, customer_id, site_name, region,
                    bandwidth_mbps, interface_type, gateway_pop, status, created_at
                ) VALUES (
                    :circuit_id, :circuit_ref, :customer_id, :site_name, :region,
                    :bandwidth_mbps, :interface_type, :gateway_pop, :status, :created_at
                );
            """, circuits)
            print(f"  [+] Loaded {len(circuits)} circuits")

            # 3. trouble_tickets (2,500)
            tickets = load_json_table("trouble_tickets")
            cursor.executemany("""
                INSERT INTO trouble_tickets (
                    ticket_id, customer_id, circuit_id, circuit_ref, severity,
                    status, symptom_category, reported_symptom, contact_person,
                    contact_phone, dt_opened, dt_closed
                ) VALUES (
                    :ticket_id, :customer_id, :circuit_id, :circuit_ref, :severity,
                    :status, :symptom_category, :reported_symptom, :contact_person,
                    :contact_phone, :dt_opened, :dt_closed
                );
            """, tickets)
            print(f"  [+] Loaded {len(tickets)} trouble tickets")

            # 4. work_orders (3,000)
            work_orders = load_json_table("work_orders")
            cursor.executemany("""
                INSERT INTO work_orders (
                    work_order_id, status, root_cause_category, root_cause_category_th,
                    specific_cause, action_taken, equipment_type, equipment_model,
                    assigned_team, field_engineer_alias, dt_dispatched, dt_restored, dt_closed
                ) VALUES (
                    :work_order_id, :status, :root_cause_category, :root_cause_category_th,
                    :specific_cause, :action_taken, :equipment_type, :equipment_model,
                    :assigned_team, :field_engineer_alias, :dt_dispatched, :dt_restored, :dt_closed
                );
            """, work_orders)
            print(f"  [+] Loaded {len(work_orders)} work orders")

            # 5. ticket_work_order_links (3,200)
            links = load_json_table("ticket_work_order_links")
            cursor.executemany("""
                INSERT INTO ticket_work_order_links (
                    link_id, ticket_id, work_order_id, relationship_type, dt_linked
                ) VALUES (
                    :link_id, :ticket_id, :work_order_id, :relationship_type, :dt_linked
                );
            """, links)
            print(f"  [+] Loaded {len(links)} ticket-work-order bridge links")

            # 6. work_order_activities (6,000)
            activities = load_json_table("work_order_activities")
            cursor.executemany("""
                INSERT INTO work_order_activities (
                    activity_id, work_order_id, activity_seq, activity_type,
                    log_detail, operator_alias, timestamp
                ) VALUES (
                    :activity_id, :work_order_id, :activity_seq, :activity_type,
                    :log_detail, :operator_alias, :timestamp
                );
            """, activities)
            print(f"  [+] Loaded {len(activities)} work order activities")

        print("  [*] Commit successful. Validating database invariants...")

        # Post-load validation
        validate_database(conn)
        print("✅ Database loading and validation completed successfully.")
        return True

    except Exception as e:
        print(f"❌ Error during database load: {e}", file=sys.stderr)
        conn.rollback()
        raise
    finally:
        conn.close()


def validate_database(conn: sqlite3.Connection) -> None:
    """
    Validates row counts, PRAGMA foreign_key_check, and PRAGMA integrity_check.
    Raises RuntimeError on any failure.
    """
    cursor = conn.cursor()

    # 1. Record counts
    total = 0
    for table, expected in EXPECTED_COUNTS.items():
        cursor.execute(f"SELECT COUNT(*) FROM {table};")
        actual = cursor.fetchone()[0]
        total += actual
        if actual != expected:
            raise RuntimeError(f"Count mismatch in {table}: actual {actual}, expected {expected}")
        print(f"    Verified {table}: {actual} rows")

    if total != 15050:
        raise RuntimeError(f"Total rows mismatch: {total} (expected 15,050)")

    # 2. Foreign key check
    cursor.execute("PRAGMA foreign_key_check;")
    fk_errors = cursor.fetchall()
    if fk_errors:
        raise RuntimeError(f"PRAGMA foreign_key_check found {len(fk_errors)} violations: {fk_errors}")
    print("    Verified PRAGMA foreign_key_check: 0 violations")

    # 3. Integrity check
    cursor.execute("PRAGMA integrity_check;")
    integrity = cursor.fetchone()[0]
    if integrity != "ok":
        raise RuntimeError(f"PRAGMA integrity_check failed: {integrity}")
    print(f"    Verified PRAGMA integrity_check: {integrity}")


if __name__ == "__main__":
    try:
        success = load_synthetic_dataset()
        sys.exit(0 if success else 1)
    except Exception:
        sys.exit(1)
