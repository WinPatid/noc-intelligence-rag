"""
NOC Intelligence — SQLite Relational Database Schema
Defines the schema for all six entity tables with full constraints,
foreign-key enforcement, and justified performance indexes.
"""

import os
import sqlite3

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DB_PATH = os.path.join(ROOT_DIR, "data", "noc_intelligence.db")


def get_db_connection(db_path: str = None) -> sqlite3.Connection:
    """
    Opens an SQLite connection with strict foreign key enforcement
    and dictionary-like row factory.
    """
    path = db_path or DEFAULT_DB_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    """
    Initializes all six entity tables and justified indexes.
    Uses CREATE TABLE IF NOT EXISTS without dropping existing structures.
    """
    cursor = conn.cursor()

    # 1. customers (50 fictional enterprise accounts)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS customers (
            customer_id         INTEGER PRIMARY KEY,
            customer_code       TEXT NOT NULL UNIQUE,
            company_name_en     TEXT NOT NULL UNIQUE,
            company_name_th     TEXT NOT NULL UNIQUE,
            company_short_name   TEXT NOT NULL,
            industry_segment    TEXT NOT NULL,
            sla_tier            TEXT NOT NULL,
            account_status      TEXT NOT NULL DEFAULT 'Active',
            created_at          TEXT NOT NULL
        );
    """)

    # 2. circuits (300 leased telecommunication links)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS circuits (
            circuit_id          INTEGER PRIMARY KEY,
            circuit_ref         TEXT NOT NULL UNIQUE,
            customer_id         INTEGER NOT NULL REFERENCES customers(customer_id) ON DELETE RESTRICT,
            site_name           TEXT NOT NULL,
            region              TEXT NOT NULL,
            bandwidth_mbps      INTEGER NOT NULL,
            interface_type      TEXT NOT NULL,
            gateway_pop         TEXT NOT NULL,
            status              TEXT NOT NULL DEFAULT 'ACTIVE',
            created_at          TEXT NOT NULL
        );
    """)

    # 3. trouble_tickets (2,500 customer incident tickets)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trouble_tickets (
            ticket_id           TEXT PRIMARY KEY,
            customer_id         INTEGER NOT NULL REFERENCES customers(customer_id) ON DELETE RESTRICT,
            circuit_id          INTEGER NOT NULL REFERENCES circuits(circuit_id) ON DELETE RESTRICT,
            circuit_ref         TEXT NOT NULL,
            severity            TEXT NOT NULL,
            status              TEXT NOT NULL,
            symptom_category    TEXT NOT NULL,
            reported_symptom    TEXT NOT NULL,
            contact_person      TEXT NOT NULL,
            contact_phone       TEXT NOT NULL,
            dt_opened           TEXT NOT NULL,
            dt_closed           TEXT
        );
    """)

    # 4. work_orders (3,000 engineering dispatch work orders)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS work_orders (
            work_order_id       TEXT PRIMARY KEY,
            status              TEXT NOT NULL,
            root_cause_category TEXT NOT NULL,
            root_cause_category_th TEXT NOT NULL,
            specific_cause      TEXT NOT NULL,
            action_taken        TEXT NOT NULL,
            equipment_type      TEXT NOT NULL,
            equipment_model     TEXT NOT NULL,
            assigned_team       TEXT NOT NULL,
            field_engineer_alias TEXT NOT NULL,
            dt_dispatched       TEXT NOT NULL,
            dt_restored         TEXT,
            dt_closed           TEXT
        );
    """)

    # 5. ticket_work_order_links (3,200 bridge records for 1:M and M:N relationships)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ticket_work_order_links (
            link_id             INTEGER PRIMARY KEY,
            ticket_id           TEXT NOT NULL REFERENCES trouble_tickets(ticket_id) ON DELETE RESTRICT,
            work_order_id       TEXT NOT NULL REFERENCES work_orders(work_order_id) ON DELETE RESTRICT,
            relationship_type   TEXT NOT NULL,
            dt_linked           TEXT NOT NULL,
            UNIQUE(ticket_id, work_order_id)
        );
    """)

    # 6. work_order_activities (6,000 technician timeline logs)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS work_order_activities (
            activity_id         INTEGER PRIMARY KEY,
            work_order_id       TEXT NOT NULL REFERENCES work_orders(work_order_id) ON DELETE RESTRICT,
            activity_seq        INTEGER NOT NULL,
            activity_type       TEXT NOT NULL,
            log_detail          TEXT NOT NULL,
            operator_alias      TEXT NOT NULL,
            timestamp           TEXT NOT NULL
        );
    """)

    # ── Justified Indexes ─────────────────────────────────────────────────────
    # Customer name lookups (Router & Intent matching)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_customers_name_en ON customers(company_name_en);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_customers_name_th ON customers(company_name_th);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_customers_short_name ON customers(company_short_name);")

    # Circuits lookups by customer and reference ID
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_circuits_customer_id ON circuits(customer_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_circuits_circuit_ref ON circuits(circuit_ref);")

    # Trouble tickets lookups by customer, circuit, status, and time
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_tickets_customer_id ON trouble_tickets(customer_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_tickets_circuit_id ON trouble_tickets(circuit_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_tickets_status ON trouble_tickets(status);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_tickets_dt_opened ON trouble_tickets(dt_opened);")

    # Bridge table traversal indexes in both directions
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_links_ticket_id ON ticket_work_order_links(ticket_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_links_work_order_id ON ticket_work_order_links(work_order_id);")

    # Work order activities chronological lookup
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_activities_wo_time ON work_order_activities(work_order_id, timestamp);")

    conn.commit()


if __name__ == "__main__":
    print(f"[*] Initializing database schema at: {DEFAULT_DB_PATH}")
    with get_db_connection(DEFAULT_DB_PATH) as c:
        init_schema(c)
    print("✅ Schema initialized successfully.")
