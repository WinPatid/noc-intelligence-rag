"""
NOC Intelligence — Parameterized SQL Query Engine
Implements clean domain query functions for ticket lookups, work orders,
bridge-table traversals, customer metrics, and activity timelines.
Strictly uses parameterized queries (? placeholders) to prevent SQL injection.
"""

import sqlite3
from typing import Optional, List, Dict, Any

from .schema import get_db_connection, DEFAULT_DB_PATH

VALID_TICKET_STATUSES = {"IN_PROGRESS", "RESOLVED", "CLOSED", "OPEN"}


def _row_to_dict(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    return dict(row)


def _rows_to_dicts(rows: List[sqlite3.Row]) -> List[Dict[str, Any]]:
    return [dict(r) for r in rows]


# ─── 1. Direct Ticket Lookup ──────────────────────────────────────────────────

def query_ticket_status(ticket_id: str, db_path: str = None) -> Optional[Dict[str, Any]]:
    """
    Looks up exact trouble ticket status joined with customer and circuit details.
    Returns None if ticket_id is not found.
    """
    sql = """
        SELECT
            t.ticket_id,
            t.customer_id,
            c.company_name_en,
            c.company_name_th,
            t.circuit_id,
            t.circuit_ref,
            cir.site_name,
            cir.region,
            t.severity,
            t.status,
            t.symptom_category,
            t.reported_symptom,
            t.contact_person,
            t.contact_phone,
            t.dt_opened,
            t.dt_closed
        FROM trouble_tickets t
        JOIN customers c ON t.customer_id = c.customer_id
        JOIN circuits cir ON t.circuit_id = cir.circuit_id
        WHERE t.ticket_id = ?;
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(sql, (ticket_id.strip(),))
        row = cursor.fetchone()
        return _row_to_dict(row)


# ─── 2. Direct Work Order Lookup ──────────────────────────────────────────────

def query_work_order_detail(work_order_id: str, db_path: str = None) -> Optional[Dict[str, Any]]:
    """
    Looks up exact engineering work order details.
    Returns None if work_order_id is not found.
    """
    sql = """
        SELECT
            w.work_order_id,
            w.status,
            w.root_cause_category,
            w.root_cause_category_th,
            w.specific_cause,
            w.action_taken,
            w.equipment_type,
            w.equipment_model,
            w.assigned_team,
            w.field_engineer_alias,
            w.dt_dispatched,
            w.dt_restored,
            w.dt_closed
        FROM work_orders w
        WHERE w.work_order_id = ?;
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(sql, (work_order_id.strip(),))
        row = cursor.fetchone()
        return _row_to_dict(row)


# ─── 3. Bridge Traversal: Ticket to Work Orders ───────────────────────────────

def query_ticket_work_orders(ticket_id: str, db_path: str = None) -> List[Dict[str, Any]]:
    """
    Traverses the bridge table to find all work orders linked to a ticket (1:1 and 1:M).
    Returns an empty list if none found or ticket does not exist.
    """
    sql = """
        SELECT
            w.work_order_id,
            l.relationship_type,
            l.dt_linked,
            w.status,
            w.root_cause_category,
            w.root_cause_category_th,
            w.specific_cause,
            w.action_taken,
            w.equipment_type,
            w.equipment_model,
            w.assigned_team,
            w.field_engineer_alias,
            w.dt_dispatched,
            w.dt_restored,
            w.dt_closed
        FROM ticket_work_order_links l
        JOIN work_orders w ON l.work_order_id = w.work_order_id
        WHERE l.ticket_id = ?
        ORDER BY l.dt_linked ASC;
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(sql, (ticket_id.strip(),))
        rows = cursor.fetchall()
        return _rows_to_dicts(rows)


# ─── 4. Customer Ticket Aggregation ───────────────────────────────────────────

def count_customer_tickets(company_name: str, optional_status: str = None, db_path: str = None) -> Dict[str, Any]:
    """
    Counts trouble tickets for a given customer name (matches English, Thai, or short name).
    Supports optional status filtering (IN_PROGRESS, RESOLVED, CLOSED, or OPEN).
    Safely rejects invalid status filters with ValueError.
    """
    if optional_status is not None:
        normalized_status = optional_status.strip().upper()
        if normalized_status not in VALID_TICKET_STATUSES:
            raise ValueError(f"Invalid status filter '{optional_status}'. Valid values: {sorted(VALID_TICKET_STATUSES)}")
    else:
        normalized_status = None

    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()

        # Step 1: Find customer
        find_cust_sql = """
            SELECT customer_id, customer_code, company_name_en, company_name_th, company_short_name
            FROM customers
            WHERE company_name_en LIKE ?
               OR company_name_th LIKE ?
               OR company_short_name LIKE ?
            ORDER BY customer_id ASC
            LIMIT 1;
        """
        search_pattern = f"%{company_name.strip()}%"
        cursor.execute(find_cust_sql, (search_pattern, search_pattern, search_pattern))
        cust = cursor.fetchone()

        if not cust:
            return {
                "found": False,
                "company_name": company_name,
                "customer_id": None,
                "status_filter": normalized_status,
                "total_tickets": 0,
                "breakdown": {}
            }

        cust_id = cust["customer_id"]

        # Step 2: Calculate counts
        if normalized_status == "OPEN":
            count_sql = "SELECT COUNT(*) FROM trouble_tickets WHERE customer_id = ? AND status = 'IN_PROGRESS';"
            cursor.execute(count_sql, (cust_id,))
        elif normalized_status:
            count_sql = "SELECT COUNT(*) FROM trouble_tickets WHERE customer_id = ? AND status = ?;"
            cursor.execute(count_sql, (cust_id, normalized_status))
        else:
            count_sql = "SELECT COUNT(*) FROM trouble_tickets WHERE customer_id = ?;"
            cursor.execute(count_sql, (cust_id,))

        matched_count = cursor.fetchone()[0]

        # Breakdown by status
        breakdown_sql = "SELECT status, COUNT(*) as cnt FROM trouble_tickets WHERE customer_id = ? GROUP BY status;"
        cursor.execute(breakdown_sql, (cust_id,))
        breakdown = {row["status"]: row["cnt"] for row in cursor.fetchall()}

        return {
            "found": True,
            "customer_id": cust_id,
            "company_name_en": cust["company_name_en"],
            "company_name_th": cust["company_name_th"],
            "status_filter": normalized_status,
            "total_tickets": matched_count,
            "breakdown": breakdown
        }


# ─── 5. Customer Circuits Lookup ──────────────────────────────────────────────

def query_customer_circuits(company_name: str, db_path: str = None) -> List[Dict[str, Any]]:
    """
    Returns all telecommunication circuits registered to the specified customer.
    """
    sql = """
        SELECT
            cir.circuit_id,
            cir.circuit_ref,
            c.company_name_en,
            cir.site_name,
            cir.region,
            cir.bandwidth_mbps,
            cir.interface_type,
            cir.gateway_pop,
            cir.status
        FROM circuits cir
        JOIN customers c ON cir.customer_id = c.customer_id
        WHERE c.company_name_en LIKE ?
           OR c.company_name_th LIKE ?
           OR c.company_short_name LIKE ?
        ORDER BY cir.circuit_id ASC;
    """
    search_pattern = f"%{company_name.strip()}%"
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(sql, (search_pattern, search_pattern, search_pattern))
        rows = cursor.fetchall()
        return _rows_to_dicts(rows)


def query_circuit_detail(circuit_ref: str, db_path: str = None) -> Optional[Dict[str, Any]]:
    """
    Looks up exact circuit record joined with customer details by circuit_ref (e.g. 'CKT-880001').
    Returns None if circuit is not found.
    """
    sql = """
        SELECT
            cir.circuit_id,
            cir.circuit_ref,
            cir.customer_id,
            c.company_name_en,
            c.company_name_th,
            cir.site_name,
            cir.region,
            cir.bandwidth_mbps,
            cir.interface_type,
            cir.gateway_pop,
            cir.status
        FROM circuits cir
        JOIN customers c ON cir.customer_id = c.customer_id
        WHERE cir.circuit_ref = ?;
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(sql, (circuit_ref.strip(),))
        row = cursor.fetchone()
        return _row_to_dict(row)


def lookup_customer(query: str, db_path: str = None) -> List[Dict[str, Any]]:
    """
    Finds customers matching query string in English, Thai, or short name.
    """
    sql = """
        SELECT customer_id, customer_code, company_name_en, company_name_th, company_short_name, industry_segment, sla_tier, account_status
        FROM customers
        WHERE company_name_en LIKE ?
           OR company_name_th LIKE ?
           OR company_short_name LIKE ?
        ORDER BY customer_id ASC;
    """
    search_pattern = f"%{query.strip()}%"
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(sql, (search_pattern, search_pattern, search_pattern))
        rows = cursor.fetchall()
        return _rows_to_dicts(rows)


def get_all_customers(db_path: str = None) -> List[Dict[str, Any]]:
    """Returns all customer entities for router caching."""
    sql = "SELECT customer_id, customer_code, company_name_en, company_name_th, company_short_name FROM customers ORDER BY customer_id ASC;"
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        return _rows_to_dicts(rows)


# ─── 6. Work Order Activities Timeline ────────────────────────────────────────

def query_work_order_activities(work_order_id: str, db_path: str = None) -> List[Dict[str, Any]]:
    """
    Retrieves chronological activity log entries for a work order.
    """
    sql = """
        SELECT
            activity_id,
            work_order_id,
            activity_seq,
            activity_type,
            log_detail,
            operator_alias,
            timestamp
        FROM work_order_activities
        WHERE work_order_id = ?
        ORDER BY activity_seq ASC, timestamp ASC;
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(sql, (work_order_id.strip(),))
        rows = cursor.fetchall()
        return _rows_to_dicts(rows)


# ─── 7. Unified Incident Package (History) ────────────────────────────────────

def query_ticket_history(ticket_id: str, db_path: str = None) -> Optional[Dict[str, Any]]:
    """
    Builds a unified operational context package for a trouble ticket:
    - Ticket details
    - All linked work orders
    - Detailed activity logs for each linked work order
    """
    ticket = query_ticket_status(ticket_id, db_path=db_path)
    if not ticket:
        return None

    work_orders = query_ticket_work_orders(ticket_id, db_path=db_path)
    for wo in work_orders:
        wo["activities"] = query_work_order_activities(wo["work_order_id"], db_path=db_path)

    ticket["work_orders"] = work_orders
    return ticket


# ─── 8. Reverse Bridge Traversal: Work Order to Tickets ───────────────────────

def query_work_order_tickets(work_order_id: str, db_path: str = None) -> List[Dict[str, Any]]:
    """
    Reverse bridge traversal: Finds all customer trouble tickets affected by a work order.
    Essential for shared backbone failure analysis (M:N).
    """
    sql = """
        SELECT
            t.ticket_id,
            t.customer_id,
            c.company_name_en,
            t.circuit_ref,
            t.severity,
            t.status,
            t.reported_symptom,
            l.relationship_type,
            l.dt_linked
        FROM ticket_work_order_links l
        JOIN trouble_tickets t ON l.ticket_id = t.ticket_id
        JOIN customers c ON t.customer_id = c.customer_id
        WHERE l.work_order_id = ?
        ORDER BY l.dt_linked ASC;
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(sql, (work_order_id.strip(),))
        rows = cursor.fetchall()
        return _rows_to_dicts(rows)
