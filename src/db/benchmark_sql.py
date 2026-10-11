"""
NOC Intelligence — Stage 2 SQL Query Latency & Query Plan Inspector
Measures exact latency and inspects EXPLAIN QUERY PLAN for all domain queries.
"""

import time
import sqlite3
from src.db.schema import get_db_connection, DEFAULT_DB_PATH
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


def benchmark_function(name, func, *args, iterations=100):
    # Warmup
    func(*args)

    start = time.perf_counter_ns()
    for _ in range(iterations):
        res = func(*args)
    duration_ns = time.perf_counter_ns() - start
    avg_ms = (duration_ns / iterations) / 1_000_000
    return avg_ms, res


def inspect_query_plans():
    print("=" * 70)
    print("EXPLAIN QUERY PLAN INSPECTION")
    print("=" * 70)

    queries = {
        "1. query_ticket_status": (
            """SELECT t.ticket_id, t.customer_id, c.company_name_en, cir.site_name, t.status
               FROM trouble_tickets t
               JOIN customers c ON t.customer_id = c.customer_id
               JOIN circuits cir ON t.circuit_id = cir.circuit_id
               WHERE t.ticket_id = ?;""",
            ("INC-2025-00001",)
        ),
        "2. query_work_order_detail": (
            "SELECT work_order_id, status, root_cause_category FROM work_orders WHERE work_order_id = ?;",
            ("WO-700001",)
        ),
        "3. query_ticket_work_orders (bridge)": (
            """SELECT w.work_order_id, l.relationship_type, w.status
               FROM ticket_work_order_links l
               JOIN work_orders w ON l.work_order_id = w.work_order_id
               WHERE l.ticket_id = ?;""",
            ("INC-2025-00001",)
        ),
        "4. count_customer_tickets (aggregate)": (
            "SELECT COUNT(*) FROM trouble_tickets WHERE customer_id = ? AND status = ?;",
            (1, "IN_PROGRESS")
        ),
        "5. query_customer_circuits": (
            """SELECT cir.circuit_id, cir.circuit_ref, c.company_name_en
               FROM circuits cir
               JOIN customers c ON cir.customer_id = c.customer_id
               WHERE c.company_name_en LIKE ?;""",
            ("%Apex Logistics%",)
        ),
        "6. query_work_order_activities": (
            "SELECT activity_id, log_detail FROM work_order_activities WHERE work_order_id = ? ORDER BY activity_seq ASC;",
            ("WO-700001",)
        ),
        "7. query_work_order_tickets (reverse bridge)": (
            """SELECT t.ticket_id, c.company_name_en, t.status
               FROM ticket_work_order_links l
               JOIN trouble_tickets t ON l.ticket_id = t.ticket_id
               JOIN customers c ON t.customer_id = c.customer_id
               WHERE l.work_order_id = ?;""",
            ("WO-700001",)
        )
    }

    with get_db_connection(DEFAULT_DB_PATH) as conn:
        cursor = conn.cursor()
        for label, (sql, params) in queries.items():
            print(f"\n[{label}]")
            cursor.execute(f"EXPLAIN QUERY PLAN {sql}", params)
            plans = cursor.fetchall()
            for p in plans:
                # plan format: (id, parent, notused, detail)
                print(f"   -> {p[3]}")


def run_latency_benchmarks():
    print("\n" + "=" * 70)
    print("MEASURED QUERY LATENCY BENCHMARKS (100 Iterations Each)")
    print("=" * 70)

    benchmarks = [
        ("query_ticket_status('INC-2025-00001')", query_ticket_status, ("INC-2025-00001",)),
        ("query_work_order_detail('WO-700001')", query_work_order_detail, ("WO-700001",)),
        ("query_ticket_work_orders('INC-2025-00001')", query_ticket_work_orders, ("INC-2025-00001",)),
        ("count_customer_tickets('Apex Logistics')", count_customer_tickets, ("Apex Logistics",)),
        ("count_customer_tickets('Apex', 'OPEN')", count_customer_tickets, ("Apex Logistics", "OPEN")),
        ("query_customer_circuits('Apex Logistics')", query_customer_circuits, ("Apex Logistics",)),
        ("query_work_order_activities('WO-700001')", query_work_order_activities, ("WO-700001",)),
        ("query_ticket_history('INC-2025-00001')", query_ticket_history, ("INC-2025-00001",)),
        ("query_work_order_tickets('WO-700001')", query_work_order_tickets, ("WO-700001",)),
    ]

    for label, fn, args in benchmarks:
        avg_ms, res = benchmark_function(label, fn, *args, iterations=100)
        items_count = len(res) if isinstance(res, (list, dict)) else 1
        print(f"  ⚡ {label:<48} : {avg_ms:6.3f} ms (result: {items_count} items/fields)")


if __name__ == "__main__":
    inspect_query_plans()
    run_latency_benchmarks()
