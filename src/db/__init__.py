"""
NOC Intelligence — Relational Database Subsystem (SQLite)
"""
from .schema import get_db_connection, init_schema, DEFAULT_DB_PATH
from .loader import load_synthetic_dataset
from .sql_engine import (
    query_ticket_status,
    query_work_order_detail,
    query_ticket_work_orders,
    count_customer_tickets,
    query_customer_circuits,
    query_work_order_activities,
    query_ticket_history,
    query_work_order_tickets,
    query_circuit_detail,
    lookup_customer,
    get_all_customers
)

__all__ = [
    "get_db_connection",
    "init_schema",
    "DEFAULT_DB_PATH",
    "load_synthetic_dataset",
    "query_ticket_status",
    "query_work_order_detail",
    "query_ticket_work_orders",
    "count_customer_tickets",
    "query_customer_circuits",
    "query_work_order_activities",
    "query_ticket_history",
    "query_work_order_tickets",
    "query_circuit_detail",
    "lookup_customer",
    "get_all_customers"
]
