"""SQLite helpers: build an in-memory database and run read-only queries."""

from __future__ import annotations

import sqlite3
import time

import pandas as pd

from src.data_ingestion import PreparedDataset

_ALLOWED_ACTIONS = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    sqlite3.SQLITE_RECURSIVE,
}


class QueryExecutionError(Exception):
    """Raised when a (validated) query fails or times out."""


def _authorizer(action, arg1, arg2, db_name, source):
    """Layer 2 of the safety net: only reads are ever permitted."""
    if action not in _ALLOWED_ACTIONS:
        return sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_READ and str(arg1).lower().startswith("sqlite_"):
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def build_connection(prepared: PreparedDataset) -> sqlite3.Connection:
    """Load the dataset into an in-memory SQLite DB, then lock it to read-only."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    prepared.sql_frame().to_sql(prepared.table_name, conn, index=False)
    conn.execute("PRAGMA query_only = ON")
    # Python 3.12+: make unknown "double-quoted" names an error instead of a string literal.
    if hasattr(conn, "setconfig") and hasattr(sqlite3, "SQLITE_DBCONFIG_DQS_DML"):
        conn.setconfig(sqlite3.SQLITE_DBCONFIG_DQS_DML, False)
    conn.set_authorizer(_authorizer)
    return conn


def run_query(
    conn: sqlite3.Connection, sql: str, timeout_seconds: int = 10
) -> pd.DataFrame:
    """Execute a validated SELECT and return a DataFrame."""
    deadline = time.monotonic() + timeout_seconds

    def _abort_if_slow() -> int:
        return 1 if time.monotonic() > deadline else 0

    conn.set_progress_handler(_abort_if_slow, 10_000)
    try:
        return pd.read_sql_query(sql, conn)
    except Exception as exc:  # sqlite3.Error, pandas DatabaseError, ...
        message = str(exc)
        if "interrupted" in message.lower():
            raise QueryExecutionError(
                f"The query took longer than {timeout_seconds} seconds."
            ) from exc
        raise QueryExecutionError(message) from exc
    finally:
        conn.set_progress_handler(None, 0)


def describe_schema(prepared: PreparedDataset, examples_per_column: int = 5) -> str:
    """Text description of the table for the LLM prompt (names, types, examples)."""
    df = prepared.df
    lines = [f'Table "{prepared.table_name}" ({len(df):,} rows). Columns:']
    for col in prepared.profile.columns:
        series = df[col.original].dropna()
        detail = ""
        if col.logical_type in ("integer", "float") and not series.empty:
            detail = f"range {series.min():g} to {series.max():g}"
        elif col.logical_type == "datetime" and not series.empty:
            detail = (
                f"range {series.min():%Y-%m-%d} to {series.max():%Y-%m-%d}; "
                "stored as TEXT in YYYY-MM-DD format"
            )
        elif col.logical_type == "text" and not series.empty:
            values = series.astype(str).value_counts().head(examples_per_column).index
            shown = ", ".join(repr(v[:40]) for v in values)
            detail = f"example values: {shown}"
        sql_type = {"integer": "INTEGER", "float": "REAL"}.get(col.logical_type, "TEXT")
        lines.append(
            f'- "{col.sql_name}" {sql_type} (original header: {col.original!r}'
            f"{'; ' + detail if detail else ''})"
        )
    return "\n".join(lines)
