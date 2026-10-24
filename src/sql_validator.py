"""Validate LLM-generated SQL before it ever touches the database.

Defence in depth: this is layer 1 (static checks). Layer 2 is the read-only
SQLite connection with an authorizer (see database.py).
"""

from __future__ import annotations

import re
from collections.abc import Iterable

MAX_SQL_LENGTH = 4000

FORBIDDEN_KEYWORDS = {
    "insert", "update", "delete", "drop", "alter", "create",
    "truncate", "attach", "detach", "pragma", "vacuum", "reindex", "analyze",
    "begin", "commit", "rollback", "savepoint", "release", "load_extension",
    "grant", "revoke", "merge", "call", "exec", "execute",
}

# Quoted strings and quoted identifiers are masked so words inside them
# (e.g. 'delete' as a value, or a column called "Update Date") don't trigger rules.
_QUOTED = re.compile(r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"|`[^`]*`|\[[^\]]*\]")


class SQLValidationError(Exception):
    """Raised when generated SQL is not safe to run."""


def _mask(sql: str) -> str:
    # \x00 can never match the identifier patterns used below.
    return _QUOTED.sub(" \x00 ", sql)


def validate_sql(
    sql: str,
    table_name: str = "data",
    max_rows: int = 200,
    allowed_columns: Iterable[str] | None = None,
) -> str:
    """Return a cleaned, row-limited SELECT, or raise SQLValidationError.

    If ``allowed_columns`` is given, every double-quoted identifier must be a known
    column, the table name, or an alias introduced with AS. This matters because SQLite
    silently treats an unknown "double-quoted" name as a *string literal*, which would
    make a wrong column name return nonsense instead of an error.
    """
    if not sql or not sql.strip():
        raise SQLValidationError("The generated SQL was empty.")

    cleaned = sql.strip()
    if len(cleaned) > MAX_SQL_LENGTH:
        raise SQLValidationError("The generated SQL is too long.")

    # Drop trailing semicolons (a single statement is fine).
    cleaned = cleaned.rstrip().rstrip(";").rstrip()
    masked = _mask(cleaned)

    if "'" in masked or '"' in masked or "`" in masked:
        raise SQLValidationError("The SQL has an unterminated quote.")
    if "--" in masked or "/*" in masked or "*/" in masked:
        raise SQLValidationError("SQL comments are not allowed.")
    if ";" in masked:
        raise SQLValidationError("Only a single SQL statement is allowed.")

    lowered = masked.lower()
    first_word = re.match(r"\s*(\w+)", lowered)
    if not first_word or first_word.group(1) not in ("select", "with"):
        raise SQLValidationError("Only SELECT queries are allowed.")

    words = set(re.findall(r"[a-z_][a-z0-9_]*", lowered))
    blocked = sorted(words & FORBIDDEN_KEYWORDS)
    if blocked:
        raise SQLValidationError(f"Forbidden keyword(s): {', '.join(blocked)}.")

    if "sqlite_" in lowered:
        raise SQLValidationError("Access to SQLite system tables is not allowed.")

    # Every table after FROM / JOIN must be our table (or a CTE defined in the query).
    cte_names = set(re.findall(r"\b([a-z_][a-z0-9_]*)\s+as\s*\(", lowered))
    allowed = {table_name.lower()} | cte_names
    for match in re.finditer(r"\b(?:from|join)\s+([a-z_][a-z0-9_]*)", lowered):
        if match.group(1) not in allowed:
            raise SQLValidationError(f"Unknown table: {match.group(1)}.")

    # The masked text hides quoted table names; check them on the original too.
    for match in re.finditer(r"\b(?:from|join)\s+[\"`\[]([^\"`\]]+)[\"`\]]", cleaned, re.I):
        if match.group(1).lower() not in allowed:
            raise SQLValidationError(f"Unknown table: {match.group(1)}.")

    if allowed_columns is not None:
        known = {c.lower() for c in allowed_columns} | allowed
        for alias in re.findall(r"\bas\s+\"((?:[^\"]|\"\")*)\"", cleaned, re.I):
            known.add(alias.lower())
        for alias in re.findall(r"\bas\s+([a-z_][a-z0-9_]*)", lowered):
            known.add(alias)
        for name in re.findall(r'"((?:[^"]|"")*)"', cleaned):
            if name.replace('""', '"').lower() not in known:
                raise SQLValidationError(f'Unknown column: "{name}".')

    # Enforce a row cap.
    if not re.search(r"\blimit\b", lowered):
        cleaned = f"{cleaned} LIMIT {int(max_rows)}"

    return cleaned
