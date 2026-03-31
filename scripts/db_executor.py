"""
DBQueryExecutor — executes SQL queries against Spider SQLite databases.
Single responsibility: run queries and return results. No reward logic.
"""

import os
import re
import sqlite3
from dataclasses import dataclass
from typing import Optional


# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_DB_ROOT = "dataset/database"
QUERY_TIMEOUT   = 10    # seconds — prevents runaway queries during training


# ── Result ────────────────────────────────────────────────────────────────────

@dataclass
class QueryResult:
    success : bool
    rows    : Optional[set]     # set of result rows, order-insensitive
    error   : Optional[str]     # error message if success=False

    @property
    def is_empty(self) -> bool:
        return self.success and len(self.rows) == 0


# ── Executor ──────────────────────────────────────────────────────────────────

class DBQueryExecutor:
    """
    Executes SQL queries against Spider SQLite databases.

    Responsibilities:
        - Resolve database path from db_id
        - Execute a SQL query and return a QueryResult
        - Validate SQL without full execution

    Not responsible for:
        - Reward computation
        - Metric calculation
        - Training logic
    """

    def __init__(self, db_root: str = DEFAULT_DB_ROOT):
        self.db_root = db_root

    # Completely empty databases — all queries return empty results
    EMPTY_DATABASES = frozenset({"music_2"})

    # Partially empty — only specific tables have no data
    EMPTY_TABLES = {
        "sakila_1"  : frozenset({"film_text", "language", "staff", "store"}),
        "formula_1" : frozenset({"pitStops", "lapTimes"}),
    }

    def is_empty_database(self, db_id: str) -> bool:
        """Returns True if the entire database has no data."""
        return db_id in self.EMPTY_DATABASES

    def references_empty_table(self, db_id: str, sql: str) -> bool:
        """
        Returns True if the SQL references a known empty table in this database.
        Uses word boundary matching to avoid false positives on column names.
        """
        empty = self.EMPTY_TABLES.get(db_id)
        if not empty:
            return False
        sql_lower = sql.lower()
        return any(
            re.search(rf'\b{re.escape(table.lower())}\b', sql_lower)
            for table in empty
        )
        """Resolve database path — supports both .db and .sqlite extensions."""
        for ext in (".db", ".sqlite"):
            path = os.path.join(self.db_root, db_id, f"{db_id}{ext}")
            if os.path.exists(path):
                return path
        return os.path.join(self.db_root, db_id, f"{db_id}.db")  # default fallback

    def execute(self, db_id: str, sql: str) -> QueryResult:
        """Execute a SQL query and return a QueryResult."""
        path = self.db_path(db_id)
        if not os.path.exists(path):
            return QueryResult(success=False, rows=None,
                               error=f"Database not found: {path}")
        try:
            conn = sqlite3.connect(path, timeout=QUERY_TIMEOUT)
            conn.execute("PRAGMA query_only = ON")
            conn.text_factory = lambda b: b.decode('utf-8', errors='replace')
            rows = set(str(r) for r in conn.execute(sql).fetchall())
            conn.close()
            return QueryResult(success=True, rows=rows, error=None)
        except sqlite3.Error as e:
            return QueryResult(success=False, rows=None, error=str(e))

    def is_valid(self, db_id: str, sql: str) -> bool:
        """Check if SQL is valid against this database schema without executing."""
        path = self.db_path(db_id)
        if not os.path.exists(path):
            return False
        try:
            conn = sqlite3.connect(path)
            conn.execute(f"EXPLAIN QUERY PLAN {sql}")
            conn.close()
            return True
        except sqlite3.Error:
            return False