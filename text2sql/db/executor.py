"""DBQueryExecutor — executes SQL queries against Spider SQLite databases."""
from __future__ import annotations

import os
import re
import sqlite3
from dataclasses import dataclass
from typing import Optional


QUERY_TIMEOUT = 10  # seconds — prevents runaway queries during training


@dataclass
class QueryResult:
    success : bool
    rows    : Optional[set]
    error   : Optional[str]

    @property
    def is_empty(self) -> bool:
        return self.success and len(self.rows) == 0


class DBQueryExecutor:
    """
    Executes SQL queries against Spider SQLite databases.
    Single responsibility: run queries, return results.
    No reward logic, no training logic.
    """

    def __init__(self, db_root: str = "dataset/database"):
        self.db_root = db_root

    def db_path(self, db_id: str) -> str:
        """Resolve database file — supports both .db and .sqlite extensions."""
        for ext in (".db", ".sqlite"):
            path = os.path.join(self.db_root, db_id, f"{db_id}{ext}")
            if os.path.exists(path):
                return path
        return os.path.join(self.db_root, db_id, f"{db_id}.db")

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
        """Check SQL validity without executing (uses EXPLAIN QUERY PLAN)."""
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
