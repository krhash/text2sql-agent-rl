"""BaseReward and RewardResult."""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from text2sql.db.executor import DBQueryExecutor, QueryResult
from text2sql.db.filter import TrainingDataFilter


@dataclass
class RewardResult:
    total          : float
    exec_acc       : float
    valid_sql      : float
    correct_tables : float
    efficiency     : float

    def __str__(self) -> str:
        return (f"reward={self.total:.4f} "
                f"[exec={self.exec_acc:.2f} "
                f"valid={self.valid_sql:.2f} "
                f"tables={self.correct_tables:.2f} "
                f"eff={self.efficiency:.2f}]")


class BaseReward(ABC):
    """
    Abstract base for all reward functions.
    Accepts optional true_sql cache — if provided, true SQL results
    are loaded from disk instead of re-executed on every reward call.
    """

    def __init__(self, db_root: str = "dataset/database",
                 true_sql_cache_path: Optional[str] = None,
                 strict_cache: bool = False):
        self.executor       = DBQueryExecutor(db_root)
        self.true_sql_cache = {}
        self.strict_cache   = strict_cache
        if true_sql_cache_path:
            with open(true_sql_cache_path) as f:
                self.true_sql_cache = json.load(f)
            print(f"Loaded true_sql cache: {len(self.true_sql_cache)} entries "
                  f"from {true_sql_cache_path}")

    @abstractmethod
    def compute(self, db_id: str, pred_sql: str, true_sql: str) -> RewardResult:
        ...

    # ── Shared metric helpers ──────────────────────────────────────────────────

    def _true_sql_result(self, db_id: str, true_sql: str) -> QueryResult:
        """Return true SQL result from cache, or fall back to live execution."""
        key = f"{db_id}||{true_sql.strip()}"
        if key in self.true_sql_cache:
            entry = self.true_sql_cache[key]
            return QueryResult(
                success=entry["success"],
                rows=set(entry["rows"]) if entry["success"] else None,
                error=entry.get("error"),
            )
        if self.strict_cache:
            raise KeyError(
                f"True SQL cache miss — key not found: '{key}'. "
                f"Run preprocess stage to rebuild the cache."
            )
        return self.executor.execute(db_id, true_sql)

    def _exec_acc(self, db_id: str, pred_sql: str, true_sql: str) -> float:
        """1.0 if pred and true SQL return identical result sets, else 0.0."""
        if not TrainingDataFilter.is_reliable(db_id, true_sql):
            return 0.0
        pred = self.executor.execute(db_id, pred_sql)
        true = self._true_sql_result(db_id, true_sql)
        if not pred.success or not true.success:
            return 0.0
        return 1.0 if pred.rows == true.rows else 0.0

    def _valid_sql(self, db_id: str, pred_sql: str) -> float:
        return 1.0 if self.executor.is_valid(db_id, pred_sql) else 0.0

    def _table_f1(self, pred_sql: str, true_sql: str) -> float:
        pred_tables = self._extract_tables(pred_sql)
        true_tables = self._extract_tables(true_sql)
        if not pred_tables or not true_tables:
            return 0.0
        overlap   = pred_tables & true_tables
        precision = len(overlap) / len(pred_tables)
        recall    = len(overlap) / len(true_tables)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def _efficiency(self, pred_sql: str, true_sql: str) -> float:
        def complexity(sql: str) -> int:
            sql = sql.lower()
            return max(0, sql.count('select') - 1) + len(re.findall(r'\bjoin\b', sql))
        delta = complexity(pred_sql) - complexity(true_sql)
        return max(0.0, 1.0 - 0.1 * delta) if delta > 0 else 1.0

    @staticmethod
    def _extract_tables(sql: str) -> set:
        sql = sql.lower()
        return set(re.findall(r'(?:from|join)\s+(\w+)(?:\s+as\s+\w+)?', sql))
