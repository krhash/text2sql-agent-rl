"""
Reward functions and training data filters for GRPO training on text-to-SQL.

Depends on DBQueryExecutor for execution — no direct DB access here.
Data quality decisions (empty tables, unreliable examples) live here,
not in the executor.
"""

import re
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from db_executor import DBQueryExecutor, QueryResult


# ── Reward Result ─────────────────────────────────────────────────────────────

@dataclass
class RewardResult:
    total           : float
    exec_acc        : float
    valid_sql       : float
    correct_tables  : float
    efficiency      : float

    def __str__(self) -> str:
        return (f"reward={self.total:.4f} "
                f"[exec={self.exec_acc:.2f} "
                f"valid={self.valid_sql:.2f} "
                f"tables={self.correct_tables:.2f} "
                f"eff={self.efficiency:.2f}]")


# ── Training Data Filter ──────────────────────────────────────────────────────

class TrainingDataFilter:
    """
    Determines whether a training example is reliable for reward computation.
    Encapsulates all data quality knowledge about the Spider dataset.

    This is a training concern — DBQueryExecutor is intentionally unaware of it.
    """

    # Databases where ALL tables are empty — execution always returns empty set
    EMPTY_DATABASES = frozenset({"music_2"})

    # Databases where SOME tables are empty — unreliable for queries using them
    EMPTY_TABLES = {
        "sakila_1"  : frozenset({"film_text", "language", "staff", "store"}),
        "formula_1" : frozenset({"pitStops", "lapTimes"}),
    }

    @classmethod
    def is_reliable(cls, db_id: str, gold_sql: str) -> bool:
        """
        Returns True if execution accuracy is a reliable reward signal
        for this example. Returns False for:
          - Fully empty databases
          - Queries referencing known empty tables
        """
        if db_id in cls.EMPTY_DATABASES:
            return False
        empty_tables = cls.EMPTY_TABLES.get(db_id)
        if empty_tables:
            sql_lower = gold_sql.lower()
            if any(re.search(rf'\b{re.escape(t.lower())}\b', sql_lower)
                   for t in empty_tables):
                return False
        return True


# ── Base Reward ───────────────────────────────────────────────────────────────

class BaseReward(ABC):
    """
    Abstract base for all reward functions.
    Accepts an optional gold_cache path — if provided, gold SQL results
    are loaded from disk instead of re-executed on every reward call.
    """

    def __init__(self, db_root: str, gold_cache_path: Optional[str] = None,
                 strict_cache: bool = False):
        """
        db_root         : path to Spider SQLite databases
        gold_cache_path : path to precomputed gold results JSON
        strict_cache    : if True, raise an error when a key is missing from
                          cache instead of falling back to live execution.
                          Set True during training, False during evaluation.
        """
        self.executor    = DBQueryExecutor(db_root)
        self.gold_cache  = {}
        self.strict_cache = strict_cache
        if gold_cache_path:
            with open(gold_cache_path) as f:
                self.gold_cache = json.load(f)
            print(f"Loaded gold cache: {len(self.gold_cache)} entries "
                  f"from {gold_cache_path}")

    @abstractmethod
    def compute(self, db_id: str, pred_sql: str, gold_sql: str) -> RewardResult:
        pass

    def _gold_result(self, db_id: str, gold_sql: str) -> QueryResult:
        """Return gold result from cache if available.
        Falls back to live execution unless strict_cache=True,
        in which case a missing key raises an error.
        """
        key = f"{db_id}||{gold_sql.strip()}"
        if key in self.gold_cache:
            entry = self.gold_cache[key]
            return QueryResult(
                success=entry["success"],
                rows=set(entry["rows"]) if entry["success"] else None,
                error=entry.get("error"),
            )
        if self.strict_cache:
            raise KeyError(
                f"Gold cache miss — key not found: '{key}'. "
                f"Run preprocess_gold_cache.py to rebuild the cache."
            )
        return self.executor.execute(db_id, gold_sql)

    @abstractmethod
    def compute(self, db_id: str, pred_sql: str, gold_sql: str) -> RewardResult:
        pass

    # ── Shared metric helpers ─────────────────────────────────────────────────

    def _exec_acc(self, db_id: str, pred_sql: str, gold_sql: str) -> float:
        """1.0 if pred and gold return identical result sets, else 0.0.
        Returns 0.0 if the example is unreliable per TrainingDataFilter.
        """
        if not TrainingDataFilter.is_reliable(db_id, gold_sql):
            return 0.0
        pred = self.executor.execute(db_id, pred_sql)
        gold = self._gold_result(db_id, gold_sql)
        if not pred.success or not gold.success:
            return 0.0
        return 1.0 if pred.rows == gold.rows else 0.0

    def _valid_sql(self, db_id: str, pred_sql: str) -> float:
        """1.0 if SQL is syntactically and structurally valid."""
        return 1.0 if self.executor.is_valid(db_id, pred_sql) else 0.0

    def _table_f1(self, pred_sql: str, gold_sql: str) -> float:
        """F1 score on table names referenced in pred vs gold SQL."""
        pred_tables = self._extract_tables(pred_sql)
        gold_tables = self._extract_tables(gold_sql)
        if not pred_tables or not gold_tables:
            return 0.0
        overlap   = pred_tables & gold_tables
        precision = len(overlap) / len(pred_tables)
        recall    = len(overlap) / len(gold_tables)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def _efficiency(self, pred_sql: str, gold_sql: str) -> float:
        """
        Structural complexity score — penalizes extra subqueries and joins.
        Returns 1.0 if pred is not more complex than gold, decreasing otherwise.
        """
        def complexity(sql: str) -> int:
            sql = sql.lower()
            return max(0, sql.count('select') - 1) + len(re.findall(r'\bjoin\b', sql))

        delta = complexity(pred_sql) - complexity(gold_sql)
        return max(0.0, 1.0 - 0.1 * delta) if delta > 0 else 1.0

    @staticmethod
    def _extract_tables(sql: str) -> set:
        sql = sql.lower()
        return set(re.findall(r'(?:from|join)\s+(\w+)(?:\s+as\s+\w+)?', sql))


# ── Binary Reward ─────────────────────────────────────────────────────────────

class BinaryReward(BaseReward):
    """
    Simple binary reward — 1.0 if execution result matches, 0.0 otherwise.
    Start here. Switch to CompositeReward if training doesn't converge.
    """

    def __init__(self, db_root: str, gold_cache_path: Optional[str] = None):
        super().__init__(db_root, gold_cache_path)

    def compute(self, db_id: str, pred_sql: str, gold_sql: str) -> RewardResult:
        exec_acc = self._exec_acc(db_id, pred_sql, gold_sql)
        return RewardResult(
            total          = exec_acc,
            exec_acc       = exec_acc,
            valid_sql      = 0.0,
            correct_tables = 0.0,
            efficiency     = 0.0,
        )


# ── Composite Reward ──────────────────────────────────────────────────────────

class CompositeReward(BaseReward):
    """
    Shaped reward combining multiple signals for stable GRPO training.

    R = 0.50 * exec_acc
      + 0.20 * valid_sql
      + 0.15 * correct_tables
      + 0.15 * efficiency

    Process rewards (correct_tables, efficiency) are 0 if SQL fails to execute.
    """

    WEIGHTS = {
        "exec_acc"       : 0.50,
        "valid_sql"      : 0.20,
        "correct_tables" : 0.15,
        "efficiency"     : 0.15,
    }

    def __init__(self, db_root: str, gold_cache_path: Optional[str] = None):
        super().__init__(db_root, gold_cache_path)

    def compute(self, db_id: str, pred_sql: str, gold_sql: str) -> RewardResult:
        valid    = self._valid_sql(db_id, pred_sql)
        exec_acc = self._exec_acc(db_id, pred_sql, gold_sql)
        tables   = self._table_f1(pred_sql, gold_sql)  if valid else 0.0
        eff      = self._efficiency(pred_sql, gold_sql) if valid else 0.0

        w = self.WEIGHTS
        total = (w["exec_acc"]       * exec_acc +
                 w["valid_sql"]      * valid     +
                 w["correct_tables"] * tables    +
                 w["efficiency"]     * eff)

        return RewardResult(
            total          = round(total, 4),
            exec_acc       = exec_acc,
            valid_sql      = valid,
            correct_tables = round(tables, 4),
            efficiency     = round(eff, 4),
        )