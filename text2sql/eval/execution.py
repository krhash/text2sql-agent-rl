"""ExecutionEvaluator — execution result set match."""
from __future__ import annotations

from typing import Optional

from text2sql.data.types import Prediction, EvalRow
from text2sql.db.filter import TrainingDataFilter
from text2sql.eval.base import SQLEvaluator


class ExecutionEvaluator(SQLEvaluator):
    """
    Runs pred_sql and true_sql against the SQLite DB, compares result sets.
    Uses the true_sql cache — ground truth is never re-executed live.
    Adds exec_acc, valid_sql, correct_tables, composite_reward, reliable columns.
    """
    name = "execution"

    def __init__(self, db_root: str = "dataset/database",
                 true_sql_cache_path: Optional[str] = None):
        # Lazy import of CompositeReward to avoid circular imports and
        # allow CPU-only use of the rest of the package
        from text2sql.reward.composite import CompositeReward
        self._reward = CompositeReward(
            db_root             = db_root,
            true_sql_cache_path = true_sql_cache_path,
        )

    def evaluate_row(self, pred: Prediction) -> EvalRow:
        result   = self._reward.compute(pred.db_id, pred.pred_sql, pred.true_sql)
        reliable = TrainingDataFilter.is_reliable(pred.db_id, pred.true_sql)
        return EvalRow(
            score   = result.exec_acc,
            details = {
                "exec_acc"        : result.exec_acc,
                "valid_sql"       : result.valid_sql,
                "correct_tables"  : result.correct_tables,
                "efficiency"      : result.efficiency,
                "composite_reward": result.total,
                "reliable"        : reliable,
            },
        )
