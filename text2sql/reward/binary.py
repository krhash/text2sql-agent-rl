"""BinaryReward — simple 1.0/0.0 execution match."""
from __future__ import annotations

from typing import Optional

from text2sql.reward.base import BaseReward, RewardResult


class BinaryReward(BaseReward):
    """
    Simple binary reward — 1.0 if execution result matches, 0.0 otherwise.
    Start here. Switch to CompositeReward if training doesn't converge.
    """

    def __init__(self, db_root: str = "dataset/database",
                 true_sql_cache_path: Optional[str] = None,
                 strict_cache: bool = False):
        super().__init__(db_root, true_sql_cache_path, strict_cache)

    def compute(self, db_id: str, pred_sql: str, true_sql: str) -> RewardResult:
        exec_acc = self._exec_acc(db_id, pred_sql, true_sql)
        return RewardResult(
            total          = exec_acc,
            exec_acc       = exec_acc,
            valid_sql      = 0.0,
            correct_tables = 0.0,
            efficiency     = 0.0,
        )
