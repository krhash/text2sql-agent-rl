"""CompositeReward — shaped multi-signal reward for stable GRPO training."""
from __future__ import annotations

from typing import Optional

from text2sql.reward.base import BaseReward, RewardResult


class CompositeReward(BaseReward):
    """
    Shaped reward combining multiple signals for stable GRPO training.

    R = 0.50 * exec_acc       (primary: did the SQL get the right answer?)
      + 0.20 * valid_sql       (did it parse and execute at all?)
      + 0.15 * correct_tables  (F1 on table names — structural partial credit)
      + 0.15 * efficiency      (not more complex than necessary)

    Process rewards (correct_tables, efficiency) are 0 if SQL fails to execute.
    """

    WEIGHTS = {
        "exec_acc"      : 0.50,
        "valid_sql"     : 0.20,
        "correct_tables": 0.15,
        "efficiency"    : 0.15,
    }

    def __init__(self, db_root: str = "dataset/database",
                 true_sql_cache_path: Optional[str] = None,
                 strict_cache: bool = False):
        super().__init__(db_root, true_sql_cache_path, strict_cache)

    def compute(self, db_id: str, pred_sql: str, true_sql: str) -> RewardResult:
        valid    = self._valid_sql(db_id, pred_sql)
        exec_acc = self._exec_acc(db_id, pred_sql, true_sql)
        tables   = self._table_f1(pred_sql, true_sql)   if valid else 0.0
        eff      = self._efficiency(pred_sql, true_sql)  if valid else 0.0

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
