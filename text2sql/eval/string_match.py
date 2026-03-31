"""StringMatchEvaluator — normalised exact string match."""
from __future__ import annotations

from text2sql.data.types import Prediction, EvalRow
from text2sql.eval.base import SQLEvaluator
from text2sql.inference.sql_utils import SQLUtils


class StringMatchEvaluator(SQLEvaluator):
    """
    Normalised exact match evaluator.
    CPU-only, instant — no database access needed.
    Used in the prompt optimization inner loop and as a fast training monitor.
    """
    name = "string_match"

    def evaluate_row(self, pred: Prediction) -> EvalRow:
        match = SQLUtils.exact_match(pred.pred_sql, pred.true_sql)
        return EvalRow(
            score   = float(match),
            details = {"exact_match": int(match)},
        )
