"""SQLEvaluator ABC and EvalResult."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd

from text2sql.data.types import Prediction, EvalRow


class SQLEvaluator(ABC):
    """
    Abstract base for all evaluators.
    Takes predictions, returns metrics DataFrame.
    Knows nothing about how predictions were generated.
    """
    name: str

    @abstractmethod
    def evaluate_row(self, pred: Prediction) -> EvalRow:
        """Evaluate a single prediction and return an EvalRow."""
        ...

    def evaluate(self, predictions: list[Prediction]) -> pd.DataFrame:
        """
        Evaluate all predictions and return a flat DataFrame.
        Columns: all Prediction fields + evaluator-specific details.
        """
        rows = []
        for pred in predictions:
            eval_row = self.evaluate_row(pred)
            row = {
                "db_id"         : pred.db_id,
                "question"      : pred.question,
                "difficulty"    : pred.difficulty,
                "true_sql"      : pred.true_sql,
                "pred_sql"      : pred.pred_sql,
                "generator_name": pred.generator_name,
                "score"         : eval_row.score,
            }
            row.update(eval_row.details)
            rows.append(row)
        return pd.DataFrame(rows)

    def score(self, predictions: list[Prediction]) -> float:
        """
        Single scalar mean score — used by optimizers for fast inner-loop evaluation.
        """
        if not predictions:
            return 0.0
        return self.evaluate(predictions)["score"].mean()
