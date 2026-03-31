"""Data contracts — the shapes data takes as it flows between pipeline stages."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class Example:
    """One Spider training or validation example."""
    db_id      : str
    question   : str
    true_sql   : str
    difficulty : str           # easy | medium | hard | extra hard
    query_toks : list[str]     # raw tokens (used by DifficultyClassifier)


@dataclass
class Prediction:
    """
    Output of any SQLGenerator for one Example.
    Serialised to predictions.csv.

    This is the serialisation boundary between generation and evaluation.
    Evaluators read this — inference never re-runs just to re-evaluate.
    """
    db_id          : str
    question       : str
    true_sql       : str
    pred_sql       : str
    raw_output     : str
    tag_found      : bool
    difficulty     : str
    generator_name : str = ""
    metadata       : dict = field(default_factory=dict)


@dataclass
class EvalRow:
    """One evaluator's output for one Prediction. Appended to eval_*.csv."""
    score   : float            # primary metric 0.0 – 1.0
    details : dict = field(default_factory=dict)


# ── Serialisation helpers ──────────────────────────────────────────────────────

def predictions_to_df(predictions: list[Prediction]) -> pd.DataFrame:
    rows = []
    for p in predictions:
        row = {
            "db_id"         : p.db_id,
            "question"      : p.question,
            "true_sql"      : p.true_sql,
            "pred_sql"      : p.pred_sql,
            "raw_output"    : p.raw_output,
            "tag_found"     : int(p.tag_found),
            "difficulty"    : p.difficulty,
            "generator_name": p.generator_name,
        }
        row.update(p.metadata)
        rows.append(row)
    return pd.DataFrame(rows)


def df_to_predictions(df: pd.DataFrame) -> list[Prediction]:
    core_cols = {"db_id", "question", "true_sql", "pred_sql",
                 "raw_output", "tag_found", "difficulty", "generator_name"}
    predictions = []
    for _, row in df.iterrows():
        extra = {k: v for k, v in row.items() if k not in core_cols}
        predictions.append(Prediction(
            db_id          = str(row.get("db_id",          "")),
            question       = str(row.get("question",       "")),
            true_sql       = str(row.get("true_sql",       "")),
            pred_sql       = str(row.get("pred_sql",       "")),
            raw_output     = str(row.get("raw_output",     "")),
            tag_found      = bool(row.get("tag_found",     False)),
            difficulty     = str(row.get("difficulty",     "")),
            generator_name = str(row.get("generator_name", "")),
            metadata       = extra,
        ))
    return predictions
