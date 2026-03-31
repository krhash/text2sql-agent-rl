"""Pipeline I/O — save and load predictions."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from text2sql.data.types import Prediction, predictions_to_df, df_to_predictions


def save_predictions(predictions: list[Prediction], path: Path | str):
    """Write predictions list to CSV."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    predictions_to_df(predictions).to_csv(path, index=False)


def load_predictions(path: Path | str) -> list[Prediction]:
    """Read predictions CSV into a list of Prediction objects."""
    df = pd.read_csv(path)
    return df_to_predictions(df)
