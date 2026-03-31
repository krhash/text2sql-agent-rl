"""SpiderDataset and DifficultyClassifier."""
from __future__ import annotations

import pandas as pd

from text2sql.data.types import Example


# ── Difficulty Classifier ──────────────────────────────────────────────────────

class DifficultyClassifier:
    """
    Derives Spider query difficulty from query_toks_no_value.
    Replicates the official Spider hardness rubric (Yu et al. 2018).
    Levels: easy | medium | hard | extra hard
    """

    COMP1 = {'WHERE', 'GROUP', 'ORDER', 'LIMIT', 'JOIN', 'OR', 'LIKE', 'HAVING'}
    COMP2 = {'EXCEPT', 'UNION', 'INTERSECT'}
    AGG   = {'COUNT', 'MAX', 'MIN', 'AVG', 'SUM'}

    @classmethod
    def classify(cls, toks: list) -> str:
        t          = [x.upper() for x in toks]
        n_comp1    = sum(1 for kw in cls.COMP1 if kw in t)
        n_comp2    = sum(1 for kw in cls.COMP2 if kw in t)
        has_nested = t.count('SELECT') > 1
        has_comp2  = n_comp2 > 0 or has_nested
        others     = cls._count_others(t)

        if n_comp1 <= 1 and others == 0 and not has_comp2:
            return 'easy'

        hard = (
            (others > 2 and n_comp1 <= 2 and not has_comp2) or
            (2 < n_comp1 <= 3 and others <= 2 and not has_comp2) or
            (n_comp1 <= 1 and others == 0 and n_comp2 == 1 and not has_nested)
        )
        medium = (
            (others <= 2 and n_comp1 <= 1 and not has_comp2) or
            (n_comp1 == 2 and others < 2 and not has_comp2)
        )

        if hard:   return 'hard'
        if medium: return 'medium'
        return 'extra hard'

    @classmethod
    def _count_others(cls, t: list) -> int:
        others = 0
        if sum(1 for tok in t if tok in cls.AGG) > 1:
            others += 1
        from_idx = t.index('FROM') if 'FROM' in t else len(t)
        if t[:from_idx].count(',') + 1 > 1:
            others += 1
        if t.count('AND') + t.count('OR') > 1:
            others += 1
        if 'GROUP' in t:
            gb_idx = t.index('GROUP')
            if t[gb_idx:gb_idx + 10].count(',') > 0:
                others += 1
        return others


# ── Spider Dataset ─────────────────────────────────────────────────────────────

class SpiderDataset:
    """Loads a Spider parquet split, attaches difficulty, supports stratified sampling."""

    def __init__(self, parquet_path: str):
        df = pd.read_parquet(parquet_path)
        df['difficulty'] = df['query_toks_no_value'].apply(DifficultyClassifier.classify)
        self.df       = df
        self.path     = parquet_path

    def load(self, n: int | None = None, random_state: int = 42) -> list[Example]:
        """
        Return all (or a stratified sample of n) examples as Example objects.
        Stratified: n//4 per difficulty level.
        """
        df = self._sample_df(n, random_state) if n else self.df
        return self._to_examples(df)

    def _sample_df(self, n: int, random_state: int) -> pd.DataFrame:
        n_per_level = n // 4
        frames = [
            group.sample(min(len(group), n_per_level), random_state=random_state)
            for _, group in self.df.groupby('difficulty')
        ]
        return pd.concat(frames).sample(frac=1, random_state=random_state).reset_index(drop=True)

    @staticmethod
    def _to_examples(df: pd.DataFrame) -> list[Example]:
        examples = []
        for _, row in df.iterrows():
            examples.append(Example(
                db_id      = str(row['db_id']),
                question   = str(row['question']),
                true_sql   = str(row['query']),
                difficulty = str(row['difficulty']),
                query_toks = list(row.get('query_toks_no_value', [])),
            ))
        return examples
