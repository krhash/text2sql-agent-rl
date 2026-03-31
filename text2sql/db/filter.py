"""TrainingDataFilter — data quality decisions for Spider reward computation."""
from __future__ import annotations

import re


class TrainingDataFilter:
    """
    Determines whether a training example is reliable for reward computation.
    Encapsulates all data quality knowledge about the Spider dataset.

    Separated from DBQueryExecutor — the executor has no business knowing
    about training data quality concerns.
    """

    # Databases where ALL tables are empty — execution always returns empty set
    EMPTY_DATABASES = frozenset({"music_2"})

    # Databases where SOME tables are empty — unreliable for those queries
    EMPTY_TABLES = {
        "sakila_1"  : frozenset({"film_text", "language", "staff", "store"}),
        "formula_1" : frozenset({"pitStops", "lapTimes"}),
    }

    @classmethod
    def is_reliable(cls, db_id: str, true_sql: str) -> bool:
        """
        Returns True if execution accuracy is a reliable reward signal.
        Returns False for:
          - Fully empty databases
          - Queries referencing known empty tables
        """
        if db_id in cls.EMPTY_DATABASES:
            return False
        empty_tables = cls.EMPTY_TABLES.get(db_id)
        if empty_tables:
            sql_lower = true_sql.lower()
            if any(re.search(rf'\b{re.escape(t.lower())}\b', sql_lower)
                   for t in empty_tables):
                return False
        return True
