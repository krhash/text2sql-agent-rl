"""SQLUtils — SQL extraction and comparison helpers."""
from __future__ import annotations

import re


class SQLUtils:
    """Stateless helpers for extracting and comparing SQL strings."""

    @staticmethod
    def extract(raw: str) -> tuple[str, bool]:
        """
        Extract SQL from between <SQL_START> and <SQL_END> tags.
        Returns (sql: str, tag_found: bool).
        Returns ("", False) if tags are missing.
        """
        m = re.search(r'<SQL_START>(.*?)<SQL_END>', raw, re.DOTALL | re.IGNORECASE)
        if not m:
            return "", False
        sql = m.group(1).strip()
        sql = sql.rstrip(';').strip()
        sql = re.sub(r'--.*',      '', sql).strip()
        sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL).strip()
        return " ".join(sql.split()), True

    @staticmethod
    def exact_match(pred: str, true: str) -> bool:
        """
        Normalised exact match — case, quotes, join type, whitespace.
        Normalisation:
          - lowercase, strip semicolons
          - double quotes → single quotes
          - INNER JOIN → JOIN
          - collapse whitespace
        """
        def norm(s: str) -> str:
            s = s.lower().strip().rstrip(';')
            s = s.replace('"', "'")
            s = s.replace('inner join', 'join')
            return " ".join(s.split())
        return norm(pred) == norm(true)
