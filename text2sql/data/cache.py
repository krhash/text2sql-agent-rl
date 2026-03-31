"""TrueSQLCacheBuilder — execute all true SQL and cache results to JSON."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from text2sql.db.executor import DBQueryExecutor


class TrueSQLCacheBuilder:
    """
    Executes all true SQL queries in a Spider split and saves results to JSON.

    Cache format:
    {
        "<db_id>||<true_sql>": {
            "rows"   : ["row1", "row2", ...],
            "success": true,
            "error"  : null
        },
        ...
    }

    Key is db_id + true_sql.
    Duplicate (db_id, true_sql) pairs are only executed once.
    """

    def __init__(self, db_root: str = "dataset/database"):
        self.executor = DBQueryExecutor(db_root)

    @staticmethod
    def cache_key(db_id: str, true_sql: str) -> str:
        return f"{db_id}||{true_sql.strip()}"

    def build(self, parquet_path: str, output_path: str) -> dict:
        df      = pd.read_parquet(parquet_path)
        cache   = {}
        skipped = 0
        failed  = 0

        print(f"Building true_sql cache for: {parquet_path}")
        print(f"Total examples: {len(df)}")

        for _, row in tqdm(df.iterrows(), total=len(df)):
            key = self.cache_key(row['db_id'], row['query'])

            if key in cache:
                skipped += 1
                continue

            result = self.executor.execute(row['db_id'], row['query'])

            if result.success:
                cache[key] = {"rows": list(result.rows), "success": True, "error": None}
            else:
                cache[key] = {"rows": [], "success": False, "error": result.error}
                failed += 1

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(cache, f)

        unique = len(cache)
        print(f"\nCache summary:")
        print(f"  Total examples  : {len(df)}")
        print(f"  Unique queries  : {unique}")
        print(f"  Skipped (dupes) : {skipped}")
        print(f"  Failed queries  : {failed}")
        print(f"  Success rate    : {(unique - failed) / unique * 100:.1f}%")
        if failed > 0:
            print(f"\n  Failed entries (will score 0 for execution accuracy):")
            for k, v in cache.items():
                if not v["success"]:
                    db_id, sql = k.split("||", 1)
                    print(f"    DB : {db_id}")
                    print(f"    Err: {v['error']}")
        print(f"\n  Saved → {output_path}")
        return cache
