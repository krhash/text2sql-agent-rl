"""
Preprocessing step — execute all gold SQL queries and cache results.

Run once before training:
    python scripts/preprocess_gold_cache.py

Produces:
    dataset/gold_cache_train.json
    dataset/gold_cache_validation.json
"""

import json
import argparse
import pandas as pd
from tqdm import tqdm

from db_executor import DBQueryExecutor


# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_DB_ROOT   = "dataset/database"
DEFAULT_TRAIN     = "dataset/train-00000-of-00001.parquet"
DEFAULT_VAL       = "dataset/validation-00000-of-00001.parquet"
DEFAULT_OUT_TRAIN = "dataset/gold_cache_train.json"
DEFAULT_OUT_VAL   = "dataset/gold_cache_validation.json"


# ── Cache Builder ─────────────────────────────────────────────────────────────

class GoldCacheBuilder:
    """
    Executes all gold SQL queries in a Spider split and saves results to JSON.

    Cache format:
    {
        "<db_id>||<gold_sql>": {
            "rows"   : ["row1", "row2", ...],   # list for JSON serialisation
            "success": true,
            "error"  : null
        },
        ...
    }

    Key is db_id + gold_sql — handles cases where same DB has multiple queries.
    Duplicate (db_id, gold_sql) pairs are only executed once.
    """

    def __init__(self, db_root: str = DEFAULT_DB_ROOT):
        self.executor = DBQueryExecutor(db_root)

    @staticmethod
    def cache_key(db_id: str, gold_sql: str) -> str:
        return f"{db_id}||{gold_sql.strip()}"

    def build(self, parquet_path: str, output_path: str) -> dict:
        df    = pd.read_parquet(parquet_path)
        cache = {}
        skipped = 0
        failed  = 0

        print(f"Building gold cache for: {parquet_path}")
        print(f"Total examples: {len(df)}")

        for _, row in tqdm(df.iterrows(), total=len(df)):
            key = self.cache_key(row['db_id'], row['query'])

            # Skip duplicates — same (db_id, gold_sql) already cached
            if key in cache:
                skipped += 1
                continue

            result = self.executor.execute(row['db_id'], row['query'])

            if result.success:
                cache[key] = {
                    "rows"   : list(result.rows),   # set → list for JSON
                    "success": True,
                    "error"  : None,
                }
            else:
                cache[key] = {
                    "rows"   : [],
                    "success": False,
                    "error"  : result.error,
                }
                failed += 1

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


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_root",   default=DEFAULT_DB_ROOT)
    parser.add_argument("--train",     default=DEFAULT_TRAIN)
    parser.add_argument("--val",       default=DEFAULT_VAL)
    parser.add_argument("--out_train", default=DEFAULT_OUT_TRAIN)
    parser.add_argument("--out_val",   default=DEFAULT_OUT_VAL)
    parser.add_argument("--split",     default="both",
                        choices=["train", "val", "both"])
    return parser.parse_args()


def main():
    args    = parse_args()
    builder = GoldCacheBuilder(args.db_root)

    if args.split in ("train", "both"):
        builder.build(args.train, args.out_train)

    if args.split in ("val", "both"):
        builder.build(args.val, args.out_val)


if __name__ == "__main__":
    main()