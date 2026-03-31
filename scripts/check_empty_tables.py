"""
Checks all Spider SQLite databases for empty tables.
Run locally before training to understand data quality.

    python scripts/check_empty_tables.py
"""

import os
import sqlite3
from collections import defaultdict


DEFAULT_DB_ROOT = "dataset/database"


def check_database(db_path: str) -> list:
    """Return list of (table_name, row_count) for all tables in a database."""
    results = []
    try:
        conn = sqlite3.connect(db_path)
        conn.text_factory = lambda b: b.decode('utf-8', errors='replace')
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        for (table,) in tables:
            try:
                count = conn.execute(f"SELECT count(*) FROM `{table}`").fetchone()[0]
                results.append((table, count))
            except sqlite3.Error as e:
                results.append((table, f"ERROR: {e}"))
        conn.close()
    except sqlite3.Error as e:
        results.append(("__db__", f"ERROR: {e}"))
    return results


def main():
    db_root    = DEFAULT_DB_ROOT
    empty_dbs  = defaultdict(list)   # db_id → [empty table names]
    total_dbs  = 0
    total_tabs = 0
    empty_tabs = 0

    for db_id in sorted(os.listdir(db_root)):
        db_path = None
        for ext in (".db", ".sqlite"):
            candidate = os.path.join(db_root, db_id, f"{db_id}{ext}")
            if os.path.exists(candidate):
                db_path = candidate
                break
        if not db_path:
            continue

        total_dbs += 1
        tables = check_database(db_path)

        for table, count in tables:
            total_tabs += 1
            if count == 0:
                empty_tabs += 1
                empty_dbs[db_id].append(table)

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"=== Empty Table Report ===\n")
    print(f"Total databases checked : {total_dbs}")
    print(f"Total tables checked    : {total_tabs}")
    print(f"Empty tables found      : {empty_tabs}")
    print(f"Databases with empties  : {len(empty_dbs)}\n")

    if empty_dbs:
        print("Databases with at least one empty table:")
        for db_id, tables in sorted(empty_dbs.items()):
            print(f"  {db_id}")
            for t in tables:
                print(f"    └── {t}  (0 rows)")
    else:
        print("✅ No empty tables found.")


if __name__ == "__main__":
    main()