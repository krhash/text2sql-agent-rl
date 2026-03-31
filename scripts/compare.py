"""scripts/compare.py — cross-run comparison table."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from text2sql.eval.report import ComparisonTable


def parse_args():
    p = argparse.ArgumentParser(description="Compare multiple pipeline runs.")
    p.add_argument("--runs",        nargs="+", required=True,
                   help="Names of runs to compare (e.g. baseline_v1 prompt_opt_v1 grpo_v1)")
    p.add_argument("--results_dir", default="results")
    p.add_argument("--output",      default=None,
                   help="Save comparison to this path (default: print to stdout)")
    return p.parse_args()


def main():
    args   = parse_args()
    table  = ComparisonTable().generate(args.runs, args.results_dir)
    print(table)
    if args.output:
        Path(args.output).write_text(table, encoding="utf-8")
        print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
