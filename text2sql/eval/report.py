"""ReportGenerator and ComparisonTable."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

DIFFICULTY_ORDER = ["easy", "medium", "hard", "extra hard"]
W = 66


def _pct(val: float | None) -> str:
    return f"{val * 100:.1f}%" if val is not None else "  N/A "


def _gap(em: float | None, ea: float | None) -> str:
    if em is None or ea is None:
        return ""
    diff = ea - em
    sign = "+" if diff >= 0 else ""
    return f"{sign}{diff * 100:.1f}%"


class ReportGenerator:
    """Generates a single-run report comparing string match vs exec accuracy."""

    def generate(
        self,
        string_df  : Optional[pd.DataFrame] = None,
        exec_df    : Optional[pd.DataFrame] = None,
        run_name   : str = "",
        model_id   : str = "unknown",
        n_samples  : int = 0,
    ) -> str:
        lines = []
        lines.append("=" * W)
        lines.append("BASELINE PIPELINE REPORT")
        lines.append(f"Run      : {run_name or 'unnamed'}")
        lines.append(f"Time     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Model    : {model_id}")
        lines.append(f"Samples  : {n_samples}")
        lines.append("=" * W)
        lines.append("")

        header = f"{'Metric':<30}  {'String Match':>12}    {'Exec Accuracy':>14}  {'Gap':>8}"
        lines.append(header)
        lines.append("-" * W)

        def em_score(sub: pd.DataFrame) -> float | None:
            if string_df is None: return None
            col = "exact_match" if "exact_match" in sub.columns else "score"
            return sub[col].mean() if col in sub.columns and len(sub) else None

        def ea_score(sub: pd.DataFrame) -> float | None:
            if exec_df is None: return None
            col = "exec_acc" if "exec_acc" in sub.columns else "score"
            return sub[col].mean() if col in sub.columns and len(sub) else None

        def line(label, s_df, e_df):
            em = em_score(s_df) if s_df is not None else None
            ea = ea_score(e_df) if e_df is not None else None
            return f"{label:<30}  {_pct(em):>12}    {_pct(ea):>14}  {_gap(em, ea):>8}"

        lines.append(line("Overall", string_df, exec_df))

        for diff in DIFFICULTY_ORDER:
            s_sub = string_df[string_df["difficulty"] == diff] if string_df is not None else None
            e_sub = exec_df[exec_df["difficulty"]   == diff] if exec_df   is not None else None
            if (s_sub is not None and len(s_sub) == 0) and (e_sub is not None and len(e_sub) == 0):
                continue
            lines.append(line(f"  {diff.capitalize()}", s_sub, e_sub))

        lines.append("")

        if exec_df is not None:
            for col, label in [
                ("valid_sql",        "Valid SQL rate"),
                ("correct_tables",   "Correct tables (F1)"),
                ("composite_reward", "Composite reward"),
            ]:
                if col in exec_df.columns:
                    lines.append(f"{'  ' + label:<30}  {exec_df[col].mean():>12.4f}")

            if "reliable" in exec_df.columns:
                n_rel   = exec_df["reliable"].sum()
                n_total = len(exec_df)
                lines.append(f"{'  Reliable examples':<30}  {int(n_rel):>5} / {n_total}")
                lines.append(f"{'  Skipped (unreliable DB)':<30}  {n_total - int(n_rel):>5}")

        lines.append("=" * W)
        return "\n".join(lines)


class ComparisonTable:
    """Side-by-side comparison of multiple runs."""

    def generate(self, run_names: list[str], results_dir: str = "results") -> str:
        run_data = {}
        for run in run_names:
            s_path = Path(results_dir) / run / "eval_string.csv"
            e_path = Path(results_dir) / run / "eval_exec.csv"
            run_data[run] = {
                "string": pd.read_csv(s_path) if s_path.exists() else None,
                "exec"  : pd.read_csv(e_path) if e_path.exists() else None,
            }

        col_w  = 14
        header = f"{'Metric':<30}" + "".join(f"  {r[:col_w]:>{col_w}}" for r in run_names)
        sep    = "=" * (30 + len(run_names) * (col_w + 2))

        lines = [sep, "COMPARISON — Spider Validation Set", sep, "", header, "-" * len(header)]

        def row(label, metric, dfs):
            vals = []
            for df in dfs:
                if df is None:
                    vals.append("  N/A")
                    continue
                col = metric if metric in df.columns else "score"
                vals.append(_pct(df[col].mean()) if col in df.columns else "  N/A")
            return f"{label:<30}" + "".join(f"  {v:>{col_w}}" for v in vals)

        string_dfs = [run_data[r]["string"] for r in run_names]
        exec_dfs   = [run_data[r]["exec"]   for r in run_names]

        lines.append(row("String Match (overall)", "exact_match", string_dfs))
        for diff in DIFFICULTY_ORDER:
            sub_s = [df[df["difficulty"] == diff] if df is not None else None for df in string_dfs]
            lines.append(row(f"  {diff.capitalize()}", "exact_match", sub_s))

        lines.append("")
        lines.append(row("Exec Accuracy (overall)", "exec_acc", exec_dfs))
        for diff in DIFFICULTY_ORDER:
            sub_e = [df[df["difficulty"] == diff] if df is not None else None for df in exec_dfs]
            lines.append(row(f"  {diff.capitalize()}", "exec_acc", sub_e))

        lines.append("")
        lines.append(row("Composite reward", "composite_reward", exec_dfs))
        lines.append(row("Valid SQL rate",   "valid_sql",        exec_dfs))
        lines.append(sep)
        return "\n".join(lines)
