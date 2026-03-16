"""
Baseline inference — Llama 3.1 8B on Spider validation set.
Run interactively : python scripts/baseline_inference.py
Run via sbatch    : sbatch jobs/run_baseline.sh

Author: Krushna Sanjay Sharma
"""

import os
import re
import json
import argparse
from datetime import datetime

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_MODEL_ID    = "meta-llama/Meta-Llama-3.1-8B-Instruct"
DEFAULT_CACHE_DIR   = "/scratch/$USER/hf_cache"
DEFAULT_DATA_PATH   = "dataset/validation-00000-of-00001.parquet"
DEFAULT_SCHEMA_PATH = "dataset/spider_schema_rows_v2.json"
DEFAULT_OUTPUT_PATH = "results/baseline_results_{datetime}.csv"
DEFAULT_N_SAMPLES   = 50
DEFAULT_DTYPE       = "bfloat16"

DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16":  torch.float16,
    "float32":  torch.float32,
}

FEW_SHOT_EXAMPLES = """
### Example 1
Schema: stadium : Stadium_ID (number) , Location (text) , Name (text) , Capacity (number)
Question: How many stadiums are there?
SQL: <SQL_START> SELECT count(*) FROM stadium <SQL_END>

### Example 2
Schema: singer : Singer_ID (number) , Name (text) , Country (text) , Age (number)
Question: What are the names of all singers from France?
SQL: <SQL_START> SELECT Name FROM singer WHERE Country = 'France' <SQL_END>

### Example 3
Schema: employee : Employee_ID (number) , Name (text) , Department (text) , Salary (number)
Question: What is the average salary of employees in each department?
SQL: <SQL_START> SELECT Department, avg(Salary) FROM employee GROUP BY Department <SQL_END>
"""


# ── Utility: Difficulty Classifier ───────────────────────────────────────────

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

        if hard:      return 'hard'
        if medium:    return 'medium'
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


# ── Prompt Builder ────────────────────────────────────────────────────────────

class PromptBuilder:
    """Builds schema-aware few-shot prompts for text-to-SQL inference."""

    SYSTEM = (
        "You are a SQL expert. Given a database schema and a question, "
        "generate the correct SQL query.\n\n"
        "You MUST follow these output rules EXACTLY:\n"
        "- You MUST wrap your SQL in tags: <SQL_START> your sql here <SQL_END>\n"
        "- You MUST output ONLY the tags and SQL — NO other text\n"
        "- You MUST NOT add explanations, comments, or markdown\n"
        "- You MUST output a SINGLE SQL statement with NO semicolon\n\n"
        "Example: <SQL_START> SELECT col FROM table WHERE condition <SQL_END>"
    )

    def __init__(self, schema_path: str):
        with open(schema_path) as f:
            schemas = json.load(f)
        self.schema_dict = {s['db_id']: s for s in schemas}

    def build(self, question: str, db_id: str) -> str:
        s = self.schema_dict[db_id]
        return (
            f"{self.SYSTEM}\n\n"
            f"Here are some examples:\n{FEW_SHOT_EXAMPLES}\n"
            "Now answer the following:\n\n"
            f"### Database Schema\n{s['Schema (values (type))']}\n\n"
            f"### Primary Keys\n{s['Primary Keys']}\n\n"
            f"### Foreign Keys\n{s['Foreign Keys']}\n\n"
            f"### Question\n{question}\n\n"
            "### SQL\n"
        )


# ── SQL Utilities ─────────────────────────────────────────────────────────────

class SQLUtils:
    """Helpers for extracting and comparing SQL strings."""

    @staticmethod
    def extract(raw: str) -> tuple:
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
    def exact_match(pred: str, gold: str) -> bool:
        """Normalised exact match — case, quotes, join type, whitespace."""
        def norm(s):
            s = s.lower().strip().rstrip(';')
            s = s.replace('"', "'")
            s = s.replace('inner join', 'join')
            return " ".join(s.split())
        return norm(pred) == norm(gold)


# ── Inference Engine ──────────────────────────────────────────────────────────

class InferenceEngine:
    """Loads model and runs greedy inference."""

    def __init__(self, model_id: str, dtype: str,
                 cache_dir: str = None, model_path: str = None):
        """
        model_id   : HuggingFace model ID — used if model_path is not set
        dtype      : bfloat16 / float16 / float32
        cache_dir  : where to download and cache model weights
        model_path : load directly from a local directory, skipping HF download
        """
        load_from = model_path if model_path else model_id
        cache     = os.path.expandvars(cache_dir) if cache_dir else None

        print(f"\nLoading model from : {load_from}")
        print(f"Cache directory    : {cache or 'HF default'}")
        print(f"dtype              : {dtype}")

        self.tokenizer = AutoTokenizer.from_pretrained(load_from, cache_dir=cache)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            load_from,
            torch_dtype=DTYPE_MAP[dtype],
            device_map="auto",
            cache_dir=cache,
        )
        self.model.eval()
        print(f"Model loaded on    : {next(self.model.parameters()).device}\n")

    def generate(self, prompt: str) -> str:
        inputs    = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=[
                    self.tokenizer.eos_token_id,
                    self.tokenizer.encode("\n\n", add_special_tokens=False)[0],
                ],
            )
        new_tokens = outputs[0][input_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)


# ── Dataset ───────────────────────────────────────────────────────────────────

class SpiderDataset:
    """Loads and samples Spider validation data."""

    def __init__(self, data_path: str):
        df = pd.read_parquet(data_path)
        df['difficulty'] = df['query_toks_no_value'].apply(DifficultyClassifier.classify)
        self.df = df
        print("Difficulty breakdown (full val set):")
        print(self.df['difficulty'].value_counts().to_string())

    def sample(self, n: int, random_state: int = 42) -> pd.DataFrame:
        n_per_level = n // 4
        frames = [
            group.sample(min(len(group), n_per_level), random_state=random_state)
            for _, group in self.df.groupby('difficulty')
        ]
        result = pd.concat(frames).sample(frac=1, random_state=random_state).reset_index(drop=True)
        print(f"\nSampled {len(result)} examples | difficulty breakdown:")
        print(result['difficulty'].value_counts().to_string())
        return result


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id",    default=DEFAULT_MODEL_ID)
    parser.add_argument("--model_path",  default=None,
                        help="Load from local directory instead of HuggingFace")
    parser.add_argument("--cache_dir",   default=DEFAULT_CACHE_DIR,
                        help="Directory to cache downloaded model weights")
    parser.add_argument("--data_path",   default=DEFAULT_DATA_PATH)
    parser.add_argument("--schema_path", default=DEFAULT_SCHEMA_PATH)
    parser.add_argument("--output_path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--n_samples",   type=int, default=DEFAULT_N_SAMPLES)
    parser.add_argument("--dtype",       default=DEFAULT_DTYPE, choices=DTYPE_MAP.keys())
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    run_ts           = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_path = args.output_path.replace("{datetime}", run_ts)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    print("Loading data...")
    dataset        = SpiderDataset(args.data_path)
    sample         = dataset.sample(args.n_samples)
    prompt_builder = PromptBuilder(args.schema_path)
    engine         = InferenceEngine(
                         args.model_id, args.dtype,
                         cache_dir=args.cache_dir,
                         model_path=args.model_path,
                     )

    results = []
    correct = 0
    print("Running inference...\n")

    for _, row in sample.iterrows():
        prompt              = prompt_builder.build(row['question'], row['db_id'])
        raw_output          = engine.generate(prompt)
        pred_sql, tag_found = SQLUtils.extract(raw_output)
        match               = SQLUtils.exact_match(pred_sql, row['query'])
        if match:
            correct += 1

        results.append({
            "db_id"      : row['db_id'],
            "question"   : row['question'],
            "gold_sql"   : row['query'],
            "raw_output" : raw_output,
            "pred_sql"   : pred_sql,
            "tag_found"  : int(tag_found),
            "difficulty" : row['difficulty'],
            "exact_match": int(match),
        })

        pd.DataFrame(results).to_csv(args.output_path, index=False)

        n = len(results)
        print(f"[{n:>3}/{len(sample)}]  "
              f"match={int(match)}  "
              f"tag={'✓' if tag_found else '✗'}  "
              f"acc={correct/n*100:.1f}%  "
              f"difficulty={row['difficulty']}")

    results_df = pd.DataFrame(results)
    print("\n" + "=" * 50)
    print("BASELINE RESULTS SUMMARY")
    print("=" * 50)
    print(f"Overall exact match : {correct}/{len(results)} = {correct/len(results)*100:.1f}%")
    print(f"Tag format followed : {results_df['tag_found'].sum()}/{len(results)}")
    print("\nBy difficulty:")
    print(results_df.groupby('difficulty')['exact_match']
                    .agg(['sum', 'count', 'mean'])
                    .rename(columns={'sum': 'correct', 'count': 'total', 'mean': 'accuracy'})
                    .assign(accuracy=lambda x: (x['accuracy'] * 100).round(1))
                    .to_string())
    print(f"\nResults saved → {args.output_path}")


if __name__ == "__main__":
    main()
    