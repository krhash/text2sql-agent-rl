# TEXT2SQL-AGENT-RL — Complete Architecture Plan

> **Goal:** Compare three approaches to improving LLM SQL generation accuracy on the Spider dataset:
> 1. **Baseline** — Llama 3.1 8B with schema-aware few-shot prompting
> 2. **Prompt Optimization** — Actor-critique agent loop, frozen weights
> 3. **GRPO + LoRA** — Reinforcement learning fine-tuning with execution reward

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [Data Contracts](#2-data-contracts)
3. [Core Abstractions](#3-core-abstractions)
4. [Component Reference](#4-component-reference)
5. [Orchestrating Class — ExperimentRunner](#5-orchestrating-class--experimentrunner)
6. [Flow of Events](#6-flow-of-events)
7. [GRPO + LoRA Training](#7-grpo--lora-training)
8. [Prompt Optimization](#8-prompt-optimization)
9. [CLI — How to Execute Each Stage](#9-cli--how-to-execute-each-stage)
10. [Results Layout](#10-results-layout)
11. [Reusable Function Reference](#11-reusable-function-reference)

---

## 1. Project Structure

```
text2sql-agent-rl/
│
├── text2sql/                         ← installable Python package (pip install -e .)
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── types.py                  ← Example, Prediction, EvalRow  (data contracts)
│   │   ├── dataset.py                ← SpiderDataset, DifficultyClassifier
│   │   ├── schema.py                 ← PromptBuilder, SchemaLoader
│   │   └── cache.py                  ← TrueSQLCacheBuilder
│   │
│   ├── db/
│   │   ├── __init__.py
│   │   ├── executor.py               ← DBQueryExecutor, QueryResult
│   │   └── filter.py                 ← TrainingDataFilter
│   │
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── base.py                   ← SQLGenerator (Protocol)
│   │   ├── engine.py                 ← InferenceEngine
│   │   ├── generator.py              ← LLMGenerator, LoRAGenerator
│   │   └── sql_utils.py              ← SQLUtils
│   │
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── base.py                   ← SQLEvaluator (ABC), EvalResult
│   │   ├── string_match.py           ← StringMatchEvaluator
│   │   ├── execution.py              ← ExecutionEvaluator
│   │   └── report.py                 ← ReportGenerator, ComparisonTable
│   │
│   ├── reward/
│   │   ├── __init__.py
│   │   ├── base.py                   ← BaseReward, RewardResult
│   │   ├── binary.py                 ← BinaryReward
│   │   └── composite.py              ← CompositeReward
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── base.py                   ← SQLOptimizer (ABC)
│   │   ├── identity.py               ← IdentityOptimizer  (baseline, no-op)
│   │   ├── prompt_opt.py             ← PromptOptimizer    (actor-critique loop)
│   │   ├── grpo.py                   ← GRPOOptimizer      (RL fine-tuning)
│   │   ├── rollout.py                ← sample_rollout(), group_advantages()
│   │   └── lora.py                   ← default_lora_config(), merge_adapter()
│   │
│   └── pipeline/
│       ├── __init__.py
│       ├── config.py                 ← RunConfig  (all run parameters)
│       ├── runner.py                 ← ExperimentRunner  (orchestrating class)
│       └── io.py                     ← save/load predictions and results
│
├── scripts/                          ← thin CLI entry points  (no business logic)
│   ├── pipeline.py                   ← main entry point, delegates to ExperimentRunner
│   └── compare.py                    ← cross-run comparison table
│
├── jobs/                             ← HPC sbatch scripts
│   ├── preprocess.sh
│   ├── inference.sh
│   └── train_grpo.sh
│
├── notebooks/
│   └── 01_explore_spider.ipynb
│
├── dataset/                          ← Spider data  (not in git)
│   ├── train-00000-of-00001.parquet
│   ├── validation-00000-of-00001.parquet
│   ├── spider_schema_rows_v2.json
│   └── database/                     ← 160 SQLite databases
│
├── results/                          ← all stage outputs, indexed by run name
├── models/                           ← LoRA checkpoints
│   └── lora/
├── setup.py
└── README.md
```

---

## 2. Data Contracts

All data that flows between stages is typed. These are the serialization boundaries.

### `text2sql/data/types.py`

```python
@dataclass
class Example:
    """One Spider row — input to any SQLGenerator."""
    db_id      : str
    question   : str
    true_sql   : str
    difficulty : str           # easy | medium | hard | extra hard
    query_toks : list[str]     # raw tokens (used by DifficultyClassifier)


@dataclass
class Prediction:
    """
    Output of any SQLGenerator for one Example.
    Written to predictions.csv.
    All evaluators read this — inference never re-runs just to re-evaluate.
    """
    db_id          : str
    question       : str
    true_sql       : str
    pred_sql       : str
    raw_output     : str
    tag_found      : bool
    difficulty     : str
    generator_name : str       # "baseline_v1" | "prompt_opt_v1" | "grpo_v1"
    metadata       : dict      # {"prompt_version": 2, "lora_step": 500, ...}


@dataclass
class EvalRow:
    """One evaluator's output for one Prediction. Appended to eval_*.csv."""
    score   : float            # primary metric 0.0 – 1.0
    details : dict             # evaluator-specific columns
```

**Key rule:** `Prediction` is the boundary between generation and evaluation. You never re-run inference to re-evaluate — you save `predictions.csv` once and re-read it with any evaluator.

---

## 3. Core Abstractions

Four interfaces that everything plugs into.

### 3.1 `SQLGenerator` (Protocol)

> Anything that takes an `Example` and produces a `Prediction`.
> Baseline, prompt-optimised, and LoRA-adapted generators all implement this.

```python
class SQLGenerator(Protocol):
    name: str

    def generate(self, example: Example) -> Prediction: ...

    def generate_batch(
        self,
        examples    : list[Example],
        progress    : bool = True,
        output_path : Path | None = None,   # incremental save — crash safe
    ) -> list[Prediction]: ...
```

**Implementations:**
- `LLMGenerator` — frozen weights, configurable `PromptBuilder`
- `LoRAGenerator(LLMGenerator)` — same interface, loads LoRA adapter on init

### 3.2 `SQLOptimizer` (ABC)

> Takes a starting generator + data. Returns a better generator.
> This is what distinguishes the three approaches.

```python
class SQLOptimizer(ABC):
    @abstractmethod
    def optimize(
        self,
        generator  : SQLGenerator,
        train_data : list[Example],
        val_data   : list[Example],
        output_dir : Path,
    ) -> SQLGenerator: ...
```

**Implementations:**

| Class | Changes | Returns |
|---|---|---|
| `IdentityOptimizer` | nothing | same `LLMGenerator` |
| `PromptOptimizer` | prompt template | `LLMGenerator` with better `PromptBuilder` |
| `GRPOOptimizer` | LoRA weights | `LoRAGenerator` pointing at checkpoint |

### 3.3 `SQLEvaluator` (ABC)

> Takes a list of predictions. Returns a metrics DataFrame.
> Knows nothing about how predictions were generated.

```python
class SQLEvaluator(ABC):
    name: str

    @abstractmethod
    def evaluate_row(self, pred: Prediction) -> EvalRow: ...

    def evaluate(self, predictions: list[Prediction]) -> pd.DataFrame:
        """Map evaluate_row over all predictions → flat DataFrame."""
        ...

    def score(self, predictions: list[Prediction]) -> float:
        """Single scalar — used by optimizers for fast inner-loop evaluation."""
        return self.evaluate(predictions)["score"].mean()
```

**Implementations:**

| Class | Metric | Speed | DB needed |
|---|---|---|---|
| `StringMatchEvaluator` | normalised exact string match | instant | no |
| `ExecutionEvaluator` | execution result set match | ~1ms/row with cache | yes |

### 3.4 `BaseReward` (ABC) — separate from Evaluator

> Used **live during GRPO training** (one scalar per completion).
> Not for batch post-hoc analysis — that's `ExecutionEvaluator`.

```python
class BaseReward(ABC):
    def compute(self, db_id: str, pred_sql: str, true_sql: str) -> RewardResult: ...
```

**Implementations:** `BinaryReward` (0/1 exec match), `CompositeReward` (weighted multi-signal).

Both share `DBQueryExecutor` with `ExecutionEvaluator` but are intentionally separate — reward functions must be fast scalars; evaluators must be rich DataFrames.

---

## 4. Component Reference

### `text2sql/data/dataset.py`

```python
class DifficultyClassifier:
    """
    Official Spider hardness rubric (Yu et al. 2018).
    Counts component keywords (WHERE, JOIN, HAVING, etc.)
    to assign easy / medium / hard / extra hard.
    """
    @classmethod
    def classify(cls, query_toks: list[str]) -> str: ...

class SpiderDataset:
    """Loads a parquet split, attaches difficulty, supports stratified sampling."""
    def __init__(self, parquet_path: str): ...
    def load(self, n: int | None = None, random_state: int = 42) -> list[Example]:
        """Stratified sample — n//4 per difficulty level."""
        ...
```

### `text2sql/data/schema.py`

```python
class PromptBuilder:
    """
    Builds few-shot schema-aware prompts.
    Serializable to/from JSON so prompt optimization can save best prompt.
    """
    def __init__(self, schema_path: str, system_prompt: str = DEFAULT_SYSTEM,
                 few_shot_examples: str = DEFAULT_EXAMPLES): ...
    def build(self, question: str, db_id: str) -> str: ...
    def save(self, path: Path): ...
    @classmethod
    def from_file(cls, path: Path) -> "PromptBuilder": ...
    def with_system(self, new_system: str) -> "PromptBuilder":
        """Return new builder with different system prompt. Immutable."""
        ...
```

### `text2sql/data/cache.py`

```python
class TrueSQLCacheBuilder:
    """
    Executes all true SQL queries in a Spider split, caches results to JSON.
    Key: "{db_id}||{true_sql}" → {"rows": [...], "success": bool, "error": str|null}
    Deduplicates — same (db_id, true_sql) only executed once.
    """
    def __init__(self, db_root: str): ...
    def build(self, parquet_path: str, output_path: str) -> dict: ...
    @staticmethod
    def cache_key(db_id: str, true_sql: str) -> str: ...
```

### `text2sql/db/executor.py`

```python
@dataclass
class QueryResult:
    success : bool
    rows    : set | None     # order-insensitive result set
    error   : str | None

class DBQueryExecutor:
    """Execute SQL against Spider SQLite databases. Single responsibility."""
    def __init__(self, db_root: str): ...
    def db_path(self, db_id: str) -> str: ...
    def execute(self, db_id: str, sql: str) -> QueryResult: ...
    def is_valid(self, db_id: str, sql: str) -> bool: ...
```

### `text2sql/db/filter.py`

```python
class TrainingDataFilter:
    """
    Flags examples where execution accuracy is an unreliable reward signal.
    Encapsulates all Spider data-quality knowledge.
    DBQueryExecutor is intentionally unaware of this.
    """
    EMPTY_DATABASES = frozenset({"music_2"})
    EMPTY_TABLES    = {"sakila_1": ..., "formula_1": ...}

    @classmethod
    def is_reliable(cls, db_id: str, true_sql: str) -> bool:
        """Returns False for fully empty DBs or queries on known empty tables."""
        ...
```

### `text2sql/inference/engine.py`

```python
class InferenceEngine:
    """Loads HuggingFace model and runs greedy generation."""
    def __init__(self, model_id: str, dtype: str,
                 cache_dir: str = None, model_path: str = None): ...
    def generate(self, prompt: str, max_new_tokens: int = 256) -> str: ...
```

### `text2sql/inference/generator.py`

```python
class LLMGenerator:
    """Frozen weights + configurable prompt. The baseline generator."""
    def __init__(self, name: str, engine: InferenceEngine,
                 prompt_builder: PromptBuilder): ...
    def with_prompt(self, new_prompt: PromptBuilder) -> "LLMGenerator":
        """Returns new generator with updated prompt. Used by PromptOptimizer."""
        ...
    def generate(self, example: Example) -> Prediction: ...
    def generate_batch(self, examples: list[Example], ...) -> list[Prediction]: ...

class LoRAGenerator(LLMGenerator):
    """Same interface as LLMGenerator. Loads LoRA adapter on init."""
    def __init__(self, name: str, base_engine: InferenceEngine,
                 lora_checkpoint: Path, prompt_builder: PromptBuilder): ...
```

### `text2sql/inference/sql_utils.py`

```python
class SQLUtils:
    @staticmethod
    def extract(raw: str) -> tuple[str, bool]:
        """Extract SQL from <SQL_START>...<SQL_END> tags. Returns (sql, tag_found)."""
        ...

    @staticmethod
    def exact_match(pred: str, true: str) -> bool:
        """
        Normalised exact match:
          - lowercase, strip semicolons
          - double quotes → single quotes
          - INNER JOIN → JOIN
          - collapse whitespace
        """
        ...
```

### `text2sql/eval/string_match.py`

```python
class StringMatchEvaluator(SQLEvaluator):
    name = "string_match"

    def evaluate_row(self, pred: Prediction) -> EvalRow:
        match = SQLUtils.exact_match(pred.pred_sql, pred.true_sql)
        return EvalRow(score=float(match), details={"exact_match": int(match)})
```

### `text2sql/eval/execution.py`

```python
class ExecutionEvaluator(SQLEvaluator):
    name = "execution"

    def __init__(self, db_root: str, true_sql_cache_path: str | None = None):
        self.reward = CompositeReward(db_root, true_sql_cache_path)

    def evaluate_row(self, pred: Prediction) -> EvalRow:
        result   = self.reward.compute(pred.db_id, pred.pred_sql, pred.true_sql)
        reliable = TrainingDataFilter.is_reliable(pred.db_id, pred.true_sql)
        return EvalRow(
            score   = result.exec_acc,
            details = {
                "exec_acc"        : result.exec_acc,
                "valid_sql"       : result.valid_sql,
                "correct_tables"  : result.correct_tables,
                "composite_reward": result.total,
                "reliable"        : reliable,
            }
        )
```

### `text2sql/eval/report.py`

```python
class ReportGenerator:
    """Single-run report: string match vs exec accuracy by difficulty."""
    def generate(self, string_df: pd.DataFrame | None,
                 exec_df: pd.DataFrame | None,
                 run_name: str, model_id: str) -> str: ...

class ComparisonTable:
    """Side-by-side comparison across multiple runs."""
    def generate(self, runs: list[str], results_dir: str) -> str: ...
```

### `text2sql/reward/composite.py`

```python
@dataclass
class RewardResult:
    total          : float
    exec_acc       : float
    valid_sql      : float
    correct_tables : float
    efficiency     : float

class CompositeReward(BaseReward):
    """
    R = 0.50 * exec_acc
      + 0.20 * valid_sql
      + 0.15 * correct_tables  (table F1)
      + 0.15 * efficiency      (structural complexity penalty)

    Uses true_sql cache — only predicted SQL is executed live.
    Skips unreliable examples via TrainingDataFilter.
    """
    WEIGHTS = {"exec_acc": 0.50, "valid_sql": 0.20,
               "correct_tables": 0.15, "efficiency": 0.15}

    def __init__(self, db_root: str, true_sql_cache_path: str | None = None): ...
    def compute(self, db_id: str, pred_sql: str, true_sql: str) -> RewardResult: ...
```

### `text2sql/training/rollout.py`

```python
def sample_rollout(
    engine        : InferenceEngine,
    prompt_builder: PromptBuilder,
    examples      : list[Example],
    group_size    : int,              # G completions per question
) -> list[list[str]]:
    """
    For each example, generate G SQL completions with temperature > 0.
    Returns shape [len(examples), G] list of raw model outputs.
    """
    ...

def group_advantages(rewards: list[list[float]]) -> list[list[float]]:
    """
    GRPO normalisation: for each group,
        advantage_i = (reward_i - mean(group)) / (std(group) + eps)
    Returns same shape as input.
    """
    ...
```

### `text2sql/training/lora.py`

```python
def default_lora_config(r: int = 16, lora_alpha: int = 32) -> LoraConfig:
    return LoraConfig(
        r              = r,
        lora_alpha     = lora_alpha,
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout   = 0.05,
        bias           = "none",
        task_type      = "CAUSAL_LM",
    )

def merge_adapter(base_model, lora_checkpoint: Path):
    """Load and merge LoRA weights into base model for inference."""
    ...
```

---

## 5. Orchestrating Class — `ExperimentRunner`

`text2sql/pipeline/runner.py`

The single class that knows about all stages, all output files, and how they wire together.

```python
class ExperimentRunner:
    """
    Orchestrates one named experiment run.

    Owns:
      - The run directory (results/<run_name>/)
      - pipeline.log
      - Skip logic (don't re-run if output exists)
      - Stage wiring (output of stage N is input to stage N+1)

    One runner per run. To compare approaches, compare runner outputs.
    """

    STAGE_ORDER = [
        "preprocess",
        "infer",
        "eval_string",
        "eval_exec",
        "optimize_prompt",
        "train_grpo",
        "report",
    ]

    def __init__(self, config: RunConfig):
        self.config  = config
        self.run_dir = Path(config.results_dir) / config.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log     = setup_logging(self.run_dir / "pipeline.log")

    def run(self, stages: list[str] | None = None, force: bool = False):
        """Run listed stages. Skip any whose output already exists (unless force)."""
        for stage in (stages or self.STAGE_ORDER):
            self._run_stage(stage, force)

    def _stage_outputs(self) -> dict[str, Path]:
        """Primary output file for each stage — existence = skip."""
        d = self.run_dir
        return {
            "preprocess"     : d / "true_sql_cache_validation.json",
            "infer"          : d / "predictions.csv",
            "eval_string"    : d / "eval_string.csv",
            "eval_exec"      : d / "eval_exec.csv",
            "optimize_prompt": d / "best_prompt.json",
            "train_grpo"     : d / "training_done.flag",
            "report"         : d / "report.txt",
        }

    def _run_stage(self, stage: str, force: bool):
        output = self._stage_outputs()[stage]
        if output.exists() and not force:
            self.log.info(f"[SKIP] {stage} — output exists: {output}")
            return
        self.log.info(f"\n{'='*60}\n  STAGE: {stage.upper()}\n{'='*60}")
        getattr(self, stage.replace("_", "_stage_", 1).replace("stage_", "", 1))()
        # simpler: self._handlers[stage]()
        self.log.info(f"[DONE] {stage}")
```

### Stage Wiring — What Each Stage Reads and Writes

| Stage | Reads | Writes |
|---|---|---|
| `preprocess` | `train.parquet`, `val.parquet` | `true_sql_cache_train.json`, `true_sql_cache_validation.json` |
| `optimize_prompt` | `train.parquet`, `val.parquet` | `best_prompt.json`, `opt_history.jsonl` |
| `train_grpo` | `train.parquet`, `true_sql_cache_train.json`, `best_prompt.json`? | `models/lora/<run>/`, `training_log.jsonl`, `training_done.flag` |
| `infer` | `val.parquet`, `best_prompt.json`?, `training_done.flag`? | `predictions.csv` |
| `eval_string` | `predictions.csv` | `eval_string.csv` |
| `eval_exec` | `predictions.csv`, `true_sql_cache_validation.json` | `eval_exec.csv` |
| `report` | `eval_string.csv`, `eval_exec.csv` | `report.txt` |

`infer` automatically picks up `best_prompt.json` if it exists (from `optimize_prompt`), and picks up the LoRA checkpoint path from `training_done.flag` if it exists (from `train_grpo`).

### `RunConfig` — `text2sql/pipeline/config.py`

```python
@dataclass
class RunConfig:
    # Identity
    run_name    : str
    results_dir : str = "results"

    # Data
    train_path  : str = "dataset/train-00000-of-00001.parquet"
    val_path    : str = "dataset/validation-00000-of-00001.parquet"
    schema_path : str = "dataset/spider_schema_rows_v2.json"
    db_root     : str = "dataset/database"

    # Cache sharing
    preprocess_split : str        = "both"       # train | val | both
    cache_run        : str | None = None         # borrow cache from another run's dir

    # Inference
    model_id       : str        = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    model_path     : str | None = None
    cache_dir      : str        = "/scratch/$USER/hf_cache"
    n_samples      : int        = 1034
    dtype          : str        = "bfloat16"
    inference_from : str | None = None           # copy predictions.csv from named run

    # Prompt optimization
    n_opt_iterations : int = 5
    opt_sample_size  : int = 100

    # GRPO training
    reward_fn  : str   = "composite"             # binary | composite
    group_size : int   = 4                       # G completions per question
    n_steps    : int   = 1000
    kl_coef    : float = 0.1
    lora_r     : int   = 16
    lora_alpha : int   = 32

    @classmethod
    def from_yaml(cls, path: str) -> "RunConfig": ...
    def to_yaml(self, path: str): ...
```

---

## 6. Flow of Events

### High-Level

```
ONCE:
  preprocess ──► true_sql_cache_train.json + true_sql_cache_validation.json

BASELINE:
  infer ──► predictions.csv
  eval_string + eval_exec ──► eval_string.csv + eval_exec.csv
  report ──► report.txt

PROMPT OPT:
  optimize_prompt ──► best_prompt.json
  infer (with best_prompt) ──► predictions.csv
  eval_string + eval_exec + report

GRPO:
  train_grpo (with best_prompt) ──► LoRA checkpoint
  infer (with LoRA model) ──► predictions.csv
  eval_string + eval_exec + report

COMPARE:
  compare --runs baseline_v1 prompt_opt_v1 grpo_v1 ──► comparison.txt
```

### Detailed Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  PREPROCESS  [CPU ~10s]                                                     │
│                                                                             │
│  train.parquet ──► TrueSQLCacheBuilder ──► true_sql_cache_train.json       │
│  val.parquet   ──► TrueSQLCacheBuilder ──► true_sql_cache_validation.json  │
│                                                                             │
│  7000 train queries → 3981 unique (3019 deduplicated), 2 failed            │
│  1034 val queries   →  564 unique ( 470 deduplicated), 0 failed            │
└─────────────────────┬───────────────────────────────────────────────────────┘
                      │ cache reused by all subsequent runs
         ┌────────────┴────────────┐
         │                         │
         ▼                         ▼
┌──────────────────┐    ┌──────────────────────────────────────────────────────┐
│  OPTIMIZE_PROMPT │    │  TRAIN_GRPO  [GPU, hours]                           │
│  [GPU, ~30 min]  │    │                                                      │
│                  │    │  for step in 1..N:                                   │
│  Actor-critique  │    │    sample K training examples                       │
│  loop over N     │    │    generate G completions each  (GPU)               │
│  iterations:     │    │    compute CompositeReward  (CPU, uses train cache)  │
│  1. gen SQL      │    │    group_advantages()                                │
│  2. critique SQL │    │    update LoRA weights  (GPU)                       │
│  3. refine prompt│    │    monitor on val_sample every 50 steps             │
│                  │    │                                                      │
│  → best_prompt   │    │  → models/lora/<run>/checkpoint-final/              │
│    .json         │    │  → training_log.jsonl                               │
└─────────┬────────┘    └──────────────────────────┬───────────────────────────┘
          │                                          │
          └──────────────┬───────────────────────────┘
                         │ both feed into infer
                         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  INFER  [GPU, ~2hrs for 1034 examples]                                      │
│                                                                             │
│  SpiderDataset(val.parquet)                                                 │
│    → stratified sample N examples (N/4 per difficulty)                      │
│    → PromptBuilder.build(question, db_id)   ← loads best_prompt.json       │
│    → InferenceEngine.generate(prompt)        ← or LoRAGenerator             │
│    → SQLUtils.extract() → pred_sql                                          │
│    → save Prediction per row (incremental — crash safe)                     │
│                                                                             │
│  Output columns: db_id, question, true_sql, pred_sql, raw_output,          │
│                  tag_found, difficulty, generator_name, metadata            │
│  → predictions.csv                                                          │
└────────────────────────────┬────────────────────────────────────────────────┘
                             │
                   ┌─────────┴──────────┐
                   │                    │
                   ▼                    ▼
┌──────────────────────────┐  ┌──────────────────────────────────────────────┐
│  EVAL_STRING  [CPU <1min]│  │  EVAL_EXEC  [CPU ~5min with cache]           │
│                          │  │                                              │
│  StringMatchEvaluator    │  │  ExecutionEvaluator                          │
│  .evaluate(predictions)  │  │  .evaluate(predictions)                      │
│                          │  │                                              │
│  Adds columns:           │  │  Adds columns:                               │
│  - exact_match (0/1)     │  │  - exec_acc (0/1)                           │
│  - score (same)          │  │  - valid_sql (0/1)                           │
│                          │  │  - correct_tables (F1)                       │
│  → eval_string.csv       │  │  - composite_reward (0–1)                    │
│                          │  │  - reliable (bool)                           │
└──────────────────────────┘  │  → eval_exec.csv                            │
                   │           └──────────────────────────────────────────────┘
                   └─────────┬──────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  REPORT  [CPU instant]                                                      │
│                                                                             │
│  ReportGenerator.generate(eval_string.csv, eval_exec.csv)                 │
│  → report.txt  (printed to stdout + saved)                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                             │ repeat across all three runs
                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  COMPARE  [CPU instant]                                                     │
│                                                                             │
│  ComparisonTable([baseline_v1, prompt_opt_v1, grpo_v1])                   │
│  → comparison.txt                                                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. GRPO + LoRA Training

### How It Works

GRPO (Group Relative Policy Optimization) is the RL algorithm used in DeepSeek-R1. It does not need a value network — it estimates advantage from within-group reward variance.

```
For each training step:

  1. SAMPLE K questions from train_data

  2. ROLLOUT — for each question, generate G completions (temperature > 0):
       completions[k][g] = model.generate(prompt, n=G)
       pred_sqls[k][g]   = SQLUtils.extract(completions[k][g])

  3. REWARD — for each completion:
       rewards[k][g] = CompositeReward.compute(
           db_id    = example.db_id,
           pred_sql = pred_sqls[k][g],
           true_sql = example.true_sql,          ← never re-executed (cache)
       ).total

  4. NORMALIZE — group-relative advantage:
       mean_r = mean(rewards[k])
       std_r  = std(rewards[k]) + 1e-8
       adv[k][g] = (rewards[k][g] - mean_r) / std_r

  5. UPDATE — policy gradient on LoRA weights only:
       loss = -sum(adv[k][g] * log_prob(completions[k][g]))
            + kl_coef * KL(current_policy || reference_policy)
       loss.backward()
       optimizer.step()    # only LoRA delta weights move

  6. MONITOR every 50 steps:
       val_sample = sample(val_data, 50)
       score      = StringMatchEvaluator.score(lora_gen.generate_batch(val_sample))
       log(step, loss, mean_reward, val_score)
```

### Why the Cache Is Critical

Without the cache, each step would require executing `K × G` true SQL queries against SQLite. With K=8 questions, G=4 completions, and 1000 steps, that's 32,000 true SQL executions per epoch. With the cache, true SQL results are looked up in ~0ms — only predicted SQL is ever executed live.

### LoRA Configuration

```python
# text2sql/training/lora.py — default config
LoraConfig(
    r              = 16,        # rank — higher = more capacity, more VRAM
    lora_alpha     = 32,        # scaling = lora_alpha / r = 2.0
    target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout   = 0.05,
    bias           = "none",
    task_type      = "CAUSAL_LM",
)
# Trainable params: ~20M out of 8B (0.25%)
# Additional VRAM: ~0.5GB — fits alongside frozen base in bfloat16 on A100 40GB
```

### What Gets Saved

```
models/lora/grpo_v1/
├── checkpoint-500/
│   ├── adapter_config.json        ← LoRA hyperparameters
│   └── adapter_model.safetensors  ← only the ~20M delta weights (not full model)
├── checkpoint-final/
└── training_log.jsonl             ← {step, loss, mean_reward, val_exact_match}
```

To evaluate a checkpoint:
```bash
python scripts/pipeline.py --run grpo_v1_eval \
    --model_path models/lora/grpo_v1/checkpoint-final \
    --stages infer eval_string eval_exec report
```

### Reward Function Choice

| Function | Formula | Start with if... |
|---|---|---|
| `BinaryReward` | 1.0 if exec match, else 0.0 | model already generates valid SQL (exec acc > 40%) |
| `CompositeReward` | 0.5·exec + 0.2·valid + 0.15·tables + 0.15·eff | model struggles early, need gradient signal even on failures |

Start with `CompositeReward`. Switch to `BinaryReward` if training collapses or the composite signal introduces noise.

---

## 8. Prompt Optimization

### Actor-Critique Loop

```
PromptOptimizer.optimize():

  current_prompt = initial_prompt
  best_score     = StringMatchEvaluator.score(baseline_gen, val_sample)
  best_prompt    = current_prompt

  for iteration in range(N):

    ┌─────────────────────────────────────────────────────┐
    │  ACTOR STEP                                         │
    │  sample 100 examples from train_data               │
    │  predictions = LLMGenerator(engine, current_prompt) │
    │               .generate_batch(sample)               │
    │  score = StringMatchEvaluator.score(predictions)   │
    └───────────────────────┬─────────────────────────────┘
                            │ failed predictions only
                            ▼
    ┌─────────────────────────────────────────────────────┐
    │  CRITIQUE STEP                                      │
    │  for each failure:                                  │
    │    critique = critic_llm.generate(                  │
    │        f"Schema: {schema}\n"                        │
    │        f"Question: {question}\n"                    │
    │        f"True SQL: {true_sql}\n"                    │
    │        f"Generated: {pred_sql}\n"                   │
    │        "Why is the generated SQL wrong? Be specific."  │
    │    )                                                │
    │  aggregated = summarize(critiques)                  │
    └───────────────────────┬─────────────────────────────┘
                            │
                            ▼
    ┌─────────────────────────────────────────────────────┐
    │  REFINE STEP                                        │
    │  new_prompt = actor_llm.generate(                   │
    │      f"Current system prompt:\n{current_prompt}\n"  │
    │      f"Score: {score*100:.1f}%\n"                  │
    │      f"Common failures:\n{aggregated}\n"            │
    │      "Write an improved system prompt."             │
    │  )                                                  │
    │  if score > best_score:                             │
    │      best_prompt = current_prompt                   │
    │      best_score  = score                            │
    │  current_prompt = new_prompt                        │
    └─────────────────────────────────────────────────────┘

  best_prompt.save(output_dir / "best_prompt.json")
  return LLMGenerator(engine, best_prompt)
```

**Key properties:**
- Weights are frozen throughout — no gradient computation
- Uses `StringMatchEvaluator` in the inner loop (fast, no DB needed)
- Can run on the same GPU as inference
- Output is a JSON-serialized `PromptBuilder` — picked up automatically by the `infer` stage

---

## 9. CLI — How to Execute Each Stage

### Install the package

```bash
cd text2sql-agent-rl
pip install -e .
```

### Stage 0 — Preprocess (once, shared by all runs)

```bash
python scripts/pipeline.py --run preprocess --stages preprocess
# or: sbatch jobs/preprocess.sh
```

Produces:
- `results/preprocess/true_sql_cache_train.json`
- `results/preprocess/true_sql_cache_validation.json`

---

### Approach 1 — Baseline

```bash
# Full baseline pipeline
python scripts/pipeline.py \
    --run baseline_v1 \
    --cache_run preprocess \
    --stages infer eval_string eval_exec report

# Eval only (skip inference if already done)
python scripts/pipeline.py \
    --run baseline_v1 \
    --stages eval_string eval_exec report

# Force re-run eval_exec
python scripts/pipeline.py \
    --run baseline_v1 \
    --stages eval_exec --force
```

---

### Approach 2 — Prompt Optimization

```bash
# Run prompt optimization, then evaluate with the best prompt
python scripts/pipeline.py \
    --run prompt_opt_v1 \
    --cache_run preprocess \
    --stages optimize_prompt infer eval_string eval_exec report

# Reuse baseline inference — just re-eval (tests if prompt changes eval at all)
python scripts/pipeline.py \
    --run prompt_opt_v1 \
    --inference_from baseline_v1 \
    --stages eval_string eval_exec report
```

---

### Approach 3 — GRPO + LoRA

```bash
# Start GRPO from prompt-optimized prompt (recommended: chain the approaches)
python scripts/pipeline.py \
    --run grpo_v1 \
    --cache_run preprocess \
    --stages train_grpo infer eval_string eval_exec report

# Evaluate a specific checkpoint without re-training
python scripts/pipeline.py \
    --run grpo_v1_chk500 \
    --model_path models/lora/grpo_v1/checkpoint-500 \
    --cache_run preprocess \
    --stages infer eval_string eval_exec report

# HPC batch job
sbatch jobs/train_grpo.sh
```

---

### Compare All Approaches

```bash
python scripts/compare.py \
    --runs baseline_v1 prompt_opt_v1 grpo_v1
```

---

### Composing Approaches (recommended order)

```bash
# 1. Preprocess once
python scripts/pipeline.py --run preprocess --stages preprocess

# 2. Baseline
python scripts/pipeline.py --run baseline_v1 --cache_run preprocess \
    --stages infer eval_string eval_exec report

# 3. Optimize prompt (uses same model as baseline)
python scripts/pipeline.py --run prompt_opt_v1 --cache_run preprocess \
    --stages optimize_prompt infer eval_string eval_exec report

# 4. GRPO starting from the optimized prompt
python scripts/pipeline.py --run grpo_v1 --cache_run preprocess \
    --stages train_grpo infer eval_string eval_exec report

# 5. Compare
python scripts/compare.py --runs baseline_v1 prompt_opt_v1 grpo_v1
```

---

## 10. Results Layout

```
results/
│
├── preprocess/                               ← stage: preprocess (shared)
│   ├── true_sql_cache_train.json
│   ├── true_sql_cache_validation.json
│   └── pipeline.log
│
├── baseline_v1/
│   ├── predictions.csv                       ← stage: infer
│   ├── eval_string.csv                       ← stage: eval_string
│   ├── eval_exec.csv                         ← stage: eval_exec
│   ├── report.txt                            ← stage: report
│   └── pipeline.log
│
├── prompt_opt_v1/
│   ├── best_prompt.json                      ← stage: optimize_prompt
│   ├── opt_history.jsonl                     ← score per iteration
│   ├── predictions.csv
│   ├── eval_string.csv
│   ├── eval_exec.csv
│   ├── report.txt
│   └── pipeline.log
│
└── grpo_v1/
    ├── predictions.csv
    ├── eval_string.csv
    ├── eval_exec.csv
    ├── report.txt
    ├── training_log.jsonl                    ← {step, loss, reward, val_score}
    ├── training_done.flag                    ← path to final checkpoint
    └── pipeline.log

models/
└── lora/
    └── grpo_v1/
        ├── checkpoint-500/
        │   ├── adapter_config.json
        │   └── adapter_model.safetensors
        └── checkpoint-final/
```

### Report Format

```
════════════════════════════════════════════════════════════
BASELINE PIPELINE REPORT
Run      : baseline_v1
Time     : 2026-03-31 10:00:00
Model    : meta-llama/Meta-Llama-3.1-8B-Instruct
Samples  : 1034
════════════════════════════════════════════════════════════

Metric                    String Match    Exec Accuracy    Gap
────────────────────────────────────────────────────────────
Overall                      29.2%           45.8%       +16.6%
  Easy                       66.7%           75.0%        +8.3%
  Medium                      8.3%           25.0%       +16.7%
  Hard                       33.3%           41.7%        +8.3%
  Extra Hard                  8.3%           16.7%        +8.3%

Valid SQL rate               95.8%
Correct tables (F1)          0.7230
Composite reward             0.5821

Reliable examples            1034 / 1034
Skipped (unreliable DB)         0
════════════════════════════════════════════════════════════
```

### Comparison Table Format

```
══════════════════════════════════════════════════════════════════════════
COMPARISON — Spider Validation Set (1034 examples)
══════════════════════════════════════════════════════════════════════════

                          Baseline    Prompt Opt    GRPO+LoRA
                         baseline_v1  prompt_opt_v1   grpo_v1
────────────────────────────────────────────────────────────────────────
String Match  (overall)    29.2%        38.1%          51.4%
  Easy                     66.7%        72.3%          81.2%
  Medium                    8.3%        18.1%          32.4%
  Hard                     33.3%        41.0%          55.6%
  Extra Hard                8.3%        12.5%          21.8%

Exec Accuracy (overall)    45.8%        54.2%          63.7%
  Easy                     75.0%        80.2%          88.1%

Composite reward           0.582        0.641          0.718
Valid SQL rate             95.8%        97.1%          98.3%
══════════════════════════════════════════════════════════════════════════
```

---

## 11. Reusable Function Reference

| Function | Module | Used by |
|---|---|---|
| `SQLUtils.extract(raw)` | `inference.sql_utils` | `LLMGenerator`, `GRPOOptimizer.rollout` |
| `SQLUtils.exact_match(pred, true)` | `inference.sql_utils` | `StringMatchEvaluator` |
| `DifficultyClassifier.classify(toks)` | `data.dataset` | `SpiderDataset`, report breakdowns |
| `TrueSQLCacheBuilder.build(parquet, out)` | `data.cache` | `preprocess` stage |
| `DBQueryExecutor.execute(db_id, sql)` | `db.executor` | `ExecutionEvaluator`, `CompositeReward` |
| `DBQueryExecutor.is_valid(db_id, sql)` | `db.executor` | `CompositeReward._valid_sql()` |
| `TrainingDataFilter.is_reliable(db_id, sql)` | `db.filter` | `ExecutionEvaluator`, `CompositeReward` |
| `CompositeReward.compute(db_id, pred, true)` | `reward.composite` | `GRPOOptimizer` (live scalar), `ExecutionEvaluator` (batch) |
| `group_advantages(rewards)` | `training.rollout` | `GRPOOptimizer` |
| `sample_rollout(engine, prompt, examples, G)` | `training.rollout` | `GRPOOptimizer` |
| `default_lora_config(r, alpha)` | `training.lora` | `GRPOOptimizer` |
| `merge_adapter(base, lora_path)` | `training.lora` | `LoRAGenerator` |
| `PromptBuilder.build(question, db_id)` | `data.schema` | `LLMGenerator`, `GRPOOptimizer`, `PromptOptimizer` |
| `PromptBuilder.save/from_file(path)` | `data.schema` | `optimize_prompt` → `infer` handoff |
| `save_predictions(preds, path)` | `pipeline.io` | `infer` stage |
| `load_predictions(path)` | `pipeline.io` | `eval_string`, `eval_exec` stages |
| `ReportGenerator.generate(...)` | `eval.report` | `report` stage |
| `ComparisonTable.generate(runs)` | `eval.report` | `compare` script |

---

## Key Design Principles

1. **`SQLGenerator` is the unit of comparison** — optimize it however you want, the evaluation pipeline never changes.

2. **`Prediction` is the serialization boundary** — generators write `predictions.csv`, evaluators read it. Inference never re-runs just to re-evaluate.

3. **Optimizers are composable** — `PromptOptimizer` output feeds into `GRPOOptimizer` input. Chain them for best results.

4. **Reward ≠ Evaluator** — `CompositeReward` returns a fast scalar for live training. `ExecutionEvaluator` returns a rich DataFrame for analysis. They share `DBQueryExecutor` but serve different masters.

5. **Cache is the performance multiplier** — true SQL executed once, looked up in 0ms during training and evaluation. Without it, GRPO is 10× slower.

6. **One result folder per run** — fully reproducible by re-running with the same `RunConfig`. Config saved to `run_config.yaml` inside the run folder.
