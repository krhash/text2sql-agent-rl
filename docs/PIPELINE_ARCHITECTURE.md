# Pipeline Architecture

This document describes the architecture of the `text2sql` package, the flow of each pipeline stage, and how data moves between stages.

---

## Class Hierarchy

### `ExperimentRunner` — Orchestrator
**Location:** `text2sql/pipeline/runner.py`

Owns one named experiment (e.g. `grpo_v1`). Executes a sequence of stages, skipping any whose primary output file already exists on disk. Reads all parameters from `RunConfig`. One runner per run; compare runs via `scripts/compare.py`.

### `SQLGenerator` — Inference Interfaces
**Location:** `text2sql/inference/`
- **`LLMGenerator`**: Frozen base model + `PromptBuilder`. Used by Baseline and Prompt Opt.
- **`LoRAGenerator`**: Loads a saved LoRA adapter checkpoint and merges it into the base model at init. Used by GRPO and SFT.

### `SQLOptimizer` — Training Interfaces
**Location:** `text2sql/training/`
- **`IdentityOptimizer`**: No-op. Returns input generator unchanged. Used for Baseline.
- **`PromptOptimizer`**: Actor-Critique loop. Iteratively rewrites the system prompt based on SQLite execution failures.
- **`SFTOptimizer`**: Supervised Fine-Tuning. Cross-entropy loss computed only on SQL completion tokens (prompt masked to -100). LoRA adapter trained with AdamW.
- **`GRPOOptimizer`**: Group Relative Policy Optimization. Stochastic rollouts + SQLite execution reward + KL-penalty policy gradient. LoRA adapter trained with AdamW.

### `BaseReward` / `SQLEvaluator` — Scoring
**Location:** `text2sql/eval/`, `text2sql/reward/`
- Ground truth SQL is **never re-executed live** during training. It is pre-cached to `true_sql_cache_train.json` during the `preprocess` stage.
- `CompositeReward`: exec accuracy (50%) + valid SQL (20%) + table F1 (20%) + efficiency (10%).
- `StringMatchEvaluator` / `ExecutionEvaluator`: used post-inference to grade `predictions.csv`.

---

## Stage Reference

| Stage | Output file | Description |
|---|---|---|
| `preprocess` | `true_sql_cache_validation.json` | Execute ground truth SQL, cache results to disk |
| `optimize_prompt` | `best_prompt.json` | Actor-Critique loop on training set |
| `train_sft` | `sft_done.flag` | SFT LoRA fine-tuning on training set |
| `train_grpo` | `training_done.flag` | GRPO RL fine-tuning on training set |
| `infer` | `predictions.csv` | Deterministic generation on validation set |
| `eval_string` | `eval_string.csv` | Syntactic exact match scoring |
| `eval_exec` | `eval_exec.csv` | SQLite execution accuracy scoring |
| `report` | `report.txt` | Difficulty-stratified summary table |

The pipeline runner skips any stage whose output file already exists. Pass `--force` to re-run a stage unconditionally.

---

## Checkpoint Layout

Both SFT and GRPO produce a structured checkpoint directory under `models/lora/<run_name>/`:

```
models/lora/<run_name>/
  checkpoint-200/      <- periodic snapshot, for resume after crash
  checkpoint-400/      <- periodic snapshot, for resume after crash
  checkpoint-best/     <- best val_score seen, used by infer stage
  checkpoint-final/    <- weights at end of training, for reference
```

- **Resume training**: auto-scans for the highest `checkpoint-<N>` on restart
- **Inference**: always loads `checkpoint-best`; flag files point to this path
- **Manual override**: `--grpo_resume_from` / `--sft_resume_from` to seed from a specific path

---

## Infer Stage — Adapter Selection

The `infer` stage supports explicit adapter selection via `--infer_model`:

| Value | Behaviour |
|---|---|
| `auto` | Auto-detects from flag files. GRPO > SFT > plain Llama. Default. |
| `none` | Frozen Llama, no adapter (equivalent to Baseline run) |
| `grpo` | Loads LoRA path from `training_done.flag` |
| `sft` | Loads LoRA path from `sft_done.flag` |
| `<path>` | Loads LoRA directly from a given filesystem path |

This enables running all four methods against the same validation set with identical inference code:

```bash
python scripts/pipeline.py --run cmp_base  --infer_model none  --stages infer eval_string eval_exec report
python scripts/pipeline.py --run cmp_sft   --infer_model sft   --stages infer eval_string eval_exec report
python scripts/pipeline.py --run cmp_grpo  --infer_model grpo  --stages infer eval_string eval_exec report
python scripts/compare.py  --runs cmp_base cmp_sft cmp_grpo
```

---

## Pipeline Flags

| Flag | Description |
|---|---|
| `--cache_run <name>` | Borrow `true_sql_cache_*.json` from another run's results folder |
| `--inference_from <name>` | Copy `predictions.csv` from another run, skip generation |
| `--infer_model` | Explicitly set which adapter the `infer` stage loads |
| `--grpo_resume_from <path>` | Seed GRPO resume from a specific checkpoint directory |
| `--sft_resume_from <path>` | Seed SFT resume from a specific checkpoint directory |
| `--checkpoint_every <N>` | How often to save a periodic checkpoint (default: 500) |
| `--force` | Re-run a stage even if its output already exists |
