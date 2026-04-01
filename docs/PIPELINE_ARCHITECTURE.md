# Pipeline Architecture & Hierarchy

This document explains the architecture of the new `text2sql` Python package, the flow of events during the pipeline, and how to execute the different training and evaluation stages.

## 🏗️ Architecture & Class Hierarchy

The codebase is organized as a modular, pip-installable module named `text2sql`. The flat scripts approach was retired in favor of a central execution manager.

### 1. `ExperimentRunner` (The Orchestrator)
**Location:** `text2sql/pipeline/runner.py`
This class controls the lifecycle of an entire experiment. It handles loading datasets, triggering the relevant optimization loops, executing inference, driving evaluations, and generating final reports. It reads exactly what parameters the experiment is using via the `RunConfig` dataclass.

### 2. `SQLGenerator` (Inference Interfaces)
**Location:** `text2sql/inference/`
The common `Protocol` for generating SQL predictions.
- **`LLMGenerator`**: Uses frozen base model weights paired with a `PromptBuilder` for few-shot generation.
- **`LoRAGenerator`**: A sub-class of `LLMGenerator` that dynamically merges a custom LoRA weights adapter checkpoint upon instantiation.

### 3. `SQLOptimizer` (Training Interfaces)
**Location:** `text2sql/training/`
An `SQLOptimizer` is essentially defined as: "Take an initial `SQLGenerator` + Training Data -> Return an **improved** `SQLGenerator`".
- **`IdentityOptimizer`**: A no-op optimizer that simply returns the input directly. Used for the baseline Llama-3.1 experiment.
- **`PromptOptimizer`**: Implements an Actor-Critique loop. It generates SQL, critiques any failed execution output using an LLM critic, refines the system prompt, and returns a new `LLMGenerator` attached to that better prompt.
- **`GRPOOptimizer`**: Implements the Reinforcement Learning Group-Relative Policy Optimization loop. It handles generating rollouts, computing the KL-penalty policy-gradient, backpropagating loss, and returning a `LoRAGenerator` initialized with the best LoRA checkpoint.

### 4. `BaseReward` & `SQLEvaluator` (Scoring Interfaces)
**Location:** `text2sql/eval/` and `text2sql/reward/`
- **`ExecutionEvaluator` / `CompositeReward`**: Executes generated SQL queries against the local SQLite databases and scores their exact row matches and efficiency.
- **True SQL Cache**: We *never* run Ground Truth SQL live! Before running any ML pipelines, the Ground Truth SQL is pre-evaluated and its expected results are stored in memory via `true_sql_cache_train.json`. 

---

## ⚙️ Complete Flow of Events

When you trigger a full job using a command like `--stages train_grpo infer eval_string eval_exec report`, the `ExperimentRunner` sequentially walks through the phases:

1. **`train_grpo` Stage**: 
   - Uses the `TrueSQLCache` to instantly verify query accuracy during LLM temperature rollouts.
   - Runs the RL alignment loop updating the LoRA weights parameters.
   - Saves a final LoRA checkpoint and writes a `training_done.flag` signal to the output directory.
2. **`infer` Stage**:
   - The runner detects the presence of the `training_done.flag`, meaning it naturally constructs a `LoRAGenerator` with those LoRA weights (rather than the default frozen `LLMGenerator`).
   - Generates and executes SQL over the validation set, creating `predictions.csv`.
3. **`eval_string` / `eval_exec` Stages**:
   - Both evaluators load and read `predictions.csv`. They *never* perform any LLM generation themselves!
   - They independently compute pure string exact matches and SQLite execution accuracies, writing to `eval_string.csv` and `eval_exec.csv`.
4. **`report` Stage**:
   - Reads both CSVs and outputs a difficulty-stratified accuracy matrix grouping the different execution and string match scores.

---

## 🏃‍♂️ How to Run the 3 Experiments

You can execute these locally or by submitting the pre-configured SLURM scripts found in the `jobs/` directory. Be sure to run Step 0 exactly once!

### Step 0. Build the Execution Cache (Mandatory Initial Step)
```bash
# Run locally:
python scripts/pipeline.py --run preprocess --stages preprocess

# Or via SLURM:
sbatch jobs/00_preprocess.sh
```

### Approach 1. Baseline Llama-3.1
Runs the model completely frozen using few-shot base prompts.
```bash
sbatch jobs/01_baseline.sh
```

### Approach 2. Prompt Optimization (Actor-Critique)
Tunes the prompt, saving `best_prompt.json`, and then automatically behaves using that prompt through inference.
```bash
sbatch jobs/02_prompt_opt.sh
```

### Approach 3. GRPO + LoRA Reinforcement Learning
Updates LoRA weights by optimizing for SQL execution accuracy reward signals over 1,000 steps.
```bash
sbatch jobs/03_grpo.sh
```

### Compare the Output Results
Generate a side-by-side comparison table showing all three models' exact match and execution accuracy metrics matrix:
```bash
sbatch jobs/04_compare.sh

# The output is sent to: results/final_comparison_matrix.txt
```

---

## 🎯 Pipeline Pointer Flags (`--cache_run` & `--inference_from`)

The `ExperimentRunner` allows you to share outputs across completely different job runs to save massive amounts of GPU time. This prevents you from having to rerun LLM inference or rebuild execution caches over and over again.

### 1. `--cache_run <run_name>`
When generating SQL and validating its accuracy against the Ground Truth SQLite executions, the pipeline needs the `true_sql` JSON caches. If you don't use this flag, the pipeline expects the caches to exist directly inside your current run folder (e.g., `results/my_new_run/`).

By passing `--cache_run preprocess`:
> *"Hey, don't look in my current folder for the Ground Truth SQLite caches. Look inside `results/preprocess/` and loan them to me!"*

### 2. `--inference_from <run_name>`
This is an incredibly helpful shortcut if you are testing new evaluation metrics, or if you want to reuse a previously generated set of predictions without spending hours spinning up the GPU.

If you pass `--inference_from baseline_v1`, the orchestrator does the following when it hits the `infer` stage:
1. It looks at `results/baseline_v1/predictions.csv`.
2. It completely copies that CSV into your current run's folder.
3. It entirely skips executing the LLM generation, moving straight to the evaluation metrics.

**Example Usage:** Let's say you write a brand new Evaluator metric called `CostEstimator` and add it to the pipeline. You want to see how the previously trained Baseline model scores on it. You would simply run:

```bash
python scripts/pipeline.py \
    --run test_new_metric \
    --inference_from baseline_v1 \
    --stages infer eval_cost_estimator report
```
