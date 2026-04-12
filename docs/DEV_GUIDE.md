# Developer Guide: Text-to-SQL RL Pipeline

This document covers environment setup, cluster deployment, pipeline architecture, and execution commands for developers working on this project.

The project uses Llama 3.1 8B Instruct optimized via three approaches: Actor-Critic Prompt Optimization, Supervised Fine-Tuning (SFT), and Group Relative Policy Optimization (GRPO) with LoRA adapters.

---

## 1. Local Environment Setup

Requires Python 3.11+.

```bash
# 1. Create and activate the conda environment
conda create -n texttosql python=3.11 -y
conda activate texttosql

# 2. Clone the repository
git clone https://github.com/krhash/text2sql-agent-rl.git
cd text2sql-agent-rl

# 3. Install the package in editable mode with all dependencies
pip install -e .

# 4. Authenticate with HuggingFace (required for gated Llama 3.1 weights)
huggingface-cli login
```

---

## 2. Cluster Deployment (HPC)

Sync your local code to the cluster. Exclude `models/` and `results/` — those are generated on the cluster and are too large to transfer.

```bash
# Preferred: rsync (supports --exclude)
rsync -avz --exclude '.git' --exclude 'models/' --exclude 'results/' --exclude '__pycache__/' \
   /path/to/local/text2sql-agent-rl/ your_username@login.discovery.neu.edu:~/TEXT2SQL-AGENT-RL/
```

```bash
# Alternative: scp (does not support folder exclusions — be careful)
scp -r /path/to/local/text2sql-agent-rl/ your_username@login.discovery.neu.edu:~/TEXT2SQL-AGENT-RL/
```

All `jobs/*.sh` scripts automatically redirect `HF_HOME` and `TRANSFORMERS_CACHE` to `/scratch/$USER/hf_cache`, preventing home directory quota exhaustion from model downloads.

---

## 3. Pipeline Architecture

The pipeline (`scripts/pipeline.py`) sequences 8 stages. State is passed between stages via output files on disk.

| Stage | Output file | Description |
|---|---|---|
| `preprocess` | `true_sql_cache_validation.json` | Execute ground-truth SQL, cache results |
| `optimize_prompt` | `best_prompt.json` | Actor-Critic prompt rewriting loop |
| `train_sft` | `sft_done.flag` | SFT LoRA fine-tuning on training set |
| `train_grpo` | `training_done.flag` | GRPO RL fine-tuning on training set |
| `infer` | `predictions.csv` | Deterministic inference on validation set |
| `eval_string` | `eval_string.csv` | Syntactic exact match scoring |
| `eval_exec` | `eval_exec.csv` | SQLite execution accuracy scoring |
| `report` | `report.txt` | Difficulty-stratified summary table |

The `infer` stage behavior is controlled by `--infer_model`:
- `auto` — auto-detects from flag files (default, backward compatible)
- `none` — plain frozen Llama, no adapter (baseline)
- `grpo` — loads LoRA from `training_done.flag`
- `sft` — loads LoRA from `sft_done.flag`
- `<path>` — loads LoRA directly from a filesystem path

---

## 4. Checkpoint Layout & Crash Recovery

Both `train_sft` and `train_grpo` write structured checkpoint directories:

```
models/lora/<run_name>/
  checkpoint-200/      <- periodic snapshot, auto-scanned for resume
  checkpoint-400/      <- periodic snapshot, auto-scanned for resume
  checkpoint-best/     <- best validation score, used by infer stage
  checkpoint-final/    <- end-of-training state, for reference
```

**Stage skipping:** If a stage's output file exists, it is skipped automatically on resubmit unless `--force` is passed.

**Inference resuming:** The `infer` stage appends to `predictions.csv` after each example. A crash at row 400/1000 resumes from row 401.

**LoRA checkpoint resume:** On restart, the trainer scans for the highest `checkpoint-<N>` directory and resumes from the next step automatically. No flags needed.

**Manual checkpoint override:**
```bash
# Resume GRPO from a specific checkpoint
sbatch --export=ALL,GRPO_RESUME_FROM=models/lora/grpo_v1/checkpoint-400 jobs/03_grpo.sh

# Resume SFT from a specific checkpoint
sbatch --export=ALL,SFT_RESUME_FROM=models/lora/sft_v1/sft/checkpoint-400 jobs/04_sft.sh

# Resume GRPO from best checkpoint (if training diverged after peak)
sbatch --export=ALL,GRPO_RESUME_FROM=models/lora/grpo_v1/checkpoint-best jobs/03_grpo.sh
```

---

## 5. Execution Reference

### A. Interactive Verification (GPU node)

Use these to verify the environment before submitting full jobs.

```bash
# Baseline
python scripts/pipeline.py \
    --run baseline_test \
    --cache_run preprocess \
    --stages infer eval_string eval_exec report \
    --n_samples 50 --dtype bfloat16

# Prompt Optimization
python scripts/pipeline.py \
    --run fast_prompt_opt \
    --cache_run preprocess \
    --stages optimize_prompt infer eval_string eval_exec report \
    --n_opt_iterations 2 --opt_sample_size 10 --n_samples 50 --dtype bfloat16

# SFT
python scripts/pipeline.py \
    --run fast_sft \
    --cache_run preprocess \
    --stages train_sft infer eval_string eval_exec report \
    --sft_n_steps 5 --batch_size 2 --checkpoint_every 2 \
    --infer_model sft --dtype bfloat16

# GRPO
python scripts/pipeline.py \
    --run fast_grpo \
    --cache_run preprocess \
    --stages train_grpo infer eval_string eval_exec report \
    --n_steps 10 --batch_size 2 --group_size 4 \
    --checkpoint_every 5 --n_samples 50 --dtype bfloat16
```

### B. Production Training (SLURM)

```bash
sbatch jobs/00_preprocess.sh     # Run once globally
sbatch jobs/01_baseline.sh       # Baseline
sbatch jobs/02_prompt_opt.sh     # Prompt optimization
sbatch jobs/04_sft.sh            # SFT fine-tuning
sbatch jobs/03_grpo.sh           # GRPO RL fine-tuning

# Override GPU type at submission for any job
sbatch --gres=gpu:h100:1 jobs/03_grpo.sh
sbatch --gres=gpu:h100:1 jobs/04_sft.sh

# Or run all stages sequentially
sbatch jobs/05_run_everything.sh
```

### C. Comparison Report

After all experiments finish, generate the cross-run comparison matrix:

```bash
python scripts/compare.py \
    --runs baseline_v1 prompt_opt_v1 sft_v1 grpo_v1 \
    --output results/final_comparison.txt
```

For full command details and resume patterns, see [`docs/HPC_TRAINING_COMMANDS.md`](HPC_TRAINING_COMMANDS.md).
