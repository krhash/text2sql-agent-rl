# Executing Training & Evaluation pipelines

This document outlines the exact `pipeline.py` sequences to run the different stages of the RL Text-To-SQL architecture. 

It covers both **Interactive Testing** (for the login / interactive nodes) and **Production SLURM batch jobs** (for the massive training runs).

---

## 1. Initial Setup: Global Preprocessing
Before running *any* model inference, you must pre-execute the ground-truth SQL datasets against SQLite. This caches the matrices directly to disk so the GRPO algorithm has a fast mathematical Reward signal without bogging down the CPU dynamically.

```bash
# INTERACTIVE / LOCAL
python scripts/pipeline.py \
    --run preprocess \
    --stages preprocess

# SLURM
sbatch jobs/00_preprocess.sh
```

---

## 2. Baseline Model
This runs `infer` across the original frozen Llama 3.1 8B Instruct model using the default, hand-engineered system prompt. It serves as your mathematical baseline.

```bash
# INTERACTIVE (Micro-test on 50 samples)
python scripts/pipeline.py \
    --run baseline_v1 \
    --cache_run preprocess \
    --stages infer eval_string eval_exec report \
    --n_samples 50 \
    --dtype bfloat16

# SLURM (Runs full Spider dataset)
sbatch jobs/01_baseline.sh
```

---

## 3. Prompt Optimization (Actor-Critique)
This discovers a brand new instructional System Prompt. The model is forced to critique its own SQL failures and organically rewrite its own System prompt to maximize Strict String Adherence over multiple iterations.

```bash
# INTERACTIVE (Micro-test constraints)
python scripts/pipeline.py \
    --run fast_prompt_opt \
    --cache_run preprocess \
    --stages optimize_prompt infer eval_string eval_exec report \
    --n_opt_iterations 2 \
    --opt_sample_size 10 \
    --n_samples 50 \
    --dtype bfloat16

# SLURM (Tests 5 iterations against a batch of 100 before scoring)
sbatch jobs/02_prompt_opt.sh
```
*Tracking file: `cat results/prompt_opt_v1/opt_history.jsonl`*

---

## 4. GRPO Reinforcement Learning (LoRA)
This structurally injects a Parameter-Efficient Fine-Tuning adapter into Llama 3.1 entirely via PyTorch. It gradients the attention weights over 1,000 steps utilizing execution advantage scoring over multiple stochastic rollouts against physical SQLite databases. 

```bash
# INTERACTIVE (Micro-test — check for OOM mapping issues)
python scripts/pipeline.py \
    --run fast_grpo \
    --cache_run preprocess \
    --stages train_grpo infer eval_string eval_exec report \
    --n_steps 10 \
    --batch_size 2 \
    --group_size 4 \
    --kl_coef 0.1 \
    --reward_fn composite \
    --n_samples 50 \
    --dtype bfloat16

# SLURM (Trains 1,000 full gradient descent steps over 6-12 hours)
sbatch jobs/03_grpo.sh
```
*Tracking file: `cat results/grpo_v1/training_log.jsonl` or `tail -f logs/grpo_<job_id>.out`*

---

## 5. End-to-End Orchestrator
To seamlessly launch the entirety of the project directly into the cluster one after another.

```bash
# Wait for node availability and run 00 -> 01 -> 02 -> 03 automatically
sbatch jobs/05_run_everything.sh
```

---

## 6. Final Project Reporting
After successfully evaluating the Baseline, Prompt Optimizer, and GRPO RL pipelines, you can run this command locally to automatically suck up all `.csv` predictions generated in `results/<run_name>/predictions.csv` and dump them into a final comparative matrix suitable for an academic paper.

```bash
python scripts/compare.py \
    --runs baseline_v1 prompt_opt_v1 grpo_v1 \
    --output results/final_paper_matrix.txt
```
