# Training & Evaluation Command Reference

All commands listed here assume you are running from inside `$PROJECT` (`~/TEXT2SQL-AGENT-RL/`) on the cluster, or from the repository root locally.

---

## 0. Preprocessing (Run once, globally)

Executes all ground-truth SQL against local SQLite databases and caches the result matrices. Every training method shares this cache via `--cache_run preprocess`.

```bash
# Interactive
python scripts/pipeline.py --run preprocess --stages preprocess

# SLURM
sbatch jobs/00_preprocess.sh
```

---

## 1. Baseline

No training. Runs frozen Llama 3.1 8B Instruct directly on the validation set with the default system prompt.

```bash
# Interactive (50 samples)
python scripts/pipeline.py \
    --run baseline_v1 \
    --cache_run preprocess \
    --stages infer eval_string eval_exec report \
    --n_samples 50 --dtype bfloat16

# SLURM (full validation set)
sbatch jobs/01_baseline.sh
```

---

## 2. Prompt Optimization (Actor-Critique)

Runs an iterative loop that rewrites the system prompt based on SQLite execution failures. Saves `best_prompt.json`, which `infer` picks up automatically.

```bash
# Interactive (2 iterations, 10 samples)
python scripts/pipeline.py \
    --run fast_prompt_opt \
    --cache_run preprocess \
    --stages optimize_prompt infer eval_string eval_exec report \
    --n_opt_iterations 2 --opt_sample_size 10 --n_samples 50 --dtype bfloat16

# SLURM (5 iterations, 100-sample optimization, full val set)
sbatch jobs/02_prompt_opt.sh
```

Monitor: `cat results/prompt_opt_v1/opt_history.jsonl`

---

## 3. SFT — Supervised Fine-Tuning

Fits a LoRA adapter using cross-entropy loss against ground-truth SQL completions. No reward computation. Faster per step than GRPO.

```bash
# Interactive (5 steps, checkpoint every 2)
python scripts/pipeline.py \
    --run fast_sft \
    --cache_run preprocess \
    --stages train_sft infer eval_string eval_exec report \
    --sft_n_steps 5 --batch_size 2 --checkpoint_every 2 \
    --infer_model sft --dtype bfloat16

# SLURM (1000 steps)
sbatch jobs/04_sft.sh

# Override GPU at submission
sbatch --gres=gpu:h100:1 jobs/04_sft.sh
```

Monitor: `cat results/sft_v1/sft_training_log.jsonl`

Checkpoint location: `models/lora/sft_v1/sft/`

---

## 4. GRPO — Reinforcement Learning (LoRA)

Fits a LoRA adapter with group-relative policy gradient, using SQLite execution accuracy as the reward signal.

```bash
# Interactive (10 steps, small batch)
python scripts/pipeline.py \
    --run fast_grpo \
    --cache_run preprocess \
    --stages train_grpo infer eval_string eval_exec report \
    --n_steps 10 --batch_size 2 --group_size 4 \
    --kl_coef 0.1 --reward_fn composite \
    --checkpoint_every 5 --n_samples 50 --dtype bfloat16

# SLURM (1000 steps)
sbatch jobs/03_grpo.sh

# Override GPU at submission
sbatch --gres=gpu:h100:1 jobs/03_grpo.sh
```

Monitor: `cat results/grpo_v1/training_log.jsonl`

Checkpoint location: `models/lora/grpo_v1/`

---

## 5. Resuming Training After a Crash

Both GRPO and SFT resume automatically when the job is resubmitted — they scan their checkpoint directory for the highest `checkpoint-<N>` and continue from the next step.

```bash
# Automatic resume (just resubmit the same job)
sbatch jobs/03_grpo.sh
sbatch jobs/04_sft.sh

# Resume GRPO from a specific checkpoint
sbatch --export=ALL,GRPO_RESUME_FROM=models/lora/grpo_v1/checkpoint-400 jobs/03_grpo.sh

# Resume SFT from a specific checkpoint
sbatch --export=ALL,SFT_RESUME_FROM=models/lora/sft_v1/sft/checkpoint-400 jobs/04_sft.sh

# Resume GRPO from best checkpoint (useful if training diverged after the peak)
sbatch --export=ALL,GRPO_RESUME_FROM=models/lora/grpo_v1/checkpoint-best jobs/03_grpo.sh
```

---

## 6. Running Inference on a Specific Method

The `--infer_model` flag controls which adapter the `infer` stage loads. Use this to evaluate a previously trained model without re-running the whole pipeline.

```bash
# Baseline (no adapter)
python scripts/pipeline.py --run cmp_base --infer_model none \
    --stages infer eval_string eval_exec report

# SFT adapter
python scripts/pipeline.py --run cmp_sft --infer_model sft \
    --stages infer eval_string eval_exec report

# GRPO adapter
python scripts/pipeline.py --run cmp_grpo --infer_model grpo \
    --stages infer eval_string eval_exec report

# Specific checkpoint path
python scripts/pipeline.py --run cmp_ckpt500 \
    --infer_model models/lora/grpo_v1/checkpoint-500 \
    --stages infer eval_string eval_exec report
```

---

## 7. Final Comparison Report

After all experiments complete, generate the cross-run comparison matrix:

```bash
python scripts/compare.py \
    --runs baseline_v1 prompt_opt_v1 sft_v1 grpo_v1 \
    --output results/final_comparison.txt
```
