#!/bin/bash
#SBATCH --job-name=baseline_v1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1          # A100 40GB minimum required
#SBATCH --mem=64GB                 # CPU RAM
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=/home/%u/TEXT2SQL-AGENT-RL/results/baseline_v1/slurm_%j.out
#SBATCH --error=/home/%u/TEXT2SQL-AGENT-RL/results/baseline_v1/slurm_%j.err

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT=$HOME/TEXT2SQL-AGENT-RL
mkdir -p $PROJECT/results/baseline_v1/

# Point HuggingFace model cache to scratch (models are large, ~16GB)
export HF_HOME=/scratch/$USER/hf_cache
export TRANSFORMERS_CACHE=/scratch/$USER/hf_cache
mkdir -p $HF_HOME

# ── Env ───────────────────────────────────────────────────────────────────────
module load anaconda3/2024.06
source activate texttosql

# ── Run ───────────────────────────────────────────────────────────────────────
echo "Job started : $(date)"
echo "Node        : $(hostname)"
echo "GPU         : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Task        : Baseline Llama 3.1 8B Inference"

# Requires the preprocess cache to have been run
python $PROJECT/scripts/pipeline.py \
    --run baseline_v1 \
    --cache_run preprocess \
    --stages infer eval_string eval_exec report

echo "Baseline experiment complete."
