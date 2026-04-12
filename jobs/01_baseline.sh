#!/bin/bash
#SBATCH --job-name=baseline_v1
#SBATCH --partition=sharing
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=48GB
#SBATCH --cpus-per-task=4
#SBATCH --time=00:59:00
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

cd $PROJECT
python $PROJECT/scripts/pipeline.py \
    --run baseline_v1 \
    --cache_run preprocess \
    --stages infer eval_string eval_exec report \
    --infer_model none \
    --dtype float16

echo "Baseline experiment complete."
