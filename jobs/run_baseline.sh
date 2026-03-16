#!/bin/bash
#SBATCH --job-name=baseline_inference
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/home/%u/TEXT2SQL-AGENT-RL/logs/baseline_%j.out
#SBATCH --error=/home/%u/TEXT2SQL-AGENT-RL/logs/baseline_%j.err

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT=$HOME/TEXT2SQL-AGENT-RL

mkdir -p $PROJECT/logs
mkdir -p $PROJECT/results

# Point HuggingFace model cache to scratch (models are large, ~16GB)
export HF_HOME=/scratch/$USER/hf_cache
export TRANSFORMERS_CACHE=/scratch/$USER/hf_cache
mkdir -p $HF_HOME

module load anaconda3/2024.06
source activate texttosql

# ── Run ───────────────────────────────────────────────────────────────────────
echo "Job started : $(date)"
echo "Node        : $(hostname)"
echo "GPU         : $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python $PROJECT/scripts/baseline_inference.py \
    --model_id    meta-llama/Meta-Llama-3.1-8B-Instruct \
    --data_path   $PROJECT/dataset/validation-00000-of-00001.parquet \
    --schema_path $PROJECT/dataset/spider_schema_rows_v2.json \
    --output_path $PROJECT/results/baseline_results.csv \
    --n_samples   50 \
    --dtype       bfloat16

echo "Job finished: $(date)"
