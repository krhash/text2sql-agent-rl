#!/bin/bash
#SBATCH --job-name=sft_v1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
# Override GPU at submission time:
#   sbatch --gres=gpu:h100:1 jobs/04_sft.sh
#   sbatch --gres=gpu:l40:1  jobs/04_sft.sh
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=/home/%u/TEXT2SQL-AGENT-RL/results/sft_v1/slurm_%j.out
#SBATCH --error=/home/%u/TEXT2SQL-AGENT-RL/results/sft_v1/slurm_%j.err

PROJECT=$HOME/TEXT2SQL-AGENT-RL
mkdir -p $PROJECT/results/sft_v1/

export HF_HOME=/scratch/$USER/hf_cache
export TRANSFORMERS_CACHE=/scratch/$USER/hf_cache
mkdir -p $HF_HOME

module load anaconda3/2024.06
source activate texttosql

echo "-- Run -------------------------------------------------------------------"
echo "Job started : $(date)"
echo "Node        : $(hostname)"
echo "GPU         : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Task        : SFT Supervised Fine-Tuning"

python $PROJECT/scripts/pipeline.py \
    --run sft_v1 \
    --cache_run preprocess \
    --stages train_sft infer eval_string eval_exec report \
    --sft_n_steps 1000 \
    --batch_size 8 \
    --checkpoint_every 200 \
    --infer_model sft

echo "SFT experiment complete."
