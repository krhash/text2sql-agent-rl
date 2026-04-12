#!/bin/bash
#SBATCH --job-name=sft_v1
#SBATCH --partition=sharing
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
# Override GPU at submission time:
#   sbatch --gres=gpu:a100:1 jobs/04_sft.sh
#   sbatch --gres=gpu:l40:1  jobs/04_sft.sh
#SBATCH --mem=48GB
#SBATCH --cpus-per-task=4
#SBATCH --time=00:59:00
#SBATCH --output=/home/%u/TEXT2SQL-AGENT-RL/results/sft_v1/slurm_%j.out
#SBATCH --error=/home/%u/TEXT2SQL-AGENT-RL/results/sft_v1/slurm_%j.err

PROJECT=$HOME/TEXT2SQL-AGENT-RL
mkdir -p $PROJECT/results/sft_v1/

export HF_HOME=/scratch/$USER/hf_cache
export TRANSFORMERS_CACHE=/scratch/$USER/hf_cache
mkdir -p $HF_HOME

module load anaconda3/2024.06
source activate texttosql

# Must cd to project root so relative paths (dataset/, results/) resolve correctly.
# SFT LoRA checkpoints are saved to models/lora/sft_v1/sft/ relative to this directory.
cd $PROJECT

echo "-- Run -------------------------------------------------------------------"
echo "Job started : $(date)"
echo "Node        : $(hostname)"
echo "GPU         : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Checkpoint  : ${SFT_RESUME_FROM:-auto (scans models/lora/sft_v1/sft/)}"
echo "Task        : SFT Supervised Fine-Tuning"

# To resume from a specific checkpoint, pass SFT_RESUME_FROM at submission:
#   SFT_RESUME_FROM=models/lora/sft_v1/sft/checkpoint-400 sbatch jobs/04_sft.sh
#   sbatch --export=ALL,SFT_RESUME_FROM=models/lora/sft_v1/sft/checkpoint-400 jobs/04_sft.sh

python $PROJECT/scripts/pipeline.py \
    --run sft_v1 \
    --cache_run preprocess \
    --stages train_sft infer eval_string eval_exec report \
    --sft_n_steps 1000 \
    --batch_size 8 \
    --checkpoint_every 50 \
    --infer_model sft \
    --dtype bfloat16 \
    ${SFT_RESUME_FROM:+--sft_resume_from $SFT_RESUME_FROM}

echo "SFT experiment complete."
