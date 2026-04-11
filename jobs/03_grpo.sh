#!/bin/bash
#SBATCH --job-name=grpo_v1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1          # A single A100 40GB/80GB handles 8B base + LoRA
# Override GPU at submission time:
#   sbatch --gres=gpu:h100:1 jobs/03_grpo.sh
#   sbatch --gres=gpu:l40:1  jobs/03_grpo.sh
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00       # RL training runs longer
#SBATCH --output=/home/%u/TEXT2SQL-AGENT-RL/results/grpo_v1/slurm_%j.out
#SBATCH --error=/home/%u/TEXT2SQL-AGENT-RL/results/grpo_v1/slurm_%j.err

PROJECT=$HOME/TEXT2SQL-AGENT-RL
mkdir -p $PROJECT/results/grpo_v1/

export HF_HOME=/scratch/$USER/hf_cache
export TRANSFORMERS_CACHE=/scratch/$USER/hf_cache
mkdir -p $HF_HOME

module load anaconda3/2024.06
source activate texttosql

echo "── Run ───────────────────────────────────────────────────────────────────────"
echo "Job started : $(date)"
echo "Node        : $(hostname)"
echo "GPU         : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Task        : GRPO Policy Gradient Optimization"

python $PROJECT/scripts/pipeline.py \
    --run grpo_v1 \
    --cache_run preprocess \
    --stages train_grpo infer eval_string eval_exec report \
    --n_steps 1000 \
    --group_size 4 \
    --kl_coef 0.1 \
    --reward_fn composite

echo "GRPO experiment complete."
