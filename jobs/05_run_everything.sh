#!/bin/bash
#SBATCH --job-name=master_run
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1          # Expected to run continuously on GPU
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00       # Max wall-time
#SBATCH --output=/home/%u/TEXT2SQL-AGENT-RL/results/master_run/slurm_%j.out
#SBATCH --error=/home/%u/TEXT2SQL-AGENT-RL/results/master_run/slurm_%j.err

PROJECT=$HOME/TEXT2SQL-AGENT-RL
mkdir -p $PROJECT/results/master_run/

export HF_HOME=/scratch/$USER/hf_cache
export TRANSFORMERS_CACHE=/scratch/$USER/hf_cache
mkdir -p $HF_HOME

module load anaconda3/2024.06
source activate texttosql

echo "── Run ───────────────────────────────────────────────────────────────────────"
echo "Job started : $(date)"
echo "Node        : $(hostname)"
echo "GPU         : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Task        : Full Pipeline End-to-End Run"

python $PROJECT/scripts/pipeline.py \
    --run master_run \
    --stages preprocess optimize_prompt train_grpo infer eval_string eval_exec report \
    --n_opt_iterations 5 \
    --n_steps 1000 \
    --group_size 4 \
    --kl_coef 0.1 \
    --reward_fn composite

echo "Master pipeline complete! Fully optimized model is ready."
