#!/bin/bash
#SBATCH --job-name=prompt_opt_v1
#SBATCH --partition=sharing
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=48GB
#SBATCH --cpus-per-task=4
#SBATCH --time=00:59:00
#SBATCH --output=/home/%u/TEXT2SQL-AGENT-RL/results/prompt_opt_v1/slurm_%j.out
#SBATCH --error=/home/%u/TEXT2SQL-AGENT-RL/results/prompt_opt_v1/slurm_%j.err

PROJECT=$HOME/TEXT2SQL-AGENT-RL
mkdir -p $PROJECT/results/prompt_opt_v1/

export HF_HOME=/scratch/$USER/hf_cache
export TRANSFORMERS_CACHE=/scratch/$USER/hf_cache
mkdir -p $HF_HOME

module load anaconda3/2024.06
source activate texttosql

echo "── Run ───────────────────────────────────────────────────────────────────────"
echo "Job started : $(date)"
echo "Node        : $(hostname)"
echo "GPU         : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Task        : Actor-Critique Prompt Optimization"

cd $PROJECT
python $PROJECT/scripts/pipeline.py \
    --run prompt_opt_v1 \
    --cache_run preprocess \
    --stages optimize_prompt infer eval_string eval_exec report \
    --dtype bfloat16

echo "Prompt optimization experiment complete."
