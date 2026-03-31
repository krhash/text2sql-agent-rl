#!/bin/bash
#SBATCH --job-name=preprocess_spider
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=00:30:00
#SBATCH --output=/home/%u/TEXT2SQL-AGENT-RL/results/preprocess/slurm_%j.out
#SBATCH --error=/home/%u/TEXT2SQL-AGENT-RL/results/preprocess/slurm_%j.err

PROJECT=$HOME/TEXT2SQL-AGENT-RL
mkdir -p $PROJECT/results/preprocess/

module load anaconda3/2024.06
source activate texttosql

echo "── Run ───────────────────────────────────────────────────────────────────────"
echo "Job started : $(date)"
echo "Node        : $(hostname)"
echo "Task        : Preprocessing Ground Truth SQLite caches"

python $PROJECT/scripts/pipeline.py \
    --run preprocess \
    --stages preprocess \
    --preprocess_split both

echo "Preprocessing complete."
