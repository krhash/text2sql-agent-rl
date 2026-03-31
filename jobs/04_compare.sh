#!/bin/bash
#SBATCH --job-name=eval_compare
#SBATCH --output=results/compare/slurm-%j.out
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00

# Activate your environment
# source venv/bin/activate  # Using venv
# conda activate text2sql   # Using conda

echo "Generating comparison report for all three models..."

# Creates a comparison report between the three approaches (assuming they all completed successfully)
python scripts/compare.py \
    --runs baseline_v1 prompt_opt_v1 grpo_v1 \
    --output results/final_comparison_matrix.txt

echo "Comparison complete! Check results/final_comparison_matrix.txt"
