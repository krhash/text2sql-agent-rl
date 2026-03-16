# TEXT2SQL-AGENT-RL

Reinforcement learning project comparing LLM improvements for text-to-SQL tasks on the Spider dataset. The baseline uses Llama 3.1 8B to generate SQL from natural language questions. Future work applies GRPO-based RL fine-tuning with LoRA to improve performance across query difficulty levels.

---

## Project Structure

```
TEXT2SQL-AGENT-RL/
    dataset/
        train-00000-of-00001.parquet        # Spider training set (7,000 examples)
        validation-00000-of-00001.parquet   # Spider validation set (1,034 examples)
        spider_schema_rows_v2.json          # DB schemas for all 160 databases
    jobs/
        run_baseline.sh                     # sbatch job script for baseline inference
    models/                                 # LoRA checkpoints saved here during training
    notebooks/
        01_explore_spider.ipynb             # EDA notebook
    scripts/
        baseline_inference.py               # Baseline inference script
    results/                                # Output CSVs from inference/eval
    logs/                                   # sbatch job logs
    README.md
```

---

## Storage Layout on HPC

| Location | What goes here |
|---|---|
| `~/TEXT2SQL-AGENT-RL/` | All project files — persistent, backed up |
| `/scratch/$USER/hf_cache/` | HuggingFace model weights (~16GB for Llama 3.1 8B) — large, not backed up |

> **Important:** Never store large model weights in your home directory. The home quota is limited. The `HF_HOME` env variable in the job scripts redirects model downloads to `/scratch` automatically.

---

## Prerequisites

- Northeastern University account with HPC access
- GlobalProtect VPN client installed ([vpn.northeastern.edu](https://vpn.northeastern.edu))
- HPC account — request at [rc.northeastern.edu](https://rc.northeastern.edu)

---

## 1. Cluster Access

### Connect to VPN
You must be on the Northeastern VPN before connecting to the cluster.
1. Open GlobalProtect
2. Connect to `vpn.northeastern.edu` with your NU credentials

### SSH into Explorer
```bash
ssh your_username@login.explorer.northeastern.edu
```

You will land on the **login node**. Do not run any compute jobs here — it is only for file management and job submission.

---

## 2. Upload Project Files

From your **local machine**, upload the project folder via `scp`:
```bash
scp -r /path/to/TEXT2SQL-AGENT-RL your_username@login.explorer.northeastern.edu:~/TEXT2SQL-AGENT-RL
```

To sync changes after the initial upload (only uploads changed files):
```bash
rsync -avz /path/to/TEXT2SQL-AGENT-RL/ your_username@login.explorer.northeastern.edu:~/TEXT2SQL-AGENT-RL/
```

Verify files landed on the cluster:
```bash
ls ~/TEXT2SQL-AGENT-RL/
```

---

## 3. Environment Setup (One Time Only)

The conda environment only needs to be created once. Do this on an interactive compute node — not the login node.

### Request an interactive GPU node
Run this from the login node:
```bash
srun --partition=gpu-interactive --gres=gpu:a100:1 --mem=64GB --time=01:00:00 --pty /bin/bash
```

Wait for the prompt to change — you will see the hostname change from `login-XX` to a compute node name like `[username@d1026 ~]$`. This confirms you are on a GPU node.

### Create the conda environment
```bash
module load anaconda3/2024.06
conda create -n texttosql python=3.11 -y
source activate texttosql

pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate pandas pyarrow
```

### Exit the compute node when done
```bash
exit
```

---

## 4. What is a Job?

The Explorer cluster is a shared resource used by hundreds of researchers simultaneously. You cannot simply SSH into a GPU machine and run code — instead you request resources through a **job scheduler** called **Slurm**, which manages the queue of all requests and allocates compute nodes fairly.

A **job** is a resource allocation request. You specify what you need — number of GPUs, CPU cores, RAM, and time — and Slurm finds an available node that meets those requirements and runs your work there. When the job finishes or times out, the node is released back to the pool for others.

```
Many users submitting jobs
         │
         ▼
    Slurm Scheduler          ← manages the queue, decides who gets what
         │
         ▼
  Available compute nodes    ← physical machines with GPUs
  [node001] [node002] ...
         │
         ▼
  Your job runs on one node for the time you requested
```

This is why you never run code on the login node — the login node is just the entry point to the cluster, it has no GPUs and is shared by everyone for file management and job submission only.

**Key things Slurm controls:**
- Which physical GPU node your job runs on (you don't choose)
- How much memory and how many CPUs you get
- Maximum wall time — your job is killed when time runs out regardless of whether it finished
- Priority in the queue — longer requested times usually mean longer waits

---

## 5. Running Jobs

There are two ways to run jobs on the cluster.

### Interactive Mode (`srun`)
Use this for development, debugging, and short sanity checks. Your terminal stays connected and you see output live. **Keep your terminal window open — if the SSH session drops, the job is killed.**

```bash
# Step 1 — from login node, request a GPU node
srun --partition=gpu-interactive --gres=gpu:a100:1 --mem=64GB --time=02:00:00 --pty /bin/bash

# Step 2 — activate environment
module load anaconda3/2024.06
source activate texttosql

# Step 3 — navigate to project and run
cd ~/TEXT2SQL-AGENT-RL
python scripts/baseline_inference.py
```

### Batch Mode (`sbatch`)
Use this for full runs and training jobs. The job runs detached in the background — you can close your terminal or disconnect and the job continues running. Output is written to a log file.

```bash
# Submit from the login node
cd ~/TEXT2SQL-AGENT-RL
sbatch jobs/run_baseline.sh

# Returns a job ID, e.g:
# Submitted batch job 1234567
```

Monitor the job:
```bash
# Check job status
squeue --me

# Watch live log output
tail -f ~/TEXT2SQL-AGENT-RL/logs/baseline_1234567.out

# Cancel a job
scancel 1234567
```

### When to use which

| Situation | Use |
|---|---|
| Testing a script works | `srun` interactive |
| Checking GPU memory / output format | `srun` interactive |
| Full dataset inference | `sbatch` |
| LoRA training (hours long) | `sbatch` always |

> **Note:** For interactive `srun` sessions, keep your terminal window open for the duration. If the SSH session drops, the job is killed immediately.

---

## 6. Baseline Inference

The baseline runs Llama 3.1 8B Instruct on Spider validation examples using a schema-aware prompt. Each question is paired with the relevant database schema and the model generates a SQL query. Results are evaluated with exact match accuracy broken down by difficulty level (easy / medium / hard / extra hard).

### Prompt format
Each example is formatted as:
```
You are an expert SQL assistant. Given a database schema and a question,
generate the correct SQL query.
Only output the SQL query, no explanation, no markdown, no code blocks.

### Database Schema
<table : col (type) | table : col (type) ...>

### Primary Keys
<table : col | ...>

### Foreign Keys
<table : col equals table : col | ...>

### Question
<natural language question>

### SQL
```

### Run a 50-example sanity check (interactive)
```bash
# From compute node
cd ~/TEXT2SQL-AGENT-RL
python scripts/baseline_inference.py --n_samples 50
```

### Run full validation set (batch)
```bash
cd ~/TEXT2SQL-AGENT-RL
sbatch jobs/run_baseline.sh
```

### Script arguments
| Argument | Default | Description |
|---|---|---|
| `--model_id` | `meta-llama/Meta-Llama-3.1-8B-Instruct` | HuggingFace model ID |
| `--data_path` | `dataset/validation-00000-of-00001.parquet` | Validation parquet |
| `--schema_path` | `dataset/spider_schema_rows_v2.json` | Schema file |
| `--output_path` | `results/baseline_results.csv` | Where to save results |
| `--n_samples` | `50` | Number of examples to run |
| `--dtype` | `bfloat16` | Model precision |

### Output
Results are saved incrementally to `results/baseline_results.csv` after every example so progress is never lost if a job times out. The CSV contains:

| Column | Description |
|---|---|
| `db_id` | Database name |
| `question` | Natural language question |
| `gold_sql` | Ground truth SQL |
| `pred_sql` | Model generated SQL |
| `difficulty` | easy / medium / hard / extra hard |
| `exact_match` | 1 if correct, 0 if not |

### GPU memory requirements
| dtype | VRAM needed |
|---|---|
| float32 | ~36 GB |
| bfloat16 | ~20 GB |
| float16 | ~20 GB |

An A100 40GB is the minimum recommended GPU for bfloat16 inference.

---

## 7. Useful Commands

```bash
# Check job queue
squeue --me

# Check home directory disk usage
du -sh ~/

# Check scratch usage
du -sh /scratch/$USER/

# Check GPU on compute node
nvidia-smi

# List available GPU types
sinfo -o "%P %G" | grep gpu

# Check conda environments
conda env list
```

---

## 8. Troubleshooting

**`ModuleNotFoundError`** — conda environment not activated:
```bash
module load anaconda3/2024.06
source activate texttosql
```

**Job killed with Signal 9** — out of memory. Increase `--mem` in the job script or srun command.

**SSH session dropped and killed the job** — use `sbatch` for any run longer than a few minutes. Interactive `srun` jobs are tied to your terminal and die if the connection drops.

**Home directory quota exceeded** — model weights are in home instead of scratch. Check:
```bash
du -sh ~/.cache/huggingface
```
If large, move to scratch and set env variable:
```bash
mv ~/.cache/huggingface /scratch/$USER/hf_cache
export HF_HOME=/scratch/$USER/hf_cache
```

**Long wait for GPU node** — request a less specific GPU type, or check availability:
```bash
sinfo -o "%P %G %t %N" | grep gpu
```