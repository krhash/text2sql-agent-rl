# HPC Environment Setup Guide

This guide walks you through exactly how to set up your Conda environment on the Northeastern Explorer HPC for the very first time, and how to activate it on all subsequent logins.

---

## 📁 0. Transfer Project Files to Cluster (Run this only once)
Before setting up the PyTorch environment, you must copy your local `text2sql-agent-rl` code securely to your home directory on the remote HPC cluster. 
Since you are on Windows, you will zip the folder to explicitly exclude massive directories like the `models/` LoRA checkpoints and local `venv/` binaries:

**1. Zip your project via Windows PowerShell:**
```powershell
# Navigate to your project folder locally
tar.exe -a -c -f project.zip --exclude=models --exclude=venv --exclude=__pycache__ --exclude=.git *
```

**2. Transfer via Secure Copy (SCP):**
```powershell
# Upload to your Explorer home directory
scp project.zip your_username@login.explorer.northeastern.edu:~/
```

**3. SSH into the cluster and unzip:**
```bash
ssh your_username@login.explorer.northeastern.edu
mkdir -p TEXT2SQL-AGENT-RL && mv project.zip TEXT2SQL-AGENT-RL/
cd TEXT2SQL-AGENT-RL
unzip project.zip
```

---

## 🛠️ 1. ONE-TIME CONDA SETUP

Do not install Heavy Machine Learning packages on the `login-XX` node! You must request an interactive compute node first.

### 1. Request an Interactive Compute Node
Log into the cluster, then run:
```bash
srun --partition=gpu-interactive --gres=gpu:a100:1 --mem=64GB --time=01:00:00 --pty /bin/bash
```
*(Wait until your prompt changes from `login-XX` to something like `d1026` — this means you are on a node with a GPU!)*

### 2. Load Anaconda & Create Your Environment
Load the global Anaconda module, then ask conda to build a completely fresh Python 3.11 sandbox named `texttosql`:
```bash
module load anaconda3/2024.06
conda create -n texttosql python=3.11 -y
```

### 3. Activate the Environment
```bash
source activate texttosql
```
*(You should now see `(texttosql)` at the very left of your terminal prompt).*

### 4. Install PyTorch & Your Project via `setup.py`
With the environment activated, navigate to your project folder and install the dependencies:
```bash
cd ~/TEXT2SQL-AGENT-RL

# 1. Install PyTorch optimized for CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 2. Use our setup.py to install the text2sql package and all other dependencies (transformers, peft, accelerate, pandas, etc.)
pip install -e ".[gpu]"
```

### 5. Login to HuggingFace
Llama-3.1-8B is gated. Tell your environment who you are:
```bash
huggingface-cli login
```
*(Paste your HuggingFace access token when prompted).*

### 6. Exit the Compute Node
You're done with setup! Give the GPU back to the cluster pool:
```bash
exit
```

---

## 🚀 EVERY OTHER TIME

You never have to run `pip install` or `conda create` again! 
Every time you SSH back into the cluster to run an experiment, just do this:

### 1. Load the Anaconda Module
Even though your environment was created, the cluster still needs to know where `conda` lives:
```bash
module load anaconda3/2024.06
```

### 2. Activate Your Environment
```bash
source activate texttosql
```

### 3. Run the Entire Pipeline
You are now ready. Navigate to your project folder and tell Slurm to run your `.sh` scripts in the background:
```bash
cd ~/TEXT2SQL-AGENT-RL

# 1. Build Ground Truth Caches (Required once)
sbatch jobs/00_preprocess.sh

# 2. Run Baseline Llama 3.1 inference
sbatch jobs/01_baseline.sh

# 3. Run Prompt Optimization Actor-Critique loop
sbatch jobs/02_prompt_opt.sh

# 4. Run GRPO + LoRA RL fine-tuning
sbatch jobs/03_grpo.sh

# 5. Generate final comparison matrix
sbatch jobs/04_compare.sh
```

> **Note:** Whenever you submit the `.sh` files inside `jobs/`, they *already contain* the `module load ...` and `source activate ...` lines. Slurm automatically activates the environment for you in the background.

---

## 🛡️ Long Running Jobs & Crash Recovery

The Explorer cluster imposes strict time limits on jobs (e.g., 24 hours), and GPU memory (OOM) failures can happen. The entire text-to-SQL pipeline is designed to be **fully resumable and crash-safe**.

If your job times out, crashes, or is preempted by the cluster, **you simply submit the exact same `sbatch` command again.**

Here is how the pipeline protects your progress:

1. **Stage Skipping (The Orchestrator):**
   If you submit `--stages train_grpo infer eval_exec` and the job crashes *after* `train_grpo` completes but *during* `infer`, running it again will **skip** the 24-hour GRPO stage instantly and resume `infer`. It does this by checking for output files like `training_done.flag`.

2. **Incremental Inference (`infer` stage):**
   The `LLMGenerator` saves predictions to `results/<run_name>/predictions.csv` incrementally after *every single example*. If you are generating 1,000 queries and the cluster crashes at query 800, restarting it will load the 800 completed queries and only generate the last 200.

3. **Iterative Prompting (`optimize_prompt` stage):**
   The Actor-Critique loop logs every iteration to `opt_history.jsonl` immediately. The best prompt discovered is permanently saved to `best_prompt.json`.

---

## 🗄️ Model Weight & Cache Management

HuggingFace loves silently downloading 16GB models and storing them completely arbitrarily inside of your `~/.cache/huggingface/` home directory.

On Northeastern's HPC cluster, doing this will completely crash your allocation by exceeding your `HOME` disk quota limit! Those giant model blocks should always be written to the scratch storage partition instead.

We solved this natively!
If you open any script inside `jobs/*.sh`, you will notice this exact line:
```bash
export HF_HOME=/scratch/$USER/hf_cache
```

This ensures that whenever you execute scripts via `sbatch`, HuggingFace downloads Llama 3.1 precisely into your high-capacity scratch storage exactly **once**. Your future jobs will read those bytes perfectly with zero repeat downloads, zero environment duplication, and zero quota blowouts!
