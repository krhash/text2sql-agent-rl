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
srun --partition=sharing --gres=gpu:v100-sxm2:1 --mem=48GB --time=00:59:00 --pty /bin/bash
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

## Long Running Jobs & Crash Recovery

The Explorer cluster imposes strict time limits and GPU OOM failures can occur. The pipeline is designed to be fully resumable.

If your job times out or crashes, resubmit the same `sbatch` command — no manual cleanup needed.

**Stage skipping:** If `train_grpo` completed but `infer` crashed, resubmitting skips the GRPO stage entirely and resumes at `infer`. The runner checks for the existence of each stage's output file before executing.

**Inference resuming:** The `infer` stage saves `predictions.csv` after every individual example. A crash at example 800/1000 resumes from example 801 on restart.

**LoRA checkpoint resume:** Both `train_grpo` and `train_sft` save periodic snapshots every N steps (default: every 200 steps). On restart, the trainer automatically scans the checkpoint directory for the highest-numbered `checkpoint-<N>` and continues from the next step.

```
models/lora/<run_name>/
  checkpoint-200/      <- periodic snapshot (for resume)
  checkpoint-400/      <- periodic snapshot (for resume)
  checkpoint-best/     <- best validation score (used by infer)
  checkpoint-final/    <- end-of-training weights
```

To resume from a specific checkpoint explicitly:
```bash
# GRPO
sbatch --export=ALL,GRPO_RESUME_FROM=models/lora/grpo_v1/checkpoint-400 jobs/03_grpo.sh

# SFT
sbatch --export=ALL,SFT_RESUME_FROM=models/lora/sft_v1/sft/checkpoint-400 jobs/04_sft.sh
```

---

## Model Weight & Cache Management

HuggingFace defaults to downloading model weights into `~/.cache/huggingface/`, which quickly exhausts the home directory quota on Northeastern's cluster.

All `jobs/*.sh` scripts include:
```bash
export HF_HOME=/scratch/$USER/hf_cache
```

This redirects all downloads to `/scratch`, which has a much larger quota. The model is downloaded once and reused across all subsequent jobs.

