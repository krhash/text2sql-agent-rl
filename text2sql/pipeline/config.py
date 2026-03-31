"""RunConfig — all parameters for one named experiment run."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class RunConfig:
    # ── Identity ────────────────────────────────────────────────────────
    run_name    : str = "run"
    results_dir : str = "results"

    # ── Data ────────────────────────────────────────────────────────────
    train_path  : str = "dataset/train-00000-of-00001.parquet"
    val_path    : str = "dataset/validation-00000-of-00001.parquet"
    schema_path : str = "dataset/spider_schema_rows_v2.json"
    db_root     : str = "dataset/database"

    # ── Cache ────────────────────────────────────────────────────────────
    preprocess_split : str           = "both"      # train | val | both
    cache_run        : Optional[str] = None        # borrow cache from named run

    # ── Inference ────────────────────────────────────────────────────────
    model_id       : str           = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    model_path     : Optional[str] = None
    cache_dir      : str           = "/scratch/$USER/hf_cache"
    n_samples      : Optional[int] = None          # None = full val set
    dtype          : str           = "bfloat16"
    inference_from : Optional[str] = None          # copy predictions from named run

    # ── Prompt optimization ──────────────────────────────────────────────
    n_opt_iterations : int = 5
    opt_sample_size  : int = 100

    # ── GRPO training ────────────────────────────────────────────────────
    reward_fn  : str   = "composite"               # binary | composite
    group_size : int   = 4
    n_steps    : int   = 1000
    kl_coef    : float = 0.1
    lora_r     : int   = 16
    lora_alpha : int   = 32
    batch_size : int   = 8
    learning_rate : float = 1e-4

    def run_dir(self) -> Path:
        return Path(self.results_dir) / self.run_name

    def cache_dir_path(self) -> Path:
        """Cache lives in the cache_run folder or this run's folder."""
        if self.cache_run:
            return Path(self.results_dir) / self.cache_run
        return self.run_dir()

    def save(self, path: Path | str):
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def from_file(cls, path: Path | str) -> "RunConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)
