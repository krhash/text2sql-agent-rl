"""scripts/pipeline.py — thin CLI wrapper around ExperimentRunner."""
import argparse
import sys
from pathlib import Path

# Allow running from project root without pip install
sys.path.insert(0, str(Path(__file__).parent.parent))

from text2sql.pipeline.config import RunConfig
from text2sql.pipeline.runner import ExperimentRunner

ALL_STAGES = [
    "preprocess", "optimize_prompt", "train_sft", "train_grpo",
    "infer", "eval_string", "eval_exec", "report",
]


def parse_args():
    p = argparse.ArgumentParser(
        description="text2sql unified pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run",          required=True,
                   help="Name of this run (output folder: results/<run>).")
    p.add_argument("--stages",       nargs="+", default=ALL_STAGES,
                   choices=ALL_STAGES,
                   help="Stages to execute.")
    p.add_argument("--force",        action="store_true",
                   help="Re-run stage even if output already exists.")
    p.add_argument("--results_dir",  default="results")
    p.add_argument("--cache_run",    default=None,
                   help="Borrow true_sql cache from this named run.")

    # Data
    p.add_argument("--train_path",   default="dataset/train-00000-of-00001.parquet")
    p.add_argument("--val_path",     default="dataset/validation-00000-of-00001.parquet")
    p.add_argument("--schema_path",  default="dataset/spider_schema_rows_v2.json")
    p.add_argument("--db_root",      default="dataset/database")
    p.add_argument("--preprocess_split", default="both", choices=["train", "val", "both"])

    # Inference
    p.add_argument("--model_id",      default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--model_path",    default=None)
    p.add_argument("--cache_dir",     default="/scratch/$USER/hf_cache")
    p.add_argument("--n_samples",     type=int, default=None,
                   help="Number of val examples to run inference on (None = all).")
    p.add_argument("--dtype",         default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--inference_from", default=None,
                   help="Reuse predictions.csv from this named run.")

    # Prompt opt
    p.add_argument("--n_opt_iterations", type=int, default=5)
    p.add_argument("--opt_sample_size",  type=int, default=100)

    # GRPO
    p.add_argument("--reward_fn",        default="composite", choices=["binary", "composite"])
    p.add_argument("--group_size",       type=int, default=4)
    p.add_argument("--n_steps",          type=int, default=1000)
    p.add_argument("--kl_coef",          type=float, default=0.1)
    p.add_argument("--lora_r",           type=int, default=16)
    p.add_argument("--lora_alpha",       type=int, default=32)
    p.add_argument("--batch_size",       type=int, default=8)
    p.add_argument("--learning_rate",    type=float, default=1e-4)
    p.add_argument("--checkpoint_every", type=int, default=500,
                   help="Save a periodic LoRA checkpoint every N steps (GRPO + SFT).")
    p.add_argument("--grpo_resume_from", default=None,
                   help="Explicit LoRA checkpoint path to seed GRPO resume from. "
                        "If omitted, auto-scans output_dir for checkpoint-<N> dirs.")

    # SFT
    p.add_argument("--sft_n_steps",      type=int, default=1000,
                   help="Number of training steps for SFT (default same as n_steps).")
    p.add_argument("--sft_resume_from",  default=None,
                   help="Explicit LoRA checkpoint path to seed SFT resume from. "
                        "If omitted, auto-scans output_dir for checkpoint-<N> dirs.")

    # Inference adapter selection
    p.add_argument("--infer_model",      default="auto",
                   help=("Which adapter to mount for the infer stage. "
                         "Options: auto | none | grpo | sft | <path-to-lora-checkpoint>"))

    return p.parse_args()


def main():
    args   = parse_args()
    config = RunConfig(
        run_name         = args.run,
        results_dir      = args.results_dir,
        cache_run        = args.cache_run,
        train_path       = args.train_path,
        val_path         = args.val_path,
        schema_path      = args.schema_path,
        db_root          = args.db_root,
        preprocess_split = args.preprocess_split,
        model_id         = args.model_id,
        model_path       = args.model_path,
        cache_dir        = args.cache_dir,
        n_samples        = args.n_samples,
        dtype            = args.dtype,
        inference_from   = args.inference_from,
        infer_model      = args.infer_model,
        n_opt_iterations = args.n_opt_iterations,
        opt_sample_size  = args.opt_sample_size,
        reward_fn        = args.reward_fn,
        group_size       = args.group_size,
        n_steps          = args.n_steps,
        kl_coef          = args.kl_coef,
        lora_r           = args.lora_r,
        lora_alpha       = args.lora_alpha,
        batch_size       = args.batch_size,
        learning_rate    = args.learning_rate,
        checkpoint_every = args.checkpoint_every,
        grpo_resume_from = args.grpo_resume_from,
        sft_n_steps      = args.sft_n_steps,
        sft_resume_from  = args.sft_resume_from,
    )
    # Save config for reproducibility
    config.run_dir().mkdir(parents=True, exist_ok=True)
    config.save(config.run_dir() / "run_config.json")

    runner = ExperimentRunner(config)
    runner.run(stages=args.stages, force=args.force)


if __name__ == "__main__":
    main()
