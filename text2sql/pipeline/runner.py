"""ExperimentRunner — the orchestrating class for the full pipeline."""
from __future__ import annotations

import logging
import shutil
import sys
from functools import cached_property
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from text2sql.pipeline.config import RunConfig


def _setup_logging(log_path: Path, run_name: str) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"pipeline.{run_name}")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger  # already set up (e.g. in tests)

    fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s")
    fh  = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    sh  = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


class ExperimentRunner:
    """
    Orchestrates one named experiment run.

    Owns:
      - run directory (results/<run_name>/)
      - pipeline.log
      - skip logic   — don't re-run a stage whose output already exists
      - stage wiring — output of stage N is resolved as input to stage N+1

    One runner per run. Compare runs via ComparisonTable or scripts/compare.py.
    """

    STAGE_ORDER = [
        "preprocess",
        "optimize_prompt",
        "train_sft",
        "train_grpo",
        "infer",
        "eval_string",
        "eval_exec",
        "report",
    ]

    def __init__(self, config: RunConfig):
        self.config  = config
        self.run_dir = config.run_dir()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log     = _setup_logging(self.run_dir / "pipeline.log", config.run_name)

    # ── Primary entry point ────────────────────────────────────────────────────

    def run(self, stages: Optional[list[str]] = None, force: bool = False):
        """
        Execute listed stages in order.
        If stages is None, runs all stages in STAGE_ORDER.
        Skips any stage whose primary output already exists unless force=True.
        """
        stages = stages or self.STAGE_ORDER
        self.log.info(f"Pipeline started — run: {self.config.run_name}")
        self.log.info(f"Stages requested : {stages}")
        self.log.info(f"Output directory : {self.run_dir}")
        for stage in stages:
            if stage not in self._handlers:
                raise ValueError(f"Unknown stage: '{stage}'. "
                                 f"Valid: {list(self._handlers.keys())}")
            self._run_stage(stage, force)
        self.log.info("\nPipeline complete.")
        self.log.info(f"All outputs in: {self.run_dir}")

    # ── Skip logic ────────────────────────────────────────────────────────────

    def _stage_outputs(self) -> dict[str, Path]:
        d = self.run_dir
        return {
            "preprocess"     : d / "true_sql_cache_validation.json",
            "optimize_prompt": d / "best_prompt.json",
            "train_sft"      : d / "sft_done.flag",
            "train_grpo"     : d / "training_done.flag",
            "infer"          : d / "predictions.csv",
            "eval_string"    : d / "eval_string.csv",
            "eval_exec"      : d / "eval_exec.csv",
            "report"         : d / "report.txt",
        }

    def _run_stage(self, stage: str, force: bool):
        output = self._stage_outputs()[stage]
        if output.exists() and not force:
            # For infer: check row count vs expected val-set size.
            # A partial predictions.csv must NOT be skipped — inference
            # must resume from the last completed row.
            if stage == "infer":
                import csv
                c = self.config
                try:
                    from text2sql.data.dataset import SpiderDataset
                    expected = len(SpiderDataset(c.val_path).load(n=c.n_samples))
                    with open(output, newline="") as f:
                        completed = max(0, sum(1 for _ in csv.reader(f)) - 1)  # subtract header
                    if completed < expected:
                        self.log.info(
                            f"[RESUME] infer — predictions.csv is incomplete "
                            f"({completed}/{expected}). Continuing from row {completed + 1}."
                        )
                        # fall through to run the stage
                    else:
                        self.log.info(f"[SKIP] infer — predictions complete ({completed}/{expected})")
                        return
                except Exception:
                    self.log.info(f"[SKIP] {stage} — output exists: {output}")
                    return
            else:
                self.log.info(f"[SKIP] {stage} — output exists: {output}")
                return
        self.log.info(f"\n{'='*60}\n  STAGE: {stage.upper()}\n{'='*60}")
        self._handlers[stage]()
        self.log.info(f"[DONE] {stage}")


    # ── Stage: preprocess ─────────────────────────────────────────────────────

    def preprocess(self):
        from text2sql.data.cache import TrueSQLCacheBuilder
        c       = self.config
        builder = TrueSQLCacheBuilder(db_root=c.db_root)
        cache_d = c.cache_dir_path()
        cache_d.mkdir(parents=True, exist_ok=True)

        if c.preprocess_split in ("train", "both"):
            out = cache_d / "true_sql_cache_train.json"
            self.log.info(f"Building train cache from: {c.train_path}")
            builder.build(c.train_path, str(out))
            self.log.info(f"  train cache → {out}")

        if c.preprocess_split in ("val", "both"):
            out = cache_d / "true_sql_cache_validation.json"
            self.log.info(f"Building val cache from: {c.val_path}")
            builder.build(c.val_path, str(out))
            self.log.info(f"  val cache → {out}")

    # ── Stage: optimize_prompt ────────────────────────────────────────────────

    def optimize_prompt(self):
        from text2sql.data.dataset import SpiderDataset
        from text2sql.data.schema import PromptBuilder
        from text2sql.eval.string_match import StringMatchEvaluator
        from text2sql.inference.engine import InferenceEngine
        from text2sql.inference.generator import LLMGenerator
        from text2sql.training.prompt_opt import PromptOptimizer
        c = self.config

        self.log.info("Loading data for prompt optimization …")
        train_data = SpiderDataset(c.train_path).load()
        val_data   = SpiderDataset(c.val_path).load(n=200)

        engine  = InferenceEngine(c.model_id, c.dtype,
                                  cache_dir=c.cache_dir, model_path=c.model_path)
        prompt  = PromptBuilder(c.schema_path)
        gen     = LLMGenerator(c.run_name, engine, prompt)

        PromptOptimizer(
            evaluator    = StringMatchEvaluator(),
            n_iterations = c.n_opt_iterations,
            sample_size  = c.opt_sample_size,
        ).optimize(gen, train_data, val_data, self.run_dir)

    # ── Stage: train_grpo ─────────────────────────────────────────────────────

    def train_grpo(self):
        from text2sql.data.dataset import SpiderDataset
        from text2sql.data.schema import PromptBuilder
        from text2sql.eval.string_match import StringMatchEvaluator
        from text2sql.inference.engine import InferenceEngine
        from text2sql.inference.generator import LLMGenerator
        from text2sql.reward.binary import BinaryReward
        from text2sql.reward.composite import CompositeReward
        from text2sql.training.grpo import GRPOOptimizer
        from text2sql.training.lora import default_lora_config
        c = self.config

        train_data = SpiderDataset(c.train_path).load()
        val_data   = SpiderDataset(c.val_path).load(n=100)
        engine     = InferenceEngine(c.model_id, c.dtype,
                                     cache_dir=c.cache_dir, model_path=c.model_path)

        # Use prompt-optimised prompt if available
        prompt_path = self.run_dir / "best_prompt.json"
        prompt = PromptBuilder.from_file(prompt_path) \
                 if prompt_path.exists() else PromptBuilder(c.schema_path)

        gen = LLMGenerator(c.run_name, engine, prompt)

        # Reward uses TRAIN cache
        train_cache = c.cache_dir_path() / "true_sql_cache_train.json"
        cache_path  = str(train_cache) if train_cache.exists() else None
        if cache_path is None:
            self.log.warning("No train cache found — falling back to live execution (slow!).")

        RewardCls = CompositeReward if c.reward_fn == "composite" else BinaryReward
        reward_fn = RewardCls(db_root=c.db_root, true_sql_cache_path=cache_path)

        # Anchor lora_output to an absolute path so it is stable across sbatch submissions
        lora_output = (Path(c.results_dir) / ".." / "models" / "lora" / c.run_name).resolve()

        # Allow explicit resume from a specific checkpoint
        if c.grpo_resume_from:
            resume_path = Path(c.grpo_resume_from)
            if not resume_path.exists():
                raise FileNotFoundError(
                    f"--grpo_resume_from path not found: {resume_path}"
                )
            # Copy the named checkpoint into output_dir as checkpoint-0 sentinel
            # so the scanner picks it up and restores weights, then continues.
            import shutil
            sentinel = lora_output / "checkpoint-0"
            if not sentinel.exists():
                self.log.info(f"Seeding resume from: {resume_path} -> {sentinel}")
                shutil.copytree(str(resume_path), str(sentinel))

        grpo_gen = GRPOOptimizer(
            reward_fn        = reward_fn,
            lora_config      = default_lora_config(r=c.lora_r, lora_alpha=c.lora_alpha),
            group_size       = c.group_size,
            n_steps          = c.n_steps,
            kl_coef          = c.kl_coef,
            batch_size       = c.batch_size,
            learning_rate    = c.learning_rate,
            checkpoint_every = c.checkpoint_every,
            output_dir       = lora_output,
            log_path         = self.run_dir / "training_log.jsonl",
            val_evaluator    = StringMatchEvaluator(),
            val_data         = val_data,
        ).optimize(gen, train_data, val_data, self.run_dir)

        # Write flag pointing to best checkpoint so infer knows where to load
        checkpoint = grpo_gen.lora_checkpoint
        (self.run_dir / "training_done.flag").write_text(str(checkpoint))
        self.log.info(f"Training complete. Checkpoint: {checkpoint}")

    # ── Stage: train_sft ────────────────────────────────────────────────────

    def train_sft(self):
        from text2sql.data.dataset import SpiderDataset
        from text2sql.data.schema import PromptBuilder
        from text2sql.eval.string_match import StringMatchEvaluator
        from text2sql.inference.engine import InferenceEngine
        from text2sql.inference.generator import LLMGenerator
        from text2sql.training.lora import default_lora_config
        from text2sql.training.sft import SFTOptimizer
        c = self.config

        train_data = SpiderDataset(c.train_path).load()
        val_data   = SpiderDataset(c.val_path).load(n=100)
        engine     = InferenceEngine(c.model_id, c.dtype,
                                     cache_dir=c.cache_dir, model_path=c.model_path)

        # Use prompt-optimised prompt if available
        prompt_path = self.run_dir / "best_prompt.json"
        prompt = PromptBuilder.from_file(prompt_path) \
                 if prompt_path.exists() else PromptBuilder(c.schema_path)

        gen = LLMGenerator(c.run_name, engine, prompt)

        # Anchor lora_output to an absolute path so it is stable across sbatch submissions
        lora_output = (Path(c.results_dir) / ".." / "models" / "lora" / c.run_name / "sft").resolve()

        # Allow explicit resume from a specific checkpoint
        if c.sft_resume_from:
            resume_path = Path(c.sft_resume_from)
            if not resume_path.exists():
                raise FileNotFoundError(
                    f"--sft_resume_from path not found: {resume_path}"
                )
            import shutil
            sentinel = lora_output / "checkpoint-0"
            if not sentinel.exists():
                self.log.info(f"Seeding SFT resume from: {resume_path} -> {sentinel}")
                shutil.copytree(str(resume_path), str(sentinel))

        sft_gen = SFTOptimizer(
            lora_config      = default_lora_config(r=c.lora_r, lora_alpha=c.lora_alpha),
            n_steps          = c.sft_n_steps,
            batch_size       = c.batch_size,
            learning_rate    = c.learning_rate,
            checkpoint_every = c.checkpoint_every,
            output_dir       = lora_output,
            log_path         = self.run_dir / "sft_training_log.jsonl",
            val_evaluator    = StringMatchEvaluator(),
            val_data         = val_data,
        ).optimize(gen, train_data, val_data, self.run_dir)

        checkpoint = sft_gen.lora_checkpoint
        (self.run_dir / "sft_done.flag").write_text(str(checkpoint))
        self.log.info(f"SFT complete. Checkpoint: {checkpoint}")

    # ── Stage: infer ──────────────────────────────────────────────────────────

    def _resolve_generator(self, engine, prompt) -> "SQLGenerator":
        """
        Determine which generator to use for the infer stage.

        --infer_model controls this explicitly:
          auto   : auto-detect from flag files (backward compat)
          none   : plain frozen Llama, no adapter
          grpo   : LoRA from training_done.flag
          sft    : LoRA from sft_done.flag
          <path> : LoRA from the given filesystem path directly
        """
        from text2sql.inference.generator import LLMGenerator, LoRAGenerator
        m = self.config.infer_model

        if m == "none":
            self.log.info("infer_model=none: using plain frozen Llama (no adapter)")
            return LLMGenerator(self.config.run_name, engine, prompt)

        elif m == "grpo":
            flag = self.run_dir / "training_done.flag"
            if not flag.exists():
                raise FileNotFoundError(
                    "--infer_model grpo: training_done.flag not found. "
                    "Run train_grpo first."
                )
            ckpt = Path(flag.read_text().strip())
            self.log.info(f"infer_model=grpo: loading GRPO checkpoint {ckpt}")
            return LoRAGenerator(self.config.run_name, engine, ckpt, prompt)

        elif m == "sft":
            flag = self.run_dir / "sft_done.flag"
            if not flag.exists():
                raise FileNotFoundError(
                    "--infer_model sft: sft_done.flag not found. "
                    "Run train_sft first."
                )
            ckpt = Path(flag.read_text().strip())
            self.log.info(f"infer_model=sft: loading SFT checkpoint {ckpt}")
            return LoRAGenerator(self.config.run_name, engine, ckpt, prompt)

        elif m == "auto":
            # Backward-compatible auto detection: GRPO > SFT > plain
            grpo_flag = self.run_dir / "training_done.flag"
            sft_flag  = self.run_dir / "sft_done.flag"
            if grpo_flag.exists():
                ckpt = Path(grpo_flag.read_text().strip())
                self.log.info(f"infer_model=auto: detected GRPO checkpoint {ckpt}")
                return LoRAGenerator(self.config.run_name, engine, ckpt, prompt)
            elif sft_flag.exists():
                ckpt = Path(sft_flag.read_text().strip())
                self.log.info(f"infer_model=auto: detected SFT checkpoint {ckpt}")
                return LoRAGenerator(self.config.run_name, engine, ckpt, prompt)
            else:
                self.log.info("infer_model=auto: no adapter found, using plain Llama")
                return LLMGenerator(self.config.run_name, engine, prompt)

        else:
            # Treat as a direct filesystem path
            ckpt = Path(m)
            if not ckpt.exists():
                raise FileNotFoundError(f"--infer_model path not found: {ckpt}")
            self.log.info(f"infer_model=<path>: loading checkpoint from {ckpt}")
            return LoRAGenerator(self.config.run_name, engine, ckpt, prompt)

    def infer(self):
        c = self.config

        # Handle --inference_from shortcut
        if c.inference_from:
            self._copy_predictions_from(c.inference_from)
            return

        from text2sql.data.dataset import SpiderDataset
        from text2sql.data.schema import PromptBuilder
        from text2sql.inference.engine import InferenceEngine
        from text2sql.pipeline.io import save_predictions

        val_data = SpiderDataset(c.val_path).load(n=c.n_samples)
        engine   = InferenceEngine(c.model_id, c.dtype,
                                   cache_dir=c.cache_dir, model_path=c.model_path)

        # Auto-load best_prompt if optimise_prompt ran
        prompt_path = self.run_dir / "best_prompt.json"
        prompt = PromptBuilder.from_file(prompt_path) \
                 if prompt_path.exists() else PromptBuilder(c.schema_path)

        # Resolve which model/adapter to use for inference
        generator = self._resolve_generator(engine, prompt)

        output_path = self.run_dir / "predictions.csv"
        self.log.info(f"Running inference on {len(val_data)} examples …")
        predictions = generator.generate_batch(val_data, output_path=output_path)
        save_predictions(predictions, output_path)

        from text2sql.inference.sql_utils import SQLUtils
        correct = sum(SQLUtils.exact_match(p.pred_sql, p.true_sql) for p in predictions)
        self.log.info(f"Inference complete: {correct}/{len(predictions)} "
                      f"= {correct/len(predictions)*100:.1f}% exact match")

    # ── Stage: eval_string ────────────────────────────────────────────────────

    def eval_string(self):
        from text2sql.eval.string_match import StringMatchEvaluator
        from text2sql.pipeline.io import load_predictions
        preds = load_predictions(self.run_dir / "predictions.csv")
        df    = StringMatchEvaluator().evaluate(preds)
        df.to_csv(self.run_dir / "eval_string.csv", index=False)
        self.log.info(f"  String match: {df['score'].mean()*100:.1f}%")

    # ── Stage: eval_exec ──────────────────────────────────────────────────────

    def eval_exec(self):
        from text2sql.eval.execution import ExecutionEvaluator
        from text2sql.pipeline.io import load_predictions
        c = self.config

        preds     = load_predictions(self.run_dir / "predictions.csv")
        val_cache = c.cache_dir_path() / "true_sql_cache_validation.json"
        cache_path = str(val_cache) if val_cache.exists() else None
        if cache_path is None:
            self.log.warning("No val cache found — falling back to live execution.")

        df = ExecutionEvaluator(db_root=c.db_root,
                               true_sql_cache_path=cache_path).evaluate(preds)
        df.to_csv(self.run_dir / "eval_exec.csv", index=False)
        self.log.info(f"  Exec accuracy: {df['score'].mean()*100:.1f}%")

    # ── Stage: report ─────────────────────────────────────────────────────────

    def report(self):
        from text2sql.eval.report import ReportGenerator
        c = self.config

        s_path = self.run_dir / "eval_string.csv"
        e_path = self.run_dir / "eval_exec.csv"
        string_df = pd.read_csv(s_path) if s_path.exists() else None
        exec_df   = pd.read_csv(e_path) if e_path.exists() else None

        if string_df is None and exec_df is None:
            raise RuntimeError("No eval results found. Run eval_string and/or eval_exec first.")

        n_samples = len(string_df if string_df is not None else exec_df)
        report_str = ReportGenerator().generate(
            string_df = string_df,
            exec_df   = exec_df,
            run_name  = c.run_name,
            model_id  = c.model_id,
            n_samples = n_samples,
        )
        (self.run_dir / "report.txt").write_text(report_str, encoding="utf-8")
        print("\n" + report_str + "\n")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _copy_predictions_from(self, source_run: str):
        src = Path(self.config.results_dir) / source_run / "predictions.csv"
        if not src.exists():
            raise FileNotFoundError(
                f"--inference_from: predictions.csv not found in run '{source_run}'"
            )
        dst = self.run_dir / "predictions.csv"
        shutil.copy2(src, dst)
        self.log.info(f"Copied predictions from {source_run}")

    @cached_property
    def _handlers(self) -> dict[str, Callable]:
        return {
            "preprocess"     : self.preprocess,
            "optimize_prompt": self.optimize_prompt,
            "train_sft"      : self.train_sft,
            "train_grpo"     : self.train_grpo,
            "infer"          : self.infer,
            "eval_string"    : self.eval_string,
            "eval_exec"      : self.eval_exec,
            "report"         : self.report,
        }
