"""SFTOptimizer - Supervised Fine-Tuning with LoRA and cross-entropy loss."""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Optional

from text2sql.data.types import Example
from text2sql.inference.base import SQLGenerator
from text2sql.training.base import SQLOptimizer

log = logging.getLogger(__name__)


class SFTOptimizer(SQLOptimizer):
    """
    Supervised Fine-Tuning via LoRA.

    Each step:
      1. Sample a batch of (prompt, true_sql) pairs from train_data.
      2. Concatenate: full_text = prompt + true_sql
      3. Tokenize and build a label tensor where prompt tokens are masked
         to -100 (loss computed only on the SQL completion tokens).
      4. Compute cross-entropy loss on completion tokens.
      5. Backpropagate through LoRA adapter, clip gradients, step AdamW.

    No reward function or rollouts required. Much faster per step than GRPO.

    Checkpoint behaviour (identical pattern to GRPOOptimizer):
      - Periodic snapshots every `checkpoint_every` steps -> checkpoint-<step>/
      - Best val_score weights saved (overwritten in-place) -> checkpoint-best/
      - On resume, scans output_dir for highest checkpoint-<N> and continues
      - sft_done.flag points to checkpoint-best (or final if no val ran)

    Returns a LoRAGenerator pointing at checkpoint-best.
    """

    def __init__(
        self,
        lora_config,
        n_steps         : int   = 1000,
        batch_size      : int   = 8,
        learning_rate   : float = 2e-4,
        checkpoint_every: int   = 500,
        output_dir      : Path  = Path("models/lora/sft"),
        log_path        : Optional[Path] = None,
        val_evaluator   = None,
        val_data        : Optional[list[Example]] = None,
        val_every       : int   = 50,
        max_length      : int   = 1024,
    ):
        self.lora_config      = lora_config
        self.n_steps          = n_steps
        self.batch_size       = batch_size
        self.learning_rate    = learning_rate
        self.checkpoint_every = checkpoint_every
        self.output_dir       = Path(output_dir)
        self.log_path         = log_path
        self.val_evaluator    = val_evaluator
        self.val_data         = val_data or []
        self.val_every        = val_every
        self.max_length       = max_length

    # -- Resume helper ---------------------------------------------------------

    @staticmethod
    def _find_latest_checkpoint(output_dir: Path) -> tuple[Optional[Path], int]:
        """
        Scan output_dir for checkpoint-<N> directories (numeric step only).
        Returns (path, step) of the highest N, or (None, 0) if none found.
        """
        best_step = 0
        best_path = None
        for p in output_dir.glob("checkpoint-*"):
            if not p.is_dir():
                continue
            stem = p.name.split("-")[-1]
            if stem.isdigit():
                s = int(stem)
                if s > best_step:
                    best_step = s
                    best_path = p
        return best_path, best_step

    # -- Label masking helper --------------------------------------------------

    @staticmethod
    def _build_labels(input_ids, prompt_len: int):
        """
        Return a label tensor with -100 for all prompt tokens.
        Loss is computed only on the completion (SQL) tokens.
        """
        import torch
        labels = input_ids.clone()
        labels[0, :prompt_len] = -100
        return labels

    def optimize(
        self,
        generator  : SQLGenerator,
        train_data : list[Example],
        val_data   : list[Example],
        output_dir : Path,
    ) -> SQLGenerator:
        """
        Run the SFT training loop.
        Returns a LoRAGenerator pointing at checkpoint-best.
        """
        try:
            import torch
            from peft import get_peft_model, PeftModel
            from torch.optim import AdamW
        except ImportError:
            raise ImportError(
                "torch and peft are required for SFT. "
                "Install with: pip install torch peft"
            )

        from text2sql.inference.generator import LLMGenerator, LoRAGenerator

        if not isinstance(generator, LLMGenerator):
            raise TypeError("SFTOptimizer requires an LLMGenerator.")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        engine         = generator.engine
        prompt_builder = generator.prompt_builder
        tokenizer      = engine.tokenizer
        device         = next(engine.model.parameters()).device

        # -- Resume or fresh start ---------------------------------------------
        resume_ckpt, resume_step = self._find_latest_checkpoint(self.output_dir)

        if resume_ckpt:
            log.info(
                f"Resuming SFT from checkpoint: {resume_ckpt} "
                f"(step {resume_step}/{self.n_steps})"
            )
            peft_model = PeftModel.from_pretrained(engine.model, str(resume_ckpt))
            peft_model.train()
        else:
            log.info("Attaching LoRA adapter (fresh start) ...")
            peft_model = get_peft_model(engine.model, self.lora_config)
            peft_model.print_trainable_parameters()
            peft_model.train()

        optimizer = AdamW(
            filter(lambda p: p.requires_grad, peft_model.parameters()),
            lr=self.learning_rate,
        )

        # Best checkpoint tracking
        best_val_score = 0.0
        best_ckpt_dir  = self.output_dir / "checkpoint-best"

        log_path = self.log_path or output_dir / "sft_training_log.jsonl"

        log.info(
            f"Starting SFT training: {self.n_steps} steps, "
            f"batch_size={self.batch_size}, lr={self.learning_rate}"
        )

        for step in range(resume_step + 1, self.n_steps + 1):
            batch = random.sample(train_data, min(self.batch_size, len(train_data)))

            peft_model.train()
            optimizer.zero_grad()

            step_loss = torch.tensor(0.0, device=device)

            for example in batch:
                # Build full text: prompt + true SQL completion
                prompt     = prompt_builder.build(example.question, example.db_id)
                completion = example.true_sql

                # Tokenize prompt alone to get prompt length for masking
                prompt_ids = tokenizer(
                    prompt, return_tensors="pt", add_special_tokens=True
                )["input_ids"]
                prompt_len = prompt_ids.shape[1]

                # Tokenize concatenated text within max_length
                full_text = prompt + completion
                inputs = tokenizer(
                    full_text, return_tensors="pt",
                    truncation=True, max_length=self.max_length,
                )
                input_ids = inputs["input_ids"].to(device)
                attn_mask = inputs["attention_mask"].to(device)

                # Mask prompt tokens from loss
                labels = self._build_labels(input_ids, prompt_len).to(device)

                outputs = peft_model(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    labels=labels,
                )
                step_loss = step_loss + outputs.loss

            step_loss = step_loss / len(batch)
            step_loss.backward()
            torch.nn.utils.clip_grad_norm_(peft_model.parameters(), 1.0)
            optimizer.step()

            # -- LOGGING -------------------------------------------------------
            if step % 10 == 0:
                log.info(f"  step={step:>4}  loss={step_loss.item():.4f}")

            # -- VALIDATION MONITOR + BEST CHECKPOINT --------------------------
            val_score = None
            val_data_to_use = val_data or self.val_data
            if self.val_evaluator and val_data_to_use and step % self.val_every == 0:
                peft_model.eval()
                engine.model = peft_model
                val_sample   = random.sample(val_data_to_use, min(50, len(val_data_to_use)))
                monitor_gen  = LLMGenerator(generator.name, engine, prompt_builder)
                preds        = monitor_gen.generate_batch(val_sample, progress=False)
                val_score    = self.val_evaluator.score(preds) * 100
                engine.model = engine.model  # restore (peft_model IS engine.model here)
                peft_model.train()
                log.info(f"  [val] step={step}  string_match={val_score:.1f}%")

                if val_score > best_val_score:
                    best_val_score = val_score
                    peft_model.save_pretrained(str(best_ckpt_dir))
                    log.info(f"  [best] New best {val_score:.1f}% -> {best_ckpt_dir}")

            entry = {
                "step": step, "loss": step_loss.item(), "val_score": val_score,
            }
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")

            # -- PERIODIC CHECKPOINT (for resume) ------------------------------
            if step % self.checkpoint_every == 0:
                ckpt = self.output_dir / f"checkpoint-{step}"
                peft_model.save_pretrained(str(ckpt))
                log.info(f"  Saved checkpoint -> {ckpt}")

        # -- FINAL CHECKPOINT --------------------------------------------------
        final_ckpt = self.output_dir / "checkpoint-final"
        peft_model.save_pretrained(str(final_ckpt))
        log.info(f"SFT training complete. Final checkpoint -> {final_ckpt}")

        infer_ckpt = best_ckpt_dir if best_ckpt_dir.exists() else final_ckpt
        log.info(f"Inference checkpoint -> {infer_ckpt}  (best val: {best_val_score:.1f}%)")

        return LoRAGenerator(
            name            = generator.name + "_sft",
            base_engine     = generator.engine,
            lora_checkpoint = infer_ckpt,
            prompt_builder  = prompt_builder,
        )
