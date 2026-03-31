"""GRPOOptimizer — RL fine-tuning with Group Relative Policy Optimization."""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Optional

from text2sql.data.types import Example
from text2sql.inference.base import SQLGenerator
from text2sql.training.base import SQLOptimizer
from text2sql.training.rollout import sample_rollout, group_advantages

log = logging.getLogger(__name__)


class GRPOOptimizer(SQLOptimizer):
    """
    LoRA fine-tuning using Group Relative Policy Optimization.

    Each training step:
      1. Sample K questions from train_data
      2. Generate G completions per question (GPU — temperature > 0)
      3. Compute reward for each completion (CPU — uses true_sql cache)
      4. Compute group-relative advantages
      5. Update LoRA weights via policy gradient + KL penalty

    The true_sql cache means ground truth is NEVER re-executed live.
    Only predicted SQL is executed (~1ms per query via SQLite).

    Returns a LoRAGenerator pointing at the final checkpoint.
    """

    def __init__(
        self,
        reward_fn,                          # BaseReward instance
        lora_config,                        # peft.LoraConfig
        group_size     : int   = 4,
        n_steps        : int   = 1000,
        kl_coef        : float = 0.1,
        batch_size     : int   = 8,         # K questions per step
        temperature    : float = 0.8,
        learning_rate  : float = 1e-4,
        output_dir     : Path  = Path("models/lora/grpo"),
        log_path       : Optional[Path] = None,
        val_evaluator  = None,              # SQLEvaluator for in-training monitoring
        val_data       : Optional[list[Example]] = None,
        val_every      : int   = 50,        # monitor every N steps
    ):
        self.reward_fn     = reward_fn
        self.lora_config   = lora_config
        self.group_size    = group_size
        self.n_steps       = n_steps
        self.kl_coef       = kl_coef
        self.batch_size    = batch_size
        self.temperature   = temperature
        self.learning_rate = learning_rate
        self.output_dir    = Path(output_dir)
        self.log_path      = log_path
        self.val_evaluator = val_evaluator
        self.val_data      = val_data or []
        self.val_every     = val_every

    def optimize(
        self,
        generator  : SQLGenerator,
        train_data : list[Example],
        val_data   : list[Example],
        output_dir : Path,
    ) -> SQLGenerator:
        """
        Run the GRPO training loop.
        Returns a LoRAGenerator pointing at the final checkpoint.
        """
        try:
            import torch
            from peft import get_peft_model
            from torch.optim import AdamW
        except ImportError:
            raise ImportError(
                "torch and peft are required for GRPO training. "
                "Install with: pip install torch peft"
            )

        from text2sql.inference.generator import LLMGenerator, LoRAGenerator
        from text2sql.inference.sql_utils import SQLUtils

        if not isinstance(generator, LLMGenerator):
            raise TypeError("GRPOOptimizer requires an LLMGenerator.")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        engine         = generator.engine
        prompt_builder = generator.prompt_builder

        # ── Attach LoRA ────────────────────────────────────────────────────────
        log.info("Attaching LoRA adapter …")
        peft_model = get_peft_model(engine.model, self.lora_config)
        peft_model.print_trainable_parameters()
        peft_model.train()

        optimizer = AdamW(
            filter(lambda p: p.requires_grad, peft_model.parameters()),
            lr=self.learning_rate,
        )

        # Reference model for KL penalty (frozen copy of base weights)
        ref_model = engine.model
        ref_model.eval()

        training_log = []
        log_path = self.log_path or output_dir / "training_log.jsonl"

        log.info(f"Starting GRPO training: {self.n_steps} steps, "
                 f"K={self.batch_size} questions, G={self.group_size} completions")

        for step in range(1, self.n_steps + 1):
            batch = random.sample(train_data, min(self.batch_size, len(train_data)))

            # ── ROLLOUT (GPU) ──────────────────────────────────────────────────
            # Temporarily swap peft_model in for sampling
            engine.model = peft_model
            all_outputs = sample_rollout(
                engine, prompt_builder, batch,
                group_size=self.group_size, temperature=self.temperature,
            )
            engine.model = ref_model  # restore reference

            # ── REWARD (CPU) ───────────────────────────────────────────────────
            all_rewards: list[list[float]] = []
            for example, group_outputs in zip(batch, all_outputs):
                group_rewards = []
                for raw_output in group_outputs:
                    pred_sql, _ = SQLUtils.extract(raw_output)
                    r = self.reward_fn.compute(example.db_id, pred_sql, example.true_sql)
                    group_rewards.append(r.total)
                all_rewards.append(group_rewards)

            mean_reward = sum(r for group in all_rewards for r in group) / \
                         (self.batch_size * self.group_size)

            # ── ADVANTAGES ────────────────────────────────────────────────────
            all_adv = group_advantages(all_rewards)

            # ── POLICY GRADIENT + KL LOSS (GPU) ───────────────────────────────
            peft_model.train()
            optimizer.zero_grad()

            total_loss = torch.tensor(0.0, requires_grad=True,
                                      device=next(peft_model.parameters()).device)

            for example, group_outputs, group_adv in zip(batch, all_outputs, all_adv):
                prompt = prompt_builder.build(example.question, example.db_id)
                for completion, adv in zip(group_outputs, group_adv):
                    full_text = prompt + completion
                    inputs    = engine.tokenizer(full_text, return_tensors="pt",
                                                 truncation=True, max_length=1024)
                    inputs    = {k: v.to(peft_model.device) for k, v in inputs.items()}

                    with torch.no_grad():
                        ref_logits  = ref_model(**inputs).logits
                    policy_logits   = peft_model(**inputs).logits

                    # Token-level log probs
                    log_probs_policy = torch.nn.functional.log_softmax(policy_logits, dim=-1)
                    log_probs_ref    = torch.nn.functional.log_softmax(ref_logits,    dim=-1)

                    target_ids = inputs["input_ids"][:, 1:]  # shift
                    lp_policy  = log_probs_policy[:, :-1].gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
                    lp_ref     = log_probs_ref[:,    :-1].gather(2, target_ids.unsqueeze(-1)).squeeze(-1)

                    # Policy gradient loss (negative because we maximise reward)
                    pg_loss = -(adv * lp_policy.mean())
                    # KL penalty
                    kl      = (lp_policy - lp_ref).mean()
                    loss    = pg_loss + self.kl_coef * kl
                    total_loss = total_loss + loss

            total_loss = total_loss / (self.batch_size * self.group_size)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(peft_model.parameters(), 1.0)
            optimizer.step()

            # ── LOGGING ───────────────────────────────────────────────────────
            if step % 10 == 0:
                log.info(f"  step={step:>4}  loss={total_loss.item():.4f}  "
                         f"mean_reward={mean_reward:.4f}")

            # ── VALIDATION MONITOR ────────────────────────────────────────────
            val_score = None
            val_data_to_use = val_data or self.val_data
            if self.val_evaluator and val_data_to_use and step % self.val_every == 0:
                peft_model.eval()
                engine.model = peft_model
                val_sample   = random.sample(val_data_to_use,
                                             min(50, len(val_data_to_use)))
                from text2sql.inference.generator import LLMGenerator
                monitor_gen  = LLMGenerator(generator.name, engine, prompt_builder)
                preds        = monitor_gen.generate_batch(val_sample, progress=False)
                val_score    = self.val_evaluator.score(preds) * 100
                engine.model = ref_model
                peft_model.train()
                log.info(f"  [val] step={step}  string_match={val_score:.1f}%")

            entry = {"step": step, "loss": total_loss.item(),
                     "mean_reward": mean_reward, "val_score": val_score}
            training_log.append(entry)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")

            # ── CHECKPOINT ────────────────────────────────────────────────────
            if step % 500 == 0:
                ckpt = self.output_dir / f"checkpoint-{step}"
                peft_model.save_pretrained(str(ckpt))
                log.info(f"  Saved checkpoint → {ckpt}")

        # ── FINAL CHECKPOINT ──────────────────────────────────────────────────
        final_ckpt = self.output_dir / "checkpoint-final"
        peft_model.save_pretrained(str(final_ckpt))
        log.info(f"Training complete. Final checkpoint → {final_ckpt}")

        return LoRAGenerator(
            name            = generator.name + "_grpo",
            base_engine     = generator.engine,
            lora_checkpoint = final_ckpt,
            prompt_builder  = prompt_builder,
        )
