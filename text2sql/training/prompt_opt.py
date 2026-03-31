"""PromptOptimizer — actor-critique loop for automatic prompt improvement."""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Optional

from text2sql.data.types import Example
from text2sql.eval.base import SQLEvaluator
from text2sql.inference.base import SQLGenerator
from text2sql.training.base import SQLOptimizer

log = logging.getLogger(__name__)


class PromptOptimizer(SQLOptimizer):
    """
    Automatic prompt optimization via actor-critique loop.

    Each iteration:
      1. ACTOR  : generate SQL on a small training sample with current prompt
      2. CRITIC : the LLM critiques failed predictions (why was the SQL wrong?)
      3. REFINE : the LLM proposes an improved system prompt from the critiques
      4. TRACK  : keep the best prompt seen across all iterations

    Weights are FROZEN throughout — only the prompt text changes.
    Returns an LLMGenerator with the improved PromptBuilder.
    """

    CRITIQUE_TEMPLATE = """You are reviewing a text-to-SQL system's mistakes.

Database schema: {schema}
Question: {question}
Correct SQL: {true_sql}
Generated SQL: {pred_sql}

Explain specifically why the generated SQL is wrong. Be concrete — reference column names, tables, and join conditions. Keep your answer to 2-3 sentences."""

    REFINE_TEMPLATE = """You are improving a text-to-SQL system prompt.

Current system prompt:
---
{current_prompt}
---

Current exact match score on {n} examples: {score:.1f}%

Common failure patterns identified by the critique agent:
{critiques}

Write an improved system prompt that addresses these failures.
Output ONLY the new prompt text — no explanations, no preamble."""

    def __init__(
        self,
        evaluator    : SQLEvaluator,
        n_iterations : int = 5,
        sample_size  : int = 100,
        max_failures : int = 10,    # max failures to send to critic per iteration
    ):
        self.evaluator    = evaluator
        self.n_iterations = n_iterations
        self.sample_size  = sample_size
        self.max_failures = max_failures

    def optimize(
        self,
        generator  : SQLGenerator,
        train_data : list[Example],
        val_data   : list[Example],
        output_dir : Path,
    ) -> SQLGenerator:
        from text2sql.inference.generator import LLMGenerator
        from text2sql.data.schema import PromptBuilder

        if not isinstance(generator, LLMGenerator):
            raise TypeError("PromptOptimizer requires an LLMGenerator.")

        output_dir.mkdir(parents=True, exist_ok=True)
        history_path = output_dir / "opt_history.jsonl"

        current_prompt = generator.prompt_builder
        best_prompt    = current_prompt
        best_score     = 0.0
        history        = []

        # Baseline score
        sample    = random.sample(train_data, min(self.sample_size, len(train_data)))
        cur_gen   = generator.with_prompt(current_prompt)
        preds     = cur_gen.generate_batch(sample, progress=False)
        score     = self.evaluator.score(preds) * 100
        best_score = score
        log.info(f"[PromptOpt] Baseline score: {score:.1f}%")

        for iteration in range(self.n_iterations):
            log.info(f"[PromptOpt] Iteration {iteration + 1}/{self.n_iterations} "
                     f"(current score: {score:.1f}%)")

            # ── ACTOR STEP ─────────────────────────────────────────────────────
            sample  = random.sample(train_data, min(self.sample_size, len(train_data)))
            cur_gen = generator.with_prompt(current_prompt)
            preds   = cur_gen.generate_batch(sample, progress=False)
            score   = self.evaluator.score(preds) * 100
            log.info(f"  Actor score: {score:.1f}%")

            # ── CRITIC STEP ────────────────────────────────────────────────────
            failures = [p for p in preds if p.pred_sql != p.true_sql][:self.max_failures]
            critiques = []
            for p in failures:
                schema = cur_gen.prompt_builder.schema_dict.get(p.db_id, {})
                critique_prompt = self.CRITIQUE_TEMPLATE.format(
                    schema   = schema.get("Schema (values (type))", ""),
                    question = p.question,
                    true_sql = p.true_sql,
                    pred_sql = p.pred_sql,
                )
                critique = generator.engine.generate(critique_prompt)
                critiques.append(critique.strip())

            if not critiques:
                log.info("  No failures this iteration — score is perfect on sample.")
                break

            # ── REFINE STEP ────────────────────────────────────────────────────
            aggregated = "\n\n".join(f"- {c}" for c in critiques)
            refine_prompt = self.REFINE_TEMPLATE.format(
                current_prompt = current_prompt.system_prompt,
                n              = len(preds),
                score          = score,
                critiques      = aggregated,
            )
            new_system = generator.engine.generate(refine_prompt, max_new_tokens=750).strip()
            
            # Clean up leakage: Instruct models often repeat the '---' delimiters we showed them
            new_system = new_system.strip("-\n `")
            if new_system.startswith("Current system prompt:"):
                new_system = new_system.replace("Current system prompt:", "").strip("-\n `")
                
            current_prompt = current_prompt.with_system(new_system)

            if score > best_score:
                best_score  = score
                best_prompt = current_prompt
                log.info(f"  New best prompt! Score: {score:.1f}%")

            hist_entry = {"iteration": iteration + 1, "score": score,
                          "n_failures": len(failures), "new_system": new_system}
            history.append(hist_entry)
            with open(history_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(hist_entry) + "\n")

        # Save best prompt for the infer stage to pick up
        best_prompt.save(output_dir / "best_prompt.json")
        log.info(f"[PromptOpt] Done. Best score: {best_score:.1f}% → {output_dir}/best_prompt.json")

        return generator.with_prompt(best_prompt)
