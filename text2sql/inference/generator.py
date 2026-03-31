"""LLMGenerator and LoRAGenerator — concrete SQLGenerator implementations."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from text2sql.data.types import Example, Prediction, predictions_to_df
from text2sql.data.schema import PromptBuilder
from text2sql.inference.engine import InferenceEngine
from text2sql.inference.sql_utils import SQLUtils


class LLMGenerator:
    """
    Frozen-weight generator with a configurable PromptBuilder.
    This is the baseline generator — same model, better prompts = PromptOptimizer output.
    """

    def __init__(self, name: str, engine: InferenceEngine,
                 prompt_builder: PromptBuilder):
        self.name           = name
        self.engine         = engine
        self.prompt_builder = prompt_builder

    def with_prompt(self, new_prompt: PromptBuilder) -> "LLMGenerator":
        """Return a new generator with a different prompt. Weights unchanged."""
        return LLMGenerator(self.name, self.engine, new_prompt)

    def generate(self, example: Example) -> Prediction:
        prompt              = self.prompt_builder.build(example.question, example.db_id)
        raw_output          = self.engine.generate(prompt)
        pred_sql, tag_found = SQLUtils.extract(raw_output)
        return Prediction(
            db_id          = example.db_id,
            question       = example.question,
            true_sql       = example.true_sql,
            pred_sql       = pred_sql,
            raw_output     = raw_output,
            tag_found      = tag_found,
            difficulty     = example.difficulty,
            generator_name = self.name,
        )

    def generate_batch(
        self,
        examples    : list[Example],
        progress    : bool = True,
        output_path : Path | None = None,
    ) -> list[Prediction]:
        """
        Generate for all examples. Saves incrementally if output_path given.
        Crash-safe: you can resume from the last saved row.
        """
        # Resume from already-generated rows
        done_ids: set[str] = set()
        predictions: list[Prediction] = []

        if output_path and Path(output_path).exists():
            df = pd.read_csv(output_path)
            from text2sql.data.types import df_to_predictions
            predictions = df_to_predictions(df)
            done_ids    = {p.db_id + p.question for p in predictions}

        remaining = [e for e in examples
                     if (e.db_id + e.question) not in done_ids]

        it = tqdm(remaining) if progress else remaining
        correct = sum(SQLUtils.exact_match(p.pred_sql, p.true_sql) for p in predictions)

        for example in it:
            pred = self.generate(example)
            predictions.append(pred)
            if SQLUtils.exact_match(pred.pred_sql, pred.true_sql):
                correct += 1

            if output_path:
                predictions_to_df(predictions).to_csv(output_path, index=False)

            n = len(predictions)
            if progress:
                acc = correct / n * 100
                it.set_postfix(acc=f"{acc:.1f}%",
                               tag="✓" if pred.tag_found else "✗",
                               diff=pred.difficulty)

        return predictions


class LoRAGenerator(LLMGenerator):
    """
    Same interface as LLMGenerator but loads a LoRA adapter at init time.
    The adapter is merged into the base model so generation is identical.
    """

    def __init__(self, name: str, base_engine: InferenceEngine,
                 lora_checkpoint: Path, prompt_builder: PromptBuilder):
        from text2sql.training.lora import merge_adapter
        merged_engine = merge_adapter(base_engine, lora_checkpoint)
        super().__init__(name, merged_engine, prompt_builder)
        self.lora_checkpoint = lora_checkpoint
