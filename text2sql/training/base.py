"""SQLOptimizer ABC — base for all generator optimization strategies."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from text2sql.data.types import Example
from text2sql.inference.base import SQLGenerator


class SQLOptimizer(ABC):
    """
    Takes a starting SQLGenerator + training data.
    Returns a new (better) SQLGenerator.

    This is the single abstraction that distinguishes the three approaches:
      - IdentityOptimizer : no-op (baseline)
      - PromptOptimizer   : tunes the prompt, weights frozen
      - GRPOOptimizer     : tunes LoRA weights, can start from tuned prompt
    """

    @abstractmethod
    def optimize(
        self,
        generator  : SQLGenerator,
        train_data : list[Example],
        val_data   : list[Example],
        output_dir : Path,
    ) -> SQLGenerator:
        """
        Run the optimization loop.
        Returns a new generator that should outperform the input generator.
        output_dir is used to save checkpoints, history, and best outputs.
        """
        ...
