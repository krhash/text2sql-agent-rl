"""IdentityOptimizer — no-op baseline optimizer."""
from __future__ import annotations

from pathlib import Path

from text2sql.data.types import Example
from text2sql.inference.base import SQLGenerator
from text2sql.training.base import SQLOptimizer


class IdentityOptimizer(SQLOptimizer):
    """
    Does nothing. Returns the input generator unchanged.
    Use this for the baseline experiment so all three stages
    go through the same Experiment.run() code path.
    """

    def optimize(
        self,
        generator  : SQLGenerator,
        train_data : list[Example],
        val_data   : list[Example],
        output_dir : Path,
    ) -> SQLGenerator:
        return generator
