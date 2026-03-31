"""SQLGenerator Protocol — the interface all generators implement."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from text2sql.data.types import Example, Prediction


@runtime_checkable
class SQLGenerator(Protocol):
    """
    Anything that turns an Example into a Prediction.

    Baseline (frozen weights + fixed prompt), prompt-optimised
    (frozen weights + tuned prompt), and LoRA-adapted generators
    all implement this same protocol.

    The evaluation pipeline and comparison table only ever see this interface —
    they don't need to know how SQL was generated.
    """
    name: str

    def generate(self, example: Example) -> Prediction:
        """Generate SQL for a single example."""
        ...

    def generate_batch(
        self,
        examples    : list[Example],
        progress    : bool = True,
        output_path : Path | None = None,
    ) -> list[Prediction]:
        """
        Generate SQL for a list of examples.
        If output_path is given, incrementally save after every row (crash-safe).
        """
        ...
