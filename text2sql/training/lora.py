"""LoRA configuration helpers."""
from __future__ import annotations

from pathlib import Path


def default_lora_config(r: int = 16, lora_alpha: int = 32):
    """
    Returns a LoRA configuration for Llama 3.1 8B.

    r=16 adds ~20M trainable params out of 8B (0.25%).
    Fits alongside frozen base in bfloat16 on A100 40GB.
    """
    try:
        from peft import LoraConfig, TaskType
    except ImportError:
        raise ImportError("peft is required for LoRA: pip install peft")

    return LoraConfig(
        r              = r,
        lora_alpha     = lora_alpha,
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout   = 0.05,
        bias           = "none",
        task_type      = TaskType.CAUSAL_LM,
    )


def merge_adapter(base_engine, lora_checkpoint: Path):
    """
    Load a saved LoRA adapter and merge it into the base engine's model.
    Returns a new InferenceEngine-like object with the merged weights.

    Used by LoRAGenerator to produce a standard InferenceEngine.
    """
    try:
        from peft import PeftModel
    except ImportError:
        raise ImportError("peft is required for LoRA: pip install peft")

    from text2sql.inference.engine import InferenceEngine

    # Create a thin wrapper that reuses the existing tokenizer
    model = PeftModel.from_pretrained(base_engine.model, str(lora_checkpoint))
    model = model.merge_and_unload()
    model.eval()

    # Reuse existing engine shell, just swap the model
    merged = object.__new__(InferenceEngine)
    merged.tokenizer       = base_engine.tokenizer
    merged.model           = model
    merged.max_new_tokens  = base_engine.max_new_tokens
    return merged
