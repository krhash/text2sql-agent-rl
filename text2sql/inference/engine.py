"""InferenceEngine — loads HuggingFace model and runs generation."""
from __future__ import annotations

import os

DTYPE_MAP = {
    "bfloat16": None,   # resolved lazily after torch import
    "float16" : None,
    "float32" : None,
}


class InferenceEngine:
    """
    Loads a HuggingFace causal LM and runs greedy text generation.

    Import of torch/transformers is deferred so that the rest of the
    package can be imported on CPU-only machines without errors.
    """

    def __init__(self, model_id: str, dtype: str = "bfloat16",
                 cache_dir: str | None = None,
                 model_path: str | None = None,
                 max_new_tokens: int = 256):
        """
        model_id       : HuggingFace model ID (used if model_path is None)
        dtype          : bfloat16 | float16 | float32
        cache_dir      : where to cache downloaded weights
        model_path     : load directly from local directory
        max_new_tokens : generation budget
        """
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        _dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16" : torch.float16,
            "float32" : torch.float32,
        }

        load_from = model_path if model_path else model_id
        cache     = os.path.expandvars(cache_dir) if cache_dir else None

        print(f"\nLoading model from : {load_from}")
        print(f"Cache directory    : {cache or 'HF default'}")
        print(f"dtype              : {dtype}")

        self.tokenizer = AutoTokenizer.from_pretrained(load_from, cache_dir=cache)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            load_from,
            torch_dtype=_dtype_map[dtype],
            device_map="auto",
            cache_dir=cache,
        )
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        print(f"Model loaded on    : {next(self.model.parameters()).device}\n")

    def generate(self, prompt: str, max_new_tokens: int | None = None) -> str:
        """Run greedy generation and return only the newly generated tokens."""
        import torch
        inputs    = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=[
                    self.tokenizer.eos_token_id,
                    self.tokenizer.encode("\n\n", add_special_tokens=False)[0],
                ],
            )
        new_tokens = outputs[0][input_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def generate_sampled(self, prompt: str, n: int = 4,
                         temperature: float = 0.8,
                         max_new_tokens: int | None = None) -> list[str]:
        """
        Generate n diverse completions with sampling.
        Used by GRPOOptimizer rollout — not for greedy inference.
        """
        import torch
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                do_sample=True,
                temperature=temperature,
                num_return_sequences=n,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        results = []
        for out in outputs:
            new_tokens = out[input_len:]
            results.append(self.tokenizer.decode(new_tokens, skip_special_tokens=True))
        return results
