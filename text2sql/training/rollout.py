"""GRPO rollout helpers — generate completions and compute group advantages."""
from __future__ import annotations

import statistics


def sample_rollout(
    engine        ,          # InferenceEngine
    prompt_builder,          # PromptBuilder
    examples      : list,    # list[Example]
    group_size    : int = 4,
    temperature   : float = 0.8,
) -> list[list[str]]:
    """
    For each example, generate group_size completions with temperature sampling.

    Returns shape [len(examples)][group_size] list of raw model output strings.
    Used by GRPOOptimizer — NOT for greedy baseline inference.
    """
    all_outputs = []
    for example in examples:
        prompt   = prompt_builder.build(example.question, example.db_id)
        outputs  = engine.generate_sampled(prompt, n=group_size, temperature=temperature)
        all_outputs.append(outputs)
    return all_outputs


def group_advantages(rewards: list[list[float]]) -> list[list[float]]:
    """
    GRPO group-relative advantage normalisation.

    For each group (one question's G completions):
        advantage_i = (reward_i - mean(group)) / (std(group) + eps)

    This is the key algorithmic difference from PPO — no value network needed.
    Groups with zero variance get zero advantage (all completions identical reward).

    Input shape  : [K questions][G completions]
    Output shape : [K questions][G completions]
    """
    eps = 1e-8
    result = []
    for group in rewards:
        mean_r = statistics.mean(group)
        std_r  = statistics.stdev(group) if len(group) > 1 else 0.0
        denom  = std_r + eps
        result.append([(r - mean_r) / denom for r in group])
    return result
