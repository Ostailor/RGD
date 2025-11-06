"""Approximate FLOP accounting for generation + oracle calls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class ModelConfig:
    name: str
    params_m: float
    hidden_dim: int
    decode_steps: int
    beam: int
    oracle_freq: int
    oracle: str


ORACLE_FLOPS = {
    "askcos": 3.5e11,  # per call equivalent, accounts for retrosynthesis solver
    "aizynth": 9.0e10,
}

GPU_EFFICIENCY = 0.35  # A100 vs theoretical peak


def transformer_flops(params_m: float, seq_len: int, hidden_dim: int) -> float:
    params = params_m * 1e6
    attention_cost = 2 * (seq_len**2) * hidden_dim
    mlp_cost = 4 * seq_len * (hidden_dim**2)
    return GPU_EFFICIENCY * (params * seq_len + attention_cost + mlp_cost)


def decode_flops(decode_steps: int, beam: int, hidden_dim: int) -> float:
    return GPU_EFFICIENCY * decode_steps * beam * (hidden_dim**2) * 4


def oracle_flops(oracle: str, calls: int) -> float:
    return ORACLE_FLOPS.get(oracle, 1e11) * calls


def estimate_total_flops(config: ModelConfig, seq_len: int = 128) -> Dict[str, float]:
    model = transformer_flops(config.params_m, seq_len, config.hidden_dim)
    decode = decode_flops(config.decode_steps, config.beam, config.hidden_dim)
    oracle_cost = oracle_flops(config.oracle, config.oracle_freq)
    total = model + decode + oracle_cost
    return {
        "model_flops": model,
        "decode_flops": decode,
        "oracle_flops": oracle_cost,
        "total_flops": total,
    }


def success_proxy(config: ModelConfig) -> float:
    base = 0.35
    scale = 0.04 * np.log1p(config.params_m)
    decode_gain = 0.01 * np.log1p(config.decode_steps)
    oracle_gain = 0.03 * np.log1p(config.oracle_freq)
    beam_penalty = 0.005 * max(config.beam - 4, 0)
    return float(np.clip(base + scale + decode_gain + oracle_gain - beam_penalty, 0.2, 0.95))

