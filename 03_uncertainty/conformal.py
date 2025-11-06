"""Conformal prediction helpers for Week 3."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np


def absolute_residuals(pred_mean: np.ndarray, targets: np.ndarray) -> np.ndarray:
    return np.abs(pred_mean - targets)


def quantile(residuals: np.ndarray, coverage: float) -> float:
    """Return conformal quantile for symmetric intervals.

    coverage is the desired marginal coverage (e.g., 0.9).
    """

    alpha = min(max(coverage, 0.0), 0.999)
    n = residuals.shape[0]
    q_level = min(1.0, math.ceil((n + 1) * alpha) / n)
    return float(np.quantile(residuals, q_level))


def conformal_interval(mean: np.ndarray, q: float) -> np.ndarray:
    lower = mean - q
    upper = mean + q
    return np.stack([lower, upper], axis=-1)


def evaluate_coverage(intervals: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean((targets >= intervals[:, 0]) & (targets <= intervals[:, 1])))

