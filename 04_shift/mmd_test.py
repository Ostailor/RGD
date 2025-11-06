"""MMD two-sample test with permutation-based p-values."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np


def rbf_kernel(x: np.ndarray, y: np.ndarray, gamma: float) -> np.ndarray:
    x_norm = np.sum(x * x, axis=1)[:, None]
    y_norm = np.sum(y * y, axis=1)[None, :]
    dist = x_norm + y_norm - 2 * np.dot(x, y.T)
    return np.exp(-gamma * np.maximum(dist, 0.0))


def compute_mmd(x: np.ndarray, y: np.ndarray, gammas: Iterable[float]) -> Tuple[float, float]:
    mmd_total = 0.0
    for gamma in gammas:
        k_xx = rbf_kernel(x, x, gamma)
        k_yy = rbf_kernel(y, y, gamma)
        k_xy = rbf_kernel(x, y, gamma)
        n = x.shape[0]
        m = y.shape[0]
        term_xx = (np.sum(k_xx) - np.trace(k_xx)) / (n * (n - 1))
        term_yy = (np.sum(k_yy) - np.trace(k_yy)) / (m * (m - 1))
        term_xy = np.sum(k_xy) / (n * m)
        mmd_total += term_xx + term_yy - 2 * term_xy
    return mmd_total / len(list(gammas)), scale_median_gamma(x, y)


def scale_median_gamma(x: np.ndarray, y: np.ndarray) -> float:
    xy = np.vstack([x, y])
    pdist = np.sum((xy[None, :, :] - xy[:, None, :]) ** 2, axis=-1)
    median = np.median(pdist)
    if median <= 0:
        median = 1.0
    return 1.0 / median


def mmd_test(x: np.ndarray, y: np.ndarray, num_permutations: int = 200) -> Tuple[float, float]:
    gamma0 = scale_median_gamma(x, y)
    gammas = [gamma0 / 2, gamma0, gamma0 * 2]
    stat, _ = compute_mmd(x, y, gammas)
    combined = np.vstack([x, y])
    n = x.shape[0]
    count = 0
    rng = np.random.default_rng(0)
    for _ in range(num_permutations):
        rng.shuffle(combined)
        x_perm = combined[:n]
        y_perm = combined[n:]
        perm_stat, _ = compute_mmd(x_perm, y_perm, gammas)
        if perm_stat >= stat:
            count += 1
    p_value = (count + 1) / (num_permutations + 1)
    return stat, p_value

