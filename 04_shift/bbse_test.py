"""Black Box Shift Estimation utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BBSEEstimate:
    weights: np.ndarray
    true_priors: np.ndarray
    estimated_priors: np.ndarray
    mae: float


def confusion_matrix(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    preds = (probs >= threshold).astype(int)
    cm = np.zeros((2, 2), dtype=np.float64)
    for j in range(2):
        mask = labels == j
        if mask.sum() == 0:
            continue
        for i in range(2):
            cm[i, j] = np.mean(preds[mask] == i)
    return cm


def bbse(cm: np.ndarray, predicted_target: np.ndarray) -> np.ndarray:
    return np.linalg.solve(cm.T, predicted_target)


def estimate_shift(probs_source: np.ndarray, labels_source: np.ndarray, probs_target: np.ndarray, true_priors_target: np.ndarray) -> BBSEEstimate:
    cm = confusion_matrix(probs_source, labels_source)
    predicted_target = np.array([
        np.mean(probs_target < 0.5),
        np.mean(probs_target >= 0.5),
    ])
    est_priors = bbse(cm, predicted_target)
    priors_source = np.array([
        np.mean(labels_source == 0),
        np.mean(labels_source == 1),
    ])
    weights = np.clip(est_priors / priors_source, 0.0, 10.0)
    mae = float(np.mean(np.abs(est_priors - true_priors_target)))
    return BBSEEstimate(weights=weights, true_priors=true_priors_target, estimated_priors=est_priors, mae=mae)
