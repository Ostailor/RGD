"""Simple ensemble-ready model abstraction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np


def featurize(smiles: List[str]) -> np.ndarray:
    """Simple hand-crafted features: length, hetero counts, aromatic ratio."""
    feats = []
    for smi in smiles:
        length = len(smi)
        hetero = sum(smi.count(atom) for atom in "NOS")
        aromatic = smi.count("c")
        feats.append(
            [
                length,
                hetero,
                aromatic,
                hetero / max(length, 1),
                aromatic / max(length, 1),
            ]
        )
    return np.array(feats, dtype=np.float32)


@dataclass
class LinearModel:
    weights: np.ndarray
    bias: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        return x @ self.weights + self.bias


def fit_linear_model(x: np.ndarray, y: np.ndarray, weight_decay: float = 1e-3) -> LinearModel:
    xtx = x.T @ x + weight_decay * np.eye(x.shape[1])
    xty = x.T @ y
    weights = np.linalg.solve(xtx, xty)
    bias = float(y.mean() - x.mean(axis=0) @ weights)
    return LinearModel(weights=weights, bias=bias)


@dataclass
class EnsembleModel:
    members: List[LinearModel]

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        member_means = np.stack([m.predict(x) for m in self.members], axis=0)
        mean = member_means.mean(axis=0)
        epistemic = member_means.var(axis=0)
        aleatoric = np.maximum(0.01, epistemic * 0.1)
        variance = epistemic + aleatoric
        return mean, variance, member_means


def load_ensemble(path: Path) -> EnsembleModel:
    payload = json.loads(path.read_text())
    members = [LinearModel(weights=np.array(m["weights"], dtype=np.float32), bias=float(m["bias"])) for m in payload["models"]]
    return EnsembleModel(members=members)
