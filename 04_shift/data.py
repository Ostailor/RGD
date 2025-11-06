"""Data helpers for Week 4 shift simulations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_PATHS = {
    "moses": ROOT / "data" / "moses.smi",
    "guacamol": ROOT / "data" / "guacamol.smi",
}


def _load_smiles(dataset: str) -> List[str]:
    path = DATA_PATHS[dataset]
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if lines and lines[0].lower() == "smiles":
        lines = lines[1:]
    return lines


def _featurize(smiles: str) -> List[float]:
    length = len(smiles)
    hetero = sum(smiles.count(atom) for atom in "NOS")
    aromatic = smiles.count("c")
    halogens = sum(smiles.count(atom) for atom in "FClBrI")
    rings = smiles.count("1")
    return [
        length,
        hetero,
        aromatic,
        halogens,
        rings,
        hetero / max(length, 1),
        aromatic / max(length, 1),
        halogens / max(length, 1),
    ]


def _compute_logits(features: np.ndarray) -> np.ndarray:
    weights = np.array([0.04, 0.9, 0.3, 0.5, -0.2, 1.2, 0.8, 0.4], dtype=np.float32)
    logits = features @ weights
    logits -= logits.mean()
    return logits


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class ShiftDataset:
    smiles: List[str]
    features: np.ndarray
    probs: np.ndarray
    labels: np.ndarray


def load_dataset(dataset: str = "moses", max_samples: int = 40000, seed: int = 0) -> ShiftDataset:
    rng = np.random.default_rng(seed)
    smiles = _load_smiles(dataset)
    if len(smiles) > max_samples:
        idx = rng.choice(len(smiles), size=max_samples, replace=False)
        smiles = [smiles[i] for i in idx]
    feats = np.array([_featurize(smi) for smi in smiles], dtype=np.float32)
    logits = _compute_logits(feats) + rng.normal(scale=0.3, size=feats.shape[0])
    probs = _sigmoid(logits)
    labels = rng.binomial(1, probs).astype(np.float32)
    return ShiftDataset(smiles=smiles, features=feats, probs=probs, labels=labels)


def split_indices(n: int, train_frac: float = 0.6, val_frac: float = 0.2, seed: int = 0) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    splits = {
        "train": np.sort(idx[:n_train]),
        "val": np.sort(idx[n_train : n_train + n_val]),
        "test": np.sort(idx[n_train + n_val :]),
    }
    return splits


def covariate_filter(features: np.ndarray, threshold: float = 40.0) -> np.ndarray:
    length = features[:, 0]
    hetero = features[:, 1]
    return (length > threshold) | (hetero >= 4)

