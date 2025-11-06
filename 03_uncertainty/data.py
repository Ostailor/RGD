"""Dataset utilities for uncertainty calibration experiments.

We reuse the Week 0–1 processed datasets to avoid additional downloads.
This module exposes thin wrappers that:
  * Load SMILES + target properties from JSON/CSV files.
  * Provide deterministic train/val/test splits (leveraging split JSONs).
  * Generate toy regression/classification targets for prototyping.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np


RAW_DATA = {
    "moses": Path("data/moses.smi"),
    "guacamol": Path("data/guacamol.smi"),
}

SPLITS = {
    "moses": Path("processed/moses/splits/scaffold_split.json"),
    "guacamol": Path("processed/guacamol/splits/scaffold_split.json"),
}


@dataclass
class DatasetSplit:
    smiles: List[str]
    targets: np.ndarray
    split: List[str]


def load_smiles(dataset: str) -> List[str]:
    path = RAW_DATA[dataset]
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if lines and lines[0].lower() == "smiles":
        lines = lines[1:]
    return lines


def toy_target(smiles: Iterable[str]) -> np.ndarray:
    values = []
    for smi in smiles:
        length = len(smi)
        aromatic = smi.count("c")
        hetero = sum(smi.count(atom) for atom in "NOS")
        value = 0.01 * length + 0.5 * np.tanh(hetero / 5) + 0.2 * (aromatic % 5)
        values.append(value)
    return np.array(values, dtype=np.float32)


def load_split(
    dataset: str,
    split_name: str = "scaffold_split",
    max_samples: Optional[int] = None,
    seed: int = 0,
) -> DatasetSplit:
    smiles = load_smiles(dataset)
    split_path = SPLITS[dataset]
    assignments = json.loads(split_path.read_text())["assignments"]
    split = [assignments.get(str(i), "train") for i in range(len(smiles))]
    targets = toy_target(smiles)

    if max_samples is not None and len(smiles) > max_samples:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(len(smiles), size=max_samples, replace=False))
        smiles = [smiles[i] for i in indices]
        split = [split[i] for i in indices]
        targets = targets[indices]
    return DatasetSplit(smiles=smiles, targets=targets, split=split)


def stratified_indices(split: DatasetSplit) -> Dict[str, np.ndarray]:
    mapping: Dict[str, List[int]] = {"train": [], "val": [], "test": []}
    for idx, label in enumerate(split.split):
        mapping.setdefault(label, []).append(idx)
    return {k: np.array(v, dtype=np.int64) for k, v in mapping.items()}
