#!/usr/bin/env python3
"""Train a toy deep ensemble for Week 3 calibration experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from .data import load_split, stratified_indices
from .models import featurize, fit_linear_model


def train_member(seed: int, x: np.ndarray, y: np.ndarray) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    noise = rng.normal(scale=0.01, size=x.shape)
    model = fit_linear_model(x + noise, y)
    return {"weights": model.weights.tolist(), "bias": model.bias}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="moses")
    parser.add_argument("--members", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("03_uncertainty/ensemble.json"))
    parser.add_argument("--max-samples", type=int, default=20000)
    args = parser.parse_args()

    split = load_split(args.dataset, max_samples=args.max_samples, seed=1337)
    indices = stratified_indices(split)["train"]
    smiles = [split.smiles[i] for i in indices]
    targets = split.targets[indices]
    x = featurize(smiles)

    ensemble: List[Dict[str, object]] = []
    for member in range(args.members):
        params = train_member(seed=member, x=x, y=targets)
        ensemble.append(params)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"dataset": args.dataset, "models": ensemble}, indent=2))
    print(f"[INFO] Saved ensemble to {args.output}")


if __name__ == "__main__":
    main()
