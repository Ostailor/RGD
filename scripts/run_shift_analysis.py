#!/usr/bin/env python3
"""Run Week 4 shift detection + estimation pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple
from importlib import import_module
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

data_module = import_module("04_shift.data")
mmd_module = import_module("04_shift.mmd_test")
bbse_module = import_module("04_shift.bbse_test")

covariate_filter = data_module.covariate_filter
load_dataset = data_module.load_dataset
split_indices = data_module.split_indices
mmd_test = mmd_module.mmd_test
estimate_shift = bbse_module.estimate_shift


def histogram_plot(length_source, length_target, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(length_source, bins=50, alpha=0.6, label="Source", density=True)
    ax.hist(length_target, bins=50, alpha=0.6, label="Shifted", density=True)
    ax.set_xlabel("SMILES length")
    ax.set_ylabel("Density")
    ax.set_title("Week 4 Covariate Shift (length distribution)")
    ax.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def resample_to_prior(probs: np.ndarray, labels: np.ndarray, desired: np.ndarray, seed: int = 7) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx_pos = np.where(labels == 1)[0]
    idx_neg = np.where(labels == 0)[0]
    n_total = len(labels)
    n_pos = max(1, int(desired[1] * n_total))
    n_neg = max(1, n_total - n_pos)
    pos_sel = rng.choice(idx_pos, size=n_pos, replace=True)
    neg_sel = rng.choice(idx_neg, size=n_neg, replace=True)
    sel = np.concatenate([pos_sel, neg_sel])
    rng.shuffle(sel)
    return probs[sel], labels[sel]


def sequential_simulation(runs: int = 200, steps: int = 200, alpha: float = 0.05) -> dict:
    threshold = 1.0 / alpha
    rng = np.random.default_rng(123)

    def simulate(mean: float) -> Tuple[float, float]:
        hits = 0
        stopping_times = []
        for _ in range(runs):
            e_value = 1.0
            stopped = steps
            for t in range(1, steps + 1):
                x = rng.normal(mean, 1.0)
                lam = 0.2
                e_value *= np.exp(lam * x - 0.5 * lam**2)
                if e_value >= threshold:
                    hits += 1
                    stopped = t
                    break
            stopping_times.append(stopped)
        return hits / runs, float(np.mean(stopping_times))

    false_alarm_rate, avg_time_null = simulate(0.0)
    detection_power, avg_time_shift = simulate(0.4)
    return {
        "false_alarm_rate": false_alarm_rate,
        "avg_time_null": avg_time_null,
        "detection_power": detection_power,
        "avg_time_shift": avg_time_shift,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="moses")
    parser.add_argument("--max-samples", type=int, default=15000)
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--metrics", type=Path, default=Path("metrics/shift_week4.json"))
    parser.add_argument("--report", type=Path, default=Path("reports/shift_week4.json"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()

    dataset = load_dataset(dataset=args.dataset, max_samples=args.max_samples, seed=4)
    splits = split_indices(len(dataset.smiles), seed=4)

    source_idx = splits["train"]
    target_idx = splits["test"]
    source_feats = dataset.features[source_idx]
    target_feats = dataset.features[target_idx]

    mask = covariate_filter(target_feats, threshold=45)
    shifted_feats = target_feats[mask]
    subset = min(len(shifted_feats), len(source_feats), 1500)
    source_block = source_feats[:subset]
    shifted_block = shifted_feats[:subset]

    stat, p_value = mmd_test(source_block, shifted_block, num_permutations=args.permutations)
    effect = float(np.mean(shifted_block[:, 0]) - np.mean(source_block[:, 0]))

    histogram_plot(
        source_block[:, 0],
        shifted_block[:, 0],
        args.figures_dir / "week4_shift_detection.png",
    )

    val_idx = splits["val"]
    probs_source = dataset.probs[val_idx]
    labels_source = dataset.labels[val_idx]

    target_probs = dataset.probs[target_idx]
    target_labels = dataset.labels[target_idx]
    desired_prior = np.array([0.3, 0.7])
    target_probs, target_labels = resample_to_prior(target_probs, target_labels, desired_prior)

    bbse_result = estimate_shift(probs_source, labels_source, target_probs, desired_prior)

    seq_metrics = sequential_simulation()

    metrics = {
        "covariate_shift": {
            "mmd_stat": stat,
            "p_value": p_value,
            "source_size": int(len(source_feats)),
            "target_size": int(len(shifted_feats)),
            "length_effect": effect,
        },
        "label_shift": {
            "true_priors": desired_prior.tolist(),
            "estimated_priors": bbse_result.estimated_priors.tolist(),
            "weights": bbse_result.weights.tolist(),
            "mae": bbse_result.mae,
        },
        "sequential_monitoring": seq_metrics,
    }

    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, indent=2))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
