#!/usr/bin/env python3
"""Compute calibration metrics, conformal coverage, and reliability plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import NormalDist
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .conformal import absolute_residuals, conformal_interval, evaluate_coverage, quantile
from .data import load_split, stratified_indices
from .models import EnsembleModel, featurize, load_ensemble


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def reliability_bins(probs: np.ndarray, labels: np.ndarray, bins: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    bucket = np.digitize(probs, edges, right=True)
    bin_acc = []
    bin_conf = []
    bin_frac = []
    for b in range(1, bins + 1):
        mask = bucket == b
        if mask.sum() == 0:
            continue
        bin_acc.append(labels[mask].mean())
        bin_conf.append(probs[mask].mean())
        bin_frac.append(mask.mean())
    return np.array(bin_acc), np.array(bin_conf), np.array(bin_frac)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    acc, conf, frac = reliability_bins(probs, labels, bins)
    return float(np.sum(np.abs(acc - conf) * frac))


def negative_log_likelihood(probs: np.ndarray, labels: np.ndarray) -> float:
    eps = 1e-8
    return float(-np.mean(labels * np.log(probs + eps) + (1 - labels) * np.log(1 - probs + eps)))


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((probs - labels) ** 2))


def build_histogram_calibrator(probs: np.ndarray, labels: np.ndarray, bins: int = 12) -> Tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    accuracies = np.zeros(bins)
    for i in range(bins):
        mask = (probs >= edges[i]) & (probs < edges[i + 1] if i < bins - 1 else probs <= edges[i + 1])
        if mask.any():
            accuracies[i] = labels[mask].mean()
        else:
            accuracies[i] = (edges[i] + edges[i + 1]) / 2
    return edges, accuracies


def apply_histogram_calibrator(probs: np.ndarray, edges: np.ndarray, accuracies: np.ndarray) -> np.ndarray:
    bins = np.digitize(probs, edges[1:-1], right=True)
    return accuracies[bins]


def load_ensemble_model(path: Path) -> EnsembleModel:
    return load_ensemble(path)


def ensemble_predictions(model: EnsembleModel, smiles: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    x = featurize(smiles)
    mean, variance, _ = model.predict(x)
    return mean, variance


def learn_affine_calibration(logits: np.ndarray, labels: np.ndarray, steps: int = 500, lr: float = 0.05) -> Tuple[float, float]:
    scale = 1.0
    bias = 0.0
    for _ in range(steps):
        z = scale * logits + bias
        probs = sigmoid(z)
        error = probs - labels
        grad_scale = np.mean(error * logits)
        grad_bias = np.mean(error)
        scale -= lr * grad_scale
        bias -= lr * grad_bias
        scale = float(np.clip(scale, -5.0, 5.0))
        bias = float(np.clip(bias, -5.0, 5.0))
    return scale, bias


def probability_from_logits(logits: np.ndarray, params: Tuple[float, float]) -> np.ndarray:
    scale, bias = params
    return sigmoid(scale * logits + bias)


def write_reliability_plot(bins_data: Dict[str, Tuple[np.ndarray, np.ndarray]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#cccccc", label="Ideal")
    for split, (acc, conf) in bins_data.items():
        ax.plot(conf, acc, marker="o", label=split)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("Reliability diagram")
    ax.legend()
    ax.grid(alpha=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_metrics_table(rows: List[Dict[str, object]], path: Path) -> None:
    import csv

    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_coverage_json(data: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def compute_binary_labels(values: np.ndarray) -> np.ndarray:
    threshold = np.median(values)
    return (values >= threshold).astype(np.float32)


def compute_conformal_metrics(mean_val, targets_val, mean_test, targets_test, variance_test):
    coverages = []
    naive = []
    residuals = absolute_residuals(mean_val, targets_val)
    coverage_grid = [0.5, 0.7, 0.8, 0.9, 0.95]
    for target in coverage_grid:
        q = quantile(residuals, target)
        intervals = conformal_interval(mean_test, q)
        cov = evaluate_coverage(intervals, targets_test)
        coverages.append({"target": target, "coverage": cov, "interval_width": float(2 * q)})

        # naive gaussian interval using z-score approx
        z_score = NormalDist().inv_cdf((1 + target) / 2)
        naive_lower = mean_test - z_score * np.sqrt(variance_test)
        naive_upper = mean_test + z_score * np.sqrt(variance_test)
        naive_cov = evaluate_coverage(np.stack([naive_lower, naive_upper], axis=-1), targets_test)
        naive.append({"target": target, "coverage": naive_cov})
    return coverages, naive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="moses")
    parser.add_argument("--ensemble", type=Path, default=Path("03_uncertainty/ensemble.json"))
    parser.add_argument("--max-samples", type=int, default=20000)
    parser.add_argument("--metrics-dir", type=Path, default=Path("metrics"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()

    split = load_split(args.dataset, max_samples=args.max_samples, seed=42)
    idx = stratified_indices(split)
    model = load_ensemble_model(args.ensemble)

    rows = []
    bins_plot_data = {}
    coverage_payload = {"dataset": args.dataset}

    stats_cache = {}

    for split_name in ("val", "test"):
        indices = idx[split_name]
        smiles = [split.smiles[i] for i in indices]
        targets = split.targets[indices]
        mean, variance = ensemble_predictions(model, smiles)
        labels = compute_binary_labels(targets)
        stats_cache[split_name] = {
            "mean": mean,
            "variance": variance,
            "targets": targets,
            "labels": labels,
        }

    mu = stats_cache["val"]["mean"].mean()
    sigma = stats_cache["val"]["mean"].std() + 1e-6
    logits_val = (stats_cache["val"]["mean"] - mu) / sigma
    calib_params = learn_affine_calibration(logits_val, stats_cache["val"]["labels"])
    probs_val_pre = probability_from_logits(logits_val, calib_params)
    hist_edges, hist_acc = build_histogram_calibrator(probs_val_pre, stats_cache["val"]["labels"])
    hist_weight = 0.5

    for split_name in ("val", "test"):
        mean = stats_cache[split_name]["mean"]
        variance = stats_cache[split_name]["variance"]
        targets = stats_cache[split_name]["targets"]
        labels = stats_cache[split_name]["labels"]

        logits = (mean - mu) / sigma
        probs_raw = probability_from_logits(logits, calib_params)
        probs_hist = apply_histogram_calibrator(probs_raw, hist_edges, hist_acc)
        probs = hist_weight * probs_hist + (1 - hist_weight) * probs_raw

        ece = expected_calibration_error(probs, labels)
        nll = negative_log_likelihood(probs, labels)
        brier = brier_score(probs, labels)

        acc, conf, _ = reliability_bins(probs, labels)
        bins_plot_data[split_name] = (acc, conf)

        rows.append({"split": split_name, "ece": ece, "nll": nll, "brier": brier})

        if split_name == "val":
            mean_val, var_val, targets_val = mean, variance, targets
        else:
            mean_test, var_test, targets_test = mean, variance, targets

    coverages, naive = compute_conformal_metrics(mean_val, targets_val, mean_test, targets_test, var_test)
    coverage_payload["conformal"] = coverages
    coverage_payload["naive"] = naive

    save_metrics_table(rows, args.metrics_dir / "calibration_table.csv")
    save_coverage_json(coverage_payload, args.metrics_dir / "coverage_week3.json")
    write_reliability_plot(bins_plot_data, args.figures_dir / "week3_reliability.png")
    print("[INFO] Calibration metrics saved.")


if __name__ == "__main__":
    main()
