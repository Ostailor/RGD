from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def run(cmd):
    subprocess.run(cmd, cwd=ROOT, check=True)


def test_load_split_sampling():
    sys.path.insert(0, str(ROOT))
    data = __import__('03_uncertainty.data', fromlist=['load_split'])
    split = data.load_split('moses', max_samples=1000, seed=1)
    assert len(split.smiles) == 1000
    assert len(split.targets) == 1000


def test_conformal_quantile_monotonic():
    sys.path.insert(0, str(ROOT))
    conformal = __import__('03_uncertainty.conformal', fromlist=['quantile'])
    residuals = np.linspace(0, 1, 100)
    q50 = conformal.quantile(residuals, 0.5)
    q90 = conformal.quantile(residuals, 0.9)
    assert q90 >= q50


def test_calibration_pipeline(tmp_path):
    ensemble_path = tmp_path / "ensemble.json"
    metrics_dir = tmp_path / "metrics"
    figures_dir = tmp_path / "figures"

    run([
        sys.executable,
        "-m",
        "03_uncertainty.train_ensemble",
        "--dataset",
        "moses",
        "--members",
        "2",
        "--max-samples",
        "500",
        "--output",
        str(ensemble_path),
    ])

    run([
        sys.executable,
        "-m",
        "03_uncertainty.calibrate",
        "--dataset",
        "moses",
        "--ensemble",
        str(ensemble_path),
        "--max-samples",
        "1000",
        "--metrics-dir",
        str(metrics_dir),
        "--figures-dir",
        str(figures_dir),
    ])

    assert (metrics_dir / "calibration_table.csv").exists()
    assert (metrics_dir / "coverage_week3.json").exists()
    assert (figures_dir / "week3_reliability.png").exists()
