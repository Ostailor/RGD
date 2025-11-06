from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def test_mmd_detects_shift():
    sys.path.insert(0, str(ROOT))
    mmd = __import__('04_shift.mmd_test', fromlist=['mmd_test']).mmd_test
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, size=(200, 4))
    y = rng.normal(0.6, 1, size=(200, 4))
    _, p = mmd(x, y, num_permutations=50)
    assert p < 0.05


def test_bbse_estimator():
    sys.path.insert(0, str(ROOT))
    bbse = __import__('04_shift.bbse_test', fromlist=['estimate_shift']).estimate_shift
    rng = np.random.default_rng(1)
    probs_source = rng.uniform(0, 1, size=1000)
    labels_source = (probs_source > 0.5).astype(float)
    probs_target = np.concatenate([
        rng.uniform(0, 0.4, size=300),
        rng.uniform(0.6, 1.0, size=700),
    ])
    labels_target = (probs_target > 0.5).astype(float)
    desired = np.array([0.3, 0.7])
    result = bbse(probs_source, labels_source, probs_target, desired)
    assert result.mae < 0.1
    assert np.all(result.weights > 0)


def test_shift_analysis_cli(tmp_path):
    metrics = tmp_path / "shift.json"
    report = tmp_path / "shift_report.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/run_shift_analysis.py",
            "--max-samples",
            "2000",
            "--permutations",
            "30",
            "--metrics",
            str(metrics),
            "--report",
            str(report),
            "--figures-dir",
            str(tmp_path),
        ],
        cwd=ROOT,
        check=True,
    )
    payload = json.loads(metrics.read_text())
    assert payload["covariate_shift"]["p_value"] < 0.1
    assert report.exists()
