from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_run_eval_cli(tmp_path):
    results = tmp_path / "results.json"
    figure = tmp_path / "summary.png"
    subprocess.run(
        [
            sys.executable,
            "06_prospective/run_eval.py",
            "--targets",
            "06_prospective/targets.json",
            "--k",
            "6",
            "--output",
            str(results),
            "--figure",
            str(figure),
            "--frontier",
            "metrics/compute_frontier_week5.json",
            "--aizynth-fallback",
        ],
        cwd=ROOT,
        check=True,
    )
    payload = json.loads(results.read_text())
    assert "askcos" in payload
    assert payload["askcos"]["feasible_at_k"] >= 0
    assert figure.exists()
