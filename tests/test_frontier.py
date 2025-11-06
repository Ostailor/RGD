from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

instrument = __import__('05_frontier.instrument_flops', fromlist=['ModelConfig', 'estimate_total_flops', 'success_proxy'])
ModelConfig = instrument.ModelConfig
estimate_total_flops = instrument.estimate_total_flops
success_proxy = instrument.success_proxy


def test_estimate_total_flops_positive():
    cfg = ModelConfig("test", 100, 768, 64, 4, 20, "askcos")
    costs = estimate_total_flops(cfg)
    assert costs["total_flops"] > 0
    assert success_proxy(cfg) > 0


def test_compute_frontier_cli(tmp_path):
    metrics_path = tmp_path / "frontier.json"
    figure_path = tmp_path / "frontier.png"
    subprocess.run(
        [
            sys.executable,
            "scripts/run_compute_frontier.py",
            "--output",
            str(metrics_path),
            "--figure",
            str(figure_path),
        ],
        cwd=ROOT,
        check=True,
    )
    data = json.loads(metrics_path.read_text())
    assert len(data) > 0
    assert figure_path.exists()
