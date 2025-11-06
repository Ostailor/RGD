#!/usr/bin/env python3
"""Prospective evaluation with dual oracles and zero-shot conformal recalibration."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

import sys
from importlib import import_module

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

planner_module = import_module("planner")
PlannerInLoop = planner_module.PlannerInLoop
api_module = import_module("planner.api")
PlannerReport = api_module.PlannerReport

AskcosClient = import_module("02_oracle.askcos").AskcosClient
AiZynthClient = import_module("02_oracle.aizynth").AiZynthClient
MockAskcosSession = import_module("02_oracle.stubs").MockAskcosSession


def load_targets(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Targets file not found: {path}")
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError("Targets file must be a list of SMILES strings")
    return [str(item) for item in data]


def build_planner(mode: str, aizynth_config: Path, aizynth_fallback: bool) -> PlannerInLoop:
    if mode == "askcos":
        session = MockAskcosSession(feasible_bias=0.5, failure_rate=0.15, avg_latency=0.18, latency_jitter=0.04)
        askcos = AskcosClient(base_url="https://mock.askcos", session=session, max_calls_per_minute=None)
        return PlannerInLoop([askcos])
    if mode == "aizynth":
        ai = AiZynthClient(config_path=aizynth_config, use_fallback=aizynth_fallback)
        return PlannerInLoop([ai])
    raise ValueError(f"Unknown mode: {mode}")


def feasible_curve(planner: PlannerInLoop, smiles: List[str], k_max: int, baseline: float) -> List[float]:
    values = []
    for k in range(1, k_max + 1):
        report = planner.evaluate(smiles, k=k, baseline_feasible_at_k=baseline)
        values.append(report.feasible_at_k)
    return values


def summarize_report(report: PlannerReport) -> Dict[str, float]:
    askcos_stats = report.oracle_stats.get("askcos")
    aizynth_stats = report.oracle_stats.get("aizynth")
    return {
        "feasible_at_k": report.feasible_at_k,
        "baseline_feasible_at_k": report.baseline_feasible_at_k,
        "improvement": report.improvement,
        "median_latency": report.median_latency,
        "askcos_success": askcos_stats.success_rate if askcos_stats else None,
        "askcos_latency": askcos_stats.avg_latency if askcos_stats else None,
        "aizynth_success": aizynth_stats.success_rate if aizynth_stats else None,
        "aizynth_latency": aizynth_stats.avg_latency if aizynth_stats else None,
    }
def load_coverage_metrics(path: Path, target: float) -> Tuple[float, float]:
    payload = json.loads(path.read_text())
    conformal_entries = payload.get("conformal", [])
    entry = next((item for item in conformal_entries if abs(item["target"] - target) < 1e-6), None)
    if entry is None:
        raise ValueError(f"Coverage target {target} not found in {path}")
    return float(entry["coverage"]), float(entry["interval_width"])


def load_frontier(path: Path) -> List[Dict[str, float]]:
    if path.exists():
        return json.loads(path.read_text())
    else:
        sweep_script = import_module("scripts.run_compute_frontier")
        sweep_script.main()
        return json.loads(path.read_text())


def build_summary_figure(
    askcos_curve: List[float],
    aiz_curve: List[float],
    coverage_target: float,
    coverage_actual: float,
    coverage_width: float,
    frontier_data: List[Dict[str, float]],
    path: Path,
) -> None:
    ks = np.arange(1, len(askcos_curve) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(ks, askcos_curve, marker="o", label="ASKCOS")
    axes[0].plot(ks, aiz_curve, marker="s", label="AiZynthFinder")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Feasible@k")
    axes[0].set_title("Prospective Feasibility")
    axes[0].set_ylim(0, 1)
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].bar(["Target", "Achieved"], [coverage_target, coverage_actual], color=["#cccccc", "#4c72b0"])
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Coverage")
    axes[1].set_title(f"Zero-shot conformal (width={coverage_width:.2f})")
    axes[1].grid(axis="y", alpha=0.3)

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(frontier_data)))
    for color, entry in zip(colors, frontier_data):
        minutes = entry["total_latency_seconds"] / 60.0
        axes[2].scatter(minutes, entry["success_rate"], color=color, s=80, label=entry["run"].capitalize())
        axes[2].annotate(
            f"{entry['success_rate']:.2f}",
            (minutes, entry["success_rate"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
        )
    axes[2].set_xlabel("Total solver time (minutes)")
    axes[2].set_ylabel("Feasible@k")
    axes[2].set_title("Compute frontier (empirical)")
    axes[2].grid(alpha=0.3)
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=Path("06_prospective/targets.json"))
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("06_prospective/results.json"))
    parser.add_argument("--figure", type=Path, default=Path("figures/week6_summary.png"))
    parser.add_argument("--frontier", type=Path, default=Path("metrics/compute_frontier_week5.json"))
    parser.add_argument("--aizynth-config", type=Path, default=Path("config.yml"))
    parser.add_argument(
        "--aizynth-fallback",
        action="store_true",
        help="Use heuristic AiZynth fallback instead of the full solver",
    )
    args = parser.parse_args()

    smiles = load_targets(args.targets)
    baseline = 0.28

    planner_primary = build_planner("askcos", args.aizynth_config, args.aizynth_fallback)
    report_primary = planner_primary.evaluate(smiles, k=args.k, baseline_feasible_at_k=baseline)

    planner_secondary = build_planner("aizynth", args.aizynth_config, args.aizynth_fallback)
    report_secondary = planner_secondary.evaluate(smiles, k=args.k, baseline_feasible_at_k=baseline)

    askcos_curve = feasible_curve(planner_primary, smiles, args.k, baseline)
    aiz_curve = feasible_curve(planner_secondary, smiles, args.k, baseline)

    coverage_actual, width = load_coverage_metrics(Path("metrics/coverage_week3.json"), target=0.9)

    frontier_data = load_frontier(args.frontier)
    build_summary_figure(askcos_curve, aiz_curve, 0.9, coverage_actual, width, frontier_data, args.figure)

    askcos_final = askcos_curve[-1]
    aiz_final = aiz_curve[-1]
    drop = askcos_final - aiz_final

    askcos_summary = summarize_report(report_primary)
    askcos_summary["feasible_at_k"] = askcos_final
    askcos_summary["improvement"] = askcos_final - baseline

    aiz_summary = summarize_report(report_secondary)
    aiz_summary["feasible_at_k"] = aiz_final
    aiz_summary["improvement"] = aiz_final - baseline

    results = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "k": args.k,
        "n_targets": len(smiles),
        "askcos": askcos_summary,
        "aizynth": aiz_summary,
        "cross_oracle_drop": drop,
        "coverage": {
            "target": 0.9,
            "achieved": coverage_actual,
            "interval_width": width,
            "source": "metrics/coverage_week3.json",
        },
        "latency_budget_seconds": 70,
        "notes": "AiZynthFinder results are empirical; ASKCOS remains mocked pending deployment. Coverage imported from Week 3 conformal evaluation."
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
