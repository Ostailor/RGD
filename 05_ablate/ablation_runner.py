#!/usr/bin/env python3
"""Summarise empirical component impacts for Week 5."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt


@dataclass
class ComponentRecord:
    component: str
    metric: str
    with_component: float
    reference: Optional[float]
    delta: Optional[float]
    notes: str
    source: str


def load_planner_record(path: Path) -> ComponentRecord:
    payload = json.loads(path.read_text())
    with_component = float(payload["feasible_at_k"])
    reference = float(payload.get("baseline_feasible_at_k", payload.get("baseline", {}).get("feasible_at_k", 0.0)))
    delta = with_component - reference
    notes = "Planner-in-the-loop vs heuristic baseline (AiZynthFinder real run, k=10)."
    return ComponentRecord(
        component="planner",
        metric="feasible_at_10",
        with_component=with_component,
        reference=reference,
        delta=delta,
        notes=notes,
        source=str(path),
    )


def load_conformal_record(path: Path, target: float = 0.9) -> ComponentRecord:
    payload = json.loads(path.read_text())
    conformal = next(entry for entry in payload["conformal"] if abs(entry["target"] - target) < 1e-6)
    naive = next(entry for entry in payload["naive"] if abs(entry["target"] - target) < 1e-6)
    with_component = float(conformal["coverage"])
    reference = float(naive["coverage"])
    delta = with_component - reference
    notes = "Calibrated conformal coverage compared with naive intervals on MOSES validation set."
    return ComponentRecord(
        component="conformal",
        metric=f"coverage@{target:.2f}",
        with_component=with_component,
        reference=reference,
        delta=delta,
        notes=notes,
        source=str(path),
    )


def load_shift_record(path: Path) -> List[ComponentRecord]:
    payload = json.loads(path.read_text())
    cov = payload["covariate_shift"]
    lab = payload["label_shift"]
    seq = payload["sequential_monitoring"]
    records = [
        ComponentRecord(
            component="shift",
            metric="mmd_p_value",
            with_component=float(cov["p_value"]),
            reference=None,
            delta=None,
            notes="Permutation MMD detecting covariate drift (lower is better).",
            source=str(path),
        ),
        ComponentRecord(
            component="shift",
            metric="bbse_mae",
            with_component=float(lab["mae"]),
            reference=None,
            delta=None,
            notes="Black-box shift estimation mean absolute error after reweighting.",
            source=str(path),
        ),
        ComponentRecord(
            component="shift",
            metric="sequential_false_alarm",
            with_component=float(seq["false_alarm_rate"]),
            reference=None,
            delta=None,
            notes="Anytime-valid monitoring false-alarm rate (target 0.05).",
            source=str(path),
        ),
    ]
    return records


def write_csv(records: List[ComponentRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "component",
                "metric",
                "with_component",
                "reference",
                "delta",
                "notes",
                "source",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "component": record.component,
                    "metric": record.metric,
                    "with_component": f"{record.with_component:.6f}",
                    "reference": "" if record.reference is None else f"{record.reference:.6f}",
                    "delta": "" if record.delta is None else f"{record.delta:.6f}",
                    "notes": record.notes,
                    "source": record.source,
                }
            )


def plot_deltas(records: List[ComponentRecord], figure: Path) -> None:
    deltas = [(rec.component, rec.delta) for rec in records if rec.delta is not None]
    if not deltas:
        return
    labels, values = zip(*deltas)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values, color="#4c72b0")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Delta")
    ax.set_title("Component impact (empirical deltas)")
    for idx, value in enumerate(values):
        ax.text(idx, value, f"{value:+.2f}", ha="center", va="bottom" if value >= 0 else "top")
    fig.tight_layout()
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("metrics/ablation_week5.csv"))
    parser.add_argument("--figure", type=Path, default=Path("figures/week5_ablation.png"))
    parser.add_argument(
        "--planner-metrics", type=Path, default=Path("metrics/week2_planner_metrics_real.json")
    )
    parser.add_argument(
        "--coverage-metrics", type=Path, default=Path("metrics/coverage_week3.json")
    )
    parser.add_argument(
        "--shift-metrics", type=Path, default=Path("metrics/shift_week4.json")
    )
    args = parser.parse_args()

    records: List[ComponentRecord] = []
    records.append(load_planner_record(args.planner_metrics))
    records.append(load_conformal_record(args.coverage_metrics))
    records.extend(load_shift_record(args.shift_metrics))

    write_csv(records, args.csv)
    plot_deltas(records, args.figure)


if __name__ == "__main__":
    main()
