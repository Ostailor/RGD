#!/usr/bin/env python3
"""Aggregate empirical compute frontier statistics from solver logs."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np


LOG_PATTERN = re.compile(
    r"feasible=(True|False).*?latency=(?P<latency>[0-9.]+)s.*?routes=(?P<routes>[0-9]+)"
)


@dataclass
class RunStats:
    name: str
    log_path: Path
    molecules: int
    feasible: int
    routes: List[int]
    latencies: List[float]

    @property
    def success_rate(self) -> float:
        return self.feasible / self.molecules if self.molecules else 0.0

    @property
    def total_latency(self) -> float:
        return float(np.sum(self.latencies)) if self.latencies else 0.0

    @property
    def median_latency(self) -> float:
        return float(np.median(self.latencies)) if self.latencies else 0.0

    @property
    def mean_routes(self) -> float:
        return float(np.mean(self.routes)) if self.routes else 0.0


def parse_log(name: str, path: Path) -> RunStats:
    feasible = 0
    latencies: List[float] = []
    routes: List[int] = []

    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {path}")

    with path.open() as handle:
        for line in handle:
            match = LOG_PATTERN.search(line)
            if not match:
                continue
            if "feasible=True" in line:
                feasible += 1
            latencies.append(float(match.group("latency")))
            routes.append(int(match.group("routes")))

    molecules = len(latencies)
    return RunStats(
        name=name,
        log_path=path,
        molecules=molecules,
        feasible=feasible,
        routes=routes,
        latencies=latencies,
    )


def default_runs() -> List[tuple[str, Path]]:
    return [
        ("baseline", Path("logs/aizynth_week2_baseline.log")),
        ("dev", Path("logs/dev_week2.log")),
        ("full", Path("logs/aizynth_week2.log")),
    ]


def collect_stats(run_specs: Iterable[tuple[str, Path]]) -> List[RunStats]:
    stats: List[RunStats] = []
    for name, path in run_specs:
        if not path.exists():
            continue
        stats.append(parse_log(name, path))
    if not stats:
        raise RuntimeError("No valid run logs found; cannot build compute frontier.")
    return stats


def save_metrics(stats: List[RunStats], output: Path) -> None:
    records = []
    for entry in stats:
        records.append(
            {
                "run": entry.name,
                "log_path": entry.log_path.as_posix(),
                "molecules": entry.molecules,
                "feasible": entry.feasible,
                "success_rate": entry.success_rate,
                "total_latency_seconds": entry.total_latency,
                "median_latency_seconds": entry.median_latency,
                "mean_routes": entry.mean_routes,
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, indent=2))


def plot_frontier(stats: List[RunStats], figure_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for entry in stats:
        total_minutes = entry.total_latency / 60.0
        ax.scatter(total_minutes, entry.success_rate, s=80, label=entry.name.capitalize())
        ax.annotate(
            f"{entry.success_rate:.2f}",
            (total_minutes, entry.success_rate),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
        )
    ax.set_xlabel("Total solver time (minutes)")
    ax.set_ylabel("Feasible@k success rate")
    ax.set_title("Empirical compute frontier (AiZynthFinder)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)


def parse_run_args(items: Iterable[str]) -> List[tuple[str, Path]]:
    runs: List[tuple[str, Path]] = []
    for item in items:
        if ":" not in item:
            raise argparse.ArgumentTypeError(
                f"Invalid run specification '{item}'. Expected format 'label:path/to/log'."
            )
        label, path = item.split(":", 1)
        runs.append((label, Path(path)))
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        dest="runs",
        action="append",
        default=[],
        help="Run specification as label:/path/to/log. Repeat for multiple runs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("metrics/compute_frontier_week5.json"),
        help="Where to store the JSON metrics.",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("figures/week5_compute_frontier.png"),
        help="Where to store the frontier plot (PNG).",
    )
    args = parser.parse_args()

    run_specs = parse_run_args(args.runs) if args.runs else default_runs()
    stats = collect_stats(run_specs)
    stats.sort(key=lambda item: item.total_latency)

    save_metrics(stats, args.output)
    plot_frontier(stats, args.figure)

    for entry in stats:
        print(
            f"[INFO] {entry.name}: success={entry.success_rate:.3f}, "
            f"total_latency={entry.total_latency:.1f}s ({entry.total_latency/60:.1f} min), "
            f"molecules={entry.molecules}"
        )


if __name__ == "__main__":
    main()
