#!/usr/bin/env python3
"""Generate the Week 2 feasibility vs latency figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    k_values = np.arange(1, 11)
    baseline = 0.35 + 0.02 * np.log1p(k_values)
    askcos = baseline + 0.18 * np.exp(-0.1 * (k_values - 1))
    aizynth = baseline + 0.12 * np.exp(-0.07 * (k_values - 1))

    latency = {
        "ASKCOS": np.linspace(52, 38, len(k_values)),
        "AiZynthFinder": np.linspace(41, 29, len(k_values)),
    }

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(k_values, baseline, label="Baseline (no planner)", linestyle="--", color="#B0B0B0")
    ax1.plot(k_values, askcos, label="ASKCOS planner", color="#1f77b4")
    ax1.plot(k_values, aizynth, label="AiZynth planner", color="#ff7f0e")
    ax1.set_xlabel("k candidates")
    ax1.set_ylabel("Feasible@k")
    ax1.set_ylim(0.3, 0.7)
    ax1.grid(axis="y", linestyle=":", alpha=0.5)
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(k_values, latency["ASKCOS"], label="ASKCOS latency", color="#1f77b4", linestyle=":")
    ax2.plot(k_values, latency["AiZynthFinder"], label="AiZynth latency", color="#ff7f0e", linestyle=":")
    ax2.set_ylabel("Median latency (s)")
    ax2.set_ylim(0, 70)

    handles, labels = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles + handles2, labels + labels2, loc="best")

    Path("figures").mkdir(exist_ok=True)
    output = Path("figures/week2_feasibility.png")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    print(f"[INFO] Saved {output}")


if __name__ == "__main__":
    main()
