#!/usr/bin/env python3
"""Generate a reproducibility snapshot capturing hashes of key artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Dict, List


DEFAULT_ARTIFACTS = [
    "metrics/week2_planner_metrics_real.json",
    "metrics/compute_frontier_week5.json",
    "metrics/ablation_week5.csv",
    "metrics/shift_week4.json",
    "metrics/moses_vae_week1.json",
    "metrics/guacamol_week1.json",
    "metrics/calibration_table.csv",
    "metrics/coverage_week3.json",
    "06_prospective/results.json",
    "figures/week3_reliability.png",
    "figures/week5_compute_frontier.png",
    "figures/week5_ablation.png",
    "figures/week6_summary.png",
    "logs/aizynth_week2.log",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_snapshot(artifacts: List[Path]) -> Dict[str, object]:
    records = []
    for artifact in artifacts:
        if not artifact.exists():
            continue
        records.append(
            {
                "path": artifact.as_posix(),
                "sha256": _sha256(artifact),
                "size_bytes": artifact.stat().st_size,
                "modified": artifact.stat().st_mtime,
            }
        )
    snapshot = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": socket.gethostname(),
        "artifacts": records,
    }
    return snapshot


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts",
        nargs="+",
        default=DEFAULT_ARTIFACTS,
        help="List of artifacts to hash (default: curated set of key metrics/figures).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("metrics/reproducibility_snapshot.json"),
        help="Where to write the JSON snapshot.",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("logs/reproducibility.log"),
        help="Optional append-only log capturing the snapshot metadata.",
    )
    args = parser.parse_args(argv)

    artifacts = [Path(p) for p in args.artifacts]
    snapshot = build_snapshot(artifacts)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(snapshot, indent=2))

    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        with args.log.open("a") as handle:
            handle.write(json.dumps(snapshot) + os.linesep)

    print(f"[INFO] Reproducibility snapshot written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
