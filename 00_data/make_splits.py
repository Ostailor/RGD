#!/usr/bin/env python3
"""Generate dataset splits (random, Bemis–Murcko scaffold, time-based) with audit logs.

Usage examples
--------------
python make_splits.py --dataset moses --input data/moses.csv --split-mode scaffold
python make_splits.py --dataset guacamol --input data/guacamol.csv --split-mode random --seed 2024
python make_splits.py --dataset moses --input data/moses.csv --split-mode all --timestamp-column record_date

The script writes split assignments to `processed/<dataset>/splits/<split_name>.json` and
summaries (coverage, leakage checks, scaffold stats) to `reports/<dataset>_<split_name>_summary.json`.

This script requires `pandas` and `rdkit`.  Install via:
  pip install pandas rdkit-pypi
Some splits (time-based) require the input to include a timestamp column in ISO format.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
except ImportError as exc:  # pragma: no cover - handled in runtime validation
    raise SystemExit(
        "RDKit is required for scaffold splitting. Install via `pip install rdkit-pypi`."
    ) from exc


DEFAULT_SPLIT_RATIOS = (0.8, 0.1, 0.1)
VALID_SPLIT_MODES = {"random", "scaffold", "time", "all"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        required=True,
        choices={"moses", "guacamol"},
        help="Dataset identifier used for output folder naming.",
    )
    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="Path to CSV file containing at minimum a SMILES column.",
    )
    parser.add_argument(
        "--split-mode",
        default="all",
        choices=VALID_SPLIT_MODES,
        help="Which split(s) to generate.",
    )
    parser.add_argument(
        "--timestamp-column",
        default=None,
        type=str,
        help="Name of column containing ISO timestamps for time-based split.",
    )
    parser.add_argument(
        "--smiles-column",
        default=None,
        type=str,
        help="Override SMILES column name (default depends on dataset).",
    )
    parser.add_argument(
        "--ratios",
        nargs=3,
        default=DEFAULT_SPLIT_RATIOS,
        type=float,
        metavar=("TRAIN", "VAL", "TEST"),
        help="Train/validation/test proportions that sum to 1.0.",
    )
    parser.add_argument(
        "--seed",
        default=1337,
        type=int,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--output-dir",
        default="processed",
        type=str,
        help="Directory to save split JSON files.",
    )
    parser.add_argument(
        "--summary-dir",
        default="reports",
        type=str,
        help="Directory to save split summary reports.",
    )
    return parser.parse_args()


def get_smiles_column(dataset: str, override: Optional[str], columns: Iterable[str]) -> str:
    if override:
        if override not in columns:
            raise ValueError(f"Provided SMILES column '{override}' not found in columns {columns}")
        return override
    default_map = {"moses": "SMILES", "guacamol": "smiles"}
    candidate = default_map.get(dataset.lower())
    if candidate and candidate in columns:
        return candidate
    if "smiles" in columns:
        return "smiles"
    if "SMILES" in columns:
        return "SMILES"
    raise ValueError(f"Could not infer SMILES column for dataset '{dataset}'. Columns: {columns}")


def validate_ratios(ratios: Tuple[float, float, float]) -> Tuple[float, float, float]:
    total = sum(ratios)
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError(f"Ratios must sum to 1.0, got {ratios} (sum={total})")
    if any(r <= 0 for r in ratios):
        raise ValueError(f"Ratios must be positive, got {ratios}")
    return ratios


def assign_random_split(
    df: pd.DataFrame,
    ratios: Tuple[float, float, float],
    seed: int,
    stratify_column: Optional[str] = None,
) -> pd.Series:
    rng = np.random.default_rng(seed)
    if stratify_column and stratify_column in df.columns:
        groups = df.groupby(stratify_column, sort=False)
        assignments = pd.Series(index=df.index, dtype="object")
        for _, group in groups:
            assignments.loc[group.index] = assign_random_split(group, ratios, seed=rng.integers(1_000_000))
        return assignments
    shuffled_indices = df.sample(frac=1.0, random_state=seed).index
    n = len(df)
    n_train = int(ratios[0] * n)
    n_val = int(ratios[1] * n)
    assignments = pd.Series(index=df.index, dtype="object")
    assignments.loc[shuffled_indices[:n_train]] = "train"
    assignments.loc[shuffled_indices[n_train : n_train + n_val]] = "val"
    assignments.loc[shuffled_indices[n_train + n_val :]] = "test"
    return assignments


def canonicalize_smiles(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)


def bemis_murcko_scaffold(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold is None:
        return None
    return Chem.MolToSmiles(scaffold, isomericSmiles=True)


def assign_scaffold_split(
    df: pd.DataFrame,
    ratios: Tuple[float, float, float],
    smiles_column: str,
) -> pd.Series:
    scaffolds: Dict[str, List[int]] = defaultdict(list)
    for idx, smiles in df[smiles_column].items():
        scaffold = bemis_murcko_scaffold(smiles)
        if scaffold is None:
            scaffold = f"invalid_{idx}"
        scaffolds[scaffold].append(idx)

    # Sort scaffolds by descending size to balance splits
    scaffold_items = sorted(scaffolds.items(), key=lambda kv: len(kv[1]), reverse=True)
    n_total = len(df)
    targets = [ratio * n_total for ratio in ratios]
    current_counts = [0.0, 0.0, 0.0]
    assignments = pd.Series(index=df.index, dtype="object")

    for scaffold, indices in scaffold_items:
        split_idx = np.argmin([current_counts[i] / targets[i] if targets[i] > 0 else float("inf") for i in range(3)])
        split_name = ["train", "val", "test"][split_idx]
        assignments.loc[indices] = split_name
        current_counts[split_idx] += len(indices)

    return assignments


def assign_time_split(
    df: pd.DataFrame,
    timestamp_column: str,
    ratios: Tuple[float, float, float],
) -> pd.Series:
    if timestamp_column not in df.columns:
        raise ValueError(f"Timestamp column '{timestamp_column}' not in dataframe columns {df.columns}")
    parsed_ts = pd.to_datetime(df[timestamp_column], utc=True, errors="coerce")
    if parsed_ts.isna().any():
        invalid = df.loc[parsed_ts.isna(), timestamp_column].unique()
        raise ValueError(f"Timestamp column '{timestamp_column}' contains non-parseable values: {invalid[:5]}")
    sorted_df = df.assign(_timestamp=parsed_ts).sort_values("_timestamp")
    n = len(sorted_df)
    n_train = int(ratios[0] * n)
    n_val = int(ratios[1] * n)
    assignments = pd.Series(index=df.index, dtype="object")
    assignments.loc[sorted_df.index[:n_train]] = "train"
    assignments.loc[sorted_df.index[n_train : n_train + n_val]] = "val"
    assignments.loc[sorted_df.index[n_train + n_val :]] = "test"
    return assignments


def summarize_split(df: pd.DataFrame, assignments: pd.Series, smiles_column: str) -> Dict[str, object]:
    counts = assignments.value_counts().to_dict()
    coverage = {split: count / len(df) for split, count in counts.items()}
    scaffolds_by_split: Dict[str, set] = defaultdict(set)
    invalid_smiles = 0
    for idx, split in assignments.items():
        smiles = df.at[idx, smiles_column]
        canonical = canonicalize_smiles(smiles)
        if canonical is None:
            invalid_smiles += 1
            scaffold = f"invalid_{idx}"
        else:
            scaffold = bemis_murcko_scaffold(canonical) or f"degenerate_{idx}"
        scaffolds_by_split[split].add(scaffold)
    scaffold_overlap = {}
    splits = ["train", "val", "test"]
    for i, a in enumerate(splits):
        for b in splits[i + 1 :]:
            overlap = scaffolds_by_split[a] & scaffolds_by_split[b]
            scaffold_overlap[f"{a}_{b}"] = len(overlap)
    return {
        "counts": counts,
        "coverage": coverage,
        "scaffold_overlap": scaffold_overlap,
        "invalid_smiles": invalid_smiles,
    }


def make_output_paths(output_dir: Path, dataset: str, split_name: str) -> Tuple[Path, Path]:
    split_dir = output_dir / dataset / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    split_path = split_dir / f"{split_name}.json"
    return split_dir, split_path


def write_split(
    dataset: str,
    split_name: str,
    assignments: pd.Series,
    summary: Dict[str, object],
    output_dir: Path,
    summary_dir: Path,
) -> None:
    split_dir, split_path = make_output_paths(output_dir, dataset, split_name)
    metadata = {
        "dataset": dataset,
        "split_name": split_name,
        "generated_at": datetime.utcnow().isoformat(),
        "commit": os.environ.get("GIT_COMMIT", "unknown"),
        "version": "week0_1",
    }
    payload = {
        "metadata": metadata,
        "assignments": assignments.to_dict(),
        "summary": summary,
    }
    split_path.write_text(json.dumps(payload, indent=2))

    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f"{dataset}_{split_name}_summary.json"
    summary_payload = {
        **metadata,
        "summary": summary,
        "n_total": len(assignments),
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2))

    print(f"[INFO] Wrote split to {split_path}")
    print(f"[INFO] Summary saved to {summary_path}")


def main() -> None:
    args = parse_args()
    ratios = validate_ratios(tuple(map(float, args.ratios)))
    df = pd.read_csv(args.input)
    smiles_column = get_smiles_column(args.dataset, args.smiles_column, df.columns)
    output_dir = Path(args.output_dir)
    summary_dir = Path(args.summary_dir)

    requested_modes = (
        {"random", "scaffold", "time"} if args.split_mode == "all" else {args.split_mode}
    )
    for mode in requested_modes:
        if mode == "random":
            assignments = assign_random_split(df, ratios, seed=args.seed)
        elif mode == "scaffold":
            assignments = assign_scaffold_split(df, ratios, smiles_column)
        elif mode == "time":
            if not args.timestamp_column:
                raise ValueError("Time-based split requested but --timestamp-column was not provided.")
            assignments = assign_time_split(df, args.timestamp_column, ratios)
        else:  # pragma: no cover
            raise ValueError(f"Unsupported split mode {mode}")

        summary = summarize_split(df, assignments, smiles_column)
        if mode == "scaffold":
            for key, overlap in summary["scaffold_overlap"].items():
                if overlap != 0:
                    raise RuntimeError(
                        f"Scaffold leakage detected between splits {key}: {overlap} overlapping scaffolds."
                    )
        split_name = f"{mode}_split"
        write_split(args.dataset, split_name, assignments, summary, output_dir, summary_dir)


if __name__ == "__main__":
    main()
