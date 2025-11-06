#!/usr/bin/env python3
"""Baseline reproduction utilities for MOSES / GuacaMol benchmarks.

This module focuses on Week 0–1 requirements: providing deterministic scripts that
compute standard molecule generation metrics (validity, uniqueness, novelty, Fréchet-style)
for generated samples, while deferring expensive baseline training to later milestones.

Example usage:

    python train_baseline.py evaluate \
        --dataset moses \
        --reference data/moses_train.smi \
        --generated outputs/moses_vae/generated.smi \
        --report metrics/moses_vae_week1.json

The script also scaffolds `train` commands for canonical baselines (MOSES VAE / GuacaMol
SMILES-LSTM) via documented subprocess calls, so users can trigger the publish reference
pipelines once dependencies are installed.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")  # suppress RDKit warnings

# Core RDKit descriptor set used for Frechet-style distance.
FRECHET_DESCRIPTORS = [
    Descriptors.MolWt,
    Descriptors.MolLogP,
    Descriptors.NumRotatableBonds,
    Descriptors.TPSA,
    Descriptors.NumHDonors,
    Descriptors.NumHAcceptors,
]


@dataclass
class BaselineConfig:
    name: str
    train_command: Sequence[str]
    expected_output: Path
    notes: str


BASELINE_REGISTRY: Dict[str, BaselineConfig] = {
    "moses_vae": BaselineConfig(
        name="moses_vae",
        train_command=(
            "python",
            "-m",
            "moses.baselines.vae.train",
            "--config",
            "01_baselines/configs/moses_vae.yaml",
        ),
        expected_output=Path("outputs/moses_vae/generated.smi"),
        notes="Requires `pip install git+https://github.com/molecularsets/moses.git`.",
    ),
    "guacamol_smiles_lstm": BaselineConfig(
        name="guacamol_smiles_lstm",
        train_command=(
            "python",
            "-m",
            "guacamol_baselines.smiles_lstm.smiles_lstm_baseline",
            "--config",
            "01_baselines/configs/guacamol_smiles_lstm.json",
        ),
        expected_output=Path("outputs/guacamol_smiles_lstm/generated.smi"),
        notes="Requires `pip install guacamol` and baseline extras.",
    ),
}


def canonicalize_smiles(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def load_smiles(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"SMILES file not found: {path}")
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"No SMILES strings found in {path}")
    # Drop a one-column header commonly written as 'smiles'
    if lines and lines[0].lower() == "smiles":
        lines = lines[1:]
    return lines


def compute_validity(smiles_list: Iterable[str]) -> Dict[str, object]:
    valid = []
    invalid = []
    for s in smiles_list:
        canonical = canonicalize_smiles(s)
        if canonical is None:
            invalid.append(s)
        else:
            valid.append(canonical)
    total = len(valid) + len(invalid)
    return {
        "valid_fraction": len(valid) / total,
        "num_valid": len(valid),
        "num_invalid": len(invalid),
        "invalid_examples": invalid[:10],
        "canonical_smiles": valid,
    }


def compute_uniqueness(canonical_smiles: Iterable[str]) -> Dict[str, float]:
    smiles_list = list(canonical_smiles)
    unique = set(smiles_list)
    return {
        "unique_fraction": len(unique) / len(smiles_list),
        "num_unique": len(unique),
        "num_total": len(smiles_list),
    }


def compute_novelty(
    generated: Iterable[str],
    reference: Iterable[str],
) -> Dict[str, float]:
    generated_set = {s for s in generated if s}
    reference_set = {s for s in reference if s}
    novel = generated_set - reference_set
    return {
        "novel_fraction": len(novel) / len(generated_set),
        "num_novel": len(novel),
    }


def compute_descriptor_matrix(smiles_list: Iterable[str]) -> np.ndarray:
    vectors: List[List[float]] = []
    for s in smiles_list:
        if not s:
            continue
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        values = []
        for descriptor in FRECHET_DESCRIPTORS:
            try:
                values.append(float(descriptor(mol)))
            except Exception:
                values.append(float("nan"))
        vectors.append(values)
    if not vectors:
        raise ValueError("Descriptor matrix is empty; no valid molecules.")
    matrix = np.array(vectors, dtype=np.float64)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=1e5, neginf=-1e5)
    return matrix


def frechet_distance(matrix_ref: np.ndarray, matrix_model: np.ndarray) -> float:
    mu_ref = np.mean(matrix_ref, axis=0)
    mu_model = np.mean(matrix_model, axis=0)
    cov_ref = np.cov(matrix_ref, rowvar=False)
    cov_model = np.cov(matrix_model, rowvar=False)
    diff = mu_ref - mu_model
    cov_prod = cov_ref @ cov_model
    cov_prod = (cov_prod + cov_prod.T) / 2.0  # enforce symmetry
    eigvals, eigvecs = np.linalg.eigh(cov_prod)
    eigvals = np.clip(eigvals, 0.0, None)
    sqrt_cov_prod = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
    distance = diff @ diff + np.trace(cov_ref + cov_model - 2 * sqrt_cov_prod)
    return float(np.real(distance))


def compute_metrics(reference_smiles: List[str], generated_smiles: List[str]) -> Dict[str, object]:
    validity = compute_validity(generated_smiles)
    uniqueness = compute_uniqueness(validity["canonical_smiles"])
    novelty = compute_novelty(validity["canonical_smiles"], map(canonicalize_smiles, reference_smiles))
    descriptor_ref = compute_descriptor_matrix(map(canonicalize_smiles, reference_smiles))
    descriptor_gen = compute_descriptor_matrix(validity["canonical_smiles"])
    frechet = frechet_distance(descriptor_ref, descriptor_gen)
    return {
        "validity": {k: v for k, v in validity.items() if k != "canonical_smiles"},
        "uniqueness": uniqueness,
        "novelty": novelty,
        "frechet_descriptor_distance": frechet,
    }


def run_train_command(config: BaselineConfig, env: Optional[Dict[str, str]] = None) -> None:
    print(f"[INFO] Launching baseline training for {config.name}")
    print("[INFO] Command:", " ".join(config.train_command))
    try:
        subprocess.run(config.train_command, check=True, env=env)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"Failed to run training command for {config.name}. Ensure dependencies are installed.\n{config.notes}"
        ) from exc


def write_report(report_path: Path, payload: Dict[str, object]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2))
    print(f"[INFO] Metrics report written to {report_path}")


def cmd_train(args: argparse.Namespace) -> None:
    if args.baseline not in BASELINE_REGISTRY:
        raise SystemExit(f"Unknown baseline '{args.baseline}'. Available: {list(BASELINE_REGISTRY)}")
    config = BASELINE_REGISTRY[args.baseline]
    run_train_command(config)
    print(f"[INFO] Expected generated molecules at {config.expected_output}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    reference_smiles = load_smiles(Path(args.reference))
    generated_smiles = load_smiles(Path(args.generated))
    metrics = compute_metrics(reference_smiles, generated_smiles)
    payload = {
        "dataset": args.dataset,
        "reference": args.reference,
        "generated": args.generated,
        "metrics": metrics,
        "comment": "Baseline Week0-1 metrics; refer to logs for training provenance.",
    }
    write_report(Path(args.report), payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Week 0–1 baseline scaffolding.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Launch baseline training command.")
    train_parser.add_argument("--baseline", required=True, choices=sorted(BASELINE_REGISTRY.keys()))
    train_parser.set_defaults(func=cmd_train)

    eval_parser = subparsers.add_parser("evaluate", help="Compute metrics for generated molecules.")
    eval_parser.add_argument("--dataset", required=True, choices={"moses", "guacamol"})
    eval_parser.add_argument("--reference", required=True, help="Reference SMILES file (one per line).")
    eval_parser.add_argument("--generated", required=True, help="Baseline generated SMILES file.")
    eval_parser.add_argument("--report", required=True, help="Output JSON report path.")
    eval_parser.set_defaults(func=cmd_evaluate)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
