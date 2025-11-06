#!/usr/bin/env python3
"""Sample molecules from the GuacaMol SMILES-LSTM baseline and write to outputs.

This helper assumes the GuacaMol baselines repository is checked out (as done
for Week 0–1) and the pretrained `model_final_0.473.pt` weights are available.
It mirrors the sampling routine used inside the GuacaMol benchmarking suite but
exposes a simple CLI so we can produce `generated.smi` directly.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def add_baseline_to_path(baseline_dir: Path) -> None:
    sys.path.insert(0, str(baseline_dir))


def load_generator(model_path: Path, device: str):
    from smiles_lstm_hc.rnn_utils import load_rnn_model  # type: ignore  # noqa: WPS433
    from smiles_lstm_hc.smiles_rnn_generator import SmilesRnnGenerator  # type: ignore  # noqa: WPS433

    model_def = model_path.with_suffix(".json")
    model = load_rnn_model(str(model_def), str(model_path), device, copy_to_cpu=True)
    return SmilesRnnGenerator(model=model, device=device)


def write_smiles(smiles: list[str], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w") as fh:
        for token in smiles:
            fh.write(f"{token}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-dir",
        default="guacamol_baselines",
        help="Path to the cloned guacamol_baselines repository.",
    )
    parser.add_argument(
        "--model-path",
        default="guacamol_baselines/smiles_lstm_hc/pretrained_model/model_final_0.473.pt",
        help="Path to the pretrained model weights.",
    )
    parser.add_argument(
        "--num",
        default=30000,
        type=int,
        help="Number of molecules to sample.",
    )
    parser.add_argument(
        "--output",
        default="outputs/guacamol_smiles_lstm/generated.smi",
        help="Destination .smi file.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device override (defaults to cuda if available else cpu).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline_dir = Path(args.baseline_dir).resolve()
    if not baseline_dir.exists():
        raise SystemExit(f"Baseline directory not found: {baseline_dir}")

    add_baseline_to_path(baseline_dir)

    model_path = Path(args.model_path).resolve()
    if not model_path.exists():
        raise SystemExit(f"Model weights not found: {model_path}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    generator = load_generator(model_path, device=device)
    smiles = generator.generate(args.num)
    dest = Path(args.output).resolve()
    write_smiles(smiles, dest)
    print(f"[INFO] Wrote {len(smiles)} GuacaMol baseline molecules to {dest}")


if __name__ == "__main__":
    main()
