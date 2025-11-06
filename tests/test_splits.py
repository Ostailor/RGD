from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

try:  # pandas may fail to import if binary deps missing
    import pandas as pd
except Exception as exc:  # noqa: BLE001
    pytest.skip(f"pandas import failed: {exc}", allow_module_level=True)

rdkit = pytest.importorskip("rdkit")


MODULE_PATH = Path(__file__).resolve().parents[1] / "00_data" / "make_splits.py"
SPEC = importlib.util.spec_from_file_location("rgd_make_splits", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_random_split_assignments_cover_all_rows():
    df = pd.DataFrame(
        {
            "SMILES": ["CCO", "CCN", "CCC", "COC", "CCCl", "CCF", "CCBr", "CC=O", "C#N", "CO"],
            "label": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    assignments = MODULE.assign_random_split(df, (0.6, 0.2, 0.2), seed=42, stratify_column="label")
    assert set(assignments.unique()) == {"train", "val", "test"}
    assert assignments.isna().sum() == 0
    assert assignments.index.equals(df.index)


def test_scaffold_split_has_no_leakage():
    df = pd.DataFrame(
        {
            "smiles": [
                "CCO",
                "CCCO",
                "CCN",
                "CCCN",
                "c1ccccc1",
                "c1ccncc1",
                "ClCCl",
                "ClCClCl",
                "BrCCBr",
                "BrCCN",
            ]
        }
    )
    assignments = MODULE.assign_scaffold_split(df, (0.5, 0.25, 0.25), smiles_column="smiles")
    summary = MODULE.summarize_split(df, assignments, smiles_column="smiles")
    assert summary["scaffold_overlap"]["train_val"] == 0
    assert summary["scaffold_overlap"]["train_test"] == 0
    assert summary["scaffold_overlap"]["val_test"] == 0


def test_time_split_respects_order():
    df = pd.DataFrame(
        {
            "SMILES": ["CCO", "CCN", "CCC", "COC", "CCCl"],
            "record_date": [
                "2020-01-01T00:00:00Z",
                "2020-06-01T00:00:00Z",
                "2021-01-01T00:00:00Z",
                "2021-06-01T00:00:00Z",
                "2022-01-01T00:00:00Z",
            ],
        }
    )
    assignments = MODULE.assign_time_split(df, "record_date", (0.4, 0.2, 0.4))
    # earliest dates should be in train
    assert assignments.iloc[0] == "train"
    assert assignments.iloc[1] == "train"
    # latest should be in test
    assert assignments.iloc[-1] == "test"


def test_write_split_outputs_json(tmp_path: Path):
    df = pd.DataFrame({"smiles": ["CCO", "CCN", "CCC"]})
    assignments = MODULE.assign_random_split(df, (0.6, 0.2, 0.2), seed=123)
    summary = MODULE.summarize_split(df, assignments, "smiles")
    MODULE.write_split(
        dataset="moses",
        split_name="random_split",
        assignments=assignments,
        summary=summary,
        output_dir=tmp_path / "processed",
        summary_dir=tmp_path / "reports",
    )
    split_path = tmp_path / "processed" / "moses" / "splits" / "random_split.json"
    summary_path = tmp_path / "reports" / "moses_random_split_summary.json"
    assert split_path.exists()
    assert summary_path.exists()
    split_payload = split_path.read_text()
    summary_payload = summary_path.read_text()
    assert '"dataset": "moses"' in split_payload
    assert '"split_name": "random_split"' in summary_payload
