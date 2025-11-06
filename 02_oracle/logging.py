"""Logging helpers for oracle calls."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from .base import OracleResult


class OracleLogger:
    """Persist oracle call metadata to disk."""

    def __init__(self, path: str | Path = "logs/oracle_calls.parquet") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, results: Iterable[OracleResult]) -> None:
        """Append results to the log file."""
        records = [result.to_record() for result in results]
        if not records:
            return
        df = pd.DataFrame(records)
        if self.path.suffix == ".parquet":
            self._append_parquet(df)
        else:
            self._append_csv(df)

    def _append_parquet(self, df: pd.DataFrame) -> None:
        if self.path.exists():
            # append by concatenating and rewriting once to keep implementation simple
            existing = pd.read_parquet(self.path)
            df = pd.concat([existing, df], ignore_index=True)
        try:
            df.to_parquet(self.path, index=False)
        except (ImportError, ValueError):
            # Fall back to JSON lines if pyarrow is unavailable
            fallback = self.path.with_suffix(".jsonl")
            self._write_jsonl(df, fallback)

    def _append_csv(self, df: pd.DataFrame) -> None:
        df.to_csv(self.path, mode="a", header=not self.path.exists(), index=False)

    def _write_jsonl(self, df: pd.DataFrame, path: Path) -> None:
        with path.open("a") as handle:
            for record in df.to_dict(orient="records"):
                handle.write(json.dumps(record) + os.linesep)
