from __future__ import annotations

import sys
from importlib import import_module, util
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _ensure_oracle_package() -> None:
    if "02_oracle" in sys.modules:
        return
    pkg_path = Path(__file__).resolve().parents[1] / "02_oracle"
    spec = util.spec_from_file_location(
        "02_oracle",
        pkg_path / "__init__.py",
        submodule_search_locations=[str(pkg_path)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to initialise 02_oracle package")
    module = util.module_from_spec(spec)
    sys.modules["02_oracle"] = module
    spec.loader.exec_module(module)


_ensure_oracle_package()

AskcosClient = import_module("02_oracle.askcos").AskcosClient
AiZynthClient = import_module("02_oracle.aizynth").AiZynthClient
OracleLogger = import_module("02_oracle.logging").OracleLogger
MockAskcosSession = import_module("02_oracle.stubs").MockAskcosSession
from planner import PlannerInLoop
from planner.api import PlannerReport


def test_askcos_client_caches_and_logs(tmp_path):
    session = MockAskcosSession(seed=7, avg_latency=0.01, latency_jitter=0.0)
    client = AskcosClient(
        base_url="https://askcos.test",
        session=session,
        max_calls_per_minute=None,
    )
    results = client.evaluate(["CCO", "N#N", "CCO"])
    assert len(results) == 3
    assert results[0].smiles == "CCO"
    assert results[1].smiles == "N#N"
    assert results[2].cache_hit is True
    # second CCO should hit cache (one HTTP call)
    assert session.calls.count("CCO") == 1
    assert results[0].status_code == 200
    assert results[0].attempts == 1

    log_path = tmp_path / "oracle.csv"
    logger = OracleLogger(log_path)
    logger.log(results)
    df = pd.read_csv(log_path)
    assert set(df["smiles"]) == {"CCO", "N#N"}
    assert "status_code" in df.columns


def test_aizynth_client_fallback():
    client = AiZynthClient(use_fallback=True)
    results = client.evaluate(["CCO", "CCCCCCCCCCCC", "c1ccccc1"])
    assert len(results) == 3
    assert any(res.feasible for res in results)
    assert all(res.oracle == "aizynth" for res in results)


def test_planner_report(tmp_path, monkeypatch):
    session = MockAskcosSession(seed=11, avg_latency=0.01, latency_jitter=0.0)
    askcos = AskcosClient(base_url="https://askcos.test", session=session)
    aizynth = AiZynthClient(use_fallback=True)
    logger = OracleLogger(tmp_path / "calls.parquet")
    planner = PlannerInLoop([askcos, aizynth], logger=logger)
    report = planner.evaluate(["CCO", "N#N", "c1ccccc1"], k=2)
    assert isinstance(report, PlannerReport)
    assert 0.0 <= report.feasible_at_k <= 1.0
    assert report.oracle_stats["askcos"].total == 3
    assert report.median_route_length is not None
    parquet_path = tmp_path / "calls.parquet"
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    else:
        fallback = parquet_path.with_suffix(".jsonl")
        df = pd.read_json(fallback, lines=True)
    assert len(df) == 6
    assert "raw_route_length" in df.columns
