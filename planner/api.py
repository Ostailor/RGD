"""Planner-in-the-loop API coordinating oracle calls and feasibility scoring."""

from __future__ import annotations

import math
from dataclasses import dataclass
import sys
from importlib import import_module, util
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List, Optional, Sequence

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

base_module = import_module("02_oracle.base")
logging_module = import_module("02_oracle.logging")

OracleClientProtocol = base_module.OracleClientProtocol
OracleResult = base_module.OracleResult
OracleLogger = logging_module.OracleLogger


@dataclass
class OracleStats:
    success_rate: float
    avg_latency: float
    p95_latency: float
    mean_score: float
    total: int
    feasible: int
    median_route_length: Optional[float]
    throughput_per_minute: float


@dataclass
class PlannerReport:
    scores: Dict[str, float]
    feasible_at_k: float
    baseline_feasible_at_k: float
    improvement: float
    oracle_stats: Dict[str, OracleStats]
    records: List[OracleResult]
    median_route_length: Optional[float]
    median_latency: float


class PlannerInLoop:
    """Coordinate several oracle clients and provide aggregate scoring."""

    def __init__(
        self,
        oracles: Sequence[OracleClientProtocol],
        logger: Optional[OracleLogger] = None,
    ) -> None:
        if not oracles:
            raise ValueError("At least one oracle client is required.")
        self.oracles = list(oracles)
        self.logger = logger or OracleLogger()

    def evaluate(
        self,
        smiles: Iterable[str],
        k: int = 10,
        baseline_feasible_at_k: Optional[float] = None,
    ) -> PlannerReport:
        """Evaluate a batch of SMILES strings and compute planner metrics."""
        smiles = list(smiles)
        oracle_results = self._evaluate_with_oracles(smiles)
        flat_results = [result for results in oracle_results.values() for result in results]
        self.logger.log(flat_results)

        aggregate_scores = self._aggregate_scores(oracle_results)
        top_k_feasible = self._feasible_at_k(oracle_results, aggregate_scores, k)
        baseline = baseline_feasible_at_k or self._baseline_feasible(smiles, k)
        improvement = top_k_feasible - baseline
        stats = self._oracle_statistics(flat_results)
        route_lengths = [
            entry.raw.get("route_length")
            for entry in flat_results
            if isinstance(entry.raw.get("route_length"), (int, float))
        ]
        median_route_length = float(median(route_lengths)) if route_lengths else None
        latencies = [entry.latency for entry in flat_results]
        median_latency = float(median(latencies)) if latencies else 0.0

        return PlannerReport(
            scores=aggregate_scores,
            feasible_at_k=top_k_feasible,
            baseline_feasible_at_k=baseline,
            improvement=improvement,
            oracle_stats=stats,
            records=flat_results,
            median_route_length=median_route_length,
            median_latency=median_latency,
        )

    def _evaluate_with_oracles(self, smiles: List[str]) -> Dict[str, List[OracleResult]]:
        results: Dict[str, List[OracleResult]] = {smi: [] for smi in smiles}
        for oracle in self.oracles:
            oracle_results = oracle.evaluate(smiles)
            for result in oracle_results:
                results[result.smiles].append(result)
        return results

    def _aggregate_scores(self, results: Dict[str, List[OracleResult]]) -> Dict[str, float]:
        aggregated: Dict[str, float] = {}
        for smi, entries in results.items():
            aggregated[smi] = self._composite_score(entries)
        return aggregated

    def _composite_score(self, entries: List[OracleResult]) -> float:
        if not entries:
            return 0.0
        raw_scores = [entry.score for entry in entries if entry.score is not None]
        base_score = float(sum(raw_scores) / len(raw_scores)) if raw_scores else 0.0
        feasibility_bonus = 0.2 if any(entry.feasible for entry in entries) else -0.1
        infeasible_penalty = 0.05 * sum(not entry.feasible for entry in entries)
        route_lengths = [
            entry.raw.get("route_length")
            for entry in entries
            if isinstance(entry.raw.get("route_length"), (int, float))
        ]
        route_penalty = 0.0
        if route_lengths:
            best_route = min(route_lengths)
            route_penalty = max(0.0, (best_route - 8) / 40.0)
        latency_penalty = 0.0
        latencies = [entry.latency for entry in entries]
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            latency_penalty = max(0.0, (avg_latency - 45.0) / 120.0)
        score = base_score + feasibility_bonus - infeasible_penalty - route_penalty - latency_penalty
        return max(0.0, min(1.0, score))

    def _feasible_at_k(
        self,
        results: Dict[str, List[OracleResult]],
        scores: Dict[str, float],
        k: int,
    ) -> float:
        sorted_smiles = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        top_k = [smi for smi, _ in sorted_smiles[:k]]
        feasibles = 0
        for smi in top_k:
            if any(entry.feasible for entry in results.get(smi, [])):
                feasibles += 1
        if k == 0:
            return 0.0
        return feasibles / k

    def _baseline_feasible(self, smiles: Sequence[str], k: int) -> float:
        """Simple heuristic baseline using molecular length as proxy."""
        scored = [(smi, self._length_score(smi)) for smi in smiles]
        top_k = [smi for smi, _ in sorted(scored, key=lambda item: item[1], reverse=True)[:k]]
        feasible = 0
        for smi in top_k:
            score = self._length_score(smi)
            has_rings = smi.count("c1") > 1
            hetero_atoms = smi.count("N") + smi.count("O") + smi.count("S")
            if score > 0.6 and not has_rings and hetero_atoms <= 1:
                feasible += 1
        return feasible / k if k else 0.0

    def _length_score(self, smiles: str) -> float:
        length = len(smiles)
        rings = smiles.count("1")
        return max(0.0, 1.0 - length / 120.0) * (1.0 - rings / 10.0)

    def _oracle_statistics(self, results: List[OracleResult]) -> Dict[str, OracleStats]:
        by_oracle: Dict[str, List[OracleResult]] = {}
        for result in results:
            by_oracle.setdefault(result.oracle, []).append(result)
        stats: Dict[str, OracleStats] = {}
        for oracle_name, entries in by_oracle.items():
            latencies = [entry.latency for entry in entries]
            latencies_sorted = sorted(latencies)
            p95_index = math.ceil(0.95 * len(latencies_sorted)) - 1
            p95_latency = latencies_sorted[max(p95_index, 0)]
            feasible = sum(entry.feasible for entry in entries)
            scores = [entry.score for entry in entries if entry.score is not None]
            mean_score = float(mean(scores)) if scores else 0.0
            route_lengths = [
                entry.raw.get("route_length")
                for entry in entries
                if isinstance(entry.raw.get("route_length"), (int, float))
            ]
            median_route = float(median(route_lengths)) if route_lengths else None
            total_latency = sum(latencies)
            throughput = 0.0
            if total_latency > 0:
                throughput = (len(entries) / total_latency) * 60.0
            stats[oracle_name] = OracleStats(
                success_rate=feasible / len(entries) if entries else 0.0,
                avg_latency=sum(latencies) / len(latencies) if latencies else 0.0,
                p95_latency=p95_latency,
                mean_score=mean_score,
                total=len(entries),
                feasible=feasible,
                median_route_length=median_route,
                throughput_per_minute=throughput,
            )
        return stats
