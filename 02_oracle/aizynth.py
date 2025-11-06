"""AiZynthFinder oracle client.

The implementation uses the public AiZynthFinder API when available. If the
library is not installed locally, the client can fall back to a lightweight
heuristic scorer so that the planner code paths remain testable.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .base import OracleResult

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional import
    from aizynthfinder.aizynthfinder import AiZynthFinder

    _AIZYNTH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _AIZYNTH_AVAILABLE = False


class AiZynthClient:
    """AiZynthFinder-based feasibility oracle."""

    name = "aizynth"

    def __init__(
        self,
        config: Optional[Dict[str, str]] = None,
        max_steps: int = 200,
        use_fallback: bool = True,
        config_path: str | Path = "config.yml",
        log_path: str | Path | None = None,
        restarts: int = 1,
    ) -> None:
        self.max_steps = max_steps
        self.config = config or {}
        self.use_fallback = use_fallback
        self.restarts = max(1, int(restarts))
        if not _AIZYNTH_AVAILABLE and not use_fallback:
            raise ImportError(
                "AiZynthFinder is not installed. Install `aizynthfinder` or enable the fallback."
            )
        self._cache: Dict[str, OracleResult] = {}
        self._config_path: Optional[Path] = None
        self._log_fh = None
        self._finder_pool: List["AiZynthFinder"] = []
        if log_path is not None:
            self._log_path = Path(log_path)
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_fh = self._log_path.open("a")
        else:
            self._log_path = None
        if _AIZYNTH_AVAILABLE and not self.use_fallback:
            self._config_path = Path(config_path)
            if not self._config_path.exists():
                raise FileNotFoundError(
                    f"AiZynthFinder config not found: {self._config_path}. "
                    "Please provide a valid YAML config."
                )
            for _ in range(self.restarts):
                finder = AiZynthFinder(configfile=str(self._config_path))
                self._ensure_policy_selection(finder)
                self._finder_pool.append(finder)

    def _ensure_policy_selection(self, finder: "AiZynthFinder") -> None:
        if not finder.expansion_policy.selection and finder.expansion_policy.items:
            finder.expansion_policy.select(finder.expansion_policy.items[0])
        if not finder.filter_policy.selection and finder.filter_policy.items:
            finder.filter_policy.select(finder.filter_policy.items)
        if not finder.stock.selection and finder.stock.items:
            finder.stock.select(finder.stock.items)

    def _log(self, message: str) -> None:
        if self._log_fh is not None:
            self._log_fh.write(message + "\n")
            self._log_fh.flush()

    def __del__(self) -> None:
        if self._log_fh is not None:
            try:
                self._log_fh.close()
            except Exception:
                pass

    def evaluate(self, smiles: Iterable[str]) -> List[OracleResult]:
        results: List[OracleResult] = []
        for smi in smiles:
            if smi in self._cache:
                cached = replace(self._cache[smi], cache_hit=True, timestamp=time.time())
                results.append(cached)
                continue
            start = time.perf_counter()
            try:
                if _AIZYNTH_AVAILABLE and not self.use_fallback:
                    result = self._evaluate_aizynth(smi)
                else:
                    result = self._evaluate_fallback(smi)
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.warning("AiZynth evaluation failed for %s: %s", smi, exc)
                self._log(f"[ERROR] {smi} :: {exc}")
                result = OracleResult(
                    smiles=smi,
                    feasible=False,
                    score=None,
                    latency=0.0,
                    oracle=self.name,
                    raw={"error": str(exc)},
                    error=str(exc),
                )
            result.latency = time.perf_counter() - start
            self._cache[smi] = result
            results.append(result)
            self._log(
                "[INFO] %s feasible=%s latency=%.3fs routes=%s"
                % (
                    smi,
                    result.feasible,
                    result.latency,
                    result.raw.get("num_routes") if result.raw else None,
                )
            )
        return results

    def _make_finder(self) -> "AiZynthFinder":
        if self._config_path is None:
            raise RuntimeError("AiZynthFinder config path is not set.")
        finder = AiZynthFinder(configfile=str(self._config_path))
        self._ensure_policy_selection(finder)
        return finder

    def _evaluate_aizynth(self, smiles: str) -> OracleResult:  # pragma: no cover - requires heavy deps
        best_result: Optional[OracleResult] = None
        best_score = float("-inf")
        while len(self._finder_pool) < self.restarts:
            self._finder_pool.append(self._make_finder())
        for attempt in range(self.restarts):
            finder = self._finder_pool[attempt]
            finder.target_smiles = smiles
            seed_value = abs(hash((smiles, attempt))) % (2**32)
            if hasattr(finder, "random_state"):
                try:
                    finder.random_state.seed(seed_value)
                except Exception:  # pragma: no cover - best effort seeding
                    pass
            finder.prepare_tree()
            search_time = finder.tree_search(show_progress=False)
            finder.build_routes()
            feasible = len(finder.routes) > 0
            stats = finder.extract_statistics()
            stats.update(
                {
                    "num_routes": len(finder.routes),
                    "solver": "aizynthfinder",
                    "restart_index": attempt,
                }
            )
            feasible = bool(stats.get("is_solved", feasible))
            score = 1.0 if feasible else 0.0
            result = OracleResult(
                smiles=smiles,
                feasible=feasible,
                score=score,
                latency=search_time,
                oracle=self.name,
                raw=stats,
                attempts=attempt + 1,
            )
            if score > best_score:
                best_score = score
                best_result = result
            elif score == best_score and best_result and result.latency < best_result.latency:
                best_result = result
            if hasattr(finder, "reset"):
                try:
                    finder.reset()
                except Exception:  # pragma: no cover - reset best effort
                    pass
        if best_result is None:
            raise RuntimeError("AiZynthFinder failed to return a result.")
        return best_result

    def _evaluate_fallback(self, smiles: str) -> OracleResult:
        """Heuristic feasibility proxy used when AiZynthFinder is unavailable."""
        length = len(smiles)
        ring_count = smiles.count("1")
        hetero = sum(smiles.count(atom) for atom in ["N", "O", "S"])
        feasible = length < 80 and ring_count <= 6
        score = max(0.0, 1.0 - (length / 120.0)) * (1.0 + hetero / 20.0)
        score = min(score, 1.0)
        route_length = max(1, math.floor(length / 10))
        stats = {
            "route_length": route_length,
            "hetero_atoms": hetero,
            "uses_fallback": True,
            "solver": "heuristic",
        }
        return OracleResult(
            smiles=smiles,
            feasible=feasible,
            score=score,
            latency=0.0,
            oracle=self.name,
            raw=stats,
            attempts=1,
        )
