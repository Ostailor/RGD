"""Common dataclasses and helper utilities for oracle integrations."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Protocol


@dataclass
class OracleResult:
    """Structured response from an oracle call."""

    smiles: str
    feasible: bool
    score: Optional[float]
    latency: float
    oracle: str
    raw: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    timestamp: float = field(default_factory=lambda: time.time())
    status_code: Optional[int] = None
    attempts: int = 1
    cache_hit: bool = False

    def to_record(self) -> Dict[str, Any]:
        """Flatten into a record for logging."""
        record = {
            "timestamp": self.timestamp,
            "smiles": self.smiles,
            "oracle": self.oracle,
            "feasible": self.feasible,
            "score": self.score,
            "latency": self.latency,
            "error": self.error,
            "status_code": self.status_code,
            "attempts": self.attempts,
            "cache_hit": self.cache_hit,
        }
        record.update({f"raw_{key}": value for key, value in self.raw.items()})
        return record


class OracleClientProtocol(Protocol):
    """Protocol describing the minimal interface required by the planner."""

    name: str

    def evaluate(self, smiles: Iterable[str]) -> List[OracleResult]:
        """Evaluate a batch of SMILES strings and return oracle results."""
        ...
