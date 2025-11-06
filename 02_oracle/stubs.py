"""Stubs and simulators used for local Week 2 development and testing."""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import requests


@dataclass
class MockAskcosResponse:
    """Lightweight stand-in for requests.Response."""

    payload: Dict[str, object]
    status_code: int

    def json(self) -> Dict[str, object]:
        return self.payload

    @property
    def text(self) -> str:
        return json.dumps(self.payload)


class MockAskcosSession:
    """Simulated ASKCOS session with controllable latency and failure rate.

    This class allows us to stress-test the planner without live credentials,
    while still exercising retry logic, latency logging, and throughput metrics.
    """

    def __init__(
        self,
        feasible_bias: float = 0.65,
        failure_rate: float = 0.05,
        avg_latency: float = 0.5,
        latency_jitter: float = 0.15,
        seed: Optional[int] = 13,
    ) -> None:
        self.feasible_bias = feasible_bias
        self.failure_rate = failure_rate
        self.avg_latency = avg_latency
        self.latency_jitter = latency_jitter
        self.rng = random.Random(seed)
        self.calls: List[str] = []

    def post(self, url: str, json: Dict[str, str], headers: Dict[str, str], timeout: float):
        smiles = json.get("smiles", "")
        self.calls.append(smiles)
        latency = max(0.0, self.rng.gauss(self.avg_latency, self.latency_jitter))
        time.sleep(min(latency, timeout))
        if self.rng.random() < self.failure_rate:
            raise requests.RequestException("Injected network failure")

        baseline = 1 / (1 + math.exp(-0.2 * (smiles.count("N") + smiles.count("O"))))
        length_penalty = max(0.0, len(smiles) - 40) / 80.0
        feasible_prob = min(0.95, max(0.1, self.feasible_bias + baseline / 4 - length_penalty))
        feasible = self.rng.random() < feasible_prob
        score = 0.2 + feasible_prob * 0.8
        route_length = max(1, math.ceil(len(smiles) / 12))
        n_routes = 1 if not feasible else 1 + int(self.rng.random() * 2)
        payload = {
            "smiles": smiles,
            "feasible": feasible,
            "score": round(score, 3),
            "route_length": route_length,
            "n_routes": n_routes,
            "latency_hint": latency,
        }
        return MockAskcosResponse(payload=payload, status_code=200)
