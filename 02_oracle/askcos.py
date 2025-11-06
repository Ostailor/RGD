"""ASKCOS oracle client."""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from dataclasses import replace
from typing import Deque, Dict, Iterable, List, Optional

import requests

from .base import OracleResult

logger = logging.getLogger(__name__)


class AskcosClient:
    """Lightweight HTTP client for the ASKCOS retrosynthesis API."""

    name = "askcos"

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        session: Optional[requests.Session] = None,
        timeout: float = 30.0,
        max_attempts: int = 3,
        backoff_factor: float = 1.5,
        max_calls_per_minute: Optional[int] = 60,
    ) -> None:
        self.base_url = base_url or os.environ.get("ASKCOS_BASE_URL", "").rstrip("/")
        if not self.base_url:
            raise ValueError(
                "ASKCOS_BASE_URL is not configured. Pass base_url explicitly or set the environment variable."
            )
        self.api_key = api_key or os.environ.get("ASKCOS_API_KEY")
        self._session = session or requests.Session()
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.backoff_factor = backoff_factor
        self.max_calls_per_minute = max_calls_per_minute
        self._timestamps: Deque[float] = deque()
        self._rate_window = 60.0
        self._cache: Dict[str, OracleResult] = {}

    def evaluate(self, smiles: Iterable[str]) -> List[OracleResult]:
        results: List[OracleResult] = []
        for smi in smiles:
            if smi in self._cache:
                cached = replace(self._cache[smi], cache_hit=True, timestamp=time.time())
                results.append(cached)
                continue
            start = time.perf_counter()
            attempts = 0
            payload = {"smiles": smi}
            headers = {"Accept": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            last_error: Optional[str] = None
            status_code: Optional[int] = None
            while attempts < self.max_attempts:
                attempts += 1
                self._respect_rate_limit()
                try:
                    response = self._session.post(
                        f"{self.base_url}/api/v2/retro/reaction",
                        json=payload,
                        headers=headers,
                        timeout=self.timeout,
                    )
                    status_code = getattr(response, "status_code", None)
                    if status_code is not None and status_code >= 400:
                        last_error = f"HTTP {status_code}: {getattr(response, 'text', '')[:200]}"
                        logger.warning(
                            "ASKCOS error",
                            extra={"smiles": smi, "status_code": status_code, "attempt": attempts},
                        )
                        time.sleep(self.backoff_factor ** attempts)
                        continue
                    data = response.json()
                    feasible = bool(data.get("feasible", data.get("success", False)))
                    score = data.get("score")
                    metadata = {
                        "route_length": data.get("route_length"),
                        "n_routes": data.get("n_routes"),
                        "status_code": status_code,
                        "attempts": attempts,
                    }
                    latency = time.perf_counter() - start
                    result = OracleResult(
                        smiles=smi,
                        feasible=feasible,
                        score=score,
                        latency=latency,
                        oracle=self.name,
                        raw=metadata,
                        status_code=status_code,
                        attempts=attempts,
                        cache_hit=False,
                    )
                    self._cache[smi] = result
                    results.append(result)
                    break
                except requests.RequestException as exc:
                    last_error = str(exc)
                    time.sleep(self.backoff_factor ** attempts)
                    logger.error(
                        "ASKCOS request exception",
                        extra={"smiles": smi, "error": last_error, "attempt": attempts},
                    )
            else:
                latency = time.perf_counter() - start
                result = OracleResult(
                    smiles=smi,
                    feasible=False,
                    score=None,
                    latency=latency,
                    oracle=self.name,
                    raw={"status_code": status_code},
                    error=last_error,
                    status_code=status_code,
                    attempts=attempts,
                    cache_hit=False,
                )
                self._cache[smi] = result
                results.append(result)
        return results

    def _respect_rate_limit(self) -> None:
        if not self.max_calls_per_minute:
            return
        now = time.monotonic()
        window_start = now - self._rate_window
        while self._timestamps and self._timestamps[0] < window_start:
            self._timestamps.popleft()
        if len(self._timestamps) >= self.max_calls_per_minute:
            sleep_time = self._rate_window - (now - self._timestamps[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
            now = time.monotonic()
            window_start = now - self._rate_window
            while self._timestamps and self._timestamps[0] < window_start:
                self._timestamps.popleft()
        self._timestamps.append(time.monotonic())
