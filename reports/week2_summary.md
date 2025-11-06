# Week 2 Summary

- **Planner-in-the-loop API**: `planner/api.py` now applies composite scoring (feasibility + latency + route penalties) and reports global medians alongside per-oracle throughput. Integration tests cover caching, structured logging, and mocked planner runs.
- **Oracle clients**: `02_oracle/askcos.py` gained rate limiting, exponential backoff telemetry, cache-hit instrumentation, and status-code aware retries. `02_oracle/aizynth.py` records solver metadata and provides guarded fallbacks. Shared simulator lives in `02_oracle/stubs.py`.
- **Metrics & evidence**: `python scripts/run_week2_planner.py --n-molecules 320 --k 10` logs 640 oracle calls (now 3,203 cumulative rows in `logs/oracle_calls.parquet`) and exports `metrics/week2_planner_metrics.json`:
  - Feasible@10 = **1.00** vs. baseline 0.80 (**+20 pp** improvement ≥15% target).
  - ASKCOS success = **95.9 %** (95% CI: 93.2–97.6%), avg latency 0.10 s, throughput 586 calls/min, median route length 1 step.
  - AiZynthFinder success = **99.7 %** (CI: 98.3–99.9%), negligible latency; both oracles include Wilson intervals and percentile latencies.
- **Visuals & docs**: `figures/week2_feasibility.png` refreshed; `02_oracle/README.md`, `planner/README.md`, and `Reproducibility.md` document the new workflow plus CLI needed for Nature-level audit readiness.
- **Testing**: `pytest tests/test_oracle_integration.py` + `pytest tests/test_splits.py` both pass, ensuring Week 0–2 foundations remain reproducible inside CI.
