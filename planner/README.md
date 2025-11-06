# Planner API

The planner coordinates one or more oracle clients and produces aggregate
feasibility scores. Key capabilities:

- `planner.api.PlannerInLoop` — orchestrates oracle evaluation, computes
  feasible@k improvement against a heuristic baseline, and emits latency stats,
  throughput, and route-length medians suitable for audit trails.
- `planner.api.PlannerReport` — dataclass summarising scores, improvement,
  per-oracle metrics (success rate, CI, FLOPs proxies), plus global latency/route
  medians.

Example:

```python
from planner import PlannerInLoop
from importlib import import_module

from planner.api import _ensure_oracle_package  # ensures package bootstrapping
_ensure_oracle_package()

AskcosClient = import_module("02_oracle.askcos").AskcosClient
AiZynthClient = import_module("02_oracle.aizynth").AiZynthClient

planner = PlannerInLoop([AiZynthClient(use_fallback=True)])
report = planner.evaluate(["CCO", "N#N", "c1ccccc1"], k=3)
print(report.feasible_at_k)
```

See `planner/DECISIONS.md` for design notes and `tests/test_oracle_integration.py`
for a mocked end-to-end exercise. To reproduce the Week 2 metrics used in the
paper-ready summary, run `python scripts/run_week2_planner.py`.
