# Week 5 — Compute Frontier

- `instrument_flops.py`: legacy FLOP estimator (retained for reference).
- `scripts/run_compute_frontier.py`: aggregates empirical success/latency trade-offs from AiZynthFinder logs (baseline, dev, full), emitting `metrics/compute_frontier_week5.json` and `figures/week5_compute_frontier.png`.
- `make reproduce-frontier`: one-command regeneration (frontier + ablations).
