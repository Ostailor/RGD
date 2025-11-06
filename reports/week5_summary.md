# Week 5 Summary — Compute Frontier & Ablations

- **Empirical frontier** (`scripts/run_compute_frontier.py`): now parses AiZynthFinder logs (`dev`, `baseline`, `full`) to report observed success/time trade-offs. Output (`metrics/compute_frontier_week5.json`, `figures/week5_compute_frontier.png`) shows success ranging 0.766→0.878 with total solver time spanning 27.7–331.8 minutes.
- **Component impacts** (`05_ablate/ablation_runner.py`): ingests recorded metrics to quantify planner and conformal gains (`metrics/ablation_week5.csv`, `figures/week5_ablation.png`). Planner-in-the-loop lifts feasible@10 by +0.50 (vs. heuristic baseline); conformal calibration improves 0.9 coverage by +0.47 over naive intervals. Shift diagnostics are reported directly from Week 4 metrics.
- **Automation**: `make reproduce-frontier` rebuilds the empirical frontier and component CSV/figure; CI test (`tests/test_frontier.py`) ensures the CLI stays functional.
- **Documentation**: Updated READMEs (`05_frontier/README.md`, `05_ablate/README.md`) and `Reproducibility.md` reflect the empirical pipeline, supporting manuscript integration.
