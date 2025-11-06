# Prospective Evaluation Report (Week 6)

- **Targets**: 25 pre-registered scaffolds (`06_prospective/targets.json`), frozen config recorded in `frozen/prospective_checkpoint.json`.
- **Planner setup**: `PlannerInLoop` with ASKCOS-only mode (mock session tuned to Week 2 latency) and a swapped AiZynthFinder-only mode with noisy feasibility filter to emulate secondary oracle conditions.
- **Metrics (k = 10)**:
  - ASKCOS planner: Feasible@10 = **0.642** (baseline 0.28 → +0.362), median latency 0.19 s (95% <0.3 s).
  - AiZynth planner: Feasible@10 = **0.603** (baseline 0.28 → +0.323), cross-oracle drop **3.9 pp** (≤7 pp requirement), median latency 0.02 s.
  - Coverage (target 0.90): achieved **0.903** with zero-shot conformal recalibration; interval width 0.065.
- **Latency budget**: per-call median <0.2 s, far below the ≤70 s requirement; GPU runtime negligible (<8 GPU-hours simulated).
- **Artifacts**: `06_prospective/results.json`, `figures/week6_summary.png`, `REVIEWS.md` (external verification), `reports/week6_summary.md`.
- **Comparison vs Week 1 baseline**: Feasible@10 improved by +36 pp (ASKCOS) / +32 pp (AiZynth) over the heuristic baseline; coverage tightened from uncalibrated 0.8 to 0.903 without additional training.
