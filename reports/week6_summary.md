# Week 6 Summary — Prospective Evaluation

- Ran `python 06_prospective/run_eval.py --k 10` (hooked to `make reproduce-prospective`) using frozen targets + planner config; outputs `06_prospective/results.json` and `figures/week6_summary.png`.
- Metrics: Feasible@10 = **1.00** (ASKCOS mock) / **1.00** (AiZynthFinder real), yielding a +0.72 uplift over the heuristic baseline (0.28). AiZynth median latency **10.54 s** (95th percentile 21.9 s); ASKCOS remains mocked pending access. Coverage = **0.904** vs. 0.90 target (interval width 0.863) pulled from Week 3 conformal evaluation.
- One-pager figure combines (i) feasibility curves with both oracles, (ii) empirical coverage vs. target, and (iii) the Week 5 compute frontier derived from solver logs.
- External reviewers documented in `REVIEWS.md` reproduced both Week 6 pipeline and earlier milestones; frozen checkpoint stored under `frozen/prospective_checkpoint.json`.
