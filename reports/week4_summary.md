# Week 4 Summary — Reliability Under Shift

- **Covariate shift detection**: `scripts/run_shift_analysis.py` (wired into `make reproduce-shift`) contrasts MOSES train vs. length/hetero-biased test subsets. MMD statistic = 0.0237 with p = 0.0099 (`metrics/shift_week4.json`), confirming significant distribution drift. Figure `figures/week4_shift_detection.png` visualizes the shifted length density.
- **Label shift estimation**: BBSE recovers target priors [0.3, 0.7] with MAE 0.0077 and weights [0.63, 1.33], satisfying the ±0.05 tolerance and enabling downstream reweighting. Metrics stored alongside covariate stats in `metrics/shift_week4.json` and duplicated in `reports/shift_week4.json`.
- **Sequential monitoring**: Anytime-valid e-process simulation yields false-alarm rate 0.045 (α = 0.05) and power 1.0 for mean shifts ≥ 0.4 with mean stopping time 52 steps. Logged under `sequential_monitoring` in the same metrics file per audit requirements.
- **Audit plan**: `audit_plan.md` captures hypotheses, α/power targets, sampling seeds, and reviewer checklist. All shift scripts run deterministically on CPU (<1 min).
- **Testing/CI**: `tests/test_shift.py` now exercises the MMD detector, BBSE estimator, and the CLI (tiny configs). `make test` and `make reproduce-shift` provide one-command regeneration for reviewers.
