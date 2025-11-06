# Week 3 Summary — Uncertainty & Calibration

- **Ensemble training**: `python -m 03_uncertainty.train_ensemble --dataset moses --members 5 --max-samples 20000` produces `03_uncertainty/ensemble.json` (5 linear members over deterministic features). Runtime <5 s on CPU.
- **Calibration metrics** (`metrics/calibration_table.csv`):
  - Validation — ECE **0.013**, NLL 0.662, Brier 0.235.
  - Test — ECE **0.026**, NLL 0.666, Brier 0.236.
  - Meets Week 3 requirement (ECE ≤ 0.03) with ~4% NLL improvement vs. raw ensemble logits.
- **Conformal coverage** (`metrics/coverage_week3.json`): coverage within ±1% of targets on the 0.5–0.95 grid; intervals are ≥35 percentage points tighter than naive Gaussian intervals while satisfying ≥12% width reduction criterion.
- **Artifacts**: `figures/week3_reliability.png` (val/test reliability curves) and `make reproduce-calibration` CI target. Reliability plot annotated with ideal diagonal and documented in `03_uncertainty/README.md`.
- **Testing/CI**: Added `tests/test_uncertainty.py` to exercise the data loader, conformal quantile monotonicity, and the full calibration runner on a 500-sample subset. `make test` now covers both Week 1 split tests and Week 3 calibration tests.
