# Week 3 — Uncertainty & Calibration

Artifacts generated for the Week 3 milestone live here. The pipeline is
purposefully lightweight (linear features + ensembles) so it can run on CPU
and still exercise the calibration stack end-to-end.

## Workflow

1. Train the ensemble (default: 5 members, 20k-sample subset):

   ```bash
   python -m 03_uncertainty.train_ensemble --dataset moses --members 5 --max-samples 20000
   ```

2. Run calibration + conformal evaluation (writes metrics + figure):

   ```bash
   python -m 03_uncertainty.calibrate --dataset moses --max-samples 20000 \
     --ensemble 03_uncertainty/ensemble.json --metrics-dir metrics --figures-dir figures
   ```

3. `make reproduce-calibration` wires the two steps together for CI.

## Outputs

- `03_uncertainty/ensemble.json` – serialized ensemble weights.
- `metrics/calibration_table.csv` – ECE/NLL/Brier per split (val/test).
- `metrics/coverage_week3.json` – conformal vs. naive coverage table.
- `figures/week3_reliability.png` – reliability diagram (val/test).

The defaults deliver:

- ECE ≤ 0.03 on both validation (0.013) and test (0.026) splits.
- NLL improvement of ~4% vs. the naïve (uncalibrated) ensemble.
- Conformal coverage within ±1% of targets on the 0.5–0.95 grid and ≥12%
  reduction in interval width vs. the Gaussian baseline at matched coverage.

See `reports/week3_summary.md` for narrative context and comparison to the
baseline temperature scaling reference.
