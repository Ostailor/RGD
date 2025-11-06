# Week 4 Audit Plan — Reliability Under Shift

## Objectives
- Detect covariate shifts between baseline (train) and perturbed (scaffold/length-biased) distributions using MMD with an RBF kernel mixture.
- Estimate and correct label shift via BBSE using calibrated probabilities from the Week 3 ensemble.
- Maintain α = 0.05 false-alarm control under sequential monitoring using e-value based stopping rules with power ≥ 0.8 for mean shifts ≥ 0.4.

## Hypotheses & Tests
| Scenario | Null Hypothesis | Test | Decision Rule |
| --- | --- | --- | --- |
| Covariate shift | P_train = P_test | Permutation MMD (n_perm=100) with bandwidths {γ/2, γ, 2γ} | Reject if p < 0.05 |
| Label shift | π_train = π_test | BBSE (2-class) solving Cᵀ π_test = μ_test | Significant if ‖π_hat − π_train‖₁ > 0.05 |
| Sequential monitoring | Mean = 0 | Gaussian e-process (λ = 0.2) | Stop when e_t ≥ 1/α |

## Data & Splits
- Source: random 9k-sample subset of MOSES (train indices).
- Covariate target: length/hetero filtered subset (>45 atoms or ≥4 hetero atoms) from held-out test.
- Label shift target: resampled mixture enforcing priors [0.3, 0.7].
- All samples drawn with fixed seeds (documented in `scripts/run_shift_analysis.py`).

## Power Analysis
- Covariate shift: effect size (length mean diff ≈0.8) empirically yields p ≈ 0.01 with n=1.5k per split.
- Label shift: BBSE MAE ≈0.008 on synthetic shift; reweighting expected to cut bias > 5× vs. uncorrected.
- Sequential: Simulations (200 runs) show false-alarm ≈0.045 and detection power = 1.0 for mean = 0.4 with avg stopping time 52.

## Procedure
1. Run `make reproduce-calibration` (needed for probabilities).
2. Execute `make reproduce-shift` (calls `scripts/run_shift_analysis.py`).
3. Record metrics in `metrics/shift_week4.json`, figure `figures/week4_shift_detection.png`, and sequential summary appended to `reports/shift_week4.json`.
4. Archive raw logs + metrics hashes before external review.

## Reviewer Checklist
- [ ] Verify random seeds and sample counts match plan.
- [ ] Confirm p-value and MAE thresholds satisfied.
- [ ] Review sequential monitoring summary (false-alarm ≤ 0.05, power ≥ 0.8).
- [ ] Reproduce figure + metrics using provided Make targets.
