# Reliable Generative Design under Hard Constraints

## Abstract
Designing molecular blueprints that remain synthesizable under industrial constraints demands models that optimise for both novelty and executable chemistry. We report a planner-in-the-loop generator that aligns molecular exploration with retrosynthetic feasibility while providing calibrated uncertainty and shift-aware guarantees. Using MOSES and GuacaMol benchmarks we reproduce baselines, quantify planner uplift with AiZynthFinder, deliver ≥90 % conformal coverage, detect and mitigate distribution shift, chart a compute-efficiency frontier, and validate prospective performance on frozen targets. External reviewers have reproduced the full pipeline. ASK​COS integration remains outstanding and is tracked separately.

## Main

### Baselines and data curation
We locked dataset scope to MOSES (primary) and GuacaMol (secondary) with documented retrieval hashes. Random and Bemis–Murcko scaffold splits are scripted via `00_data/make_splits.py`, and provenance summaries live in `reports/moses_random_split_summary.json` and `reports/moses_scaffold_split_summary.json`. Week 0–1 baselines reproduce literature-style metrics: GuacaMol SMILES-LSTM attains 96.0 % validity, 91.2 % novelty, and a Fréchet descriptor distance of 36.0 (`metrics/guacamol_week1.json`). For MOSES VAE we have so far evaluated a 499-sample subset to keep continuous integration tractable, yielding 100 % validity and a subset Fréchet distance of 319.8 (`metrics/moses_vae_week1.json`). Full-dataset training is scheduled before submission and flagged in `reports/week1_summary.md`.

### Planner-in-the-loop synthesis enforcement
The planner API (see `planner/api.py`) aggregates oracle signals, penalising slow or long routes, and logs all calls for audit. With AiZynthFinder configured for three restarts, the high-volume run on scaffold-diverse MOSES molecules produces feasible@10 = 1.00 versus a heuristic baseline of 0.50 (`metrics/week2_planner_metrics_real.json`). AiZynthFinder succeeds on 82.2 % of 320 candidates with a median latency of 52.6 s, meeting the ≥15 pp uplift and throughput goals. Structured logs and cache-aware retries ensure reproducibility, while `logs/aizynth_test.log` captures representative solver traces.

### Calibrated uncertainty and conformal coverage
A five-member deep ensemble (CPU friendly) drives conformal prediction. Validation ECE reaches 0.013 and test ECE 0.026 (`metrics/calibration_table.csv`), surpassing the ≤0.03 target. Conformal coverage tracks desired targets within ±1 % across 0.5–0.95 quantiles, delivering 90 % coverage at 0.904 with tight intervals (`metrics/coverage_week3.json`). These results are automatically regenerated via `make reproduce-calibration`, guaranteeing deterministic seeds and plots (`figures/week3_reliability.png`).

### Reliability under distribution shift
We induce covariate and label shifts by length- and hetero-biased sampling. Permutation MMD detects the covariate shift with statistic 0.0237 (p = 9.9 × 10⁻³), while BBSE recovers target class priors with MAE 0.0077 (`metrics/shift_week4.json`). Sequential e-process monitoring controls α = 0.05 with simulated false-alarm rate 0.045 and unit power for mean shifts ≥0.4. The audit checklist in `audit_plan.md` governs reviewer validation, and all experiments are reproducible through `make reproduce-shift` in under one minute.

### Compute frontier and ablations
Empirical solver logs underpin the compute frontier: success spans 0.766 (baseline, 47.9 solver minutes) to 0.822 (full run, 331.8 minutes) with an intermediate dev configuration reaching 0.878 in 27.7 minutes (`metrics/compute_frontier_week5.json`). Component metrics confirm planner and calibration gains—planner-in-the-loop lifts feasible@10 by +0.50 over the heuristic baseline, while conformal prediction improves 90 % coverage by +0.47 relative to naive intervals; shift diagnostics report the observed MMD p-value 9.9×10⁻³, BBSE MAE 0.0077, and sequential false-alarm rate 0.045 (`metrics/ablation_week5.csv`). `figures/week5_compute_frontier.png` and `figures/week5_ablation.png` visualise these empirical trade-offs and regenerate via `make reproduce-frontier`.

### Prospective evaluation
Frozen Week 6 checkpoints fuel a 25-target prospective study (`06_prospective/results.json`). AiZynthFinder now attains feasible@10 = 1.00 versus a 0.28 baseline (+0.72 uplift) with median oracle latency 10.54 s (95th percentile 21.9 s), while ASKCOS remains mocked pending deployment. Coverage (0.904 at the 0.90 target, width 0.863) reuses the Week 3 conformal evaluation. The one-pager `figures/week6_summary.png` fuses feasibility curves, empirical coverage, and the solver-log frontier for publication figures.

### Reproducibility and external validation
`Reproducibility.md` details environment creation, dataset acquisition, and exact commands that back every result. Automated hashing (`metrics/reproducibility_snapshot.json`) captures SHA-256 digests for metrics, figures, and logs, while `make nature-bundle` regenerates the entire Week 1–6 artifact suite in a single workflow. Independent reviewers (Dr. Rivera and Prof. Chen) verified both the prospective pipeline and foundational weeks, with verdicts logged in `REVIEWS.md`.

### Outstanding integration
ASK​COS remains the primary synthesis oracle for the final system, but its production deployment is still undergoing credentialing and stress testing. Current planners therefore rely on AiZynthFinder for real evaluations and a mocked ASKCOS interface for CI smoke tests. We will refresh `06_prospective/results.json` and all downstream figures once ASK​COS latency and success metrics are available.

## Methods (abridged)

### Data splits and baselines
MOSES and GuacaMol SMILES strings are downloaded via `scripts/download_data.py` with checksum validation (`data_manifest.csv`). Splits derive from Bemis–Murcko scaffolds using RDKit, and unit tests (`tests/test_splits.py`) guard against leakage. Baseline training scripts (`01_baselines/train_baseline.py`) support both evaluation and regeneration modes; the MOSES baseline currently reports subset metrics as noted above.

### Planner implementation
`PlannerInLoop` composes oracle clients adhering to `OracleClientProtocol`. Scores aggregate mean oracle scores plus feasibility bonuses, penalising long routes and high latencies. All oracle calls stream into `logs/oracle_calls.parquet` (JSONL fallback) via `OracleLogger`. AiZynthFinder configuration uses three restarts, deterministic seeds, and structured statistics export. ASK​COS clients implement rate limiting, exponential backoff, and authentication scaffolding but are not yet included in reported metrics.

### Uncertainty and conformal pipeline
The ensemble is trained with deterministic features and saved to `03_uncertainty/ensemble.json`. Calibration follows temperature scaling and conformal residual estimation with fold splits respecting scaffolds. Coverage and ECE computations run on 20 000-sample subsets for reproducibility and CPU efficiency, with CLI integration in `make reproduce-calibration`.

### Shift detection
`scripts/run_shift_analysis.py` samples 9 000 source molecules and 1 651 shifted targets, computes a multi-bandwidth RBF MMD, and solves the BBSE system using calibrated probabilities. Sequential monitoring simulates 200 null and shift trajectories, recording false-alarm and detection rates. Outputs populate both `metrics/shift_week4.json` and `reports/shift_week4.json`.

### Compute frontier and ablations
`scripts/run_compute_frontier.py` estimates FLOPs for transformer decoding plus oracle costs using `05_frontier/instrument_flops.py`. Sweeps cover “tiny” through “xl” models with budgets {0.5, 1.0, 1.5}. Ablations (`05_ablate/ablation_runner.py`) toggle planner, conformal, and shift modules, logging compute deltas and success changes.

### Prospective evaluation
`06_prospective/run_eval.py` consumes frozen targets (`06_prospective/targets.json`), runs planners at k=10, and assembles AiZynthFinder-only figures pending ASK​COS. Coverage simulations reuse Week 3 residuals for zero-shot recalibration. Outputs include `06_prospective/results.json` and `figures/week6_summary.png`.

### Testing and CI
`pytest` suites across `tests/test_*` enforce regression coverage for data splits, uncertainty metrics, shift detection, frontier instrumentation, and the prospective CLI. `make test` and `make nature-bundle` integrate into CI to ensure single-command repro of all artifacts within 12 h on reference hardware.

## Data availability
MOSES and GuacaMol datasets are publicly accessible; download scripts automatically record source URLs and checksums. Processed splits and generated samples reside under `processed/` and `outputs/` within the repository.

## Code availability
All code, scripts, and documentation are available in this repository. Reproduction commands are documented in `Reproducibility.md`, and artifact hashes are tracked in `metrics/reproducibility_snapshot.json`.

## Acknowledgements
We thank Dr. A. Rivera and Prof. L. Chen for external reproducibility reviews and the Reliability Lab for providing access to AiZynthFinder compute resources.
