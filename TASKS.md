# Reliable Generative Design under Hard Constraints — TASKS

This document expands the execution plan into concrete tasks, deliverables, and quantitative metrics that serve both (i) a NeurIPS/ICLR (methods) + JCIM/J. Cheminf. (domain) submission package—with escalation path to Nature—and (ii) recommendation letter evidence highlighting rank, ownership, and reproducible impact.

## North-Star Outcome
- Planner-in-the-loop molecular generator that enforces hard constraints via synthesis oracles (ASKCOS primary, AiZynthFinder secondary).
- Calibrated uncertainty estimates (deep ensembles + conformal prediction) with targeted coverage guarantees.
- Audit-first evaluation demonstrating reliability under distribution shift, compute-efficiency trade-offs, and pre-registered decision rules.
- Reproducibility artifacts (frozen environment, seeded runs, CI) enabling one-command regeneration of baselines and proposed method.
- Publication-grade figures: accuracy vs. FLOPs compute frontier, coverage vs. target curve, oracle feasibility histogram.

## Research Paper Readiness (Nature) Metrics
- **Empirical coverage**: target ±3% (e.g., 90% target → 87–93%) on prospective evaluation, reported with confidence intervals.
- **Constraint adherence**: oracle-feasible@k ≥ 35% on primary test set; median route length/time reduced ≥15% vs. baseline.
- **Reliability under shift**: statistically significant MMD/label-shift detection (α = 0.05) with quantified performance degradation and mitigation using uncertainty filters.
- **Compute frontier**: FLOPs vs. success rate curve including oracle latency, exhibiting ≥10% improvement in success per FLOP over baseline; plots explicitly show FLOPs on the x-axis with an aligned oracle wall-time axis.
- **Reproducibility**: CI workflow that re-runs full pipeline (`make reproduce-all`) in <12h on reference GPU; environment lockfile and hash-verified outputs.
- **Ablation evidence**: each component removal degrades primary success metric by ≥5%, reported with paired statistical tests, alongside the canonical calibration trio (ECE, NLL, reliability plots per Guo et al.).

## Recommendation Letter Evidence (Quantitative Hooks)
- **Ownership**: Documented decisions (audit_plan.md, annotated READMEs) with timestamps and version tags; weekly Git tags `milestone-weekX`.
- **Rank/Impact**: Comparative leaderboard vs. published baselines showing ≥2σ improvement on core metrics; include Nature submission-ready comparison table.
- **Throughput**: Oracle throughput logs demonstrating ≥95% uptime, average latency ≤60s per query with >1,000 logged calls.
- **Leadership**: CI adoption rate (≥3 distinct pipelines) and review checklist usage tracked via `ci_metrics.json`.
- **Mentorship/Audit Mindset**: Risk-limiting audit scripts with user documentation, plus reproducibility sign-off from two external reviewers (tracked via `REVIEWS.md`).

## Cross-Cutting Tasks (All Weeks)
- Maintain `Reproducibility.md` and `CHANGELOG.md` with run commands, seeds, and artifact hashes.
- Keep `metrics/` directory updated with JSON/CSV exports for each experiment (naming convention: `YYYYMMDD_*`).
- Automate figure generation (`make figures`) to ensure parity between paper plots and LOR-ready graphics.
- Record weekly summary memos (`reports/weekX_summary.md`) capturing accomplishments, blockers, and numeric KPIs.
- Use RDChiral + RDKit stereo handling for template application, chiral validity checks, and deduplication.

---

## Week 0–1 — Scope & Baselines
**Objectives**
- Lock dataset, baseline models, and evaluation splits without leakage.
- Stand up reproducible baseline pipeline with CI.

We use MOSES or GuacaMol with Bemis–Murcko scaffold or time-based splits via RDKit; we’ll report both random and scaffold splits.

**Tasks**
- Implement scaffold/time split script (`00_data/make_splits.py`) with unit tests covering scaffold uniqueness and time-based leakage.
- Curate dataset README citing MoleculeNet practices; include data licenses and retrieval hashes.
- Reproduce MOSES/GuacaMol baselines (`01_baselines/train_baseline.py`) tracking validity, novelty, Fréchet chemnet distance.
- Create `Reproducibility.md` describing hardware/software requirements; implement `make reproduce-baselines`.
- Configure CI job (`.github/workflows/reproduce.yml`) executing baseline reproduction on fixed seed.

**Deliverables**
- `00_data/` with scripts, README, unit tests (`tests/test_splits.py`).
- `01_baselines/` with training script, evaluation notebook, and summary table.
- `Reproducibility.md` and passing CI run badge/screenshot.

**Metrics**
- Split leakage tests: 0 failures across ≥100 sampled scaffolds.
- Baseline reproduction gap ≤2% from reported literature metrics.
- CI runtime ≤4h with completion log stored in `logs/ci_week1.txt`.
- Dataset retrieval reproducibility: checksum verification automated (100% pass).

**Evidence for Letter**
- Document `scope_decisions.md` with rationale for dataset/oracle choices and reviewer-style justifications.
- Capture CI uptime and mean runtime for Week 1 (≥90% success, <10 retries).

---

## Week 2 — Planner-in-the-Loop API
**Objectives**
- Integrate ASKCOS and AiZynthFinder oracles into generator loop.
- Quantify feasibility and latency trade-offs.

**Tasks**
- Implement `02_oracle/askcos.py` HTTP client with exponential backoff, rate limiting, and structured logging.
- Implement `02_oracle/aizynth.py` local runner with caching and failure handling.
- Design planner API (`planner/api.py`) that scores K proposals and penalizes infeasible routes.
- Log all oracle interactions (`logs/oracle_calls.parquet`) capturing latency, status codes, and route metadata.
- Produce README figure showing feasibility@k vs. latency using toy batch (10 molecules).

**Deliverables**
- `02_oracle/` module with integration tests (`tests/test_oracle_integration.py`).
- `planner/` package with API documentation and example usage.
- `figures/week2_feasibility.png` embedded in README.

**Metrics**
- Oracle success rate ≥80% on known set, with 95% CI reported.
- Average API latency ≤60s (ASKCOS) and ≤45s (AiZynthFinder) with 95th percentile tracked.
- Feasibility@10 improvement ≥15% vs. baseline generator without planner.
- Report (a) feasible@k, (b) median route length/step count, and (c) end-to-end latency (generator + oracle), citing AiZynthFinder’s <1 min/route reference latency while noting ASKCOS variability.

**Evidence for Letter**
- Ownership log: `planner/DECISIONS.md` summarizing interface contracts.
- Throughput: ≥500 logged oracle calls with uptime ≥95%.
- Latency chart annotated to show optimization decisions.

---

## Week 3 — Uncertainty & Calibration
**Objectives**
- Implement deep ensembles and conformal prediction for calibrated uncertainty.
- Quantify calibration via multiple metrics.

**Tasks**
- Train N=5 ensemble models (`03_uncertainty/train_ensemble.py`) with deterministic seeding.
- Implement conformal wrapper (`03_uncertainty/conformal.py`) generating prediction sets and intervals.
- Compute reliability diagrams, ECE, NLL, Brier score (`03_uncertainty/calibrate.py`).
- Update CI target `make reproduce-calibration`.
- Generate coverage vs. target plot and calibration tables for paper appendix.

**Deliverables**
- `03_uncertainty/` scripts, configs, and plots.
- `metrics/coverage_week3.json`, `metrics/calibration_table.csv`.
- CI logs demonstrating deterministic calibration pipeline.

**Metrics**
- Empirical coverage within ±3% of targets (50–95% grid) on validation and prospective sets.
- ECE ≤0.03 and NLL improved ≥10% vs. baseline softmax temperatures.
- Prediction set size reduced ≥12% with conformal vs. naive ensembles at fixed coverage.

**Evidence for Letter**
- Reliability plots annotated with audit commentary (`figures/week3_reliability.png`).
- Calibration memo (`reports/week3_summary.md`) citing standards and peer comparisons.

---

## Week 4 — Reliability Under Shift
**Objectives**
- Detect and quantify performance under covariate/label shift.
- Pre-register audit procedures with risk-limiting controls.

**Tasks**
- Implement MMD two-sample test (`04_shift/mmd_test.py`) with kernel selection justification.
- Implement BBSE label-shift estimator (`04_shift/bbse_test.py`) with calibration-aware adjustments.
- Design experimental protocol inducing artificial covariate shift (e.g., scaffolds, temporal drift).
- Author `audit_plan.md` (pre-registration: α=0.05, power 0.8, stopping rules using confidence sequences and anytime-valid confidence sequences/e-processes to control false positives during monitoring).
- Run sequential monitoring simulation to validate anytime-valid inference.

**Deliverables**
- `04_shift/` scripts, configs, shift simulation notebooks.
- `reports/shift_week4.json` summarizing p-values, effect sizes, mitigation outcomes.
- `audit_plan.md` with reviewer-style checklist.

**Metrics**
- Significant shift detection (p < 0.05) on induced shifts with power ≥0.8 (confidence via bootstrapping).
- Post-mitigation performance degradation reduced ≥20% compared to unmitigated run.
- Confidence sequence coverage ≥90% under sequential monitoring.

**Evidence for Letter**
- Demonstrated distinction between “can’t” vs. “won’t” failures documented in `reports/week4_summary.md`.
- Audit log: completion of pre-registration sign-off by external reviewer (record signature/date).

---

## Week 5 — Compute Frontier & Ablations
**Objectives**
- Quantify compute vs. performance trade-offs and component contributions.

**Tasks**
- Instrument FLOP counting per model (`05_frontier/instrument_fLOPs.py`) and log oracle compute costs.
- Perform sweeps over model size, decoding budget, oracle frequency (via `05_frontier/frontier.ipynb`).
- Run ablations: no-oracle, oracle-late, oracle-every-step; no-conformal vs. conformal; ASKCOS vs. AiZynthFinder.
- Export compute frontier plot and ablation table for paper main text.

**Deliverables**
- `05_frontier/frontier.ipynb` with reproducible pipeline.
- `05_ablate/ablation_runner.py` + `metrics/ablation_week5.csv`.
- `figures/week5_compute_frontier.png` & `figures/week5_ablation.png`.

**Metrics**
- Compute efficiency: ≥10% higher feasibility per FLOP vs. strongest baseline.
- Ablations: each removed component yields ≥5% drop in feasibility@k and/or coverage; statistical significance p < 0.05 (paired tests).
- Oracle latency breakdown highlighting compute budget allocation (≤40% time in oracle calls after optimizations), with compute frontier plots showing FLOPs plus oracle wall-time contributions.

**Evidence for Letter**
- Experimental design note (`05_ablate/EXPERIMENT_DESIGN.md`) detailing hypotheses, controls, and statistical tests.
- Leadership metric: documentation of team or collaborator usage of `make frontier` target (≥2 adopters).

---

## Week 6 — Prospective Evaluation & Cross-Domain Reuse
**Objectives**
- Validate system on pre-registered prospective targets and demonstrate oracle swap generality.

**Tasks**
- Freeze model checkpoints and config seeds (`frozen/` directory) prior to prospective evaluation.
- Evaluate on held-out target list with zero-shot conformal recalibration (`06_prospective/run_eval.py`).
- Swap oracles (ASKCOS ↔ AiZynthFinder) using identical planner interface; optionally integrate docking oracle if time permits.
- Compile final report (`06_prospective/report.md`) summarizing feasibility, coverage, latency, and generalization.
- Assemble one-pager figure combining coverage curve, feasibility histogram, compute frontier for paper & letter.

**Deliverables**
- `06_prospective/results.json`, `06_prospective/report.md`.
- `figures/week6_summary.png` (three-panel figure ready for Nature submission & LOR).
- `REVIEWS.md` capturing external reproducibility validation outcomes.

**Metrics**
- Prospective feasibility@k ≥35% (ASKCOS) and ≥30% (AiZynthFinder) with improvement vs. baseline.
- Coverage at 90% target within ±3%; latency budget maintained ≤8 GPU-hours and ≤70s median oracle call.
- Cross-oracle generality: performance drop ≤7% when swapping primary/secondary oracle.

**Evidence for Letter**
- Prospective evaluation sign-off with timestamps, independent witness (mentor/advisor) recorded.
- Quantitative summary table showing improvements relative to Week 1 baselines, highlighting ownership of gains.

---

## Submission & Packaging (Parallel Weeks 6–7)
- Draft Nature manuscript sections: Methods (planner-in-the-loop, uncertainty), Results (tables/figures), Supplement (audit plan, CI details). Maintain `paper/` directory with Overleaf sync.
- Prepare `LOR_metrics_summary.pdf` containing:
  - Leaderboard table vs. baselines.
  - Compute frontier, coverage, feasibility plots.
  - CI uptime chart & oracle latency histogram.
  - Bullet list of leadership/ownership contributions with quantitative evidence.
- Conduct reproducibility dry run with fresh environment, document steps in `Reproducibility.md`.
- Finalize artifact DOI (Zenodo/OSF) with hashed releases (`release_notes.md`).
- Submit Nature preprint to arXiv/BioMed Central as interim dissemination; log submission metadata.

## Ongoing Monitoring (Post-Week 7)
- Weekly `metrics/trends.csv` update showing moving averages for success, latency, coverage.
- Quarterly audit rerun using stored scripts to ensure long-term reproducibility.
- Maintain backlog in `TASKS_BACKLOG.md` for future extensions (e.g., active learning, docking oracle integration).
