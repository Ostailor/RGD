# Reliable Generative Design (RGD)

RGD is an end-to-end benchmark and reference implementation for reliable molecular generative design under hard constraints. The project curates open datasets, baseline models, uncertainty-aware evaluation, distribution-shift analysis, and retrospective/prospective planning experiments. Every stage is scripted and reproducible to support weekly milestone reviews and audit trails.

## Repository Layout
- `00_data/` scripts for dataset acquisition, cleaning, and split generation.
- `01_baselines/` wrappers around GuacaMol and MOSES generative baselines plus evaluation CLI.
- `02_oracle/` interfaces for AiZynthFinder, ASKCOS, and oracle logging utilities.
- `03_uncertainty/` ensemble training, calibration, conformal prediction, and plots.
- `04_shift/` label/latent shift detection tests and sequential monitoring utilities.
- `05_ablate/` experiment runner for ablation sweeps feeding Week 5 summaries.
- `05_frontier/` FLOPs instrumentation and compute frontier reporting.
- `06_prospective/` prospective evaluation pipeline and reviewer-facing reports.
- `planner/` high-level planner-in-the-loop orchestration (Week 2+ scenarios).
- `metrics/`, `figures/`, `logs/`, `reports/`, `outputs/` materialized artifacts.
- `Makefile` task shortcuts; `Reproducibility.md` detailed reproduction notes.

## Prerequisites
- Python 3.10+
- CUDA-capable GPU recommended for generative training (CPU works for evaluation-only runs).
- Conda or `uv`/`pip` for environment management.
- Git LFS for large datasets (`brew install git-lfs && git lfs install`).

## Quick Start
```bash
# 1. Create environment
conda create -n rgd python=3.10
conda activate rgd
pip install -r requirements.txt

# 2. Fetch datasets (writes to data/)
python scripts/download_data.py --dataset moses --dest data/
python scripts/download_data.py --dataset guacamol --dest data/

# 3. Generate standard splits
python 00_data/make_splits.py --dataset moses --input data/moses.smi --split-mode random --smiles-column smiles
python 00_data/make_splits.py --dataset guacamol --input data/guacamol.smi --split-mode scaffold --smiles-column smiles

# 4. Reproduce the baseline evaluation bundle
make reproduce-baselines
```
Large raw files (CSV/SMI/SMILES) must be tracked via Git LFS or excluded with `.gitignore`. A manifest with expected SHA256 hashes is stored in `data_manifest.csv`.

## Milestone Pipelines
- **Week 0–1 (Baselines):** `make reproduce-baselines` runs dataset prep and baseline metric computation, saving JSON summaries under `metrics/`.
- **Week 2 (Planner-in-the-loop):** `make reproduce-planner` stresses oracle integrations (AiZynth/ASKCOS) and collects feasibility/improvement metrics.
- **Week 3 (Uncertainty & Calibration):** `make reproduce-calibration` trains the ensemble, calibrates prediction sets, and exports reliability plots.
- **Week 4 (Shift Detection):** `make reproduce-shift` runs BBSE/MMD tests and writes reports to `metrics/shift_week4.json` and `reports/`.
- **Week 5 (Compute Frontier & Ablations):** `make reproduce-frontier` and `python 05_ablate/ablation_runner.py --csv metrics/ablation_week5.csv --figure figures/week5_ablation.png` evaluate compute trade-offs.
- **Week 6 (Prospective Evaluation):** `make reproduce-prospective` executes planning against curated target lists with audit logs.
- **Nature Bundle:** `make nature-bundle` stitches the Week 2, 5, and 6 runs to produce the publication package.

Each command stamps logs with the active Git commit, fixed random seed, and execution metadata to aid external audit.

## Key CLI Examples
```bash
# Evaluate MOSES VAE baseline against the Week 1 reference set
python 01_baselines/train_baseline.py evaluate \
  --dataset moses \
  --reference data/moses_week1_reference.smi \
  --generated outputs/moses_vae/generated.smi \
  --report metrics/moses_vae_week1.json

# Train and apply the uncertainty ensemble (Week 3)
python -m 03_uncertainty.train_ensemble --dataset moses --members 5 --max-samples 20000 --output 03_uncertainty/ensemble.json
python -m 03_uncertainty.calibrate --dataset moses --ensemble 03_uncertainty/ensemble.json --metrics-dir metrics --figures-dir figures

# Run shift analysis (Week 4)
python scripts/run_shift_analysis.py \
  --dataset moses --max-samples 15000 --permutations 100 \
  --metrics metrics/shift_week4.json --report reports/shift_week4.json --figures-dir figures
```

## Testing & Continuous Integration
- Unit tests: `pytest tests`
- Targeted smoke tests: `pytest tests/test_oracle_integration.py` (planner), `tests/test_uncertainty.py`, `tests/test_frontier.py`
- Make targets invoked in CI workflows mirror the above pipelines to guarantee reproducibility.

## Artifacts & Logging
- `metrics/*.json` primary quantitative outputs consumed by dashboards.
- `figures/*.png` publication-ready charts (reliability curves, compute frontier, shift plots).
- `logs/` contains planner traces, oracle call audits, and reproducibility snapshots.
- `reports/` human-readable summaries for weekly status and reviewer sign-off.

All scripts emit deterministic outputs given fixed seeds and respect the `GIT_COMMIT` environment variable for provenance.

## Contributing
1. Fork and clone the repository (`git clone git@github.com:<user>/RGD.git`).
2. Install prereqs and ensure Git LFS is active if manipulating large data assets.
3. Add or update scripts/tests. Please accompany new pipelines with metrics or figure regression tests when feasible.
4. Run the relevant `make reproduce-*` target plus `pytest` before submitting a PR.

For detailed reproducibility guidance, consult `Reproducibility.md`, `audit_plan.md`, and `scope_decisions.md`.
