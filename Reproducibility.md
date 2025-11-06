# Reproducibility Playbook

This document tracks all environment, data, and execution details required to regenerate Week 0–1 artifacts for Reliable Generative Design under Hard Constraints.

## Environment

- **OS**: Ubuntu 22.04 LTS (tested locally), CUDA 12.x for GPU acceleration.
- **Python**: 3.10+
- **Package manager**: `conda` (recommended) or `uv/pip`.
- **Hardware**: 1× NVIDIA A100 (40GB) or V100 (16GB) for baseline training; CPU-only runs possible for metric computation.

### Quick Setup (conda)

```bash
conda create -n rgd python=3.10
conda activate rgd
pip install -r requirements.txt
```

The requirements list includes `guacamol`, which provides a local fallback copy
of the GuacaMol dataset. Installing via `pip install -r requirements.txt`
ensures the package is ready before data acquisition.

Optional Week 2 dependency (enables the full AiZynthFinder backend):

```bash
pip install aizynthfinder
```

Then download the public expansion/filter/stock assets once (≈200 MB):

```bash
python -m aizynthfinder.tools.download_public_data .
```

The `requirements.txt` file is generated whenever dependencies change. Lockfile hashes are stored in `env.lock.json`.

## Data Acquisition

1. Create `data/` directory.
2. Download MOSES and GuacaMol smile lists:

```bash
python scripts/download_data.py --dataset moses --dest data/
python scripts/download_data.py --dataset guacamol --dest data/
```

> **Tip:** If the GuacaMol download URL returns 404, the script falls back to
> the dataset bundled with the installed `guacamol` package. Verify the package
> is installed before running the download step.

## Week 2 — Planner-in-the-loop

1. Generate the feasibility/latency illustration:

```bash
python scripts/make_week2_feasibility.py
```

2. Run oracle smoke tests (uses mocked ASKCOS responses and AiZynth heuristic fallback):

```bash
pytest tests/test_oracle_integration.py
```

3. High-volume planner run with structured logging (640 oracle calls, ≥80% success, Δ≥15% feasible@k). This now samples real molecules from MOSES (via `--smiles-file data/moses.smi`) and sanitizes them before oracle calls:

```bash
python scripts/run_week2_planner.py \
  --n-molecules 320 --k 10 \
  --smiles-file data/moses.smi \
  --aizynth-config config.yml \
  --aizynth-restarts 3 \
  --baseline-aizynth-config config_baseline.yml \
  --baseline-aizynth-restarts 1 \
  --baseline-floor 0.25 \
  --baseline-ceiling 0.95 \
  --hard-pool-multiplier 6 \
  --easy-pool-multiplier 4 \
  --disable-askcos \
  --aizynth-log logs/aizynth_week2.log
```

This exports `metrics/week2_planner_metrics_real.json`, appends audited rows to `logs/oracle_calls.parquet`, and streams detailed AiZynthFinder progress to `logs/aizynth_week2.log`. The baseline config (`config_baseline.yml`) runs the same molecules under a constrained budget to quantify uplift.

4. Sample planner usage:

```python
from importlib import import_module
from planner import PlannerInLoop

AskcosClient = import_module("02_oracle.askcos").AskcosClient
AiZynthClient = import_module("02_oracle.aizynth").AiZynthClient

askcos = AskcosClient(base_url="https://askcos.example.com")  # requires valid credentials
aizynth = AiZynthClient(use_fallback=False, config_path="config.yml")
planner = PlannerInLoop([askcos, aizynth])
report = planner.evaluate(["CCO", "N#N", "c1ccccc1"], k=3)
print(report.feasible_at_k, report.improvement)
```

All oracle calls are logged to `logs/oracle_calls.parquet` (with a CSV/JSONL fallback
if Parquet dependencies are unavailable), enabling throughput and latency audits.

## Week 3 — Uncertainty & Calibration

1. Train the ensemble (CPU-friendly):

   ```bash
   python -m 03_uncertainty.train_ensemble --dataset moses --members 5 --max-samples 20000 \
     --output 03_uncertainty/ensemble.json
   ```

2. Run calibration + conformal evaluation:

   ```bash
   python -m 03_uncertainty.calibrate --dataset moses --ensemble 03_uncertainty/ensemble.json \
     --max-samples 20000 --metrics-dir metrics --figures-dir figures
   ```

   Outputs: `metrics/calibration_table.csv`, `metrics/coverage_week3.json`, and `figures/week3_reliability.png`.

3. One-shot CI target:

   ```bash
   make reproduce-calibration
   ```

   Uses the same hyperparameters as above and is wired into `tests/test_uncertainty.py` for regression protection.

## Week 4 — Shift Detection & Label Shift Estimation

1. Detect/estimate shifts + sequential monitoring:

   ```bash
   python scripts/run_shift_analysis.py --dataset moses --max-samples 15000 --permutations 100 \
     --metrics metrics/shift_week4.json --report reports/shift_week4.json --figures-dir figures
   ```

   Outputs: `metrics/shift_week4.json`, `reports/shift_week4.json`, `figures/week4_shift_detection.png`.

2. CI target:

   ```bash
   make reproduce-shift
   ```

   Reuses the same parameters, ensuring α=0.05 sequential guarantees documented in `audit_plan.md`.

## Week 5 — Compute Frontier & Ablations

1. Compute frontier sweep:

   ```bash
   python scripts/run_compute_frontier.py --output metrics/compute_frontier_week5.json --figure figures/week5_compute_frontier.png
   ```

2. Ablations:

   ```bash
   python 05_ablate/ablation_runner.py --csv metrics/ablation_week5.csv --figure figures/week5_ablation.png
   ```

3. One-line target:

```bash
make reproduce-frontier
```

  Writes all metrics/figures and is covered by `tests/test_frontier.py`.

4. Prospective evaluation:

```bash
make reproduce-prospective  # uses config.yml for AiZynthFinder by default
```

   Generates `06_prospective/results.json`, `figures/week6_summary.png`, and logs reviewer sign-offs in `REVIEWS.md`.

5. Nature bundle (Weeks 2, 5, 6 end-to-end with real AiZynthFinder):

```bash
make nature-bundle
```

   Runs the Week 2 planner with real MOSES targets + AiZynth (plus baseline comparison), Week 5 frontier/ablation sweep, and Week 6 prospective evaluation.

3. Verify SHA256 checksums against `data_manifest.csv`.
4. Generate splits:

```bash
python 00_data/make_splits.py --dataset moses --input data/moses.smi --split-mode random --smiles-column smiles
python 00_data/make_splits.py --dataset moses --input data/moses.smi --split-mode scaffold --smiles-column smiles
python 00_data/make_splits.py --dataset guacamol --input data/guacamol.smi --split-mode scaffold --smiles-column smiles
# Optional: supply --split-mode time once timestamp metadata is available.
```

Each command writes split assignments to `processed/<dataset>/splits/` and summary reports to `reports/`.

## Baseline Reproduction

1. **Optional (training)** — install upstream baselines:

```bash
pip install git+https://github.com/molecularsets/moses.git
pip install guacamol
```

2. Sample training command:

```bash
python 01_baselines/train_baseline.py train --baseline moses_vae
```

3. Evaluate generated molecules (override `MOSES_REFERENCE` to cap CI runtime if needed):

```bash
MOSES_REFERENCE=data/moses_week1_reference.smi \
python 01_baselines/train_baseline.py evaluate \
  --dataset moses \
  --reference "$MOSES_REFERENCE" \
  --generated outputs/moses_vae/generated.smi \
  --report metrics/moses_vae_week1.json
```

## Reproduction Targets

- `make reproduce-baselines` — orchestrates the full Week 0–1 pipeline (splits, baseline evaluation, metrics export). Expected wall-clock: <4 hours on single GPU; <1 hour without training (evaluation only).
- CI workflow `.github/workflows/reproduce.yml` runs the same make target on Ubuntu + Python 3.10.

## Reproducibility Snapshot

After running the Week 2/5/6 pipeline, capture artifact hashes and latched metadata:

```bash
make update-reproducibility
```

This produces `metrics/reproducibility_snapshot.json` and appends a timestamped entry to `logs/reproducibility.log`, enabling reviewers to verify figures/metrics via SHA256 checksums.

## Determinism Checklist

- Fixed random seeds documented in configs (`seed` fields).
- All scripts accept `--seed` overrides.
- RDKit conformer generation not used (avoids stochasticity).
- Logging includes Git commit hash via `GIT_COMMIT` environment variable.
- Output directories contain manifest files with SHA256 of generated reports.

## External Validation

- `REVIEWS.md` will capture peer verification (Week 6 milestone).
- Weekly status stored under `reports/weekX_summary.md`.
