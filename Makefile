PYTHON ?= python
DATA_DIR ?= data
MOSES_REFERENCE ?= $(DATA_DIR)/moses.smi
GUACAMOL_REFERENCE ?= $(DATA_DIR)/guacamol.smi

.PHONY: download-data generate-splits evaluate-baselines reproduce-baselines test oracle-smoke week2-metrics reproduce-calibration reproduce-shift reproduce-frontier reproduce-prospective reproduce-week2-real update-reproducibility nature-bundle

# Download MOSES and GuacaMol data to data/
download-data:
	$(PYTHON) scripts/download_data.py --dataset moses --dest $(DATA_DIR)
	$(PYTHON) scripts/download_data.py --dataset guacamol --dest $(DATA_DIR)

# Generate random + scaffold splits for datasets (requires data to be downloaded)
generate-splits:
	@test -f $(DATA_DIR)/moses.csv || (echo "Missing MOSES dataset in $(DATA_DIR). Run 'make download-data' first." && exit 1)
	@if test -f processed/moses/splits/random_split.json; then \
		echo "[INFO] MOSES random split already generated; skipping"; \
	else \
		$(PYTHON) 00_data/make_splits.py --dataset moses --input $(DATA_DIR)/moses.smi --split-mode random --smiles-column smiles; \
	fi
	@if test -f processed/moses/splits/scaffold_split.json; then \
		echo "[INFO] MOSES scaffold split already generated; skipping"; \
	else \
		$(PYTHON) 00_data/make_splits.py --dataset moses --input $(DATA_DIR)/moses.smi --split-mode scaffold --smiles-column smiles; \
	fi
	@test -f $(DATA_DIR)/guacamol.smi || (echo "Missing GuacaMol dataset in $(DATA_DIR). Run 'make download-data' first." && exit 1)
	@if test -f processed/guacamol/splits/scaffold_split.json; then \
		echo "[INFO] GuacaMol scaffold split already generated; skipping"; \
	else \
		$(PYTHON) 00_data/make_splits.py --dataset guacamol --input $(DATA_DIR)/guacamol.smi --split-mode scaffold --smiles-column smiles; \
	fi

# Evaluate baseline samples (assumes generated molecules are present)
evaluate-baselines:
	@test -f outputs/moses_vae/generated.smi || echo "[WARN] outputs/moses_vae/generated.smi missing; skip MOSES baseline eval"
	@if test -f outputs/moses_vae/generated.smi; then \
		$(PYTHON) 01_baselines/train_baseline.py evaluate --dataset moses --reference $(MOSES_REFERENCE) --generated outputs/moses_vae/generated.smi --report metrics/moses_vae_week1.json; \
	fi
	@test -f outputs/guacamol_smiles_lstm/generated.smi || echo "[WARN] outputs/guacamol_smiles_lstm/generated.smi missing; skip GuacaMol baseline eval"
	@if test -f outputs/guacamol_smiles_lstm/generated.smi; then \
		$(PYTHON) 01_baselines/train_baseline.py evaluate --dataset guacamol --reference $(GUACAMOL_REFERENCE) --generated outputs/guacamol_smiles_lstm/generated.smi --report metrics/guacamol_week1.json; \
	fi

# Full Week 0-1 pipeline (no-op if generation artifacts missing)
reproduce-baselines: download-data generate-splits test evaluate-baselines
	@echo "[INFO] Reproduction pipeline finished. Review metrics/ for outputs."

# Run unit tests (dataset independent)
test:
	$(PYTHON) -m pytest tests/test_splits.py tests/test_uncertainty.py tests/test_shift.py tests/test_frontier.py tests/test_prospective.py

oracle-smoke:
	$(PYTHON) -m pytest tests/test_oracle_integration.py

week2-metrics:
	$(PYTHON) scripts/run_week2_planner.py --n-molecules 320 --k 10

reproduce-calibration:
	$(PYTHON) -m 03_uncertainty.train_ensemble --dataset moses --members 5 --max-samples 20000 --output 03_uncertainty/ensemble.json
	$(PYTHON) -m 03_uncertainty.calibrate --dataset moses --ensemble 03_uncertainty/ensemble.json --max-samples 20000 --metrics-dir metrics --figures-dir figures

reproduce-shift:
	$(PYTHON) scripts/run_shift_analysis.py --dataset moses --max-samples 15000 --permutations 100 --metrics metrics/shift_week4.json --report reports/shift_week4.json --figures-dir figures

reproduce-frontier:
	$(PYTHON) scripts/run_compute_frontier.py --output metrics/compute_frontier_week5.json --figure figures/week5_compute_frontier.png
	$(PYTHON) 05_ablate/ablation_runner.py --csv metrics/ablation_week5.csv --figure figures/week5_ablation.png

reproduce-prospective:
	$(PYTHON) 06_prospective/run_eval.py --targets 06_prospective/targets.json --k 10 --output 06_prospective/results.json --figure figures/week6_summary.png

reproduce-week2-real:
	$(PYTHON) scripts/run_week2_planner.py --n-molecules 320 --k 10 --smiles-file data/moses.smi --aizynth-config config.yml --aizynth-restarts 3 --baseline-aizynth-config config_baseline.yml --baseline-aizynth-restarts 1 --baseline-floor 0.25 --baseline-ceiling 0.95 --hard-pool-multiplier 6 --easy-pool-multiplier 4 --disable-askcos --aizynth-log logs/aizynth_week2.log --output metrics/week2_planner_metrics_real.json

update-reproducibility:
	$(PYTHON) scripts/update_reproducibility.py --output metrics/reproducibility_snapshot.json --log logs/reproducibility.log

nature-bundle: reproduce-baselines reproduce-calibration reproduce-shift reproduce-week2-real reproduce-frontier reproduce-prospective update-reproducibility
	@echo "[INFO] Nature bundle complete: Week2 (real AiZynth), Week5 frontier, Week6 prospective."
