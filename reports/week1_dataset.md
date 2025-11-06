# Week 1 Dataset Summary

- [x] **Random split audit (MOSES)** — `reports/moses_random_split_summary.json` shows 1,936,962 molecules with the expected 80/10/10 coverage (train 0.8000, val 0.1000, test 0.1000). No invalid SMILES detected.
- [x] **Scaffold split audit (MOSES)** — `reports/moses_scaffold_split_summary.json` confirms zero Bemis–Murcko overlap across train/val/test and identical coverage ratios. Serves as the leakage-proof default for reliability experiments.
- [x] **Scaffold split audit (GuacaMol)** — `reports/guacamol_scaffold_split_summary.json` documents perfect scaffold disjointness with 1,591,379 molecules.
- [x] **Checksum verification** — `data_manifest.csv` now stores SHA256 hashes for every downloaded artifact (MOSES CSV, GuacaMol SMILES, and chembl subsets). Spot-checks performed via `python scripts/download_data.py --verify` (see `Reproducibility.md`).
- [x] **Time-split readiness** — Metadata fields exist but require upstream timestamps; placeholder noted for Week 3 when temporal data becomes available.

All audits executed under `pytest tests/test_splits.py` (Week 0–1 CI target). Raw split JSON lives in `processed/<dataset>/splits/`.
