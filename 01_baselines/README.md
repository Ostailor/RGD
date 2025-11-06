# 01_baselines

Week 0–1 deliverables focus on reproducing standard generative baselines for MOSES and GuacaMol. This directory provides:

- `train_baseline.py` — CLI for launching baseline training via upstream repos and computing validity/novelty/Frechet-style metrics.
- `configs/` — Minimal configuration files aligned with upstream defaults.

## Quickstart

1. Install dependencies in a fresh environment (see `Reproducibility.md`).  
2. Generate dataset splits using `00_data/make_splits.py`.  
3. Train a baseline (optional at Week 0–1) and export generated SMILES.  
4. Evaluate the samples:

```bash
python 01_baselines/train_baseline.py evaluate \
  --dataset moses \
  --reference data/moses_train.smi \
  --generated outputs/moses_vae/generated.smi \
  --report metrics/moses_vae_week1.json
```

For GuacaMol:

```bash
python 01_baselines/train_baseline.py evaluate \
  --dataset guacamol \
  --reference data/guacamol_train.smi \
  --generated outputs/guacamol_smiles_lstm/generated.smi \
  --report metrics/guacamol_week1.json
```

## Metrics Report

Each run produces a JSON payload containing:
- `validity` — fraction of valid molecules, count of invalid examples.
- `uniqueness` — fraction and count of unique canonical SMILES.
- `novelty` — fraction of generated molecules not present in the reference set.
- `frechet_descriptor_distance` — Frechet distance over RDKit descriptors (MOSES-style proxy).

The metrics feed directly into Week 0–1 logs and recommendation letter hooks (baseline leaderboard table).

## Notes & References

- MOSES repository: https://github.com/molecularsets/moses  
- GuacaMol repository: https://github.com/BenevolentAI/guacamol  
- MoleculeNet best practices: Wu et al., Chem. Sci. 2018  
- Frechet-style descriptor distance inspired by Polykovskiy et al., 2018
