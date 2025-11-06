# 00_data

This module tracks dataset acquisition, curation, and splitting for the Reliable Generative Design project. We focus on public benchmarks used in molecule generation research:

- **MOSES** (Molecular Sets)  
  Source: https://github.com/molecularsets/moses  
  License: MIT  
  Canonical reference: _Polykovskiy et al., Molecular Sets (MOSES): A Benchmarking Platform for Molecular Generation_, arXiv:1811.12823.

- **GuacaMol**  
  Source: https://github.com/BenevolentAI/guacamol  
  License: MIT  
  Canonical reference: _Brown et al., GuacaMol: Benchmarking Models for de Novo Molecular Design_, Journal of Chemical Information and Modeling (2019).

## Split Strategy

Following MoleculeNet guidance and recent reproducibility reports:

1. **Random Split** (baseline) — stratified sampling by target label (if available) or uniform sampling otherwise.
2. **Bemis–Murcko Scaffold Split** — generated via RDKit. Ensures scaffolds in the validation/test sets are unseen during training to test out-of-distribution generalization.
3. **Time-Based Split** — relies on metadata timestamps (`record_date` column, ISO 8601). Enforces chronological split to control for temporal leakage.

We will publish both random and scaffold splits by default, with optional time-based splits when the source dataset includes timestamp metadata.

## Artifacts

| File | Description |
| --- | --- |
| `make_splits.py` | Entry point for generating and validating dataset splits. |
| `schemas/` | JSON schema files describing expected input columns (to be added as datasets are integrated). |
| `README.md` | This document. |

Generated split indices are stored under `processed/<dataset>/splits/<split_name>.json`. Each JSON file includes:

- metadata block (dataset version, commit hash, generator, timestamp)
- split assignment for each molecule ID (`train`, `val`, `test`)
- scaffold/time statistics used for audits

## Hashes & Integrity

Dataset downloads must be tracked via `data_manifest.csv` with SHA256 hashes. See `Reproducibility.md` for exact commands.

## Reproducibility Checklist

- [ ] Download dataset via documented script and verify SHA256 hash.  
- [ ] Run `python make_splits.py --dataset <name> --split-mode scaffold`.  
- [ ] Confirm `tests/test_splits.py` passes (`pytest tests/test_splits.py`).  
- [ ] Record split summary in `reports/week1_dataset.md` including scaffold uniqueness, class balance, and timestamp coverage.

## References

- Wu et al., _MoleculeNet: A Benchmark for Molecular Machine Learning_, Chem. Sci. 2018.
- Bemis & Murcko, _The Properties of Known Drugs. 1. Molecular Frameworks_, J. Med. Chem. 1996.
- Stanley et al., _On the Validity of Molecular Splits_, arXiv:2106.11039.
