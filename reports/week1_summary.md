# Week 1 Summary

- GuacaMol baseline (SMILES-LSTM) completed on CPU. Evaluation artifacts:
  - `guacamol_baselines/smiles_lstm_hc/distribution_learning_results.json`
  - `metrics/guacamol_week1.json`
- MOSES VAE baseline evaluated on a 500-sample reference subset to keep Week 1 CI runs tractable:
  - Generated molecules: `outputs/moses_vae/generated.smi`
  - Metrics report: `metrics/moses_vae_week1.json` (validity 1.00, frechet 53.44 vs. subset)
  - Full-dataset evaluation is queued once GPU/compute budget frees up; subset provenance noted in `metrics/moses_vae_week1.json`.
