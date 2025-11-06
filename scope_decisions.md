# Scope Decisions

- Dataset focus: MOSES (primary), GuacaMol (secondary) following MoleculeNet to expose both random and scaffold splits.
- Oracle selection: ASKCOS primary, AiZynthFinder secondary (Week 2 integration).
- Planner contracts defined in `planner/DECISIONS.md`; oracle logging centralised via `logs/oracle_calls.parquet` for auditability.
- Split policy: Generate and publish random + Bemis-Murcko scaffold splits; add time-based splits when timestamps exist (e.g., MOSES).
- Metric targets: Validity >= 0.9, novelty >= 0.8, Frechet descriptor distance improvement vs. baselines tracked weekly.
- CI commitment: `make reproduce-baselines` runs on every PR touching data or baselines.
