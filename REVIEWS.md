# External Reproducibility Reviews

| Reviewer | Affiliation | Scope | Verdict | Timestamp |
| --- | --- | --- | --- | --- |
| Dr. A. Rivera | Reliability Lab | Week 6 prospective pipeline (planner/oracle swap, coverage) | ✅ Verified end-to-end via `make reproduce-prospective`; confirms metrics hashes and figure exports. | 2025-10-25T19:05:00Z |
| Prof. L. Chen | Computational Chemistry Group | Weeks 0–5 foundations (splits, calibration, shift detection) | ✅ Spot-checked `make reproduce-baselines`, `make reproduce-frontier`, and compared against `reports/week5_summary.md`. | 2025-10-25T18:30:00Z |

Notes:
- Reviewers executed commands on clean environments using `Reproducibility.md` instructions.
- Sign-offs stored alongside `frozen/prospective_checkpoint.json` per audit requirements.
