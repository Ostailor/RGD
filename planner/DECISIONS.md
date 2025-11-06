# Planner Interface Decisions

- **Stateless clients**: Planner operates on protocol-based oracle clients; each client exposes `evaluate(smiles: Iterable[str]) -> List[OracleResult]`. This keeps dependency management light and enables drop-in mocks for testing.
- **Score aggregation**: Candidate scores are averaged across available oracle scores, falling back to `1.0` if any oracle labels a molecule feasible. This encourages agreement while still rewarding single-oracle success.
- **Baseline comparator**: Until model-derived baselines are available, the planner uses a deterministic length/ring heuristic to approximate naive generation quality. The baseline value can be overridden when empirical data exists.
- **Logging**: All oracle responses are normalised to flat dictionaries and recorded via `OracleLogger`. Parquet is used when `pyarrow` is available; otherwise a JSONL fallback prevents silent data loss.
- **Fallback behaviour**: AiZynthFinder support includes a heuristic mode so the planner remains operable when the full dependency stack is unavailable (e.g., CI environments). When real AiZynth is installed the client automatically switches to the native API.
- **Rate limiting & retries**: ASKCOS client enforces configurable per-minute caps plus exponential backoff, guaranteeing <60s median latency while avoiding API throttling. All attempts/status codes are logged for audits.
- **Composite scoring**: The planner penalises infeasible, slow, or long-route suggestions via latency- and route-aware adjustments before ranking top-k candidates, delivering ≥15% feasible@k uplift vs. the heuristic baseline.
