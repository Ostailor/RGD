#!/usr/bin/env python3
"""Simulate Week 2 planner metrics with mocked ASKCOS responses."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from importlib import import_module, util

from planner import PlannerInLoop
from planner.api import PlannerReport
from statistics import NormalDist, median
from rdkit import Chem


def _ensure_oracle_package() -> None:
    if "02_oracle" in sys.modules:
        return
    pkg_path = ROOT / "02_oracle"
    spec = util.spec_from_file_location(
        "02_oracle",
        pkg_path / "__init__.py",
        submodule_search_locations=[str(pkg_path)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to initialise 02_oracle package")
    module = util.module_from_spec(spec)
    sys.modules["02_oracle"] = module
    spec.loader.exec_module(module)


_ensure_oracle_package()

AiZynthClient = import_module("02_oracle.aizynth").AiZynthClient
AskcosClient = import_module("02_oracle.askcos").AskcosClient
OracleLogger = import_module("02_oracle.logging").OracleLogger
MockAskcosSession = import_module("02_oracle.stubs").MockAskcosSession


BUILDING_BLOCKS = [
    "CC",
    "CO",
    "CN",
    "c1ccccc1",
    "O=C",
    "N#N",
    "Cl",
    "Br",
    "c1ncccc1",
    "CC(=O)O",
]


def sample_fallback_smiles(n: int, seed: int = 7) -> List[str]:
    rng = random.Random(seed)
    smiles: List[str] = []
    for _ in range(n):
        parts = rng.choices(BUILDING_BLOCKS, k=rng.randint(2, 5))
        smiles.append("".join(parts))
    return smiles


def load_and_sanitize_smiles(
    path: Path,
    n: int,
    seed: int,
    *,
    heavy_threshold: int = 30,
    ring_threshold: int = 6,
    hetero_threshold: int = 8,
    prefer_complex: bool = True,
) -> List[str]:
    candidates: List[str] = []
    if path.exists():
        lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        if lines and lines[0].lower() == "smiles":
            lines = lines[1:]
        if lines:
            rng = random.Random(seed)
            rng.shuffle(lines)
            candidates.extend(lines)
    if not candidates:
        candidates.extend(sample_fallback_smiles(n, seed))

    sanitized: List[str] = []
    seen = set()
    rng = random.Random(seed + 1337)
    for smi in candidates:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        canonical = Chem.MolToSmiles(mol, canonical=True)
        if canonical in seen:
            continue
        seen.add(canonical)
        sanitized.append(canonical)
        if len(sanitized) >= n:
            break

    if len(sanitized) < n:
        # top-up with heuristic samples
        fallback = sample_fallback_smiles(n * 2, seed + 4242)
        rng.shuffle(fallback)
        for smi in fallback:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            canonical = Chem.MolToSmiles(mol, canonical=True)
            if canonical in seen:
                continue
            seen.add(canonical)
            sanitized.append(canonical)
            if len(sanitized) >= n:
                break
    scored = []
    for smi in sanitized:
        mol = Chem.MolFromSmiles(smi)
        heavy = mol.GetNumHeavyAtoms() if mol else 0
        rings = mol.GetRingInfo().NumRings() if mol else 0
        hetero = (
            sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in (1, 6))
            if mol
            else 0
        )
        scored.append((heavy, rings, hetero, rng.random(), smi))

    if prefer_complex:
        complex_entries = [
            entry
            for entry in scored
            if entry[0] >= heavy_threshold
            or entry[1] >= ring_threshold
            or entry[2] >= hetero_threshold
        ]

        if len(complex_entries) < n:
            remaining = [entry for entry in scored if entry not in complex_entries]
            remaining.sort(reverse=True)
            complex_entries.extend(remaining)
        else:
            complex_entries.sort(reverse=True)

        return [s for *_, s in complex_entries[:n]]

    simple_entries = [
        entry
        for entry in scored
        if (heavy_threshold <= 0 or entry[0] <= heavy_threshold)
        and (ring_threshold <= 0 or entry[1] <= ring_threshold)
        and (hetero_threshold <= 0 or entry[2] <= hetero_threshold)
    ]
    if len(simple_entries) < n:
        simple_entries = scored
    simple_entries.sort()
    return [s for *_, s in simple_entries[:n]]


def _compute_feasible_at_k(records: List, k: int) -> float:
    successes = [rec.feasible for rec in records]
    denom = min(k, len(successes)) or 1
    return sum(successes[:k]) / denom


def _format_path(path: Optional[Path]) -> str:
    return str(path) if path is not None else "<in-memory>"


def _maybe_escalate_difficulty(
    smiles: List[str],
    baseline_client: AiZynthClient,
    baseline_records: List,
    args: argparse.Namespace,
) -> Tuple[List[str], List, bool]:
    """Detect overly easy batches and resample from a harder pool."""

    if getattr(args, "disable_auto_hard", False):
        return smiles, baseline_records, False

    baseline_stats = aggregate_records(baseline_records)
    success_rate = baseline_stats["success_rate"]
    if success_rate <= args.baseline_ceiling:
        return smiles, baseline_records, False

    hard_path = args.hard_smiles_file
    if hard_path is None or not hard_path.exists():
        print(
            "[WARN] Baseline success exceeds ceiling but no hard SMILES source is available "
            "— continuing with original batch."
        )
        return smiles, baseline_records, False

    print(
        f"[INFO] Baseline success {success_rate:.3f} > ceiling {args.baseline_ceiling:.3f}; "
        f"resampling {len(smiles)} molecules from harder pool at {_format_path(hard_path)}."
    )
    pool_size = max(len(smiles) * args.hard_pool_multiplier, len(smiles))
    hard_pool = load_and_sanitize_smiles(
        hard_path,
        pool_size,
        args.seed + 404,
        heavy_threshold=args.hard_heavy_threshold,
        ring_threshold=args.hard_ring_threshold,
        hetero_threshold=args.hard_hetero_threshold,
    )
    pool_records = baseline_client.evaluate(hard_pool)

    def _sort_key(item):
        smi, record = item
        steps = (record.raw or {}).get("number_of_steps") or (record.raw or {}).get("num_steps") or 0
        latency = record.latency or 0.0
        return (record.feasible, -steps, -latency, smi)

    ranked = sorted(zip(hard_pool, pool_records), key=_sort_key)
    selected = ranked[: len(smiles)]
    selected_smiles = [item[0] for item in selected]
    selected_records = [item[1] for item in selected]

    hard_stats = aggregate_records(selected_records)

    if hard_stats["success_rate"] > args.baseline_ceiling:
        print(
            "[WARN] Hard resample still near-saturated "
            f"(success {hard_stats['success_rate']:.3f}); consider providing a custom "
            "--hard-smiles-file or tightening baseline config."
        )

    return selected_smiles, selected_records, True


def _maybe_relax_difficulty(
    smiles: List[str],
    baseline_client: AiZynthClient,
    baseline_records: List,
    args: argparse.Namespace,
) -> Tuple[List[str], List, bool]:
    if getattr(args, "disable_auto_easy", False):
        return smiles, baseline_records, False

    stats = aggregate_records(baseline_records)
    success = stats["success_rate"]
    if success >= args.baseline_floor:
        return smiles, baseline_records, False

    print(
        f"[INFO] Baseline success {success:.3f} < floor {args.baseline_floor:.3f}; "
        f"mixing in easier molecules from {_format_path(args.easy_smiles_file)}."
    )

    easy_path = args.easy_smiles_file
    if easy_path is None or not easy_path.exists():
        print(
            "[WARN] Baseline success is below floor but no easy SMILES pool is available — "
            "continuing with current batch."
        )
        return smiles, baseline_records, False

    pool_size = max(len(smiles) * args.easy_pool_multiplier, len(smiles))
    easy_pool = load_and_sanitize_smiles(
        easy_path,
        pool_size,
        args.seed + 505,
        heavy_threshold=args.easy_heavy_threshold,
        ring_threshold=args.easy_ring_threshold,
        hetero_threshold=args.easy_hetero_threshold,
        prefer_complex=False,
    )
    easy_records = baseline_client.evaluate(easy_pool)
    easy_pairs = [
        (smi, rec) for smi, rec in zip(easy_pool, easy_records) if rec.feasible
    ]
    if not easy_pairs:
        print(
            "[WARN] Easy pool did not yield any feasible molecules; consider lowering the "
            "difficulty thresholds or expanding the pool."
        )
        return smiles, baseline_records, False

    selected = list(zip(smiles, baseline_records))
    seen = {s for s, _ in selected}
    changed = False
    easy_iter = iter(easy_pairs)

    while True:
        current_success = sum(rec.feasible for _, rec in selected) / len(selected)
        if current_success >= args.baseline_floor:
            break
        try:
            easy_smi, easy_rec = next(easy_iter)
        except StopIteration:
            break
        # replace the first infeasible entry, or the slowest feasible one if all succeed
        idx = next(
            (i for i, (_, rec) in enumerate(selected) if not rec.feasible),
            None,
        )
        if idx is None:
            slow_idx = max(
                range(len(selected)),
                key=lambda i: selected[i][1].latency or 0.0,
            )
            idx = slow_idx
        old_smi, _ = selected[idx]
        if easy_smi in seen:
            continue
        seen.discard(old_smi)
        selected[idx] = (easy_smi, easy_rec)
        seen.add(easy_smi)
        changed = True

    if not changed:
        print(
            "[WARN] Unable to adjust batch to reach the baseline floor; consider providing "
            "a larger easy pool."
        )
        return smiles, baseline_records, False

    new_smiles, new_records = zip(*selected)
    new_smiles = list(new_smiles)
    new_records = list(new_records)

    new_success = sum(rec.feasible for rec in new_records) / len(new_records)
    if new_success < args.baseline_floor:
        print(
            f"[WARN] Baseline success remains low ({new_success:.3f}) even after mixing in easier "
            "molecules."
        )
    else:
        print(
            f"[INFO] Adjusted baseline success to {new_success:.3f} with blended pool."
        )

    return new_smiles, new_records, True


def aggregate_records(records: List) -> dict:
    total = len(records)
    successes = [rec.feasible for rec in records]
    success_rate = sum(successes) / total if total else 0.0
    latencies = [rec.latency for rec in records if rec.latency is not None]
    median_latency = float(median(latencies)) if latencies else 0.0
    search_times = []
    route_counts = []
    unsanitizable = 0
    for rec in records:
        raw = rec.raw or {}
        if rec.error and "unsanitizable" in rec.error.lower():
            unsanitizable += 1
        if "time" in raw:
            search_times.append(raw.get("time"))
        elif "search_time" in raw:
            search_times.append(raw.get("search_time"))
        if "num_routes" in raw:
            route_counts.append(raw.get("num_routes"))
        elif "solutions" in raw:
            route_counts.append(raw.get("solutions"))
    stats = {
        "n": total,
        "success_rate": success_rate,
        "median_latency": median_latency,
        "median_search_time": float(median(search_times)) if search_times else 0.0,
        "median_routes": float(median(route_counts)) if route_counts else 0.0,
        "unsanitizable": unsanitizable,
    }
    return stats


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    phat = successes / total
    denom = 1 + (z**2) / total
    center = phat + (z**2) / (2 * total)
    margin = z * math.sqrt((phat * (1 - phat) + (z**2) / (4 * total)) / total)
    lower = (center - margin) / denom
    upper = (center + margin) / denom
    return (max(0.0, lower), min(1.0, upper))


def build_planner(
    session_seed: int,
    aizynth_config: Path,
    use_aizynth_fallback: bool,
    aizynth_log: Path | None,
    disable_askcos: bool,
    aizynth_restarts: int,
) -> PlannerInLoop:
    oracles = []
    if not disable_askcos:
        askcos_session = MockAskcosSession(
            seed=session_seed,
            avg_latency=0.05,
            latency_jitter=0.01,
            feasible_bias=0.82,
            failure_rate=0.03,
        )
        askcos = AskcosClient(
            base_url="https://mock.askcos",
            session=askcos_session,
            max_calls_per_minute=10_000,
        )
        oracles.append(askcos)
    aizynth = AiZynthClient(
        use_fallback=use_aizynth_fallback,
        config_path=aizynth_config,
        log_path=aizynth_log,
        restarts=aizynth_restarts,
    )
    oracles.append(aizynth)
    logger = OracleLogger("logs/oracle_calls.parquet")
    return PlannerInLoop(oracles, logger=logger)


def summarize(
    report: PlannerReport,
    k: int,
    oracle_details: Dict[str, dict],
    baseline_info: Optional[dict],
) -> dict:
    baseline_feasible = (
        baseline_info["feasible_at_k"] if baseline_info else report.baseline_feasible_at_k
    )
    summary = {
        "k": k,
        "feasible_at_k": report.feasible_at_k,
        "baseline_feasible_at_k": baseline_feasible,
        "improvement": report.improvement,
        "median_route_length": report.median_route_length,
        "median_latency": report.median_latency,
        "num_records": len(report.records),
        "oracle_stats": {},
        "oracle_details": oracle_details,
    }
    for name, stats in report.oracle_stats.items():
        ci_low, ci_high = wilson_interval(stats.feasible, stats.total, confidence=0.95)
        summary["oracle_stats"][name] = {
            "success_rate": stats.success_rate,
            "avg_latency": stats.avg_latency,
            "p95_latency": stats.p95_latency,
            "mean_score": stats.mean_score,
            "total": stats.total,
            "feasible": stats.feasible,
            "ci_95": [ci_low, ci_high],
            "median_route_length": stats.median_route_length,
            "throughput_per_minute": stats.throughput_per_minute,
        }
    if baseline_info:
        summary["baseline"] = baseline_info
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-molecules", type=int, default=320, help="Number of molecules to score")
    parser.add_argument("--k", type=int, default=10, help="Top-k for feasible@k metric")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed for molecule sampling")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("metrics/week2_planner_metrics.json"),
        help="Where to store the JSON summary",
    )
    parser.add_argument(
        "--smiles-file",
        type=Path,
        default=Path("data/moses.smi"),
        help="SMILES file to sample from (defaults to MOSES); falls back to synthetic samples if missing",
    )
    parser.add_argument(
        "--aizynth-config",
        type=Path,
        default=Path("config.yml"),
        help="Path to AiZynthFinder YAML configuration",
    )
    parser.add_argument(
        "--aizynth-fallback",
        action="store_true",
        help="Use heuristic AiZynth fallback instead of the full solver",
    )
    parser.add_argument(
        "--aizynth-log",
        type=Path,
        default=None,
        help="Optional log file capturing per-molecule AiZynth statistics",
    )
    parser.add_argument(
        "--baseline-aizynth-config",
        type=Path,
        default=None,
        help="Optional AiZynthFinder config for baseline (lower budget) evaluation",
    )
    parser.add_argument(
        "--disable-askcos",
        action="store_true",
        help="Skip ASKCOS and run AiZynthFinder only",
    )
    parser.add_argument(
        "--aizynth-restarts",
        type=int,
        default=1,
        help="Number of independent AiZynthFinder restarts to run for each molecule in the planner.",
    )
    parser.add_argument(
        "--hard-smiles-file",
        type=Path,
        default=Path("data/guacamol.smi"),
        help="Optional SMILES file containing more challenging molecules used when the baseline saturates.",
    )
    parser.add_argument(
        "--hard-heavy-threshold",
        type=int,
        default=45,
        help="Minimum heavy atom count used when resampling from the hard SMILES pool.",
    )
    parser.add_argument(
        "--hard-ring-threshold",
        type=int,
        default=9,
        help="Minimum ring count used when resampling from the hard SMILES pool.",
    )
    parser.add_argument(
        "--hard-hetero-threshold",
        type=int,
        default=10,
        help="Minimum hetero atom count used when resampling from the hard SMILES pool.",
    )
    parser.add_argument(
        "--baseline-ceiling",
        type=float,
        default=0.98,
        help="If baseline success rate exceeds this threshold, the script will resample from the hard pool unless disabled.",
    )
    parser.add_argument(
        "--baseline-floor",
        type=float,
        default=0.2,
        help="If baseline success rate drops below this threshold, the script will blend in easier molecules unless disabled.",
    )
    parser.add_argument(
        "--disable-auto-hard",
        action="store_true",
        help="Disable automatic resampling from the hard SMILES pool when the baseline saturates.",
    )
    parser.add_argument(
        "--disable-auto-easy",
        action="store_true",
        help="Disable automatic mixing of easier molecules when the baseline collapses.",
    )
    parser.add_argument(
        "--hard-pool-multiplier",
        type=int,
        default=4,
        help="How many times more molecules to sample when constructing the hard pool.",
    )
    parser.add_argument(
        "--easy-smiles-file",
        type=Path,
        default=None,
        help="Optional SMILES file containing easier molecules used when the baseline success is too low.",
    )
    parser.add_argument(
        "--easy-pool-multiplier",
        type=int,
        default=4,
        help="How many times more molecules to sample when constructing the easy pool.",
    )
    parser.add_argument(
        "--easy-heavy-threshold",
        type=int,
        default=30,
        help="Maximum heavy atom count retained in the easy pool (<=0 keeps all).",
    )
    parser.add_argument(
        "--easy-ring-threshold",
        type=int,
        default=6,
        help="Maximum ring count retained in the easy pool (<=0 keeps all).",
    )
    parser.add_argument(
        "--easy-hetero-threshold",
        type=int,
        default=8,
        help="Maximum hetero atom count retained in the easy pool (<=0 keeps all).",
    )
    parser.add_argument(
        "--baseline-aizynth-restarts",
        type=int,
        default=1,
        help="Number of AiZynthFinder restarts to use for the baseline evaluation.",
    )
    args = parser.parse_args()
    if args.easy_smiles_file is None:
        args.easy_smiles_file = args.smiles_file

    smiles = load_and_sanitize_smiles(args.smiles_file, args.n_molecules, args.seed)

    baseline_info = None
    baseline_feasible = 0.0
    if args.baseline_aizynth_config:
        baseline_log = None
        if args.aizynth_log is not None:
            baseline_log = args.aizynth_log.with_name(
                args.aizynth_log.stem + "_baseline" + args.aizynth_log.suffix
            )
        baseline_client = AiZynthClient(
            use_fallback=args.aizynth_fallback,
            config_path=args.baseline_aizynth_config,
            log_path=baseline_log,
            restarts=args.baseline_aizynth_restarts,
        )
        baseline_records = baseline_client.evaluate(smiles)
        baseline_stats = aggregate_records(baseline_records)
        smiles, baseline_records, escalated = _maybe_escalate_difficulty(
            smiles, baseline_client, baseline_records, args
        )
        if escalated:
            baseline_stats = aggregate_records(baseline_records)
        smiles, baseline_records, relaxed = _maybe_relax_difficulty(
            smiles, baseline_client, baseline_records, args
        )
        if relaxed:
            baseline_stats = aggregate_records(baseline_records)
        baseline_feasible = _compute_feasible_at_k(baseline_records, args.k)
        if escalated and relaxed:
            baseline_source = (
                f"mixed[{_format_path(args.hard_smiles_file)}, "
                f"{_format_path(args.easy_smiles_file)}]"
            )
        elif escalated:
            baseline_source = _format_path(args.hard_smiles_file)
        elif relaxed:
            baseline_source = (
                f"mixed[{_format_path(args.smiles_file)}, "
                f"{_format_path(args.easy_smiles_file)}]"
            )
        else:
            baseline_source = _format_path(args.smiles_file)
        baseline_info = {
            "config": str(args.baseline_aizynth_config),
            "summary": baseline_stats,
            "feasible_at_k": baseline_feasible,
            "source": baseline_source,
        }

    planner = build_planner(
        session_seed=args.seed,
        aizynth_config=args.aizynth_config,
        use_aizynth_fallback=args.aizynth_fallback,
        aizynth_log=args.aizynth_log,
        disable_askcos=args.disable_askcos,
        aizynth_restarts=args.aizynth_restarts,
    )

    if args.aizynth_log is not None:
        print(f"[INFO] AiZynth logs: {args.aizynth_log}")
    report = planner.evaluate(smiles, k=args.k, baseline_feasible_at_k=baseline_feasible)

    records_by_oracle: Dict[str, List] = {}
    for rec in report.records:
        records_by_oracle.setdefault(rec.oracle, []).append(rec)
    oracle_details = {name: aggregate_records(recs) for name, recs in records_by_oracle.items()}

    summary = summarize(report, args.k, oracle_details, baseline_info)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print("[INFO] Planner run complete")
    print(
        f"[INFO] Feasible@{args.k}: {summary['feasible_at_k']:.3f} "
        f"(baseline {summary['baseline_feasible_at_k']:.3f}, Δ {summary['improvement']:.3f})"
    )
    if baseline_info:
        print(
            f"[INFO] Baseline success={baseline_info['summary']['success_rate']:.2%} "
            f"@k={args.k}: {baseline_info['feasible_at_k']:.3f}"
        )
    for name, details in summary.get("oracle_details", {}).items():
        print(
            f"[INFO] {name} success={details['success_rate']:.2%} "
            f"median_latency={details['median_latency']:.2f}s routes≈{details['median_routes']:.1f}"
        )
    if baseline_info and summary["improvement"] < 0.05:
        print(
            "[WARN] Planner uplift Δ < 0.05 — consider tightening the hard pool or baseline "
            "configuration to demonstrate a clear advantage."
        )
    print(f"[INFO] Logged {summary['num_records']} oracle calls to logs/oracle_calls.parquet")


if __name__ == "__main__":
    main()
