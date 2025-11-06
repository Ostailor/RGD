#!/usr/bin/env python3
"""Download benchmark datasets for Reliable Generative Design Week 0-1."""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path
from typing import Dict, Iterable
import subprocess
import shutil

try:
    from importlib import resources
except ImportError:  # pragma: no cover
    import importlib_resources as resources  # type: ignore

import requests


DATASETS: Dict[str, Dict[str, str]] = {
    "moses": {
        "url": "https://media.githubusercontent.com/media/molecularsets/moses/master/data/dataset_v1.csv",
        "filename": "moses.csv",
        "sha256": "bb47a94d347afd476d3828b5e26dceeabc42a2d8cf92a791d00349f22fea0d8b",
        "description": "MOSES dataset (181k molecules). Source: Polykovskiy et al., 2018.",
    },
    "guacamol": {
        "url": "https://raw.githubusercontent.com/BenevolentAI/guacamol/master/data/guacamol_v1_all.smiles",
        "filename": "guacamol.smiles",
        "sha256": "",
        "description": "GuacaMol dataset (1.4M molecules). Source: Brown et al., 2019.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(DATASETS.keys()),
        help="Dataset to download.",
    )
    parser.add_argument(
        "--dest",
        default="data",
        type=str,
        help="Destination directory.",
    )
    return parser.parse_args()


def sha256sum(path: Path) -> str:
    hash_obj = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()


def download(dataset: str, dest: Path) -> Path:
    info = DATASETS[dataset]
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / info["filename"]

    existing = locate_existing_dataset(dataset, dest, info)
    if existing is not None:
        print(f"[INFO] Found existing {dataset} dataset at {existing}. Skipping download.")
        return existing

    print(f"[INFO] Downloading {dataset}: {info['description']}")
    response = requests.get(info["url"], timeout=60)
    used_fallback = False
    try:
        response.raise_for_status()
    except requests.HTTPError as err:
        if dataset == "guacamol" and err.response is not None and err.response.status_code == 404:
            print("[WARN] Remote GuacaMol dataset not found at primary URL; attempting package fallback.")
            try:
                target, digest = copy_guacamol_from_package(dest, target)
            except RuntimeError:
                print("[WARN] Package fallback unavailable; generating GuacaMol dataset via Docker.")
                target, digest = generate_guacamol_with_docker(dest)
            used_fallback = True
        else:
            raise
    else:
        target.write_bytes(response.content)
        digest = sha256sum(target)
    if not used_fallback and info["sha256"] and digest != info["sha256"]:
        target.unlink(missing_ok=True)  # type: ignore[arg-type]
        raise RuntimeError(
            f"SHA256 mismatch for {dataset}: expected {info['sha256']} got {digest}"
        )
    print(f"[INFO] Saved to {target} (sha256={digest})")
    if dataset == "moses":
        smiles_path = dest / "moses.smi"
        with target.open("r", newline="") as src, smiles_path.open("w", newline="") as out:
            reader = csv.DictReader(src)
            writer = csv.writer(out)
            writer.writerow(["smiles"])
            for row in reader:
                smiles = (row.get("SMILES") or row.get("smiles") or "").strip()
                if smiles:
                    writer.writerow([smiles])
        print(f"[INFO] Extracted SMILES with header to {smiles_path}")
    elif dataset == "guacamol":
        smiles_path = dest / "guacamol.smi"
        with smiles_path.open("w", newline="") as out:
            writer = csv.writer(out)
            writer.writerow(["smiles"])
            for line in target.read_text().splitlines():
                line = line.strip()
                if line:
                    writer.writerow([line])
        target = smiles_path
        digest = sha256sum(target)
        print(f"[INFO] Normalized SMILES with header to {smiles_path}")
    return target


def copy_guacamol_from_package(dest: Path, target: Path) -> tuple[Path, str]:
    try:
        data_path = resources.files("guacamol") / "data" / "guacamol_v1_all.smiles"  # type: ignore[attr-defined]
    except (AttributeError, ModuleNotFoundError):
        data_path = None
    if data_path is None or not data_path.is_file():
        raise RuntimeError(
            "GuacaMol dataset unavailable. Install `guacamol` package or provide data manually."
        )
    dest.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data_path.read_bytes())
    digest = sha256sum(target)
    print(f"[INFO] Copied dataset from installed guacamol package to {target} (sha256={digest})")
    # bypass hash check for package-sourced data
    smiles_path = dest / "guacamol.smi"
    with smiles_path.open("w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["smiles"])
        for line in target.read_text().splitlines():
            line = line.strip()
            if line:
                writer.writerow([line])
    digest = sha256sum(smiles_path)
    print(f"[INFO] Normalized SMILES with header to {smiles_path}")
    return smiles_path, digest


def locate_existing_dataset(dataset: str, dest: Path, info: Dict[str, str]) -> Path | None:
    direct_target = dest / info["filename"]
    if direct_target.exists():
        if info["sha256"]:
            try:
                digest = sha256sum(direct_target)
            except OSError:
                digest = None
            else:
                if digest == info["sha256"]:
                    return direct_target
                print(
                    f"[WARN] Existing {dataset} file at {direct_target} has unexpected hash "
                    f"{digest}; continuing to refresh."
                )
        else:
            return direct_target

    if dataset == "guacamol":
        combined = dest / "guacamol.smi"
        if combined.exists():
            return combined

        split_dir = dest / "guacamol"
        splits = [
            "chembl24_canon_train.smiles",
            "chembl24_canon_dev-valid.smiles",
            "chembl24_canon_test.smiles",
        ]
        if all((split_dir / name).exists() for name in splits):
            print("[INFO] Detected generated GuacaMol split files; creating guacamol.smi.")
            combined_data = combine_guacamol_splits(split_dir, splits, combined)
            print(f"[INFO] Wrote combined GuacaMol SMILES to {combined} ({combined_data} rows).")
            return combined
    return None


def combine_guacamol_splits(split_dir: Path, filenames: Iterable[str], output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["smiles"])
        for fname in filenames:
            path = split_dir / fname
            with path.open() as src:
                for line in src:
                    line = line.strip()
                    if not line:
                        continue
                    writer.writerow([line])
                    count += 1
    return count


def generate_guacamol_with_docker(dest: Path) -> tuple[Path, str]:
    if shutil.which("docker") is None:
        raise RuntimeError("Docker is required to generate the GuacaMol dataset; please install Docker Desktop.")

    repo_root = Path(__file__).resolve().parent.parent
    external_dir = repo_root / "external"
    repo_dir = external_dir / "guacamol"
    external_dir.mkdir(parents=True, exist_ok=True)

    if not repo_dir.exists():
        if shutil.which("git") is None:
            raise RuntimeError("Git is required to clone the GuacaMol repository for Docker generation.")
        print("[INFO] Cloning GuacaMol repository for Docker-based dataset generation.")
        subprocess.run(
            ["git", "clone", "https://github.com/BenevolentAI/guacamol.git", str(repo_dir)],
            check=True,
            cwd=external_dir,
        )

    print("[INFO] Building guacamol-deps Docker image (linux/amd64). This may take several minutes.")
    subprocess.run(
        [
            "docker",
            "build",
            "--platform=linux/amd64",
            "-t",
            "guacamol-deps",
            "-f",
            "dockers/Dockerfile",
            ".",
        ],
        check=True,
        cwd=repo_dir,
    )

    workspace = repo_root.resolve()
    try:
        dest_relative = dest.resolve().relative_to(workspace)
    except ValueError as exc:
        raise RuntimeError(
            f"Destination directory {dest} must be inside the project workspace {workspace} for Docker generation."
        ) from exc

    output_dir = dest / "guacamol"
    output_dir.mkdir(parents=True, exist_ok=True)

    docker_output = f"/workspace/{dest_relative}/guacamol"
    print("[INFO] Running guacamol-deps container to generate ChEMBL-based splits (this can take an hour).")
    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--platform=linux/amd64",
            "-v",
            f"{workspace}:/workspace",
            "-w",
            "/workspace/external/guacamol",
            "guacamol-deps",
            "python",
            "-m",
            "guacamol.data.get_data",
            "-o",
            docker_output,
        ],
        check=True,
    )

    splits = [
        "chembl24_canon_train.smiles",
        "chembl24_canon_dev-valid.smiles",
        "chembl24_canon_test.smiles",
    ]
    combined = dest / "guacamol.smi"
    combine_guacamol_splits(output_dir, splits, combined)
    digest = sha256sum(combined)
    print(f"[INFO] Generated GuacaMol dataset via Docker. Combined SMILES at {combined} (sha256={digest}).")
    return combined, digest


def main() -> None:
    args = parse_args()
    try:
        download(args.dataset, Path(args.dest))
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
