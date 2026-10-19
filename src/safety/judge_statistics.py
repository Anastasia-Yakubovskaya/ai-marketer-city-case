import sys
import hashlib
import json
from datetime import datetime
from math import ceil, log
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "assumptions.yaml"
OUT_TABLES = ROOT / "outputs" / "tables"
OUT_META = ROOT / "outputs" / "meta"


def validate_config(cfg: dict) -> None:
    required = ["confidence_level", "target_recalls", "example_clean_sample_sizes"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Missing required keys in judge_statistics config: {missing}")


def zero_failure_upper_bound(n: int, confidence_level: float = 0.95) -> float:
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be in (0, 1)")
    alpha = 1.0 - confidence_level
    return 1.0 - alpha ** (1.0 / n)


def required_zero_failure_sample(target_recall: float, confidence_level: float = 0.95) -> int:
    if not 0 < target_recall < 1:
        raise ValueError("target_recall must be in (0, 1)")
    alpha = 1.0 - confidence_level

    return ceil(log(alpha) / log(target_recall))


def cluster_adjusted_requirement(n_independent: int, cluster_size: int, icc: float) -> dict:
    if cluster_size < 1:
        raise ValueError("cluster_size must be >= 1")
    if not 0 <= icc < 1:
        raise ValueError("icc must be in [0, 1)")
    design_effect = 1.0 + (cluster_size - 1) * icc
    return {
        "n_assuming_independence": n_independent,
        "cluster_size": cluster_size,
        "icc": icc,
        "design_effect": design_effect,
        "n_required_adjusted": ceil(n_independent * design_effect),
    }


def cohens_kappa(confusion_matrix) -> float:
    m = np.asarray(confusion_matrix, dtype=float)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError("confusion_matrix must be square")
    total = m.sum()
    if total <= 0:
        raise ValueError("confusion_matrix must contain observations")
    observed = np.trace(m) / total
    row = m.sum(axis=1) / total
    col = m.sum(axis=0) / total
    expected = float((row * col).sum())
    if expected >= 1.0 - 1e-12:


        return float("nan")
    return float((observed - expected) / (1.0 - expected))


def kappa_under_accuracy(prevalence: float, sensitivity: float, specificity: float) -> float:
    p, s, sp = prevalence, sensitivity, specificity
    m = np.array([
        [p * s * s + (1 - p) * (1 - sp) ** 2,
         p * s * (1 - s) + (1 - p) * (1 - sp) * sp],
        [p * (1 - s) * s + (1 - p) * sp * (1 - sp),
         p * (1 - s) ** 2 + (1 - p) * sp * sp],
    ])
    return cohens_kappa(m)


def build_tables(cfg: dict):
    confidence = float(cfg["confidence_level"])
    unit = str(cfg.get("unit_of_observation", "creative"))
    tables = {}


    evidence_rows = []
    for n in cfg["example_clean_sample_sizes"]:
        n = int(n)
        upper = zero_failure_upper_bound(n, confidence)
        evidence_rows.append({
            "n_clean_zero_misses": n,
            "unit_of_observation": unit,
            "confidence_level": confidence,
            "upper_miss_rate": upper,
            "lower_recall_bound": 1.0 - upper,
            "note": "upper bound, not a point estimate; independence assumed",
        })
    tables["evidence"] = pd.DataFrame(evidence_rows)


    requirement_rows = []
    for recall in cfg["target_recalls"]:
        recall = float(recall)
        n = required_zero_failure_sample(recall, confidence)
        requirement_rows.append({
            "target_recall": recall,
            "max_miss_rate": 1.0 - recall,
            "confidence_level": confidence,
            "required_zero_miss_examples": n,
            "achieved_upper_miss_rate": zero_failure_upper_bound(n, confidence),
            "unit_of_observation": unit,
        })
    tables["requirements"] = pd.DataFrame(requirement_rows)


    clustering = cfg.get("clustering")
    if clustering:
        cluster_rows = []
        for _, row in tables["requirements"].iterrows():
            adj = cluster_adjusted_requirement(
                int(row["required_zero_miss_examples"]),
                int(clustering["cluster_size"]),
                float(clustering["icc"]),
            )
            adj["target_recall"] = float(row["target_recall"])
            cluster_rows.append(adj)
        tables["clustering"] = pd.DataFrame(cluster_rows)


    kappa_cfg = cfg.get("kappa_sensitivity", {})
    sensitivity = float(kappa_cfg.get("assumed_sensitivity", 0.95))
    specificity = float(kappa_cfg.get("assumed_specificity", 0.99))
    grid = kappa_cfg.get("prevalence_grid", [0.005, 0.01, 0.02, 0.05, 0.10])
    kappa_rows = []
    for prev in grid:
        kappa_rows.append({
            "violation_prevalence": float(prev),
            "assumed_sensitivity": sensitivity,
            "assumed_specificity": specificity,
            "cohens_kappa": kappa_under_accuracy(float(prev), sensitivity, specificity),
            "status": "illustration, not a measurement",
        })
    tables["kappa"] = pd.DataFrame(kappa_rows)

    return tables


def compute_config_hash(cfg: dict) -> str:
    cfg_str = json.dumps(cfg, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(cfg_str.encode("utf-8")).hexdigest()[:16]


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        full_cfg = yaml.safe_load(f)

    cfg = full_cfg["judge_statistics"]
    validate_config(cfg)

    tables = build_tables(cfg)

    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_META.mkdir(parents=True, exist_ok=True)

    filenames = {
        "evidence": "judge_upper_bounds.csv",
        "requirements": "judge_required_samples.csv",
        "clustering": "judge_clustering_adjustment.csv",
        "kappa": "kappa_prevalence_sensitivity.csv",
    }
    for key, frame in tables.items():
        frame.to_csv(OUT_TABLES / filenames[key], index=False)

    metadata = {
        "script": "judge_statistics.py",
        "timestamp": datetime.now().isoformat(),
        "config_hash": compute_config_hash(cfg),
        "deterministic": True,
        "seed_required": False,
        "tables_written": sorted(tables.keys()),
    }
    with open(OUT_META / "judge_statistics.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("Zero-miss evidence (upper bound, not a point estimate):")
    print(tables["evidence"].to_string(index=False))
    print("\nRequired sample sizes for a target recall:")
    print(tables["requirements"].to_string(index=False))
    if "clustering" in tables:
        print("\nCluster-adjusted requirement (conservative):")
        print(tables["clustering"].to_string(index=False))
    print("\nKappa prevalence sensitivity (illustration):")
    print(tables["kappa"].to_string(index=False))
    print(f"\nConfig hash: {metadata['config_hash']}")
    print(f"Saved to: {OUT_TABLES}")


if __name__ == "__main__":
    main()
