import sys
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "assumptions.yaml"
SEEDS_PATH = ROOT / "config" / "seeds.yaml"
OUT_TABLES = ROOT / "outputs" / "tables"
OUT_META = ROOT / "outputs" / "meta"


def validate_config(cfg: dict) -> None:
    required = ["n_simulations", "as_is", "to_be", "analysis_as_is_days",
                "analysis_to_be_days", "observation_windows_days"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Missing required keys in process_model config: {missing}")


def tri(rng, params, size):
    low, mode, high = params
    if not (low <= mode <= high):
        raise ValueError(f"Triangular params must satisfy low <= mode <= high, got {params}")
    return rng.triangular(low, mode, high, size=size)


def critical_path(rng, steps, n):
    finish = {}
    for name, spec in steps.items():
        # Этап стартует после самой поздней из зависимостей.
        deps = spec.get("depends_on", [])
        if deps:

            dep_arrays = np.stack([finish[d] for d in deps])
            start = np.max(dep_arrays, axis=0)
        else:
            start = np.zeros(n)
        finish[name] = start + tri(rng, spec["days"], n)

    return np.max(np.stack(list(finish.values())), axis=0)


def summarize(arr, metric, scenario, window=None):
    return {
        "metric": metric,
        "scenario": scenario,
        "observation_window_days": window if window is not None else 0,
        "p10": float(np.quantile(arr, 0.10)),
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
        "mean": float(np.mean(arr)),
    }


def simulate_process(cfg, seed=42):
    rng = np.random.default_rng(seed)
    n = int(cfg["n_simulations"])
    rows = []


    totals = {}
    for state in ("as_is", "to_be"):
        totals[state] = critical_path(rng, cfg[state], n)
        rows.append(summarize(totals[state], "time_to_launch", state))


    analysis_as_is = tri(rng, cfg["analysis_as_is_days"], n)
    analysis_to_be = tri(rng, cfg["analysis_to_be_days"], n)

    for w in cfg["observation_windows_days"]:
        for state, analysis in [("as_is", analysis_as_is), ("to_be", analysis_to_be)]:
            full_time = totals[state] + w + analysis
            rows.append(summarize(full_time, "time_to_evidence", state, window=w))

    return pd.DataFrame(rows)


def compute_config_hash(cfg: dict) -> str:
    cfg_str = json.dumps(cfg, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(cfg_str.encode("utf-8")).hexdigest()[:16]


def main():
    if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")
    if not SEEDS_PATH.exists():
        raise FileNotFoundError(f"Seeds not found: {SEEDS_PATH}")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        full_cfg = yaml.safe_load(f)
    cfg = full_cfg["process_model"]
    validate_config(cfg)

    with open(SEEDS_PATH, "r", encoding="utf-8") as f:
        seeds = yaml.safe_load(f)
    seed = int(seeds["global_seed"])

    out = simulate_process(cfg, seed)

    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_META.mkdir(parents=True, exist_ok=True)

    out.to_csv(OUT_TABLES / "process_summary.csv", index=False)

    metadata = {
        "script": "process_model.py",
        "timestamp": datetime.now().isoformat(),
        "config_hash": compute_config_hash(cfg),
        "seed": seed,
        "n_simulations": int(cfg["n_simulations"]),
        "rows": len(out),
    }
    with open(OUT_META / "process_model.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(out.to_string(index=False))
    print()
    print(f"Config hash: {metadata['config_hash']}")
    print(f"Seed: {seed}, simulations: {metadata['n_simulations']}")
    print(f"Saved: {OUT_TABLES / 'process_summary.csv'}")


if __name__ == "__main__":
    main()
