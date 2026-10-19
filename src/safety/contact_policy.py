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
    required = ["n_clients", "n_weeks", "weekly_campaign_pressure_grid",
                "message_cap_per_week", "sms_cap_per_week", "sms_share",
                "optout_share", "value_distribution"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Missing required keys in contact_policy_model config: {missing}")


def simulate_local_checks(rng, n_clients, weekly_pressure, sms_share, optout_share,
                          message_cap=4, sms_cap=1):
    candidate_count = rng.poisson(weekly_pressure, size=n_clients)
    opted_out = rng.random(n_clients) < optout_share
    sent_count = np.where(opted_out, 0, candidate_count)

    sms_count = np.zeros(n_clients, dtype=int)
    active = sent_count > 0
    sms_count[active] = rng.binomial(sent_count[active], sms_share)

    total_violation = sent_count > message_cap
    sms_violation = sms_count > sms_cap
    optout_violation = opted_out & (sent_count > 0)
    any_violation = total_violation | sms_violation | optout_violation

    return pd.DataFrame({
        'candidate_count': candidate_count,
        'sent_count': sent_count,
        'sms_count': sms_count,
        'opted_out': opted_out,
        'total_violation': total_violation,
        'sms_violation': sms_violation,
        'optout_violation': optout_violation,
        'any_violation': any_violation,
    })


def _select_for_one_client(values, is_sms, message_cap, sms_cap):
    order = np.argsort(values)[::-1]
    selected = []
    total_used = 0
    sms_used = 0

    for idx in order:
        if total_used >= message_cap:
            break
        if is_sms[idx]:
            if sms_used >= sms_cap:
                continue
            sms_used += 1
        selected.append(idx)
        total_used += 1

    return np.asarray(selected, dtype=int)


def simulate_central_arbiter(rng, n_clients, weekly_pressure, sms_share, optout_share,
                             value_params, message_cap=4, sms_cap=1):
    candidate_count = rng.poisson(weekly_pressure, size=n_clients)
    opted_out = rng.random(n_clients) < optout_share

    sent_count = np.zeros(n_clients, dtype=int)
    sms_count = np.zeros(n_clients, dtype=int)
    candidate_value = np.zeros(n_clients, dtype=float)
    selected_value = np.zeros(n_clients, dtype=float)

    low = float(value_params['low'])
    mode = float(value_params['mode'])
    high = float(value_params['high'])

    for i, n in enumerate(candidate_count):
        if n == 0:
            continue

        values = rng.triangular(low, mode, high, size=n)
        candidate_value[i] = values.sum()

        if opted_out[i]:
            continue

        is_sms = rng.random(n) < sms_share
        selected = _select_for_one_client(values, is_sms, message_cap, sms_cap)
        sent_count[i] = len(selected)
        sms_count[i] = int(is_sms[selected].sum()) if len(selected) else 0
        selected_value[i] = float(values[selected].sum()) if len(selected) else 0.0

    total_violation = sent_count > message_cap
    sms_violation = sms_count > sms_cap
    optout_violation = opted_out & (sent_count > 0)
    any_violation = total_violation | sms_violation | optout_violation

    feasible_candidate_value = candidate_value[~opted_out].sum()
    value_retention = (selected_value.sum() / feasible_candidate_value
                       if feasible_candidate_value > 0 else 1.0)

    return pd.DataFrame({
        'candidate_count': candidate_count,
        'sent_count': sent_count,
        'sms_count': sms_count,
        'opted_out': opted_out,
        'candidate_value': candidate_value,
        'selected_value': selected_value,
        'total_violation': total_violation,
        'sms_violation': sms_violation,
        'optout_violation': optout_violation,
        'any_violation': any_violation,
    }), value_retention


def summarize(df, strategy, pressure, value_retention=np.nan):
    return {
        'strategy': strategy,
        'weekly_pressure': float(pressure),
        'clients': int(len(df)),
        'mean_sent_messages': float(df['sent_count'].mean()),
        'p95_sent_messages': float(df['sent_count'].quantile(0.95)),
        'total_cap_violation_rate': float(df['total_violation'].mean()),
        'sms_cap_violation_rate': float(df['sms_violation'].mean()),
        'optout_violation_rate': float(df['optout_violation'].mean()),
        'any_policy_violation_rate': float(df['any_violation'].mean()),
        'value_retention': (float(value_retention) if not np.isnan(value_retention) else np.nan),
        'value_loss': (float(1 - value_retention) if not np.isnan(value_retention) else np.nan),
    }


def run_pressure_grid(config, seed=42):
    base_rng = np.random.default_rng(seed)
    rows = []

    for pressure in config['weekly_campaign_pressure_grid']:
        week_rows = {'local_checks': [], 'central_arbiter': []}

        for _ in range(int(config['n_weeks'])):
            local = simulate_local_checks(
                np.random.default_rng(int(base_rng.integers(0, 2**32 - 1))),
                int(config['n_clients']), float(pressure), float(config['sms_share']),
                float(config['optout_share']), int(config['message_cap_per_week']),
                int(config['sms_cap_per_week']))
            week_rows['local_checks'].append(summarize(local, 'local_checks', pressure))

            central, retention = simulate_central_arbiter(
                np.random.default_rng(int(base_rng.integers(0, 2**32 - 1))),
                int(config['n_clients']), float(pressure), float(config['sms_share']),
                float(config['optout_share']), config['value_distribution'],
                int(config['message_cap_per_week']), int(config['sms_cap_per_week']))
            week_rows['central_arbiter'].append(
                summarize(central, 'central_arbiter', pressure, retention))

        for strategy, values in week_rows.items():
            tmp = pd.DataFrame(values)
            row = {'strategy': strategy, 'weekly_pressure': float(pressure),
                   'clients': int(config['n_clients'])}
            for col in tmp.select_dtypes(include=[np.number]).columns:
                if col not in {'weekly_pressure', 'clients'}:
                    row[col] = float(tmp[col].mean())
            rows.append(row)

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
    cfg = full_cfg["contact_policy_model"]
    validate_config(cfg)

    with open(SEEDS_PATH, "r", encoding="utf-8") as f:
        seeds = yaml.safe_load(f)
    seed = int(seeds["global_seed"])

    result = run_pressure_grid(cfg, seed)

    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_META.mkdir(parents=True, exist_ok=True)

    result.to_csv(OUT_TABLES / "contact_policy_violations.csv", index=False)

    metadata = {
        "script": "contact_policy.py",
        "timestamp": datetime.now().isoformat(),
        "config_hash": compute_config_hash(cfg),
        "seed": seed,
        "n_clients": int(cfg["n_clients"]),
        "n_weeks": int(cfg["n_weeks"]),
        "pressure_levels": len(cfg["weekly_campaign_pressure_grid"]),
        "rows": len(result),
    }
    with open(OUT_META / "contact_policy.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(result.to_string(index=False))
    print(f"\nConfig hash: {metadata['config_hash']}")
    print(f"Seed: {seed}, clients: {metadata['n_clients']}, weeks: {metadata['n_weeks']}")
    print(f"Saved to: {OUT_TABLES / 'contact_policy_violations.csv'}")


if __name__ == "__main__":
    main()
