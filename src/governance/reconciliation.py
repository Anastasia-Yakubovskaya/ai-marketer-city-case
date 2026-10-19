
import hashlib
import json
from datetime import datetime
from pathlib import Path
import sys
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "assumptions.yaml"
OUT_TABLES = ROOT / "outputs" / "tables"
OUT_META = ROOT / "outputs" / "meta"


def validate_config(cfg: dict) -> None:
    required = [
        "team_fte",
        "fte_cost_m_rub_per_month",
        "reported_team_cost_m_rub_per_month",
        "reported_launch_days",
        "case_step_days",
    ]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Missing required keys in reconciliation config: {missing}")


def reconcile(cfg: dict) -> pd.DataFrame:
    rows = []


    team_fte = float(cfg["team_fte"])
    fte_cost = float(cfg["fte_cost_m_rub_per_month"])
    reported_team_cost = float(cfg["reported_team_cost_m_rub_per_month"])
    # Несостыковки исходного кейса не исправляем молча: фиксируем обе версии рядом.
    calculated_team_cost = team_fte * fte_cost

    rows.append({
        "check": "team_cost",
        "reported_value": reported_team_cost,
        "calculated_value": calculated_team_cost,
        "unit": "m RUB/month",
        "absolute_gap": calculated_team_cost - reported_team_cost,
        "relative_gap": calculated_team_cost / reported_team_cost - 1,
        "status": "MISMATCH" if abs(calculated_team_cost - reported_team_cost) > 1e-9 else "OK",
        "interpretation": (
            f"{team_fte:g} FTE x {fte_cost:g} = {calculated_team_cost:.2f}, "
            f"while the case reports {reported_team_cost:g}. "
            f"Gap {abs(calculated_team_cost - reported_team_cost):.2f} m RUB/month "
            f"is named explicitly as a reconciliation item."
        ),
    })


    # Простая сумма шагов — диагностическая проверка; process_model отдельно допускает параллельность.
    step_days = cfg["case_step_days"]
    lower = sum(float(v[0]) for v in step_days.values())
    mode = sum(float(v[1]) for v in step_days.values())
    upper = sum(float(v[2]) for v in step_days.values())
    reported_launch = float(cfg["reported_launch_days"])

    rows.append({
        "check": "launch_duration_lower_bound",
        "reported_value": reported_launch,
        "calculated_value": lower,
        "unit": "working days",
        "absolute_gap": lower - reported_launch,
        "relative_gap": lower / reported_launch - 1,
        "status": "MISMATCH" if lower > reported_launch else "OK",
        "interpretation": (
            f"Sum of minimum step durations ({lower:.1f} days) exceeds the stated "
            f"{reported_launch:g} days; some steps must overlap or sit outside the "
            f"measured lead time."
        ),
    })

    rows.append({
        "check": "launch_duration_mode_sum",
        "reported_value": reported_launch,
        "calculated_value": mode,
        "unit": "working days",
        "absolute_gap": mode - reported_launch,
        "relative_gap": mode / reported_launch - 1,
        "status": "MISMATCH" if abs(mode - reported_launch) > 1e-9 else "OK",
        "interpretation": (
            f"Naive sequential summation of mode durations ({mode:.1f} days) is not a "
            f"valid reconstruction of the {reported_launch:g}-day process; "
            f"parallelism is modeled in process/process_model.py."
        ),
    })

    rows.append({
        "check": "launch_duration_upper_bound",
        "reported_value": reported_launch,
        "calculated_value": upper,
        "unit": "working days",
        "absolute_gap": upper - reported_launch,
        "relative_gap": upper / reported_launch - 1,
        "status": "MISMATCH" if upper > reported_launch else "OK",
        "interpretation": (
            f"Upper-end sequential duration ({upper:.1f} days) quantifies the maximum "
            f"inconsistency before modeling parallelism."
        ),
    })

    return pd.DataFrame(rows)


def compute_config_hash(cfg: dict) -> str:
    cfg_str = json.dumps(cfg, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(cfg_str.encode("utf-8")).hexdigest()[:16]


def main():
    if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Config not found: {CONFIG_PATH}\n"
            f"Create config/assumptions.yaml with a 'reconciliation' section."
        )

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        full_cfg = yaml.safe_load(f)

    cfg = full_cfg["reconciliation"]
    validate_config(cfg)

    out = reconcile(cfg)

    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_META.mkdir(parents=True, exist_ok=True)

    out.to_csv(OUT_TABLES / "reconciliation.csv", index=False)

    metadata = {
        "script": "reconciliation.py",
        "timestamp": datetime.now().isoformat(),
        "config_path": str(CONFIG_PATH.relative_to(ROOT)),
        "config_hash": compute_config_hash(cfg),
        "checks": len(out),
        "mismatches": int((out["status"] == "MISMATCH").sum()),
    }
    with open(OUT_META / "reconciliation.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(out.to_string(index=False))
    print()
    print(f"Config hash: {metadata['config_hash']}")
    print(f"Checks: {metadata['checks']}, mismatches: {metadata['mismatches']}")
    print(f"Saved: {OUT_TABLES / 'reconciliation.csv'}")


if __name__ == "__main__":
    main()
