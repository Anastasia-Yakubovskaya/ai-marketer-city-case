import sys
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "assumptions.yaml"
INCIDENTS_PATH = ROOT / "data" / "processed" / "incidents.csv"
OUT_TABLES = ROOT / "outputs" / "tables"
OUT_META = ROOT / "outputs" / "meta"


NORMALIZER = 10_000

VALID_SEVERITIES = {"low", "medium", "high", "critical"}

REQUIRED_INCIDENT_COLUMNS = {"severity", "exposed_contacts"}


def validate_config(cfg: dict) -> None:
    required = ["monthly_budget_points", "severity_weights", "burn_thresholds"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Missing required keys in error_budget config: {missing}")

    weights = cfg["severity_weights"]
    missing_sev = VALID_SEVERITIES - set(weights.keys())
    if missing_sev:
        raise ValueError(f"severity_weights missing severities: {sorted(missing_sev)}")

    thresholds = cfg["burn_thresholds"]
    required_th = ["normal_max", "freeze_max", "reduce_max"]
    missing_th = [k for k in required_th if k not in thresholds]
    if missing_th:
        raise ValueError(f"burn_thresholds missing keys: {missing_th}")

    n, f, r = (float(thresholds[k]) for k in required_th)
    if not (0 < n <= f <= r):
        raise ValueError(
            f"burn_thresholds must satisfy 0 < normal_max <= freeze_max <= reduce_max, "
            f"got {n}, {f}, {r}"
        )

    if float(cfg["monthly_budget_points"]) <= 0:
        raise ValueError("monthly_budget_points must be positive")


def validate_incidents(inc: pd.DataFrame) -> None:
    if inc is None or inc.empty:
        return
    missing = REQUIRED_INCIDENT_COLUMNS - set(inc.columns)
    if missing:
        raise ValueError(f"Incident data missing columns: {sorted(missing)}")
    bad_sev = set(inc["severity"].dropna()) - VALID_SEVERITIES
    if bad_sev:
        raise ValueError(f"Unknown severities in incident data: {sorted(bad_sev)}")
    if pd.to_numeric(inc["exposed_contacts"], errors="coerce").isna().any():
        raise ValueError("exposed_contacts must be numeric")
    if (pd.to_numeric(inc["exposed_contacts"], errors="coerce") < 0).any():
        raise ValueError("exposed_contacts must be non-negative")


def incident_points(severity: str, exposed_contacts: float, weights: dict) -> float:
    if severity not in weights:
        raise ValueError(f"Unknown severity: {severity}")
    return float(weights[severity]) * max(float(exposed_contacts), 1.0) / NORMALIZER


def action(burn: float, thresholds: dict, critical: bool = False) -> str:
    if critical:
        return "STOP_AND_DEMOTE"
    if burn <= float(thresholds["normal_max"]):
        return "NORMAL"
    if burn <= float(thresholds["freeze_max"]):
        return "FREEZE_PROMOTION"
    if burn <= float(thresholds["reduce_max"]):
        return "REDUCE_COVERAGE"
    return "DEMOTE_TO_HUMAN_APPROVAL"


def evaluate(inc: Optional[pd.DataFrame], cfg: dict) -> dict:
    budget = float(cfg["monthly_budget_points"])
    weights = cfg["severity_weights"]
    thresholds = cfg["burn_thresholds"]

    if inc is None or inc.empty:
        return {
            "budget_points": budget,
            "used_points": 0.0,
            "burn_rate": 0.0,
            "critical_incident": False,
            "incident_count": 0,
            "action": action(0.0, thresholds, False),
        }

    validate_incidents(inc)
    used = sum(
        incident_points(s, n, weights)
        for s, n in zip(inc["severity"], inc["exposed_contacts"])
    )
    critical = bool((inc["severity"] == "critical").any())
    burn = used / budget if budget > 0 else float("inf")
    return {
        "budget_points": budget,
        "used_points": used,
        "burn_rate": burn,
        "critical_incident": critical,
        "incident_count": len(inc),
        "action": action(burn, thresholds, critical),
    }


def build_scenarios() -> dict[str, pd.DataFrame]:
    raw = {
        "clean_month": [],
        "few_low_small": [
            ("low", 2000, 1), ("low", 1500, 2), ("low", 3000, 3),
        ],
        "one_medium": [
            ("medium", 5000, 2),
        ],
        "one_high": [
            ("high", 20000, 3),
        ],
        "one_critical_small": [
            ("critical", 5000, 2),
        ],
        "one_critical_large": [
            ("critical", 100000, 3),
        ],
        "mixed_escalating": [
            ("low", 3000, 1),
            ("medium", 8000, 2),
            ("high", 15000, 3),
            ("medium", 6000, 4),
        ],
        "two_high_one_critical": [
            ("high", 10000, 1),
            ("high", 12000, 2),
            ("critical", 8000, 3),
        ],
    }
    return {
        name: pd.DataFrame(rows, columns=["severity", "exposed_contacts", "week"])
        for name, rows in raw.items()
    }


def scenario_table(scenarios: dict, cfg: dict) -> pd.DataFrame:
    rows = []
    for name, inc in scenarios.items():
        status = evaluate(inc, cfg)
        status["scenario"] = name
        rows.append(status)
    df = pd.DataFrame(rows)
    cols = ["scenario", "incident_count", "used_points", "budget_points",
            "burn_rate", "critical_incident", "action"]
    return df[cols].sort_values("burn_rate", ascending=False).reset_index(drop=True)


def monthly_trajectory(scenarios: dict, cfg: dict, weeks: int = 4) -> pd.DataFrame:
    budget = float(cfg["monthly_budget_points"])
    weights = cfg["severity_weights"]
    thresholds = cfg["burn_thresholds"]
    rows = []
    for name, inc in scenarios.items():
        cumulative = 0.0
        critical_seen = False
        for w in range(1, weeks + 1):
            week_inc = inc[inc["week"] == w] if "week" in inc.columns else pd.DataFrame()
            for _, r in week_inc.iterrows():
                cumulative += incident_points(r["severity"], r["exposed_contacts"], weights)
                if r["severity"] == "critical":
                    critical_seen = True
            burn = cumulative / budget if budget > 0 else float("inf")
            rows.append({
                "scenario": name,
                "week": w,
                "cumulative_points": cumulative,
                "burn_rate": burn,
                "critical_seen": critical_seen,
                "action": action(burn, thresholds, critical_seen),
            })
    return pd.DataFrame(rows)


def live_status(cfg: dict) -> Optional[pd.DataFrame]:
    if not INCIDENTS_PATH.exists():
        print(f"[skip] no live incident feed at {INCIDENTS_PATH}; "
              f"scenarios and trajectory are still produced.")
        return None
    inc = pd.read_csv(INCIDENTS_PATH, encoding="utf-8")
    status = evaluate(inc, cfg)
    status["source"] = "live_incidents.csv"
    return pd.DataFrame([status])


def compute_config_hash(cfg: dict) -> str:
    cfg_str = json.dumps(cfg, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(cfg_str.encode("utf-8")).hexdigest()[:16]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        full_cfg = yaml.safe_load(f)


    if "error_budget" in full_cfg:
        cfg = full_cfg["error_budget"]
    elif "autonomy_rule" in full_cfg and "error_budget" in full_cfg["autonomy_rule"]:
        cfg = full_cfg["autonomy_rule"]["error_budget"]
    else:
        raise KeyError("error_budget section not found in config")

    validate_config(cfg)

    scenarios = build_scenarios()
    st = scenario_table(scenarios, cfg)
    traj = monthly_trajectory(scenarios, cfg)
    live = live_status(cfg)

    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_META.mkdir(parents=True, exist_ok=True)

    st.to_csv(OUT_TABLES / "error_budget_scenarios.csv", index=False)
    traj.to_csv(OUT_TABLES / "error_budget_trajectory.csv", index=False)
    if live is not None:
        live.to_csv(OUT_TABLES / "error_budget_status.csv", index=False)

    metadata = {
        "script": "error_budget.py",
        "timestamp": datetime.now().isoformat(),
        "config_hash": compute_config_hash(cfg),
        "normalizer_contacts": NORMALIZER,
        "scenarios": len(scenarios),
        "weeks": 4,
        "live_feed_present": live is not None,
        "tables_written": [
            "error_budget_scenarios.csv",
            "error_budget_trajectory.csv",
        ] + (["error_budget_status.csv"] if live is not None else []),
    }
    with open(OUT_META / "error_budget.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("Error budget scenarios (month-end view):")
    print(st.to_string(index=False))
    print("\nWeekly trajectory (when thresholds are crossed):")
    print(traj.to_string(index=False))
    if live is not None:
        print("\nLive status from incidents.csv:")
        print(live.to_string(index=False))
    print(f"\nConfig hash: {metadata['config_hash']}")
    print(f"Normalizer: {NORMALIZER} contacts = 1 base incident unit")
    print(f"Saved to: {OUT_TABLES}")


if __name__ == "__main__":
    main()
