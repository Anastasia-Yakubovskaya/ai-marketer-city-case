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
DATA_PATH = ROOT / "data" / "processed" / "campaign_portfolio_assumptions.csv"
OUT_TABLES = ROOT / "outputs" / "tables"
OUT_META = ROOT / "outputs" / "meta"

SCORE_COLUMNS = [
    "value",
    "repeatability",
    "data_readiness",
    "automation_readiness",
    "risk",
]

REQUIRED_COLUMNS = [
    "campaign_id",
    "service",
    "channel",
    "offer_class",
    "description",
] + SCORE_COLUMNS

WEIGHT_KEYS = [
    "value",
    "repeatability",
    "data_readiness",
    "automation_readiness",
    "risk_penalty",
]

THRESHOLD_KEYS = [
    "pilot_first_min_score",
    "second_wave_min_score",
    "pilot_first_max_risk",
    "second_wave_max_risk",
]


def validate_config(cfg: dict) -> None:
    if "weights" not in cfg:
        raise ValueError("Missing portfolio.weights in config")
    if "thresholds" not in cfg:
        raise ValueError("Missing portfolio.thresholds in config")

    weights = cfg["weights"]
    thresholds = cfg["thresholds"]

    missing_weights = [k for k in WEIGHT_KEYS if k not in weights]
    if missing_weights:
        raise ValueError(f"Missing portfolio weights: {missing_weights}")

    missing_thresholds = [k for k in THRESHOLD_KEYS if k not in thresholds]
    if missing_thresholds:
        raise ValueError(f"Missing portfolio thresholds: {missing_thresholds}")

    for key in WEIGHT_KEYS:
        value = float(weights[key])
        if value < 0:
            raise ValueError(f"Portfolio weight {key} must be non-negative, got {value}")

    benefit_weight_sum = sum(float(weights[k]) for k in WEIGHT_KEYS if k != "risk_penalty")
    if benefit_weight_sum <= 0:
        raise ValueError("Sum of benefit weights must be positive")

    if float(thresholds["pilot_first_min_score"]) < float(thresholds["second_wave_min_score"]):
        raise ValueError("pilot_first_min_score must be >= second_wave_min_score")


def validate_dataframe(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in portfolio CSV: {missing}")

    if df.empty:
        raise ValueError("Portfolio CSV is empty")

    for col in SCORE_COLUMNS:
        series = pd.to_numeric(df[col], errors="coerce")
        if series.isna().any():
            bad_rows = df.loc[series.isna(), col]
            raise ValueError(f"Column {col} contains non-numeric values: {bad_rows.tolist()}")
        if (series < 1).any() or (series > 5).any():
            raise ValueError(f"Column {col} must contain values in range 1-5")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def config_hash(cfg: dict) -> str:
    cfg_str = json.dumps(cfg, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(cfg_str.encode("utf-8")).hexdigest()[:16]


def score_portfolio(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    validate_dataframe(df)

    weights = cfg["weights"]
    thresholds = cfg["thresholds"]

    x = df.copy()


    for col in SCORE_COLUMNS:
        x[col] = pd.to_numeric(x[col], errors="raise")
        x[f"{col}_normalized"] = (x[col] - 1.0) / 4.0

    benefit_weight_sum = sum(float(weights[k]) for k in WEIGHT_KEYS if k != "risk_penalty")

    x["priority_score"] = (
        float(weights["value"]) * x["value_normalized"]
        + float(weights["repeatability"]) * x["repeatability_normalized"]
        + float(weights["data_readiness"]) * x["data_readiness_normalized"]
        + float(weights["automation_readiness"]) * x["automation_readiness_normalized"]
        - float(weights["risk_penalty"]) * x["risk_normalized"]
    )


    x["score_share_of_max_benefit"] = x["priority_score"] / benefit_weight_sum

    pilot_mask = (
        (x["priority_score"] >= float(thresholds["pilot_first_min_score"]))
        & (x["risk"] <= float(thresholds["pilot_first_max_risk"]))
    )

    second_mask = (
        (x["priority_score"] >= float(thresholds["second_wave_min_score"]))
        & (x["risk"] <= float(thresholds["second_wave_max_risk"]))
    )

    x["recommended_wave"] = np.select(
        [pilot_mask, second_mask],
        ["PILOT_FIRST", "SECOND_WAVE"],
        default="HUMAN_FIRST",
    )

    return x.sort_values("priority_score", ascending=False).reset_index(drop=True)


def summarize_portfolio(scored: pd.DataFrame) -> pd.DataFrame:
    summary = (
        scored.groupby("recommended_wave")
        .agg(
            campaigns=("campaign_id", "count"),
            avg_priority_score=("priority_score", "mean"),
            min_priority_score=("priority_score", "min"),
            max_priority_score=("priority_score", "max"),
            avg_risk=("risk", "mean"),
            max_risk=("risk", "max"),
        )
        .reset_index()
    )

    wave_order = ["PILOT_FIRST", "SECOND_WAVE", "HUMAN_FIRST"]
    summary["recommended_wave"] = pd.Categorical(
        summary["recommended_wave"],
        categories=wave_order,
        ordered=True,
    )
    summary = summary.sort_values("recommended_wave").reset_index(drop=True)

    summary["campaign_share"] = summary["campaigns"] / summary["campaigns"].sum()

    return summary


def main():
    if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Portfolio data not found: {DATA_PATH}\n"
            f"Create data/processed/campaign_portfolio_assumptions.csv first."
        )

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        full_cfg = yaml.safe_load(f)

    cfg = full_cfg["portfolio"]
    validate_config(cfg)

    df = pd.read_csv(DATA_PATH, encoding="utf-8")
    scored = score_portfolio(df, cfg)
    summary = summarize_portfolio(scored)

    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_META.mkdir(parents=True, exist_ok=True)

    scored.to_csv(OUT_TABLES / "campaign_portfolio_priority.csv", index=False)
    summary.to_csv(OUT_TABLES / "campaign_portfolio_summary.csv", index=False)

    metadata = {
        "script": "portfolio.py",
        "timestamp": datetime.now().isoformat(),
        "config_path": str(CONFIG_PATH.relative_to(ROOT)),
        "data_path": str(DATA_PATH.relative_to(ROOT)),
        "config_hash": config_hash(cfg),
        "data_hash": file_hash(DATA_PATH),
        "rows": len(scored),
        "wave_counts": summary.set_index("recommended_wave")["campaigns"].to_dict(),
    }

    with open(OUT_META / "portfolio_priority.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("Portfolio priority scores:")
    print(
        scored[
            [
                "campaign_id",
                "service",
                "channel",
                "offer_class",
                "priority_score",
                "risk",
                "recommended_wave",
            ]
        ].to_string(index=False)
    )

    print("\nWave summary:")
    print(summary.to_string(index=False))

    print(f"\nConfig hash: {metadata['config_hash']}")
    print(f"Data hash: {metadata['data_hash']}")
    print(f"Saved to: {OUT_TABLES}")


if __name__ == "__main__":
    main()
