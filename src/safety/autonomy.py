import sys
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "assumptions.yaml"
OUT_TABLES = ROOT / "outputs" / "tables"
OUT_META = ROOT / "outputs" / "meta"


def validate_config(cfg: dict) -> None:
    required = [
        "confidence_level",
        "tiers",
        "damage_per_contact_rub",
        "max_expected_damage_rub",
        "canary_share",
        "canary_detection_prob",
        "true_error_rate_grid",
        "sprt",
        "error_budget",
    ]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Missing required keys in autonomy_rule config: {missing}")

    for tier_name, tier in cfg["tiers"].items():
        if "target_recall" not in tier:
            raise ValueError(f"Tier {tier_name} missing target_recall")


def cp_upper_bound(k: int, n: int, confidence_level: float) -> float:
    if n <= 0:
        raise ValueError("n must be positive")
    if k < 0 or k > n:
        raise ValueError("k must be in [0, n]")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be in (0, 1)")

    alpha = 1.0 - confidence_level


    if k == n:
        return 1.0


    return float(stats.beta.ppf(1.0 - alpha, k + 1, n - k))


def required_zero_failure_sample(target_recall: float, confidence_level: float = 0.95) -> int:
    if not 0 < target_recall < 1:
        raise ValueError("target_recall must be in (0, 1)")
    alpha = 1.0 - confidence_level
    return int(math.ceil(math.log(alpha) / math.log(target_recall)))


def max_allowed_errors(n: int, p_bad: float, alpha: float) -> int:
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 < p_bad < 1:
        raise ValueError("p_bad must be in (0, 1)")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")

    best = -1
    for k in range(n + 1):
        prob = float(stats.binom.cdf(k, n, p_bad))
        if prob <= alpha + 1e-12:
            best = k
        else:
            break
    return best


def canary_effectiveness_phi(q: float, pi: float) -> float:
    if not 0 <= q <= 1:
        raise ValueError("q must be in [0, 1]")
    if not 0 <= pi <= 1:
        raise ValueError("pi must be in [0, 1]")
    return float(q + (1.0 - q) * (1.0 - pi))


def build_thresholds(cfg: dict) -> pd.DataFrame:
    confidence = float(cfg["confidence_level"])
    d = float(cfg["damage_per_contact_rub"])
    ed_max = float(cfg["max_expected_damage_rub"])
    q = float(cfg["canary_share"])
    pi = float(cfg["canary_detection_prob"])
    phi = canary_effectiveness_phi(q, pi)

    rows = []

    for tier_name, tier in cfg["tiers"].items():
        target_recall = float(tier["target_recall"])
        unacceptable_miss_rate = 1.0 - target_recall
        alpha = 1.0 - confidence

        n = required_zero_failure_sample(target_recall, confidence)
        k = max_allowed_errors(n, unacceptable_miss_rate, alpha)

        if k < 0:

            raise ValueError(
                f"Tier {tier_name}: no promotion rule can control type-I error "
                f"at n={n}, p_bad={unacceptable_miss_rate}, alpha={alpha}"
            )

        # Верхняя граница ошибки, а не наблюдаемая доля ошибок, определяет допустимый охват.
        p_u = cp_upper_bound(k, n, confidence)


        if p_u <= 0:
            r_max_no_canary = float("inf")
            r_max_with_canary = float("inf")
        else:
            r_max_no_canary = ed_max / (p_u * d)
            r_max_with_canary = ed_max / (p_u * d * phi)

        rows.append({
            "tier": tier_name,
            "description": tier.get("description", ""),
            "target_recall": target_recall,
            "unacceptable_miss_rate": unacceptable_miss_rate,
            "confidence_level": confidence,
            "required_creatives_n": n,
            "max_allowed_critical_errors_k": k,
            "upper_miss_rate_p_u": p_u,
            "damage_per_contact_rub": d,
            "max_expected_damage_rub": ed_max,
            "canary_share_q": q,
            "canary_detection_prob_pi": pi,
            "canary_phi": phi,
            "max_contacts_without_canary": r_max_no_canary,
            "max_contacts_with_canary": r_max_with_canary,
            "promotion_condition": f"observe <= {k} critical errors among {n} reviewed creatives",
        })

    return pd.DataFrame(rows)


def build_promotion_oc_curve(cfg: dict, thresholds: pd.DataFrame) -> pd.DataFrame:
    grid = [float(x) for x in cfg["true_error_rate_grid"]]
    rows = []

    for _, tier in thresholds.iterrows():
        n = int(tier["required_creatives_n"])
        k = int(tier["max_allowed_critical_errors_k"])
        for p in grid:
            prob = float(stats.binom.cdf(k, n, p))
            rows.append({
                "tier": tier["tier"],
                "n": n,
                "k": k,
                "true_error_rate": p,
                "probability_of_promotion": prob,
                "type_I_error_target": 1.0 - float(tier["confidence_level"]),
            })

    return pd.DataFrame(rows)


def build_sprt_demotion(cfg: dict) -> pd.DataFrame:
    sprt = cfg["sprt"]
    p0 = float(sprt["p0"])
    p1 = float(sprt["p1"])
    alpha = float(sprt["alpha"])
    beta = float(sprt["beta"])

    if not (0 < p0 < p1 < 1):
        raise ValueError("SPRT requires 0 < p0 < p1 < 1")
    if not (0 < alpha < 1 and 0 < beta < 1):
        raise ValueError("SPRT alpha and beta must be in (0, 1)")

    # SPRT нужен для быстрого понижения тира при деградации, а не для доказательства качества с нуля.
    A = math.log(beta / (1.0 - alpha))
    B = math.log((1.0 - beta) / alpha)

    inc_error = math.log(p1 / p0)
    inc_ok = math.log((1.0 - p1) / (1.0 - p0))

    return pd.DataFrame([{
        "test": "sprt_demotion",
        "h0_acceptable_error_rate_p0": p0,
        "h1_bad_error_rate_p1": p1,
        "alpha_false_demotion": alpha,
        "beta_missed_bad_rate": beta,
        "lower_boundary_A": A,
        "upper_boundary_B": B,
        "log_increment_if_error": inc_error,
        "log_increment_if_ok": inc_ok,
        "decision_rule": (
            "accumulate log-likelihood ratio; if LLR >= B, lower the Green tier; "
            "if LLR <= A, continue; otherwise keep monitoring"
        ),
    }])


def build_error_budget(cfg: dict) -> pd.DataFrame:
    eb = cfg["error_budget"]
    monthly_points = float(eb["monthly_budget_points"])
    weights = eb["severity_weights"]
    thresholds = eb["burn_thresholds"]

    scenarios = [
        {
            "scenario": "clean_month",
            "low": 0,
            "medium": 0,
            "high": 0,
            "critical": 0,
        },
        {
            "scenario": "one_low",
            "low": 1,
            "medium": 0,
            "high": 0,
            "critical": 0,
        },
        {
            "scenario": "one_medium",
            "low": 0,
            "medium": 1,
            "high": 0,
            "critical": 0,
        },
        {
            "scenario": "one_high",
            "low": 0,
            "medium": 0,
            "high": 1,
            "critical": 0,
        },
        {
            "scenario": "one_critical",
            "low": 0,
            "medium": 0,
            "high": 0,
            "critical": 1,
        },
        {
            "scenario": "two_medium_one_high",
            "low": 0,
            "medium": 2,
            "high": 1,
            "critical": 0,
        },
    ]

    rows = []
    for sc in scenarios:
        points = (
            sc["low"] * float(weights["low"])
            + sc["medium"] * float(weights["medium"])
            + sc["high"] * float(weights["high"])
            + sc["critical"] * float(weights["critical"])
        )
        burn = points / monthly_points if monthly_points > 0 else float("inf")

        if burn <= float(thresholds["normal_max"]):
            action = "normal: Green tier may expand eligible reach"
        elif burn <= float(thresholds["freeze_max"]):
            action = "freeze: pause further Green-tier expansion"
        elif burn <= float(thresholds["reduce_max"]):
            action = "reduce: lower tier / increase human review"
        else:
            action = "stop: freeze Green tier and start incident review"

        rows.append({
            "scenario": sc["scenario"],
            "low": sc["low"],
            "medium": sc["medium"],
            "high": sc["high"],
            "critical": sc["critical"],
            "points_consumed": points,
            "monthly_budget_points": monthly_points,
            "burn_rate": burn,
            "action": action,
        })

    return pd.DataFrame(rows)


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

    cfg = full_cfg["autonomy_rule"]
    validate_config(cfg)

    thresholds = build_thresholds(cfg)
    oc = build_promotion_oc_curve(cfg, thresholds)
    sprt = build_sprt_demotion(cfg)
    error_budget = build_error_budget(cfg)

    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_META.mkdir(parents=True, exist_ok=True)

    thresholds.to_csv(OUT_TABLES / "autonomy_thresholds.csv", index=False)
    oc.to_csv(OUT_TABLES / "autonomy_promotion_oc.csv", index=False)
    sprt.to_csv(OUT_TABLES / "autonomy_sprt_demotion.csv", index=False)
    error_budget.to_csv(OUT_TABLES / "autonomy_error_budget.csv", index=False)

    metadata = {
        "script": "autonomy.py",
        "timestamp": datetime.now().isoformat(),
        "config_hash": compute_config_hash(cfg),
        "deterministic": True,
        "seed_required": False,
        "tables_written": [
            "autonomy_thresholds.csv",
            "autonomy_promotion_oc.csv",
            "autonomy_sprt_demotion.csv",
            "autonomy_error_budget.csv",
        ],
    }
    with open(OUT_META / "autonomy_rule.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("Autonomy thresholds:")
    print(thresholds.to_string(index=False))
    print("\nPromotion OC curve (probability of promotion by true error rate):")
    print(oc.to_string(index=False))
    print("\nSPRT demotion monitoring:")
    print(sprt.to_string(index=False))
    print("\nError budget scenarios:")
    print(error_budget.to_string(index=False))
    print(f"\nConfig hash: {metadata['config_hash']}")
    print(f"Saved to: {OUT_TABLES}")


if __name__ == "__main__":
    main()
