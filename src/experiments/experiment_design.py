import sys
import hashlib
import json
from datetime import datetime
from math import ceil, sqrt
from pathlib import Path

import pandas as pd
import yaml
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "assumptions.yaml"
OUT_TABLES = ROOT / "outputs" / "tables"
OUT_META = ROOT / "outputs" / "meta"


def validate_config(cfg: dict) -> None:
    required = ["alpha", "power", "baseline_conversion", "relative_lift",
                "cuped_rho_squared", "audience_sizes", "holdout_share",
                "persistent_holdout_share", "noninferiority"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Missing required keys in experiment_design config: {missing}")


def two_proportion_sample_size(p_control: float, p_treatment: float,
                               alpha: float = 0.05, power: float = 0.80) -> int:
    if not (0 < p_control < 1 and 0 < p_treatment < 1):
        raise ValueError("probabilities must be in (0, 1)")
    effect = abs(p_treatment - p_control)
    if effect == 0:
        raise ValueError("effect must be non-zero")
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)
    variance = p_control * (1 - p_control) + p_treatment * (1 - p_treatment)
    return ceil((z_alpha + z_beta) ** 2 * variance / effect ** 2)


def cuped_adjusted_sample_size(n_per_arm: int, rho_squared: float) -> int:
    if not 0 <= rho_squared < 1:
        raise ValueError("rho_squared must be in [0, 1)")
    return ceil(n_per_arm * (1.0 - rho_squared))


def power_two_proportion_unequal(n_control: int, n_treatment: int,
                                 p_control: float, p_treatment: float,
                                 alpha: float = 0.05) -> float:
    if min(n_control, n_treatment) <= 0:
        raise ValueError("arm sizes must be positive")
    p_pool = (n_control * p_control + n_treatment * p_treatment) / (n_control + n_treatment)
    se0 = sqrt(p_pool * (1 - p_pool) * (1 / n_control + 1 / n_treatment))
    se1 = sqrt(p_control * (1 - p_control) / n_control + p_treatment * (1 - p_treatment) / n_treatment)
    delta = abs(p_treatment - p_control)
    z_alpha = norm.ppf(1 - alpha / 2)

    noncentral = delta / se1
    critical = z_alpha * se0 / se1
    return float(norm.cdf(-critical - noncentral) + 1 - norm.cdf(critical - noncentral))


def mde_for_audience(total_n: int, holdout_share: float, baseline: float,
                     alpha: float = 0.05, target_power: float = 0.80) -> dict:
    n_control = max(1, round(total_n * holdout_share))
    n_treatment = total_n - n_control
    lo, hi = 0.0, min(0.50, 1.0 - baseline - 1e-9)
    # Бинарным поиском подбираем эффект, при котором мощность впервые достигает цели.
    for _ in range(80):
        mid = (lo + hi) / 2
        p_treat = baseline + mid
        pw = power_two_proportion_unequal(n_control, n_treatment, baseline, p_treat, alpha)
        if pw >= target_power:
            hi = mid
        else:
            lo = mid
    abs_mde = hi
    return {
        "audience_size": total_n,
        "holdout_share": holdout_share,
        "n_control": n_control,
        "n_treatment": n_treatment,
        "baseline_conversion": baseline,
        "mde_absolute": abs_mde,
        "mde_relative": abs_mde / baseline,
        "target_power": target_power,
        "alpha": alpha,
    }


def noninferiority_sample_size(p_human: float, p_ai: float, margin_absolute: float,
                               alpha_one_sided: float = 0.05, power: float = 0.80) -> int:
    if margin_absolute <= 0:
        raise ValueError("margin_absolute must be positive")
    distance = (p_ai - p_human) + margin_absolute
    if distance <= 0:
        raise ValueError("expected effect does not lie inside the non-inferiority region")
    z_alpha = norm.ppf(1 - alpha_one_sided)
    z_beta = norm.ppf(power)
    variance = p_human * (1 - p_human) + p_ai * (1 - p_ai)
    return ceil(((z_alpha + z_beta) ** 2 * variance) / distance ** 2)


def srm_check(observed_treatment: int, observed_control: int,
              expected_share_treatment: float = 0.5) -> dict:
    total = observed_treatment + observed_control
    if total <= 0:
        raise ValueError("total must be positive")
    expected_treatment = total * expected_share_treatment
    expected_control = total * (1.0 - expected_share_treatment)
    chi2 = ((observed_treatment - expected_treatment) ** 2 / expected_treatment +
            (observed_control - expected_control) ** 2 / expected_control)

    from scipy.stats import chi2 as chi2_dist
    p_value = 1.0 - chi2_dist.cdf(chi2, df=1)
    return {
        "check": "randomization_srm",
        "observed_treatment": observed_treatment,
        "observed_control": observed_control,
        "expected_treatment": expected_treatment,
        "expected_control": expected_control,
        "chi2": chi2,
        "p_value": p_value,
        "threshold": 0.001,
        "status": "pass" if p_value >= 0.001 else "alert",
    }


def build_outputs(cfg: dict):
    alpha = float(cfg["alpha"])
    power = float(cfg["power"])
    p0 = float(cfg["baseline_conversion"])
    relative_lift = float(cfg["relative_lift"])
    p1 = p0 * (1 + relative_lift)

    n = two_proportion_sample_size(p0, p1, alpha, power)
    n_cuped = cuped_adjusted_sample_size(n, float(cfg["cuped_rho_squared"]))
    sample_summary = pd.DataFrame([{
        "comparison": "communication_vs_holdout",
        "baseline_conversion": p0,
        "treatment_conversion": p1,
        "relative_lift": relative_lift,
        "alpha": alpha,
        "power": power,
        "n_per_arm_without_cuped": n,
        "cuped_rho_squared": float(cfg["cuped_rho_squared"]),
        "n_per_arm_with_cuped_approx": n_cuped,
    }])

    mde = pd.DataFrame([
        mde_for_audience(int(N), float(cfg["holdout_share"]), p0, alpha, power)
        for N in cfg["audience_sizes"]
    ])

    ni = cfg["noninferiority"]
    ni_n = noninferiority_sample_size(
        float(ni["human_conversion"]),
        float(ni["expected_ai_conversion"]),
        float(ni["margin_absolute"]),
        float(ni["alpha_one_sided"]),
        float(ni["power"]),
    )
    noninferiority = pd.DataFrame([{
        "comparison": "ai_vs_human_noninferiority",
        "human_conversion": float(ni["human_conversion"]),
        "expected_ai_conversion": float(ni["expected_ai_conversion"]),
        "noninferiority_margin_absolute": float(ni["margin_absolute"]),
        "alpha_one_sided": float(ni["alpha_one_sided"]),
        "power": float(ni["power"]),
        "n_per_arm": ni_n,
    }])


    srm = srm_check(observed_treatment=50032, observed_control=49968)
    srm_df = pd.DataFrame([srm])

    return sample_summary, mde, noninferiority, srm_df


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

    cfg = full_cfg["experiment_design"]
    validate_config(cfg)

    sample_summary, mde, noninferiority, srm_df = build_outputs(cfg)

    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_META.mkdir(parents=True, exist_ok=True)

    sample_summary.to_csv(OUT_TABLES / "experiment_sample_size.csv", index=False)
    mde.to_csv(OUT_TABLES / "mde_by_audience.csv", index=False)
    srm_df.to_csv(OUT_TABLES / "srm_example.csv", index=False)

    metadata = {
        "script": "experiment_design.py",
        "timestamp": datetime.now().isoformat(),
        "config_hash": compute_config_hash(cfg),
        "tables_written": [
            "experiment_sample_size.csv",
            "mde_by_audience.csv",
            "srm_example.csv",
        ],
    }
    with open(OUT_META / "experiment_design.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(sample_summary.to_string(index=False))
    print("\nMDE by audience:\n", mde.to_string(index=False))
    print("\nConversion proxy non-inferiority (diagnostic only):\n", noninferiority.to_string(index=False))
    print("\nSRM check example:\n", srm_df.to_string(index=False))
    print(f"\nConfig hash: {metadata['config_hash']}")
    print(f"Saved to: {OUT_TABLES}")


if __name__ == "__main__":
    main()
