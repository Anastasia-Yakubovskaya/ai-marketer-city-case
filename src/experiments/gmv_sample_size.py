from math import ceil
from pathlib import Path

import pandas as pd
import yaml
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]


def var_y(p: float, c: float) -> float:
    return p * (c**2 + 1 - p)


def n_superiority(
    p0: float,
    lift: float,
    c: float,
    alpha: float = 0.05,
    power: float = 0.8,
    rho2: float = 0.0,
) -> int:
    p1 = p0 * (1 + lift)
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    n = z**2 * (var_y(p1, c) + var_y(p0, c)) / (p1 - p0) ** 2
    return ceil(n * (1 - rho2))


def n_noninferiority(
    p_h: float,
    c: float,
    margin: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    z = norm.ppf(1 - alpha) + norm.ppf(power)
    return ceil(z**2 * 2 * var_y(p_h, c) / margin**2)


def economic_delta_power(g: dict, delta_break_even: float = 0.046) -> pd.DataFrame:
    p_control = float(g["baseline_conversion"])
    p_human = float(g["human_conversion"])
    effect_human = p_human - p_control
    p_ai = p_control + effect_human * (1 + delta_break_even)
    alpha = float(g["alpha"])
    power = float(g["power"])
    rho2 = float(g["cuped_rho_squared"])
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)

    rows = []
    for cv in [0.0, 1.0, 2.0]:
        n = ceil(z**2 * (var_y(p_human, cv) + var_y(p_ai, cv)) / (p_ai - p_human) ** 2)
        rows.append(
            {
                "order_value_cv": cv,
                "economic_delta": delta_break_even,
                "human_incremental_effect_pp": 100 * effect_human,
                "ai_minus_human_effect_pp": 100 * (p_ai - p_human),
                "n_per_arm": n,
                "n_per_arm_cuped": ceil(n * (1 - rho2)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    with open(ROOT / "config" / "assumptions.yaml", encoding="utf-8") as file:
        g = yaml.safe_load(file)["gmv"]

    p0 = float(g["baseline_conversion"])
    lift = float(g["relative_lift"])
    alpha = float(g["alpha"])
    power = float(g["power"])
    rho2 = float(g["cuped_rho_squared"])

    base = n_superiority(p0, lift, 0.0, alpha, power)
    sample_rows = []
    for cv in [0.0] + [float(x) for x in g["order_value_cv"]]:
        n = n_superiority(p0, lift, cv, alpha, power)
        sample_rows.append(
            {
                "order_value_cv": cv,
                "n_per_arm": n,
                "ratio_vs_conversion": round(n / base, 3),
                "n_per_arm_cuped": n_superiority(p0, lift, cv, alpha, power, rho2),
            }
        )

    p_human = float(g["human_conversion"])
    effect = p_human - p0
    ni_rows = []
    for cv in [0.0] + [float(x) for x in g["order_value_cv"]]:
        for margin_share in g["noninferiority_margin_share_of_effect"]:
            margin = float(margin_share) * effect
            ni_rows.append(
                {
                    "order_value_cv": cv,
                    "margin_share_of_effect": margin_share,
                    "margin_pp": round(100 * margin, 3),
                    "n_per_arm": n_noninferiority(p_human, cv, margin, alpha, power),
                }
            )

    tables = ROOT / "outputs" / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    sample_df = pd.DataFrame(sample_rows)
    ni_df = pd.DataFrame(ni_rows)
    delta_df = economic_delta_power(g)

    sample_df.to_csv(tables / "gmv_sample_size.csv", index=False)
    ni_df.to_csv(tables / "gmv_noninferiority.csv", index=False)
    delta_df.to_csv(tables / "economic_delta_power.csv", index=False)

    print(sample_df.to_string(index=False))
    print(ni_df.to_string(index=False))
    print("\nEconomic delta feasibility:")
    print(delta_df.to_string(index=False))


if __name__ == "__main__":
    main()
