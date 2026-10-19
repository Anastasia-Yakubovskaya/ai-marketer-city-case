import hashlib
import json
import sys
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
    required = [
        "n_simulations",
        "horizon_months",
        "breakeven_horizon_months",
        "labor_cost_m_rub_per_month",
        "communication_budget_m_rub_per_month",
        "labor_share_removed",
        "capacity_monetization",
        "roi0",
        "delta",
        "pilot_cost",
        "post_pilot_opex",
        "coverage_ramp",
        "max_coverage_share",
        "coverage_sensitivity_share",
        "discount_rate_annual",
        "error_benefit",
    ]
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ValueError(f"Missing required keys in economics config: {missing}")


def tri(rng, params, size):
    low, mode, high = map(float, params)
    if not low <= mode <= high:
        raise ValueError(f"Triangular params must satisfy low <= mode <= high, got {params}")
    if low == high:
        return np.full(size, low) if size is not None else low
    return rng.triangular(low, mode, high, size=size)


def build_ramp(cfg, horizon):
    max_share = float(cfg["max_coverage_share"])
    ramp = np.ones(horizon) * max_share
    given = cfg["coverage_ramp"]
    last_month = 0
    last_value = 0.0

    for month_str, value in given.items():
        month = int(month_str)
        if 1 <= month <= horizon:
            ramp[month - 1] = float(value)
            if month > last_month:
                last_month = month
                last_value = float(value)

    target_month = min(24, horizon)
    if target_month > last_month:
        step = (max_share - last_value) / (target_month - last_month)
        for month in range(last_month + 1, target_month + 1):
            ramp[month - 1] = min(max_share, last_value + step * (month - last_month))

    for month in range(target_month, horizon):
        ramp[month] = max_share
    return ramp


def scaled_ramp(cfg: dict, horizon: int, share: float) -> np.ndarray:
    base = build_ramp(cfg, horizon)
    return base * float(share) / float(cfg["max_coverage_share"])


def monthly_costs(cfg, horizon, n, rng):
    pilot = tri(rng, cfg["pilot_cost"], n)
    opex = tri(rng, cfg["post_pilot_opex"], n)
    costs = np.zeros((n, horizon))
    costs[:, :6] = (pilot / 6)[:, None]
    costs[:, 6:] = opex[:, None]
    return costs


def full_coverage_benefit(cfg, n, rng):
    labor = float(cfg["labor_cost_m_rub_per_month"])
    communication = float(cfg["communication_budget_m_rub_per_month"])

    labor_share = tri(rng, cfg["labor_share_removed"], n)
    monetization = tri(rng, cfg["capacity_monetization"], n)
    roi0 = tri(rng, cfg["roi0"], n)
    delta = tri(rng, cfg["delta"], n)

    return {
        "communications": communication * roi0 * delta,
        "labor_capacity": labor * labor_share * monetization,
        "errors_avoided": tri(rng, cfg["error_benefit"], n),
        "delta": delta,
        "roi0": roi0,
        "a": labor_share,
        "m": monetization,
    }


def payback_month(benefit_full, costs, ramp, scenario):
    # В conservative не считаем высвобождённую ёмкость денежным эффектом.
    if scenario == "conservative":
        benefit = benefit_full["communications"]
    elif scenario == "base":
        benefit = benefit_full["communications"] + benefit_full["labor_capacity"]
    else:
        raise ValueError(f"unknown scenario {scenario}")

    per_month = (benefit + benefit_full["errors_avoided"])[:, None] * ramp[None, :]
    cumulative = np.cumsum(per_month - costs, axis=1)
    paid = cumulative >= 0
    has_payback = paid.any(axis=1)
    first_month = np.argmax(paid, axis=1) + 1
    return np.where(has_payback, first_month, np.nan)


def npv_at_horizon(benefit_full, costs, ramp, horizon, rate_annual, scenario):
    if scenario == "conservative":
        benefit = benefit_full["communications"]
    elif scenario == "base":
        benefit = benefit_full["communications"] + benefit_full["labor_capacity"]
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    total_benefit = benefit + benefit_full["errors_avoided"]
    per_month = total_benefit[:, None] * ramp[None, :]
    net = per_month[:, :horizon] - costs[:, :horizon]
    monthly_rate = (1 + rate_annual) ** (1 / 12) - 1
    discount = (1 + monthly_rate) ** (-np.arange(1, horizon + 1))
    return np.sum(net * discount[None, :], axis=1)


def percentile_row(series, label):
    series = np.asarray(series, dtype=float)
    return {
        "metric": label,
        "p10": float(np.nanpercentile(series, 10)),
        "p50": float(np.nanpercentile(series, 50)),
        "p90": float(np.nanpercentile(series, 90)),
        "mean": float(np.nanmean(series)),
    }


def deterministic_payback_month(cfg, delta_value, scenario, ramp, horizon):
    mode = lambda values: float(values[1])
    labor = float(cfg["labor_cost_m_rub_per_month"])
    communication = float(cfg["communication_budget_m_rub_per_month"])
    pilot = mode(cfg["pilot_cost"])
    opex = mode(cfg["post_pilot_opex"])
    error_benefit = mode(cfg["error_benefit"])

    communication_benefit = communication * mode(cfg["roi0"]) * delta_value
    if scenario == "conservative":
        full_benefit = communication_benefit
    else:
        capacity = labor * mode(cfg["labor_share_removed"]) * mode(cfg["capacity_monetization"])
        full_benefit = communication_benefit + capacity

    costs = np.zeros(horizon)
    costs[:6] = pilot / 6
    costs[6:] = opex
    cumulative = np.cumsum((full_benefit + error_benefit) * ramp - costs)
    if not (cumulative >= 0).any():
        return np.nan
    return int(np.argmax(cumulative >= 0) + 1)


def break_even_delta(cfg, scenario, ramp, horizon, target_month):
    low, high = -0.10, 0.50
    for _ in range(60):
        mid = (low + high) / 2
        month = deterministic_payback_month(cfg, mid, scenario, ramp, horizon)
        if np.isnan(month) or month > target_month:
            low = mid
        else:
            high = mid
    return high


def coverage_case(cfg, seed, ramp, label):
    n = int(cfg["n_simulations"])
    horizon = int(cfg["horizon_months"])
    rng = np.random.default_rng(seed)
    costs = monthly_costs(cfg, horizon, n, rng)
    benefit = full_coverage_benefit(cfg, n, rng)
    target = int(cfg["breakeven_horizon_months"])

    rows = []
    for scenario in ("conservative", "base"):
        payback = payback_month(benefit, costs, ramp, scenario)
        rows.append(
            {
                "coverage_case": label,
                "scenario": scenario,
                "payback_p10": float(np.nanpercentile(payback, 10)),
                "payback_p50": float(np.nanpercentile(payback, 50)),
                "payback_p90": float(np.nanpercentile(payback, 90)),
                "prob_payback_24m": float(np.nanmean(payback <= 24)),
                "prob_never_within_60m": float(np.mean(np.isnan(payback))),
                "breakeven_delta_24m": break_even_delta(
                    cfg, scenario, ramp, horizon, target
                ),
            }
        )
    return pd.DataFrame(rows)


def simulate(cfg, seed):
    rng = np.random.default_rng(seed)
    n = int(cfg["n_simulations"])
    horizon = int(cfg["horizon_months"])
    ramp = build_ramp(cfg, horizon)
    costs = monthly_costs(cfg, horizon, n, rng)
    benefit = full_coverage_benefit(cfg, n, rng)

    summary = pd.DataFrame(
        [
            percentile_row(benefit["communications"], "communications_mln_month"),
            percentile_row(benefit["labor_capacity"], "labor_capacity_mln_month"),
            percentile_row(benefit["errors_avoided"], "errors_avoided_mln_month"),
            percentile_row(
                benefit["communications"]
                + benefit["labor_capacity"]
                + benefit["errors_avoided"],
                "total_mln_month",
            ),
        ]
    )

    payback_rows = []
    for scenario in ("conservative", "base"):
        payback = payback_month(benefit, costs, ramp, scenario)
        payback_rows.append(
            {
                "scenario": scenario,
                "payback_p10": float(np.nanpercentile(payback, 10)),
                "payback_p50": float(np.nanpercentile(payback, 50)),
                "payback_p90": float(np.nanpercentile(payback, 90)),
                "prob_payback_24m": float(np.nanmean(payback <= 24)),
                "prob_payback_36m": float(np.nanmean(payback <= 36)),
                "prob_never_within_horizon": float(np.mean(np.isnan(payback))),
            }
        )
    payback = pd.DataFrame(payback_rows)

    target = int(cfg["breakeven_horizon_months"])
    breakeven = pd.DataFrame(
        [
            {
                "scenario": scenario,
                "breakeven_horizon_months": target,
                "breakeven_delta": break_even_delta(
                    cfg, scenario, ramp, horizon, target
                ),
                "note": "economic planning threshold, not a statistical M6 proof",
            }
            for scenario in ("conservative", "base")
        ]
    )

    rate = float(cfg["discount_rate_annual"])
    param_specs = {
        "delta": cfg["delta"],
        "roi0": cfg["roi0"],
        "labor_share_removed": cfg["labor_share_removed"],
        "capacity_monetization": cfg["capacity_monetization"],
        "pilot_cost": cfg["pilot_cost"],
        "post_pilot_opex": cfg["post_pilot_opex"],
    }
    tornado_rows = []
    for name, params in param_specs.items():
        for label, value in (("p10", params[0]), ("p90", params[2])):
            local = dict(cfg)
            local[name] = [value, value, value]
            rng_local = np.random.default_rng(seed)
            local_benefit = full_coverage_benefit(local, n, rng_local)
            local_costs = monthly_costs(local, horizon, n, rng_local)
            npv = npv_at_horizon(
                local_benefit, local_costs, ramp, target, rate, "base"
            )
            tornado_rows.append(
                {
                    "parameter": name,
                    "point": label,
                    "value": float(value),
                    "npv_median_mln": float(np.nanmedian(npv)),
                }
            )
    tornado_long = pd.DataFrame(tornado_rows)
    tornado = tornado_long.pivot_table(
        index="parameter", columns="point", values="npv_median_mln"
    )
    tornado["range"] = (tornado["p90"] - tornado["p10"]).abs()
    tornado = tornado.reset_index().sort_values("range", ascending=False)

    rng_trajectory = np.random.default_rng(seed)
    benefit_trajectory = full_coverage_benefit(cfg, n, rng_trajectory)
    costs_trajectory = monthly_costs(cfg, horizon, n, rng_trajectory)
    per_month = (
        benefit_trajectory["communications"]
        + benefit_trajectory["labor_capacity"]
        + benefit_trajectory["errors_avoided"]
    )[:, None] * ramp[None, :]
    median_benefit = np.median(per_month, axis=0)
    median_cost = np.median(costs_trajectory, axis=0)
    trajectory = pd.DataFrame(
        {
            "month": np.arange(1, horizon + 1),
            "coverage_ramp": ramp,
            "median_benefit_mln": median_benefit,
            "median_cost_mln": median_cost,
            "median_net_mln": median_benefit - median_cost,
            "cumulative_net_mln": np.cumsum(median_benefit - median_cost),
        }
    )

    return summary, payback, breakeven, tornado, trajectory


def compute_config_hash(cfg: dict) -> str:
    cfg_str = json.dumps(cfg, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(cfg_str.encode("utf-8")).hexdigest()[:16]


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    with open(CONFIG_PATH, encoding="utf-8") as file:
        cfg = yaml.safe_load(file)["economics"]
    validate_config(cfg)

    with open(SEEDS_PATH, encoding="utf-8") as file:
        seed = int(yaml.safe_load(file)["global_seed"])

    summary, payback, breakeven, tornado, trajectory = simulate(cfg, seed)
    horizon = int(cfg["horizon_months"])
    full_ramp = build_ramp(cfg, horizon)
    half_share = float(cfg["coverage_sensitivity_share"])
    half_ramp = scaled_ramp(cfg, horizon, half_share)
    coverage_sensitivity = pd.concat(
        [
            coverage_case(cfg, seed, full_ramp, "full_eligible_contour"),
            coverage_case(cfg, seed, half_ramp, f"{half_share:.0%}_eligible_value"),
        ],
        ignore_index=True,
    )

    modal_payback = pd.DataFrame(
        [
            {
                "coverage_case": "full_eligible_contour",
                "scenario": scenario,
                "delta": float(cfg["delta"][1]),
                "modal_payback_month": deterministic_payback_month(
                    cfg,
                    float(cfg["delta"][1]),
                    scenario,
                    full_ramp,
                    horizon,
                ),
            }
            for scenario in ("conservative", "base")
        ]
    )

    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_META.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_TABLES / "economics_summary.csv", index=False)
    payback.to_csv(OUT_TABLES / "payback_scenarios.csv", index=False)
    breakeven.to_csv(OUT_TABLES / "breakeven.csv", index=False)
    tornado.to_csv(OUT_TABLES / "tornado.csv", index=False)
    trajectory.to_csv(OUT_TABLES / "economics_trajectory.csv", index=False)
    coverage_sensitivity.to_csv(
        OUT_TABLES / "economics_coverage_sensitivity.csv", index=False
    )
    modal_payback.to_csv(OUT_TABLES / "economics_modal_payback.csv", index=False)

    metadata = {
        "script": "economics.py",
        "timestamp": datetime.now().isoformat(),
        "config_hash": compute_config_hash(cfg),
        "seed": seed,
        "n_simulations": int(cfg["n_simulations"]),
        "horizon_months": horizon,
        "coverage_interpretation": "100% means the full eligible post-pilot contour, not all campaign types",
    }
    with open(OUT_META / "economics.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    print("Benefit at full eligible coverage (mln RUB/month):")
    print(summary.to_string(index=False))
    print("\nPayback by scenario:")
    print(payback.to_string(index=False))
    print("\nCoverage sensitivity:")
    print(coverage_sensitivity.to_string(index=False))
    print("\nModal payback:")
    print(modal_payback.to_string(index=False))
    print("\nTornado:")
    print(tornado.to_string(index=False))


if __name__ == "__main__":
    main()
