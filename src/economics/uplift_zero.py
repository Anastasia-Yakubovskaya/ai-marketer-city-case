import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.economics import economics as econ


def run(cfg: dict, seed: int = 42, n: int = 20000) -> pd.DataFrame:
    local = dict(cfg)
    local["delta"] = [0.0, 0.0, 0.0]

    rng = np.random.default_rng(seed)
    horizon = int(local["horizon_months"])
    ramp = econ.build_ramp(local, horizon)
    costs = econ.monthly_costs(local, horizon, n, rng)
    benefit = econ.full_coverage_benefit(local, n, rng)

    rows = []
    for scenario in ("conservative", "base"):
        capacity = benefit["labor_capacity"] if scenario == "base" else 0.0
        monthly = capacity + benefit["errors_avoided"]
        gross_pilot = costs[:, :6].sum(axis=1)
        early_benefit = (monthly[:, None] * ramp[None, :6]).sum(axis=1)
        loss_if_stopped = gross_pilot - early_benefit
        payback = econ.payback_month(benefit, costs, ramp, scenario)
        npv24 = econ.npv_at_horizon(
            benefit,
            costs,
            ramp,
            24,
            float(local["discount_rate_annual"]),
            scenario,
        )

        rows.append(
            {
                "scenario": scenario,
                "pilot_gross_p50": float(np.nanpercentile(gross_pilot, 50)),
                "loss_if_stopped_at_m6_p10": float(np.nanpercentile(loss_if_stopped, 10)),
                "loss_if_stopped_at_m6_p50": float(np.nanpercentile(loss_if_stopped, 50)),
                "loss_if_stopped_at_m6_p90": float(np.nanpercentile(loss_if_stopped, 90)),
                "prob_payback_36m": float(np.nanmean(payback <= 36)),
                "npv24_p10": float(np.nanpercentile(npv24, 10)),
                "npv24_p50": float(np.nanpercentile(npv24, 50)),
                "npv24_p90": float(np.nanpercentile(npv24, 90)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    with open(ROOT / "config" / "assumptions.yaml", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)["economics"]

    result = run(cfg)
    tables = ROOT / "outputs" / "tables"
    meta = ROOT / "outputs" / "meta"
    tables.mkdir(parents=True, exist_ok=True)
    meta.mkdir(parents=True, exist_ok=True)

    result.round(3).to_csv(tables / "uplift_zero.csv", index=False)
    with open(meta / "uplift_zero.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "script": "uplift_zero.py",
                "timestamp": datetime.now().isoformat(),
                "rows": result.round(3).to_dict("records"),
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
    print(result.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
