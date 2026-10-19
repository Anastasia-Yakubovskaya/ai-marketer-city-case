import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]


def capacity_bridge(cf: dict, cb: dict) -> pd.DataFrame:
    rate = cf["team_cost_m_rub_per_month"] / 24.0
    rows = []

    for role, values in cb["roles"].items():
        released = values["fte"] * values["automatable_share"]
        rows.append(
            {
                "role": role,
                "fte": values["fte"],
                "automatable_share": values["automatable_share"],
                "fte_released_full": released,
                "m_rub_per_month_full": released * rate,
                "type": "ASSUMPTION",
            }
        )

    review_fte = (
        cf["campaigns_per_month"]
        * cb["review_hours_per_campaign"]
        / cb["hours_per_fte_month"]
    )
    rows.append(
        {
            "role": "manual review scenario (120 x 2 h)",
            "fte": review_fte,
            "automatable_share": 1.0,
            "fte_released_full": review_fte,
            "m_rub_per_month_full": review_fte * rate,
            "type": "ASSUMPTION",
        }
    )
    return pd.DataFrame(rows)


def budget_check(cf: dict, ab: dict) -> pd.DataFrame:
    weeks_per_month = 52 / 12
    rows = []

    for clients_m in ab["reachable_clients_options_m"]:
        contacts = clients_m * 1e6 * cf["messages_per_week"] * weeks_per_month
        push_cost = contacts * cf["push_cost_rub"] / 1e6
        share = push_cost / cf["communication_budget_m_rub_per_month"]
        rows.append(
            {
                "reachable_clients_m": clients_m,
                "max_contacts_m": round(contacts / 1e6, 1),
                "max_push_spend_m_rub": round(push_cost, 2),
                "share_of_40m": round(share, 3),
                "claimed_share": ab["claimed_share_of_budget"],
                "claim_feasible": share >= ab["claimed_share_of_budget"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    with open(ROOT / "config" / "assumptions.yaml", encoding="utf-8") as file:
        full = yaml.safe_load(file)

    cf = full["case_facts"]
    capacity = capacity_bridge(cf, full["capacity_bridge"])
    budget = budget_check(cf, full["addressable_budget"])

    tables = ROOT / "outputs" / "tables"
    meta = ROOT / "outputs" / "meta"
    tables.mkdir(parents=True, exist_ok=True)
    meta.mkdir(parents=True, exist_ok=True)

    capacity.round(3).to_csv(tables / "capacity_bridge.csv", index=False)
    budget.to_csv(tables / "addressable_budget.csv", index=False)

    role_rows = capacity[
        (capacity["type"] == "ASSUMPTION")
        & ~capacity["role"].str.startswith("manual review")
    ]
    review_row = capacity[capacity["role"].str.startswith("manual review")].iloc[0]

    metadata = {
        "script": "capacity_budget.py",
        "timestamp": datetime.now().isoformat(),
        "assumed_fte_released_full": float(role_rows.fte_released_full.sum()),
        "assumed_m_rub_full": float(role_rows.m_rub_per_month_full.sum()),
        "review_scenario_fte": float(review_row.fte_released_full),
        "push_absorbs_share_4m": float(budget.iloc[0].share_of_40m),
        "push_absorbs_share_12m": float(budget.iloc[1].share_of_40m),
        "claim_65pct_feasible": bool(budget.claim_feasible.any()),
    }
    with open(meta / "capacity_budget.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    print(capacity.round(2).to_string(index=False))
    print(budget.to_string(index=False))


if __name__ == "__main__":
    main()
