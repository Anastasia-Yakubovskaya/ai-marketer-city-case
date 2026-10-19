import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]


def build_up(cfg: dict) -> dict:
    team = cfg["team_fte"] * cfg["fte_cost_m_rub_per_month"] * cfg["months"]
    infrastructure = [float(value) for value in cfg["infra_inference_total_m_rub"]]
    contingency_share = float(cfg["contingency_share"])
    gross = [
        team + infra + contingency_share * (team + infra)
        for infra in infrastructure
    ]

    post_pilot = cfg["post_pilot"]
    fte_cost = cfg["fte_cost_m_rub_per_month"]
    opex = [
        post_pilot["team_fte"][idx] * fte_cost
        + post_pilot["infra_m_rub_per_month"][idx]
        + post_pilot["annotation_m_rub_per_month"][idx]
        for idx in range(3)
    ]
    return {
        "team": team,
        "infra": infrastructure,
        "contingency_share": contingency_share,
        "gross": gross,
        "opex": opex,
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    with open(ROOT / "config" / "assumptions.yaml", encoding="utf-8") as file:
        full = yaml.safe_load(file)

    values = build_up(full["pilot_cost_bottom_up"])
    round2 = lambda x: round(x, 2)
    contingency = [
        values["contingency_share"] * (values["team"] + infra)
        for infra in values["infra"]
    ]

    rows = [
        (
            "team: 5 FTE x 0.35 x 6 months",
            values["team"],
            values["team"],
            values["team"],
            "case average FTE rate used as proxy",
        ),
        (
            "infrastructure and inference (total)",
            *values["infra"],
            "ASSUMPTION: confirm internal chargeback",
        ),
        (
            "contingency 8% of team + infrastructure",
            *contingency,
            "ASSUMPTION",
        ),
        ("PILOT GROSS CASH COST", *values["gross"], "sum"),
        (
            "post-pilot run cost per month",
            *values["opex"],
            "ASSUMPTION: 2-4 FTE + infra + annotation",
        ),
    ]
    table = pd.DataFrame(
        [(name, round2(low), round2(mode), round2(high), note) for name, low, mode, high, note in rows],
        columns=["component", "low", "mode", "high", "type_and_comment"],
    )

    tables = ROOT / "outputs" / "tables"
    meta = ROOT / "outputs" / "meta"
    tables.mkdir(parents=True, exist_ok=True)
    meta.mkdir(parents=True, exist_ok=True)
    table.to_csv(tables / "pilot_cost_build_up.csv", index=False)

    economics = full["economics"]
    metadata = {
        "script": "pilot_cost.py",
        "timestamp": datetime.now().isoformat(),
        "gross": [round2(value) for value in values["gross"]],
        "opex": [round2(value) for value in values["opex"]],
        "matches_economics_config": (
            [round2(value) for value in values["gross"]]
            == [float(value) for value in economics["pilot_cost"]]
            and [round2(value) for value in values["opex"]]
            == [float(value) for value in economics["post_pilot_opex"]]
        ),
    }
    with open(meta / "pilot_cost.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
