import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]


def raw_rates(
    pre_rate: float,
    post_rate: float,
    pre_share: dict,
    post_share: dict,
    factor: float,
) -> dict:
    return {
        key: (pre_rate * pre_share[key] + post_rate * post_share[key]) * factor
        for key in pre_share
    }


def required_judge_recall(raw: dict, coverage: dict, target: float):
    residual_other = sum(
        raw[key] * (1 - coverage[key])
        for key in raw
        if key != "typos_tone"
    )
    room = target - residual_other
    if room < 0:
        return None
    return max(0.0, 1 - room / raw["typos_tone"])


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    with open(ROOT / "config" / "assumptions.yaml", encoding="utf-8") as file:
        full = yaml.safe_load(file)
    case = full["case_facts"]
    target_cfg = full["judge_target"]

    rows = []
    for factor in target_cfg["agent_raw_error_factor"]:
        raw = raw_rates(
            case["prelaunch_error_rate"],
            case["postlaunch_error_rate"],
            target_cfg["pre_launch_share"],
            target_cfg["post_launch_share"],
            factor,
        )
        for eligibility_recall in target_cfg["control_recall"]["offer_unavailable"]:
            coverage = {
                key: value
                for key, value in target_cfg["control_recall"].items()
                if key != "offer_unavailable"
            }
            coverage["offer_unavailable"] = eligibility_recall
            coverage["typos_tone"] = 0.0

            residual_high = sum(
                raw[key] * (1 - coverage[key])
                for key in raw
                if target_cfg["severity"][key] == "high"
            )
            for target in target_cfg["targets_post_launch_error_rate"]:
                recall = required_judge_recall(raw, coverage, target)
                rows.append(
                    {
                        "agent_raw_error_factor": factor,
                        "eligibility_recall": eligibility_recall,
                        "target_post_error_rate": target,
                        "required_judge_recall": None if recall is None else round(recall, 3),
                        "residual_high_severity_rate": round(residual_high, 5),
                    }
                )

    result = pd.DataFrame(rows)
    recommended = target_cfg["recommended"]
    selected = result[
        (result.agent_raw_error_factor == recommended["agent_raw_error_factor"])
        & (result.eligibility_recall == recommended["eligibility_recall"])
        & (result.target_post_error_rate == recommended["target"])
    ].iloc[0]

    today_high = case["postlaunch_error_rate"] * (
        target_cfg["post_launch_share"]["wrong_segment"]
        + target_cfg["post_launch_share"]["offer_unavailable"]
    )

    tables = ROOT / "outputs" / "tables"
    meta = ROOT / "outputs" / "meta"
    tables.mkdir(parents=True, exist_ok=True)
    meta.mkdir(parents=True, exist_ok=True)
    result.to_csv(tables / "judge_target.csv", index=False)

    metadata = {
        "script": "judge_target.py",
        "timestamp": datetime.now().isoformat(),
        "recommended": recommended,
        "required_judge_recall_recommended": float(selected.required_judge_recall),
        "today_high_severity_post_launch_rate": today_high,
        "residual_high_severity_recommended": float(selected.residual_high_severity_rate),
        "taxonomy_status": "ASSUMPTION until validated on real incident data",
    }
    with open(meta / "judge_target.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    print(result.to_string(index=False))
    print("recommended:", selected.to_dict())
    print("today high-severity post-launch rate:", round(today_high, 5))


if __name__ == "__main__":
    main()
