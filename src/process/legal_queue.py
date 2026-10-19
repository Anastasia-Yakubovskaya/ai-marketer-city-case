import copy
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

from src.process import process_model


def simulate(full: dict, seed: int = 42, n: int = 50000) -> pd.DataFrame:
    process_cfg = full["process_model"]
    legal_cfg = full["legal_queue"]
    rng = np.random.default_rng(seed)

    template_launch = process_model.critical_path(rng, process_cfg["to_be"], n)
    steps = {}
    for name, spec in process_cfg["to_be"].items():
        current = copy.deepcopy(spec)
        if name == "human_one_click":
            steps["legal_review"] = {
                "days": legal_cfg["queue_days"],
                "depends_on": ["safety_checks"],
            }
            current["depends_on"] = ["legal_review"]
        steps[name] = current

    new_launch = process_model.critical_path(rng, steps, n)
    new_share = float(legal_cfg["new_campaign_share"])
    blended = np.where(rng.random(n) < new_share, new_launch, template_launch)

    observation_window = int(legal_cfg["observation_window_days"])
    analysis = process_model.tri(rng, process_cfg["analysis_to_be_days"], n)
    rows = []
    for group, launch in (
        ("template (pre-approved)", template_launch),
        ("new (legal queue)", new_launch),
        ("blended", blended),
    ):
        q10, q50, q90 = np.quantile(launch, [0.1, 0.5, 0.9])
        evidence_p50, evidence_p90 = np.quantile(
            launch + observation_window + analysis,
            [0.5, 0.9],
        )
        rows.append(
            {
                "group": group,
                "launch_p10": q10,
                "launch_p50": q50,
                "launch_p90": q90,
                "time_to_evidence_p50_w14": evidence_p50,
                "time_to_evidence_p90_w14": evidence_p90,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    with open(ROOT / "config" / "assumptions.yaml", encoding="utf-8") as file:
        full = yaml.safe_load(file)
    result = simulate(full)

    tables = ROOT / "outputs" / "tables"
    meta = ROOT / "outputs" / "meta"
    tables.mkdir(parents=True, exist_ok=True)
    meta.mkdir(parents=True, exist_ok=True)
    result.round(3).to_csv(tables / "process_legal_queue.csv", index=False)

    with open(meta / "process_legal_queue.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "script": "legal_queue.py",
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
