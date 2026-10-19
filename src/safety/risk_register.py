import json
from datetime import datetime
from pathlib import Path
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[2]
REQUIRED = ["id", "failure_mode", "effect", "severity", "occurrence", "detectability", "preventive_control", "detection", "occurrence_after", "detectability_after", "owner"]


def compute_fmea(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing: raise ValueError(f"FMEA data missing columns: {missing}")
    x = df.copy()
    for c in ("severity", "occurrence", "detectability", "occurrence_after", "detectability_after"):
        if not x[c].between(1, 10).all(): raise ValueError(f"{c} must be within 1..10")
    # RPN используем только для приоритизации: это экспертный скоринг, а не вероятность инцидента.
    x["rpn_before"] = x.severity * x.occurrence * x.detectability
    x["rpn_after"] = x.severity * x.occurrence_after * x.detectability_after
    x["reduction_x"] = (x.rpn_before / x.rpn_after).round(1)
    return x.sort_values("rpn_before", ascending=False).reset_index(drop=True)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    df = compute_fmea(pd.read_csv(ROOT / "data" / "processed" / "fmea_assumptions.csv"))
    (ROOT / "outputs" / "tables").mkdir(parents=True, exist_ok=True); (ROOT / "outputs" / "meta").mkdir(parents=True, exist_ok=True)
    df.to_csv(ROOT / "outputs" / "tables" / "fmea.csv", index=False, encoding="utf-8-sig")
    tot = dict(rpn_before=int(df.rpn_before.sum()), rpn_after=int(df.rpn_after.sum()), risks=len(df), top_residual=df.sort_values("rpn_after", ascending=False).head(3).id.tolist())
    json.dump({"script": "risk_register.py", "timestamp": datetime.now().isoformat(), "expert_assumptions": True, **tot}, open(ROOT / "outputs" / "meta" / "fmea.json", "w"), indent=2)
    print(df[["id", "failure_mode", "rpn_before", "rpn_after"]].to_string(index=False)); print(tot)


if __name__ == "__main__":
    main()
