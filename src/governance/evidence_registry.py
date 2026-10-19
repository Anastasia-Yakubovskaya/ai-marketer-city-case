from pathlib import Path
import pandas as pd
import sys

ALLOWED = {"CASE_FACT", "DERIVED_FACT", "ASSUMPTION", "MODEL_RESULT", "PILOT_TARGET"}


def validate_registry(df):
    required = {"metric", "value", "unit", "type", "source", "rationale", "slides"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Не хватает колонок: {sorted(missing)}")
    bad_types = set(df["type"].dropna()) - ALLOWED
    if bad_types:
        raise ValueError(f"Неизвестные типы: {sorted(bad_types)}")
    return True


def main():
    if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
    df = pd.read_csv("data/processed/evidence_registry.csv")
    validate_registry(df)
    Path("outputs/reports").mkdir(parents=True, exist_ok=True)
    rows = [f"- {kind}: {(df.type == kind).sum()}" for kind in sorted(ALLOWED)]
    text = "# Проверка реестра доказательств\n\n" + "\n".join(rows) + "\n"
    Path("outputs/reports/evidence_registry_check.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
