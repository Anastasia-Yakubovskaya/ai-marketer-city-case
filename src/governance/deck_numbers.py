import json
import sys
from math import ceil
from pathlib import Path

import pandas as pd
import yaml
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "outputs" / "tables"
META = ROOT / "outputs" / "meta"


def ni_sample_size(p: float, margin: float, alpha: float = 0.05, power: float = 0.8) -> int:
    z = norm.ppf(1 - alpha) + norm.ppf(power)
    return ceil(z**2 * 2 * p * (1 - p) / margin**2)


def build() -> tuple[dict, pd.DataFrame]:
    with open(ROOT / "config" / "assumptions.yaml", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    case = cfg["case_facts"]
    numbers = {}
    rows = []

    def add(metric, value, unit, evidence_type, source, rationale, slides):
        rows.append(
            {
                "metric": metric,
                "value": value,
                "unit": unit,
                "type": evidence_type,
                "source": source,
                "rationale": rationale,
                "slides": slides,
            }
        )

    case_metrics = [
        ("campaigns_per_month", "campaigns/month", "1,3"),
        ("team_cost_m_rub_per_month", "m RUB/month", "1,8"),
        ("communication_budget_m_rub_per_month", "m RUB/month", "1,8"),
        ("launch_days_as_is", "days", "1,2"),
        ("analysis_days_as_is", "days", "2"),
    ]
    for key, unit, slides in case_metrics:
        add(key, case[key], unit, "CASE_FACT", "case", "stated in the case", slides)

    labor = case["team_cost_m_rub_per_month"]
    communication = case["communication_budget_m_rub_per_month"]
    add(
        "labor_share",
        round(100 * labor / (labor + communication), 2),
        "%",
        "DERIVED_FACT",
        "case facts",
        "8.4/(8.4+40)",
        "8",
    )

    reconciliation = pd.read_csv(TABLES / "reconciliation.csv")
    numbers["reconciliation_mismatches"] = int(
        (reconciliation.status == "MISMATCH").sum()
    )
    reconciliation_index = reconciliation.set_index("check")
    add(
        "fte_cost_gap",
        float(reconciliation_index.loc["team_cost", "absolute_gap"]),
        "m RUB/month",
        "DERIVED_FACT",
        "reconciliation.csv",
        "26 FTE x 0.35 = 9.1 vs 8.4 stated in the case",
        "17",
    )
    process_cfg = cfg["process_model"]
    review_minutes = process_cfg["one_click_active_review_minutes"]
    add(
        "one_click_active_review_target",
        f"{review_minutes[0]}-{review_minutes[1]}",
        "minutes",
        "PILOT_TARGET",
        "config/assumptions.yaml",
        "active time spent reading the prepared approval card",
        "6",
    )
    add(
        "approval_lead_time_p50_target",
        process_cfg["approval_lead_time_p50_target_days"],
        "days",
        "PILOT_TARGET",
        "config/assumptions.yaml",
        "queue lead time is measured separately from active review time",
        "6,11",
    )

    add(
        "naive_sequential_days_range",
        "11.0-12.0",
        "days",
        "DERIVED_FACT",
        "reconciliation.csv",
        "naive sum of listed steps differs from stated 9-day lead time",
        "14",
    )

    process = pd.read_csv(TABLES / "process_summary.csv")
    launch = process[process.metric == "time_to_launch"].set_index("scenario")
    numbers["launch_tobe_p50"] = float(launch.loc["to_be", "p50"])
    add(
        "launch_time_to_be_p50",
        round(numbers["launch_tobe_p50"], 2),
        "days",
        "MODEL_RESULT",
        "process_model.py",
        "simulation of target process",
        "1,2,14",
    )

    evidence = process[process.metric == "time_to_evidence"]
    numbers["loop"] = {}
    for window in (7, 14, 28):
        as_is = float(
            evidence[
                (evidence.scenario == "as_is")
                & (evidence.observation_window_days == window)
            ]["mean"].iloc[0]
        )
        to_be = float(
            evidence[
                (evidence.scenario == "to_be")
                & (evidence.observation_window_days == window)
            ]["mean"].iloc[0]
        )
        reduction = 1 - to_be / as_is
        numbers["loop"][window] = {"as_is": as_is, "to_be": to_be, "cut": reduction}
        add(
            f"time_to_evidence_cut_W{window}",
            round(100 * reduction, 1),
            "%",
            "MODEL_RESULT",
            "process_model.py",
            f"{as_is:.1f} -> {to_be:.1f} days",
            "1,2,14",
        )

    add(
        "waiting_share_scenario",
        "40-80%",
        "% of 9-day lead time",
        "ASSUMPTION",
        "config/assumptions.yaml",
        "illustrative decomposition; measure in shadow mode",
        "2",
    )

    autonomy = pd.read_csv(TABLES / "autonomy_thresholds.csv").set_index("tier")
    numbers["autonomy"] = {}
    for tier, row in autonomy.iterrows():
        values = {
            "n": int(row.required_creatives_n),
            "pu": float(row.upper_miss_rate_p_u),
            "r0": float(row.max_contacts_without_canary),
            "r1": float(row.max_contacts_with_canary),
        }
        numbers["autonomy"][tier] = values
        add(
            f"autonomy_{tier}",
            values["n"],
            "creatives",
            "MODEL_RESULT",
            "autonomy.py",
            f"p_U={values['pu']:.3f}; reach {values['r0']/1e6:.2f}m / {values['r1']/1e6:.2f}m with canary",
            "6",
        )
    add(
        "canary_share_q",
        cfg["autonomy_rule"]["canary_share"],
        "share",
        "ASSUMPTION",
        "config/assumptions.yaml",
        "pilot design assumption",
        "6,9",
    )

    contact = pd.read_csv(TABLES / "contact_policy_violations.csv")
    p3_local = contact[
        (contact.strategy == "local_checks") & (contact.weekly_pressure == 3)
    ].iloc[0]
    p3_central = contact[
        (contact.strategy == "central_arbiter") & (contact.weekly_pressure == 3)
    ].iloc[0]
    add(
        "local_violation_rate_pressure3",
        round(100 * float(p3_local.any_policy_violation_rate), 1),
        "%",
        "MODEL_RESULT",
        "contact_policy.py",
        "synthetic simulation: 20k clients x 5 weeks",
        "4,16",
    )
    add(
        "central_value_loss_pressure3",
        round(100 * float(p3_central.value_loss), 1),
        "%",
        "MODEL_RESULT",
        "contact_policy.py",
        "synthetic value loss to enforce policy",
        "4,16",
    )

    gmv = pd.read_csv(TABLES / "gmv_sample_size.csv").set_index("order_value_cv")
    numbers["gmv"] = {
        str(key): {
            "n": int(row.n_per_arm),
            "ratio": float(row.ratio_vs_conversion),
            "n_cuped": int(row.n_per_arm_cuped),
        }
        for key, row in gmv.iterrows()
    }
    add(
        "n_campaign_vs_holdout_conversion",
        numbers["gmv"]["0.0"]["n"],
        "clients/arm",
        "MODEL_RESULT",
        "gmv_sample_size.py",
        "conversion proxy; primary metric remains GMV",
        "7",
    )
    experiment_cfg = cfg["experiment_design"]
    add(
        "experiment_control_share",
        round(100 * float(experiment_cfg["holdout_share"]), 1),
        "%",
        "ASSUMPTION",
        "config/assumptions.yaml",
        "control share used inside MDE / individual experiment calculations",
        "7",
    )
    persistent = experiment_cfg["persistent_holdout_share"]
    add(
        "persistent_holdout_share",
        f"{100*float(persistent[0]):.0f}-{100*float(persistent[1]):.0f}",
        "%",
        "PILOT_TARGET",
        "config/assumptions.yaml",
        "separate long-term holdout for cumulative effect and communication fatigue",
        "7",
    )
    for cv in ("1.0", "2.0", "3.0"):
        add(
            f"n_campaign_vs_holdout_gmv_cv{cv}",
            numbers["gmv"][cv]["n"],
            "clients/arm",
            "MODEL_RESULT",
            "gmv_sample_size.py",
            f"GMV primary metric, order-value CV={cv}",
            "7",
        )

    gmv_ni = pd.read_csv(TABLES / "gmv_noninferiority.csv")
    numbers["gmv_ni"] = {
        f"{row.order_value_cv:.1f}|{row.margin_share_of_effect:.1f}": int(row.n_per_arm)
        for row in gmv_ni.itertuples()
        if float(row.order_value_cv) > 0
    }
    for cv in (1.0, 2.0):
        for margin in (0.1, 0.2, 0.3):
            key = f"{cv:.1f}|{margin:.1f}"
            add(
                f"n_ai_vs_human_gmv_cv{cv:.0f}_margin{int(margin*100)}pct",
                numbers["gmv_ni"][key],
                "clients/arm",
                "MODEL_RESULT",
                "gmv_sample_size.py",
                "AI vs human non-inferiority on GMV",
                "7",
            )

    delta_power = pd.read_csv(TABLES / "economic_delta_power.csv")
    numbers["economic_delta_power"] = delta_power.to_dict("records")
    for row in delta_power.itertuples():
        add(
            f"economic_delta_power_cv{row.order_value_cv:.0f}",
            int(row.n_per_arm_cuped),
            "clients/arm with CUPED",
            "MODEL_RESULT",
            "gmv_sample_size.py",
            "sample needed to statistically resolve 4.6% growth of incremental effect; too large for M6 pilot",
            "7,11,12",
        )

    economics = pd.read_csv(TABLES / "payback_scenarios.csv").set_index("scenario")
    breakeven = pd.read_csv(TABLES / "breakeven.csv").set_index("scenario")
    coverage = pd.read_csv(TABLES / "economics_coverage_sensitivity.csv")
    numbers["econ"] = {}
    for scenario in economics.index:
        values = {
            "p50": float(economics.loc[scenario, "payback_p50"]),
            "p10": float(economics.loc[scenario, "payback_p10"]),
            "p90": float(economics.loc[scenario, "payback_p90"]),
            "p24": float(economics.loc[scenario, "prob_payback_24m"]),
            "be": float(breakeven.loc[scenario, "breakeven_delta"]),
        }
        numbers["econ"][scenario] = values
        add(
            f"payback_p50_{scenario}_full_eligible",
            values["p50"],
            "months",
            "MODEL_RESULT",
            "economics.py",
            f"full eligible contour; P(<=24m)={values['p24']:.3f}; break-even={values['be']:.4f}",
            "8,17",
        )

    half_cons = coverage[
        (coverage.coverage_case == "50%_eligible_value")
        & (coverage.scenario == "conservative")
    ].iloc[0]
    add(
        "payback_p50_conservative_50pct_eligible",
        float(half_cons.payback_p50),
        "months",
        "MODEL_RESULT",
        "economics.py",
        f"P(<=24m)={half_cons.prob_payback_24m:.3f}; break-even={half_cons.breakeven_delta_24m:.4f}",
        "8,17",
    )

    pilot_cost = pd.read_csv(TABLES / "pilot_cost_build_up.csv").set_index("component")
    gross = pilot_cost.loc["PILOT GROSS CASH COST"]
    numbers["pilot_gross"] = {
        "low": float(gross.low),
        "mode": float(gross["mode"]),
        "high": float(gross.high),
    }
    add(
        "pilot_gross_cash_cost",
        numbers["pilot_gross"]["mode"],
        "m RUB",
        "MODEL_RESULT",
        "pilot_cost.py",
        f"range {numbers['pilot_gross']['low']}-{numbers['pilot_gross']['high']}",
        "1,8,10,12,17",
    )

    judge_target = json.load(open(META / "judge_target.json", encoding="utf-8"))
    numbers["judge_target"] = judge_target
    add(
        "judge_recall_model_minimum",
        round(100 * judge_target["required_judge_recall_recommended"], 1),
        "%",
        "MODEL_RESULT",
        "judge_target.py",
        "model-derived minimum under assumed error taxonomy",
        "5,15",
    )
    add(
        "judge_recall_operational_target",
        85,
        "%",
        "PILOT_TARGET",
        "pilot decision",
        "rounded operating threshold above model minimum",
        "5,9,11,12",
    )

    legal = pd.read_csv(TABLES / "process_legal_queue.csv").set_index("group")
    add(
        "launch_blended_p50_with_legal_queue",
        round(float(legal.loc["blended", "launch_p50"]), 2),
        "days",
        "MODEL_RESULT",
        "legal_queue.py",
        "50/50 split is an assumption, not an organizer fact",
        "1,2,14",
    )

    capacity = json.load(open(META / "capacity_budget.json", encoding="utf-8"))
    numbers["capacity_budget"] = capacity
    add(
        "manual_review_scenario_fte",
        round(capacity["review_scenario_fte"], 2),
        "FTE",
        "ASSUMPTION",
        "capacity_budget.py",
        "120 campaigns x assumed 2h / 174h; validate in shadow mode",
        "3,17",
    )

    uplift_zero = pd.read_csv(TABLES / "uplift_zero.csv").set_index("scenario")
    add(
        "loss_if_uplift_zero_stopped_m6_conservative_p50",
        round(float(uplift_zero.loc["conservative", "loss_if_stopped_at_m6_p50"]), 1),
        "m RUB",
        "MODEL_RESULT",
        "uplift_zero.py",
        "stop after pilot if economic effect is absent",
        "8,12",
    )

    fmea = pd.read_csv(TABLES / "fmea.csv")
    add(
        "fmea_risk_count",
        int(len(fmea)),
        "risks",
        "ASSUMPTION",
        "risk_register.py",
        "expert ranking only; S/O/D are not measured probabilities",
        "9,18",
    )

    return numbers, pd.DataFrame(rows)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    numbers, registry = build()
    META.mkdir(parents=True, exist_ok=True)
    with open(META / "deck_numbers.json", "w", encoding="utf-8") as file:
        json.dump(numbers, file, ensure_ascii=False, indent=2, default=float)
    registry.to_csv(ROOT / "data" / "processed" / "evidence_registry.csv", index=False)
    print(registry[["metric", "value", "type", "slides"]].to_string(index=False))


if __name__ == "__main__":
    main()
