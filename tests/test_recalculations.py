import importlib.util

import pandas as pd
import yaml


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CFG = yaml.safe_load(open("config/assumptions.yaml", encoding="utf-8"))


def test_pilot_cost_bottom_up_uses_case_rate_and_matches_economics_config():
    module = load("src/economics/pilot_cost.py", "pc")
    build_up = module.build_up(CFG["pilot_cost_bottom_up"])
    assert abs(build_up["team"] - 10.5) < 1e-9
    assert [round(x, 2) for x in build_up["gross"]] == [float(x) for x in CFG["economics"]["pilot_cost"]]
    assert [round(x, 2) for x in build_up["opex"]] == [float(x) for x in CFG["economics"]["post_pilot_opex"]]


def test_gmv_formula_reduces_to_conversion_when_cv_zero_and_grows_with_cv():
    module = load("src/experiments/gmv_sample_size.py", "gs")
    n0 = module.n_superiority(0.05, 0.2, 0.0)
    n1 = module.n_superiority(0.05, 0.2, 1.0)
    n3 = module.n_superiority(0.05, 0.2, 3.0)

    # При CV=0 формула должна почти совпадать с бинарной конверсией; рост CV обязан увеличивать выборку.
    assert 8000 < n0 < 8300
    assert 1.9 < n1 / n0 < 2.2
    assert n3 > 9 * n0


def test_judge_target_formula_and_infeasibility():
    module = load("src/safety/judge_target.py", "jt")
    judge_cfg = CFG["judge_target"]
    raw = module.raw_rates(
        0.08,
        0.015,
        judge_cfg["pre_launch_share"],
        judge_cfg["post_launch_share"],
        1.0,
    )
    assert abs(sum(raw.values()) - 0.095) < 1e-9

    coverage = {
        "contact_policy": 1.0,
        "offer_unavailable": 0.95,
        "text_vs_conditions": 0.95,
        "wrong_segment": 0.8,
        "typos_tone": 0.0,
    }
    assert 0.70 < module.required_judge_recall(raw, coverage, 0.015) < 0.75
    assert module.required_judge_recall(raw, coverage, 0.001) is None


def test_addressable_budget_claim_is_infeasible_at_policy_cap():
    module = load("src/economics/capacity_budget.py", "cb")
    budget = module.budget_check(CFG["case_facts"], CFG["addressable_budget"])
    assert not budget.claim_feasible.any()
    assert abs(budget.iloc[0].max_push_spend_m_rub - 6.93) < 0.02


def test_capacity_bridge_review_scenario_fte():
    module = load("src/economics/capacity_budget.py", "cb2")
    capacity = module.capacity_bridge(CFG["case_facts"], CFG["capacity_bridge"])
    row = capacity[capacity.role.str.startswith("manual review")].iloc[0]
    assert abs(row.fte_released_full - 120 * 2 / 174) < 1e-9
    assert row["type"] == "ASSUMPTION"


def test_legal_queue_increases_launch_time_for_new_campaigns():
    module = load("src/process/legal_queue.py", "lq")
    result = module.simulate(CFG, n=20000).set_index("group")
    assert result.loc["new (legal queue)", "launch_p50"] > result.loc["template (pre-approved)", "launch_p50"] + 0.5


def test_uplift_zero_conservative_never_pays_back():
    module = load("src/economics/uplift_zero.py", "uz")
    result = module.run(CFG["economics"], n=5000).set_index("scenario")

    # При нулевом uplift conservative-сценарий не должен случайно окупаться за счёт capacity.
    assert result.loc["conservative", "prob_payback_36m"] < 0.01
    assert result.loc["conservative", "loss_if_stopped_at_m6_p50"] > 10
