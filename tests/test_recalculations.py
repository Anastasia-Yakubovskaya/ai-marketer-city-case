import importlib.util, yaml, pandas as pd
def load(path, name):
    s = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
CFG = yaml.safe_load(open("config/assumptions.yaml", encoding="utf-8"))

def test_pilot_cost_bottom_up_uses_case_rate_and_matches_economics_config():
    m = load("src/economics/pilot_cost.py", "pc"); b = m.build_up(CFG["pilot_cost_bottom_up"])
    assert abs(b["team"] - 10.5) < 1e-9
    assert [round(x, 2) for x in b["gross"]] == [float(x) for x in CFG["economics"]["pilot_cost"]]
    assert [round(x, 2) for x in b["opex"]] == [float(x) for x in CFG["economics"]["post_pilot_opex"]]

def test_gmv_formula_reduces_to_conversion_when_cv_zero_and_grows_with_cv():
    m = load("src/experiments/gmv_sample_size.py", "gs")
    n0 = m.n_superiority(0.05, 0.2, 0.0); n1 = m.n_superiority(0.05, 0.2, 1.0); n3 = m.n_superiority(0.05, 0.2, 3.0)
    assert 8000 < n0 < 8300 and 1.9 < n1 / n0 < 2.2 and n3 > 9 * n0

def test_judge_target_formula_and_infeasibility():
    m = load("src/safety/judge_target.py", "jt"); jt = CFG["judge_target"]
    raw = m.raw_rates(0.08, 0.015, jt["pre_launch_share"], jt["post_launch_share"], 1.0)
    assert abs(sum(raw.values()) - 0.095) < 1e-9
    cov = {"contact_policy": 1.0, "offer_unavailable": 0.95, "text_vs_conditions": 0.95, "wrong_segment": 0.8, "typos_tone": 0.0}
    assert 0.70 < m.required_judge_recall(raw, cov, 0.015) < 0.75
    assert m.required_judge_recall(raw, cov, 0.001) is None

def test_addressable_budget_claim_is_infeasible_at_policy_cap():
    m = load("src/economics/capacity_budget.py", "cb"); bc = m.budget_check(CFG["case_facts"], CFG["addressable_budget"])
    assert not bc.claim_feasible.any() and abs(bc.iloc[0].max_push_spend_m_rub - 6.93) < 0.02

def test_capacity_bridge_review_scenario_fte():
    m = load("src/economics/capacity_budget.py", "cb2")
    cb = m.capacity_bridge(CFG["case_facts"], CFG["capacity_bridge"])
    row = cb[cb.role.str.startswith("manual review")].iloc[0]
    assert abs(row.fte_released_full - 120 * 2 / 174) < 1e-9
    assert row["type"] == "ASSUMPTION"

def test_legal_queue_increases_launch_time_for_new_campaigns():
    m = load("src/process/legal_queue.py", "lq"); df = m.simulate(CFG, n=20000).set_index("group")
    assert df.loc["new (legal queue)", "launch_p50"] > df.loc["template (pre-approved)", "launch_p50"] + 0.5

def test_uplift_zero_conservative_never_pays_back():
    m = load("src/economics/uplift_zero.py", "uz"); df = m.run(CFG["economics"], n=5000).set_index("scenario")
    assert df.loc["conservative", "prob_payback_36m"] < 0.01 and df.loc["conservative", "loss_if_stopped_at_m6_p50"] > 10
