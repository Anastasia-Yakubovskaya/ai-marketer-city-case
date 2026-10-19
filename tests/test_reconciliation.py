import importlib.util
import yaml


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_known_team_cost_mismatch_is_detected():
    mod = load_module("src/governance/reconciliation.py", "reconciliation")
    cfg = yaml.safe_load(open("config/assumptions.yaml", encoding="utf-8"))["reconciliation"]
    out = mod.reconcile(cfg).set_index("check")
    assert abs(out.loc["team_cost", "calculated_value"] - 9.1) < 1e-12
    assert out.loc["team_cost", "status"] == "MISMATCH"


def test_naive_step_sum_exceeds_reported_launch_time():
    mod = load_module("src/governance/reconciliation.py", "reconciliation2")
    cfg = yaml.safe_load(open("config/assumptions.yaml", encoding="utf-8"))["reconciliation"]
    out = mod.reconcile(cfg).set_index("check")
    assert out.loc["launch_duration_lower_bound", "calculated_value"] > 9
