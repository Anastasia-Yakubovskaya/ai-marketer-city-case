import importlib.util

import pandas as pd
import yaml


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_process():
    module = load("src/process/process_model.py", "pm")
    config = yaml.safe_load(open("config/assumptions.yaml"))["process_model"]
    output = module.simulate_process(config, 42)
    launch = output[output.metric == "time_to_launch"].set_index("scenario")
    assert launch.loc["to_be", "p50"] < launch.loc["as_is", "p50"]


def test_portfolio():
    module = load("src/economics/portfolio.py", "po")
    config = yaml.safe_load(open("config/assumptions.yaml"))["portfolio"]
    data = pd.DataFrame(
        [
            ["G", "x", "push", "repeat", "good", 5, 5, 5, 5, 1],
            ["B", "x", "mixed", "new", "bad", 2, 1, 1, 1, 5],
        ],
        columns=[
            "campaign_id", "service", "channel", "offer_class", "description",
            "value", "repeatability", "data_readiness", "automation_readiness", "risk",
        ],
    )
    scored = module.score_portfolio(data, config)
    if "priority_score" in scored.columns:
        scored = scored.sort_values("priority_score", ascending=False)
    assert scored.iloc[0].campaign_id == "G"


def test_error_budget():
    module = load("src/safety/error_budget.py", "eb")
    config = yaml.safe_load(open("config/assumptions.yaml"))["error_budget"]
    incidents = pd.DataFrame([{"severity": "critical", "exposed_contacts": 100}])

    # Критический инцидент должен останавливать контур независимо от накопленного месячного burn rate.
    result = module.evaluate(incidents, config)
    assert result["action"] == "STOP_AND_DEMOTE"
