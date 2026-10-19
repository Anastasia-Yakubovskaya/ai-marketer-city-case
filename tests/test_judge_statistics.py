import importlib.util


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_zero_failure_upper_bound_50_is_about_5_8_percent():
    mod = load_module("src/safety/judge_statistics.py", "judge")
    value = mod.zero_failure_upper_bound(50, 0.95)
    assert abs(value - 0.05816) < 0.0002


def test_98_recall_needs_149_clean_examples():
    mod = load_module("src/safety/judge_statistics.py", "judge2")
    assert mod.required_zero_failure_sample(0.98, 0.95) == 149


def test_99_recall_needs_299_clean_examples():
    mod = load_module("src/safety/judge_statistics.py", "judge3")
    assert mod.required_zero_failure_sample(0.99, 0.95) == 299


def test_kappa_perfect_agreement_is_one():
    mod = load_module("src/safety/judge_statistics.py", "judge4")
    assert mod.cohens_kappa([[10,0],[0,20]]) == 1.0
