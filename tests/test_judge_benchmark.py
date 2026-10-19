import importlib.util, yaml

def load(path, name):
    s = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

def _run(seed=42):
    m = load("src/safety/judge_benchmark.py", "jb"); cfg = yaml.safe_load(open("config/assumptions.yaml"))["judge_benchmark"]
    df = m.generate_corpus(cfg["class_counts"], int(cfg["n_clean"]), seed); x = m.evaluate(df); return m, x, m.summarize(x)

def test_corpus_size_and_determinism():
    _, x1, _ = _run(); _, x2, _ = _run(); assert len(x1) == 1500 and list(x1.text) == list(x2.text)

def test_no_false_alarms_in_default_mode_and_numeric_classes_caught():
    m, x, (by_class, summary) = _run(); s = summary.set_index("group"); bc = by_class.set_index("error_class")
    assert s.loc["false_alarm_clean", "recall"] == 0.0
    assert bc.loc["wrong_date", "recall"] > 0.9 and bc.loc["wrong_min", "recall"] > 0.9 and bc.loc["missing_condition", "recall"] > 0.9

def test_rules_do_not_replace_judge_for_tone_and_hidden_offers():
    m, x, (by_class, summary) = _run(); bc = by_class.set_index("error_class")
    assert bc.loc["invented_offer", "recall"] < 0.8 and bc.loc["pressure", "recall"] < 0.8

def test_required_llm_recall_formula():
    m = load("src/safety/judge_benchmark.py", "jb2"); assert abs(m.required_llm_recall(0.8, 0.98) - 0.9) < 1e-9
