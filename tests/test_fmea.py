import importlib.util, pandas as pd
def load(path, name):
    s = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
def test_fmea_rpn_and_reduction():
    m = load("src/safety/risk_register.py", "fm"); df = pd.read_csv("data/processed/fmea_assumptions.csv"); r = m.compute_fmea(df)
    assert (r.rpn_before == r.severity * r.occurrence * r.detectability).all() and r.rpn_after.sum() < r.rpn_before.sum()
def test_fmea_rejects_out_of_scale():
    m = load("src/safety/risk_register.py", "fm2"); df = pd.read_csv("data/processed/fmea_assumptions.csv"); df.loc[0, "severity"] = 11
    try: m.compute_fmea(df); assert False
    except ValueError: pass
