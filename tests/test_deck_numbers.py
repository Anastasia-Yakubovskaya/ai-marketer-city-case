import importlib.util, math

def load(path, name):
    s = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

def test_ni_sample_size_matches_config_and_scales_with_margin():
    m = load("src/governance/deck_numbers.py", "dn"); assert m.ni_sample_size(0.06, 0.003) == 77489
    assert 8.9 < m.ni_sample_size(0.06, 0.001) / m.ni_sample_size(0.06, 0.003) < 9.1

def test_registry_has_no_empty_values_and_valid_types():
    m = load("src/governance/deck_numbers.py", "dn2"); N, reg = m.build()
    assert reg.value.notna().all() and set(reg.type) <= {"CASE_FACT", "DERIVED_FACT", "ASSUMPTION", "MODEL_RESULT", "PILOT_TARGET"}
