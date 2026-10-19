import importlib.util


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_sample_size_for_5_to_6_percent_is_about_8k_per_arm():
    mod = load_module("src/experiments/experiment_design.py", "exp")
    n = mod.two_proportion_sample_size(0.05, 0.06, 0.05, 0.80)
    assert 7500 <= n <= 9000


def test_cuped_reduces_required_sample():
    mod = load_module("src/experiments/experiment_design.py", "exp2")
    # При rho²=0.30 ожидаем примерно 30% снижения требуемой выборки.
    assert mod.cuped_adjusted_sample_size(8100, 0.30) == 5670


def test_mde_decreases_with_larger_audience():
    mod = load_module("src/experiments/experiment_design.py", "exp3")
    small = mod.mde_for_audience(20000, 0.10, 0.05)["mde_relative"]
    large = mod.mde_for_audience(50000, 0.10, 0.05)["mde_relative"]
    assert large < small


def test_noninferiority_margin_smaller_means_more_sample():
    mod = load_module("src/experiments/experiment_design.py", "exp4")
    n1 = mod.noninferiority_sample_size(0.06, 0.06, 0.003)
    n2 = mod.noninferiority_sample_size(0.06, 0.06, 0.006)
    # Чем строже допустимое ухудшение, тем больше данных нужно для non-inferiority.
    assert n1 > n2
