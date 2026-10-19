import importlib.util
import numpy as np
import yaml


def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_central_arbiter_never_violates_caps():
    mod=load_module('src/safety/contact_policy.py','cp1')
    df,ret=mod.simulate_central_arbiter(np.random.default_rng(1),5000,8.0,0.4,0.1,
        {'low':0.0,'mode':1.0,'high':3.0},4,1)
    # Центральный арбитр должен соблюдать оба hard-cap независимо от нагрузки.
    assert df.sent_count.max() <= 4
    assert df.sms_count.max() <= 1
    assert not df.any_violation.any()
    assert 0 <= ret <= 1


def test_local_violations_increase_with_pressure():
    mod=load_module('src/safety/contact_policy.py','cp2')
    low=mod.simulate_local_checks(np.random.default_rng(2),50000,1.0,0.15,0.08,4,1)
    high=mod.simulate_local_checks(np.random.default_rng(3),50000,5.0,0.15,0.08,4,1)
    # Без общего арбитра рост давления кампаний должен увеличивать число конфликтов политики.
    assert high.any_violation.mean() > low.any_violation.mean()


def test_optout_suppression():
    mod=load_module('src/safety/contact_policy.py','cp3')
    df,_=mod.simulate_central_arbiter(np.random.default_rng(4),5000,5.0,0.2,0.5,
        {'low':0.0,'mode':1.0,'high':3.0},4,1)
    assert (df.loc[df.opted_out,'sent_count']==0).all()


def test_grid_has_both_strategies():
    mod=load_module('src/safety/contact_policy.py','cp4')
    with open('config/assumptions.yaml','r',encoding='utf-8') as f:
        cfg=yaml.safe_load(f)['contact_policy_model']
    cfg=dict(cfg); cfg['n_clients']=1000; cfg['n_weeks']=2; cfg['weekly_campaign_pressure_grid']=[2.0,4.0]
    out=mod.run_pressure_grid(cfg,42)
    assert set(out.strategy)=={'local_checks','central_arbiter'}
    assert set(out.weekly_pressure)=={2.0,4.0}
