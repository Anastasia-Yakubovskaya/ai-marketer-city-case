import importlib.util,pandas as pd,yaml
def load(path,name):
    s=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
def test_process():
    m=load('src/process/process_model.py','pm'); c=yaml.safe_load(open('config/assumptions.yaml'))['process_model']; o=m.simulate_process(c,42); x=o[o.metric=='time_to_launch'].set_index('scenario'); assert x.loc['to_be','p50']<x.loc['as_is','p50']
def test_portfolio():
    m=load('src/economics/portfolio.py','po'); cfg=yaml.safe_load(open('config/assumptions.yaml'))['portfolio']
    d=pd.DataFrame([['G','x','push','repeat','good',5,5,5,5,1],['B','x','mixed','new','bad',2,1,1,1,5]],columns=['campaign_id','service','channel','offer_class','description','value','repeatability','data_readiness','automation_readiness','risk'])
    r=m.score_portfolio(d,cfg).sort_values('priority_score',ascending=False) if 'priority_score' in m.score_portfolio(d,cfg).columns else m.score_portfolio(d,cfg)
    assert r.iloc[0].campaign_id=='G'
def test_error_budget():
    m=load('src/safety/error_budget.py','eb'); cfg=yaml.safe_load(open('config/assumptions.yaml'))['error_budget']
    d=pd.DataFrame([{'severity':'critical','exposed_contacts':100}]); r=m.evaluate(d,cfg); assert r['action']=='STOP_AND_DEMOTE'
