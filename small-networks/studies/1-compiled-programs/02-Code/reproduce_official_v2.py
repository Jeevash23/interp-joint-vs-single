"""Bounded source-pinned CAE reproduction, with an explicit restricted-domain adaptation."""
from v2_resources import ROOT, guard, profile
guard(admission=True)
import importlib.util
import json
import sys
import numpy as np

vendor = ROOT/'02-Code/vendor/cae'
sys.path.insert(0,str(vendor/'test'))
from utils import load_system, collect_scores
suite=load_system(vendor/'test/11_controlled_experiments.py','official_controlled_suite')
ctrl=load_system('11_controlled_experiments.py','official_controlled_system')
from causal_abstraction import SystemState

out=ROOT/'04-Results/v2/official-cae-reproduction.json'
rows=[]
selected={'CAE_down_nf','CAE_up_nf','CAE_down','CAE_up'}
for mode in ('official','root_only_adaptation'):
    for index in range(20):
        guard()
        for cond in ('valid','xor_leak'):
            c=suite.build_exp2(cond,index)
            kwargs=dict(c.task_kwargs)
            if mode=='root_only_adaptation':
                kwargs['intervention_domain']=['X']
                kwargs['max_interventions']=1
            tasks=[t for t in c.tasks if t.name in selected]
            scores=collect_scores(c.engine.run_tasks(tasks,**kwargs),{})
            if mode=='official':
                iia=[m for m in c.analytical if type(m).__name__=='IIAMetric']
                scores.update(collect_scores({},c.engine.run_analytical_metrics(iia,sampler=c.sampler,n_samples=500)))
            rows.append({'mode':mode,'run_index':index,'condition':cond,
                         'settings':kwargs,'scores':scores})
        out.write_text(json.dumps({'rows':rows,'profile':profile()},indent=2)+'\n')
        print(mode,index+1,flush=True)

model=ctrl.XORBackupLowLevelModel(xor_active=False)
base=SystemState(values={'x':np.array([[0.]])})
test=model.forward_with_interventions(base,{'a':np.array([[1.]])})
diagnostic={k:np.asarray(v).tolist() for k,v in test.values.items()}
saved=json.loads(out.read_text())
saved['unmodified_source_semantics_check']={'input_x':0,'requested_a':1,'returned_state':diagnostic,
    'mediator_write_honored':bool(test.values['a'][0,0]==1)}
saved['adaptations']=['20 finite runs; direct factory invocation avoids CLI hardcoded runs=100',
 'four CAE metrics plus IIA only; all use official implementations',
 'root_only_adaptation restricts substantive intervention_domain to X and max_interventions=1; unmapped-variable faithfulness remains enabled in its metrics; separate from official results']
out.write_text(json.dumps(saved,indent=2)+'\n')
print(json.dumps(saved['profile']))
