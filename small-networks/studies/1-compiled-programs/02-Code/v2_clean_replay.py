"""Two-program replay in a fresh numerical/Tracr environment, no Torch required.
Recompiles models and repeats every assay with saved learned projectors. It does
not retrain DAS; that limitation is explicit in the replay record.
"""
from v2_resources import ROOT,guard,profile
from v2_model import build_program,Network,qualify
from v2_panel import evaluate,controls
import numpy as np
import json,gzip,sys,importlib.metadata as metadata

guard(admission=True);result=[]
for family,seed in [('lookup',10000),('additive',20000)]:
    model,meta=build_program(family,seed);net=Network.from_compiler(model,meta);gates,raw=qualify(model,net)
    folder=ROOT/f'03-Data/v2/final/{family}-{seed}';saved=Network.load(folder)
    exact=all(np.array_equal(v,saved.params[k][n]) for k,d in net.params.items() for n,v in d.items())
    assert exact and np.array_equal(net.rotation,saved.rotation)
    cs=np.load(folder/'candidates.npz');names=list(json.loads((ROOT/f'04-Results/v2/final/{family}-{seed}.json').read_text())['labels'])
    candidates={n:(cs[n+'_a'],cs[n+'_b']) for n in names}
    checks=controls(net,raw,candidates);records,summaries=evaluate(net,candidates)
    with gzip.open(ROOT/f'04-Results/v2/final/{family}-{seed}-queries.jsonl.gz','rt') as f:previous=[json.loads(x) for x in f]
    assert len(records)==len(previous)
    for a,b in zip(records,previous):
        assert {k:v for k,v in a.items() if k!='seconds'}=={k:v for k,v in b.items() if k!='seconds'}
    result.append({'family':family,'seed':seed,'parameter_arrays_bitwise_equal':exact,
                   'all_scientific_query_fields_exactly_equal':True,'queries':len(records),'gates':gates})
out={'all_pass':True,'python':sys.version,'executable':sys.executable,'programs':result,
     'scope':'Fresh pinned numerical+Tracr installation, models recompiled, saved learned candidates, no DAS retraining.',
     'packages':{x.metadata['Name']:x.version for x in metadata.distributions()},'profile':profile()}
(ROOT/'04-Results/v2/clean-environment-replay.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
