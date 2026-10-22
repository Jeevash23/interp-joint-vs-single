"""Resumable final run. Requires a frozen, matching source manifest."""
from v2_resources import ROOT,guard,profile
from v2_candidates import make_candidates,reference_label
from v2_model import build_program,Network,qualify,INPUTS
from v2_panel import evaluate,controls
import numpy as np
import json,gzip,hashlib,time,sys,traceback

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def run_one(family,seed,split='final'):
    guard(admission=True);started=time.perf_counter()
    model,meta=build_program(family,seed);net=Network.from_compiler(model,meta)
    gates,raw=qualify(model,net)
    folder=ROOT/f'03-Data/v2/{split}/{family}-{seed}';net.save(folder)
    # Duplicate audit does not exclude or resample: all frozen seeds are kept.
    duplicate=[]
    for old in (ROOT/'03-Data/v2').glob('*/*/model.json'):
        if old.parent==folder:continue
        prior=json.loads(old.read_text())['meta']
        if prior['family']==family and prior['table']==meta['table']:duplicate.append(str(old.relative_to(ROOT)))
    candidates,discovery=make_candidates(net,raw,steps=200)
    np.savez_compressed(folder/'candidates.npz',**{n+'_'+v:p[i] for n,p in candidates.items() for i,v in enumerate(('a','b'))})
    labels={n:reference_label(net,raw,*p) for n,p in candidates.items()}
    checks=controls(net,raw,candidates)
    queries,summaries=evaluate(net,candidates)
    dest=ROOT/f'04-Results/v2/{split}';dest.mkdir(exist_ok=True)
    with gzip.open(dest/f'{family}-{seed}-queries.jsonl.gz','wt') as f:
        for q in queries:f.write(json.dumps(q,separators=(',',':'))+'\n')
    result={'family':family,'seed':seed,'meta':meta,'gates':gates,'labels':labels,
            'discovery':discovery,'controls':checks,'summaries':summaries,
            'duplicates':duplicate,'query_records':len(queries),'seconds':time.perf_counter()-started,
            'profile':profile()}
    path=dest/f'{family}-{seed}.json';tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(path)
    return result

if __name__=='__main__':
    protocol=ROOT/'06-Study-History/V2-FROZEN-MANIFEST.json'
    frozen=json.loads(protocol.read_text())
    for path,sha in frozen['source_sha256'].items():assert digest(ROOT/path)==sha,('changed_frozen_source',path)
    guard(admission=True)
    for family,seeds in frozen['final_seeds'].items():
        for seed in seeds:
            target=ROOT/f'04-Results/v2/final/{family}-{seed}.json'
            if target.exists():continue
            try:
                row=run_one(family,seed)
                print(family,seed,'qualified',round(row['seconds'],3),'seconds',flush=True)
            except Exception as e:
                dest=ROOT/'04-Results/v2/final';dest.mkdir(exist_ok=True)
                (dest/f'{family}-{seed}-FAILED.json').write_text(json.dumps({'family':family,'seed':seed,'error':str(e),'traceback':traceback.format_exc()},indent=2))
                raise
    print('All frozen programs finished.',profile(),flush=True)
