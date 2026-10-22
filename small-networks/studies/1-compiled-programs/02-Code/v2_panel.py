"""Explicit uncached validation queries and finite instrumentation controls."""
from v2_resources import guard
import time
import numpy as np
from v2_model import INPUTS

BUDGETS=[12,24,60,120]
COSTS={'ordinary':3,'full_single':3,'heldout_single':3,'joint':4,
       'faithfulness':3,'raw_latent':4,'transport':6,'natural':1}
METHODS=list(COSTS)
TOL=1e-4

def queries(seed,method,n):
    # Joint, latent, transport and natural share a nested query stream.
    kind=method if method in ('ordinary','full_single','heldout_single','faithfulness') else 'joint'
    rng=np.random.default_rng(seed+390101+METHODS.index(kind)*1009)
    restricted=np.flatnonzero((INPUTS[:,:2]<2).all(axis=1));rows=[]
    for i in range(n):
        v=i%2
        while True:
            candidates=restricted if kind=='ordinary' else np.arange(27)
            ri=int(rng.choice(candidates));da=int(rng.choice(candidates));db=int(rng.choice(candidates))
            r,a,b=INPUTS[ri],INPUTS[da],INPUTS[db]
            target=r.copy()
            if kind=='joint':
                if r[0]==a[0] or r[1]==b[1]:continue
                target[:2]=[a[0],b[1]]
            elif kind!='faithfulness':
                if r[v]==a[v]:continue
                target[v]=a[v]
                if kind=='heldout_single' and not (target[:2]==2).any():continue
            break
        rows.append({'ri':ri,'da':da,'db':db,'v':v,'anchor':int(rng.integers(27)),
                     'target':target.tolist()})
    return rows

def evaluate(net,candidates):
    records=[];summaries=[]
    def state(ix):
        _,layers=net.native_numpy(INPUTS[[ix]])
        raw=layers[net.site][:,-1]
        return raw,net.encode_site(raw)
    for method,cost in COSTS.items():
        plan=queries(net.meta['seed'],method,max(BUDGETS)//cost)
        for name,(pa,pb) in candidates.items():
            elapsed=0;fails=0;native_calls=0;suffix_calls=0
            for i,q in enumerate(plan):
                if i%20==0:guard()
                started=time.perf_counter();expected=float(net.oracle(q['target'])[0])
                if method=='natural':
                    pred=float(net.native_numpy([q['target']])[0][0,-1]);native_calls+=1
                    error=abs(pred-expected)
                else:
                    raw,z=state(q['ri']);_,za=state(q['da']);native_calls+=2
                    if method in ('ordinary','full_single','heldout_single'):
                        p=pa if q['v']==0 else pb;zp=z+(za-z)@p
                    elif method=='faithfulness':
                        zp=z+(za-z)@(np.eye(net.d)-pa-pb)
                        expected=float(net.oracle(INPUTS[[q['ri']]])[0])
                    else:
                        _,zb=state(q['db']);native_calls+=1
                        zp=z+(za-z)@pa+(zb-z)@pb
                    if method in ('raw_latent','transport'):
                        _,ls=net.native_numpy([q['target']]);rn=ls[net.site][:,-1]
                        delta=zp-net.encode_site(rn);native_calls+=1
                        if method=='raw_latent':
                            pred=float(np.linalg.norm(delta));expected=0.;error=abs(pred)
                        else:
                            ra,az=state(q['anchor']);native_calls+=1
                            pred=float(net.suffix(net.replace_site(ra,az+delta))[0]);suffix_calls+=1
                            expected=float(net.oracle(INPUTS[[q['anchor']]])[0]);error=abs(pred-expected)
                    else:
                        pred=float(net.suffix(net.replace_site(raw,zp))[0]);suffix_calls+=1;error=abs(pred-expected)
                duration=time.perf_counter()-started;elapsed+=duration
                assert np.isfinite(pred) and np.isfinite(error)
                failed=bool(error>TOL);fails+=failed
                records.append({'candidate':name,'method':method,'query':i,**q,'observed':pred,
                    'expected':expected,'error':error,'failed':failed,'seconds':duration,
                    'cumulative_cost':cost*(i+1)})
                if cost*(i+1) in BUDGETS:
                    summaries.append({'candidate':name,'method':method,'budget':cost*(i+1),
                        'queries':i+1,'failures':fails,'accepted':not bool(fails),'seconds':elapsed,
                        'native_calls':native_calls,'suffix_calls':suffix_calls})
    return records,summaries

def controls(net,raw,candidates):
    z=net.encode_site(raw);summary={};all_self=True;max_locality=0.;max_order=0.
    for name,(pa,pb) in candidates.items():
        unchanged=z+(z-z)@pa+(z-z)@pb
        self_err=float(np.max(np.abs(net.suffix(net.replace_site(raw,unchanged))-net.oracle(INPUTS))))
        all_self &= self_err<TOL
        q=queries(net.meta['seed'],'joint',12)
        ri=np.array([x['ri'] for x in q]);da=np.array([x['da'] for x in q]);db=np.array([x['db'] for x in q])
        both=z[ri]+(z[da]-z[ri])@pa+(z[db]-z[ri])@pb
        first=z[ri]+(z[da]-z[ri])@pa
        sequential=first+(z[db]-first)@pb
        reverse=z[ri]+(z[db]-z[ri])@pb
        reverse=reverse+(z[da]-reverse)@pa
        order=float(max(np.max(np.abs(both-sequential)),np.max(np.abs(both-reverse))));max_order=max(max_order,order)
        patched=net.replace_site(raw[ri],both)
        outside=[x for x in range(raw.shape[1]) if x not in net.pool]
        loc=float(np.max(np.abs(patched[:,outside]-raw[ri][:,outside])));max_locality=max(max_locality,loc)
        irrelevant=[]
        for v,p in enumerate((pa,pb)):
            pairs=np.array([(i,j) for i in range(27) for j in range(27) if INPUTS[i,v]==INPUTS[j,v]])
            ir,dr=pairs.T;zp=z[ir]+(z[dr]-z[ir])@p
            out=net.suffix(net.replace_site(raw[ir],zp))
            irrelevant.append(int(np.sum(np.abs(out-net.oracle(INPUTS[ir]))>TOL)))
        # Equivalent change of orthonormal coordinates around this exact adapter.
        rot,_=np.linalg.qr(np.random.default_rng(net.meta['seed']+779).normal(size=(net.d,net.d)))
        zz=z@rot;pp=rot.T@pa@rot;qq=rot.T@pb@rot
        equiv=(zz[ri]+(zz[da]-zz[ri])@pp+(zz[db]-zz[ri])@qq)@rot.T
        sym=float(np.max(np.abs(equiv-both)))
        summary[name]={'self_max_error':self_err,'order_max_error':order,'write_locality_max_error':loc,
                       'irrelevant_donor_output_failures':irrelevant,'irrelevant_pairs_per_variable':243,
                       'rotation_equivalence_max_error':sym}
        assert order<1e-8 and loc==0 and sym<1e-8
    assert all_self
    # Match each compiler patch's length, randomize only its direction.
    pa,pb=candidates['compiler'];rng=np.random.default_rng(net.meta['seed']+661)
    q=queries(net.meta['seed'],'joint',30);pairs=[]
    for t in q:
        ri,da,db=t['ri'],t['da'],t['db']
        delta=(z[da]-z[ri])@pa+(z[db]-z[ri])@pb
        direction=rng.normal(size=net.d);direction*=np.linalg.norm(delta)/np.linalg.norm(direction)
        out=float(net.suffix(net.replace_site(raw[[ri]],z[[ri]]+direction))[0])
        wanted=float(net.oracle(t['target'])[0])
        pairs.append({'target':t['target'],'norm':float(np.linalg.norm(delta)),
                      'random_norm':float(np.linalg.norm(direction)),'observed':out,'expected':wanted,'failed':abs(out-wanted)>TOL})
    # Legal-domain entry point is integer triples in 0..2; invalid donors are
    # rejected before forward execution, not counted as mechanistic failures.
    bad=[[3,0,0],[-1,1,1],[0,1],[0,.5,2]]
    for x in bad:
        try:net.native_numpy(x)
        except ValueError:pass
        else:raise AssertionError(('inadmissible_donor_accepted',x))
    return {'all_instrumentation_gates_pass':True,'candidates':summary,
            'matched_norm_random_direction':pairs,'inadmissible_donors_rejected':bad}

if __name__=='__main__':
    from v2_resources import ROOT,profile
    from v2_model import Network
    import json,gzip
    guard(admission=True);out=[]
    for family in ('lookup','additive'):
        for seed in range(4):
            folder=ROOT/f'03-Data/v2/development/{family}-{seed}';net=Network.load(folder)
            c=np.load(folder/'candidates.npz');names=sorted(set(n[:-2] for n in c.files))
            cand={n:(c[n+'_a'],c[n+'_b']) for n in names}
            _,ls=net.native_numpy(INPUTS);raw=ls[net.site][:,-1]
            checks=controls(net,raw,cand);records,summary=evaluate(net,cand)
            out.append({'family':family,'seed':seed,'summaries':summary,'controls':checks})
            print(family,seed,'panel passed',flush=True)
    (ROOT/'04-Results/v2/development-panel.json').write_text(json.dumps({'rows':out,'profile':profile()},indent=2)+'\n')
