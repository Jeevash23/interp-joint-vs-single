"""Frozen candidate procedures: reference controls, mean contrasts and DAS adaptation.

DAS uses the standard rotated-coordinate interchange equation, optimized on
single-variable counterfactual output loss. This small CPU implementation is
an adaptation, not a rerun of Grant et al.'s trained-MLP/DAS experiment.
"""
from v2_resources import guard
import time
import numpy as np
# Import before Haiku: its optional-Flax module confuses torch's source lookup.
import torch
import torch._dynamo
from v2_model import INPUTS

NAMES=['compiler','compiler_null','mean_full','mean_restricted','das_restricted','random']

def basis(z,x,variable):
    values=np.unique(x[:,variable])
    means=np.array([z[x[:,variable]==v].mean(axis=0) for v in values])
    _,s,vt=np.linalg.svd(means-means.mean(axis=0),full_matrices=False)
    return vt[s>1e-6].T

def make_candidates(net,raw,steps=200):
    started=time.perf_counter();d=net.d;ka=len(net.a_idx);kb=len(net.b_idx)
    eye=np.eye(d);q=net.rotation
    pa=q.T@np.diag([1]*ka+[0]*(d-ka))@q
    pb=q.T@np.diag([0]*ka+[1]*kb+[0]*3)@q
    nn=np.zeros((d,d));nn[-3:,-3:]=np.eye(3)-np.ones((3,3))/3
    data={'compiler':(pa,pb),'compiler_null':(pa+q.T@nn@q,pb)}
    z=net.encode_site(raw)
    mask=(INPUTS[:,:2]<2).all(axis=1)
    for name,selection in [('mean_full',np.ones(27,dtype=bool)),('mean_restricted',mask)]:
        ua=basis(z[selection],INPUTS[selection],0);ub=basis(z[selection],INPUTS[selection],1)
        data[name]=(ua@ua.T,ub@ub.T)
    mean_seconds=time.perf_counter()-started
    learned,log=fit_das(net,raw[mask],INPUTS[mask],steps=steps)
    data['das_restricted']=learned
    r=2 if net.categorical else 1
    u,_=np.linalg.qr(np.random.default_rng(net.meta['seed']+910101).normal(size=(d,d)))
    data['random']=(u[:,:r]@u[:,:r].T,u[:,r:2*r]@u[:,r:2*r].T)
    for name,(a,b) in data.items():
        assert np.isfinite(a).all() and np.isfinite(b).all()
        assert np.max(np.abs(a@a-a))<1e-8 and np.max(np.abs(b@b-b))<1e-8
        assert np.max(np.abs(a@b))<1e-8,(name,np.max(np.abs(a@b)))
    return data,{'mean_contrast_seconds':mean_seconds,'native_activation_examples_full':27,
                 'native_activation_examples_restricted':12,'das':log,
                 'total_seconds':time.perf_counter()-started}

def fit_das(net,raw,inputs,steps):
    import torch
    torch.set_num_threads(2)
    started=time.perf_counter();d=net.d;r=2 if net.categorical else 1
    z=net.encode_site(raw);ua=basis(z,inputs,0);ub=basis(z,inputs,1)
    # Initialization uses only the discovery sample's conditional means.
    random=np.random.default_rng(net.meta['seed']+451337).normal(size=(d,d))
    init,_=np.linalg.qr(np.column_stack([ua,ub,random]))
    order=[0,2,1,3]+list(range(4,d)) if r==2 else list(range(d))
    q0=torch.tensor(init[:,order],dtype=torch.float64)
    theta=torch.nn.Parameter(torch.zeros((d,d),dtype=torch.float64))
    ri,di,vi=np.array([(i,j,v) for i in range(len(inputs)) for j in range(len(inputs))
                       for v in (0,1) if inputs[i,v]!=inputs[j,v]]).T
    target=inputs[ri].copy();target[np.arange(len(ri)),vi]=inputs[di,vi]
    expected=net.oracle(target)
    if net.categorical:
        y=np.zeros((len(ri),len(net.out_idx)))
        for i,v in enumerate(expected):y[i,net.output_values.index(int(v))]=1
    else:y=expected[:,None]
    rt=torch.tensor(raw[ri],dtype=torch.float64);zt=torch.tensor(z[ri],dtype=torch.float64)
    dz=torch.tensor(z[di]-z[ri],dtype=torch.float64)
    tar=torch.tensor(y,dtype=torch.float64);selector=torch.tensor(vi==0)[:,None]
    w1,b1,w2,b2=[torch.tensor(v,dtype=torch.float64) for v in (net.w1,net.b1,net.w2,net.b2)]
    inv=torch.tensor(net.rotation.T,dtype=torch.float64);ident=torch.eye(d,dtype=torch.float64)
    optim=torch.optim.Adam([theta],lr=.02)
    def projectors():
        s=(theta-theta.T)/2
        u=q0@torch.linalg.solve(ident-s,ident+s)
        return u[:,:r]@u[:,:r].T,u[:,r:2*r]@u[:,r:2*r].T
    history=[];best_loss=float('inf');best=None;updates=0
    for step in range(steps):
        if step%25==0:guard()
        a,b=projectors();edited=zt+torch.where(selector,dz@a,dz@b)
        h=rt.clone();h[:,net.pool]=edited@inv
        pred=(h+torch.relu(h@w1+b1)@w2+b2)[:,net.out_idx]
        loss=((pred-tar)**2).mean()/(1 if net.categorical else 100)
        assert torch.isfinite(loss)
        lv=float(loss.detach())
        if lv<best_loss:best_loss=lv;best=(a.detach().numpy().copy(),b.detach().numpy().copy())
        if step==0 or (step+1)%25==0:history.append({'step':step+1,'loss':float(loss.detach())})
        # Do not let Adam turn floating-point roundoff at an exact solution
        # into an artificial counterexample. Selected solely on training loss.
        if lv<=1e-12:break
        optim.zero_grad();loss.backward();optim.step()
        updates+=1
    aa,bb=best
    patched=z[ri]+np.where((vi==0)[:,None],(z[di]-z[ri])@aa,(z[di]-z[ri])@bb)
    out=net.suffix(net.replace_site(raw[ri],patched))
    return (aa,bb),{'max_steps':steps,'updates':updates,'evaluations':step+1,'best_training_loss':best_loss,
                    'convergence_loss':1e-12,'learning_rate':.02,'training_pairs':len(ri),
                    'differentiable_suffix_examples':(step+1)*len(ri),
                    'initialization':'restricted conditional mean contrasts + seeded random completion',
                    'history':history,'training_output_agreement':float(np.mean(np.isclose(out,expected,atol=1e-4,rtol=0))),
                    'seconds':time.perf_counter()-started}

def reference_label(net,raw,pa,pb):
    """Exhaustive linear carrier laws; never uses any suffix/joint output."""
    z=net.encode_site(raw);diff=z[None,:,:]-z[:,None,:]
    carriers=net.a_idx+net.b_idx;k=len(net.a_idx);true=raw[:,carriers]
    full_delta=true[None,:,:]-true[:,None,:]
    errors=[];witnesses=[]
    for v,p in enumerate((pa,pb)):
        actual=(diff@p@net.rotation.T)[...,:len(carriers)]
        wanted=np.zeros_like(actual)
        wanted[...,:k] = full_delta[...,:k] if v==0 else 0
        wanted[...,k:] = full_delta[...,k:] if v==1 else 0
        err=np.abs(actual-wanted);ix=np.unravel_index(np.argmax(err),err.shape)
        errors.append(float(err[ix]));witnesses.append({'recipient':INPUTS[ix[0]].tolist(),
             'donor':INPUTS[ix[1]].tolist(),'carrier_component':int(ix[2]),'actual':float(actual[ix]),'wanted':float(wanted[ix])})
    commute=float(np.max(np.abs(pa@pb-pb@pa)))
    return {'reference_consistent':max(errors)<1e-4 and commute<1e-8,
            'max_carrier_errors':errors,'commutator_max_abs':commute,'witnesses':witnesses,
            'reference_recipient_donor_pairs_per_variable':729}

if __name__=='__main__':
    from v2_resources import ROOT,profile
    from v2_model import Network
    import json,sys
    guard(admission=True);rows=[]
    for family in ('lookup','additive'):
        for seed in range(4):
            net=Network.load(ROOT/f'03-Data/v2/development/{family}-{seed}')
            _,layers=net.native_numpy(INPUTS);raw=layers[net.site][:,-1]
            candidates,discovery=make_candidates(net,raw,steps=int(sys.argv[1]) if len(sys.argv)>1 else 200)
            labels={name:reference_label(net,raw,*p) for name,p in candidates.items()}
            row={'family':family,'seed':seed,'discovery':discovery,'labels':labels};rows.append(row)
            folder=ROOT/f'03-Data/v2/development/{family}-{seed}'
            np.savez_compressed(folder/'candidates.npz',**{name+'_'+v:p[i] for name,p in candidates.items() for i,v in enumerate(('a','b'))})
            print(family,seed,discovery['das']['training_output_agreement'],discovery['total_seconds'],{n:l['reference_consistent'] for n,l in labels.items()},flush=True)
            (ROOT/'04-Results/v2/development-candidates.json').write_text(json.dumps({'rows':rows,'profile':profile()},indent=2)+'\n')
