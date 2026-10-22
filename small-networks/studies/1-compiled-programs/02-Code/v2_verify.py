"""Independent saved-record verifier; imports no experiment implementation.

It shares NumPy and the declared semantic definition, but uses explicit per-head
attention and direct matrix operations for independent execution/composition.
"""
from v2_resources import ROOT,guard,profile
import gzip,json,itertools,hashlib
import numpy as np

X=np.array(list(itertools.product(range(3),repeat=3)))

def load(folder):
    meta=json.loads((folder/'model.json').read_text());w=dict(np.load(folder/'weights.npz'));return meta,w

def forward(meta,w):
    tm=meta['token_map'];emb=w['token_embed|embeddings'];pos=w['pos_embed|embeddings'][:4]
    all_sites=[];outs=[];labels=meta['labels'];oi=[i for i,x in enumerate(labels) if x.startswith('answer_')]
    for inp in X:
        token=[tm['BOS']]+[tm[str(v)] for v in inp];h=emb[token]+pos
        for layer in range(meta['num_layers']):
            prefix=f'transformer/layer_{layer}'
            q,k,v=[h@w[prefix+'/attn/'+s+'|w']+w[prefix+'/attn/'+s+'|b'] for s in ['query','key','value']]
            chunks=[];dim=meta['key_size']
            for head in range(meta['num_heads']):
                sl=slice(head*dim,(head+1)*dim)
                logits=(q[:,sl]@k[:,sl].T)/np.sqrt(dim)
                att=np.array([np.exp(row-row.max())/np.exp(row-row.max()).sum() for row in logits])
                chunks.append(att@v[:,sl])
            h+=np.concatenate(chunks,axis=1)@w[prefix+'/attn/linear|w']+w[prefix+'/attn/linear|b']
            if layer==meta['num_layers']-1:all_sites.append(h[-1].copy())
            a=h@w[prefix+'/mlp/linear_1|w']+w[prefix+'/mlp/linear_1|b'];a[a<0]=0
            h+=a@w[prefix+'/mlp/linear_2|w']+w[prefix+'/mlp/linear_2|b']
        val=h[-1,oi]
        outs.append(meta['output_values'][val.argmax()] if meta['categorical'] else val[0])
    return np.array(all_sites),np.array(outs)

def verify():
    guard(admission=True);details=[];seen={};records_count=0;full_census=[];audit=[];differences=[]
    frozen=json.loads((ROOT/'06-Study-History/V2-FROZEN-MANIFEST.json').read_text())
    for p,h in frozen['source_sha256'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h
    saved_analysis=json.loads((ROOT/'04-Results/v2/analysis.json').read_text());counts={}
    for family,seeds in frozen['final_seeds'].items():
        for seed in seeds:
            guard();folder=ROOT/f'03-Data/v2/final/{family}-{seed}'
            r=json.loads((ROOT/f'04-Results/v2/final/{family}-{seed}.json').read_text())
            meta,w=load(folder);raw,out=forward(meta,w);table=np.array(meta['meta']['table']);want=table[X[:,0],X[:,1]]
            assert np.max(np.abs(out-want))<1e-4
            c=dict(np.load(folder/'candidates.npz'));lab=meta['labels']
            ai=[i for i,s in enumerate(lab) if s.startswith('var_a_')];bi=[i for i,s in enumerate(lab) if s.startswith('var_b_')]
            ni=[lab.index('tokens:'+str(i)) for i in range(3)];pool=ai+bi+ni;q=w['rotation'];z=raw[:,pool]@q
            ab=raw[:,ai+bi];k=len(ai);dim=len(ai+bi)
            prefix=f"transformer/layer_{meta['num_layers']-1}/mlp/"
            oi=[i for i,s in enumerate(lab) if s.startswith('answer_')]
            def logits(h):
                mid=np.maximum(h@w[prefix+'linear_1|w']+w[prefix+'linear_1|b'],0)
                return (h+mid@w[prefix+'linear_2|w']+w[prefix+'linear_2|b'])[:,oi]
            def suffix(h):
                val=logits(h)
                return np.array(meta['output_values'])[val.argmax(axis=1)] if meta['categorical'] else val[:,0]
            def edited(ri,da,db,name):
                a,b=c[name+'_a'],c[name+'_b'];base=z[ri]
                # Independently compose actual replacement operations in order.
                first=base@(np.eye(len(q))-a)+z[da]@a
                second=first@(np.eye(len(q))-b)+z[db]@b
                both=base+(z[da]-base)@a+(z[db]-base)@b
                assert np.max(np.abs(second-both))<1e-8
                h=raw[ri].copy();h[:,pool]=second@q.T;return h
            for name,l in r['labels'].items():
                ok=True;worst=0
                for v in (0,1):
                    p=c[name+('_a' if v==0 else '_b')]
                    for ri in range(27):
                        for di in range(27):
                            h=z[ri]@(np.eye(len(q))-p)+z[di]@p
                            actual=(h@q.T)[:dim];target=ab[ri].copy()
                            if v==0:target[:k]=ab[di,:k]
                            else:target[k:]=ab[di,k:]
                            worst=max(worst,float(np.max(np.abs(actual-target))))
                ok=worst<1e-4
                assert ok==l['reference_consistent'],(family,seed,name,worst,l)
            with gzip.open(ROOT/f'04-Results/v2/final/{family}-{seed}-queries.jsonl.gz','rt') as f:records=[json.loads(line) for line in f]
            records_count+=len(records)
            for s in r['summaries']:
                subset=[x for x in records if x['method']==s['method'] and x['candidate']==s['candidate'] and x['cumulative_cost']<=s['budget']]
                failed=sum(x['error']>1e-4 for x in subset)
                assert failed==s['failures'] and (failed==0)==s['accepted'] and len(subset)==s['queries']
                assert all(x['failed']==(x['error']>1e-4) and abs(abs(x['observed']-x['expected'])-x['error'])<1e-10 for x in subset)
                valid=r['labels'][s['candidate']]['reference_consistent'];key=(s['budget'],s['method'],valid)
                pair=counts.setdefault(key,[0,0]);pair[0]+=s['accepted'];pair[1]+=1
            # Independently rerun every saved assay, with explicit margin audit.
            for name in r['labels']:
                for method in ('ordinary','full_single','heldout_single','joint','faithfulness','raw_latent','transport','natural'):
                    rows=[x for x in records if x['candidate']==name and x['method']==method]
                    ri=np.array([x['ri'] for x in rows]);da=np.array([x['da'] for x in rows]);db=np.array([x['db'] for x in rows]);vi=np.array([x['v'] for x in rows])
                    a,b=c[name+'_a'],c[name+'_b'];zz=z[ri]
                    target=np.array([x['target'] for x in rows]);target_index=target[:,0]*9+target[:,1]*3+target[:,2]
                    wanted=table[target[:,0],target[:,1]]
                    if method=='natural':vals=None;pred=out[target_index]
                    else:
                        if method in ('ordinary','full_single','heldout_single'):
                            zz=zz+np.where((vi==0)[:,None],(z[da]-zz)@a,(z[da]-zz)@b)
                        elif method=='faithfulness':
                            zz=zz+(z[da]-zz)@(np.eye(len(q))-a-b);wanted=table[X[ri,0],X[ri,1]]
                        else:zz=zz+(z[da]-zz)@a+(z[db]-zz)@b
                        if method=='raw_latent':
                            pred=np.linalg.norm(zz-z[target_index],axis=1);wanted=np.zeros(len(rows));vals=None
                        else:
                            if method=='transport':
                                anchor=np.array([x['anchor'] for x in rows]);zz=z[anchor]+zz-z[target_index]
                                h=raw[anchor].copy();wanted=table[X[anchor,0],X[anchor,1]]
                            else:h=raw[ri].copy()
                            h[:,pool]=zz@q.T;vals=logits(h)
                            pred=np.array(meta['output_values'])[vals.argmax(axis=1)] if meta['categorical'] else vals[:,0]
                    assert np.max(np.abs(wanted-[x['expected'] for x in rows]))<1e-4
                    if meta['categorical'] and vals is not None:
                        wanted_idx=np.array([meta['output_values'].index(int(t)) for t in wanted]);actual=vals[np.arange(len(rows)),wanted_idx]
                        other=vals.copy();other[np.arange(len(rows)),wanted_idx]=-np.inf
                        signed=actual-other.max(axis=1)
                        states=np.where(signed>1e-4,'pass',np.where(signed< -1e-4,'fail','indeterminate'))
                        for i,row in enumerate(rows):
                            if pred[i]!=row['observed']:
                                saved_idx=meta['output_values'].index(int(row['observed']))
                                gap=float(vals[i].max()-vals[i,saved_idx]);assert gap<=1e-4,(family,seed,name,method,i,gap)
                                differences.append({'family':family,'seed':seed,'candidate':name,'method':method,'query':i,'saved':row['observed'],'independent':float(pred[i]),'score_gap':gap})
                    else:
                        assert np.max(np.abs(pred-[x['observed'] for x in rows]))<1e-4,(family,seed,name,method)
                        states=np.where(np.abs(pred-wanted)>1e-4,'fail','pass')
                    for budget in (12,24,60,120):
                        which=np.array([x['cumulative_cost']<=budget for x in rows]);ss=states[which]
                        status='reject' if 'fail' in ss else ('indeterminate' if 'indeterminate' in ss else 'accept')
                        audit.append({'family':family,'seed':seed,'candidate':name,'method':method,'budget':budget,
                            'reference_consistent':r['labels'][name]['reference_consistent'],'status':status,
                            'robust_failed_queries':int(np.sum(ss=='fail')),'indeterminate_queries':int(np.sum(ss=='indeterminate'))})
            if seed==seeds[0]:
                triples=np.array(list(itertools.product(range(27),repeat=3)));ri,da,db=triples.T
                for name in r['labels']:
                    outputs=[]
                    for start in range(0,len(ri),729):
                        sl=slice(start,start+729);outputs.extend(suffix(edited(ri[sl],da[sl],db[sl],name)))
                    wanted=table[X[da,0],X[db,1]];fail=np.abs(np.array(outputs)-wanted)>1e-4
                    full_census.append({'family':family,'seed':seed,'candidate':name,'joint_triples':len(ri),'output_failures':int(fail.sum()),'note':'Argmax census may include near ties; use margin audit for scientific rejection claims.'})
            key=json.dumps(meta['meta']['table']);seen.setdefault(family+key,[]).append(seed)
            details.append({'family':family,'seed':seed,'independent_native_max_error':float(np.max(np.abs(out-want))),
                            'reference_labels_checked':6,'saved_assay_queries_reexecuted':len(records)})
    for metric in saved_analysis['metrics']:
        for v,label in [(False,'false_acceptance'),(True,'valid_acceptance')]:
            count=counts[(metric['budget'],metric['method'],v)]
            assert count==[metric[label]['accepted'],metric[label]['denominator']]
    assert records_count==saved_analysis['query_records']
    margin_summary=[]
    for method in ('ordinary','full_single','heldout_single','joint','faithfulness','raw_latent','transport','natural'):
        for budget in (12,24,60,120):
            rows=[x for x in audit if x['method']==method and x['budget']==budget]
            margin_summary.append({'method':method,'budget':budget,**{('valid_' if v else 'invalid_')+status:sum(x['reference_consistent']==v and x['status']==status for x in rows) for v in (False,True) for status in ('accept','reject','indeterminate')}})
    result={'all_pass':True,'verification_scope':'Score-gap-aware independent replay; exact argmax equality failed in near ties, preserved separately.',
            'frozen_source_hashes_match':True,'programs':len(details),'query_records_reaggregated':records_count,
            'saved_assay_predictions_reexecuted':records_count,'program_details':details,'exhaustive_joint_census':full_census,
            'categorical_argmax_disagreements':len(differences),'largest_tied_score_gap':max((x['score_gap'] for x in differences),default=0),
            'margin_audit_summary':margin_summary,
            'duplicate_final_tables':[v for v in seen.values() if len(v)>1],
            'independence_limit':'Separate equations and source; same NumPy, saved weights, semantic table and AI author. No external review.',
            'profile':profile()}
    (ROOT/'04-Results/v2/independent-verification.json').write_text(json.dumps(result,indent=2)+'\n')
    (ROOT/'04-Results/v2/margin-audit.json').write_text(json.dumps({'decisions':audit,'argmax_disagreements':differences},indent=2)+'\n')
    print({k:v for k,v in result.items() if k not in ('program_details','exhaustive_joint_census')})

if __name__=='__main__':verify()
