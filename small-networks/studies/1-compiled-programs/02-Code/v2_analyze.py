"""Program-cluster summaries. No patch is treated as an independent sample."""
from v2_resources import ROOT,guard,profile
import json,csv,time
import numpy as np
from scipy.stats import beta

def analysis():
    guard(admission=True)
    programs=[json.loads(p.read_text()) for p in sorted((ROOT/'04-Results/v2/final').glob('*.json')) if not p.name.endswith('FAILED.json')]
    assert len(programs)==80
    methods=list(dict.fromkeys(s['method'] for s in programs[0]['summaries']));budgets=[12,24,60,120]
    rng=np.random.default_rng(881109);indices=[]
    for family in ('lookup','additive'):
        ix=np.array([i for i,p in enumerate(programs) if p['family']==family]);assert len(ix)==40
        indices.append(ix[rng.integers(40,size=(10000,40))])
    draws=np.concatenate(indices,axis=1)
    data=[];metrics=[]
    for i,p in enumerate(programs):
        for s in p['summaries']:
            data.append({'program':i,'family':p['family'],'seed':p['seed'],**s,
                         'valid':p['labels'][s['candidate']]['reference_consistent']})
    def stat(rows):
        num=np.zeros(80);den=np.zeros(80)
        for r in rows:num[r['program']]+=r['accepted'];den[r['program']]+=1
        rate=num.sum()/den.sum() if den.sum() else None
        if rate is None:return {'accepted':0,'denominator':0,'rate':None,'ci95':None}
        dist=num[draws].sum(axis=1)/den[draws].sum(axis=1)
        return {'accepted':int(num.sum()),'denominator':int(den.sum()),'rate':float(rate),
                'ci95':np.quantile(dist,[.025,.975]).tolist()}
    for budget in budgets:
        for method in methods:
            rows=[r for r in data if r['budget']==budget and r['method']==method]
            invalid=[r for r in rows if not r['valid']];valid=[r for r in rows if r['valid']]
            fa=stat(invalid);va=stat(valid)
            metrics.append({'budget':budget,'method':method,'false_acceptance':fa,'valid_acceptance':va,
                'false_rejections':len(valid)-va['accepted'],
                'native_calls':sum(r['native_calls'] for r in rows),'suffix_calls':sum(r['suffix_calls'] for r in rows),
                'total_seconds':sum(r['seconds'] for r in rows),
                'median_candidate_seconds':float(np.median([r['seconds'] for r in rows]))})
    numerators=[];den=np.zeros(80)
    for method in ['joint','full_single']:
        arr=np.zeros(80)
        for r in data:
            if r['budget']==120 and r['method']==method and not r['valid']:
                arr[r['program']]+=r['accepted']
                if method=='joint':den[r['program']]+=1
        numerators.append(arr)
    delta=numerators[0]-numerators[1];boot=delta[draws].sum(axis=1)/den[draws].sum(axis=1)
    primary={'contrast':'joint minus full_single false acceptance, budget 120',
             'difference':float(delta.sum()/den.sum()),'ci95':np.quantile(boot,[.025,.975]).tolist(),
             'invalid_denominator':int(den.sum()),'bootstrap_programs':80,'bootstrap_replicates':10000}
    ordinary_set={(r['program'],r['candidate']) for r in data if r['method']=='ordinary' and r['budget']==12 and not r['valid'] and r['accepted']}
    conditional={m:stat([r for r in data if r['budget']==120 and r['method']==m and (r['program'],r['candidate']) in ordinary_set]) for m in methods}
    breakdown=[]
    for family in ('lookup','additive'):
        for candidate in programs[0]['labels']:
            for method in methods:
                rows=[r for r in data if r['family']==family and r['candidate']==candidate and r['budget']==120 and r['method']==method]
                breakdown.append({'family':family,'candidate':candidate,'method':method,
                    'valid':sum(r['valid'] for r in rows),'total':len(rows),
                    'accepted':sum(r['accepted'] for r in rows),'invalid_accepted':sum(r['accepted'] and not r['valid'] for r in rows),
                    'valid_rejected':sum(not r['accepted'] and r['valid'] for r in rows)})
    zero_bounds={}
    for m in methods:
        rs=[r for r in data if r['method']==m and r['budget']==120]
        failed=set(r['program'] for r in rs if r['valid'] and not r['accepted'])
        zero_bounds[m]={'programs_with_any_valid_rejection':len(failed),'programs':80,
            'exact_binomial_ci95':[float(beta.ppf(.025,len(failed),81-len(failed))) if failed else 0.,
                                   float(beta.ppf(.975,len(failed)+1,80-len(failed))) if len(failed)<80 else 1.]}
    out={'programs':80,'families':{'lookup':40,'additive':40},'candidate_instances':480,
        'reference_valid':sum(x['reference_consistent'] for p in programs for x in p['labels'].values()),
        'reference_invalid':sum(not x['reference_consistent'] for p in programs for x in p['labels'].values()),
        'metrics':metrics,'primary':primary,'conditional_ordinary12_denominator':len(ordinary_set),
        'conditional':conditional,'breakdown':breakdown,'program_level_rejection_bounds':zero_bounds,
        'all_gates_pass':all(p['gates']['all_pass'] and p['controls']['all_instrumentation_gates_pass'] for p in programs),
        'max_native_numpy_difference':max(p['gates']['native_numpy_max_abs'] for p in programs),
        'duplicate_tables':{p['family']+'-'+str(p['seed']):p['duplicates'] for p in programs if p['duplicates']},
        'query_records':sum(p['query_records'] for p in programs),
        'total_per_program_seconds':sum(p['seconds'] for p in programs),
        'max_peak_rss_kib':max(p['profile']['peak_rss_kib'] for p in programs),
        'discovery_seconds':sum(p['discovery']['total_seconds'] for p in programs),
        'discovery_gradient_suffix_examples':sum(p['discovery']['das']['differentiable_suffix_examples'] for p in programs),
        'training_agreement_range':[min(p['discovery']['das']['training_output_agreement'] for p in programs),max(p['discovery']['das']['training_output_agreement'] for p in programs)],
        'profile':profile()}
    dest=ROOT/'04-Results/v2';(dest/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
    with (dest/'candidate-panel-results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    with (dest/'primary-table.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['method','invalid_accepted','invalid_total','false_acceptance','fa_lo','fa_hi','valid_accepted','valid_total','false_rejected','native_calls','suffix_calls','total_seconds'])
        for m in metrics:
            if m['budget']!=120:continue
            a,b=m['false_acceptance'],m['valid_acceptance']
            writer.writerow([m['method'],a['accepted'],a['denominator'],a['rate'],*a['ci95'],b['accepted'],b['denominator'],m['false_rejections'],m['native_calls'],m['suffix_calls'],m['total_seconds']])
    print(json.dumps({k:v for k,v in out.items() if k not in ('metrics','breakdown','conditional','program_level_rejection_bounds')},indent=2))

if __name__=='__main__':analysis()
