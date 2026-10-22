"""Static research figures and explicit error examples from saved records."""
from v2_resources import ROOT,guard
import json,gzip
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from v2_verify import load,forward,X

guard(admission=True);base=ROOT/'04-Results/v2';out=base/'figures';out.mkdir(exist_ok=True)
a=json.loads((base/'analysis.json').read_text());v=json.loads((base/'independent-verification.json').read_text())
names={'ordinary':'Restricted singles','full_single':'Full-domain singles','heldout_single':'Held-out singles',
       'joint':'Joint patches','faithfulness':'Complement resampling','raw_latent':'Raw latent distance',
       'transport':'Divergence transport','natural':'Natural execution'}
colors={'ordinary':'#bf6c27','full_single':'#165d81','heldout_single':'#4d96a8','joint':'#744aa4',
        'faithfulness':'#237b58','raw_latent':'#bd4267','transport':'#6f7b26','natural':'#8796a4'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
ms=[m for m in a['metrics'] if m['budget']==120];ys=np.arange(len(ms))
fig,axs=plt.subplots(1,2,figsize=(9,4.7),sharey=True,gridspec_kw={'wspace':.22})
for ax,key,title in zip(axs,['false_acceptance','valid_acceptance'],['Inconsistent claims accepted (lower is better)','Consistent claims accepted (higher is better)']):
    values=[m[key]['rate']*100 for m in ms]
    ax.barh(ys,values,color=[colors[m['method']] for m in ms],height=.65)
    for i,(m,value) in enumerate(zip(ms,values)):
        ax.text(min(value+1,97),i,str(m[key]['accepted'])+'/'+str(m[key]['denominator']),va='center',ha='right' if value>95 else 'left',color='white' if value>95 else '#213342',fontsize=9)
    ax.set_xlim(0,102);ax.set_xticks([0,25,50,75,100]);ax.set_xlabel('% accepted');ax.set_title(title,fontsize=10,pad=10);ax.grid(axis='x',alpha=.13);ax.set_axisbelow(True)
axs[0].set_yticks(ys,[names[m['method']] for m in ms]);axs[0].invert_yaxis()
fig.subplots_adjust(left=.23,right=.98,bottom=.16,top=.87)
fig.text(.23,.94,'Final comparison • 120 state-access calls per candidate',fontsize=13,weight='bold')
fig.text(.23,.035,'Frozen binary decisions. Margin audit confirms all primary rejections except one restricted-single case.',fontsize=8)
fig.savefig(out/'primary-comparison.png');fig.savefig(out/'primary-comparison.pdf');plt.close(fig)

fig,axs=plt.subplots(1,2,figsize=(9,4.6))
methods=['ordinary','full_single','heldout_single','joint','faithfulness','transport']
for method in methods:
    rows=[x for x in a['metrics'] if x['method']==method];x=[z['budget'] for z in rows];y=np.array([z['false_acceptance']['rate']*100 for z in rows])
    ci=np.array([z['false_acceptance']['ci95'] for z in rows])*100
    axs[0].plot(x,y,'o-',color=colors[method],label=names[method],lw=1.7,ms=4);axs[0].fill_between(x,ci[:,0],ci[:,1],color=colors[method],alpha=.12)
    audit=[z for z in v['margin_audit_summary'] if z['method']==method]
    lo=np.array([z['invalid_accept'] for z in audit])/160*100
    hi=np.array([z['invalid_accept']+z['invalid_indeterminate'] for z in audit])/160*100
    axs[1].plot(x,hi,'o-',color=colors[method],lw=1.7,ms=4);axs[1].fill_between(x,lo,hi,color=colors[method],alpha=.12)
for ax in axs:
    ax.set_xscale('log');ax.set_xticks([12,24,60,120],['12','24','60','120']);ax.set_ylim(-2,61)
    ax.set_xlabel('State-access budget');ax.set_ylabel('Inconsistent claims accepted (%)');ax.grid(alpha=.15)
axs[0].set_title('Frozen decisions + clustered 95% intervals',fontsize=10)
axs[1].set_title('Sensitivity: ambiguous cases may pass',fontsize=10)
fig.legend(*axs[0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False,fontsize=8.5)
fig.subplots_adjust(bottom=.23,left=.08,right=.98,top=.84,wspace=.3)
fig.suptitle('Coverage resolves the failures; a joint advantage was not found',fontsize=13,weight='bold',y=.98)
fig.savefig(out/'cost-curves.png');fig.savefig(out/'cost-curves.pdf');plt.close(fig)

# Derive a readable, robust single/joint counterexample, no handpicked program.
folder=ROOT/'03-Data/v2/final/lookup-10000';meta,w=load(folder);raw,_=forward(meta,w)
cs=dict(np.load(folder/'candidates.npz'));lab=meta['labels'];pool=[i for i,s in enumerate(lab) if s.startswith(('var_a_','var_b_'))]+[lab.index('tokens:'+str(i)) for i in range(3)]
q=w['rotation'];z=raw[:,pool]@q;pa,pb=cs['mean_restricted_a'],cs['mean_restricted_b'];oi=[i for i,s in enumerate(lab) if s.startswith('answer_')]
with gzip.open(base/'final/lookup-10000-queries.jsonl.gz','rt') as f:rows=[json.loads(x) for x in f]
examples=[]
for method in ['full_single','joint']:
    for row in rows:
        if row['candidate']!='mean_restricted' or row['method']!=method:continue
        ri,da,db=row['ri'],row['da'],row['db'];zz=z[ri].copy()
        if method=='joint':zz+=(z[da]-z[ri])@pa+(z[db]-z[ri])@pb
        else:zz+=(z[da]-z[ri])@(pa if row['v']==0 else pb)
        h=raw[ri].copy();h[pool]=zz@q.T;prefix=f"transformer/layer_{meta['num_layers']-1}/mlp/"
        scores=(h+np.maximum(h@w[prefix+'linear_1|w']+w[prefix+'linear_1|b'],0)@w[prefix+'linear_2|w']+w[prefix+'linear_2|b'])[oi]
        ei=meta['output_values'].index(int(row['expected']));gap=float(scores.max()-scores[ei])
        if gap>1e-4:
            examples.append({'method':method,'recipient':X[ri].tolist(),'donor_a':X[da].tolist(),'donor_b':X[db].tolist(),
                'target':row['target'],'predicted':row['observed'],'expected':row['expected'],
                'score_gap':gap,'raw_carriers_after':(zz@q.T)[:6].tolist(),'query':row['query'],'cost':row['cumulative_cost']});break
stats={'program':'lookup-10000','table':meta['meta']['table'],'examples':examples}
allp=[json.loads(p.read_text()) for p in (base/'final').glob('*.json')]
stats['matched_norm_controls']={family:{'failures':sum(t['failed'] for p in allp if p['family']==family for t in p['controls']['matched_norm_random_direction']),
                                      'total':1200} for family in ('lookup','additive')}
stats['irrelevant_donor_failures']={family:{name:sum(sum(p['controls']['candidates'][name]['irrelevant_donor_output_failures']) for p in allp if p['family']==family) for name in allp[0]['labels']} for family in ('lookup','additive')}
(base/'error-analysis.json').write_text(json.dumps(stats,indent=2)+'\n')
print(json.dumps(stats,indent=2))
