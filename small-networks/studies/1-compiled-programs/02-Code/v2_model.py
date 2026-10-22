"""Tracr program construction and an independent NumPy execution path.

The NumPy attention equations follow the pinned Apache-2.0 Tracr architecture;
all weights are supplied by the unmodified compiler. Native JAX execution is
used independently to qualify every program before interpretation assays.
"""
from v2_resources import ROOT, guard
import itertools
import json
from dataclasses import dataclass
import numpy as np
from tracr.rasp import rasp
from tracr.compiler import compiling

INPUTS=np.array(list(itertools.product(range(3),repeat=3)),dtype=int)

def build_program(family,seed):
    rng=np.random.default_rng(seed)
    rasp.RASPExpr._ids=itertools.count(1)
    sel_a=rasp.Select(rasp.indices,rasp.Map(lambda i:0,rasp.indices),rasp.Comparison.EQ)
    sel_b=rasp.Select(rasp.indices,rasp.Map(lambda i:1,rasp.indices),rasp.Comparison.EQ)
    if family=='lookup':
        attempts=0
        while True:
            table=rng.integers(0,5,size=(3,3));attempts+=1
            if len(set(map(tuple,table)))==3 and len(set(map(tuple,table.T)))==3:
                break
        a=rasp.Aggregate(sel_a,rasp.tokens).named('var_a')
        b=rasp.Aggregate(sel_b,rasp.tokens).named('var_b')
        answer=rasp.SequenceMap(lambda x,y:int(table[x,y]),a,b).named('answer')
        meta={'family':family,'seed':seed,'table':table.tolist(),'generator_draws':attempts,'a_values':[0,1,2],'b_values':[0,1,2]}
    elif family=='additive':
        av=rng.choice(np.arange(1,11),size=3,replace=False)
        bv=rng.choice(np.arange(1,11),size=3,replace=False)
        # The pinned compiler restricts numerical Aggregate to binary inputs.
        # Select categorical tokens first, then map their numerical values.
        ca=rasp.Aggregate(sel_a,rasp.tokens)
        cb=rasp.Aggregate(sel_b,rasp.tokens)
        a=rasp.numerical(rasp.Map(lambda x:float(av[x]),ca)).named('var_a')
        b=rasp.numerical(rasp.Map(lambda x:float(bv[x]),cb)).named('var_b')
        answer=rasp.numerical(rasp.LinearSequenceMap(a,b,1,1)).named('answer')
        table=av[:,None]+bv[None,:]
        meta={'family':family,'seed':seed,'table':table.tolist(),'generator_draws':1,'a_values':av.tolist(),'b_values':bv.tolist()}
    else: raise ValueError(family)
    compiled=compiling.compile_rasp_to_model(answer,vocab={0,1,2},max_seq_len=3,compiler_bos='BOS',mlp_exactness=100)
    return compiled,meta


@dataclass
class Network:
    meta:dict
    labels:list
    params:dict
    num_heads:int
    key_size:int
    num_layers:int
    token_map:dict
    output_values:list
    categorical:bool
    rotation:np.ndarray

    def __post_init__(self):
        self.a_idx=[i for i,s in enumerate(self.labels) if s.startswith('var_a_')]
        self.b_idx=[i for i,s in enumerate(self.labels) if s.startswith('var_b_')]
        self.n_idx=[self.labels.index('tokens:'+str(i)) for i in range(3)]
        self.pool=self.a_idx+self.b_idx+self.n_idx
        self.out_idx=[i for i,s in enumerate(self.labels) if s.startswith('answer_')]
        self.d=len(self.pool)
        assert self.rotation.shape==(self.d,self.d)
        self.site=2*self.num_layers-2 # immediately after the final attention
        self.w1=self.params[f'transformer/layer_{self.num_layers-1}/mlp/linear_1']['w']
        self.b1=self.params[f'transformer/layer_{self.num_layers-1}/mlp/linear_1']['b']
        self.w2=self.params[f'transformer/layer_{self.num_layers-1}/mlp/linear_2']['w']
        self.b2=self.params[f'transformer/layer_{self.num_layers-1}/mlp/linear_2']['b']

    @classmethod
    def from_compiler(cls,m,meta):
        assert not m.model_config.layer_norm and not m.model_config.causal
        params={k:{x:np.asarray(y,dtype=np.float64) for x,y in v.items()} for k,v in m.params.items()}
        pool=[s for s in m.residual_labels if s.startswith(('var_a_','var_b_'))]+[f'tokens:{i}' for i in range(3)]
        q,r=np.linalg.qr(np.random.default_rng(meta['seed']+773311).normal(size=(len(pool),len(pool))))
        q=q@np.diag(np.where(np.diag(r)>=0,1.,-1.))
        categorical=meta['family']=='lookup'
        values=list(m.output_encoder.decode(list(range(len(m.output_encoder.encoding_map))))) if categorical else []
        tm={str(k):v for k,v in m.input_encoder.encoding_map.items()}
        return cls(meta,m.residual_labels,params,m.model_config.num_heads,m.model_config.key_size,m.model_config.num_layers,tm,values,categorical,q)

    def native_numpy(self,inputs):
        inputs=np.asarray(inputs)
        if (inputs.ndim not in (1,2) or inputs.shape[-1]!=3
                or not np.issubdtype(inputs.dtype,np.integer)
                or np.any(inputs<0) or np.any(inputs>2)):
            raise ValueError('Legal inputs/donors must be integer triples in 0..2')
        inputs=inputs.reshape(-1,3)
        tokens=np.array([[self.token_map['BOS']]+[self.token_map[str(x)] for x in row] for row in inputs])
        h=self.params['token_embed']['embeddings'][tokens]+self.params['pos_embed']['embeddings'][None,:4,:]
        layers=[]
        for layer in range(self.num_layers):
            name=f'transformer/layer_{layer}'
            proj=[]
            for kind in ('query','key','value'):
                p=self.params[f'{name}/attn/{kind}']
                proj.append((h@p['w']+p['b']).reshape(len(h),4,self.num_heads,self.key_size))
            q,k,v=proj
            logits=np.einsum('bthd,bshd->bhts',q,k)/np.sqrt(self.key_size)
            logits-=logits.max(axis=-1,keepdims=True)
            weights=np.exp(logits);weights/=weights.sum(axis=-1,keepdims=True)
            att=np.einsum('bhts,bshd->bthd',weights,v).reshape(len(h),4,-1)
            p=self.params[f'{name}/attn/linear'];h=h+att@p['w']+p['b'];layers.append(h.copy())
            p1=self.params[f'{name}/mlp/linear_1'];p2=self.params[f'{name}/mlp/linear_2']
            h=h+np.maximum(h@p1['w']+p1['b'],0)@p2['w']+p2['b'];layers.append(h.copy())
        return self.decode(h),layers

    def decode(self,h):
        z=h[...,self.out_idx]
        return np.array(self.output_values)[np.argmax(z,axis=-1)] if self.categorical else z[...,0]

    def suffix(self,site):
        return self.decode(site+np.maximum(site@self.w1+self.b1,0)@self.w2+self.b2)

    def encode_site(self,raw_site):
        return np.asarray(raw_site)[...,self.pool]@self.rotation

    def replace_site(self,raw_site,z):
        patched=raw_site.copy();patched[...,self.pool]=z@self.rotation.T
        return patched

    def oracle(self,inputs):
        x=np.asarray(inputs).reshape(-1,3)
        t=np.asarray(self.meta['table'])
        return t[x[:,0],x[:,1]]

    def save(self,folder):
        folder.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(folder/'weights.npz',rotation=self.rotation,**{k+'|'+x:y for k,v in self.params.items() for x,y in v.items()})
        (folder/'model.json').write_text(json.dumps({'meta':self.meta,'labels':self.labels,'num_heads':self.num_heads,'key_size':self.key_size,'num_layers':self.num_layers,'token_map':self.token_map,'output_values':self.output_values,'categorical':self.categorical,'site':self.site},indent=2)+'\n')

    @classmethod
    def load(cls,folder):
        meta=json.loads((folder/'model.json').read_text());meta.pop('site')
        a=np.load(folder/'weights.npz');p={}
        for key in a.files:
            if key=='rotation':continue
            name,v=key.split('|');p.setdefault(name,{})[v]=a[key]
        return cls(params=p,rotation=a['rotation'],**meta)


def qualify(m,net):
    import jax.numpy as jnp
    encoded=np.array([m.input_encoder.encode(['BOS']+list(map(int,row))) for row in INPUTS])
    native=m.forward(m.params,jnp.asarray(encoded))
    native_layers=[np.asarray(h) for h in native.transformer_output.residuals]
    _,layers=net.native_numpy(INPUTS)
    delta=max(float(np.max(np.abs(x-y))) for x,y in zip(native_layers,layers))
    assert delta<1e-4,('native_vs_numpy',delta)
    official=np.asarray(native.unembedded_output)[:,-1]
    if net.categorical:official=np.array(net.output_values)[official.astype(int)]
    assert np.allclose(official,net.oracle(INPUTS),atol=1e-4,rtol=0),('native_oracle',official)
    outputs=net.suffix(layers[net.site][:,-1])
    assert np.allclose(outputs,net.oracle(INPUTS),atol=1e-4,rtol=0)
    raw=layers[net.site][:,-1];z=net.encode_site(raw)
    assert np.max(np.abs(raw-net.replace_site(raw,z)))<1e-10
    # Structural qualification: nuisance pool dimensions cannot affect final readout.
    assert np.max(np.abs(net.w1[net.n_idx]))==0
    assert np.max(np.abs(net.w2[:,net.a_idx+net.b_idx]))==0
    assert np.max(np.abs(raw[:,net.out_idx]))<1e-8
    assert np.isfinite(np.concatenate([h.ravel() for h in layers])).all()
    return {'native_numpy_max_abs':delta,'native_oracle_inputs':27,'all_pass':True,
            'site':net.site,'residual_width':len(net.labels),'pool_width':net.d,
            'parameters':sum(x.size for v in net.params.values() for x in v.values())},raw

if __name__=='__main__':
    guard(admission=True)
    import sys
    family=sys.argv[1] if len(sys.argv)>1 else 'lookup'
    rows=[]
    for seed in range(4):
        guard();m,meta=build_program(family,seed);net=Network.from_compiler(m,meta)
        gates,raw=qualify(m,net);net.save(ROOT/f'03-Data/v2/development/{family}-{seed}')
        rows.append({'family':family,'seed':seed,**gates});print(rows[-1],flush=True)
    from v2_resources import profile
    (ROOT/f'04-Results/v2/qualification-{family}.json').write_text(json.dumps({'rows':rows,'profile':profile()},indent=2)+'\n')
