"""Read-only discovery exports with isolated random streams."""
import copy,os,random
from contextlib import contextmanager
from dataclasses import fields
import numpy as np
import torch
from .utils import atomic_json
@contextmanager
def isolated_rng():
    py,ns=random.getstate(),np.random.get_state()
    try:
        with torch.random.fork_rng():yield
    finally:random.setstate(py);np.random.set_state(ns)
def enabled():return os.environ.get('FEDGAPS_DIAGNOSTICS')=='1'
def export_initial(r):
    if not enabled():return
    from .prototypes import signatures
    with isolated_rng():
        model=copy.deepcopy(r.model);model.load_state_dict(r.state['servers'][0])
        export_stats(r,signatures(model,r.data,r.c,r.device),None,'initial',0)
def geometry(nodes,mask):
    norm=nodes/np.maximum(np.linalg.norm(nodes,axis=-1,keepdims=True),1e-12)
    same,different=[],[]
    for c in range(len(nodes)):
        ds=np.flatnonzero(mask[c]);same.extend(float(1-norm[c,d]@norm[c,e]) for j,d in enumerate(ds) for e in ds[j+1:])
    for d in range(mask.shape[1]):
        cs=np.flatnonzero(mask[:,d]);different.extend(float(1-norm[c,d]@norm[e,d]) for j,c in enumerate(cs) for e in cs[j+1:])
    a=float(np.mean(same)) if same else None;b=float(np.mean(different)) if different else None
    return dict(same_class_cross_domain=a,different_class_same_domain=b,separation_gap=b-a if a is not None and b is not None else None)
def export_stats(r,stats,bank,stage,round_id):
    if not enabled():return
    from .prototypes import graph_bank
    with isolated_rng():
        if bank is None:bank=graph_bank(stats,r.data,r.model,r.c)
        arrays={f.name:getattr(bank,f.name).detach().cpu().numpy().copy() for f in fields(bank)}
        n,c,f=len(stats),r.classes,r.model.feature_dim
        arrays['client_prototypes']=np.zeros((n,c,f),dtype=np.float32);arrays['client_mask']=np.zeros((n,c),dtype=bool)
        keys=['count','dispersion','raw_support','reliability','tempered_support']
        for k in keys:arrays['client_'+k]=np.zeros((n,c))
        for i,classes in enumerate(stats):
            for label,stat in classes.items():
                arrays['client_mask'][i,label]=True;arrays['client_prototypes'][i,label]=stat.prototype
                for k in keys:arrays['client_'+k][i,label]=getattr(stat,k)
        arrays['client_distance']=r.data.core.proposed_distance(stats,r.c.prototype_mix,r.c.overlap_penalty)
        root=r.out/'diagnostics';root.mkdir(exist_ok=True)
        np.savez_compressed(root/f'{stage}.npz',**arrays)
        atomic_json(root/f'{stage}.json',dict(stage=stage,round=round_id,method=r.c.method,classes=list(r.data.core.CLASS_NAMES),domains=list(r.data.core.DOMAIN_NAMES),clients=[{k:x[k] for k in ['id','domain','semantic_profile']} for x in r.data.clients],raw=geometry(arrays['raw_nodes'],arrays['mask']),smoothed=geometry(arrays['smooth_nodes'],arrays['mask']),note='No-graph affinities are diagnostic; smoothing is disabled. Selected exports use the restored warm-up checkpoint.'))
