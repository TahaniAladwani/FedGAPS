import argparse
from pathlib import Path
import numpy as np
from icde.config import Config
from icde.data import load_bases, labels_of, prepare

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    c=Config(dataset='office10',data_root=str(a.data_root),output_root=str(a.output),
             partition_mode='planted',planted_sampling='finite')
    core,train,test=load_bases(c)
    for name,base in zip(core.DOMAIN_NAMES,train):
        counts=np.bincount(labels_of(base),minlength=len(core.CLASS_NAMES))
        print(name,dict(zip(core.CLASS_NAMES,map(int,counts))),flush=True)
    for seed in (42,123):
        c.seed=seed
        data=prepare(c)
        print('Seed',seed,'partition verified:',data.manifest_path)
        for info in data.manifest.get('allocation_diagnostics',[]):print(info)
