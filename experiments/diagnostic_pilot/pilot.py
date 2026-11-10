"""Run matched export-on/export-off pairs and verify exact numerical equality."""
import argparse,json,os,subprocess,sys
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
PROJECT = Path(__file__).resolve().parent
required = [PROJECT / 'icde/config.py', PROJECT / 'icde/__init__.py', PROJECT / 'reference/run_digits5.py']
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise SystemExit('Incomplete project extraction. Extract the entire ZIP, not just the notebook or pilot.py. Missing: ' + ', '.join(missing))
sys.path.insert(0, str(PROJECT))
from icde.config import Config
METHODS=['ours','ours_no_graph','ours_no_reliability']
def equal(a,b,path='root'):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b),path
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b),path
    elif isinstance(a,dict):
        assert a.keys()==b.keys(),path
        for k in a:equal(a[k],b[k],path+'/'+str(k))
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):equal(x,y,path+'/'+str(i))
    else:assert a==b,path

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',default='data');p.add_argument('--output',default='outputs/diagnostic_pilot');p.add_argument('--device',default='cuda');p.add_argument('--smoke',action='store_true');p.add_argument('--resume',action='store_true');p.add_argument('--workers',type=int,default=0);p.add_argument('--batch-size',type=int,default=64);p.add_argument('--amp',action='store_true');args=p.parse_args()
    root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=True)
    if args.smoke and args.output=='outputs/diagnostic_pilot':raise ValueError('Use a separate --output for smoke tests')
    reports=[];partition=None
    for method in METHODS:
        outputs=[]
        for mode in ['off','on']:
            c=Config(dataset='digits5',seed=42,alpha=.1,method=method,data_root=str(Path(args.data_root).resolve()),output_root=str(root/mode),device=args.device)
            c.workers=args.workers;c.batch_size=args.batch_size;c.amp=args.amp
            if args.smoke:
                c.dataset='synthetic';c.backbone='tiny';c.pretrained=False;c.image_size=32;c.batch_size=32;c.rounds=5;c.warmup_rounds=3;c.head_only_rounds=0;c.clients_per_domain=2;c.threads=2
            c.validate();config=root/f'{method}_{mode}.json';config.write_text(json.dumps(asdict(c),indent=2))
            env=dict(os.environ,FEDGAPS_DIAGNOSTICS='1' if mode=='on' else '0',FEDGAPS_CAPTURE_PREDICTIONS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
            command=[sys.executable,'-m','icde','run','--config',str(config)]
            if args.resume:command.append('--resume')
            subprocess.run(command,env=env,cwd=PROJECT,check=True)
            out=Path(c.output_root)/c.dataset/f'alpha_{c.alpha:g}'/f'seed_{c.seed}'/f'{method}-{c.identity()}'
            current=json.loads((out/'partition.json').read_text())
            if partition is None:partition=current
            equal(partition,current,'partition');outputs.append(out)
        a,b=outputs
        for name in ['history.json','clustering.json']:
            if (a/name).exists():equal(json.loads((a/name).read_text()),json.loads((b/name).read_text()),name)
        ma,mb=[json.loads((x/'metrics.json').read_text()) for x in outputs]
        for k in ['predictive','clustering','selection','warmup_selection','communication','rounds_executed']:equal(ma[k],mb[k],k)
        sa,sb=[torch.load(x/'selected.pt',map_location='cpu',weights_only=False)['snapshot'] for x in outputs];equal(sa,sb,'selected weights')
        with np.load(a/'test_predictions.npz') as x,np.load(b/'test_predictions.npz') as y:
            assert x.files==y.files
            for k in x.files:equal(x[k],y[k],'test/'+k)
        for name in ['distance.npy','membership.npy']:equal(np.load(a/name),np.load(b/name),name)
        assert (b/'diagnostics/initial.npz').exists() and (b/'diagnostics/selected.npz').exists()
        reports.append(dict(method=method,exact_match=True,output_on=str(b)))
        (root/'verification.json').write_text(json.dumps(dict(dataset=c.dataset,partition_identical=True,checks=reports),indent=2))
    print('PASS: all three variants match exactly, including test probabilities.')
if __name__=='__main__':main()
