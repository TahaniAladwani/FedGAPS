import argparse
import copy
import fcntl
import json
from pathlib import Path
from icde.config import Config
from icde.utils import atomic_json
from study import DiscoveryRunner, MatchedRunner, AccountingRunner, audit_partition
from summarize_study import summarize


def jobs(partition):
    if partition=='dirichlet':
        return [(name,'fuzzy',2) for name in ('counts','reliability_counts','unaligned','aligned')]
    return [('aligned','fuzzy',k) for k in (1,2,3,4)]+[('aligned',g,2) for g in ('agglomerative','spectral','kmeans')]+[('counts','fuzzy',2)]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',choices=['domainnet10','digits5','office10','synthetic'],default='domainnet10')
    p.add_argument('--suite',choices=['dirichlet','planted','both'],default='both')
    p.add_argument('--data-root',type=Path,default=Path('data'))
    p.add_argument('--output',type=Path,default=Path('outputs/collaborator_study'))
    p.add_argument('--seeds',nargs='+',type=int,default=[42,123])
    p.add_argument('--device',default='cuda');p.add_argument('--workers',type=int,default=4)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--samples-per-client',type=int,default=500)
    p.add_argument('--planted-sampling',choices=['finite','quota'],default='finite')
    p.add_argument('--reuse-planted',type=Path,help='Previous FedGAPS_Planted_Groups output root; requires quota sampling')
    p.add_argument('--amp',action='store_true');p.add_argument('--resume',action='store_true')
    p.add_argument('--plan-only',action='store_true')
    a=p.parse_args()
    if len(set(a.seeds))!=len(a.seeds):p.error('Seeds must be unique')
    if a.reuse_planted and (a.planted_sampling!='quota' or a.suite=='dirichlet'):
        p.error('--reuse-planted requires --planted-sampling quota and a suite including planted')
    a.output=a.output.resolve();a.data_root=a.data_root.resolve()
    if a.reuse_planted:a.reuse_planted=a.reuse_planted.resolve()
    a.output.mkdir(parents=True,exist_ok=True)
    partitions=['dirichlet','planted'] if a.suite=='both' else [a.suite]
    if a.reuse_planted and a.suite=='both':partitions=['planted','dirichlet']
    planned=[]
    for partition in partitions:
        for seed in a.seeds:
            for discovery in (['aligned','unaligned'] if partition=='dirichlet' else ['aligned']):
                planned.append(dict(partition=partition,seed=seed,stage='discovery',variant=discovery))
            for name,g,k in jobs(partition):
                planned.append(dict(partition=partition,seed=seed,stage='downstream',variant=name,grouping=g,k=k))
            if partition=='planted':
                for name in ('fedavg','domain_balanced_fedavg'):
                    planned.append(dict(partition=partition,seed=seed,stage='control',variant=name,k=1))
    plan={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items() if k not in ('resume','plan_only')}
    plan['jobs']=planned;plan['job_count']=len(planned)
    plan['expected_new_training_jobs']=len(planned)-(10*len(a.seeds) if a.reuse_planted else 0)
    path=a.output/'plan.json'
    if path.exists() and json.loads(path.read_text())!=plan:raise ValueError('Plan differs. Use a fresh --output directory.')
    atomic_json(path,plan)
    print(f"{len(planned)} logical jobs: "+', '.join(f'{q}={sum(j["partition"]==q for j in planned)}' for q in partitions),flush=True)
    print(f"Expected new training jobs: {plan['expected_new_training_jobs']}",flush=True)
    if a.reuse_planted:print('Compatible legacy jobs will be imported, not retrained. All requested old jobs must pass validation.',flush=True)
    if a.plan_only:return
    with (a.output/'.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for partition in partitions:
            for seed in a.seeds:
                base=Config(dataset=a.dataset,data_root=str(a.data_root),seed=seed,partition_mode=partition,
                    planted_sampling=a.planted_sampling,device=a.device,workers=a.workers,amp=a.amp,
                    batch_size=a.batch_size,planted_samples=a.samples_per_client,early_stop_patience=51,
                    image_size=96 if a.dataset=='digits5' else 224,alpha=.1)
                if a.dataset=='synthetic':
                    base.backbone='tiny';base.pretrained=False;base.image_size=32;base.workers=0
                    base.rounds=4;base.warmup_rounds=2;base.head_only_rounds=0;base.threads=1
                caches={}
                for name in (['aligned','unaligned'] if partition=='dirichlet' else ['aligned']):
                    c=copy.deepcopy(base);c.method='ours_no_graph' if name=='unaligned' else 'ours'
                    c.output_root=str(a.output/'discovery'/partition/name)
                    runner=DiscoveryRunner(c,a.resume)
                    audit_dir=a.output/'audits'/partition;audit_dir.mkdir(parents=True,exist_ok=True)
                    audit_partition(runner.data,audit_dir,seed)
                    cache_path=a.output/'cache'/partition/f'{name}_seed{seed}.pt'
                    if a.reuse_planted and partition=='planted' and not cache_path.exists():
                        from reuse_legacy import import_discovery
                        import_discovery(runner,a.reuse_planted,cache_path)
                    caches[name]=runner.obtain(cache_path)
                for name,g,k in jobs(partition):
                    c=copy.deepcopy(base);c.comparison=name;c.grouping=g;c.clusters=k
                    c.output_root=str(a.output/'runs'/partition/f'{name}_{g}_k{k}')
                    runner=MatchedRunner(c,a.resume)
                    runner.start_from(caches['aligned'],caches.get('unaligned'))
                    if a.reuse_planted and partition=='planted' and name=='aligned' and not (runner.out/'metrics.json').exists():
                        from reuse_legacy import import_result
                        import_result(runner,a.reuse_planted,caches['aligned'])
                    else:runner.run()
                    summarize(a.output)
                if partition=='planted':
                    for balanced in (False,True):
                        c=copy.deepcopy(base);c.method='fedavg';c.clusters=1;c.domain_balanced=balanced
                        label='domain_balanced_fedavg' if balanced else 'fedavg'
                        c.output_root=str(a.output/'runs'/partition/label)
                        runner=AccountingRunner(c,a.resume)
                        if runner.data.manifest!=caches['aligned']['manifest']:raise ValueError('Control partition mismatch')
                        if a.reuse_planted and not (runner.out/'metrics.json').exists():
                            from reuse_legacy import import_result
                            import_result(runner,a.reuse_planted,caches['aligned'])
                        else:runner.run()
                        summarize(a.output)


if __name__=='__main__':main()
