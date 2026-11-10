import argparse
import csv
from dataclasses import asdict
import itertools
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from .config import Config, DATASETS, METHODS, BASELINES, ABLATIONS
from .utils import atomic_json


def load_config(path):
    raw = json.loads(Path(path).read_text()) if path else {}
    return Config(**raw)


def matrix(config_path, optional=False, ablations=False):
    raw = json.loads(Path(config_path).read_text())
    shared = raw.get('common', {})
    methods = raw.get('methods', list(BASELINES[:6])+['ours'])
    if optional:
        methods = list(dict.fromkeys(methods+['fpl','fedsoft']))
    if ablations:
        methods = list(dict.fromkeys(methods+list(ABLATIONS)))
    configs = []
    for dataset,alpha,seed,method in itertools.product(raw['datasets'],raw.get('alphas',[.1,.5]),raw.get('seeds',[42,123,2024]),methods):
        values = dict(shared)
        values.update(raw.get('dataset_overrides',{}).get(dataset,{}))
        values.update(raw.get('method_overrides',{}).get(method,{}))
        values.update(dataset=dataset,alpha=alpha,seed=seed,method=method)
        configs.append(Config(**values).validate())
    return configs


def summarize(root):
    groups = {}
    for path in sorted(Path(root).glob('*/*/seed_*/*/metrics.json')):
        result = json.loads(path.read_text())
        if result.get('status') != 'completed':
            continue
        config = dict(result['config'])
        seed = config.pop('seed')
        selection = result.get('selection',[])
        result['resources'] = dict(seconds=result['seconds'],rounds_executed=result['rounds_executed'])
        warmup = result.get('warmup_selection')
        if warmup:
            result['resources'].update(warmup_best_round=warmup['best_round'],warmup_stopped_round=warmup['stopped_round'])
        if selection:
            result['resources'].update(best_round_mean=float(np.mean([x['best_round'] for x in selection])),
                stopped_round_mean=float(np.mean([x['stopped_round'] for x in selection])))
        config.pop('output_root',None)
        # Never mix different hyperparameter settings, backbones, or runtime flags.
        key = json.dumps(config,sort_keys=True)
        entry = groups.setdefault(key,{'config':config,'runs':{}})
        if seed in entry['runs']:
            raise ValueError(f'Duplicate seed {seed} for one configuration; use a clean result root')
        entry['runs'][seed] = result
    rows = []
    for entry in groups.values():
        config,runs = entry['config'],entry['runs']
        for block in ('predictive','clustering','communication','resources'):
            keys = set().union(*(set(r.get(block) or {}) for r in runs.values()))
            for key in sorted(keys):
                values = [(seed,(r.get(block) or {}).get(key)) for seed,r in sorted(runs.items())]
                values = [(seed,value) for seed,value in values if isinstance(value,(int,float)) and not isinstance(value,bool)]
                if not values:
                    continue
                numbers = [v for _,v in values]
                rows.append(dict(dataset=config['dataset'],alpha=config['alpha'],method=config['method'],
                    metric=f'{block}.{key}',mean=float(np.mean(numbers)),
                    sample_std=float(np.std(numbers,ddof=1)) if len(numbers)>1 else None,
                    n_seeds=len(numbers),seeds=' '.join(str(s) for s,_ in values),
                    complete_requested_seeds=set(s for s,_ in values)=={42,123,2024},
                    settings=json.dumps(config,sort_keys=True)))
    root = Path(root)
    root.mkdir(parents=True,exist_ok=True)
    atomic_json(root/'summary.json',rows)
    if rows:
        with (root/'summary.csv').open('w',newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    write_report(root, rows)
    print(f'Summarized {len(groups)} configurations into {root / "summary.csv"}')
    return rows


def write_report(root, rows):
    lines = ['# Experiment summary', '',
        'Values are mean ± sample standard deviation over completed seeds. Accuracies are fractions.',
        'Synthetic runs are pipeline tests only. Missing seeds are explicitly marked.', '']
    for block in ('predictive', 'clustering', 'communication', 'resources'):
        lines.extend([f'## {block.title()}', '',
            '| Dataset | α | Method | Settings | Metric | Mean ± SD | Seeds | Complete |',
            '|---|---:|---|---|---|---:|---|---|'])
        for row in rows:
            if not row['metric'].startswith(block+'.'):
                continue
            deviation = 'n/a' if row['sample_std'] is None else f"{row['sample_std']:.5g}"
            short = hashlib.sha256(row['settings'].encode()).hexdigest()[:8]
            lines.append(f"| {row['dataset']} | {row['alpha']} | {row['method']} | {short} | "
                f"{row['metric'].split('.',1)[1]} | {row['mean']:.5g} ± {deviation} | "
                f"{row['seeds']} | {'yes' if row['complete_requested_seeds'] else 'NO'} |")
        lines.append('')
    pending = {}
    for path in root.glob('coverage_plan-*.json'):
        for item in json.loads(path.read_text()).get('pending_runs',[]):
            pending[(item['dataset'],item['seed'],item['alpha'],item['baseline']['id'])] = item
    if pending:
        lines.extend(['## Pending requested baseline', '',
            f'{len(pending)} requested runs are not implemented because the full algorithm or verified author code is missing.',
            'These are not completed experiments and have no fabricated metric values.', ''])
        for identifier in sorted({x['baseline']['id'] for x in pending.values()}):
            example = next(x['baseline'] for x in pending.values() if x['baseline']['id']==identifier)
            lines.append(f"- {example['title']} — DOI: {example['doi']} — {example['status']}")
        lines.append('')
    (root/'report.md').write_text('\n'.join(lines)+'\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description='ICDE reproducible federated experiment suite')
    commands = parser.add_subparsers(dest='command',required=True)
    single = commands.add_parser('run',help='Run one configuration')
    single.add_argument('--config')
    for name,typ in [('dataset',str),('method',str),('seed',int),('alpha',float),('device',str),('data_root',str),('output_root',str)]:
        single.add_argument('--'+name.replace('_','-'),type=typ)
    single.add_argument('--resume',action='store_true')
    preflight = commands.add_parser('preflight',help='Check data and write the shared partition without training')
    preflight.add_argument('--config',required=True)
    sweep = commands.add_parser('sweep',help='Plan or execute a reproducible experiment matrix')
    sweep.add_argument('--config',default='configs/paper.json')
    sweep.add_argument('--optional',action='store_true')
    sweep.add_argument('--ablations',action='store_true')
    sweep.add_argument('--execute',action='store_true',help='Without this, only write the plan')
    sweep.add_argument('--resume',action='store_true')
    sweep.add_argument('--device')
    sweep.add_argument('--data-root')
    sweep.add_argument('--output-root')
    sweep.add_argument('--shard-index',type=int,default=0)
    sweep.add_argument('--shard-count',type=int,default=1)
    sweep.add_argument('--keep-going',action='store_true')
    summary = commands.add_parser('summarize')
    summary.add_argument('--output-root',default='outputs')
    smoke = commands.add_parser('smoke',help='Offline end-to-end checks with synthetic data')
    smoke.add_argument('--output-root',default='outputs/smoke')
    smoke.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS))
    smoke.add_argument('--device',default='cpu')
    args = parser.parse_args(argv)
    if args.command == 'summarize':
        summarize(args.output_root)
        return
    if args.command in ('run','preflight'):
        config = load_config(args.config)
        for key in ('dataset','method','seed','alpha','device','data_root','output_root'):
            if getattr(args,key,None) is not None:
                setattr(config,key,getattr(args,key))
        config.validate()
        if args.command=='preflight':
            from .data import prepare
            data = prepare(config)
            print(f'{len(data.clients)} clients; {len(data.core.CLASS_NAMES)} classes; partition: {data.manifest_path}')
            return
        from .engine import run
        result = run(config,args.resume)
        print(json.dumps(result['predictive'],indent=2))
        return
    if args.command == 'smoke':
        configs = [Config(dataset='synthetic',method=m,output_root=args.output_root,backbone='tiny',
            pretrained=False,image_size=32,batch_size=32,rounds=4,warmup_rounds=2,head_only_rounds=0,
            early_stop_min_rounds=1,clients_per_domain=2,device=args.device,workers=0,threads=2) for m in args.methods]
    else:
        if not 0 <= args.shard_index < args.shard_count:
            parser.error('Require 0 <= shard-index < shard-count')
        configs = matrix(args.config,args.optional,args.ablations)
        for config in configs:
            for name in ('device','data_root','output_root'):
                if getattr(args,name) is not None:
                    setattr(config,name,getattr(args,name))
            config.validate()
        configs = configs[args.shard_index::args.shard_count]
    pending_runs = []
    if args.command == 'sweep':
        requested = json.loads(Path(args.config).read_text())
        for dataset,alpha,seed,baseline in itertools.product(requested['datasets'],requested.get('alphas',[.1,.5]),requested.get('seeds',[42,123,2024]),requested.get('pending_baselines',[])):
            pending_runs.append(dict(dataset=dataset,alpha=alpha,seed=seed,baseline=baseline))
        pending_runs = pending_runs[args.shard_index::args.shard_count]
    plans = []
    for config in configs:
        root = Path(config.output_root)/'plans'
        path = root/f'{config.dataset}-{config.method}-{config.identity()}.json'
        atomic_json(path,asdict(config))
        command = [sys.executable,'-m','icde','run','--config',str(path)]
        if getattr(args,'resume',False):
            command.append('--resume')
        plans.append(dict(config=str(path),command=command))
    if configs:
        atomic_json(Path(configs[0].output_root)/f'plan-{getattr(args,"shard_index",0)}.json',plans)
        atomic_json(Path(configs[0].output_root)/f'coverage_plan-{getattr(args,"shard_index",0)}.json',
            dict(runnable_runs=len(plans),pending_runs=pending_runs,total_requested_runs=len(plans)+len(pending_runs)))
    print(f'{len(plans)} runnable experiments planned; {len(pending_runs)} requested runs pending algorithm source',flush=True)
    if args.command=='sweep' and not args.execute:
        return
    failures = []
    for i,plan in enumerate(plans):
        print(f'[{i+1}/{len(plans)}] {plan["config"]}',flush=True)
        completed = subprocess.run(plan['command'],check=False)
        if completed.returncode:
            failures.append(plan)
            if not getattr(args,'keep_going',False):
                break
    for root in sorted(set(c.output_root for c in configs)):
        summarize(root)
    if failures:
        raise SystemExit(f'{len(failures)} experiments failed; inspect failure.json and terminal output')

if __name__=='__main__':
    main()
