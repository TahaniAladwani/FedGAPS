"""One-factor DomainNet-10 sweep with a shared default and matched partitions."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))
from icde.config import Config

def settings():
    return [(v, .5) for v in [.01, .05, .1, .2, .5]] + [(.1, v) for v in [.25, .75, 1.0]]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', default='data')
    p.add_argument('--output', default='outputs/domainnet10_sensitivity')
    p.add_argument('--device', default='cuda')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--amp', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--plan-only', action='store_true')
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    if a.smoke and a.output == 'outputs/domainnet10_sensitivity':
        p.error('Use a separate --output for synthetic smoke tests')
    root = Path(a.output).resolve(); root.mkdir(parents=True, exist_ok=True)
    plan = []
    for seed in [42, 123]:
        for strength, beta in settings():
            c = Config(dataset='domainnet10', method='ours', alpha=.1, seed=seed,
                       data_root=str(Path(a.data_root).resolve()), output_root=str(root),
                       device=a.device, workers=a.workers, batch_size=a.batch_size,
                       image_size=224, amp=a.amp, graph_lambda=strength, tempering_beta=beta)
            if a.smoke:
                c.dataset='synthetic'; c.backbone='tiny'; c.pretrained=False
                c.image_size=32; c.rounds=3; c.warmup_rounds=2; c.head_only_rounds=0
                c.clients_per_domain=2; c.threads=2
            c.validate()
            config=root/f'seed_{seed}_lambda_{strength:g}_beta_{beta:g}.json'
            output=root/c.dataset/'alpha_0.1'/f'seed_{seed}'/f'ours-{c.identity()}'
            plan.append(dict(seed=seed,graph_lambda=strength,tempering_beta=beta,
                             config=str(config),output=str(output),resolved=asdict(c)))
    old=root/'plan.json'
    if old.exists() and json.loads(old.read_text()) != plan:
        raise SystemExit('Settings differ from the saved plan. Choose a new --output directory.')
    old.write_text(json.dumps(plan,indent=2))
    for item in plan:
        Path(item['config']).write_text(json.dumps(item['resolved'],indent=2))
    print('16 runs: DomainNet-10, alpha=0.1, seeds 42/123; 8 configurations per seed.',flush=True)
    if a.plan_only:return
    env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8',FEDGAPS_DIAGNOSTICS='0',FEDGAPS_CAPTURE_PREDICTIONS='0')
    env['PYTHONPATH']=str(PROJECT)+os.pathsep+env.get('PYTHONPATH','')
    partitions={}
    for index,item in enumerate(plan,1):
        print(f"Run {index}/16: seed={item['seed']} lambda={item['graph_lambda']} beta={item['tempering_beta']}",flush=True)
        cmd=[sys.executable,'-m','icde','run','--config',item['config']]
        if a.resume:cmd.append('--resume')
        subprocess.run(cmd,cwd=PROJECT,env=env,check=True)
        partition=json.loads((Path(item['output'])/'partition.json').read_text())
        seed=item['seed']
        if seed in partitions and partition!=partitions[seed]:
            raise RuntimeError(f'Partition mismatch for seed {seed}')
        partitions[seed]=partition
        (root/'partition_check.json').write_text(json.dumps(dict(completed_runs=index,matched_within_seed=True),indent=2))
        subprocess.run([sys.executable,str(PROJECT/'summarize_sensitivity.py'),'--output',str(root)],env=env,check=True)

if __name__=='__main__':main()
