"""Four matched-protocol FedGAPS runs with read-only geometry exports."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
PROJECT = Path(__file__).resolve().parent
required = ['icde/__init__.py', 'icde/config.py', 'icde/engine.py', 'reference/run_digits5.py']
missing = [name for name in required if not (PROJECT / name).is_file()]
if missing:
    raise SystemExit(f'Incomplete project at {PROJECT}. Extract the entire ZIP into a new folder. Missing: {missing}')
sys.path.insert(0, str(PROJECT))
from icde.config import Config

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', default='data')
    p.add_argument('--output', default='outputs/geometry_4runs')
    p.add_argument('--device', default='cuda')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--amp', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--plan-only', action='store_true')
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    root = Path(args.output).resolve()
    if args.smoke and args.output == 'outputs/geometry_4runs':
        p.error('Use a separate output directory for synthetic smoke tests')
    root.mkdir(parents=True, exist_ok=True)
    plan = []
    for alpha in (0.1, 0.5):
        for seed in (42, 123):
            c = Config(dataset='office10', method='ours', alpha=alpha, seed=seed,
                       data_root=str(Path(args.data_root).resolve()), output_root=str(root),
                       device=args.device, workers=args.workers,
                       batch_size=args.batch_size, amp=args.amp, image_size=224)
            if args.smoke:
                c.dataset='synthetic'; c.backbone='tiny'; c.pretrained=False
                c.image_size=32; c.rounds=5; c.warmup_rounds=3
                c.head_only_rounds=0; c.clients_per_domain=2; c.threads=2
            c.validate()
            config = root / f'alpha_{alpha}_seed_{seed}.json'
            config.write_text(json.dumps(asdict(c), indent=2))
            out = root / c.dataset / f'alpha_{alpha:g}' / f'seed_{seed}' / f'ours-{c.identity()}'
            plan.append(dict(alpha=alpha, seed=seed, config=str(config), output=str(out)))
    (root/'plan.json').write_text(json.dumps(plan, indent=2))
    print('Four Office-10 FedGAPS runs: alpha 0.1 and 0.5, seeds 42 and 123.', flush=True)
    if args.plan_only:
        return
    env = dict(os.environ, FEDGAPS_DIAGNOSTICS='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    env['PYTHONPATH'] = str(PROJECT) + os.pathsep + env.get('PYTHONPATH', '')
    for item in plan:
        command = [sys.executable, '-m', 'icde', 'run', '--config', item['config']]
        if args.resume:
            command.append('--resume')
        subprocess.run(command, cwd=PROJECT, env=env, check=True)
    subprocess.run([sys.executable, str(PROJECT/'summarize_geometry.py'), '--output', str(root)], check=True)

if __name__ == '__main__':
    main()
