"""Compare all stages using the intersection of observed class-domain nodes."""
import argparse
import csv
import json
from pathlib import Path
import sys
PROJECT = Path(__file__).resolve().parent
required = ['icde/__init__.py', 'icde/config.py', 'icde/engine.py', 'reference/run_digits5.py']
missing = [name for name in required if not (PROJECT / name).is_file()]
if missing:
    raise SystemExit(f'Incomplete project at {PROJECT}. Extract the entire ZIP into a new folder. Missing: {missing}')
sys.path.insert(0, str(PROJECT))
import numpy as np
from icde.diagnostics import geometry

MEASURES = ('same_class_cross_domain', 'different_class_same_domain')
STAGES = ('initial_raw', 'selected_raw', 'selected_smoothed')

def summarize(root):
    rows = []
    for run in json.loads((root/'plan.json').read_text()):
        folder = Path(run['output'])/'diagnostics'
        with np.load(folder/'initial.npz') as initial, np.load(folder/'selected.npz') as selected:
            mask = initial['mask'].astype(bool) & selected['mask'].astype(bool)
            arrays = (initial['raw_nodes'], selected['raw_nodes'], selected['smooth_nodes'])
            if not all(np.isfinite(a[mask]).all() for a in arrays):
                raise ValueError(f'Nonfinite prototypes in {folder}')
            metrics = [geometry(a, mask) for a in arrays]
        meta = json.loads((folder/'selected.json').read_text())
        for measure in MEASURES:
            row = dict(alpha=run['alpha'], seed=run['seed'], measure=measure,
                       selected_round=meta['round'], observed_nodes=int(mask.sum()),
                       same_class_pairs=int(sum(n*(n-1)//2 for n in mask.sum(axis=1))),
                       different_class_pairs=int(sum(n*(n-1)//2 for n in mask.sum(axis=0))))
            for stage, values in zip(STAGES, metrics):
                row[stage] = values[measure]
            if any(row[s] is None for s in STAGES):
                raise ValueError(f'No comparable pairs for {measure} in {folder}')
            row['training_change'] = row['selected_raw']-row['initial_raw']
            row['smoothing_change'] = row['selected_smoothed']-row['selected_raw']
            rows.append(row)
    with (root/'geometry_per_run.csv').open('w', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=[]
    for alpha in (0.1, 0.5):
        for measure in MEASURES:
            group=[r for r in rows if r['alpha']==alpha and r['measure']==measure]
            row=dict(alpha=alpha,measure=measure,n_seeds=len(group))
            for stage in STAGES:
                values=[r[stage] for r in group]
                row[stage+'_mean']=float(np.mean(values))
                row[stage+'_sd']=float(np.std(values,ddof=1))
            summary.append(row)
    (root/'geometry_summary.json').write_text(json.dumps(summary,indent=2))
    lines=[r'\begin{table}[t]',r'\centering',r'\caption{Prototype cosine distances across discovery stages (mean $\pm$ sample standard deviation over two seeds). Comparisons within each run use identical observed pairs.}',r'\label{tab:geometry}',r'\setlength{\tabcolsep}{3pt}',r'\scriptsize',r'\begin{tabular}{@{}clccc@{}}',r'\toprule',r'$\alpha$ & Pairs & Initial & Learned & Smoothed \\',r'\midrule']
    for r in summary:
        label='Same class' if r['measure']==MEASURES[0] else 'Different classes'
        cells=[f"{r[s+'_mean']:.3f} $\\pm$ {r[s+'_sd']:.3f}" for s in STAGES]
        lines.append(f"{r['alpha']} & {label} & "+' & '.join(cells)+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\par\smallskip\raggedright\scriptsize Initial: before discovery. Learned: selected checkpoint before smoothing. Smoothed: same checkpoint after smoothing. Same-class pairs span domains; different-class pairs share a domain.',r'\end{table}']
    (root/'geometry_table.tex').write_text('\n'.join(lines)+'\n')
    print('Saved geometry_per_run.csv, geometry_summary.json, geometry_table.tex in',root)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    summarize(Path(p.parse_args().output).resolve())
