import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from study import write_csv

CATEGORIES=('discovery_models','discovery_summaries','auxiliary_discovery_models',
            'auxiliary_discovery_summaries','selection_models','selection_summaries','downstream_models')
METRICS=('accuracy','worst_domain_accuracy','planted_ari','planted_nmi','domain_mi_nats',
         'domain_nmi','domain_ari','cross_domain_precision_at_k','realized_clusters',
         'active_models_per_client','seconds','distance_seconds','clustering_seconds',
         'upload_gb','download_gb','total_gb')+tuple(f'{c}_{direction}_gb' for c in CATEGORIES for direction in ('upload','download'))


def summarize(root):
    root=Path(root); rows=[]
    for path in sorted(root.glob('runs/**/metrics.json')):
        r=json.loads(path.read_text()); c=r['config']; info=r.get('comparison',{})
        label=info.get('variant','domain_balanced_fedavg' if c['domain_balanced'] else 'fedavg')
        row=dict(dataset=c['dataset'],partition=c['partition_mode'],variant=label,
                 grouping=c['grouping'] if c['method'].startswith('ours') else 'none',
                 k=c['clusters'],seed=c['seed'],partition_hash=r['partition_hash'])
        row.update({key:None for key in METRICS})
        row.update(accuracy=100*r['predictive']['accuracy'],worst_domain_accuracy=100*r['predictive']['worst_domain_accuracy'],seconds=r['seconds'])
        for key in METRICS:
            if key in (r.get('clustering') or {}):row[key]=r['clustering'][key]
        row['distance_seconds']=info.get('distance_seconds'); row['clustering_seconds']=info.get('clustering_seconds')
        comm=r['communication']
        for direction in ('upload','download','total'):row[direction+'_gb']=comm[direction+'_bytes']/1e9
        for name in CATEGORIES:
            for direction in ('upload','download'):
                row[f'{name}_{direction}_gb']=comm['components'].get(name,{}).get(direction+'_bytes',0)/1e9
        rows.append(row)
    write_csv(root/'per_run.csv',rows)
    communication=[{k:v for k,v in r.items() if k in ('dataset','partition','variant','grouping','k','seed') or k.endswith('_gb')} for r in rows]
    write_csv(root/'communication_per_run.csv',communication)
    grouped=defaultdict(list)
    for row in rows:
        grouped[(row['dataset'],row['partition'],row['variant'],row['grouping'],row['k'])].append(row)
    summary=[]
    for key,group in sorted(grouped.items()):
        if len({r['seed'] for r in group})!=len(group):raise ValueError('Duplicate results for one seed')
        row=dict(zip(('dataset','partition','variant','grouping','k'),key)); row['n_seeds']=len(group)
        for name in METRICS:
            values=[r[name] for r in group if r[name] is not None]
            row[name+'_mean']=float(np.mean(values)) if values else None
            row[name+'_sd']=float(np.std(values,ddof=1)) if len(values)>1 else None
        summary.append(row)
    write_csv(root/'summary.csv',summary)
    paired=[]
    for row in rows:
        if row['grouping']!='fuzzy' or row['k']!=2:continue
        reference='reliability_counts' if row['partition']=='dirichlet' else 'counts'
        matches=[r for r in rows if r['partition']==row['partition'] and r['seed']==row['seed'] and
                 r['dataset']==row['dataset'] and r['variant']==reference and r['k']==2 and r['grouping']=='fuzzy']
        if not matches:continue
        base=matches[0]
        if row['partition_hash']!=base['partition_hash']:raise ValueError('Paired partitions differ')
        pair={k:row[k] for k in ('dataset','partition','variant','seed')};pair['reference']=reference
        for key in ('accuracy','worst_domain_accuracy','planted_ari','planted_nmi','cross_domain_precision_at_k','total_gb'):
            pair[key+'_difference']=row[key]-base[key] if row[key] is not None and base[key] is not None else None
        paired.append(pair)
    write_csv(root/'paired_differences.csv',paired)
    for partition in ('dirichlet','planted'):
        lines=[r'\begin{table*}[t]',r'\centering\scriptsize',
               r'\caption{Matched collaborator-selection study: '+partition+r'. Mean $\pm$ sample standard deviation over completed seeds. Accuracy is in percent; domain MI is in nats.}',
               r'\begin{tabular}{llrrrrrrr}\toprule',
               r'Variant & Grouping ($K$) & Seeds & Acc. & Worst domain & ARI & NMI & Domain MI & P@3\\\midrule']
        for r in summary:
            if r['partition']!=partition:continue
            def fmt(key):
                mean,sd=r[key+'_mean'],r[key+'_sd']
                if mean is None:return '--'
                return f'{mean:.3f}' + (f' $\\pm$ {sd:.3f}' if sd is not None else '')
            variant=r['variant'].replace('_',r'\_')
            lines.append(f"{variant} & {r['grouping']} ({r['k']}) & {r['n_seeds']} & "+' & '.join(fmt(x) for x in
                ('accuracy','worst_domain_accuracy','planted_ari','planted_nmi','domain_mi_nats','cross_domain_precision_at_k'))+r' \\')
        lines += [r'\bottomrule\end{tabular}',r'\end{table*}']
        (root/f'{partition}_results.tex').write_text('\n'.join(lines)+'\n')
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    summarize(p.parse_args().output)
