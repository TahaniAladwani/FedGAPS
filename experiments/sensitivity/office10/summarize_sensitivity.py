"""Export partial results; plot only complete two-seed comparisons."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def write_csv(path,rows):
    if not rows:return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.output);plan=json.loads((root/'plan.json').read_text());rows=[]
    for item in plan:
        file=Path(item['output'])/'metrics.json'
        if not file.exists():continue
        m=json.loads(file.read_text())
        row={k:item[k] for k in ['seed','graph_lambda','tempering_beta']}
        row.update({k:100*m['predictive'][k] for k in ['accuracy','balanced_accuracy','worst_domain_accuracy','worst_class_recall']})
        row['rounds_executed']=m['rounds_executed'];rows.append(row)
    write_csv(root/'sensitivity_per_run.csv',rows)
    print(f'Summary: {len(rows)}/16 runs available.',flush=True)
    if len(rows)!=16:return
    summary=[]
    for parameter,values,other,default in [('graph_lambda',[.01,.05,.1,.2,.5],'tempering_beta',.5),('tempering_beta',[.25,.5,.75,1.],'graph_lambda',.1)]:
        for value in values:
            group=[r for r in rows if r[parameter]==value and r[other]==default]
            assert {r['seed'] for r in group}=={42,123} and len(group)==2
            row=dict(parameter=parameter,value=value,n_seeds=2)
            for metric in ['accuracy','balanced_accuracy','worst_domain_accuracy','worst_class_recall']:
                v=[r[metric] for r in group];row[metric+'_mean']=float(np.mean(v));row[metric+'_sd']=float(np.std(v,ddof=1))
            summary.append(row)
    write_csv(root/'sensitivity_summary.csv',summary)
    plt.rcParams.update({'font.size':10,'font.weight':'bold','axes.labelweight':'bold','axes.titleweight':'bold','pdf.fonttype':42})
    fig,axs=plt.subplots(1,2,figsize=(8,3.5),layout='constrained')
    for ax,param,label,default in zip(axs,['graph_lambda','tempering_beta'],['Graph strength','Reliability tempering'],[.1,.5]):
        group=[r for r in summary if r['parameter']==param];x=[r['value'] for r in group]
        for metric,title,color in [('accuracy','Accuracy','#1565c0'),('worst_domain_accuracy','Worst-domain accuracy','#b71c1c')]:
            ax.errorbar(x,[r[metric+'_mean'] for r in group],yerr=[r[metric+'_sd'] for r in group],marker='o',capsize=3,color=color,label=title)
        ax.axvline(default,color='gray',ls=':',label='Default')
        if param=='graph_lambda':ax.set_xscale('log')
        ax.set_xticks(x,[f'{v:g}' for v in x]);ax.set_xlabel(label);ax.grid(alpha=.2)
    axs[0].set_ylabel('Accuracy (%)')
    handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=3,frameon=False)
    fig.suptitle('Office-10 | Dirichlet α = 0.1')
    for ext in ['png','pdf']:fig.savefig(root/f'sensitivity.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)

if __name__=='__main__':main()
