"""Render class-domain edges and prototype geometry from pilot exports."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='outputs/diagnostic_pilot');args=p.parse_args()
    for root in Path(args.output).glob('on/*/alpha_*/seed_*/*/diagnostics'):
        meta=json.loads((root/'selected.json').read_text());bank=np.load(root/'selected.npz')
        c=len(meta['classes']);cols=5;rows=(c+cols-1)//cols
        fig,axes=plt.subplots(rows,cols,figsize=(15,3*rows),squeeze=False)
        maximum=max(float(bank['edge_weights'].max()),1e-8)
        for i,ax in enumerate(axes.flat):
            if i>=c:ax.axis('off');continue
            im=ax.imshow(bank['edge_weights'][i],vmin=0,vmax=maximum,cmap='viridis')
            ax.set_title(str(meta['classes'][i]));ax.set_xticks(range(len(meta['domains'])),meta['domains'],rotation=60,ha='right',fontsize=7);ax.set_yticks(range(len(meta['domains'])),meta['domains'],fontsize=7)
        fig.suptitle(meta['method']+' | Selected discovery graph affinities')
        fig.tight_layout();fig.savefig(root/'class_domain_edges.pdf');plt.close(fig)
        records=[json.loads(f.read_text()) for f in sorted(root.glob('round_*.json'))]
        fig,ax=plt.subplots(figsize=(7,4))
        for kind,style in [('raw','--'),('smoothed','-')]:
            for key,color in [('same_class_cross_domain','tab:blue'),('different_class_same_domain','tab:orange')]:
                ax.plot([r['round'] for r in records],[r[kind][key] for r in records],style,color=color,label=kind+' '+key.replace('_',' '))
        ax.set(xlabel='Warm-up round',ylabel='Cosine distance');ax.legend(fontsize=7);fig.tight_layout();fig.savefig(root/'geometry_history.pdf');plt.close(fig)
        print(root)
if __name__=='__main__':main()
