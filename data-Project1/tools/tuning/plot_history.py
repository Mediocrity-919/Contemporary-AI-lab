"""Plot saved first-stage parameter records; no model training is performed."""
from pathlib import Path
import csv
import json
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=Path(__file__).resolve().parents[3]
OUT=BASE/'experiments/reports/20261014_report/figures';OUT.mkdir(parents=True,exist_ok=True)
CHECKS=BASE/'experiments/runs/20261014_report_review'
CHECKS.mkdir(parents=True,exist_ok=True)
rows=[]
for name in ['summary.csv','summary_LR.csv','summary_SVM.csv','summary_MNB.csv','summary_MLP.csv']:
    with (BASE/'experiments/tables'/name).open(encoding='utf-8-sig') as f:rows.extend(csv.DictReader(f))

def value(label,key):
    m=re.search(r'(?:'+key+r'=|_'+key+r')([0-9]+(?:\.[0-9]*)?(?:e[+-]?[0-9]+)?)(?=,|_|$)',label)
    return float(m.group(1)) if m else None

series={'LR':{},'Linear SVM':{},'NB':{},'MLP (hidden=200, ReLU)':{}}
for r in rows:
    p=r['params'];model=r['model'];y=100*float(r['val_acc'])
    if model=='LR' and ('iter500' in p or 'iter=500' in p):key='LR';x=value(p,'C')
    elif model=='SVM' and ('kernel-linear' in p or 'kernel=linear' in p):key='Linear SVM';x=value(p,'C')
    elif model in ['MNB','MultinomialNB'] and 'fitpriorFalse' not in p:key='NB';x=value(p,'alpha')
    elif model=='MLP-torch' and value(p,'hidden')==200 and ('actrelu' in p or 'act=relu' in p) and '3seed' not in p:key='MLP (hidden=200, ReLU)';x=value(p,'wd')
    else:continue
    if x is not None and x not in series[key]:series[key][x]={'accuracy':y,'run':r['note']}
    # First-round values take precedence for duplicate configurations.

fig,axes=plt.subplots(2,2,figsize=(7.3,3.6))
for ax,(label,points),color in zip(axes.flat,series.items(),['#3974a5','#4b8b51','#b47531','#865ba5']):
    xs=sorted(points);ys=[points[x]['accuracy'] for x in xs]
    ax.semilogx(xs,ys,'o-',color=color,markersize=4,linewidth=1.2)
    ax.set_title(label,fontsize=10);ax.set_ylabel('Accuracy (%)',fontsize=9)
    ax.set_xlabel('C' if label in ['LR','Linear SVM'] else 'alpha' if label=='NB' else 'weight decay',fontsize=9)
    ax.tick_params(labelsize=8);ax.grid(alpha=.23)
    best=max(range(len(xs)),key=lambda i:ys[i]);ax.annotate(f'{ys[best]:.2f}',(xs[best],ys[best]),xytext=(3,-12) if label.startswith('MLP') else (3,5),textcoords='offset points',fontsize=8,color=color)
    lo,hi=min(ys),max(ys);ax.set_ylim(max(0,lo-max(1,(hi-lo)*.08)),min(100,hi+max(1,(hi-lo)*.18)))
    if label.startswith('MLP'):ax.set_ylim(0,100);ax.set_yticks([0,25,50,75,100])
fig.suptitle('Initial word-TF-IDF tuning | same 1,465-row validation',fontsize=11)
fig.tight_layout();fig.savefig(OUT/'initial_tuning.png',dpi=220,bbox_inches='tight');plt.close(fig)
(CHECKS/'historical_plot_sources.json').write_text(json.dumps(series,ensure_ascii=False,indent=2),encoding='utf-8')
print('Historical curves from saved records:',OUT/'initial_tuning.png')
