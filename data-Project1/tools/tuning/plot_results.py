"""Draw static scientific figures from frozen metrics; this script does not train."""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=Path(__file__).resolve().parents[3]
d=json.loads((BASE/'experiments/runs/20261008_tuning/manifest/checks/plot_data.json').read_text(encoding='utf-8'))
OUT=BASE/'experiments/reports/20261008_tuning/figures';OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':180})

fig,axes=plt.subplots(1,3,figsize=(8.2,2.85),gridspec_kw={'width_ratios':[1.65,1.1,.8]})
cs=sorted({r['params']['C'] for r in d['A']});weights=sorted({r['params']['char_weight'] for r in d['A']})
a=np.array([[100*next(r['acc'] for r in d['A'] if r['params']['C']==c and r['params']['char_weight']==w) for c in cs] for w in weights])
caps=sorted({(r['params']['word_features'],r['params']['char_features']) for r in d['B']});bcs=sorted({r['params']['C'] for r in d['B']})
b=np.array([[100*max(r['acc'] for r in d['B'] if (r['params']['word_features'],r['params']['char_features'])==cap and r['params']['C']==c) for c in bcs] for cap in caps])
c=np.array([[100*max(r['acc'] for r in d['C'] if r['params']['min_df']==df and r['params']['stop_words']==stop) for stop in ['english',None]] for df in [1,2,3]])
arrays=[a,b,c]
lo=min(x.min() for x in arrays);hi=max(x.max() for x in arrays)
for ax,z,title,xt,yt in zip(axes,arrays,['A: C and character weight','B: feature capacities',f'C: min_df (C={d["C"][0]["params"]["C"]:g})'],[[f'{v:g}' for v in cs],[f'{v:g}' for v in bcs],['English','None']],[[f'{v:g}' for v in weights],[f'W{w//1000}k/Ch{ch//1000}k' for w,ch in caps],['1','2','3']]):
    ax.imshow(z,cmap='YlGnBu',vmin=lo,vmax=hi,aspect='auto')
    ax.set_xticks(range(len(xt)),xt,fontsize=8.5);ax.set_yticks(range(len(yt)),yt,fontsize=8.5)
    ax.set_title(title,fontsize=10)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):ax.text(j,i,f'{z[i,j]:.2f}',ha='center',va='center',fontsize=8.5,color='white' if z[i,j]>(lo+hi)/2 else '#172b3a')
axes[0].set_xlabel('C');axes[0].set_ylabel('Character weight')
axes[1].set_xlabel('C');axes[2].set_xlabel('Word stop words');axes[2].set_ylabel('min_df')
fig.suptitle('Fixed validation accuracy (%) | 1,474 samples',fontsize=11)
fig.tight_layout();fig.savefig(OUT/'tuning_grid.png',bbox_inches='tight');plt.close(fig)

colors=plt.get_cmap('tab10').colors
fig,axes=plt.subplots(1,2,figsize=(7.5,3.1),gridspec_kw={'width_ratios':[1.05,1]})
winner=max(d['cv'],key=lambda r:(r['acc'],r['f1']))
for i,r in enumerate(d['cv']):
    mean=100*r['acc'];sd=100*r['sd'];axes[0].errorbar(mean,i,xerr=sd,fmt='o',capsize=4,color=colors[i],markersize=5)
    axes[0].text(mean+sd+.07,i,f'{mean:.2f}',va='center',fontsize=8)
axes[0].set_yticks(range(len(d['cv'])),[r['label']+(' *' if r['key']==winner['key'] else '') for r in d['cv']],fontsize=9)
axes[0].invert_yaxis();axes[0].set_xlabel('Accuracy (%)');axes[0].set_title('Mean and fold sample SD',fontsize=10)
lo=min(100*(r['acc']-r['sd']) for r in d['cv']);hi=max(100*(r['acc']+r['sd']) for r in d['cv'])
axes[0].set_xlim(lo-.3,hi+.7);axes[0].grid(axis='x',alpha=.2)
keys=list(dict.fromkeys(['char_reference',winner['key'],'MLP']))
for key in keys:
    r=next(r for r in d['cv'] if r['key']==key);i=d['cv'].index(r)
    axes[1].plot(range(1,6),100*np.array(r['folds']),'-o',markersize=4,label=r['label'],color=colors[i])
axes[1].set_xticks(range(1,6));axes[1].set_xlabel('Same fold');axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title('Paired fold results',fontsize=10);axes[1].legend(fontsize=8);axes[1].grid(alpha=.2)
fig.suptitle('Frozen candidates: stratified 5-fold comparison',fontsize=11)
fig.tight_layout();fig.savefig(OUT/'cv_comparison.png',bbox_inches='tight');plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(7.5,2.65))
for i,path in enumerate(d['mlp_runs']):
    root=BASE/path;c=json.loads((root/'config.json').read_text(encoding='utf-8'))
    with (root/'history.csv').open(encoding='utf-8') as f:rows=list(csv.DictReader(f))
    epochs=[int(r['epoch']) for r in rows];label=f'seed {c["random_state"]}'
    axes[0].plot(epochs,[float(r['train_loss']) for r in rows],label=label,color=colors[i])
    axes[1].plot(epochs,[100*float(r['val_acc']) for r in rows],label=label,color=colors[i])
    axes[1].axvline(c['best_epoch'],color=colors[i],alpha=.22,linestyle='--')
axes[0].set_ylabel('Training cross-entropy');axes[1].set_ylabel('Validation accuracy (%)')
for ax in axes:ax.set_xlabel('Epoch');ax.legend(fontsize=8);ax.grid(alpha=.2)
fig.suptitle('Best MLP configuration: three initializations',fontsize=11)
fig.tight_layout();fig.savefig(OUT/'mlp_curves.png',bbox_inches='tight');plt.close(fig)
print('Rendered three figures from frozen experiment data:',OUT)
