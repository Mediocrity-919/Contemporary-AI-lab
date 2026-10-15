"""Run the frozen revision plan; resume completed runs from its manifest."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from train_experiment import BASE, DATA, RUNS, TABLES, COLS, run_experiment, save_json, timestamp, append_summary
from prepare_data import extract_subject

SESSION=RUNS/'20261008_revision_manifest'
SESSION.mkdir(parents=True,exist_ok=True)
CHECKS=SESSION/'checks'
CHECKS.mkdir(parents=True,exist_ok=True)
MANIFEST=SESSION/'manifest.json'
state=json.loads(MANIFEST.read_text(encoding='utf-8')) if MANIFEST.exists() else {'session':'revision_v7','status':'running','records':{}}
save_json(SESSION/'config.json',{'model':'revision-manifest','plan':'experiments/reports/20261008_revision/revision_plan.md','data_version':'v7_keepall','seed':42,'conda_environment':'pytorch_env','summary_columns':COLS})
tr=pd.read_csv(DATA/'train.csv',keep_default_na=False)
va=pd.read_csv(DATA/'val.csv',keep_default_na=False)
full=pd.concat([tr,va],ignore_index=True)
assert len(full)==7368 and full.original_row.nunique()==7368
assert len(tr)==5894 and len(va)==1474

def run(key,model,params,train=tr,val=va,seed=42,stage='grid',column='text_clean',fold=None,fixed_epochs=None,artifact=False):
    if key in state['records']:
        d=RUNS/state['records'][key]
        if (d/'metrics.json').exists():
            return {'name':d.name,'directory':str(d),'config':json.loads((d/'config.json').read_text(encoding='utf-8')),'metrics':json.loads((d/'metrics.json').read_text(encoding='utf-8'))}
    result=run_experiment(model,train,val,params,seed,stage,column,fold,fixed_epochs,artifact)
    state['records'][key]=result['name'];save_json(MANIFEST,state)
    return result

def rank(r):
    return r['metrics']['val_acc'],r['metrics']['val_macro_f1']

def table(records,path):
    rows=[]
    for r in records:
        row={'time':r['name'][:15],'model':r['config']['model'],'params':r['config']['params_label'],'note':r['name']}
        row.update({k:round(r['metrics'].get(k,float('nan')),4) for k in COLS[3:9]})
        rows.append(row)
    pd.DataFrame(rows,columns=COLS).to_csv(TABLES/path,index=False)

def aggregate(key,records,stage,extra=None):
    if key in state['records']:
        d=RUNS/state['records'][key]
        return {'name':d.name,'directory':str(d),'config':json.loads((d/'config.json').read_text(encoding='utf-8')),'metrics':json.loads((d/'metrics.json').read_text(encoding='utf-8'))}
    cfg=dict(records[0]['config']);cfg['stage']=stage;cfg['source_runs']=[r['name'] for r in records]
    cfg['params_label']=cfg['params_label']+f',{stage},n={len(records)}'
    if extra:cfg.update(extra)
    met={k:float(np.mean([r['metrics'][k] for r in records])) for k in COLS[3:9]}
    met['std']={k:float(np.std([r['metrics'][k] for r in records],ddof=1)) for k in COLS[3:9]}
    met['individual_val_accs']=[r['metrics']['val_acc'] for r in records]
    met['individual_val_macro_f1s']=[r['metrics']['val_macro_f1'] for r in records]
    met['params']=cfg['params_label']
    name=f'{timestamp()}_{cfg["model"]}_{stage}_{key}'
    d=RUNS/name;d.mkdir(exist_ok=False);save_json(d/'config.json',cfg);save_json(d/'metrics.json',met)
    append_summary(name,cfg,met);state['records'][key]=name;save_json(MANIFEST,state)
    return {'name':name,'directory':str(d),'config':cfg,'metrics':met}

print('STAGE 1: keep a complete fixed validation split; isolate filtering',flush=True)
pre=[]
pre.append(run('filter_train_only','SVM',{'C':.5},train=tr.loc[~tr.legacy_remove],stage='filter_train_only',column='text_clean_v6'))
pre.append(run('v6_keepall','SVM',{'C':.5},stage='v6_keepall',column='text_clean_v6'))
base=run('v7_baseline_svm','SVM',{'C':.5},stage='v7_keepall');pre.append(base)
table(pre,'revision_preprocessing.csv')

print('STAGE 2: CPU models and feature grid',flush=True)
words=[];chars=[];lrs=[];nbs=[]
for w in [20000,50000,100000]:
    for c in [.3,.5,1.,2.]:
        r=base if w==20000 and c==.5 else run(f'svm_word{w}_C{c}','SVM',{'word_features':w,'C':c})
        words.append(r)
for c in [.3,.5,1.]:
    chars.append(run(f'svm_wordchar_C{c}','SVM',{'C':c,'word_features':20000,'char_features':50000}))
for c in [1.,10.,100.,300.]:
    lrs.append(run(f'lr_C{c}','LR',{'C':c}))
for a in [.03,.08,.1,.3,1.]:
    nbs.append(run(f'nb_alpha{a}','MNB',{'alpha':a}))
table(words+chars,'revision_features.csv');table(lrs,'revision_LR.csv');table(nbs,'revision_MNB.csv')

print('STAGE 3: MLP training parameters and local regularization',flush=True)
mlps=[]
for lr in [.0005,.001,.002]:
    for b in [128,256]:
        mlps.append(run(f'mlp_lr{lr}_b{b}_wd5e-5','MLP-torch',{'lr':lr,'batch_size':b}))
chosen=max(mlps,key=rank)
for wd in [3e-5,1e-4]:
    p={**chosen['config']['params'],'wd':wd}
    mlps.append(run(f'mlp_local_wd{wd}','MLP-torch',p))
table(mlps,'revision_MLP.csv')

print('STAGE 4: three seeds for the two best MLP configurations',flush=True)
mlp_aggregates=[]
for idx,r in enumerate(sorted(mlps,key=rank,reverse=True)[:2]):
    rs=[r]
    for seed in [7,2024]:
        rs.append(run(f'mlp_finalist{idx}_seed{seed}','MLP-torch',r['config']['params'],seed=seed,stage='multiseed',artifact=True))
    budget=int(np.median([z['metrics']['best_epoch'] for z in rs]))
    mlp_aggregates.append(aggregate(f'mlp_3seed_{idx}',rs,'3seed',{'seeds':[42,7,2024],'frozen_epochs':budget,'epoch_budget_rule':'median best epoch of these three tuning runs; frozen before CV/full training'}))
table(mlp_aggregates,'revision_MLP_seeds.csv')
mlp_best=max(mlp_aggregates,key=rank)

print('STAGE 5: frozen-candidate five-fold split sensitivity',flush=True)
word_best=max(words,key=rank);char_best=max(chars,key=rank)
candidates=[('word',word_best),('wordchar',char_best),('mlp',mlp_best)]
folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(full,full.target))
cv=[]
for label,r in candidates:
    rs=[]
    for fold,(ti,vi) in enumerate(folds,1):
        rs.append(run(f'cv_{label}_fold{fold}',r['config']['model'],r['config']['params'],train=full.iloc[ti],val=full.iloc[vi],stage='cv_stratified',fold=fold,fixed_epochs=mlp_best['config']['frozen_epochs'] if label=='mlp' else None))
    agg=aggregate(f'cv_{label}',rs,'cv5',{'candidate':label,'fold_seed':42,'evaluation_note':'non-nested post-selection sensitivity; MLP epoch budget frozen before outer-fold scoring'})
    cv.append(agg)
table(cv,'revision_cv_summary.csv')

print('STAGE 6: grouped-title diagnostic for frozen linear candidate',flush=True)
linear_best=max([z for z in cv if z['config']['model']=='SVM'],key=rank)
titles=full.text.map(extract_subject).str.lower().str.replace(r'\s+',' ',regex=True).str.strip()
groups=np.array([s if s else f'__empty_{i}' for i,s in enumerate(titles)])
rs=[]
for fold,(ti,vi) in enumerate(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(full,full.target,groups),1):
    assert not (set(groups[ti])&set(groups[vi]))
    rs.append(run(f'grouped_fold{fold}','SVM',linear_best['config']['params'],train=full.iloc[ti],val=full.iloc[vi],stage='cv_subjectgroup',fold=fold))
grouped=aggregate('cv_subjectgroup',rs,'groupcv5',{'grouping':'lowercase normalized Subject after removing Re prefix; proxy only','paired_random_candidate':linear_best['name']})
table([linear_best,grouped],'revision_group_diagnostic.csv')

selection=max(cv,key=rank)
frozen={'model':selection['config']['model'],'params':selection['config']['params'],'selected_cv_run':selection['name'],'mean_cv_acc':selection['metrics']['val_acc'],'std_cv_acc':selection['metrics']['std']['val_acc'],'mean_cv_macro_f1':selection['metrics']['val_macro_f1'],'epochs':mlp_best['config']['frozen_epochs'] if selection['config']['model']=='MLP-torch' else None,'selection_rule':'max mean accuracy over the frozen 5-fold candidates; macro-F1 breaks exact ties','limitations':'non-nested CV on data also used for candidate development; no independent test labels'}
save_json(CHECKS/'revision_selection.json',frozen)
main=[max(lrs,key=rank),max(nbs,key=rank),max(words+chars,key=rank),mlp_best]
table(main,'revision_best_models.csv')

print('STAGE 7: full-data prediction with the frozen selected configuration',flush=True)
test=pd.read_csv(DATA/'test_clean.csv',keep_default_na=False)
final=run('final_fulltrain',frozen['model'],frozen['params'],train=full,val=test,stage='final_fulltrain_v7',fixed_epochs=frozen['epochs'],artifact=True)
frozen['final_run']=final['name'];frozen['prediction_file']=final['config']['prediction_file'];save_json(CHECKS/'revision_selection.json',frozen)
state['status']='complete';state['selected']=frozen;save_json(MANIFEST,state)
save_json(SESSION/'metrics.json',{'completed_model_runs':len(state['records']),'selected':frozen,'data_config':json.loads((DATA/'data_config.json').read_text(encoding='utf-8'))})
print('COMPLETE',json.dumps(frozen,ensure_ascii=False,indent=2),flush=True)
