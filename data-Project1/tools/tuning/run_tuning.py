"""Complete the finite, staged lab1 tuning plan in the activated conda environment."""
from pathlib import Path
import copy
import hashlib
import json
import os
import shutil
import sys
import traceback
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, train_test_split

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import train_experiment as t
from prepare_data import clean_text_keepall, extract_subject

BATCH='20261008_tuning'
t.configure_output(BATCH)
BASE=t.BASE; SESSION=t.RUNS/'manifest'; SESSION.mkdir(parents=True,exist_ok=True)
CHECKS=SESSION/'checks';CHECKS.mkdir(exist_ok=True)
MANIFEST=SESSION/'manifest.json'

class Tee:
    def __init__(self, stream, log):self.stream=stream;self.log=log
    def write(self,s):self.stream.write(s);self.log.write(s);self.log.flush();return len(s)
    def flush(self):self.stream.flush();self.log.flush()

log=(SESSION/'progress.log').open('a',encoding='utf-8',buffering=1)
sys.stdout=Tee(sys.stdout,log);sys.stderr=Tee(sys.stderr,log)
tr=pd.read_csv(t.DATA/'train.csv',keep_default_na=False)
va=pd.read_csv(t.DATA/'val.csv',keep_default_na=False)
full=pd.concat([tr,va],ignore_index=True)
assert len(tr)==5894 and len(va)==1474 and len(full)==7368 and full.original_row.nunique()==7368
assert not set(tr.original_row)&set(va.original_row)
expected=json.loads((t.DATA/'data_config.json').read_text(encoding='utf-8'))
for p,h in expected['source_sha256'].items():assert hashlib.sha256((BASE/p).read_bytes()).hexdigest()==h
assert sys.executable.lower().replace('/','\\').endswith('anaconda_envs\\envs\\pytorch_env\\python.exe')

state=json.loads(MANIFEST.read_text(encoding='utf-8')) if MANIFEST.exists() else {'batch':BATCH,'status':'running','records':{},'stages':{},'reused_keys':[]}
fingerprint={'trainer_sha256':hashlib.sha256(Path(t.__file__).read_bytes()).hexdigest(),'plan_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'train_hash':t.row_hash(tr,'text_clean'),'val_hash':t.row_hash(va,'text_clean')}
if 'fingerprint' in state:assert state['fingerprint']==fingerprint,'Source/data changed: use a new batch.'
state['fingerprint']=fingerprint
index={}
summary_names=set(pd.read_csv(t.SUMMARY).note) if t.SUMMARY.exists() else set()

def store():
    state['updated_at']=t.timestamp();t.save_json(MANIFEST,state)

def params(model,p):
    z=t.complete_params(model,p)
    for k in ['C','alpha','wd','lr','char_weight','dropout','max_df']:
        if k in z:z[k]=float(z[k])
    return z

def signature(model,p,seed,train_hash,val_hash,epochs):
    return json.dumps([model,params(model,p),int(seed),train_hash,val_hash,epochs],sort_keys=True)

def load(path):
    d=BASE/path if not Path(path).is_absolute() else Path(path)
    return {'name':d.name,'directory':str(d),'path':str(d.relative_to(BASE)),'config':json.loads((d/'config.json').read_text(encoding='utf-8')),'metrics':json.loads((d/'metrics.json').read_text(encoding='utf-8'))}

for root in [BASE/'experiments/runs/20261008_revision',t.RUNS]:
    for p in root.glob('*/config.json'):
        if not (p.parent/'metrics.json').exists():continue
        r=load(str(p.parent.relative_to(BASE)));c=r['config']
        if c.get('model') not in ['LR','MNB','SVM','MLP-torch'] or 'source_runs' in c or c.get('text_column')!='text_clean':continue
        if 'val_acc' not in r['metrics']:continue
        index[signature(c['model'],c['params'],c['random_state'],c['train_sha256'],c['val_sha256'],c.get('fixed_epochs'))]=r

def rank(r):return r['metrics']['val_acc'],r['metrics']['val_macro_f1']
def unique(records):return list({r['path']:r for r in records}.values())

def run(key,model,p,train=tr,val=va,seed=42,stage='grid',fold=None,epochs=None,force=False,artifact=False):
    p=params(model,p)
    sig=signature(model,p,seed,t.row_hash(train,'text_clean'),t.row_hash(val,'text_clean'),epochs)
    state['current_key']=key;store()
    if key in state['records']:
        r=load(state['records'][key]);c=r['config']
        assert signature(c['model'],c['params'],c['random_state'],c['train_sha256'],c['val_sha256'],c.get('fixed_epochs'))==sig
        return r
    if not force and not artifact and sig in index:
        r=index[sig];state['reused_keys'].append(key)
        print('REUSE',key,r['name'],flush=True)
    else:
        r=t.run_experiment(model,train,val,p,seed,stage,fold=fold,fixed_epochs=epochs,artifact=artifact)
        r['path']=str(Path(r['directory']).relative_to(BASE));index[sig]=r
    if 'val_acc' in r['metrics'] and r['name'] not in summary_names:
        # Fresh runs are already appended by the training entry point.
        present=set(pd.read_csv(t.SUMMARY).note) if t.SUMMARY.exists() else set()
        if r['name'] not in present:t.append_summary(r['name'],r['config'],r['metrics'])
        summary_names.add(r['name'])
    state['records'][key]=r['path'];store()
    return r

def table(records,name):
    rows=[]
    for r in unique(records):
        c=r['config'];m=r['metrics'];p=params(c['model'],c['params'])
        label=t.param_string(c['model'],p,c['random_state'])
        if c.get('source_runs'):label+=f",{c['stage']},n={len(c['source_runs'])}"
        z={'time':r['name'][:15],'model':c['model'],'params':label,'note':r['name']}
        z.update({k:round(m.get(k,float('nan')),4) for k in t.COLS[3:9]});rows.append(z)
    pd.DataFrame(rows,columns=t.COLS).to_csv(t.TABLES/name,index=False)

def aggregate(key,records,stage,extra=None):
    if key in state['records']:return load(state['records'][key])
    c=copy.deepcopy(records[0]['config']);c['stage']=stage;c['fold']=None
    c['source_runs']=[r['name'] for r in records];c['source_run_paths']=[r['path'] for r in records]
    c['params']=params(c['model'],c['params']);c['params_label']=t.param_string(c['model'],c['params'],c['random_state'])+f',{stage},n={len(records)}'
    if extra:c.update(extra)
    m={k:float(np.mean([r['metrics'][k] for r in records])) for k in t.COLS[3:9]}
    m['std']={k:float(np.std([r['metrics'][k] for r in records],ddof=1)) for k in t.COLS[3:9]}
    m['individual_val_accs']=[r['metrics']['val_acc'] for r in records]
    m['individual_val_macro_f1s']=[r['metrics']['val_macro_f1'] for r in records]
    name=f'{t.timestamp()}_{c["model"]}_{stage}_{key}'
    d=t.RUNS/name;d.mkdir();t.save_json(d/'config.json',c);t.save_json(d/'metrics.json',m)
    r={'name':name,'directory':str(d),'path':str(d.relative_to(BASE)),'config':c,'metrics':m}
    t.append_summary(name,c,m);summary_names.add(name);state['records'][key]=r['path'];store();return r

def stage(name,records):
    state['stage']=name;state['stages'][name]=[r['path'] for r in unique(records)];store()
    print('STAGE COMPLETE',name,len(unique(records)),flush=True)

def seeds(key,r):
    rs=[run(f'{key}_seed{seed}','MLP-torch',r['config']['params'],seed=seed,stage='multiseed') for seed in [42,7,2024]]
    budget=int(np.median([x['metrics']['best_epoch'] for x in rs]))
    return aggregate(key,rs,'3seed',{'seeds':[42,7,2024],'frozen_epochs':budget})

def nearby(c):
    values=[.03,.1,.2,.3,.5,1.]
    i=min(range(len(values)),key=lambda j:abs(values[j]-c))
    return values[max(0,i-1):min(len(values),i+2)]

def main():
    store();print('ANACONDA ENVIRONMENT',sys.executable,'PID',os.getpid(),flush=True)
    old_manifest=json.loads((BASE/'experiments/runs/20261008_revision/20261008_revision_manifest/manifest.json').read_text(encoding='utf-8'))
    oldroot='experiments/runs/20261008_revision/'
    baseline=load(oldroot+old_manifest['records']['svm_wordchar_C0.3'])
    parity=run('baseline_parity','SVM',baseline['config']['params'],stage='parity',force=True)
    assert pd.read_csv(Path(parity['directory'])/'val_predictions.csv').equals(pd.read_csv(Path(baseline['directory'])/'val_predictions.csv'))
    state['baseline_parity_passed']=True;store()

    # A: C and relative feature weights.
    a=[]
    for weight in [.5,1.,2.]:
        for c in [.03,.1,.2,.3,.5,1.]:
            a.append(run(f'A_c{c}_w{weight}','SVM',{'C':c,'word_features':20000,'char_features':50000,'char_weight':weight},stage='A_C_weight'))
    stage('A',a);svms=list(a);best=max(a,key=rank);cstar=best['config']['params']['C']
    neighbors=nearby(cstar);other=next(x for x in neighbors if x!=cstar)
    b=[]
    for w in [20000,50000]:
        for ch in [30000,50000,100000]:
            for c in [cstar,other]:
                p={**best['config']['params'],'word_features':w,'char_features':ch,'C':c}
                b.append(run(f'B_w{w}_ch{ch}_c{c}','SVM',p,stage='B_capacity'))
    stage('B',b);svms+=b;best=max(svms,key=rank)
    cs=[]
    for df in [1,2,3]:
        for stop in ['english',None]:
            p={**best['config']['params'],'min_df':df,'stop_words':stop}
            cs.append(run(f'C_df{df}_stop{stop}','SVM',p,stage='C_tfidf'))
    for j,r in enumerate(sorted(cs,key=rank,reverse=True)[:2]):
        for c in nearby(r['config']['params']['C']):
            cs.append(run(f'C_local{j}_c{c}','SVM',{**r['config']['params'],'C':c},stage='C_local_C'))
    stage('C',cs);svms+=cs;best=max(svms,key=rank)
    ds=[]
    for key,change in [('word_unigram',{'word_ngram':(1,1)}),('char36',{'char_ngram':(3,6)}),('linear_tf',{'sublinear_tf':False})]:
        ds.append(run('D_'+key,'SVM',{**best['config']['params'],**change},stage='D_ablation'))
    stage('D',ds);svms+=ds;svms=unique(svms);table(svms,'summary_SVM.csv')

    # E: historically strong MLP region, with actual per-seed metrics.
    e=[];mlp_records=[]
    for hidden in [200,225,250]:
        for wd in [3e-5,4e-5,5e-5]:
            e.append(run(f'E_h{hidden}_wd{wd}','MLP-torch',{'hidden':hidden,'wd':wd,'lr':.001,'batch_size':256},stage='E_MLP_local'))
    mlp_records+=e;egs=[seeds(f'E_finalist{i}',r) for i,r in enumerate(sorted(e,key=rank,reverse=True)[:2])]
    mlp_records+=egs;stage('E',e+egs)
    f=[]
    for i,r in enumerate(sorted(egs,key=rank,reverse=True)):
        f.append(run(f'F_input50k_{i}','MLP-torch',{**r['config']['params'],'word_features':50000},stage='F_MLP_50k'))
    f50=seeds('F_50k_best',max(f,key=rank));mlp_records+=f+[f50]
    anchor=max(egs+[f50],key=rank);training=[]
    for dropout in [0.,.1,.2]:
        for scheduler in ['none','step']:
            p={**anchor['config']['params'],'dropout':dropout,'scheduler':scheduler}
            training.append(run(f'F_drop{dropout}_{scheduler}','MLP-torch',p,stage='F_MLP_training'))
    fgs=[seeds(f'F_finalist{i}',r) for i,r in enumerate(sorted(training,key=rank,reverse=True)[:2])]
    mlp_records+=training+fgs;stage('F',f+[f50]+training+fgs)
    mlp_best=max(egs+[f50]+fgs,key=rank);table(mlp_records,'summary_MLP.csv')

    # G/H: requested local baselines; also test LR on the selected representation.
    lrs=[run(f'G_c{c}','LR',{'C':c},stage='G_LR') for c in [1.,3.,10.,30.,100.,300.]]
    representation=max(svms,key=rank)['config']['params']
    for c in [1.,3.,10.,30.]:lrs.append(run(f'G_shared_c{c}','LR',{**representation,'C':c},stage='G_LR_features'))
    stage('G',lrs)
    nbs=[run(f'H_alpha{alpha}','MNB',{'alpha':alpha},stage='H_NB') for alpha in [.03,.08,.1,.2,.3,.5,.7,1.]]
    stage('H',nbs);table(lrs+nbs,'summary_baselines.csv')
    lr_best=max(lrs,key=rank);nb_best=max(nbs,key=rank)
    table([max(svms,key=rank),mlp_best,lr_best,nb_best],'best_models.csv')

    # Frozen candidates, identical outer folds; no outer-fold checkpoint selection.
    candidates=[('word_reference',load(oldroot+old_manifest['records']['svm_word100000_C1.0'])),('char_reference',baseline)]
    top=[];seen=set()
    for r in sorted(svms,key=rank,reverse=True):
        sig=json.dumps(params('SVM',r['config']['params']),sort_keys=True)
        if sig not in seen:top.append(r);seen.add(sig)
        if len(top)==2:break
    candidates += [(f'tuned_svm{i}',r) for i,r in enumerate(top)]
    candidates += [('MLP',mlp_best),('LR',lr_best),('MNB',nb_best)]
    cv=[];folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(full,full.target))
    for label,r in candidates:
        rs=[];budgets=[]
        for fold,(ti,vi) in enumerate(folds,1):
            training_df=full.iloc[ti];validation_df=full.iloc[vi];epochs=None
            if r['config']['model']=='MLP-torch':
                inner_tr,inner_va=train_test_split(training_df,test_size=.2,stratify=training_df.target,random_state=1000+fold)
                assert not set(inner_tr.original_row)&set(validation_df.original_row)
                inner=run(f'cv_MLP_inner{fold}','MLP-torch',r['config']['params'],train=inner_tr,val=inner_va,stage='cv_inner_epoch',fold=fold)
                epochs=inner['metrics']['best_epoch'];budgets.append(epochs)
            rs.append(run(f'cv_{label}_fold{fold}',r['config']['model'],r['config']['params'],train=training_df,val=validation_df,stage='cv_outer',fold=fold,epochs=epochs))
        cv.append(aggregate('cv_'+label,rs,'cv5',{'candidate':label,'fold_seed':42,'inner_selected_epochs':budgets,'final_epochs':int(np.median(budgets)) if budgets else None,'evaluation_note':'post-selection non-nested candidate comparison; MLP epochs selected inside each outer training fold, never on its scoring fold'}))
    stage('CV',cv);table(cv,'summary_cv.csv');winner=max(cv,key=rank)

    # A title-group diagnostic estimates a different target and cannot choose the winner.
    grouped=[]
    if winner['config']['model']!='MLP-torch':
        titles=full.text.map(extract_subject).str.lower().str.replace(r'\s+',' ',regex=True).str.strip()
        groups=np.array([v if v else f'__empty_{i}' for i,v in enumerate(titles)])
        for fold,(ti,vi) in enumerate(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(full,full.target,groups),1):
            assert not set(groups[ti])&set(groups[vi])
            grouped.append(run(f'group_{fold}',winner['config']['model'],winner['config']['params'],train=full.iloc[ti],val=full.iloc[vi],stage='group_diagnostic',fold=fold))
        group=aggregate('group_diagnostic',grouped,'groupcv5',{'paired_random_run':winner['path'],'target':'unseen normalized Subject; imperfect discussion-thread proxy'})
        state['group_diagnostic']=group['path']

    frozen={'model':winner['config']['model'],'params':winner['config']['params'],'cv_run':winner['path'],'cv_accuracy':winner['metrics']['val_acc'],'cv_macro_f1':winner['metrics']['val_macro_f1'],'cv_sd':winner['metrics']['std']['val_acc'],'epochs':winner['config'].get('final_epochs'),'selection_rule':'highest mean five-fold accuracy, then mean macro-F1; finite tested candidates','limitations':'candidate development and evaluation share this dataset; no independent labeled test'}
    state['selected']=frozen;store();t.save_json(SESSION/'best_parameters.json',frozen)

    raw_test=pd.read_csv(BASE/'data-Project1/test_data_unlabeled.csv',keep_default_na=False)
    test=pd.DataFrame({'original_row':np.arange(len(raw_test)),'text_clean':raw_test.text.map(lambda x:clean_text_keepall(str(x)))})
    saved_test=pd.read_csv(t.DATA/'test_clean.csv',keep_default_na=False)
    assert test.text_clean.equals(saved_test.text_clean),'Raw-test cleaning differs from recorded v7'
    final=run('final_fulltrain',frozen['model'],frozen['params'],train=full,val=test,stage='final_fulltrain',epochs=frozen['epochs'],artifact=True,force=True)
    pred=pd.read_csv(BASE/final['config']['prediction_file'])
    assert list(pred)==['id','target'] and len(pred)==2457 and np.array_equal(pred.id,np.arange(2457)) and pred.target.isin(range(10)).all()
    public=t.PREDICTIONS/'predictions.csv';shutil.copyfile(BASE/final['config']['prediction_file'],public)
    shutil.copyfile(public,BASE/'data-Project1/predictions.csv')
    frozen.update({'final_run':final['path'],'prediction_file':str(public.relative_to(BASE)),'submission_file':'data-Project1/predictions.csv'})
    state['selected']=frozen;state['status']='complete';state['current_key']=None;store()
    t.save_json(SESSION/'best_parameters.json',frozen)
    t.save_json(SESSION/'metrics.json',{'record_keys':len(state['records']),'unique_run_paths':len(set(state['records'].values())),'reused_keys':len(state['reused_keys']),'stages':{k:len(v) for k,v in state['stages'].items()},'selected':frozen})
    print('COMPLETE',json.dumps(frozen,ensure_ascii=False,indent=2),flush=True)

try:main()
except BaseException as error:
    state['status']='failed';state['error']=str(error);store();traceback.print_exc();raise
