"""Parameter-driven lab1 training. Outputs use the existing runs/tables schema."""
import argparse
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import re
import time
import warnings

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from threadpoolctl import threadpool_limits

BASE = Path(__file__).resolve().parents[1]
RUNS = BASE / 'experiments/runs/20261008_revision'
TABLES = BASE / 'experiments/tables/20261008_revision'
DATA = BASE / 'experiments/data_processed/v7_keepall'
COLS = ['time','model','params','train_acc','val_acc','val_macro_p','val_macro_r','val_macro_f1','val_weighted_f1','note']
SUMMARY = TABLES / 'summary_revision_v7.csv'
PREDICTIONS = BASE / 'experiments/predictions'
FEATURE_CACHE = {}
FEATURE_DEFAULTS = {'word_features':20000,'char_features':0,'char_weight':1.0,'min_df':2,'max_df':1.0,'stop_words':'english','word_ngram':(1,2),'char_ngram':(3,5),'sublinear_tf':True}

def configure_output(batch):
    """Keep a complete experiment batch separate from frozen older records."""
    global RUNS, TABLES, SUMMARY, PREDICTIONS
    if not re.fullmatch(r'[A-Za-z0-9_-]+',batch):
        raise ValueError('Batch must be a simple directory name')
    RUNS=BASE/'experiments/runs'/batch
    TABLES=BASE/'experiments/tables'/batch
    SUMMARY=TABLES/'summary_tuning.csv'
    PREDICTIONS=BASE/'experiments/predictions'/batch
    for path in [RUNS,TABLES,PREDICTIONS]:path.mkdir(parents=True,exist_ok=True)

def complete_params(model, params=None):
    defaults={'LR':{'C':100,'max_iter':500},'MNB':{'alpha':.08,'fit_prior':True},'SVM':{'C':.5},'MLP-torch':{'hidden':200,'wd':5e-5,'lr':.001,'batch_size':128,'activation':'relu','max_epochs':60,'patience':8,'dropout':0.0,'scheduler':'none'}}
    p={**FEATURE_DEFAULTS,**defaults[model],**(params or {})}
    p['word_ngram']=tuple(p['word_ngram']);p['char_ngram']=tuple(p['char_ngram'])
    if p['char_weight']<=0 or p['min_df']<1:raise ValueError('Invalid feature parameters')
    return p

def timestamp():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')

def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')

def row_hash(df, text_column):
    value = '\0'.join(f'{i}:{t}' for i,t in zip(df.original_row,df[text_column].astype(str)))
    return hashlib.sha256(value.encode('utf-8')).hexdigest()

def param_string(model, p, seed):
    core = {'LR': ['C','max_iter'], 'MNB':['alpha','fit_prior'], 'SVM':['C'], 'MLP-torch':['hidden','wd','lr','batch_size','activation']}[model]
    values = ','.join(f'{k}={p[k]}' for k in core if k in p)
    values+=f',word={p["word_features"]},char={p["char_features"]},cw={p["char_weight"]:g},df={p["min_df"]},stop={p["stop_words"]},wn={p["word_ngram"]},cn={p["char_ngram"]},tf={p["sublinear_tf"]}'
    if model=='MLP-torch':values+=f',dropout={p["dropout"]:g},scheduler={p["scheduler"]}'
    return values+f',seed={seed}'

def append_summary(name, cfg, met):
    row = {'time':name[:15], 'model':cfg['model'], 'params':cfg['params_label'], 'note':name}
    row.update({k:round(met[k],4) if k in met else np.nan for k in COLS[3:9]})
    pd.DataFrame([row],columns=COLS).to_csv(SUMMARY, mode='a', header=not SUMMARY.exists(), index=False)

def features(tr, va, p, column, seed):
    settings={k:p.get(k,v) for k,v in FEATURE_DEFAULTS.items()}
    key = (row_hash(tr,column),row_hash(va,column) if va is not None else '',json.dumps(settings,sort_keys=True))
    if key in FEATURE_CACHE:
        return FEATURE_CACHE[key]
    word = TfidfVectorizer(max_features=settings['word_features'],ngram_range=settings['word_ngram'],sublinear_tf=settings['sublinear_tf'],stop_words=settings['stop_words'],min_df=settings['min_df'],max_df=settings['max_df'])
    xt = word.fit_transform(tr[column].astype(str))
    xv = word.transform(va[column].astype(str)) if va is not None else None
    vectors = {'word':word}
    if p.get('char_features',0):
        char = TfidfVectorizer(analyzer='char_wb',ngram_range=settings['char_ngram'],max_features=p['char_features'],min_df=settings['min_df'],max_df=settings['max_df'],sublinear_tf=settings['sublinear_tf'],dtype=np.float32)
        a = char.fit_transform(tr[column].astype(str))
        b = char.transform(va[column].astype(str)) if va is not None else None
        xt = hstack([xt,a*settings['char_weight']]).tocsr()
        xv = hstack([xv,b*settings['char_weight']]).tocsr() if b is not None else None
        vectors['char'] = char
    result = xt, xv, vectors
    # Keep just the latest representations: CV must not retain every fold.
    if len(FEATURE_CACHE) >= 2:
        FEATURE_CACHE.clear()
    FEATURE_CACHE[key] = result
    return result

def scores(y, p):
    return {'val_acc':float(accuracy_score(y,p)), 'val_macro_p':float(precision_score(y,p,average='macro',zero_division=0)), 'val_macro_r':float(recall_score(y,p,average='macro',zero_division=0)), 'val_macro_f1':float(f1_score(y,p,average='macro',zero_division=0)), 'val_weighted_f1':float(f1_score(y,p,average='weighted',zero_division=0)), 'report':classification_report(y,p,output_dict=True,zero_division=0), 'confusion_matrix':confusion_matrix(y,p,labels=range(10)).tolist(), 'correct':int((np.asarray(y)==p).sum())}

def run_experiment(model, tr, va, params=None, seed=42, stage='grid', column='text_clean', fold=None, fixed_epochs=None, artifact=False):
    p = complete_params(model,params)
    ts = timestamp()
    label = param_string(model,p,seed)
    slug = re.sub(r'[^A-Za-z0-9_.-]+','_',label)
    digest=hashlib.sha256(label.encode()).hexdigest()[:8]
    name = f'{ts}_{model}_{stage[:24]}_{slug[:52]}_{digest}' + (f'_fold{fold}' if fold is not None else '')
    d = RUNS / name
    suffix=1
    while d.exists():
        suffix+=1; d=RUNS/f'{name}_n{suffix}'
    name=d.name; d.mkdir(parents=True)
    impl={'LR':'sklearn.linear_model.LogisticRegression','MNB':'sklearn.naive_bayes.MultinomialNB','SVM':'sklearn.svm.LinearSVC','MLP-torch':'torch.nn.Linear-ReLU/Tanh-Linear'}[model]
    cfg={'model':model,'implementation':impl,'data_version':'v7_keepall' if column=='text_clean' else 'v6_clean_fixed_raw_split','stage':stage,'fold':fold,'random_state':seed,'params':p,'params_label':label,'text_column':column,'train_rows':len(tr),'val_rows':len(va) if va is not None else 0,'train_sha256':row_hash(tr,column),'val_sha256':row_hash(va,column) if va is not None else None,'tfidf':{'max_features':p['word_features'],'ngram_range':[1,2],'sublinear_tf':True,'stop_words':'english','min_df':2,'norm':'l2','word_dtype':'float64','char_features':p['char_features'],'char_analyzer':'char_wb','char_ngram':[3,5]},'fixed_epochs':fixed_epochs,'selection':'accuracy primary; macro-F1 breaks exact accuracy ties','fit_boundary':'vectorizers fitted on this training split only'}
    cfg['tfidf'].update({'ngram_range':list(p['word_ngram']),'char_ngram':list(p['char_ngram']),'sublinear_tf':p['sublinear_tf'],'stop_words':p['stop_words'],'min_df':p['min_df'],'max_df':p['max_df'],'char_weight':p['char_weight'],'post_concat_normalization':False})
    save_json(d/'config.json',cfg)
    import sklearn, sys
    cfg['environment']={'python':sys.version,'executable':sys.executable,'sklearn':sklearn.__version__,'numpy':np.__version__,'pandas':pd.__version__}
    start=time.perf_counter(); xt,xv,vectors=features(tr,va,p,column,seed)
    ytr=tr.target.to_numpy().astype(np.int64)
    yva=va.target.to_numpy().astype(np.int64) if va is not None and 'target' in va else None
    xt_rows=xt.getnnz(axis=1)
    cfg['actual_features']=int(xt.shape[1]);cfg['zero_train_vectors']=int((xt_rows==0).sum())
    cfg['zero_val_vectors']=int((xv.getnnz(axis=1)==0).sum()) if xv is not None else None
    caught=[];history=[]
    if model!='MLP-torch':
        clf = {'LR':lambda:LogisticRegression(C=p['C'],max_iter=p['max_iter'],random_state=seed), 'MNB':lambda:MultinomialNB(alpha=p['alpha'],fit_prior=p['fit_prior']), 'SVM':lambda:LinearSVC(C=p['C'],max_iter=5000,tol=1e-4,random_state=seed)}[model]()
        cfg['classifier_params']=clf.get_params()
        with warnings.catch_warnings(record=True) as ws, threadpool_limits(limits=4):
            warnings.simplefilter('always');clf.fit(xt,ytr)
        caught=[str(w.message) for w in ws]
        pt=clf.predict(xt);pv=clf.predict(xv) if xv is not None else None
        if artifact:
            joblib.dump({'vectorizers':vectors,'classifier':clf,'text_column':column,'params':p},d/'model.joblib')
        niter=np.asarray(clf.n_iter_).tolist() if hasattr(clf,'n_iter_') else None
    else:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader,TensorDataset
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(4)
        device='cuda' if torch.cuda.is_available() else 'cpu'
        cfg['environment'].update({'torch':torch.__version__,'cuda':torch.version.cuda,'device':device,'gpu':torch.cuda.get_device_name(0) if device=='cuda' else None,'deterministic_algorithms':True,'CUBLAS_WORKSPACE_CONFIG':os.environ['CUBLAS_WORKSPACE_CONFIG']})
        class MLP(nn.Module):
            def __init__(self):
                super().__init__();self.fc1=nn.Linear(xt.shape[1],p['hidden']);self.act=nn.ReLU() if p['activation']=='relu' else nn.Tanh();self.fc2=nn.Linear(p['hidden'],10)
                self.drop=nn.Dropout(p['dropout']) if p['dropout'] else nn.Identity()
            def forward(self,x):
                return self.fc2(self.drop(self.act(self.fc1(x))))
        clf=MLP().to(device)
        xtrain=torch.from_numpy(xt.astype(np.float32).toarray());yv=torch.from_numpy(ytr)
        generator=torch.Generator().manual_seed(seed)
        dl=DataLoader(TensorDataset(xtrain,yv),batch_size=p['batch_size'],shuffle=True,generator=generator)
        val_tensor=torch.from_numpy(xv.astype(np.float32).toarray()).to(device) if xv is not None else None
        opt=torch.optim.Adam(clf.parameters(),lr=p['lr'],weight_decay=p['wd'])
        scheduler=torch.optim.lr_scheduler.StepLR(opt,step_size=5,gamma=.5) if p['scheduler']=='step' else None
        loss_fn=nn.CrossEntropyLoss();best=(-1.,-1.);best_state=None;best_epoch=0;stale=0
        def predict(tensor):
            clf.eval();pred=[]
            with torch.no_grad():
                for i in range(0,len(tensor),512):
                    pred.append(clf(tensor[i:i+512].to(device)).argmax(1).cpu().numpy())
            return np.concatenate(pred)
        limit=fixed_epochs if fixed_epochs is not None else p['max_epochs']
        for ep in range(1,limit+1):
            clf.train();total=0.
            for xb,yb in dl:
                xb,yb=xb.to(device),yb.to(device);opt.zero_grad();loss=loss_fn(clf(xb),yb);loss.backward();opt.step();total+=loss.item()*len(xb)
            h={'epoch':ep,'train_loss':total/len(tr),'lr':float(opt.param_groups[0]['lr'])}
            if yva is not None:
                pp=predict(val_tensor);acc=float(accuracy_score(yva,pp));f1=float(f1_score(yva,pp,average='macro'))
                h.update({'val_acc':acc,'val_macro_f1':f1})
                if fixed_epochs is None:
                    primary_improved=acc>best[0]
                    if (acc,f1)>best:
                        best=(acc,f1);best_state=copy.deepcopy(clf.state_dict());best_epoch=ep
                    stale=0 if primary_improved else stale+1
            history.append(h)
            if scheduler is not None:scheduler.step()
            if fixed_epochs is None and stale>=p['patience']:
                break
        if fixed_epochs is None and best_state is not None:
            clf.load_state_dict(best_state)
        else:
            best_epoch=ep
        pt=predict(xtrain);pv=predict(val_tensor) if val_tensor is not None else None
        cfg['parameter_count']=sum(w.numel() for w in clf.parameters())
        cfg['best_epoch']=best_epoch;cfg['stop_epoch']=ep
        pd.DataFrame(history).to_csv(d/'history.csv',index=False)
        if artifact:
            torch.save(clf.state_dict(),d/'best_weights.pth');joblib.dump(vectors,d/'vectorizers.joblib')
        niter=ep
    cfg['secs_including_features']=time.perf_counter()-start
    met={'params':label,'train_acc':float(accuracy_score(ytr,pt)),'secs':cfg['secs_including_features'],'n_iter':niter,'warnings':caught}
    if model=='MLP-torch':met.update({'best_epoch':cfg['best_epoch'],'stop_epoch':cfg['stop_epoch']})
    if yva is not None:
        met.update(scores(yva,pv))
        pd.DataFrame({'original_row':va.original_row.to_numpy(),'target':yva,'prediction':pv,'correct':yva==pv}).to_csv(d/'val_predictions.csv',index=False)
    elif pv is not None:
        if not np.array_equal(va.original_row.to_numpy(),np.arange(len(va))):
            raise ValueError('Test order changed')
        predfile=PREDICTIONS/f'{name}.csv';predfile.parent.mkdir(parents=True,exist_ok=True)
        pd.DataFrame({'id':va.original_row.to_numpy(),'target':pv}).to_csv(predfile,index=False)
        cfg['prediction_file']=str(predfile.relative_to(BASE));cfg['prediction_rows']=len(pv)
    save_json(d/'config.json',cfg);save_json(d/'metrics.json',met)
    pd.DataFrame({'original_row':tr.original_row.to_numpy()}).to_csv(d/'train_rows.csv',index=False)
    if yva is not None:append_summary(name,cfg,met)
    (d/'stdout.txt').write_text(f'{name}\n'+json.dumps({k:v for k,v in met.items() if k not in ['report','confusion_matrix']},indent=2),encoding='utf-8')
    print(name,{k:met[k] for k in ['train_acc','val_acc','val_macro_f1','best_epoch','secs'] if k in met},flush=True)
    return {'name':name,'directory':str(d),'config':cfg,'metrics':met}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=['LR','MNB','SVM','MLP-torch'],required=True)
    parser.add_argument('--batch',default=None)
    parser.add_argument('--data-dir',type=Path,default=DATA)
    parser.add_argument('--C',type=float,default=None);parser.add_argument('--alpha',type=float,default=None)
    parser.add_argument('--max-iter',type=int,default=None);parser.add_argument('--hidden',type=int,default=None)
    parser.add_argument('--wd',type=float,default=None);parser.add_argument('--lr',type=float,default=None)
    parser.add_argument('--batch-size',type=int,default=None);parser.add_argument('--word-features',type=int,default=20000)
    parser.add_argument('--char-features',type=int,default=0);parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--char-weight',type=float,default=1.)
    parser.add_argument('--min-df',type=int,default=2);parser.add_argument('--stop-words',choices=['english','none'],default='english')
    parser.add_argument('--word-ngram',type=int,nargs=2,default=[1,2]);parser.add_argument('--char-ngram',type=int,nargs=2,default=[3,5])
    parser.add_argument('--linear-tf',action='store_true');parser.add_argument('--dropout',type=float,default=0.)
    parser.add_argument('--scheduler',choices=['none','step'],default='none')
    parser.add_argument('--epochs',type=int,default=None);parser.add_argument('--full-train',action='store_true')
    parser.add_argument('--stage',default='manual');parser.add_argument('--artifact',action='store_true')
    args=parser.parse_args()
    if args.batch:configure_output(args.batch)
    p={k:v for k,v in {'C':args.C,'alpha':args.alpha,'max_iter':args.max_iter,'hidden':args.hidden,'wd':args.wd,'lr':args.lr,'batch_size':args.batch_size,'word_features':args.word_features,'char_features':args.char_features}.items() if v is not None}
    p.update({'char_weight':args.char_weight,'min_df':args.min_df,'stop_words':None if args.stop_words=='none' else args.stop_words,'word_ngram':tuple(args.word_ngram),'char_ngram':tuple(args.char_ngram),'sublinear_tf':not args.linear_tf})
    if args.model=='MLP-torch':p.update({'dropout':args.dropout,'scheduler':args.scheduler})
    tr=pd.read_csv(args.data_dir/'train.csv',keep_default_na=False);va=pd.read_csv(args.data_dir/'val.csv',keep_default_na=False)
    if args.full_train:
        tr=pd.concat([tr,va],ignore_index=True);va=pd.read_csv(args.data_dir/'test_clean.csv',keep_default_na=False)
        if args.model=='MLP-torch' and args.epochs is None:parser.error('Full MLP training requires a frozen --epochs budget')
    run_experiment(args.model,tr,va,p,args.seed,args.stage,fixed_epochs=args.epochs,artifact=args.artifact or args.full_train)

if __name__=='__main__':main()
