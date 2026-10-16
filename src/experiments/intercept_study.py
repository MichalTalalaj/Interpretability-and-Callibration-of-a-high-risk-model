"""Explicit adaptive follow-up: separate intercept treatment from class weighting."""
from copy import deepcopy
from pathlib import Path
import argparse
import hashlib
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from .data import ROOT,load_protocol,development_data,write_json,digest,check_preservation,preservation_snapshot,runtime_environment
from .transforms import pipeline,CheckedLogisticRegression
from .recipes import Recipe
from .protocol import partitions,fit_recipe,summarize


def run(path,resume=False,output_override=None):
    cfg=load_protocol(path)
    if output_override is not None:cfg['output']=str(Path(output_override).resolve())
    output=Path(cfg['output'])
    if output.exists() and not resume: raise FileExistsError(output)
    output.mkdir(exist_ok=True)
    source=[Path(path).resolve(),*sorted((ROOT/'src').rglob('*.py'))]
    X,y,excluded,base=development_data(cfg)
    identity=dict(protocol=cfg,environment=runtime_environment(),code_signature=hashlib.sha256(''.join(digest(p) for p in source).encode()).hexdigest(),
                  data_sha256=digest(base['data']['path']))
    if (output/'identity.json').exists() and json.loads((output/'identity.json').read_text())!=identity:
        raise ValueError('Cannot resume changed follow-up')
    write_json(output/'identity.json',identity)
    write_json(output/'environment.json',identity['environment'])
    own_snapshot=output/'preserved_files.json'
    original=ROOT/'artifacts/portfolio_v1/preserved_files.json'
    if not own_snapshot.exists():
        write_json(own_snapshot,json.loads(original.read_text()) if original.exists() else preservation_snapshot(output))
    specifications=[]
    for name,clip,solver,penalty,intercept_scaling in [
        ('lr_lbfgs_standard',False,'lbfgs','l2',1),
        ('lr_lbfgs_clip',True,'lbfgs','l2',1),
        ('lr_l1_intercept',True,'liblinear','l1',100)]:
        estimator=pipeline(CheckedLogisticRegression(solver=solver,penalty=penalty,intercept_scaling=intercept_scaling,
            C=.03,max_iter=1000,tol=.001,random_state=42),clip=clip)
        grid={'classifier__C':[.003,.01,.03,.1] if penalty=='l1' else [.0001,.003,.03,.3]}
        specifications.append(Recipe(name,'glassbox','Adaptive intercept-treatment diagnostic; matched preprocessing/C grid.',estimator,grid))
    write_json(output/'experiment_plan.json',[dict(name=r.name,hypothesis=r.hypothesis,grid=r.grid) for r in specifications])
    records=[];predictions=[];start=time.perf_counter()
    for repeat,seed in enumerate(cfg['outer_seeds']):
        for fold,(train,assess) in enumerate(StratifiedKFold(n_splits=cfg['outer_folds'],shuffle=True,random_state=seed).split(X,y)):
            parts=partitions(X,y,cfg,repeat,fold,train,assess)
            assert not set(excluded)&set(np.concatenate(list(parts.values())))
            parent=output/f'repeat_{repeat}/fold_{fold}';parent.mkdir(parents=True,exist_ok=True)
            write_json(parent/'partitions.json',{k:dict(indices=v.tolist(),rows=len(v),events=int(y.loc[v].sum())) for k,v in parts.items()})
            current=[];frames={}
            for recipe in specifications:
                folder=parent/recipe.name
                if (folder/'results.json').exists():
                    values=json.loads((folder/'results.json').read_text());frame=pd.read_csv(folder/'assessment_predictions.csv',float_precision='round_trip')
                else:
                    print(f'intercept follow-up repeat {repeat+1} fold {fold+1}: {recipe.name}',flush=True)
                    values,frame=fit_recipe(recipe,X,y,parts,cfg,folder,seed+fold*100)
                for v in values:v.update(repeat=repeat,fold=fold)
                frame['repeat']=repeat;frame['fold']=fold;records.extend(values);current.extend(values);predictions.append(frame);frames[recipe.name]=frame
            best=sorted([v for v in current if v['mode']=='raw'],key=lambda v:(-v['inner_ap'],v['recipe']))[0]['recipe']
            for value in current:
                if value['recipe']==best:
                    selected=deepcopy(value);selected.update(recipe='selector_intercept',selected_recipe=best);records.append(selected)
            frame=frames[best].copy();frame['selected_recipe']=best;frame['recipe']='selector_intercept';predictions.append(frame)
    summarize(output,records,pd.concat(predictions,ignore_index=True),cfg)
    preserved=json.loads(own_snapshot.read_text())
    write_json(output/'completion.json',dict(complete=True,seconds=time.perf_counter()-start,
        preserved_historical_files=check_preservation(preserved),assessment='Adaptive development follow-up, not untouched final evaluation.',
        historical_test_re_evaluated=False))
    print('Intercept follow-up complete',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,default=ROOT/'config/portfolio_intercept.yaml')
    parser.add_argument('--resume',action='store_true');parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args();run(args.config,args.resume,args.output_dir)
