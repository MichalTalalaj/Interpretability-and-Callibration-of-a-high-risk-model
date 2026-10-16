"""Repeated nested development CV with independent calibration and threshold sets."""
from copy import deepcopy
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
import warnings
import joblib
import numpy as np
import pandas as pd
from joblib import parallel_backend
from sklearn.base import clone
from sklearn.metrics import average_precision_score,roc_auc_score
from sklearn.model_selection import GridSearchCV,StratifiedKFold,train_test_split,permutation_test_score
from threadpoolctl import threadpool_limits
from src.evaluation import classification_metrics
from src.tuning import select_threshold
from .data import ROOT,digest,write_json,load_protocol,development_data,preservation_snapshot,check_preservation,audit,runtime_environment
from .recipes import recipes
from .calibration import MonotoneSigmoid


METRIC_CONFIG=dict(threshold=.5,psi_bins=10,psi_epsilon=1e-6,calibration_bins=8)


def partitions(X,y,protocol,repeat,fold,outer_train,outer_assessment):
    seed=protocol['outer_seeds'][repeat]+100*(fold+1)
    train=X.index.to_numpy()[outer_train]; assessment=X.index.to_numpy()[outer_assessment]
    fit,remaining=train_test_split(train,test_size=1-protocol['fit_fraction'],stratify=y.loc[train],random_state=seed)
    cal,threshold=train_test_split(remaining,test_size=protocol['threshold_fraction']/(1-protocol['fit_fraction']),
                                   stratify=y.loc[remaining],random_state=seed+1)
    sets={'fit':fit,'calibration':cal,'threshold':threshold,'assessment':assessment}
    for name,left in sets.items():
        if set(np.unique(y.loc[left]))!={0,1}: raise ValueError(f'Partition {name} lacks a class')
        for right_name,right in sets.items():
            if name!=right_name and set(left)&set(right): raise ValueError('Partition overlap')
    return sets


def evaluate(y,p,threshold):
    cfg={**METRIC_CONFIG,'threshold':float(threshold)}
    metrics=classification_metrics(y,p,cfg)
    tn,fp=metrics['confusion_matrix'][0]; fn,tp=metrics['confusion_matrix'][1]
    metrics.update(false_positives=fp,false_negatives=fn,true_positives=tp,true_negatives=tn,
                   flagged=tp+fp,flagged_fraction=(tp+fp)/len(y),f2=5*tp/(5*tp+4*fn+fp) if tp+fn else 0)
    return metrics


def fit_recipe(recipe,X,y,parts,protocol,folder,seed):
    folder.mkdir(parents=True,exist_ok=True)
    fit=parts['fit']; cal=parts['calibration']; th=parts['threshold']; assess=parts['assessment']
    inner=StratifiedKFold(n_splits=protocol['inner_folds'],shuffle=True,random_state=seed)
    cv=list(inner.split(X.loc[fit],y.loc[fit]))
    write_json(folder/'inner_splits.json',[dict(train=fit[a].tolist(),validation=fit[b].tolist()) for a,b in cv])
    search=GridSearchCV(recipe.estimator,recipe.grid,scoring={'ap':'average_precision','roc_auc':'roc_auc'},
        refit='ap',cv=cv,n_jobs=protocol['search_jobs'],error_score=np.nan,return_train_score=True)
    start=time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with threadpool_limits(limits=1),parallel_backend('threading',n_jobs=protocol['search_jobs']):
            search.fit(X.loc[fit],y.loc[fit])
    search_seconds=time.perf_counter()-start
    results=pd.DataFrame(search.cv_results_)
    results.to_csv(folder/'inner_cv.csv',index=False)
    warning_messages=sorted(set(str(w.message) for w in caught))
    if not np.isfinite(search.best_score_): raise RuntimeError(f'No converged candidate for {recipe.name}')
    base=search.best_estimator_
    cal_probs=base.predict_proba(X.loc[cal])[:,1]
    sigmoid=MonotoneSigmoid(protocol['sigmoid_C']).fit(cal_probs,y.loc[cal])
    raw_th=base.predict_proba(X.loc[th])[:,1]; raw_assess=base.predict_proba(X.loc[assess])[:,1]
    train_probabilities=base.predict_proba(X.loc[fit])[:,1]
    output=[]; predictions=[]
    for mode in ['raw','sigmoid']:
        t_probs=raw_th if mode=='raw' else sigmoid.predict(raw_th)
        a_probs=raw_assess if mode=='raw' else sigmoid.predict(raw_assess)
        selected=select_threshold(y.loc[th],t_probs,protocol['threshold_beta'])
        metric=evaluate(y.loc[assess],a_probs,selected['threshold'])
        output.append(dict(recipe=recipe.name,family=recipe.family,mode=mode,inner_ap=float(search.best_score_),
            inner_roc_auc=float(search.cv_results_['mean_test_roc_auc'][search.best_index_]),
            inner_train_ap=float(search.cv_results_['mean_train_ap'][search.best_index_]),
            fitted_train_ap=float(average_precision_score(y.loc[fit],train_probabilities)),
            fitted_train_roc_auc=float(roc_auc_score(y.loc[fit],train_probabilities)),
            search_seconds=search_seconds,best_params=search.best_params_,threshold_selection=selected,
            retained_raw_fields=len(base.named_steps['features'].selected_columns_),
            transformed_features=len(base.named_steps['preprocessor'].get_feature_names_out()),
            active_coefficients=int(np.count_nonzero(base.named_steps['classifier'].coef_)) if hasattr(base.named_steps['classifier'],'coef_') else None,
            calibration_slope=sigmoid.slope_,calibration_intercept=sigmoid.intercept_,
            failed_candidates=int(results.mean_test_ap.isna().sum()),warnings=warning_messages,metrics=metric))
        frame=pd.DataFrame({'row_index':assess,'actual':y.loc[assess].to_numpy(),
            'probability':a_probs,'raw_probability':raw_assess,
            'prediction':(a_probs>=selected['threshold']).astype(int),'threshold':selected['threshold'],
            'recipe':recipe.name,'family':recipe.family,'mode':mode})
        predictions.append(frame)
    joblib.dump(dict(pipeline=base,calibrator=sigmoid,partitions=parts,recipe=recipe.name,seed=seed),folder/'model.joblib')
    write_json(folder/'results.json',output)
    pd.concat(predictions,ignore_index=True).to_csv(folder/'assessment_predictions.csv',index=False)
    return output,pd.concat(predictions,ignore_index=True)


def summarize(output,records,predictions,protocol):
    rows=[]
    for record in records:
        values={k:v for k,v in record.items() if k not in ['metrics','warnings','threshold_selection','best_params']}
        values.update({k:v for k,v in record['metrics'].items() if not isinstance(v,(dict,list))})
        values['threshold_f2']=record['threshold_selection']['score']
        values['best_params']=json.dumps(record['best_params'],sort_keys=True)
        rows.append(values)
    folds=pd.DataFrame(rows); folds.to_csv(output/'fold_results.csv',index=False)
    predictions.to_csv(output/'development_predictions.csv',index=False)
    summary=folds.groupby(['recipe','family','mode']).agg(
        folds=('pr_auc','size'),ap_mean=('pr_auc','mean'),ap_sd=('pr_auc','std'),
        roc_auc_mean=('roc_auc','mean'),roc_auc_sd=('roc_auc','std'),brier_mean=('brier_score','mean'),
        log_loss_mean=('log_loss','mean'),recall_mean=('recall','mean'),precision_mean=('precision','mean'),
        false_positives_mean=('false_positives','mean'),flagged_fraction_mean=('flagged_fraction','mean'),
        f2_mean=('f2','mean'),inner_ap_mean=('inner_ap','mean'),inner_train_ap_mean=('inner_train_ap','mean'),
        fitted_train_ap_mean=('fitted_train_ap','mean'),raw_fields_mean=('retained_raw_fields','mean'),
        transformed_features_mean=('transformed_features','mean'),active_coefficients_mean=('active_coefficients','mean'),
        search_seconds_total=('search_seconds','sum'),failed_candidates=('failed_candidates','sum')).reset_index()
    summary.sort_values(['mode','ap_mean'],ascending=[True,False]).to_csv(output/'comparison.csv',index=False)
    repeat_rows=[]
    for (recipe,mode,repeat),frame in predictions.groupby(['recipe','mode','repeat']):
        assert not frame.row_index.duplicated().any()
        repeat_rows.append(dict(recipe=recipe,mode=mode,repeat=int(repeat),rows=len(frame),events=int(frame.actual.sum()),
            pooled_ap=average_precision_score(frame.actual,frame.probability),
            pooled_roc_auc=roc_auc_score(frame.actual,frame.probability)))
    pd.DataFrame(repeat_rows).to_csv(output/'repeat_results.csv',index=False)
    # Compare whole-recipe paired fold differences. These folds overlap in training; SD is descriptive.
    contrasts=[('lr_weighted','lr_standard'),('lr_clip','lr_standard'),('lr_robust','lr_clip'),
        ('lr_rare','lr_clip'),('lr_no_indicators','lr_clip'),('lr_drop_sparse','lr_clip'),
        ('lr_redundancy','lr_clip'),('lr_grouped','lr_clip'),('lr_l1','lr_clip'),
        ('xgb_search','xgb_legacy'),('xgb_search','xgb_historical_selected'),
        ('xgb_weighted','xgb_search'),('xgb_rare','xgb_search')]
    paired=[]
    for challenger,reference in contrasts:
        for mode in ['raw','sigmoid']:
            left=folds[(folds.recipe==challenger)&(folds['mode']==mode)].set_index(['repeat','fold'])
            right=folds[(folds.recipe==reference)&(folds['mode']==mode)].set_index(['repeat','fold'])
            if left.empty or right.empty: continue
            for metric in ['pr_auc','brier_score','log_loss','f2','flagged_fraction']:
                diff=left[metric]-right[metric]
                paired.append(dict(challenger=challenger,reference=reference,mode=mode,metric=metric,
                    mean_difference=float(diff.mean()),sd_difference=float(diff.std()),
                    min_difference=float(diff.min()),max_difference=float(diff.max()),
                    positive_differences=int((diff>0).sum()),folds=len(diff)))
    pd.DataFrame(paired).to_csv(output/'paired_ablations.csv',index=False)
    write_json(output/'summary_notes.json',dict(
        assessment='Repeated nested DEVELOPMENT CV; historical test excluded. Not fresh external assessment.',
        averaging='Headline AP is mean outer-fold AP; pooled per-repeat AP is separate because fold score scales differ.',
        uncertainty='Fold SD and paired differences are descriptive; overlapping training sets and repeated rows are dependent.',
        selection='Family selector chooses a recipe using inner CV only in each outer fold; recipe tables are all reported.',
        sample_efficiency='Each outer fit uses 60% of its training partition; 20% calibration and 20% threshold data remain disjoint.',
        calibration='Raw and monotone-sigmoid outputs share the identical base fit; calibrator C=1 is fixed, not outer-selected.'))
    return folds,summary


def permutation_control(protocol,X,y,output):
    path=output/'permutation_control.json'
    if path.exists(): return
    estimator=next(r.estimator for r in recipes() if r.name=='lr_clip')
    estimator.set_params(classifier__C=.03)
    cv=StratifiedKFold(n_splits=3,shuffle=True,random_state=42)
    start=time.perf_counter()
    with parallel_backend('threading',n_jobs=protocol['search_jobs']):
        observed,null,p=permutation_test_score(estimator,X,y,cv=cv,scoring='average_precision',
            n_permutations=protocol['permutation_repetitions'],n_jobs=protocol['search_jobs'],random_state=42)
    write_json(path,dict(observed_mean_cv_ap=float(observed),null_scores=null.tolist(),
        monte_carlo_p=float(p),permutations=len(null),seconds=time.perf_counter()-start,
        scope='Fixed predeclared clipped L2 C=.03 benchmark, 3-fold development CV; not corrected for broader model search.',
        caveat='Exchangeability of borrower observations is unverified; this is a diagnostic, not proof of a signal ceiling.'))


def run(protocol_path,resume=False,only=None,output_override=None):
    protocol=load_protocol(protocol_path); output=Path(protocol['output'])
    if output_override is not None:
        output=Path(output_override).resolve();protocol['output']=str(output)
    if only: protocol['only_recipes']=only
    signature_files=[Path(protocol_path).resolve(),*sorted((ROOT/'src').rglob('*.py'))]
    signature=hashlib.sha256(''.join(digest(p) for p in signature_files).encode()).hexdigest()
    if output.exists() and not resume: raise FileExistsError(f'{output} exists; use --resume with the unchanged protocol')
    output.mkdir(parents=True,exist_ok=True)
    X,y,excluded,base_cfg=development_data(protocol)
    identity=dict(protocol=protocol,code_signature=signature,environment=runtime_environment(),
                  source_hashes={str(p.relative_to(ROOT)):digest(p) for p in signature_files if p.is_relative_to(ROOT)},
                  data_sha256=digest(base_cfg['data']['path']),
                  exclusion_sha256=digest(protocol['historical_test_manifest']))
    if (output/'identity.json').exists():
        if json.loads((output/'identity.json').read_text())!=identity: raise ValueError('Cannot resume changed code, data or protocol')
    else:
        write_json(output/'identity.json',identity)
        write_json(output/'preserved_files.json',preservation_snapshot(output))
    write_json(output/'exclusions.json',dict(historical_test_indices=excluded.tolist(),development_indices=X.index.tolist()))
    audit(protocol,X,y,excluded,base_cfg,output)
    available=recipes()
    if only:
        available=[r for r in available if r.name in only]
        if {r.name for r in available}!=set(only): raise ValueError('Unknown recipe name')
    write_json(output/'experiment_plan.json',[dict(name=r.name,family=r.family,hypothesis=r.hypothesis,grid=r.grid,
                                                  eligible_for_family_selection=r.selectable) for r in available])
    versions={p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','scipy','xgboost','joblib','PyYAML']}
    write_json(output/'environment.json',dict(python=platform.python_version(),packages=versions,
        parallelism='2 threading search workers; each XGBoost estimator n_jobs=1'))
    records=[]; predictions=[]; run_start=time.perf_counter()
    for repeat,seed in enumerate(protocol['outer_seeds']):
        outer=StratifiedKFold(n_splits=protocol['outer_folds'],shuffle=True,random_state=seed)
        for fold,(train,assessment) in enumerate(outer.split(X,y)):
            parts=partitions(X,y,protocol,repeat,fold,train,assessment)
            assert not set(excluded)&set(np.concatenate(list(parts.values())))
            parent=output/f'repeat_{repeat}/fold_{fold}'; parent.mkdir(parents=True,exist_ok=True)
            write_json(parent/'partitions.json',{name:dict(indices=indices.tolist(),rows=len(indices),events=int(y.loc[indices].sum())) for name,indices in parts.items()})
            current=[]; fold_predictions={}
            for recipe in available:
                folder=parent/recipe.name
                if (folder/'results.json').exists() and (folder/'assessment_predictions.csv').exists() and (folder/'model.joblib').exists():
                    values=json.loads((folder/'results.json').read_text()); frame=pd.read_csv(folder/'assessment_predictions.csv',float_precision='round_trip')
                else:
                    print(f'repeat {repeat+1}/{len(protocol["outer_seeds"])} fold {fold+1}/{protocol["outer_folds"]}: {recipe.name}',flush=True)
                    values,frame=fit_recipe(recipe,X,y,parts,protocol,folder,seed+fold*100)
                for value in values: value.update(repeat=repeat,fold=fold)
                frame['repeat']=repeat;frame['fold']=fold
                current.extend(values);records.extend(values);predictions.append(frame);fold_predictions[recipe.name]=frame
            for family in ['glassbox','blackbox']:
                eligible=[r.name for r in available if r.family==family and r.selectable]
                candidates=[v for v in current if v['recipe'] in eligible and v['mode']=='raw']
                if not candidates: continue
                best=sorted(candidates,key=lambda v:(-v['inner_ap'],v['recipe']))[0]
                for value in current:
                    if value['recipe']==best['recipe']:
                        selected=deepcopy(value); selected.update(recipe='selector_'+family,selected_recipe=best['recipe']);records.append(selected)
                selected_frame=fold_predictions[best['recipe']].copy();selected_frame['selected_recipe']=best['recipe'];selected_frame['recipe']='selector_'+family
                predictions.append(selected_frame)
            # Checkpoint summaries contain complete folds only and do not drive any next-fold choice.
            summarize(output,records,pd.concat(predictions,ignore_index=True),protocol)
    permutation_control(protocol,X,y,output)
    preserved=json.loads((output/'preserved_files.json').read_text())
    write_json(output/'completion.json',dict(complete=True,seconds_this_invocation=time.perf_counter()-run_start,
        preserved_historical_files=check_preservation(preserved),outer_fold_evaluations=len(records),
        assessment_rows_per_recipe=len(X)*len(protocol['outer_seeds']),historical_test_re_evaluated=False))
    print(f'Completed development experiments: {output}',flush=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'config/portfolio.yaml')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--only',nargs='+',help='Diagnostic runs only; use a separate output configuration')
    parser.add_argument('--output-dir',type=Path,help='New immutable artifact directory for an independent reproduction')
    args=parser.parse_args(argv)
    run(args.config,args.resume,args.only,args.output_dir)


if __name__=='__main__': main()
