"""Audit every saved partition/prediction and quantify conditional paired uncertainty."""
from itertools import combinations
from pathlib import Path
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from .data import ROOT,load_protocol,development_data,write_json,check_preservation
from .protocol import evaluate
from .interpretation import raw_field


def validate_run(path):
    output=Path(path);folds=pd.read_csv(output/'fold_results.csv')
    preds=pd.read_csv(output/'development_predictions.csv',float_precision='round_trip')
    cfg=json.loads((output/'identity.json').read_text())['protocol']
    X,y,excluded,_=development_data(cfg)
    if set(preds.row_index)&set(excluded): raise ValueError('Historical test used')
    audits=[];fit_diagnostics=[]
    for parent in sorted(output.glob('repeat_*/fold_*')):
        parts=json.loads((parent/'partitions.json').read_text())
        sets={k:set(v['indices']) for k,v in parts.items()}
        assert set.union(*sets.values())==set(X.index)
        for a,b in combinations(sets,2): assert not sets[a]&sets[b]
        for directory in sorted(p for p in parent.iterdir() if p.is_dir()):
            if not (directory/'model.joblib').exists():continue
            bundle=joblib.load(directory/'model.joblib');base=bundle['pipeline'];cal=bundle['calibrator']
            saved=pd.read_csv(directory/'assessment_predictions.csv',float_precision='round_trip')
            raw=saved[saved['mode']=='raw']
            assert set(raw.row_index)==sets['assessment']
            predicted=base.predict_proba(X.loc[raw.row_index])[:,1]
            np.testing.assert_allclose(predicted,raw.probability,rtol=1e-7,atol=1e-9)
            calibrated=saved[saved['mode']=='sigmoid']
            np.testing.assert_allclose(cal.predict(predicted),calibrated.probability,rtol=1e-7,atol=1e-9)
            for record in json.loads((directory/'results.json').read_text()):
                frame=saved[saved['mode']==record['mode']]
                metrics=evaluate(frame.actual,frame.probability,record['metrics']['threshold'])
                for key in ['pr_auc','roc_auc','brier_score','log_loss','precision','recall','f2','flagged']:
                    np.testing.assert_allclose(metrics[key],record['metrics'][key],rtol=1e-7,atol=1e-10)
            classifier=base.named_steps['classifier']
            iterations=np.asarray(classifier.n_iter_) if hasattr(classifier,'n_iter_') else None
            if iterations is not None:assert np.all(iterations<classifier.max_iter)
            fit_indices=parts['fit']['indices']
            fit_diagnostics.append(dict(directory=str(directory.relative_to(output)),recipe=directory.name,
                solver=getattr(classifier,'solver',None),C=getattr(classifier,'C',None),
                max_observed_iterations=int(iterations.max()) if iterations is not None else None,
                declared_max_iterations=getattr(classifier,'max_iter',None),
                base_intercept=float(classifier.intercept_[0]) if hasattr(classifier,'coef_') else None,
                fitted_mean_raw_score=float(base.predict_proba(X.loc[fit_indices])[:,1].mean()),
                fit_event_rate=float(y.loc[fit_indices].mean()),calibration_slope=cal.slope_,
                calibration_events=cal.events_,calibration_intercept=cal.intercept_))
            inner=json.loads((directory/'inner_splits.json').read_text())
            for split in inner:
                assert not set(split['train'])&set(split['validation'])
                assert set(split['train'])|set(split['validation'])==sets['fit']
            audits.append(dict(directory=str(directory.relative_to(output)),fit_rows=len(sets['fit']),
                calibration_rows=len(sets['calibration']),threshold_rows=len(sets['threshold']),assessment_rows=len(raw)))
    counts=preds.groupby(['recipe','mode','repeat']).row_index.agg(['size','nunique'])
    assert (counts['size']==len(X)).all() and (counts['nunique']==len(X)).all()
    for _,frame in preds.groupby(['recipe','mode','repeat']):
        np.testing.assert_array_equal(y.loc[frame.row_index],frame.actual)
    write_json(output/'validation.json',dict(passed=True,saved_models_checked=len(audits),
        per_model=audits,historical_exclusion_rows=len(excluded),development_rows=len(X),
        results='All partition roles disjoint; all inner folds fit-only; serialized models reproduce all assessment predictions and metrics.',
        reproducibility_scope='Exact predictions checked in the pinned local environment; runtime is hardware-dependent.'))
    pd.DataFrame(fit_diagnostics).to_csv(output/'model_fit_diagnostics.csv',index=False)
    return folds,preds,X,y


def bootstrap(output,preds,X,y,repetitions):
    targets=[('selector_glassbox','dummy'),('selector_blackbox','dummy'),
             ('selector_blackbox','selector_glassbox'),('lr_clip','lr_standard'),('lr_grouped','lr_clip')]
    pos=np.flatnonzero(y.to_numpy()==1);neg=np.flatnonzero(y.to_numpy()==0)
    groups=[]
    for (repeat,fold),frame in preds[(preds.recipe=='dummy')&(preds['mode']=='raw')].groupby(['repeat','fold']):
        order=frame.row_index.to_numpy();indices=X.index.get_indexer(order);labels=y.loc[order].to_numpy()
        scores={}
        for name in set(sum(([a,b] for a,b in targets),[])):
            sub=preds[(preds.recipe==name)&(preds['mode']=='raw')&(preds['repeat']==repeat)&(preds.fold==fold)]
            scores[name]=sub.set_index('row_index').loc[order].probability.to_numpy()
        groups.append((indices,labels,scores))
    rng=np.random.default_rng(2026);samples={key:[] for key in targets}
    for _ in range(repetitions):
        weights=np.bincount(np.r_[rng.choice(pos,len(pos),replace=True),rng.choice(neg,len(neg),replace=True)],minlength=len(X))
        values={name:[] for name in groups[0][2]}
        for indices,labels,scores in groups:
            for name,p in scores.items(): values[name].append(average_precision_score(labels,p,sample_weight=weights[indices]))
        for key in targets:samples[key].append(float(np.mean(values[key[0]])-np.mean(values[key[1]])))
    result=[]
    for (challenger,reference),values in samples.items():
        result.append(dict(challenger=challenger,reference=reference,mode='raw',
            mean_bootstrap_difference=float(np.mean(values)),p025=float(np.quantile(values,.025)),p975=float(np.quantile(values,.975)),
            repetitions=repetitions))
    write_json(output/'conditional_bootstrap.json',dict(comparisons=result,
        resampling='Stratified ORIGINAL development rows; the same multiplicities are used across recipes, folds and repeats.',
        estimand='Difference in mean of six outer-fold APs, conditional on the already fitted scores and observed prevalence.',
        warning='Intervals do not include model retraining, adaptive experiment design, unknown borrower dependence or future distribution shift. Not evidence of statistical superiority.'))


def selection_stability(output):
    rows=[];sets={}
    for directory in Path(output).glob('repeat_*/fold_*/*'):
        if not (directory/'model.joblib').exists():continue
        bundle=joblib.load(directory/'model.joblib');base=bundle['pipeline'];raw=base.named_steps['features'].selected_columns_
        clf=base.named_steps['classifier'];names=base.named_steps['preprocessor'].get_feature_names_out()
        active=set(raw)
        if hasattr(clf,'coef_'):
            active={raw_field(name,raw) for name,weight in zip(names,clf.coef_[0]) if abs(weight)>1e-10}
        recipe=directory.name;fold_key=directory.parent.parent.name+'/'+directory.parent.name
        sets.setdefault(recipe,[]).append((fold_key,active))
        for feature in raw:rows.append(dict(recipe=recipe,fold=fold_key,raw_field=feature,active=feature in active))
    pd.DataFrame(rows).to_csv(Path(output)/'feature_retention.csv',index=False)
    comparisons=[]
    for recipe,folds in sets.items():
        for (a,left),(b,right) in combinations(folds,2):
            comparisons.append(dict(recipe=recipe,fold_a=a,fold_b=b,jaccard=len(left&right)/len(left|right) if left|right else 1.,
                                    active_fields_a=len(left),active_fields_b=len(right)))
    pd.DataFrame(comparisons).to_csv(Path(output)/'feature_selection_stability.csv',index=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=ROOT/'artifacts/portfolio_v1')
    parser.add_argument('--followup-dir',type=Path,default=ROOT/'artifacts/portfolio_intercept_v1')
    parser.add_argument('--skip-bootstrap',action='store_true',help='Recheck saved artifacts without repeating the existing bootstrap')
    args=parser.parse_args()
    main_path=args.run_dir.resolve();followup=args.followup_dir.resolve()
    protocol=json.loads((main_path/'identity.json').read_text())['protocol']
    _,preds,X,y=validate_run(main_path);validate_run(followup)
    selection_stability(main_path);selection_stability(followup)
    if not args.skip_bootstrap:bootstrap(main_path,preds,X,y,protocol['bootstrap_repetitions'])
    historical=json.loads((main_path/'preserved_files.json').read_text())
    write_json(main_path/'preservation_check.json',dict(passed=True,unchanged_files=check_preservation(historical)))
    print('All saved models, role isolation, predictions and metric checks passed.',flush=True)


if __name__=='__main__':main()
