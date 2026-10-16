"""Held-out development explanations across errors, backgrounds and LIME seeds."""
from pathlib import Path
from itertools import combinations
import argparse
import time
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits
from .data import ROOT,load_protocol,development_data,write_json


def raw_field(feature,raw_columns):
    value=feature.split('__',1)[-1]
    if value.startswith('missingindicator_'): value=value[len('missingindicator_'):]
    for column in sorted(raw_columns,key=len,reverse=True):
        if value==column or value.startswith(column+'_'): return column
    raise ValueError(f'No raw-field lineage for {feature}')


def connected_groups(frame,columns,limit=.98):
    numeric=[c for c in columns if pd.api.types.is_numeric_dtype(frame[c])]
    corr=frame[numeric].corr(method='spearman',min_periods=100).abs()
    parent={c:c for c in columns}
    def root(c):
        while parent[c]!=c: c=parent[c]
        return c
    for i,left in enumerate(numeric):
        for right in numeric[i+1:]:
            if corr.loc[left,right]>=limit:
                a,b=root(left),root(right)
                parent[max(a,b)]=min(a,b)
    return {c:root(c) for c in columns}


def run(protocol_path,run_dir=None):
    import shap
    from lime.lime_tabular import LimeTabularExplainer
    protocol=load_protocol(protocol_path)
    if run_dir is not None:protocol['output']=str(Path(run_dir).resolve())
    output=Path(protocol['output'])/'interpretation'
    if output.exists(): raise FileExistsError(f'{output} already exists; preserve it and use a new protocol output for another study')
    output.mkdir()
    X,y,excluded,_=development_data(protocol)
    repeat=protocol['interpretation_repeat'];fold=protocol['interpretation_fold']
    plan=pd.read_json(Path(protocol['output'])/'experiment_plan.json')
    summaries={};all_cases=[];all_lime=[];all_weights=[];start=time.perf_counter()
    for family in ['glassbox','blackbox']:
        print('Explaining development '+family,flush=True)
        parent=Path(protocol['output'])/f'repeat_{repeat}/fold_{fold}'
        eligible=plan[(plan.family==family)&plan.eligible_for_family_selection]['name'].tolist()
        candidates=[]
        for name in eligible:
            import json
            candidates.extend(v for v in json.loads((parent/name/'results.json').read_text()) if v['mode']=='raw')
        recipe=sorted(candidates,key=lambda v:(-v['inner_ap'],v['recipe']))[0]['recipe']
        directory=parent/recipe
        bundle=joblib.load(directory/'model.joblib');base=bundle['pipeline'];calibrator=bundle['calibrator'];parts=bundle['partitions']
        before=joblib.hash(bundle)
        assessment=X.loc[parts['assessment']];fit=X.loc[parts['fit']]
        assert not set(excluded)&(set(fit.index)|set(assessment.index))
        pre=base.named_steps['preprocessor'];feature_filter=base.named_steps['features'];classifier=base.named_steps['classifier']
        transform=lambda frame:pre.transform(feature_filter.transform(frame))
        values=transform(assessment);names=pre.get_feature_names_out().tolist();raw=feature_filter.selected_columns_
        lineage=[raw_field(name,raw) for name in names]
        predictions=pd.read_csv(directory/'assessment_predictions.csv',float_precision='round_trip')
        frame=predictions[predictions['mode']=='sigmoid'].set_index('row_index').loc[assessment.index].copy()
        np.testing.assert_allclose(base.predict_proba(assessment)[:,1],frame.raw_probability,atol=1e-7)
        frame['cell']=np.select([(frame.actual==1)&(frame.prediction==1),(frame.actual==0)&(frame.prediction==1),
            (frame.actual==1)&(frame.prediction==0)],['TP','FP','FN'],default='TN')
        cases=[]
        for cell in ['TP','FP','FN','TN']:
            candidates=frame[frame.cell==cell]
            chosen=candidates.sample(min(len(candidates),protocol['explanation_rows_per_confusion_cell']),random_state=42)
            cases.extend(chosen.index.tolist())
        positions=[assessment.index.get_loc(row) for row in cases]
        arrays=[];baselines=[];global_means=[];errors=[]
        for seed in protocol['explanation_stability_seeds']:
            background=transform(fit.sample(min(len(fit),protocol['explanation_background_rows']),random_state=seed))
            explainer=shap.LinearExplainer(classifier,background) if family=='glassbox' else shap.TreeExplainer(
                classifier,data=background,feature_perturbation='interventional',model_output='raw')
            explanation=explainer(values)
            phi=np.asarray(explanation.values);margin=np.asarray(explanation.base_values)
            check=float(np.max(np.abs(expit(phi.sum(axis=1)+margin)-frame.raw_probability.to_numpy())))
            if check>1e-5: raise ValueError(f'SHAP additivity error {check}')
            arrays.append(phi);baselines.append(margin);global_means.append(np.abs(phi).mean(axis=0));errors.append(check)
        np.savez_compressed(output/f'{family}_shap.npz',values=np.asarray(arrays),base_values=np.asarray(baselines),
            row_indices=assessment.index.to_numpy(),feature_names=np.asarray(names),case_indices=np.asarray(cases),
            raw_probability=frame.raw_probability.to_numpy(),calibrated_probability=frame.probability.to_numpy(),
            actual=frame.actual.to_numpy(),predicted=frame.prediction.to_numpy())
        global_means=np.asarray(global_means)
        weights=classifier.coef_[0] if family=='glassbox' else classifier.feature_importances_
        pd.DataFrame(dict(feature=names,raw_field=lineage,native_weight=weights,
            shap_mean_abs=global_means.mean(axis=0),shap_background_sd=global_means.std(axis=0))).to_csv(output/f'{family}_features.csv',index=False)
        group_map=connected_groups(fit,raw)
        cluster_names=sorted(set(group_map.values()))
        grouped=[]
        for cluster in cluster_names:
            indices=[j for j,column in enumerate(lineage) if group_map[column]==cluster]
            contributions=arrays[0][:,indices].sum(axis=1)
            grouped.append(dict(group=cluster,members=';'.join(sorted(c for c in raw if group_map[c]==cluster)),
                mean_absolute_group_shap=float(np.abs(contributions).mean()),transformed_terms=len(indices)))
        pd.DataFrame(grouped).sort_values('mean_absolute_group_shap',ascending=False).to_csv(output/f'{family}_correlated_groups.csv',index=False)
        comparisons=[]
        for a,b in combinations(range(len(global_means)),2):
            topa=set(np.argsort(global_means[a])[-10:]);topb=set(np.argsort(global_means[b])[-10:])
            comparisons.append(dict(seed_a=protocol['explanation_stability_seeds'][a],seed_b=protocol['explanation_stability_seeds'][b],
                top10_jaccard=len(topa&topb)/len(topa|topb),global_rank_spearman=float(spearmanr(global_means[a],global_means[b]).statistic)))
        local_sign=[]
        for i in positions:
            top=np.argsort(np.abs(arrays[0][i]))[-5:]
            local_sign.append(float(np.mean([np.mean(np.sign(arrays[0][i,top])==np.sign(other[i,top])) for other in arrays[1:]])))
        lime_start=time.perf_counter()
        for seed in protocol['explanation_stability_seeds']:
            # Freeze the same training background so this study isolates perturbation RNG, not both effects.
            background=transform(fit.sample(min(len(fit),protocol['explanation_background_rows']),random_state=42))
            lime=LimeTabularExplainer(background,feature_names=names,class_names=['non-default','default'],
                mode='classification',random_state=seed,discretize_continuous=False)
            for row,position in zip(cases,positions):
                with threadpool_limits(limits=1):
                    ex=lime.explain_instance(values[position],classifier.predict_proba,labels=(1,),num_features=min(15,len(names)),
                                             num_samples=protocol['lime_perturbations'])
                raw_p=float(frame.loc[row,'raw_probability']);local=float(np.asarray(ex.local_pred).ravel()[0])
                all_lime.append(dict(family=family,recipe=recipe,row_index=row,cell=frame.loc[row,'cell'],seed=seed,
                    model_probability=raw_p,calibrated_probability=float(frame.loc[row,'probability']),
                    local_prediction=local,absolute_local_error=abs(local-raw_p),weighted_training_r2=float(ex.score),
                    perturbations=protocol['lime_perturbations'],scope='uncalibrated base probability; LIME standardized transformed coordinates'))
                for j,weight in ex.local_exp[1]:
                    all_weights.append(dict(family=family,row_index=row,seed=seed,feature=names[j],raw_field=lineage[j],weight=float(weight)))
        for row in cases:
            item=frame.loc[row]
            all_cases.append(dict(family=family,recipe=recipe,row_index=row,cell=item['cell'],actual=int(item.actual),
                raw_probability=float(item.raw_probability),calibrated_probability=float(item.probability),threshold=float(item.threshold),
                prediction=int(item.prediction),repeat=repeat,fold=fold))
        summaries[family]=dict(selected_recipe=recipe,repeat=repeat,fold=fold,explained_assessment_rows=len(assessment),
            assessment_events=int(frame.actual.sum()),transformed_features=len(names),retained_raw_fields=len(raw),
            stratified_case_rows=len(cases),cell_counts={k:int(v) for k,v in frame.cell.value_counts().items()},
            background_rows=min(len(fit),protocol['explanation_background_rows']),
            seeds=protocol['explanation_stability_seeds'],shap_additivity_max_error=max(errors),
            background_stability=comparisons,mean_top5_local_sign_agreement=float(np.mean(local_sign)),
            calibration_slope=calibrator.slope_,calibration_intercept=calibrator.intercept_,
            lime_seconds=time.perf_counter()-lime_start,
            scope='SHAP/native explain base log odds. LIME approximates base probability. Sigmoid decisions shown separately.',
            sampling='All assessment rows for SHAP; up to 8 random cases per confusion cell for LIME. Not population-weighted sampling.')
        assert joblib.hash(bundle)==before,'Interpretation changed model state'
    case_frame=pd.DataFrame(all_cases);lime_frame=pd.DataFrame(all_lime);weight_frame=pd.DataFrame(all_weights)
    case_frame.to_csv(output/'cases.csv',index=False);lime_frame.to_csv(output/'lime_diagnostics.csv',index=False)
    weight_frame.to_csv(output/'lime_weights.csv',index=False)
    stability=[]
    for (family,row),data in weight_frame.groupby(['family','row_index']):
        sets={seed:set(frame.feature) for seed,frame in data.groupby('seed')}
        for a,b in combinations(sets,2):
            stability.append(dict(family=family,row_index=int(row),seed_a=int(a),seed_b=int(b),
                selected15_jaccard=len(sets[a]&sets[b])/len(sets[a]|sets[b])))
    pd.DataFrame(stability).to_csv(output/'lime_stability.csv',index=False)
    for family in summaries:
        lime=lime_frame[lime_frame.family==family]
        summaries[family]['lime_summary']=dict(explanations=len(lime),cases=int(lime.row_index.nunique()),
            r2_median=float(lime.weighted_training_r2.median()),r2_p10=float(lime.weighted_training_r2.quantile(.1)),
            absolute_error_median=float(lime.absolute_local_error.median()),absolute_error_p90=float(lime.absolute_local_error.quantile(.9)),
            selected15_jaccard_mean=float(pd.DataFrame(stability).query('family == @family').selected15_jaccard.mean()))
    write_json(output/'summary.json',dict(families=summaries,total_seconds=time.perf_counter()-start,
        historical_test_used=False,case_selection='Fixed outer development repeat 0, fold 0; stratified random successes/errors, seed 42.',
        limits=['One predeclared outer model per family, not all fitted models.',
                'Background stability does not establish retraining stability or causality.',
                'LIME perturbs transformed one-hot fields independently; synthetic profiles can be implausible.',
                'Weighted training R2 is local in-sample surrogate fit, not out-of-sample faithfulness.']))
    print('Interpretation complete: '+str(output),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,default=ROOT/'config/portfolio.yaml')
    parser.add_argument('--run-dir',type=Path);args=parser.parse_args();run(args.config,args.run_dir)
