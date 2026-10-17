"""Generate the current portfolio report/deck/theory from completed local evidence."""
from pathlib import Path
import json
import sys
from xml.sax.saxutils import escape
import re
import argparse
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import expit
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve,roc_curve,average_precision_score
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak,Image,Table,TableStyle
from PIL import Image as PILImage
import tools.report_rendering as renderer
from tools.report_references import SOURCES

RUN=ROOT/'artifacts/portfolio_v1'
FOLLOW=ROOT/'artifacts/portfolio_intercept_v1'
REPORT=ROOT/'reports/generated'
FIGURES=REPORT/'figures'
NAVY,TEAL,GOLD,RED,GRAY=renderer.NAVY,renderer.TEAL,renderer.GOLD,renderer.RED,renderer.GRAY


def j(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def save(fig,name):
    target=FIGURES/(name+'.png');fig.savefig(target,dpi=180,bbox_inches='tight',facecolor='white');plt.close(fig);return target


def load():
    for directory in [REPORT,FIGURES]:directory.mkdir(parents=True,exist_ok=True)
    for run in [RUN,FOLLOW]:
        if not j(run/'completion.json')['complete'] or not j(run/'validation.json')['passed']:raise ValueError('Incomplete/unvalidated experiment')
    data={}
    for key,filename in [('summary','comparison.csv'),('folds','fold_results.csv'),('predictions','development_predictions.csv'),
        ('paired','paired_ablations.csv'),('stability','feature_selection_stability.csv')]:
        data[key]=pd.read_csv(RUN/filename,float_precision='round_trip')
    data['follow_summary']=pd.read_csv(FOLLOW/'comparison.csv')
    data['follow_folds']=pd.read_csv(FOLLOW/'fold_results.csv')
    data['all_summary']=pd.concat([data['summary'].assign(study='planned'),data['follow_summary'].assign(study='adaptive follow-up')],ignore_index=True)
    data['audit']=j(RUN/'data_audit.json');data['historical']=j(RUN/'historical_diagnosis.json')
    data['interpretation']=j(RUN/'interpretation/summary.json');data['bootstrap']=j(RUN/'conditional_bootstrap.json')
    data['permutation']=j(RUN/'permutation_control.json');data['completion']=j(RUN/'completion.json')
    data['cases']=pd.read_csv(RUN/'interpretation/cases.csv');data['lime']=pd.read_csv(RUN/'interpretation/lime_diagnostics.csv')
    data['feature_audit']=pd.read_csv(RUN/'feature_audit.csv')
    data['fit_diagnostics']=pd.read_csv(RUN/'model_fit_diagnostics.csv')
    return data


def row(data,name,mode='sigmoid'):
    return data['all_summary'][(data['all_summary'].recipe==name)&(data['all_summary']['mode']==mode)].iloc[0]


def charts(data):
    plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,
        'text.color':NAVY,'axes.labelcolor':NAVY,'xtick.color':GRAY,'ytick.color':GRAY,'axes.titleweight':'bold'})
    result={}
    fig,axes=plt.subplots(1,2,figsize=(11.8,4.5))
    frame=data['feature_audit'].sort_values('missing_fraction').tail(10)
    axes[0].barh([renderer.short(f,36) for f in frame.feature],frame.missing_fraction*100,color=TEAL)
    axes[0].set(xlabel='Development missing values (%)',title='Six fields are >99% missing',xlim=(0,105))
    cat=data['feature_audit'].query('type == "categorical"').sort_values('unique_observed')
    axes[1].barh(cat.feature,cat.unique_observed,color=GOLD)
    axes[1].set(xlabel='Observed categories in 2,400 development rows',title='Industry encoding dominates cardinality')
    for ax in axes:ax.tick_params(axis='y',labelsize=8.5);ax.grid(axis='x',alpha=.15)
    fig.tight_layout();result['quality']=save(fig,'data_quality')
    for family,title in [('glassbox','Logistic ablations'),('blackbox','Tree benchmarks and searches')]:
        frame=data['summary'].query('mode == "raw" and family == @family').sort_values('ap_mean')
        fig,ax=plt.subplots(figsize=(10,6 if family=='glassbox' else 4.5))
        ax.errorbar(frame.ap_mean,np.arange(len(frame)),xerr=frame.ap_sd,fmt='o',color=TEAL if family=='glassbox' else GOLD,capsize=3)
        ax.set_yticks(range(len(frame)),frame.recipe,fontsize=10);ax.axvline(146/2400,ls=':',color=GRAY,label='Development prevalence')
        ax.set(title=title+' / all six outer folds',xlabel='Mean outer-development AP ± fold SD');ax.grid(axis='x',alpha=.15);ax.legend(fontsize=9)
        fig.tight_layout();result[family+'_ablations']=save(fig,family+'_ablations')
    comparisons=data['paired'].query('mode == "raw" and metric == "pr_auc"')
    fig,ax=plt.subplots(figsize=(10.5,5.1))
    labels=[]
    for i,(_,record) in enumerate(comparisons.iterrows()):
        left=data['folds'].query('recipe == @record.challenger and mode == "raw"').set_index(['repeat','fold'])
        right=data['folds'].query('recipe == @record.reference and mode == "raw"').set_index(['repeat','fold'])
        values=(left.pr_auc-right.pr_auc).to_numpy()
        ax.scatter(values,np.full(len(values),i),color=TEAL,alpha=.6,s=28);ax.scatter(record.mean_difference,i,color=NAVY,marker='D',s=45)
        labels.append(record.challenger+' − '+record.reference)
    ax.set_yticks(range(len(labels)),labels,fontsize=9);ax.axvline(0,ls=':',color=GRAY)
    ax.set(xlabel='Paired change in outer AP; dots = folds, diamond = mean',title='Controlled ablations include failures');ax.grid(axis='x',alpha=.15)
    fig.tight_layout();result['paired']=save(fig,'paired_ablations')
    names=['lr_standard','lr_weighted','lr_clip','lr_historical_selected','xgb_legacy','xgb_search','xgb_weighted','selector_glassbox','selector_blackbox']
    fig,ax=plt.subplots(figsize=(10,5))
    for i,name in enumerate(names):
        raw=row(data,name,'raw');cal=row(data,name,'sigmoid')
        ax.plot([raw.brier_mean,cal.brier_mean],[i,i],color='#9EADB7')
        ax.scatter(raw.brier_mean,i,color=RED,s=45,label='Raw' if i==0 else None)
        ax.scatter(cal.brier_mean,i,color=TEAL,s=45,label='Independent sigmoid' if i==0 else None)
    ax.set_yticks(range(len(names)),names);ax.set(xlabel='Mean outer Brier loss (lower is better)',title='Calibration repairs probability levels, not signal')
    ax.axvline(row(data,'dummy','raw').brier_mean,color=GRAY,ls=':',label='Dummy prior');ax.legend(fontsize=9);ax.grid(axis='x',alpha=.15)
    fig.tight_layout();result['calibration_changes']=save(fig,'calibration_changes')
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.4))
    for name,color in [('selector_glassbox',TEAL),('selector_blackbox',GOLD)]:
        for mode,style in [('raw','--'),('sigmoid','-')]:
            frame=data['predictions'].query('recipe == @name and mode == @mode and repeat == 0')
            observed,predicted=calibration_curve(frame.actual,frame.probability,n_bins=8,strategy='quantile')
            axes[0].plot(predicted,observed,ls=style,marker='o',ms=4,color=color,label=name.replace('selector_','')+' '+mode)
        frame=data['predictions'].query('recipe == @name and mode == "sigmoid" and repeat == 0')
        axes[1].hist(frame.probability,bins=np.linspace(0,.4,25),color=color,alpha=.5,label=name.replace('selector_',''))
    axes[0].plot([0,1],[0,1],':',color=GRAY);axes[0].set(xlabel='Mean score in quantile bin',ylabel='Observed default rate',title='Repeat 0 / pooled outer-development reliability',xlim=(0,1),ylim=(0,1))
    axes[1].set(xlabel='Calibrated score (display 0–0.4)',ylabel='Development assessment rows',title='Independent sigmoid scores')
    for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.15)
    fig.tight_layout();result['reliability']=save(fig,'reliability')
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.4))
    for name,color in [('selector_glassbox',TEAL),('selector_blackbox',GOLD)]:
        frame=data['predictions'].query('recipe == @name and mode == "raw" and repeat == 0')
        fpr,tpr,_=roc_curve(frame.actual,frame.probability);precision,recall,_=precision_recall_curve(frame.actual,frame.probability)
        label=name.replace('selector_','')
        axes[0].plot(fpr,tpr,color=color,label=label)
        axes[1].step(recall,precision,color=color,where='post',label=label+f' pooled AP={average_precision_score(frame.actual,frame.probability):.3f}')
    axes[0].plot([0,1],[0,1],':',color=GRAY);axes[0].set(title='Repeat 0 / pooled outer-development ROC',xlabel='False-positive rate',ylabel='Recall')
    axes[1].axhline(146/2400,color=GRAY,ls=':');axes[1].set(title='Repeat 0 / pooled PR',xlabel='Recall',ylabel='Precision',ylim=(0,1))
    for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.15)
    fig.tight_layout();result['ranking']=save(fig,'ranking')
    fig,ax=plt.subplots(figsize=(9.5,4.3));p=data['permutation']
    ax.hist(p['null_scores'],bins=18,color='#9AB8C5',edgecolor='white');ax.axvline(p['observed_mean_cv_ap'],color=TEAL,lw=3,label=f'Observed fixed-pipeline AP {p["observed_mean_cv_ap"]:.3f}')
    ax.set(title='99 shuffled-label controls / fixed clipped L2 C=.03',xlabel='Mean 3-fold development AP',ylabel='Permutations');ax.legend(fontsize=10)
    fig.tight_layout();result['permutation']=save(fig,'permutation')
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.4))
    for ax,family in zip(axes,['glassbox','blackbox']):
        frame=data['lime'].query('family == @family')
        values=[frame.query('cell == @cell').weighted_training_r2.to_numpy() for cell in ['TP','FP','FN','TN']]
        ax.boxplot(values,tick_labels=['TP','FP','FN','TN'],showfliers=False)
        ax.set(title=family+' / local LIME fit',xlabel='Confusion cell of calibrated decision',ylabel='Weighted synthetic-neighborhood R²')
        ax.grid(axis='y',alpha=.15)
    fig.tight_layout();result['lime']=save(fig,'lime_fidelity')
    renderer.ASSETS=FIGURES
    for family in ['glassbox','blackbox']:
        features=pd.read_csv(RUN/f'interpretation/{family}_features.csv')
        result[family+'_shap']=renderer.bar_chart(features.feature,features.shap_mean_abs,
            family+' / 800 held-out development rows','Mean absolute base log-odds SHAP',family+'_shap',color=TEAL if family=='glassbox' else GOLD)
        groups=pd.read_csv(RUN/f'interpretation/{family}_correlated_groups.csv')
        result[family+'_groups']=renderer.bar_chart(groups.group,groups.mean_absolute_group_shap,
            family+' / correlated-field groups','Mean |sum of group SHAP|, log odds',family+'_groups',color=TEAL if family=='glassbox' else GOLD)
        result[family+'_native']=renderer.bar_chart(features.feature,features.native_weight,
            family+' / native output','Coefficient, log odds' if family=='glassbox' else 'Normalized mean split gain',
            family+'_native',signed=family=='glassbox',color=GOLD)
        array=np.load(RUN/f'interpretation/{family}_shap.npz')
        cases=data['cases'].query('family == @family and cell == "FN"')
        case=cases.iloc[0] if len(cases) else data['cases'].query('family == @family').iloc[0]
        position=int(np.flatnonzero(array['row_indices']==int(case.row_index))[0])
        values=array['values'][0,position];names=array['feature_names'];baseline=float(array['base_values'][0,position])
        order=np.argsort(np.abs(values))[::-1][:8]
        labels=[renderer.short(names[k],42) for k in order]+[f'Other {len(values)-len(order)} terms']
        effects=list(values[order])+[float(values.sum()-values[order].sum())]
        fig,ax=plt.subplots(figsize=(10.4,4.7));current=baseline
        for n,effect in enumerate(effects):
            previous=current
            ax.barh(n,abs(effect),left=min(current,current+effect),color=RED if effect>=0 else TEAL)
            current+=effect;ax.text((previous+current)/2,n,f'{effect:+.3f}',fontsize=8,color='white',va='center',ha='center',weight='bold')
        ax.set_yticks(range(len(labels)),labels,fontsize=9);ax.invert_yaxis()
        ax.axvline(baseline,ls=':',color=GRAY,label=f'Baseline {baseline:.3f}')
        ax.axvline(current,ls='--',color=NAVY,label=f'Final margin {current:.3f}')
        ax.set(title=f'{family} / {case.cell} development row {int(case.row_index)}',xlabel='Cumulative base-model log odds; background seed 42')
        ax.legend(fontsize=8,loc='lower right');fig.tight_layout();result[family+'_local']=save(fig,family+'_local_error')
    return result


def table_md(frame,columns,formatters=None):
    formatters=formatters or {}
    rows=['| '+' | '.join(label for key,label in columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    for _,record in frame.iterrows():
        values=[formatters.get(key,lambda v:str(v))(record[key]) for key,label in columns]
        rows.append('| '+' | '.join(values)+' |')
    return '\n'.join(rows)


def report_markdown(data):
    summary=data['all_summary'];primary=summary[summary.recipe.isin(['dummy','selector_glassbox','selector_blackbox'])&(summary['mode']=='sigmoid')]
    formats={key:(lambda v:f'{v:.4f}') for key in ['ap_mean','ap_sd','roc_auc_mean','brier_mean','log_loss_mean','f2_mean']}
    formats.update({key:(lambda v:f'{v:.1%}') for key in ['precision_mean','recall_mean','flagged_fraction_mean']})
    formats['false_positives_mean']=lambda v:f'{v:.1f}'
    columns=[('recipe','Procedure'),('ap_mean','AP mean'),('ap_sd','AP SD'),('roc_auc_mean','ROC-AUC'),('brier_mean','Brier'),('log_loss_mean','Log loss'),('precision_mean','Precision'),('recall_mean','Recall'),('false_positives_mean','FP / 800'),('flagged_fraction_mean','Flagged')]
    lines=['# Portfolio audit: reproducible development evidence','',
        'Experiment snapshot: 16 October 2026. Assessment uses repeated nested **development CV**, excluding the already inspected 600 historical test rows. There is no untouched final assessment cohort. The results do not establish production readiness.','',
        '## Main findings','',
        '- The data contain real redundancy, severe missingness, rare categories, infinite values and extreme ratios. Timing, default horizon and borrower identity remain undocumented.',
        '- Clipping numerical extremes is a useful development ablation. Extra feature selection is mixed: group-wise association selection can reduce dimensions, while blanket correlation pruning can discard useful information.',
        '- Both model-family selection procedures beat a constant-score benchmark in these development results. Their AP differences are small relative to fold variation; no statistical or universal superiority is claimed.',
        '- Calibration repairs probability levels but cannot create ranking signal. Strong liblinear regularization also penalizes the intercept; a labeled follow-up compares unpenalized-intercept L2 and weakly penalized-intercept L1.',
        '- LIME is much less faithful/stable for the studied tree model than for logistic regression. All explanations concern the base classifier, with calibrated decisions shown separately.','',
        '## Protocol and scope','',
        'The historical test membership is frozen in `config/historical_test_indices.csv`, with dataset and index hashes in `config/dataset_identity.json`. Original train + validation membership becomes a 2,400-row development pool containing 146 defaults. No new test set is created by reshuffling.','',
        'There are two fixed outer seeds (42, 43), three stratified outer folds per repeat, and three inner folds. In each outer split, 1,600 rows are available for development. These are split into 960 model-fit/search rows, 320 calibration rows, 320 threshold rows; the remaining 800 development rows form the outer assessment fold. Around 58–59 fit events and 19–20 events per calibration/threshold subset make this a conservative but noisy design. Each inner model fit has approximately 640 rows and 39 events.','',
        'Preprocessing, raw-field selection, category pooling, clipping, imputation and scaling fit inside each inner training fold. Hyperparameters are selected by mean inner AP. Calibration uses a fixed-C, nonnegative-slope sigmoid on independent, naturally imbalanced calibration rows. F2 thresholds are selected on a third disjoint subset. Both raw and calibrated outputs share exactly the same fitted base classifier. No resampling or early stopping is used.','',
        'A nested family selector chooses among eligible recipes using **inner AP only**, before accessing the outer labels. Its outer results assess that selection procedure. The full ablation table is descriptive development evidence and is not used to choose a retrospectively optimal seed. Separate held-out roles use less fitting data than historical experiments, so the new CV scores must not be compared as if they were historical test gains.','',
        '## Primary procedure-level results','',table_md(primary,columns,formats),'',
        'Values are means over six outer folds (each 800 rows). AP SD is a descriptive fold SD, not a confidence interval; the folds share training observations and each development row is assessed twice. Precision, recall and counts use development-selected F2 thresholds, not a business-cost rule.','',
        '## Data diagnosis: verified facts and unresolved hypotheses','',
        f'- {data["audit"]["raw_numeric_infinite_cells"]:,} infinite numeric cells in the full CSV; the loader converts them to missing values. No exact duplicate records. The historically dropped field has one unique value.',
        '- Eight exact duplicate predictor pairs and 66 development numeric pairs with |Spearman rho| ≥ .98 (at least 100 co-observations). Example exact pairs include `wsk_ROS` / `wsk_rent_sprzedazy` and `wsk_koszty_przychody` / `wsk_poziom_kosztow_operacyjnych`.',
        '- Seven development predictors exceed 95% missingness, including six above 99%. Six observed-value near-constants have a modal share ≥99.5%. No configured predictor is wholly missing or strictly constant over all development rows; fold-level constants still occur.',
        '- `pkdKod` has 368 development categories and 104 singleton levels. The largest absolute ratio is about 1.09 × 10^17. Near-zero denominators are a plausible explanation, but source formulas/units are not supplied, so those values are not declared invalid merely because they are large.',
        '- Two coincident four-field financial profiles cover four non-default rows. The pairs differ on 3 and 17 other fields; matching low-information profiles are not verified borrower IDs. No borrower-aware split can be justified as a complete solution from these signatures.',
        '- No usable timestamp or borrower key is documented. An automated substring name scan flags `podatek_dochodowy` because it contains “date”; manual schema review rejects that substring match as timestamp evidence. The age feature is not an observation date.',
        '- Default = 1 is verified as a binary label, but its horizon, event definition, statement dates, feature availability at decision time, repeated-borrower identity and collection process are unknown. Post-outcome leakage cannot be ruled out from names alone.','',
        'Full per-field missingness, cardinality, ranges and unverified availability are recorded in `docs/evidence/portfolio_v1/feature_audit.csv`. Correlation pairs and the frozen historical diagnosis are included alongside it.','',
        '## Historical diagnosis (exploratory only)','',
        'Original logistic AP was 0.611 in training versus 0.106 in validation. Original XGBoost was 0.991 in training versus 0.186 in validation. Those are large overfitting gaps, visible before consulting test scores. The previous selected XGBoost had mean inner training AP 0.244 versus CV AP 0.157; reducing variance also restricted fitted signal. Its fixed lambda=100, gamma=2 and child weight=10 can produce shallow or constant trees. This is evidence to investigate excessive regularization, not proof of one causal explanation for every score change.','',
        'Inspecting the actual old tree structures finds 0/300 constant trees and 10.76 mean leaves for the baseline, versus 40/250 constant trees and 2.46 mean leaves (maximum four) for the selected model. This verifies strong fitted restriction; it does not alone prove that weaker regularization will generalize.','',
        'The frozen historical configurations are re-evaluated under the same development protocol as all other recipes. Their rows (`lr_legacy`, `lr_historical_selected`, `xgb_legacy`, `xgb_historical_selected`) provide controlled comparisons without selecting anything from the historical test.','',
        '## All planned ablations','',
        'All selected logistic fits finish before the declared 1,000-iteration cap. Candidates that reach the cap are rejected rather than silently scored as successful; this recorded study has zero failed candidates. Per-model iteration counts, intercepts, raw fitted mean scores, fit prevalence and calibration parameters are retained in `model_fit_diagnostics.csv`. The intercept diagnosis is therefore auditable against actual fitted states, not inferred from test losses.','']
    planned=summary.query('study == "planned" and mode == "raw"').sort_values('recipe')
    cols=[('recipe','Recipe'),('ap_mean','AP'),('ap_sd','SD'),('inner_ap_mean','Inner AP'),('fitted_train_ap_mean','Fit AP'),('raw_fields_mean','Raw fields'),('transformed_features_mean','Dimensions'),('active_coefficients_mean','Active coefficients'),('search_seconds_total','Search seconds')]
    fmt={**formats,**{k:(lambda v:f'{v:.3f}') for k in ['inner_ap_mean','fitted_train_ap_mean']},**{k:(lambda v:'—' if pd.isna(v) else f'{v:.1f}') for k in ['raw_fields_mean','transformed_features_mean','active_coefficients_mean','search_seconds_total']}}
    lines += [table_md(planned,cols,fmt),'',
        'Recipe-level runtime is cumulative inner search plus base refit. Raw/sigmoid rows share that cost and must not be added together. Selector rows inherit the winning recipe’s cost and **do not represent the full family search budget**; full procedure cost includes all eligible candidate recipes. Total run time and environment are in completion.json/environment.json.','',
        'L2 C values are .0001, .003, .03, .3; clipped L1 uses .003, .01, .03, .1. Numerical clipping is fit-only at 1%/99%. Robust scaling is isolated after clipping. Rare-category cutoffs 5/20, missingness cutoffs .95/.99, correlation cutoffs .90/.98 and supervised raw-field counts 25/75/150 are selected in inner CV. Whole categorical fields are retained or removed together, never individual low-ranked one-hot columns.','',
        'The grouped supervised selector ranks raw numerical fields by ANOVA association and categorical fields by a rare-pooled chi-square association heuristic. Scores are **not inferential p-values** and are not interpreted causally. The redundancy reducer greedily retains a deterministic, better-observed representative by absolute Spearman correlation; it preserves complete categorical groups. Clipping, rare pooling and embedded sparsity are separate controlled comparisons.','',
        'The eight XGBoost candidates pair (150 trees,.08), (300,.04), (600,.02) with depths 2/3 and moderate L2/child/split controls. Identical candidate paths are used for unweighted, fold-weighted and rare-category variants. This replaces the former broad but sparsely covered Cartesian search. Tree counts are selected by inner validation; no early-stopping set is reused.','',
        '## Paired ablation differences','']
    p=data['paired'].query('mode == "raw" and metric == "pr_auc"')
    lines += [table_md(p,[('challenger','Challenger'),('reference','Reference'),('mean_difference','Mean ΔAP'),('sd_difference','SD ΔAP'),('positive_differences','Positive folds'),('folds','Folds')],
        {'mean_difference':lambda v:f'{v:+.4f}','sd_difference':lambda v:f'{v:.4f}','positive_differences':lambda v:str(int(v)),'folds':lambda v:str(int(v))}),'',
        'Every contrast uses paired partitions. Greater AP is better; negative differences are retained. More feature removal does not consistently help. The individual outer ablation rankings are not a fresh confirmation of the best recipe selected after viewing this table.','',
        '## Adaptive intercept follow-up','',
        'Strong regularization of liblinear’s synthetic intercept can pull even an unweighted score toward .5. This was diagnosed using development fits after the initial plan began. The additional study is therefore explicitly **adaptive development work**, not independent confirmation. The original planned results remain unchanged.','',
        'Two L2 recipes use lbfgs (unpenalized intercept), with identical C grids, with and without clipping. A third clipped L1 recipe changes only intercept_scaling from 1 to 100, weakening its intercept penalty. The L2 ablation changes both solver and intercept treatment; its results alone do not isolate the numerical solver from intercept penalization. All six outer partitions, calibration/threshold roles and seeds are identical.','',
        table_md(summary.query('study == "adaptive follow-up" and mode == "raw"'),[('recipe','Adaptive recipe'),('ap_mean','AP'),('ap_sd','SD'),('brier_mean','Brier'),('log_loss_mean','Log loss')],formats),'',
        'The follow-up improves probability behavior without assigning an arbitrary prior shift. It does not establish that the entire dataset’s signal has been exhausted, nor that the adaptive winner has unbiased final performance.','',
        '## Ranking, probabilities and decisions','',
        'The sigmoid slope is constrained nonnegative and has an L2 penalty with C=1; the intercept is unpenalized. This keeps complexity low with only about 19–20 calibration events and prevents learning a reverse risk ordering. Zero-slope calibrators become constant. Positive-slope calibration preserves within-fold order except numerical clipping/ties, so AP and optimal F2 decisions often stay unchanged while Brier/log loss change. It does not automatically preserve pooled rankings across different fitted folds.','',
        'No isotonic calibration is used with these small subsets. Both raw and sigmoid rows, reliability curves, log loss and Brier are retained. Lower Brier alone is not proof of calibration, and reliability-bin estimates are noisy. F2 expresses a recall preference, not an estimated lending-cost function; flagged volume and false positives remain central outcomes.','',
        '## Uncertainty and a signal diagnostic','',
        f'The fixed predeclared clipped L2 C=.03 benchmark has 3-fold development AP {data["permutation"]["observed_mean_cv_ap"]:.4f}; its 99 shuffled-label controls give Monte Carlo p={data["permutation"]["monte_carlo_p"]:.2f}. This diagnostic supports some association under row-exchangeability, not a performance ceiling, a corrected significance test for all searches, or verified borrower independence. It uses a separate fixed-pipeline CV diagnostic, not the nested-family headline procedure.','']
    boot=pd.DataFrame(data['bootstrap']['comparisons'])
    lines += [table_md(boot,[('challenger','Challenger'),('reference','Reference'),('p025','Conditional 2.5%'),('p975','Conditional 97.5%')],{'p025':lambda v:f'{v:+.4f}','p975':lambda v:f'{v:+.4f}'}),'',
        'The paired bootstrap resamples the original development rows, stratified by outcome, sharing row multiplicities across both repeats and recipes. It estimates differences in mean outer AP **conditional on the fitted predictions and observed prevalence**. It does not retrain models, correct adaptive design or account for unknown borrower clustering. These intervals must not be presented as statistical superiority of the whole future training procedure.','',
        '## Interpretation and stability','',
        'Interpretation uses predeclared outer repeat 0/fold 0, with each family’s recipe chosen solely by that fold’s inner AP. SHAP covers all 800 held-out development rows per family, over three 100-row fit-only backgrounds (seeds 42/43/44). Historical test rows are excluded from interpretation. Global importances average absolute base-log-odds SHAP; correlated raw fields are also grouped using fit-only |rho|≥.98 connected components, summing signed effects before taking mean absolute magnitude. A connected group may link fields whose mutual correlation is below .98 through a chain.','']
    for family,values in data['interpretation']['families'].items():
        lm=values['lime_summary'];bg=values['background_stability']
        lines += [f'### {family}', '',
            f'The explained fold selected `{values["selected_recipe"]}`: {values["transformed_features"]} transformed columns, {values["assessment_events"]} assessment defaults. All 800 rows are explained; 8 random cases each from TP/FP/FN/TN produce 32 case studies. Native coefficients/gain are exported separately. SHAP maximum probability reconstruction error is {values["shap_additivity_max_error"]:.3g}.', '',
            f'Across three backgrounds, mean top-10 Jaccard is {np.mean([b["top10_jaccard"] for b in bg]):.3f}; mean global rank Spearman is {np.mean([b["global_rank_spearman"] for b in bg]):.3f}. Background stability is not the same as retraining stability.', '',
            f'LIME has {lm["explanations"]} explanations (32 cases × 3 seeds), each using 2,000 perturbations. Median weighted synthetic-neighborhood R² is {lm["r2_median"]:.3f}, its 10th percentile {lm["r2_p10"]:.3f}; median absolute local-probability error {lm["absolute_error_median"]:.4f}, 90th percentile {lm["absolute_error_p90"]:.4f}. Selected-15-feature Jaccard averages {lm["selected15_jaccard_mean"]:.3f}.', '']
    stability=data['stability'].groupby('recipe').agg(mean_jaccard=('jaccard','mean'),min_jaccard=('jaccard','min'),comparisons=('jaccard','size')).reset_index()
    stability=stability[stability.recipe.isin(['lr_grouped','lr_l1','lr_redundancy','lr_drop_sparse'])]
    lines += ['### Cross-fit retained/active field stability','',table_md(stability,[('recipe','Recipe'),('mean_jaccard','Mean Jaccard'),('min_jaccard','Minimum'),('comparisons','Fold pairs')],
        {'mean_jaccard':lambda v:f'{v:.3f}','min_jaccard':lambda v:f'{v:.3f}','comparisons':lambda v:str(int(v))}),'',
        'Jaccard uses retained raw fields for explicit selectors and nonzero raw-field membership for L1. Keeping every field trivially gives high retention overlap; it is not evidence of stable effect magnitudes. Pairwise fold fits share training rows, so these are descriptive stability measurements.','',
        'SHAP/native are base log odds, LIME is a base-probability surrogate, and the sigmoid is a separate score mapping. Exact base reconstruction does not establish economic causality. The balanced success/error sample is intentionally not population-weighted; LIME can generate fractional one-hot values and implausible financial combinations. Its weighted R² is in-sample synthetic fidelity, not held-out faithfulness. The full case table, weights and diagnostics are retained. Error case studies show the first randomly sampled FN for each family, with its signed SHAP terms and actual raw/calibrated score and threshold; these are errors, not favorable maximum-score examples.','',
        '## Validation and reproducibility','',
        'All 126 saved base models across the planned and adaptive runs reproduce their assessment predictions. Fit, calibration, threshold and assessment roles are disjoint, inner folds use fit rows only, and metric/count exports are recomputed from predictions. Automated checks cover assessment-label isolation, frozen transforms, monotone calibration, clone/weight behavior and portable historical exclusions.','',
        'Saved-model verification is distinct from refitting. Independent refits reproduce the recorded selected settings, outer metrics and development predictions in the pinned environment. Inner L1 search scores vary slightly; exact inner-search bitwise reproducibility is not claimed and the cause was not isolated. Refit records are reproduction_check.json and publication_check.json in docs/evidence/portfolio_v1. The explanation study was not repeated in the full-refit check.','',
        'The reproduction guide is docs/reproduction.md. Experiment directories are immutable by default; resumption rejects changed code, data or configuration. The frozen exclusion manifest contains 600 original indices without labels. A CRLF/LF-normalized dataset hash rejects content or ordering changes; the original raw-byte hash is retained.','',

        '## What is still needed before deployment','',
        'A documented target horizon/event definition; borrower identifiers and statement/decision/outcome timestamps; verified feature availability and source formulas; appropriate group/time development splits; a new external or future cohort; specified costs and capacity; uncertainty studies that include refitting; subgroup error/calibration analysis where justified; monitoring/label-maturation procedures and operational ownership. None is claimed as implemented by this research benchmark.','',
        '## Sources','',
        'Primary method references: [scikit-learn CV](https://scikit-learn.org/1.5/modules/cross_validation.html), [LogisticRegression](https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.LogisticRegression.html), [calibration](https://scikit-learn.org/1.5/modules/calibration.html), [XGBoost](https://xgboost.readthedocs.io/en/stable/parameter.html), [SHAP paper](https://arxiv.org/abs/1705.07874), [LIME paper](https://arxiv.org/abs/1602.04938). Project measurements are from the supplied CSV and saved experiment outputs.', '']
    text='\n'.join(lines)
    (REPORT/'technical_report.md').write_text(text,encoding='utf-8')
    return text


def presentation(data,c):
    d=renderer.Deck()
    def page(title,subtitle='',dark=False):
        return d.page(title,subtitle,source='Development CV evidence • historical test excluded • docs/portfolio-audit.md',dark=dark)
    p=page('Credit modeling: defensible evidence','A reproducible portfolio audit of logistic regression and XGBoost',True)
    d.text(p,'Signal, uncertainty and failure analysis',55,185,850,30,'#FFFFFF',True)
    d.text(p,'Data audit → controlled ablations → nested development CV → calibration → interpretation',55,253,845,23,'#63D0D0')
    d.text(p,'Historical test repeatedly inspected; excluded from every new experiment.',55,385,850,18,'#FFFFFF')
    p=page('The outcome is a stronger study, not a score claim','Both families have modest development signal. The evidence also shows overfitting and costly operating points.')
    a=row(data,'selector_glassbox');b=row(data,'selector_blackbox');dummy=row(data,'dummy')
    d.cards(p,[('Glassbox selector AP',f'{a.ap_mean:.3f}',f'Fold SD {a.ap_sd:.3f}',TEAL),('Blackbox selector AP',f'{b.ap_mean:.3f}',f'Fold SD {b.ap_sd:.3f}',GOLD),('Dummy AP',f'{dummy.ap_mean:.3f}','Natural-prevalence reference',NAVY)])
    d.bullets(p,['Outlier handling matters; additional feature removal is mixed.',
        'Separate calibration repairs probability levels. Neither family has proven universal superiority.'],y=327,size=20,gap=18)
    p=page('Data provenance remains the main deployment blocker','Binary labels are observed; a default horizon, decision dates and borrower identities are not documented.')
    d.table(p,['Verified','Unresolved'],[
        ['3,000 rows / 183 defaults','Default event definition and horizon'],
        ['218 configured raw predictors','Feature availability at decision time'],
        ['No exact duplicate records','Repeated borrowers cannot be excluded'],
        ['Historical 600-row exclusion frozen','No new external or temporal cohort'],
        ['2,400 development rows / 146 defaults','No verified timestamps for temporal validation']],widths=[.49,.51],rowh=49,size=14)
    p=page('The audit found concrete quality problems','These are measured problems; near-zero denominators and post-outcome leakage remain hypotheses.')
    d.image(p,c['quality'])
    p=page('Redundancy is real, but removal needs evidence','Eight exact duplicate field pairs; 66 numerical pairs with |Spearman rho| ≥ .98 on development data.')
    d.bullets(p,['Example exact duplicates: wsk_ROS / wsk_rent_sprzedazy; wsk_koszty_przychody / wsk_poziom_kosztow_operacyjnych.',
        'Some ratios reach about 10^17; 1,014 infinite cells occur in the supplied full CSV.',
        'Two coincident four-field profiles are not verified borrower IDs. No complete borrower-aware split is supported.',
        'Every raw field’s prediction-time availability remains explicitly unverified.'],size=18,gap=14)
    p=page('Historical evidence showed overfit before the test','Old training, validation and selected-CV results are diagnosis only; they do not select new models.')
    d.table(p,['Historical configuration','Train AP','Validation AP','Selected CV AP'],[
        ['Logistic baseline','0.611','0.106','—'],['XGBoost baseline','0.991','0.186','—'],
        ['Balanced L1 selection','0.298','0.140','0.137'],['Heavily regularized XGBoost','0.230','0.172','0.157']],widths=[.43,.19,.19,.19],rowh=54,size=15)
    d.text(p,'The old tree winner reduced variance and fitted signal. Investigate regularization and sampling, not test-tune.',50,453,850,15,GRAY)
    p=page('Repeated nested development CV','Two fixed repeats (42, 43) × three outer folds; three-fold inner AP selection.')
    stages=[('960 rows','Fit and inner search','≈58–59 defaults'),('320 rows','Independent sigmoid','≈19–20 defaults'),('320 rows','Independent F2 threshold','≈19–20 defaults'),('800 rows','Outer development assessment','≈48–49 defaults')]
    for i,(number,title,count) in enumerate(stages):
        x=43+223*i;d.box(p,x,183,205,178,'#F0F5F6');d.text(p,number,x+13,200,179,27,NAVY,True)
        d.text(p,title,x+13,249,178,17,TEAL,True);d.text(p,count,x+13,321,178,12,GRAY)
    d.text(p,'All four roles are disjoint. This costs fitting data, but avoids sharing calibration and threshold labels.',48,411,865,17)
    d.text(p,'Random stratification is a documented limitation, not a substitute for missing borrower/time metadata.',48,459,865,14,GRAY)
    p=page('A small set of explicit experiment families','All ablations use the same outer roles; feature settings are selected inside inner folds.')
    d.table(p,['Question','Controlled comparison'],[
        ['Imbalance','Unweighted vs balanced logistic; identical tree candidates with fold weights'],
        ['Extreme values / scaling','1%/99% train-only clipping; robust vs standard scaling after clipping'],
        ['Missing / rare data','Indicators off; sparse-field cutoffs; rare-level pooling'],
        ['Feature selection','L1; raw-field Spearman redundancy; whole-field supervised top-k'],
        ['Tree search','Matched rate/tree-count paths, moderate complexity controls, frozen references']],widths=[.25,.75],rowh=48,size=13)
    p=page('The full logistic ablation table is retained','Mean ± SD over six outer folds. SD is descriptive; these folds are dependent.')
    d.image(p,c['glassbox_ablations'],y=145,h=341)
    p=page('Trees: moderate controls, matched search paths','No early stopping: tree counts are inner-selected. Weighted and unweighted searches share eight candidates.')
    d.image(p,c['blackbox_ablations'])
    p=page('Paired changes include negative findings','Dots show all six paired outer-fold AP changes; diamonds are means. No favorable seed was selected.')
    d.image(p,c['paired'],y=143,h=344)
    p=page('Feature selection is not automatically an improvement','Whole raw fields preserve one-hot groups; dimensions and active sets vary across training folds.')
    names=['lr_clip','lr_rare','lr_drop_sparse','lr_redundancy','lr_grouped','lr_l1']
    table=[]
    for name in names:
        v=row(data,name,'raw');table.append([name,f'{v.raw_fields_mean:.1f}',f'{v.transformed_features_mean:.1f}',f'{v.active_coefficients_mean:.1f}',f'{v.ap_mean:.3f}'])
    d.table(p,['Recipe','Raw fields','Dimensions','Active coeff.','AP'],table,widths=[.35,.16,.18,.18,.13],rowh=42,size=13)
    p=page('Calibration is a separate probability experiment','A regularized monotone sigmoid fits natural-prevalence rows independently of model fitting and thresholds.')
    d.image(p,c['calibration_changes'])
    p=page('A fitting diagnosis: the intercept was penalized','Adaptive follow-up, reported separately. It uses the same folds and C grids; it is not independent confirmation.')
    d.bullets(p,['liblinear penalizes its synthetic intercept. Very small C can pull even an unweighted score toward 0.5.',
        'L2 follow-up: lbfgs with an unpenalized intercept, with and without clipping.',
        'L1 follow-up: same solver, with intercept_scaling=100 to weaken the intercept penalty.'],size=18,gap=14)
    v=row(data,'lr_lbfgs_clip','raw');base=row(data,'lr_clip','raw')
    d.text(p,f'Clipped L2 raw Brier: liblinear {base.brier_mean:.3f} → lbfgs {v.brier_mean:.3f}.',55,422,850,21,NAVY,True)
    p=page('Primary procedure-level results','Each family chooses its recipe by inner AP only. These are development estimates, not a new test result.')
    table=[]
    for name in ['dummy','selector_glassbox','selector_blackbox']:
        v=row(data,name);table.append([name.replace('selector_',''),f'{v.ap_mean:.3f} ± {v.ap_sd:.3f}',f'{v.roc_auc_mean:.3f}',f'{v.brier_mean:.4f}',f'{v.log_loss_mean:.3f}'])
    d.table(p,['Procedure','AP ± SD','ROC-AUC','Brier','Log loss'],table,widths=[.25,.25,.17,.17,.16],rowh=62,size=15)
    d.text(p,'Sigmoid outputs shown. The adaptive intercept study is separate; it does not replace these planned estimates.',50,442,850,15,GRAY)
    p=page('Ranking curves need scope','Repeat 0 pooled outer-development scores. Pooled AP and mean fold AP are different summaries.')
    d.image(p,c['ranking'])
    p=page('Probability reliability remains uncertain','Repeat 0 pooled outer-development predictions; the fitted calibration subsets contain only ≈20 defaults each.')
    d.image(p,c['reliability'])
    p=page('F2 thresholds still create substantial workload','F2 is a stated recall preference, not a monetary cost model. Values are outer-fold means.')
    table=[]
    for name in ['dummy','selector_glassbox','selector_blackbox','lr_lbfgs_clip']:
        v=row(data,name);table.append([name.replace('selector_',''),f'{v.precision_mean:.1%}',f'{v.recall_mean:.1%}',f'{v.false_positives_mean:.1f}',f'{v.flagged_fraction_mean:.1%}'])
    d.table(p,['Procedure','Precision','Recall','FP / 800','Flagged'],table,widths=[.35,.17,.16,.16,.16],rowh=57,size=14)
    d.text(p,'Last row is the adaptive follow-up. No threshold was selected from the historical test or outer labels.',50,453,850,14,GRAY)
    p=page('There is signal; no ceiling is established','Fixed clipped L2 C=.03, three-fold development CV, 99 shuffled-label controls.')
    d.image(p,c['permutation'])
    p=page('Uncertainty rules out a strong winner claim','Original rows are paired across repeats in the bootstrap; model refitting is not included.')
    comparison=next(v for v in data['bootstrap']['comparisons'] if v['challenger']=='selector_blackbox' and v['reference']=='selector_glassbox')
    d.cards(p,[('Conditional ΔAP interval',f'[{comparison["p025"]:+.3f}, {comparison["p975"]:+.3f}]','Tree selector minus logistic selector',NAVY)],y=165)
    d.bullets(p,['Fold SD, repeat-level results and paired ablations are all retained.',
        'The interval conditions on fitted predictions and prevalence; unknown borrower clustering is not modeled.',
        'Adaptive experimentation and future distribution shift require a new external / temporal cohort.'],y=322,size=18,gap=14)
    p=page('Both families: strengths and costs','Interpretability does not compensate for poor probability estimates or incomplete data provenance.')
    d.table(p,['','Logistic regression','XGBoost'],[
        ['Strength','Signed additive margins; sparse variants','Nonlinearities and interactions'],
        ['Weakness','Representation and extreme-value sensitivity','Overfit and tuning complexity'],
        ['Probability issue','Weighting and intercept penalization','Small calibration sets; score shrinkage'],
        ['Interpretation','Exact native terms; Linear SHAP','Gain has no direction; Tree SHAP'],
        ['Observed limit','No clear family-level AP win','Low LIME fidelity in the studied model']],widths=[.19,.405,.405],rowh=49,size=13)
    p=page('Expanded interpretation, without favorable-case selection','Predeclared outer repeat 0/fold 0; recipe chosen by inner AP. No historical test explanations are added.')
    d.cards(p,[('SHAP rows','800 / family','All held-out rows in that outer fold',TEAL),('Success / error cases','32 / family','8 random TP, FP, FN, TN cases',NAVY),('LIME fits','192 total','3 seeds × 32 cases × 2 families',GOLD)])
    d.bullets(p,['Three fit-only SHAP backgrounds test reference sensitivity.',
        'Native/SHAP explain base log odds. LIME approximates base probability; calibrated decisions are separate.'],y=330,size=19,gap=17)
    for family in ['glassbox','blackbox']:
        p=page(f'Native {family}: actual fitted output','Coefficient signs describe the base logistic margin; tree gain is unsigned training split importance.')
        d.image(p,c[family+'_native'])
    for family in ['glassbox','blackbox']:
        p=page(f'Global SHAP: {family}, 800 development rows','Mean absolute base-log-odds attribution averaged across three training backgrounds.')
        d.image(p,c[family+'_shap'])
    for family in ['glassbox','blackbox']:
        case=data['cases'].query('family == @family and cell == "FN"').iloc[0]
        p=page(f'Error analysis: {family}, a missed default','First randomly sampled FN in the fixed development fold; signed effects explain the base classifier.')
        d.image(p,c[family+'_local'],y=147,h=288)
        d.text(p,f'Row {int(case.row_index)}: actual 1; base {case.raw_probability:.4f} → sigmoid {case.calibrated_probability:.4f} < t {case.threshold:.4f}.',50,453,850,16,NAVY,True)
    p=page('Correlated fields can redistribute credit','Fit-only correlation groups sum signed SHAP terms before taking mean absolute magnitude.')
    d.image(p,c['glassbox_groups'])
    p=page('Explanation stability has several meanings','Background stability does not establish retraining stability, causal validity or subgroup fairness.')
    table=[]
    for family,values in data['interpretation']['families'].items():
        bg=values['background_stability'];lm=values['lime_summary']
        table.append([family,f'{np.mean([v["top10_jaccard"] for v in bg]):.3f}',f'{np.mean([v["global_rank_spearman"] for v in bg]):.3f}',f'{lm["selected15_jaccard_mean"]:.3f}'])
    d.table(p,['Family','SHAP top-10 Jaccard','SHAP rank correlation','LIME selected-15 Jaccard'],table,widths=[.18,.28,.27,.27],rowh=70,size=14)
    d.text(p,'Active/retained raw-field Jaccard across all six fitted folds is also exported for every recipe.',50,427,850,17,GRAY)
    p=page('LIME fidelity is weak for this tree model','32 cases/family × 3 seeds × 2,000 perturbations; synthetic-neighborhood R² is not held-out faithfulness.')
    d.image(p,c['lime'])
    p=page('Successes and errors are all retained','Balanced confusion-cell sampling is illustrative; it is not a population-weighted case distribution.')
    grouped=data['cases'].groupby(['family','cell']).agg(cases=('row_index','size'),raw=('raw_probability','median'),cal=('calibrated_probability','median')).reset_index()
    table=[[r.family,r.cell,str(r.cases),f'{r.raw:.3f}',f'{r.cal:.3f}'] for _,r in grouped.iterrows()]
    d.table(p,['Family','Cell','Cases','Median base','Median sigmoid'],table,widths=[.23,.14,.14,.24,.25],rowh=35,size=12)
    p=page('Reproducibility is part of the result','Saved-model predictions and partition role boundaries were checked independently.')
    d.bullets(p,['126 saved base models: assessment predictions and metrics recomputed after loading.',
        'Automated checks cover outer-label isolation, frozen preprocessing and portable historical exclusions.',
        'Refit checks are distinct from saved-model checks; inner L1 search scores vary slightly between runs.',
        'The experiment plan, fold tables, runtimes, environment and validation records support independent review.'],size=19,gap=15)
    p=page('What would make this deployment evidence?','Implemented: rigorous local development study. Proposed: external validation and operational design.')
    d.bullets(p,['Obtain borrower identifiers, dates, target horizon and verified prediction-time feature lineage.',
        'Use appropriate group/time development splits, then assess a genuinely new cohort.',
        'Define error costs, review capacity, subgroup checks and monitoring ownership.',
        'Validate recalibration and explanation stability under refitting and distribution shift.'],size=19,gap=16)
    p=page('Evidence and methods','Technical report, theory guide and compact machine-readable evidence accompany this presentation.')
    d.bullets(p,['docs/portfolio-audit.md and docs/evidence/portfolio_v1: complete evidence and limitations.',
        'config/portfolio.yaml: planned study; portfolio_intercept.yaml: labeled adaptive follow-up.',
        'Primary sources: scikit-learn CV/scoring/calibration; XGBoost; SHAP (Lundberg & Lee); LIME (Ribeiro et al.).',
        'Presentation: reports/portfolio_v1. Reproduction: docs/reproduction.md. Theory source: tools/portfolio_theory.md.'],size=18,gap=16)
    return d


def pdf_styles():
    return dict(body=ParagraphStyle('pbody',fontName='ReportSans',fontSize=10.2,leading=15,spaceAfter=9,textColor=colors.HexColor(NAVY)),
        small=ParagraphStyle('psmall',fontName='ReportSans',fontSize=8,leading=11,spaceAfter=6,textColor=colors.HexColor(GRAY)),
        title=ParagraphStyle('ptitle',fontName='ReportSans-Bold',fontSize=24,leading=31,spaceAfter=16,textColor=colors.HexColor(NAVY)),
        heading=ParagraphStyle('pheading',fontName='ReportSans-Bold',fontSize=17,leading=23,spaceBefore=14,spaceAfter=12,textColor=colors.HexColor(NAVY)))


def markup(text):
    text=escape(text)
    text=re.sub(r'\*\*(.*?)\*\*',r'<b>\1</b>',text)
    text=re.sub(r'`([^`]+)`',r'<font color="#087F8C">\1</font>',text)
    text=re.sub(r'\[([^]]+)\]\((https?[^)]+)\)',r'<a href="\2" color="#087F8C">\1</a>',text)
    return text


def footer(canvas,doc):
    width,height=doc.pagesize
    canvas.setFont('ReportSans',8);canvas.setFillColor(colors.HexColor(GRAY))
    canvas.drawString(43,28,'DEVELOPMENT EVIDENCE • HISTORICAL TEST EXCLUDED • 16 OCT 2026')
    canvas.drawRightString(width-43,28,str(doc.page))


def technical_pdf(text):
    styles=pdf_styles();styles['body'].fontSize=9.2;styles['body'].leading=13.6
    blocks=[];lines=text.splitlines();index=0
    while index<len(lines):
        line=lines[index]
        if line.startswith('|'):
            table=[]
            while index<len(lines) and lines[index].startswith('|'):
                values=[v.strip() for v in lines[index].strip('|').split('|')]
                if not all(set(v)<=set('-: ') for v in values):table.append(values)
                index+=1
            count=len(table[0]);width=750
            widths=[140]+[(width-140)/(count-1)]*(count-1) if count>1 else [width]
            t=Table([[Paragraph(markup(v),styles['small']) for v in row] for row in table],colWidths=widths,repeatRows=1)
            t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#EAF5F5')),('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
            blocks += [t,Spacer(1,12)];continue
        if line.startswith('# '):blocks.append(Paragraph(markup(line[2:]),styles['title']))
        elif line.startswith('## '):blocks.append(Paragraph(markup(line[3:]),styles['heading']))
        elif line.startswith('### '):blocks.append(Paragraph(markup(line[4:]),styles['heading']))
        elif line.strip():blocks.append(Paragraph(markup(line),styles['body']))
        index+=1
    doc=SimpleDocTemplate(str(REPORT/'technical_report.pdf'),pagesize=(A4[1],A4[0]),
        leftMargin=43,rightMargin=43,topMargin=40,bottomMargin=50,title='Portfolio audit: complete development evidence')
    doc.build(blocks,onFirstPage=footer,onLaterPages=footer)


def theory_pdf(data,c):
    styles=pdf_styles();source=(Path(__file__).parent/'portfolio_theory.md').read_text(encoding='utf-8')
    chapters=[]
    for block in re.split(r'^## ',source,flags=re.M)[1:]:
        title,body=block.split('\n',1);chapters.append((title,body))
    story=[Spacer(1,65),Paragraph('Credit-model theory and the evidence it supports',styles['title']),
        Paragraph('Companion to the reproducible portfolio audit • 16 October 2026',styles['body']),Spacer(1,25),
        Paragraph('Current scope: repeated nested development CV, controlled preprocessing/selection, independent calibration and thresholds, expanded explanation diagnostics. The already inspected historical test is excluded. No new external cohort or deployment readiness is claimed.',styles['body']),
        Paragraph('The guide separates model theory, implemented experiment design, observed development results and unresolved questions. The adaptive intercept follow-up is labeled throughout.',styles['body']),PageBreak(),Paragraph('Contents',styles['title'])]
    for n,(title,_) in enumerate(chapters,1):
        story.append(Paragraph(f'<a href="#theory_{n}" color="{TEAL}">{escape(title)}</a>',styles['small']))
    story.append(PageBreak())
    for n,(title,body) in enumerate(chapters,1):
        story.append(Paragraph(f'<a name="theory_{n}"/>'+escape(title),styles['heading']))
        for paragraph in re.split(r'\n\s*\n',body.strip()):
            story.append(Paragraph(markup(' '.join(paragraph.splitlines())),styles['body']))
        story.append(PageBreak())
    story.append(Paragraph('Current numerical evidence',styles['title']))
    for name in ['dummy','selector_glassbox','selector_blackbox','lr_lbfgs_clip']:
        v=row(data,name);label='adaptive follow-up' if name.startswith('lr_lbfgs') else 'planned procedure'
        story.append(Paragraph(f'<b>{name}</b> ({label}, sigmoid): mean outer AP {v.ap_mean:.4f} ± SD {v.ap_sd:.4f}; ROC-AUC {v.roc_auc_mean:.4f}; Brier {v.brier_mean:.5f}; log loss {v.log_loss_mean:.4f}; precision {v.precision_mean:.1%}; recall {v.recall_mean:.1%}; mean false positives {v.false_positives_mean:.1f} / 800; flagged {v.flagged_fraction_mean:.1%}.',styles['body']))
    story.append(Paragraph('These are descriptive development estimates from the specified training/calibration/threshold budget. The family-selector rows account for inner recipe selection; individual ablation rankings and the intercept follow-up do not become external validation merely because they are reproducible.',styles['body']))
    story.append(PageBreak())
    for key,title in [('calibration_changes','Probability ablation: raw versus independent sigmoid'),('paired','Controlled paired AP changes'),
        ('glassbox_shap','Glassbox SHAP over all 800 held-out development rows'),('blackbox_shap','Blackbox SHAP over all 800 held-out development rows'),
        ('lime','LIME fidelity across successes and errors'),('glassbox_local','A logistic error: local base-log-odds explanation'),
        ('blackbox_local','A tree-model error: local base-log-odds explanation')]:
        story.append(Paragraph(title,styles['heading']))
        with PILImage.open(c[key]) as image:ratio=image.height/image.width
        story.append(Image(str(c[key]),width=475,height=475*ratio))
        story.append(Paragraph('Recorded development outputs; historical test rows excluded. Interpretation targets the base classifier, while sigmoid probabilities and decisions are separate. Full case-level weights, scope notes and diagnostics are retained locally and summarized in the technical report.',styles['body']))
        story.append(PageBreak())
    story.append(Paragraph('Primary references and implementation provenance',styles['title']))
    extra=[('R21','Nested and repeated cross-validation','https://scikit-learn.org/1.5/modules/cross_validation.html'),
        ('R22','Univariate ANOVA selection','https://scikit-learn.org/1.5/modules/generated/sklearn.feature_selection.f_classif.html'),
        ('R23','Rare-category grouping with OneHotEncoder','https://scikit-learn.org/1.5/modules/generated/sklearn.preprocessing.OneHotEncoder.html'),
        ('R24','Permutation test score','https://scikit-learn.org/1.5/modules/generated/sklearn.model_selection.permutation_test_score.html')]
    for i,(ref,title,url) in enumerate(SOURCES+extra):
        if i in [12]:story += [PageBreak(),Paragraph('References (continued)',styles['title'])]
        story.append(Paragraph(f'<b>[{ref}] {escape(title)}</b><br/><a href="{escape(url)}" color="{TEAL}">{escape(url)}</a>',styles['small']))
    story.append(Paragraph('Implementation: src/experiments/{data,transforms,recipes,calibration,protocol,intercept_study,interpretation,validation}.py. Complete comparison: docs/portfolio-audit.md. Machine-readable evidence: docs/evidence/portfolio_v1. Local original code/config snapshots: artifacts/*/source_snapshot.',styles['body']))
    doc=SimpleDocTemplate(str(REPORT/'theory_guide.pdf'),pagesize=A4,leftMargin=55,rightMargin=55,topMargin=50,bottomMargin=57,
        title='Theory for the reproducible credit-model portfolio',author='Local research project')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)




def report_readme(slides):
    return f'''# Credit-model results presentation

- [Slide PDF](results_presentation.pdf)
- [Editable PowerPoint](results_presentation.pptx)

The {slides}-slide presentation compares regularized logistic regression and XGBoost,
covering data limitations, nested development evaluation, preprocessing ablations,
calibration, decision thresholds and native, SHAP and LIME outputs.
The previously inspected historical test is excluded from the development study.

Report generation uses completed, validated experiments. It also produces
`technical_report.md`, `technical_report.pdf`, `theory_guide.pdf` and chart assets.
The repository reproduction guide describes inputs, commands and output paths.
'''


def main():
    global RUN,FOLLOW,REPORT,FIGURES
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=RUN);parser.add_argument('--followup-dir',type=Path,default=FOLLOW)
    parser.add_argument('--report-dir',type=Path,default=REPORT)
    args=parser.parse_args();RUN=args.run_dir.resolve();FOLLOW=args.followup_dir.resolve();REPORT=args.report_dir.resolve();FIGURES=REPORT/'figures'
    data=load();renderer.register_fonts();renderer.OUT=REPORT
    figures=charts(data);text=report_markdown(data);technical_pdf(text)
    deck=presentation(data,figures);deck.save();theory_pdf(data,figures)
    (REPORT/'README.md').write_text(report_readme(len(deck.pages)),encoding='utf-8')
    print(data['all_summary'].query('mode == "sigmoid"')[['recipe','ap_mean','ap_sd','brier_mean']].to_string(index=False))
    print(f'Generated {len(deck.pages)} slides and current technical/theory PDFs: {REPORT}')


if __name__=='__main__':main()
