"""Presentation narrative and local theory-document assembly."""
from pathlib import Path
import re
import numpy as np
from scipy.special import expit
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image, Table, TableStyle, KeepTogether
from reportlab.lib.pagesizes import A4
from xml.sax.saxutils import escape

NAVY, TEAL, GOLD, RED, GRAY = '#152B3A', '#087F8C', '#B36B00', '#B83B52', '#546774'

SOURCES = [
('R1','Logistic regression: penalties, C and class weights','https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.LogisticRegression.html'),
('R2','XGBoost: A Scalable Tree Boosting System (Chen & Guestrin, 2016)','https://arxiv.org/abs/1603.02754'),
('R3','XGBoost parameter reference','https://xgboost.readthedocs.io/en/release_2.1.0/parameter.html'),
('R4','RandomizedSearchCV and multiple scoring metrics','https://scikit-learn.org/1.5/modules/generated/sklearn.model_selection.RandomizedSearchCV.html'),
('R5','StratifiedKFold','https://scikit-learn.org/1.5/modules/generated/sklearn.model_selection.StratifiedKFold.html'),
('R6','Preprocessing: standardization and encoding','https://scikit-learn.org/1.5/modules/preprocessing.html'),
('R7','Missing-value imputation','https://scikit-learn.org/1.5/modules/impute.html'),
('R8','F-beta score','https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.fbeta_score.html'),
('R9','ROC-AUC','https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.roc_auc_score.html'),
('R10','Average precision','https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.average_precision_score.html'),
('R11','Brier loss','https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.brier_score_loss.html'),
('R12','Log loss','https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.log_loss.html'),
('R13','Probability calibration','https://scikit-learn.org/1.5/modules/calibration.html'),
('R14','A Unified Approach to Interpreting Model Predictions (Lundberg & Lee, 2017)','https://arxiv.org/abs/1705.07874'),
('R15','SHAP LinearExplainer','https://shap.readthedocs.io/en/latest/generated/shap.LinearExplainer.html'),
('R16','SHAP TreeExplainer','https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html'),
('R17','Why Should I Trust You? (Ribeiro, Singh & Guestrin, 2016)','https://arxiv.org/abs/1602.04938'),
('R18','LIME tabular API','https://lime-ml.readthedocs.io/en/latest/lime.html#module-lime.lime_tabular'),
('R19','SciPy two-sample Kolmogorov-Smirnov statistic','https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html'),
('R20','Metrics and scoring overview','https://scikit-learn.org/1.5/modules/model_evaluation.html'),
]


def make_deck(d,r,c):
    gm=r['metrics'][('glassbox','tuned')]['test']; bm=r['metrics'][('blackbox','tuned')]['test']
    row=r['case_row']; i=r['case_position']; actual=int(r['y'].loc[row])
    p=d.page('Interpretable credit models', 'Current results, tradeoffs and explanation outputs',dark=True)
    d.text(p,'A fair comparison needs three views:',55,180,830,27,'#FFFFFF',True)
    d.text(p,'ranking quality  /  default detection  /  probability quality',55,238,830,24,'#63D0D0')
    d.text(p,'Logistic regression and XGBoost • baseline vs tuned • 3,000 records',55,340,830,17,'#FFFFFF')
    d.text(p,'Local research snapshot • calibration disabled • 16 October 2026',55,395,830,15,'#CFDCE4')
    p=d.page('More recall, with substantial tradeoffs','The selected experiments improve detection, but do not improve every objective.')
    d.cards(p,[('Glassbox recall','37.8%','Baseline: 10.8%',TEAL),('Blackbox recall','73.0%','Baseline: 2.7%',GOLD),('Test defaults','37 / 600','Limited evidence for generalization',NAVY)])
    d.bullets(p,['Glassbox AP rises slightly; its weighted probabilities substantially overestimate default risk.',
        'Blackbox AP falls despite higher recall. Both selected thresholds create more false positives.'],y=325,size=19,gap=20)
    p=d.page('A 6.1% default rate changes evaluation','An all-non-default classifier is 93.9% accurate on the full dataset, yet detects no defaults.')
    d.image(p,c['data'])
    p=d.page('One shared data pipeline','Every learned transformation is fitted inside training folds; test labels do not select parameters.')
    stages=[('218 raw\nfeatures','Median/mode\nimputation'),('Training-only\nencoding','Scale numeric values\nfor logistic only'),('574 fitted\nfeatures','Missing indicators\n+ constant removal'),('Classifier\nscore','Validation F2\nthreshold')]
    for j,(title,body) in enumerate(stages):
        x=45+j*223; d.box(p,x,183,204,175,'#F0F5F6'); d.text(p,title,x+14,200,175,23,NAVY,True); d.text(p,body,x+14,282,177,14,GRAY)
    d.text(p,'Same 60 / 20 / 20 split and random seed 42. No post-hoc 4% PD shift; no supervised WoE encoding.',50,406,855,16)
    p=d.page('Two model families, different strengths','Interpretability and nonlinear flexibility are different design choices. [R1–R3]')
    d.table(p,['Dimension','Glassbox: logistic','Blackbox: XGBoost'],[
        ['Function','Additive linear log odds','Sum of nonlinear decision trees'],
        ['Strength','Signed, exact decomposition','Interactions and nonlinear effects'],
        ['Main limitation','Restricted functional form','More complex; stronger overfit controls'],
        ['Native explanation','Coefficients and local terms','Global split-gain importance'],
        ['Post-hoc explanations','Linear SHAP; LIME','Tree SHAP; LIME']],widths=[.20,.40,.40],rowh=48,size=13)
    p=d.page('Fresh searches: class weighting was tested','40 random candidates × 5 stratified training folds per model; select by mean average precision. [R4–R5]')
    d.table(p,['Selected setting','Glassbox','Blackbox'],[
        ['Core parameters','L1 penalty; C = 0.1','250 trees; depth 4; learning rate .03'],
        ['Imbalance handling','Balanced class weights','Unweighted winner; weighted options tested'],
        ['Regularization','max_iter 1,000; tol .001','L2 = 100; gamma = 2; child weight = 10'],
        ['CV AP ± fold SD','0.1370 ± 0.0358','0.1569 ± 0.0176'],
        ['Validation F2 threshold','0.566104','0.056098']],widths=[.23,.34,.43],rowh=48,size=12.5)
    p=d.page('Cross-validation shows variation','Fold standard deviations describe these CV folds; they are not confidence intervals for the selected winner.')
    d.image(p,c['cv'])
    p=d.page('Test results: all four configurations','AP is average precision. Calibration is disabled in every configuration shown.')
    rows=[]
    for n,v,label in [('glassbox','baseline','Glassbox baseline'),('glassbox','tuned','Glassbox tuned'),('blackbox','baseline','Blackbox baseline'),('blackbox','tuned','Blackbox tuned')]:
        m=r['metrics'][(n,v)]['test']; rows.append([label,f'{m["roc_auc"]:.3f}',f'{m["pr_auc"]:.3f}',f'{m["precision"]:.1%}',f'{m["recall"]:.1%}',f'{m["brier_score"]:.3f}'])
    d.table(p,['Configuration','ROC-AUC','AP','Precision','Recall','Brier ↓'],rows,widths=[.28,.14,.12,.16,.14,.16],rowh=52,size=14)
    d.text(p,'Different selected thresholds change precision and recall; ranking metrics compare scores across thresholds.',50,443,850,14,GRAY)
    p=d.page('Ranking did not improve consistently','The original blackbox baseline has the highest test AP and ROC-AUC of these four runs. [R9–R10]')
    d.image(p,c['ranking'])
    p=d.page('Every gain in recall has an operating cost','Shared test set: 563 non-defaults and 37 defaults. Counts, not normalized percentages.')
    d.image(p,c['confusion'],y=158,h=285)
    d.text(p,'Tuned glassbox: 14 detected defaults / 104 false positives. Tuned blackbox: 27 / 242.',50,456,855,17,NAVY,True)
    p=d.page('Thresholds trade precision for recall','Dashed markers were selected on validation. Test curves are descriptive and were not used for selection. [R8]')
    d.image(p,c['thresholds'])
    p=d.page('Weighted glassbox scores are not reliable PDs','Brier: 0.0765 → 0.2238; ECE: '+f'{r["metrics"][("glassbox","baseline")]["test"]["ece"]:.3f} → {gm["ece"]:.3f}. Reliability curves confirm overprediction. [R11–R13]')
    d.image(p,c['calibration'])
    p=d.page('Glassbox: transparent, with clear limits','Selected model: balanced L1 logistic regression; 574 transformed input dimensions.')
    active=int(np.count_nonzero(r['explanations']['glassbox']['native']['coefficients']))
    d.box(p,45,165,420,287,'#EAF5F5'); d.box(p,485,165,430,287,'#FBF1EF')
    d.text(p,'PROS',65,182,380,15,TEAL,True); d.bullets(p,['Exact signed log-odds terms',f'L1 retains {active} nonzero coefficients','Test AP 0.122 → 0.129'],x=65,y=226,w=375,size=19,gap=16)
    d.text(p,'CONS',505,182,385,15,RED,True); d.bullets(p,['No automatic nonlinear interactions','Correlated inputs complicate attribution','Probability calibration deteriorated'],x=505,y=226,w=375,size=19,gap=16)
    p=d.page('Blackbox: flexible, but the baseline ranks better','Selected experiment is strongly regularized XGBoost; class weighting was not selected.')
    d.box(p,45,165,420,287,'#FBF4E8'); d.box(p,485,165,430,287,'#FBF1EF')
    d.text(p,'PROS',65,182,380,15,GOLD,True); d.bullets(p,['Learns nonlinearities and interactions','Exact Tree SHAP for the chosen game','27 / 37 test defaults detected'],x=65,y=226,w=375,size=19,gap=16)
    d.text(p,'CONS',505,182,385,15,RED,True); d.bullets(p,['Test AP 0.193 → 0.115','242 / 563 non-defaults flagged','Gain provides no effect direction'],x=505,y=226,w=375,size=19,gap=16)
    p=d.page('What each explanation actually explains','All outputs concern the base classifier and its transformed features; calibration is excluded.')
    d.table(p,['Method','Output / units','Scope / limitation'],[
        ['Logistic native','Coefficient; x × coefficient, log odds','Exact additive terms; correlated inputs'],
        ['XGBoost gain','Normalized average split gain','Global importance; no direction'],
        ['Linear / Tree SHAP','Signed contributions, log odds','20 test rows; 100 training background rows'],
        ['Supplementary LIME','Local surrogate coefficients','One case; probability approximation']],widths=[.22,.37,.41],rowh=58,size=13)
    d.text(p,'A positive signed effect raises the model score; an explanation is not a causal explanation of default.',50,454,850,14,GRAY)
    p=d.page('Native glassbox: signed coefficients','Numerical terms use training-standardized features; one-hot terms describe category indicators. [R1]')
    d.image(p,c['glassbox_native'])
    p=d.page('Native blackbox: gain importance','Higher gain means more useful training splits, not a higher default probability or a causal effect. [R3]')
    d.image(p,c['blackbox_native'])
    p=d.page('Global SHAP: importance across the explained sample','Mean absolute log-odds attribution; 20 sampled test rows, not the full population. [R14–R16]')
    d.image(p,c['glassbox_global_shap'],x=25,y=157,w=455,h=312)
    d.image(p,c['blackbox_global_shap'],x=480,y=157,w=455,h=312)
    for name in ['glassbox','blackbox']:
        ex=r['explanations'][name]; base=ex['shap']['base_values'][i]; final=float(base+np.asarray(ex['shap']['values'])[i].sum()); probability=ex['base_default_probability'][i]
        p=d.page(f'Local SHAP: {name}, the same test record',f'Row {row}; actual label {actual}. Chosen among saved positive cases by highest blackbox score; illustrative, not representative.')
        d.image(p,c[name+'_local_shap'],y=151,h=290)
        threshold=r['metrics'][(name,'tuned')]['test']['threshold']
        decision='flagged default' if probability>=threshold else 'not flagged: missed default'
        d.text(p,f'Background {base:.3f} + SHAP {final-base:+.3f} = {final:.3f}; score {probability:.3f}, t={threshold:.3f}.',50,447,860,14,NAVY,True)
        d.text(p,f'Decision: {decision}.',50,475,860,12,RED if probability<threshold else TEAL,True)
    p=d.page('Native and SHAP use different reference points','Both reconstruct the same logistic prediction, but their individual terms answer different questions.')
    ex=r['explanations']['glassbox']; native=ex['native']; local=np.asarray(native['local_log_odds_contributions'])[i]
    base=ex['shap']['base_values'][i]; shaps=np.asarray(ex['shap']['values'])[i]
    d.cards(p,[('Native intercept',f'{native["intercept"]:.3f}','Reference: zero transformed inputs',TEAL),('SHAP expected margin',f'{base:.3f}','Reference: training background',NAVY)],y=160)
    d.text(p,f'Native: {native["intercept"]:.3f} + {local.sum():+.3f} = {native["intercept"]+local.sum():.3f} log odds',55,325,850,23,NAVY,True)
    d.text(p,f'SHAP: {base:.3f} + {shaps.sum():+.3f} = {base+shaps.sum():.3f} log odds',55,377,850,23,NAVY,True)
    d.text(p,'Do not add log-odds effects directly to probabilities. Apply the sigmoid to the final margin.',55,443,850,15,GRAY)
    p=d.page('LIME: a supplementary local approximation','One saved test record; 100 background rows; 1,000 perturbations; 15 selected surrogate terms. [R17–R18]')
    d.image(p,c['glassbox_lime'],x=25,y=151,w=455,h=270); d.image(p,c['blackbox_lime'],x=480,y=151,w=455,h=270)
    for j,name in enumerate(['glassbox','blackbox']):
        lime=r['lime'][name]
        d.text(p,f'{name}: model {lime["model_probability"]:.3f}; surrogate {lime["surrogate_local_prediction"]:.3f}; weighted R² {lime["weighted_training_r2"]:.3f}',40+j*460,438,435,12,NAVY,True)
    d.text(p,'R² is in-sample surrogate fit on synthetic points. Coefficients are not exact SHAP-like contributions.',45,477,870,11,GRAY)
    p=d.page('Interpretation requires scope and assumptions','These are model-behavior explanations, with several limits. [R14–R18]')
    d.bullets(p,['Correlated financial variables can redistribute credit between features; raw labels need a data dictionary.',
        'Interventional SHAP changes feature dependence assumptions; background choice affects the explanation.',
        'LIME can create fractional one-hot combinations and implausible financial profiles.',
        'One local case and 20 SHAP rows do not establish stability, fairness or population-wide explanations.'],size=18,gap=14)
    p=d.page('The next experiments follow from the evidence','No model is established as deployment-ready by these results.')
    d.bullets(p,['Calibrate weighted scores using independent development predictions, then select the operating threshold separately.',
        'Define default horizon, borrower grouping, time ordering, decision costs and review capacity.',
        'Use nested or repeated development validation; reserve a new external / temporal cohort for assessment.',
        'Expand explanation sampling and assess stability, subgroup performance, missingness and drift.'],size=18,gap=14)
    p=d.page('Reproducible and entirely local','Presentation figures use exported predictions and explanations. Saved models were loaded without fitting.')
    d.table(p,['Local output','Purpose'],[
        ['results_presentation.pptx / .pdf','Editable slides and a matching PDF'],
        ['theory_guide.pdf','Definitions, equations, worked examples, sources'],
        ['reports/data/feature_outputs.csv','Exact native weights and mean |SHAP| values'],
        ['reports/data/supplementary_lime.json','Case weights and surrogate diagnostics'],
        ['reports/data/manifest.json','Input hashes and report-generation versions']],widths=[.47,.53],rowh=47,size=13)
    p=d.page('Sources and provenance','Full clickable bibliography is in theory_guide.pdf; project artifacts are the source of all numerical results.')
    d.bullets(p,['R1, R4–R13, R20: scikit-learn 1.5.2 documentation; preprocessing, tuning, scoring and calibration.',
        'R2–R3: Chen & Guestrin (2016) and XGBoost parameter documentation.',
        'R14–R16: Lundberg & Lee (2017), SHAP LinearExplainer and TreeExplainer documentation.',
        'R17–R18: Ribeiro, Singh & Guestrin (2016) and LIME tabular documentation.',
        'Local evidence: artifacts/{glassbox,blackbox}, artifacts/selected and artifacts/tuned.'],size=17,gap=14)
    return d


def make_theory(result,charts,out):
    content=(Path(__file__).parent/'theory_content.md').read_text(encoding='utf-8')
    chapters=[]
    for block in re.split(r'^## ',content,flags=re.M)[1:]:
        title,body=block.split('\n',1); chapters.append((title,body.strip()))
    styles={
        'title':ParagraphStyle('title',fontName='ReportSans-Bold',fontSize=27,leading=35,textColor=colors.HexColor(NAVY),spaceAfter=18),
        'heading':ParagraphStyle('heading',fontName='ReportSans-Bold',fontSize=20,leading=26,textColor=colors.HexColor(NAVY),spaceAfter=15),
        'body':ParagraphStyle('body',fontName='ReportSans',fontSize=10.4,leading=15.5,spaceAfter=10,textColor=colors.HexColor(NAVY)),
        'small':ParagraphStyle('small',fontName='ReportSans',fontSize=8.4,leading=12,spaceAfter=8,textColor=colors.HexColor(GRAY)),
        'sub':ParagraphStyle('sub',fontName='ReportSans-Bold',fontSize=11.3,leading=16,spaceBefore=6,spaceAfter=8,textColor=colors.HexColor(TEAL)),
    }
    def markup(text):
        value=escape(text)
        value=re.sub(r'\*\*(.*?)\*\*',r'<b>\1</b>',value)
        for ref,_,_ in SOURCES: value=re.sub(r'\['+ref+r'\]',f'<a href="#ref_{ref}" color="{TEAL}">[{ref}]</a>',value)
        return value
    def para(text,style='body'): return Paragraph(markup(text),styles[style])
    def figure(path,width=475):
        from PIL import Image as PILImage
        with PILImage.open(path) as image:
            ratio=image.height/image.width
        return Image(str(path),width=width,height=width*ratio)
    story=[Spacer(1,70),para('UNDERSTANDING THE CURRENT CREDIT-MODEL EXPERIMENTS','small'),
        para('Theory, evidence and interpretation','title'),
        para('A companion to the logistic-regression and XGBoost results presentation.'),
        Spacer(1,22),para('Snapshot: 16 October 2026. English edition. Local-only output.'),
        para('This guide separates mathematical definitions, the repository’s exact implementation, observed results and proposed next experiments. It is a technical research explanation, not a claim of deployment readiness.'),
        Spacer(1,28),para('Reading path','sub'),
        para('Chapters 1–6: data and probability foundations. Chapters 7–11: the models and training procedure. Chapters 12–18: evaluation, calibration, thresholds and monitoring. Chapters 19–25: interpretation methods and their outputs. Chapters 26–28: evidence, reproducibility and a glossary.'),
        PageBreak(),para('Contents','heading')]
    for index,(title,_) in enumerate(chapters,1):
        story.append(Paragraph(f'<a href="#chapter_{index}" color="{TEAL}">{escape(title)}</a>  •  {index+2}',styles['small']))
    story += [para('Appendices: result table, numerical explanation audit, figures and references.','small'),PageBreak()]
    attachments={'3. Data scope and class imbalance':'data','11. Cross-validation and search uncertainty':'cv',
        '13. ROC-AUC, Gini and KS':'ranking','16. Calibration and weighted scores':'calibration',
        '21. Native tree gain importance':'blackbox_native'}
    for number,(title,body) in enumerate(chapters):
        story.append(Paragraph(f'<a name="chapter_{number+1}"/>'+escape(title),styles['heading']))
        for paragraph in re.split(r'\n\s*\n',body):
            paragraph=paragraph.strip()
            if not paragraph: continue
            if paragraph.startswith('### '): story.append(para(paragraph[4:],'sub'))
            else: story.append(para(' '.join(paragraph.splitlines())))
        if title in attachments:
            path=charts[attachments[title]]
            from PIL import Image as PILImage
            with PILImage.open(path) as im: width,height=im.size
            story += [Spacer(1,5),Image(str(path),width=475,height=475*height/width),para('Actual saved project outputs; see presentation and local data exports for detail.','small')]
        story.append(PageBreak())
    story.append(para('Appendix A. Complete test-result table','heading'))
    headers=['Configuration','ROC-AUC','AP','Precision','Recall','F1','Brier']
    rows=[headers]
    for name in ['glassbox','blackbox']:
        for variant in ['baseline','tuned']:
            m=result['metrics'][(name,variant)]['test']
            rows.append([f'{name} {variant}']+[f'{m[key]:.4f}' for key in ['roc_auc','pr_auc','precision','recall','f1','brier_score']])
    tab=Table([[Paragraph(escape(x),styles['small']) for x in row] for row in rows],colWidths=[116]+[60]*6)
    tab.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#EAF5F5')),('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),9),('LINEBELOW',(0,0),(-1,0),.5,colors.HexColor(TEAL))]))
    story += [tab,Spacer(1,16),para('All configurations use the same 600 test rows. Tuned thresholds were selected by validation F2; baseline thresholds were 0.5. Calibration was disabled. AP is average precision, rather than trapezoidal PR area.'),
        figure(charts['confusion']),
        para('The tuned glassbox has 104 false positives and 23 false negatives; the tuned blackbox has 242 and 10. These count differences are a major operational consequence.'),PageBreak()]
    story.append(para('Appendix B. Explanation numerical audit','heading'))
    i=result['case_position']; row=result['case_row']
    story.append(para(f'Worked case: original DataFrame row index {row}, actual default label {int(result["y"].loc[row])}. It was selected from the saved explanation sample, choosing the positive case with the highest blackbox score. This choice is illustrative, not a random case study or an additional test-selection step.'))
    for name,ex in result['explanations'].items():
        base=float(ex['shap']['base_values'][i]); total=float(np.asarray(ex['shap']['values'])[i].sum()); lime=result['lime'][name]
        story += [para(name.capitalize(),'sub'),para(f'SHAP: expected margin {base:.6f} + attribution sum {total:+.6f} = {base+total:.6f}; sigmoid gives {expit(base+total):.6f}. Exported score: {ex["base_default_probability"][i]:.6f}. Maximum SHAP reconstruction error over all 20 explained rows: {result["checks"][name+"_shap_probability_max_error"]:.3g}.'),
            para(f'Supplementary LIME: true model score {lime["model_probability"]:.6f}; local surrogate estimate {lime["surrogate_local_prediction"]:.6f}; weighted in-sample R² {lime["weighted_training_r2"]:.6f}. This R² describes the sampled perturbation neighborhood, not held-out predictive accuracy or a calibrated probability. It is not a global measure of explanation reliability.')]
    story += [para('Native and SHAP feature values are exported to reports/data/feature_outputs.csv; supplementary LIME values are in reports/data/supplementary_lime.json. Feature prefixes and full machine-readable names are retained there. Plots shorten some labels without changing their numerical values.'),PageBreak()]
    for key,title in [('glassbox_native','Appendix C. Native logistic coefficients'),('glassbox_global_shap','Appendix D. Glassbox global SHAP'),('blackbox_global_shap','Appendix E. Blackbox global SHAP'),('glassbox_local_shap','Appendix F. Glassbox local SHAP'),('blackbox_local_shap','Appendix G. Blackbox local SHAP')]:
        story += [para(title,'heading'),figure(charts[key]),para('The values come from the saved selected-model explanation artifacts. Global SHAP summaries use only 20 test records; local plots use the same illustrative test row. Contributions are in base-model log odds. Any omitted local features are summed into an explicit remainder term.'),PageBreak()]
    story.append(para('References and provenance','heading'))
    story.append(para('Primary documentation and original method papers, checked 16 October 2026. Package versions used by the project remain the pinned local versions; live documentation can describe newer releases. Numerical project claims come from the local artifacts, not the references below.','small'))
    for index,(ref,title,url) in enumerate(SOURCES):
        if index==10:
            story += [PageBreak(),para('References (continued)','heading')]
        story.append(Paragraph(f'<a name="ref_{ref}"/><b>[{ref}] {escape(title)}</b><br/><a href="{escape(url)}" color="{TEAL}">{escape(url)}</a>',styles['small']))
    story += [Spacer(1,15),para('Local implementation sources: src/data/loader.py; src/data/preprocessing.py; src/models/base.py; src/models/glassbox.py; src/models/blackbox.py; src/models/weighted_xgboost.py; src/tuning.py; src/evaluation/metrics.py; src/explainability/explainers.py. Experimental sources: artifacts/selected, artifacts/tuned and the original baseline artifacts.','small'),
        para('Input-file SHA-256 hashes, package versions and numerical audit checks are recorded alongside this PDF in reports/data/manifest.json and report_summary.json. Models and original artifacts were not changed to create this report. The additional LIME outputs were generated locally from the already fitted models.','small')]
    def footer(canvas,doc):
        canvas.setStrokeColor(colors.HexColor('#D9E2E7')); canvas.line(55,46,A4[0]-55,46)
        canvas.setFont('ReportSans',8); canvas.setFillColor(colors.HexColor(GRAY)); canvas.drawString(55,32,'LOCAL RESEARCH SNAPSHOT • 16 OCT 2026')
        canvas.drawRightString(A4[0]-55,32,str(doc.page))
    doc=SimpleDocTemplate(str(out/'theory_guide.pdf'),pagesize=A4,rightMargin=55,leftMargin=55,
        topMargin=52,bottomMargin=62,title='Theory, evidence and interpretation of credit models',author='Local project report')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
