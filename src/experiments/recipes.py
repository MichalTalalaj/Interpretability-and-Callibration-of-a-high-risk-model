"""Declared ablations and matched candidate spaces; no historical-test feedback."""
from dataclasses import dataclass
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier
from src.models.weighted_xgboost import FoldWeightedXGBClassifier
from .transforms import pipeline, CheckedLogisticRegression


@dataclass
class Recipe:
    name: str
    family: str
    hypothesis: str
    estimator: object
    grid: object
    selectable: bool=True


def recipes(seed=42):
    out=[]
    def lr(name,hypothesis,*,grid=None,penalty='l2',class_weight=None,selectable=True,**pre):
        clf=CheckedLogisticRegression(solver='liblinear',penalty=penalty,class_weight=class_weight,
            C=.03,max_iter=1000,tol=.001,random_state=seed)
        settings={'classifier__C':[.0001,.003,.03,.3]}
        if grid: settings.update(grid)
        out.append(Recipe(name,'glassbox',hypothesis,pipeline(clf,**pre),settings,selectable))
    out.append(Recipe('dummy','dummy','Natural-prevalence constant-score benchmark.',
                      pipeline(DummyClassifier(strategy='prior'),scale='none'),{},False))
    lr('lr_standard','Reference: current all-field preprocessing with stronger tuned L2.')
    lr('lr_weighted','Balanced versus unweighted L2, with identical candidates.',class_weight='balanced')
    lr('lr_clip','Train-fitted 1%/99% clipping versus standard L2.',clip=True)
    lr('lr_robust','Robust versus standard scaling after identical clipping.',clip=True,scale='robust')
    lr('lr_rare','Pool rare categories after clipping; keep raw one-hot groups intact.',
       clip=True,min_frequency=10,grid={'preprocessor__columns__categorical__onehot__min_frequency':[5,20]})
    lr('lr_no_indicators','Missingness indicators versus no indicators, after clipping.',clip=True,indicators=False)
    lr('lr_drop_sparse','Drop almost-unobserved raw fields; missingness cutoff selected inside CV.',
       clip=True,grid={'features__missing_limit':[.95,.99]})
    lr('lr_redundancy','Greedy Spearman redundancy reduction; cutoff selected inside CV.',
       clip=True,grid={'features__correlation_limit':[.90,.98]})
    lr('lr_grouped','Select complete raw fields by univariate association; field count selected inside CV.',
       clip=True,grid={'features__top_k':[25,75,150]})
    lr('lr_l1','Embedded L1 versus L2 after identical clipping.',clip=True,penalty='l1',
       grid={'classifier__C':[.003,.01,.03,.1]})
    lr('lr_legacy','Frozen original C=2 unweighted logistic reference under the new protocol.',
       grid={'classifier__C':[2.]},selectable=False)
    lr('lr_historical_selected','Frozen prior selected balanced L1 C=.1 reference, new development protocol.',
       penalty='l1',class_weight='balanced',grid={'classifier__C':[.1]},selectable=False)
    common=dict(objective='binary:logistic',eval_metric='logloss',tree_method='hist',n_jobs=1,
                random_state=seed,subsample=.8,colsample_bytree=.8)
    legacy=XGBClassifier(**common,n_estimators=300,max_depth=4,learning_rate=.05,reg_lambda=1)
    out.append(Recipe('xgb_legacy','blackbox','Frozen original tree configuration; quantify overfitting on development.',
                      pipeline(legacy,scale='none'),{},False))
    old=XGBClassifier(**{**common,'subsample':1.,'colsample_bytree':1.},n_estimators=250,max_depth=4,
                      learning_rate=.03,min_child_weight=10,gamma=2,reg_lambda=100,reg_alpha=0)
    out.append(Recipe('xgb_historical_selected','blackbox','Frozen heavily regularized prior winner; investigate underfitting.',
                      pipeline(old,scale='none'),{},False))
    candidates=[]
    for trees,rate in [(150,.08),(300,.04),(600,.02)]:
        for depth in [2,3]:
            candidates.append({'classifier__n_estimators':[trees],'classifier__learning_rate':[rate],
                'classifier__max_depth':[depth],'classifier__min_child_weight':[1],
                'classifier__reg_lambda':[5.],'classifier__gamma':[0.]})
    for trees,rate in [(300,.04),(600,.02)]:
        candidates.append({'classifier__n_estimators':[trees],'classifier__learning_rate':[rate],
            'classifier__max_depth':[3],'classifier__min_child_weight':[5],
            'classifier__reg_lambda':[20.],'classifier__gamma':[.2]})
    for name,multiplier,rare,hypothesis in [
        ('xgb_search',None,None,'Matched learning-rate/tree-count paths and moderate complexity controls.'),
        ('xgb_weighted',1.,None,'Full fold-local class weighting with the identical eight tree candidates.'),
        ('xgb_rare',None,10,'Rare-level pooling with the identical eight unweighted tree candidates.')]:
        clf=FoldWeightedXGBClassifier(**common,balance_multiplier=multiplier,reg_alpha=0)
        out.append(Recipe(name,'blackbox',hypothesis,pipeline(clf,scale='none',min_frequency=rare),candidates))
    return out
