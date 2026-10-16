"""Read-only diagnosis using the portable exclusion manifest and frozen history."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
import pandas as pd
from src.config import load_config
from src.data import CreditDataLoader
from src.experiments.data import load_protocol, development_data

root=Path(__file__).resolve().parents[1]
dev,yd,historical,cfg=development_data(load_protocol(root/'config/portfolio.yaml'))
raw=pd.read_csv(cfg['data']['path'],encoding='utf-8-sig')
X,y=CreditDataLoader(cfg).load()
print('Columns:', json.dumps(raw.columns.tolist(),ensure_ascii=True))
print('Shapes:',raw.shape,dev.shape,'development events',int(yd.sum()))
print('Target values:',raw['default'].value_counts(dropna=False).to_dict())
print('Full duplicate raw rows:',int(raw.duplicated().sum()),'duplicate predictors:',int(X.duplicated().sum()))
print('Dev exact duplicate predictors:',int(dev.duplicated().sum()))
print('All missing:',dev.columns[dev.isna().all()].tolist())
print('Constants:',dev.columns[dev.nunique(dropna=False)<=1].tolist())
print('Highest missingness:',dev.isna().mean().sort_values(ascending=False).head(20).to_dict())
print('Categoricals:')
for c in cfg['data']['categorical_features']:
    print(c,'unique',dev[c].nunique(),'top',dev[c].value_counts().head(8).to_dict(),'singleton levels',int((dev[c].value_counts()==1).sum()))
numeric=dev.drop(columns=cfg['data']['categorical_features'])
print('Absolute max top:',numeric.abs().max().sort_values(ascending=False).head(15).to_dict())
print('Potential group / time fields:',[c for c in raw if any(word in c.lower() for word in ['date','data','rok','year','reg','nip','krs','id','czas','time'])])
profile=['wsk_liczba_dni_istnienia','Aktywa','Kapital_wlasny','przychody_sprzedazy']
same=dev[profile].duplicated(keep=False)
print('Matching four-field profiles:',dev.loc[same,profile].to_dict(orient='index'))
for _,group in dev.loc[same].groupby(profile,dropna=False):
    rows=group.index.tolist()
    for left,right in zip(rows,rows[1:]):
        equal=(dev.loc[left].eq(dev.loc[right])|(dev.loc[left].isna()&dev.loc[right].isna()))
        print('Profile pair',left,right,'differing fields',int((~equal).sum()),'labels',int(yd[left]),int(yd[right]))
history=json.loads((root/'docs/evidence/portfolio_v1/historical_diagnosis.json').read_text())
print('Frozen historical results below are exploratory; no new historical-test assessment.')
for name in ['glassbox','blackbox']:
    for variant in ['baseline','selected']:
        metrics=history[name][variant]
        print(name,variant,{s:{k:metrics[s][k] for k in ['roc_auc','pr_auc','brier_score','log_loss','recall']}
                            for s in ['train','validation','test']})
    print(name,'old selected CV row',history[name]['previous_selected_cv'])
