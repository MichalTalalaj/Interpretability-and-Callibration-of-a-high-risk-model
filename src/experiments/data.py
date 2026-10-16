"""Immutable historical exclusions and a development-only feature audit."""
from pathlib import Path
import hashlib
import json
import re
import platform
import importlib.metadata
import joblib
import numpy as np
import pandas as pd
import yaml
from src.config import load_config
from src.data import CreditDataLoader


ROOT=Path(__file__).resolve().parents[2]


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dataset_digest(path):
    """Preserve content/ordering identity across Git's CRLF/LF checkout conversion."""
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n',b'\n')).hexdigest()


def runtime_environment():
    return dict(python=platform.python_version(),packages={name:importlib.metadata.version(name) for name in
        ['numpy','pandas','scikit-learn','scipy','xgboost','joblib','PyYAML']})


def write_json(path,content):
    Path(path).write_text(json.dumps(content,indent=2,allow_nan=False,default=_json_default),encoding='utf-8')


def _json_default(value):
    if isinstance(value,np.generic): return value.item()
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,Path): return str(value)
    raise TypeError(f'Not JSON serializable: {type(value)}')


def load_protocol(path):
    path=Path(path).resolve()
    cfg=yaml.safe_load(path.read_text(encoding='utf-8'))
    for key in ['base_config','historical_test_manifest','output']:
        cfg[key]=str((path.parent/cfg[key]).resolve())
    if 'dataset_identity' in cfg:
        cfg['dataset_identity']=str((path.parent/cfg['dataset_identity']).resolve())
    if not np.isclose(sum(cfg[k] for k in ['fit_fraction','calibration_fraction','threshold_fraction']),1):
        raise ValueError('Development fractions must sum to one')
    if min(cfg[k] for k in ['fit_fraction','calibration_fraction','threshold_fraction'])<=0:
        raise ValueError('All three development partitions must be nonempty')
    return cfg


def development_data(protocol):
    config=load_config(protocol['base_config'])
    identity=json.loads(Path(protocol['dataset_identity']).read_text()) if 'dataset_identity' in protocol else None
    if identity and dataset_digest(config['data']['path'])!=identity['sha256']:
        raise ValueError('Dataset content/order changed: do not reuse historical row exclusions on changed data')
    X,y=CreditDataLoader(config).load()
    excluded=pd.read_csv(protocol['historical_test_manifest'],usecols=['row_index']).row_index.to_numpy()
    if len(set(excluded))!=len(excluded) or not set(excluded)<=set(X.index):
        raise ValueError('Invalid historical exclusion manifest')
    if identity and hashlib.sha256(np.asarray(excluded,dtype=np.int64).tobytes()).hexdigest()!=identity['indices_sha256']:
        raise ValueError('Historical exclusion membership/order differs from the frozen original')
    # Verify exact membership against both saved baseline index manifests; do not read test labels.
    other=ROOT/'artifacts/blackbox/predictions_test.csv'
    if other.exists():
        if set(pd.read_csv(other,usecols=['row_index']).row_index)!=set(excluded):
            raise ValueError('Historical baseline test memberships differ')
    development=X.index[~X.index.isin(excluded)]
    baseline=Path(protocol['historical_test_manifest']).parent
    original_dev=set()
    if all((baseline/f'predictions_{split}.csv').exists() for split in ['train','validation']):
        for split in ['train','validation']:
            original_dev.update(pd.read_csv(baseline/f'predictions_{split}.csv',usecols=['row_index']).row_index)
        if original_dev!=set(development): raise ValueError('Development pool differs from historical train + validation')
    elif identity is None:
        raise ValueError('Need either original split manifests or a frozen dataset/exclusion identity')
    return X.loc[development],y.loc[development],excluded,config


def preservation_snapshot(output):
    files=[]
    for parent in [ROOT/'artifacts',ROOT/'reports']:
        if parent.exists():
            files.extend(p for p in parent.rglob('*') if p.is_file() and not p.is_relative_to(output)
                         and not p.is_relative_to(ROOT/'reports/portfolio_v1'))
    return {str(p.relative_to(ROOT)):digest(p) for p in files}


def check_preservation(snapshot):
    changed=[p for p,h in snapshot.items() if not (ROOT/p).is_file() or digest(ROOT/p)!=h]
    if changed: raise ValueError(f'Historical files changed: {changed}')
    return len(snapshot)


def audit(protocol,X,y,excluded,config,output):
    raw=pd.read_csv(config['data']['path'],encoding=config['data']['encoding'],sep=config['data']['separator'])
    numeric=X.select_dtypes(include=np.number)
    records=[]
    for c in X:
        observed=X[c].dropna(); counts=observed.value_counts(); numeric_type=c in numeric
        record=dict(feature=c,type='numeric' if numeric_type else 'categorical',missing_fraction=float(X[c].isna().mean()),
            unique_observed=int(observed.nunique()),singleton_levels=int((counts==1).sum()),
            modal_observed_fraction=float(counts.iloc[0]/len(observed)) if len(observed) else 1.,
            min=float(observed.min()) if numeric_type and len(observed) else None,
            max=float(observed.max()) if numeric_type and len(observed) else None,
            q01=float(observed.quantile(.01)) if numeric_type and len(observed) else None,
            q50=float(observed.quantile(.5)) if numeric_type and len(observed) else None,
            q99=float(observed.quantile(.99)) if numeric_type and len(observed) else None,
            prediction_time_availability='undocumented; not verified',
            role='configured predictor; field semantics not inferred from its name')
        records.append(record)
    pd.DataFrame(records).to_csv(output/'feature_audit.csv',index=False)
    corr=numeric.corr(method='spearman',min_periods=100)
    pairs=[]
    for i,left in enumerate(corr):
        for j in range(i+1,len(corr)):
            value=corr.iloc[i,j]
            if np.isfinite(value) and abs(value)>=.98:
                right=corr.columns[j]
                pairs.append(dict(left=left,right=right,spearman=float(value),
                                  co_observed=int(X[[left,right]].notna().all(axis=1).sum())))
    pd.DataFrame(pairs,columns=['left','right','spearman','co_observed']).to_csv(output/'correlated_pairs.csv',index=False)
    signatures={}
    for c in X:
        key=hashlib.sha256(pd.util.hash_pandas_object(X[c],index=False).to_numpy().tobytes()).hexdigest()
        signatures.setdefault(key,[]).append(c)
    exact_groups=[columns for columns in signatures.values() if len(columns)>1]
    profile=['wsk_liczba_dni_istnienia','Aktywa','Kapital_wlasny','przychody_sprzedazy']
    profiles=X[profile].duplicated(keep=False) if set(profile)<=set(X) else pd.Series(False,index=X.index)
    possible_ids=[c for c in raw if re.search(r'(^id$|(^|_)id($|_)|nip|regon|krs|borrower|client|customer|date|timestamp|(^|_)rok($|_)|year)',c,re.I)]
    metadata=dict(target='default',observed_values=sorted(raw['default'].unique().tolist()),
        default_definition='not documented beyond binary label',horizon='unknown',observation_date='unknown',
        feature_availability_at_prediction='unverified for every configured field',borrower_key='not found in supplied schema',
        timestamp='not found in supplied schema; age is not a record date',
        repeated_borrowers='cannot establish or exclude without a borrower identifier',name_based_id_time_candidates=possible_ids)
    details=dict(dataset_sha256=digest(config['data']['path']),historical_manifest_sha256=digest(protocol['historical_test_manifest']),
        total_rows=len(raw),development_rows=len(X),development_events=int(y.sum()),excluded_rows=len(excluded),
        raw_columns_with_target=len(raw.columns),configured_predictors=len(X.columns),
        full_exact_duplicate_rows=int(raw.duplicated().sum()),development_duplicate_predictor_rows=int(X.duplicated().sum()),
        development_exact_duplicate_field_groups=exact_groups,
        raw_dropped_field=dict(name='szczegolnaFormaPrawna_Symbol',unique_values=int(raw['szczegolnaFormaPrawna_Symbol'].nunique()),
            status='historically excluded; no new target-derived inclusion choice'),
        development_all_missing=[r['feature'] for r in records if r['missing_fraction']==1],
        development_near_constant_observed=[r['feature'] for r in records if r['modal_observed_fraction']>=.995],
        development_more_than_95_percent_missing=[r['feature'] for r in records if r['missing_fraction']>.95],
        correlated_numeric_pairs_abs_rho_ge_098=len(pairs),
        identical_four_field_profile_rows=int(profiles.sum()),profile_fields=profile,
        profile_caveat='A coincident financial profile is not a verified borrower identifier.',
        raw_numeric_infinite_cells=int(np.isinf(raw.select_dtypes(include=np.number)).sum().sum()),
        metadata=metadata,
        verified=['No exact duplicate records in the supplied full CSV.',
                  'Heavy missingness, high category cardinality and extreme numerical ranges are present.',
                  'Historical test row membership is excluded and retained verbatim.'],
        hypotheses=['Near-zero financial denominators may generate extreme ratios; source formulas are unavailable.',
                    'Repeated borrowers, post-outcome feature leakage and temporal dependence cannot be ruled out.',
                    'Limited signal and label quality are possible explanations; no performance ceiling is established.'])
    write_json(output/'data_audit.json',details)
    required=[ROOT/f'artifacts/{name}/metrics.json' for name in ['glassbox','blackbox']]
    required += [ROOT/f'artifacts/selected/{name}/metrics.json' for name in ['glassbox','blackbox']]
    required += [ROOT/f'artifacts/{parent}blackbox/model.joblib' for parent in ['', 'selected/']]
    if not all(path.exists() for path in required):
        frozen=ROOT/'docs/evidence/portfolio_v1/historical_diagnosis.json'
        if not frozen.exists(): raise FileNotFoundError('Frozen historical audit evidence is required when original artifacts are absent')
        historical=json.loads(frozen.read_text())
        historical['provenance']='Frozen descriptive historical evidence; no historical models fitted or assessed during this reproduction.'
        write_json(output/'historical_diagnosis.json',historical)
        return details
    historical={}
    for name in ['glassbox','blackbox']:
        historical[name]={}
        for version,parent in [('baseline',ROOT/'artifacts'),('selected',ROOT/'artifacts/selected')]:
            historical[name][version]=json.loads((parent/name/'metrics.json').read_text())
        cv=pd.read_csv(ROOT/'artifacts/tuned'/name/'cv_results.csv')
        columns=['mean_train_average_precision','mean_test_average_precision','std_test_average_precision',
                 'mean_train_roc_auc','mean_test_roc_auc']
        historical[name]['previous_selected_cv']={c:float(cv.iloc[0][c]) for c in columns}
        historical[name]['search_candidates']=len(cv)
        historical[name]['candidate_coverage']={c:cv[c].fillna('none').value_counts().to_dict()
                                                for c in cv if c.startswith('param_')}
    for version,parent in [('baseline',ROOT/'artifacts'),('selected',ROOT/'artifacts/selected')]:
        model=joblib.load(parent/'blackbox/model.joblib').pipeline_.named_steps['classifier']
        trees=[json.loads(t) for t in model.get_booster().get_dump(dump_format='json')]
        def leaves(node): return 1 if 'leaf' in node else sum(leaves(child) for child in node['children'])
        counts=[leaves(t) for t in trees]
        historical['blackbox'][version]['tree_structure']=dict(trees=len(trees),constant_trees=sum(c==1 for c in counts),
            average_leaves=float(np.mean(counts)),max_leaves=max(counts))
    write_json(output/'historical_diagnosis.json',historical)
    return details
