"""Archive the original completed local studies, requiring their ignored artifacts.

This maintenance utility refreshes recorded evidence and is not a clean-checkout
reproduction command. Use the experiment modules in README.md for reproduction.
"""
from pathlib import Path
import hashlib
import json
import shutil
import importlib.metadata
import platform
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
evidence=ROOT/'docs/evidence/portfolio_v1'
evidence.mkdir(parents=True,exist_ok=True)
inventory=sorted(f'{distribution.metadata["Name"]}=={distribution.version}' for distribution in importlib.metadata.distributions())
(evidence/'environment-freeze.txt').write_text(
    f'# Recorded {platform.python_implementation()} {platform.python_version()} / {platform.system()} environment.\n'
    '# Package inventory, including optional reporting dependencies; not a universal hash-locked environment.\n'
    +'\n'.join(inventory)+'\n',encoding='utf-8')
run=ROOT/'artifacts/portfolio_v1'
for filename in ['comparison.csv','fold_results.csv','repeat_results.csv','paired_ablations.csv',
    'conditional_bootstrap.json','permutation_control.json','data_audit.json','feature_audit.csv',
    'correlated_pairs.csv','historical_diagnosis.json','environment.json','experiment_plan.json',
    'completion.json','validation.json','preservation_check.json','reproduction_check.json','feature_selection_stability.csv','model_fit_diagnostics.csv']:
    if (run/filename).exists():shutil.copy2(run/filename,evidence/filename)
followup=ROOT/'artifacts/portfolio_intercept_v1'
for filename in ['comparison.csv','fold_results.csv','repeat_results.csv','completion.json','validation.json','feature_selection_stability.csv','model_fit_diagnostics.csv']:
    if (followup/filename).exists():shutil.copy2(followup/filename,evidence/('intercept_'+filename))
for filename in ['summary.json','cases.csv','lime_diagnostics.csv','lime_stability.csv']:
    if (run/'interpretation'/filename).exists():shutil.copy2(run/'interpretation'/filename,evidence/('interpretation_'+filename))
historical=pd.read_csv(ROOT/'artifacts/glassbox/predictions_test.csv',usecols=['row_index'])
historical.to_csv(ROOT/'config/historical_test_indices.csv',index=False)
dataset=next(ROOT.glob('*.csv'))
(ROOT/'config/dataset_identity.json').write_text(json.dumps(dict(
    dataset_filename=dataset.name,sha256=hashlib.sha256(dataset.read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
    recorded_raw_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
    hash_normalization='Only CRLF-to-LF newline normalization; content, feature and row ordering remain protected.',
    historical_test_rows=len(historical),indices_sha256=hashlib.sha256(historical.row_index.to_numpy().tobytes()).hexdigest(),
    source='Original glassbox predictions_test.csv row_index column, copied verbatim without labels.',
    interpretation='Already inspected historical test; never a new or untouched final assessment.'),indent=2),encoding='utf-8')
for directory in [run,followup]:
    snapshot=directory/'source_snapshot'
    if not snapshot.exists():
        for folder in ['src','config']:
            for path in (ROOT/folder).rglob('*'):
                if path.is_file() and '__pycache__' not in path.parts:
                    destination=snapshot/path.relative_to(ROOT);destination.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(path,destination)
        (snapshot/'requirements.txt').write_bytes((ROOT/'requirements.txt').read_bytes())
print('Frozen tracked evidence and exact historical exclusion membership.')
