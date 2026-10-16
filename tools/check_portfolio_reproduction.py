"""Cold-refit two complete recipes on all six folds and compare saved predictions."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
import pandas as pd
import yaml
from src.experiments.data import ROOT,load_protocol,write_json
from src.experiments.protocol import run
from src.experiments.validation import validate_run

cfg=load_protocol(ROOT/'config/portfolio.yaml')
cfg['permutation_repetitions']=1
cfg['output']=str(ROOT/'artifacts/portfolio_reproduction_check')
path=ROOT/'artifacts/portfolio_reproduction_config.yaml'
path.write_text(yaml.safe_dump(cfg),encoding='utf-8')
run(path,only=['dummy','lr_clip'])
validate_run(cfg['output'])
expected=pd.read_csv(ROOT/'artifacts/portfolio_v1/development_predictions.csv',float_precision='round_trip')
actual=pd.read_csv(Path(cfg['output'])/'development_predictions.csv',float_precision='round_trip')
expected=expected[expected.recipe.isin(['dummy','lr_clip'])].set_index(['recipe','mode','repeat','fold','row_index']).sort_index()
actual=actual[actual.recipe.isin(['dummy','lr_clip'])].set_index(['recipe','mode','repeat','fold','row_index']).sort_index()
assert expected.index.equals(actual.index)
for column in ['probability','raw_probability','threshold','prediction']:
    np.testing.assert_allclose(expected[column],actual[column],rtol=1e-10,atol=1e-12)
write_json(ROOT/'artifacts/portfolio_v1/reproduction_check.json',dict(passed=True,
    recipes=['dummy','lr_clip'],outer_folds=6,prediction_rows=len(actual),
    scope='Cold candidate searches and refits using the portable tracked exclusion manifest; raw/sigmoid probabilities, thresholds and labels reproduced.',
    no_historical_test_assessment=True))
print('Cold refits reproduced all saved predictions for both recipes on all six folds.')
