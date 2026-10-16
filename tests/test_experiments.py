"""Behavioral isolation tests for the portfolio protocol and grouped selectors."""
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from src.experiments.transforms import FinancialFeatures,pipeline,CheckedLogisticRegression
from src.experiments.calibration import MonotoneSigmoid
from src.experiments.recipes import Recipe,recipes
from src.experiments.protocol import partitions,fit_recipe
from src.experiments.data import ROOT,load_protocol,development_data,dataset_digest,audit


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(12)
        self.X=pd.DataFrame({'a':rng.normal(size=240),'category':['a','b','rare']*80})
        self.X['duplicate']=self.X.a
        self.X['sparse']=np.nan
        self.X.loc[:3,'sparse']=100
        self.y=pd.Series(([0]*7+[1])*30)
        self.protocol=dict(outer_seeds=[42],fit_fraction=.6,calibration_fraction=.2,threshold_fraction=.2,
            inner_folds=2,search_jobs=1,sigmoid_C=1,threshold_beta=2)

    def test_selectors_freeze_training_statistics_and_preserve_raw_groups(self):
        selector=FinancialFeatures(clip_quantile=.01,missing_limit=.95,correlation_limit=.98)
        fitted=selector.fit(self.X,self.y)
        self.assertNotIn('sparse',fitted.selected_columns_)
        self.assertEqual(len(set(['a','duplicate'])&set(fitted.selected_columns_)),1)
        self.assertIn('category',fitted.selected_columns_)
        before=joblib.hash(fitted)
        holdout=self.X.copy();holdout['a']=1e25;holdout['category']='unseen'
        transformed=fitted.transform(holdout)
        self.assertEqual(before,joblib.hash(fitted))
        self.assertEqual(list(transformed),fitted.selected_columns_)
        supervised=FinancialFeatures(clip_quantile=.01,top_k=2).fit(self.X,self.y)
        self.assertEqual(len(supervised.selected_columns_),2)
        self.assertEqual(set(supervised.selected_columns_),set(clone(supervised).fit(self.X,self.y).selected_columns_))

    def test_four_roles_are_disjoint_and_reproducible(self):
        p=partitions(self.X,self.y,self.protocol,0,0,np.arange(160),np.arange(160,240))
        again=partitions(self.X,self.y,self.protocol,0,0,np.arange(160),np.arange(160,240))
        self.assertEqual(set.union(*(set(v) for v in p.values())),set(self.X.index))
        for name,indices in p.items():
            np.testing.assert_array_equal(indices,again[name])
            for other,right in p.items():
                if name!=other:self.assertFalse(set(indices)&set(right))

    def test_calibration_monotonicity_and_invalid_inputs(self):
        p=np.linspace(.001,.99,240)
        cal=MonotoneSigmoid().fit(p,self.y)
        predictions=cal.predict(p)
        self.assertTrue(np.all(np.diff(predictions)>=0))
        self.assertTrue(np.isfinite(predictions).all())
        self.assertTrue(((predictions>=0)&(predictions<=1)).all())
        self.assertEqual(cal.events_,30)
        with self.assertRaises(ValueError):MonotoneSigmoid().fit([np.nan,.2],[0,1])

    def test_assessment_labels_cannot_change_fitted_models_calibration_or_thresholds(self):
        p=partitions(self.X,self.y,self.protocol,0,0,np.arange(160),np.arange(160,240))
        recipe=Recipe('test','glassbox','isolation',pipeline(CheckedLogisticRegression(
            solver='liblinear',max_iter=1000,random_state=42),clip=True),{'classifier__C':[.01,.1]})
        changed=self.y.copy();changed.loc[p['assessment']]=1-changed.loc[p['assessment']]
        with tempfile.TemporaryDirectory() as tmp:
            one,first=fit_recipe(recipe,self.X,self.y,p,self.protocol,Path(tmp)/'one',42)
            two,second=fit_recipe(recipe,self.X,changed,p,self.protocol,Path(tmp)/'two',42)
            a=joblib.load(Path(tmp)/'one/model.joblib');b=joblib.load(Path(tmp)/'two/model.joblib')
            self.assertEqual(joblib.hash(a['pipeline']),joblib.hash(b['pipeline']))
            self.assertEqual(joblib.hash(a['calibrator']),joblib.hash(b['calibrator']))
            np.testing.assert_allclose(first.probability,second.probability)
            np.testing.assert_allclose(first.threshold,second.threshold)
            self.assertEqual(one[0]['best_params'],two[0]['best_params'])

    def test_weighted_and_unweighted_xgb_candidates_are_identical(self):
        mapping={r.name:r for r in recipes()}
        self.assertEqual(mapping['xgb_search'].grid,mapping['xgb_weighted'].grid)
        self.assertEqual(mapping['xgb_search'].grid,mapping['xgb_rare'].grid)
        self.assertIsNone(mapping['xgb_search'].estimator.named_steps['classifier'].balance_multiplier)
        self.assertEqual(mapping['xgb_weighted'].estimator.named_steps['classifier'].balance_multiplier,1)

    def test_portable_exclusions_need_no_ignored_baseline_artifacts(self):
        cfg=load_protocol(ROOT/'config/portfolio.yaml')
        with tempfile.TemporaryDirectory() as folder,patch('src.experiments.data.ROOT',Path(folder)):
            X,y,excluded,_=development_data(cfg)
            self.assertEqual((len(X),int(y.sum()),len(excluded)),(2400,146,600))
            self.assertFalse(set(X.index)&set(excluded))
            bad=Path(folder)/'bad_identity.json'
            import json
            identity=json.loads(Path(cfg['dataset_identity']).read_text());identity['sha256']='invalid'
            bad.write_text(json.dumps(identity));cfg['dataset_identity']=str(bad)
            with self.assertRaises(ValueError):development_data(cfg)

    def test_dataset_identity_normalizes_only_line_endings(self):
        with tempfile.TemporaryDirectory() as folder:
            lf=Path(folder)/'lf.csv';crlf=Path(folder)/'crlf.csv';reordered=Path(folder)/'changed.csv'
            lf.write_bytes(b'a,b\n1,2\n3,4\n');crlf.write_bytes(b'a,b\r\n1,2\r\n3,4\r\n')
            reordered.write_bytes(b'a,b\n3,4\n1,2\n')
            self.assertEqual(dataset_digest(lf),dataset_digest(crlf))
            self.assertNotEqual(dataset_digest(lf),dataset_digest(reordered))

    def test_data_audit_can_use_frozen_evidence_without_old_models(self):
        cfg=load_protocol(ROOT/'config/portfolio.yaml')
        X,y,excluded,base=development_data(cfg)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);target=root/'docs/evidence/portfolio_v1';target.mkdir(parents=True)
            shutil.copy2(ROOT/'docs/evidence/portfolio_v1/historical_diagnosis.json',target/'historical_diagnosis.json')
            output=root/'audit';output.mkdir()
            with patch('src.experiments.data.ROOT',root):
                result=audit(cfg,X,y,excluded,base,output)
            self.assertEqual(result['development_events'],146)
            self.assertIn('Frozen descriptive historical evidence',(output/'historical_diagnosis.json').read_text())


if __name__=='__main__':unittest.main()
