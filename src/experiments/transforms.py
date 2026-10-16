"""Fold-fitted, group-preserving financial preprocessing ablations."""
import warnings
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import f_classif, VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted


class NonConvergedFit(ValueError):
    """A non-converged candidate must not silently win a search."""


class CheckedLogisticRegression(LogisticRegression):
    def fit(self, X, y, sample_weight=None):
        super().fit(X,y,sample_weight=sample_weight)
        if np.any(self.n_iter_>=self.max_iter):
            raise NonConvergedFit('Logistic candidate did not converge within its declared budget')
        return self


class FinancialFeatures(TransformerMixin, BaseEstimator):
    """Select raw fields, preserving every one-hot group as a single raw field.

    Correlation grouping is a deterministic greedy Spearman representative set.
    Supervised scores are univariate association heuristics, not inferential p values.
    Every statistic is learned by fit; transform uses frozen columns and cutoffs.
    """
    def __init__(self, clip_quantile=None, missing_limit=1.01, correlation_limit=None, top_k=None):
        self.clip_quantile=clip_quantile
        self.missing_limit=missing_limit
        self.correlation_limit=correlation_limit
        self.top_k=top_k

    def fit(self,X,y=None):
        self.feature_names_in_=np.asarray(X.columns,dtype=object)
        self.n_features_in_=len(X.columns)
        missing=X.isna().mean()
        columns=[c for c in X if missing[c]<self.missing_limit]
        if not columns: raise ValueError('Missingness filter removed all fields')
        numeric=[c for c in columns if pd.api.types.is_numeric_dtype(X[c])]
        self.lower_=self.upper_=None
        if self.clip_quantile is not None:
            if not 0<self.clip_quantile<.5: raise ValueError('Clipping quantile must lie in (0, .5)')
            self.lower_=X[numeric].quantile(self.clip_quantile)
            self.upper_=X[numeric].quantile(1-self.clip_quantile)
        frame=self._clip(X[columns].copy())
        self.redundancy_groups_={}
        if self.correlation_limit is not None:
            if not 0<self.correlation_limit<=1: raise ValueError('Invalid correlation limit')
            corr=frame[numeric].corr(method='spearman',min_periods=max(10,min(100,len(X)//4))).abs().fillna(0)
            kept=[]
            for c in sorted(numeric,key=lambda c:(missing[c],c)):
                match=next((k for k in kept if corr.loc[c,k]>=self.correlation_limit),None)
                if match is None: kept.append(c)
                else: self.redundancy_groups_.setdefault(match,[]).append(c)
            columns=[c for c in columns if c not in numeric or c in kept]
        self.association_scores_={}
        if self.top_k is not None:
            if y is None: raise ValueError('Supervised selection requires labels')
            labels=np.asarray(y)
            for c in columns:
                series=frame[c]
                if pd.api.types.is_numeric_dtype(series):
                    values=series.fillna(series.median()).fillna(0).to_numpy(dtype=float)
                    if np.ptp(values)==0: p=1.0
                    else:
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore',RuntimeWarning)
                            _,ps=f_classif(values.reshape(-1,1),labels)
                        p=float(ps[0]) if np.isfinite(ps[0]) else 1.0
                else:
                    values=series.fillna('__MISSING__').astype(str)
                    counts=values.value_counts()
                    values=values.where(values.map(counts)>=10,'__RARE__')
                    contingency=pd.crosstab(values,labels)
                    p=float(chi2_contingency(contingency,correction=False)[1]) if len(contingency)>1 else 1.0
                self.association_scores_[c]=-np.log10(max(p,1e-300))
            order=sorted(columns,key=lambda c:(-self.association_scores_[c],c))[:int(self.top_k)]
            columns=[c for c in columns if c in order]
        self.selected_columns_=columns
        return self

    def _clip(self,frame):
        if self.lower_ is not None:
            columns=[c for c in self.lower_.index if c in frame]
            frame[columns]=frame[columns].clip(lower=self.lower_[columns],upper=self.upper_[columns],axis=1)
        return frame

    def transform(self,X):
        check_is_fitted(self,'selected_columns_')
        missing=set(self.feature_names_in_)-set(X.columns)
        if missing: raise ValueError(f'Missing expected fields: {sorted(missing)}')
        return self._clip(X[self.selected_columns_].copy())

    def get_feature_names_out(self,input_features=None):
        check_is_fitted(self,'selected_columns_')
        return np.asarray(self.selected_columns_,dtype=object)


def preprocessing(scale='standard', indicators=True, min_frequency=None):
    numeric=[('imputer',SimpleImputer(strategy='median',add_indicator=indicators,keep_empty_features=True))]
    if scale=='standard': numeric.append(('scaler',StandardScaler()))
    elif scale=='robust': numeric.append(('scaler',RobustScaler()))
    elif scale!='none': raise ValueError(f'Unknown scale {scale}')
    categorical=Pipeline([
        ('imputer',SimpleImputer(strategy='most_frequent',keep_empty_features=True)),
        ('onehot',OneHotEncoder(handle_unknown='infrequent_if_exist' if min_frequency else 'ignore',
                               min_frequency=min_frequency,sparse_output=False))])
    return Pipeline([
        ('columns',ColumnTransformer([
            ('numeric',Pipeline(numeric),make_column_selector(dtype_include=np.number)),
            ('categorical',categorical,make_column_selector(dtype_exclude=np.number))],verbose_feature_names_out=True)),
        ('variance',VarianceThreshold(0))])


def pipeline(classifier, *, clip=False, scale='standard', indicators=True, min_frequency=None,
             missing_limit=1.01, correlation_limit=None, top_k=None):
    return Pipeline([
        ('features',FinancialFeatures(clip_quantile=.01 if clip else None,missing_limit=missing_limit,
                                     correlation_limit=correlation_limit,top_k=top_k)),
        ('preprocessor',preprocessing(scale,indicators,min_frequency)),
        ('classifier',classifier)])
