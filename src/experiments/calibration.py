"""Low-complexity monotone sigmoid fitted on an independent natural-prevalence set."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted


def logits(probabilities):
    p=np.asarray(probabilities,dtype=float)
    if p.ndim!=1 or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('Expected finite one-dimensional probabilities')
    p=np.clip(p,1e-12,1-1e-12)
    return np.log(p)-np.log1p(-p)


class MonotoneSigmoid(BaseEstimator):
    """L2-penalized nonnegative slope, unpenalized intercept; sum-loss convention."""
    def __init__(self,C=1.): self.C=C

    def fit(self,probabilities,y):
        z=logits(probabilities); labels=np.asarray(y,dtype=float)
        if labels.shape!=z.shape or set(np.unique(labels))!={0.,1.}: raise ValueError('Both classes required')
        if self.C<=0: raise ValueError('Calibration C must be positive')
        def objective(theta):
            a,b=theta; margin=a*z+b; residual=expit(margin)-labels
            value=np.logaddexp(0,margin).sum()-labels@margin + a*a/(2*self.C)
            gradient=np.array([residual@z+a/self.C,residual.sum()])
            return value,gradient
        prevalence=labels.mean()
        fit=minimize(objective,[1.,np.log(prevalence/(1-prevalence))-z.mean()],jac=True,
                     method='L-BFGS-B',bounds=[(0,None),(None,None)],options={'maxiter':1000,'ftol':1e-12})
        if not fit.success: raise RuntimeError(f'Calibration did not converge: {fit.message}')
        self.slope_,self.intercept_=map(float,fit.x)
        self.rows_,self.events_=len(labels),int(labels.sum())
        return self

    def predict(self,probabilities):
        check_is_fitted(self,'slope_')
        return expit(self.slope_*logits(probabilities)+self.intercept_)
