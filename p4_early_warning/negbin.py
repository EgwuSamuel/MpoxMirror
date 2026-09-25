"""
Negative-binomial (NB2, log link) regression with the dispersion estimated by
maximum likelihood.

statsmodels' GLM NegativeBinomial family silently fixes alpha = 1. The mpox
counts are far more over-dispersed than that (MLE alpha ~ 50-80), so a fixed
alpha = 1 gives confidence intervals and p-values that are much too narrow.
All NB analyses in the manuscript use this helper instead.
"""
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm


def design(df: pd.DataFrame, predictors) -> pd.DataFrame:
    X = df[predictors].fillna(0).astype(np.float64)
    return sm.add_constant(X, has_constant="add")


def fit_negbin(y, X: pd.DataFrame):
    """Fit NB2 by MLE, starting from the alpha=1 GLM solution (stable start)."""
    y = np.asarray(y, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        glm = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=1.0)).fit(maxiter=200)
        start = np.append(glm.params.values, 1.0)
        res = sm.NegativeBinomial(y, X, loglike_method="nb2").fit(
            start_params=start, method="newton", maxiter=200, disp=0)
        if not res.mle_retvals.get("converged", False):
            res = sm.NegativeBinomial(y, X, loglike_method="nb2").fit(
                start_params=start, method="bfgs", maxiter=2000, disp=0)
    if not res.mle_retvals.get("converged", False):
        raise RuntimeError("NB2 maximum-likelihood fit did not converge")
    return res


def irr_table(res, predictors) -> dict:
    ci = res.conf_int()
    return {p: {"irr": float(np.exp(res.params[p])),
                "lo95": float(np.exp(ci.loc[p, 0])),
                "hi95": float(np.exp(ci.loc[p, 1])),
                "pvalue": float(res.pvalues[p]),
                "significant": bool(res.pvalues[p] < 0.05)}
            for p in predictors if p in res.params.index}


class PredictiveNegBin:
    """
    NB2 predictor for the walk-forward baseline system.

    In the early walk-forward folds (training data <= 2021, ~30 positive
    state-weeks) the NB maximum-likelihood estimate does not exist
    (quasi-separation: coefficients diverge). To use one rule in every fold,
    the baseline is fitted with a minimal ridge penalty (lam = 1e-4 on
    standardised coefficients, intercept unpenalised) and alpha chosen by
    profile likelihood. Where the MLE exists this is numerically ~identical to
    it. Inference (IRRs, CIs, p-values) always uses the unpenalised fit_negbin.
    """
    LAM = 1e-4

    def __init__(self, predictors):
        self.predictors = list(predictors)

    def _z(self, df):
        X = df[self.predictors].fillna(0).astype(np.float64)
        return sm.add_constant((X - self.mu) / self.sd, has_constant="add").values

    def fit(self, df, y):
        from scipy.optimize import minimize_scalar
        X = df[self.predictors].fillna(0).astype(np.float64)
        self.mu, self.sd = X.mean(), X.std().replace(0, 1.0)
        Z, y = self._z(df), np.asarray(y, dtype=np.float64)
        pw = np.r_[0.0, np.ones(Z.shape[1] - 1)] * self.LAM

        def _fit(alpha):
            m = sm.GLM(y, Z, family=sm.families.NegativeBinomial(alpha=alpha))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = m.fit_regularized(alpha=pw, L1_wt=0.0, maxiter=500)
            return r.params, m.family.loglike(y, m.predict(r.params))

        opt = minimize_scalar(lambda la: -_fit(np.exp(la))[1],
                              bounds=(np.log(1e-3), np.log(1e4)), method="bounded")
        self.alpha = float(np.exp(opt.x))
        self.params, _ = _fit(self.alpha)
        return self

    def predict(self, df):
        return np.exp(self._z(df) @ self.params)
