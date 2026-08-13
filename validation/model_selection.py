"""
Return-distribution model selection: GBM vs GJR-GARCH vs 2-state HMM.

Diagnostics:
  - Ljung-Box test on standardized residuals and squared residuals
  - Jarque-Bera normality test on standardized residuals
  - AIC / BIC

A model is "trusted" if its residual diagnostics pass at the configured
significance level. If none pass, GBM is returned as fallback (flagged).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox

import config
from engine.garch import GARCHFit, fit_garch
from engine.regime_switching import HMMFit, fit_hmm


@dataclass
class GBMFit:
    """Simple GBM (Gaussian) baseline model."""
    mu: float
    sigma: float
    aic: float
    bic: float


@dataclass
class ModelDiagnostics:
    """Diagnostics for a fitted return model."""
    model_name: str
    aic: float
    bic: float
    lb_pvalue_residuals: float    # Ljung-Box on standardized residuals
    lb_pvalue_sq_residuals: float # Ljung-Box on squared residuals
    jb_pvalue: float              # Jarque-Bera on residuals
    trusted: bool                 # passes all diagnostic checks
    fit: Any                      # the fit object


@dataclass
class ModelSelectionResult:
    """Result of model selection step."""
    diagnostics: list[ModelDiagnostics] = field(default_factory=list)
    trusted_models: dict[str, Any] = field(default_factory=dict)  # name -> fit
    best_model_name: str = 'gbm'
    fallback: bool = False


def _fit_gbm(returns: pd.Series) -> GBMFit:
    """Fit a simple GBM (mean + constant vol) to returns."""
    mu = float(returns.mean())
    sigma = float(returns.std())
    n = len(returns)
    # Log-likelihood for Gaussian
    if sigma <= 0:
        sigma = 1e-8
    ll = -0.5 * n * (np.log(2 * np.pi * sigma ** 2) + 1)
    aic = -2 * ll + 2 * 2  # 2 parameters: mu, sigma
    bic = -2 * ll + 2 * np.log(n)
    return GBMFit(mu=mu, sigma=sigma, aic=aic, bic=bic)


def _gbm_standardized_residuals(returns: pd.Series, fit: GBMFit) -> np.ndarray:
    return ((returns.values - fit.mu) / max(fit.sigma, 1e-8))


def _garch_standardized_residuals(returns: pd.Series, fit: GARCHFit) -> np.ndarray:
    """Extract standardized residuals from the GARCH fit."""
    try:
        resid = fit.result.resid / fit.scale  # back to original scale
        cond_vol = fit.result.conditional_volatility / fit.scale
        std_resid = resid / np.maximum(cond_vol, 1e-10)
        return std_resid.values
    except Exception:
        return _gbm_standardized_residuals(returns, GBMFit(
            mu=float(returns.mean()), sigma=float(returns.std()), aic=0.0, bic=0.0
        ))


def _hmm_standardized_residuals(returns: pd.Series, fit: HMMFit) -> np.ndarray:
    """Standardize returns by the HMM state-conditional mean and std."""
    try:
        X = returns.values.reshape(-1, 1)
        states = fit.model.predict(X)
        means = fit.model.means_.flatten()
        stds = np.sqrt(fit.model.covars_.flatten())
        state_mean = means[states]
        state_std = np.maximum(stds[states], 1e-10)
        return (returns.values - state_mean) / state_std
    except Exception:
        return returns.values


def _run_diagnostics(
    model_name: str,
    std_resid: np.ndarray,
    aic: float,
    bic: float,
    fit: Any,
    sig: float = config.DIAGNOSTICS_SIGNIFICANCE,
    lags: int = config.LJUNG_BOX_LAGS,
) -> ModelDiagnostics:
    """Run Ljung-Box and Jarque-Bera on standardized residuals."""
    resid = pd.Series(std_resid).dropna()
    if len(resid) < lags + 5:
        return ModelDiagnostics(
            model_name=model_name, aic=aic, bic=bic,
            lb_pvalue_residuals=0.0, lb_pvalue_sq_residuals=0.0,
            jb_pvalue=0.0, trusted=False, fit=fit,
        )

    try:
        lb_res = acorr_ljungbox(resid, lags=[lags], return_df=True)
        lb_p = float(lb_res['lb_pvalue'].iloc[0])
    except Exception:
        lb_p = 0.0

    try:
        lb_sq = acorr_ljungbox(resid ** 2, lags=[lags], return_df=True)
        lb_sq_p = float(lb_sq['lb_pvalue'].iloc[0])
    except Exception:
        lb_sq_p = 0.0

    try:
        _, jb_p = stats.jarque_bera(resid)
    except Exception:
        jb_p = 0.0

    # Model is trusted if residuals are white noise (LB p > sig)
    # AND squared residuals show no remaining autocorrelation (LB_sq p > sig)
    trusted = lb_p > sig and lb_sq_p > sig

    return ModelDiagnostics(
        model_name=model_name,
        aic=aic,
        bic=bic,
        lb_pvalue_residuals=lb_p,
        lb_pvalue_sq_residuals=lb_sq_p,
        jb_pvalue=jb_p,
        trusted=trusted,
        fit=fit,
    )


def select_model(
    returns: pd.Series,
    min_garch_history: int = config.MIN_GARCH_HISTORY,
    min_hmm_history: int = config.MIN_HMM_HISTORY,
    sig: float = config.DIAGNOSTICS_SIGNIFICANCE,
) -> ModelSelectionResult:
    """Fit and compare GBM, GJR-GARCH, and 2-state HMM.

    Models that pass diagnostic checks are marked as trusted.
    If none pass, GBM is returned as fallback.

    Parameters
    ----------
    returns : log-return series
    min_garch_history : minimum bars needed to fit GARCH
    min_hmm_history : minimum bars needed to fit HMM
    sig : significance level for diagnostic tests

    Returns
    -------
    ModelSelectionResult
    """
    result = ModelSelectionResult()
    n = len(returns)

    # --- GBM (always fit) ---
    gbm_fit = _fit_gbm(returns)
    gbm_resid = _gbm_standardized_residuals(returns, gbm_fit)
    gbm_diag = _run_diagnostics('gbm', gbm_resid, gbm_fit.aic, gbm_fit.bic, gbm_fit, sig)
    result.diagnostics.append(gbm_diag)

    # --- GJR-GARCH ---
    if n >= min_garch_history:
        try:
            garch_fit = fit_garch(returns)
            garch_resid = _garch_standardized_residuals(returns, garch_fit)
            garch_diag = _run_diagnostics(
                'garch', garch_resid, garch_fit.aic, garch_fit.bic, garch_fit, sig
            )
            result.diagnostics.append(garch_diag)
        except Exception:
            pass

    # --- HMM ---
    if n >= min_hmm_history:
        try:
            hmm_fit = fit_hmm(returns)
            hmm_resid = _hmm_standardized_residuals(returns, hmm_fit)
            hmm_diag = _run_diagnostics(
                'hmm', hmm_resid, hmm_fit.aic, hmm_fit.bic, hmm_fit, sig
            )
            result.diagnostics.append(hmm_diag)
        except Exception:
            pass

    # Collect trusted models
    for diag in result.diagnostics:
        if diag.trusted:
            result.trusted_models[diag.model_name] = diag.fit

    if result.trusted_models:
        # Best by AIC
        result.best_model_name = min(
            result.trusted_models,
            key=lambda k: next(d.aic for d in result.diagnostics if d.model_name == k)
        )
        result.fallback = False
    else:
        # Fallback: GBM
        result.trusted_models['gbm'] = gbm_fit
        result.best_model_name = 'gbm'
        result.fallback = True

    return result
