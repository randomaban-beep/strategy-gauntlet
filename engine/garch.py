"""
GJR-GARCH(1,1) model fitting and simulation via the arch library.

Returns are rescaled by *100 for numerical stability during fitting;
AIC/BIC are adjusted back to the original scale for comparability.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate.base import ARCHModelResult


@dataclass
class GARCHFit:
    """Result from fitting GJR-GARCH(1,1)."""
    result: ARCHModelResult
    scale: float  # rescaling factor applied before fitting
    aic: float    # AIC on original scale
    bic: float    # BIC on original scale
    alpha: float  # ARCH coefficient
    beta: float   # GARCH coefficient
    gamma: float  # leverage coefficient
    omega: float  # constant
    mu: float = 0.0  # constant mean on original scale


def fit_garch(returns: pd.Series, scale: float = 100.0) -> GARCHFit:
    """Fit a GJR-GARCH(1,1) model to a return series.

    Parameters
    ----------
    returns : log-return series
    scale : multiplicative rescaling applied before fitting (default 100)

    Returns
    -------
    GARCHFit dataclass
    """
    scaled = returns * scale
    model = arch_model(scaled, vol='GARCH', p=1, o=1, q=1, dist='Normal', rescale=False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = model.fit(disp='off', show_warning=False)

    params = res.params
    # AIC/BIC from arch are on the scaled series; adjust log-likelihood by -2*n*log(scale)
    # to get approximate AIC/BIC on the original scale
    n = len(returns)
    ll_correction = n * np.log(scale)  # |d(scaled)/d(original)| = scale
    aic = res.aic + 2 * ll_correction
    bic = res.bic + 2 * ll_correction

    return GARCHFit(
        result=res,
        scale=scale,
        aic=aic,
        bic=bic,
        alpha=float(params.get('alpha[1]', params.get('alpha1', 0.0))),
        beta=float(params.get('beta[1]', params.get('beta1', 0.0))),
        gamma=float(params.get('gamma[1]', params.get('gamma1', 0.0))),
        omega=float(params.get('omega', 0.0)) / (scale ** 2),
        mu=float(params.get('mu', 0.0)) / scale,
    )


def simulate_garch_paths(
    fit: GARCHFit,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate paths from a fitted GJR-GARCH(1,1) model.

    Parameters
    ----------
    fit : GARCHFit from fit_garch
    n_steps : number of steps per path
    n_paths : number of paths
    rng : seeded numpy Generator

    Returns
    -------
    np.ndarray of shape (n_paths, n_steps) with log returns on original scale
    """
    omega = fit.omega * (fit.scale ** 2)  # scaled omega
    alpha = fit.alpha
    beta = fit.beta
    gamma = fit.gamma

    paths = np.zeros((n_paths, n_steps))
    for i in range(n_paths):
        h = np.zeros(n_steps)
        r = np.zeros(n_steps)
        # Initialize variance
        h[0] = omega / max(1.0 - alpha - 0.5 * gamma - beta, 1e-8)
        z = rng.standard_normal(n_steps)
        r[0] = np.sqrt(max(h[0], 1e-12)) * z[0]
        for t in range(1, n_steps):
            indicator = 1.0 if r[t - 1] < 0 else 0.0
            h[t] = omega + (alpha + gamma * indicator) * r[t - 1] ** 2 + beta * h[t - 1]
            h[t] = max(h[t], 1e-12)
            r[t] = np.sqrt(h[t]) * z[t]
        paths[i] = (r / fit.scale) + fit.mu  # back to original scale with mean
    return paths
