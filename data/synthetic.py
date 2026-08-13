"""
Synthetic market data generators for offline testing.

All generators are seeded for reproducibility. volume is simulated as
lognormal noise (mean 1e6, sigma 0.3) independent of price.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Any


def _make_volume(n: int, rng: np.random.Generator) -> np.ndarray:
    """Simulate daily volume as lognormal noise."""
    return rng.lognormal(mean=13.8, sigma=0.3, size=n)  # ~1e6 median


def gbm_series(
    n: int = 1000,
    mu: float = 0.05,
    sigma: float = 0.20,
    seed: int = 0,
) -> pd.DataFrame:
    """Simulate a Geometric Brownian Motion price series.

    Parameters
    ----------
    n : number of daily bars
    mu : annualized drift
    sigma : annualized volatility
    seed : RNG seed

    Returns
    -------
    DataFrame with columns close, volume, log_return (length n).
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252
    log_rets = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(n)
    close = 100.0 * np.exp(np.cumsum(log_rets))
    volume = _make_volume(n, rng)
    return pd.DataFrame({'close': close, 'volume': volume, 'log_return': log_rets})


def garch_series(
    n: int = 2000,
    omega: float = 1e-6,
    alpha: float = 0.07,
    beta: float = 0.88,
    gamma: float = 0.05,
    seed: int = 1,
) -> pd.DataFrame:
    """Simulate a GJR-GARCH(1,1) series with leverage effect.

    The conditional variance follows:
        h_t = omega + (alpha + gamma * I(z_{t-1}<0)) * r_{t-1}^2 + beta * h_{t-1}
    where I is the indicator function.

    Parameters
    ----------
    n : number of bars
    omega, alpha, beta, gamma : GJR-GARCH parameters
    seed : RNG seed

    Returns
    -------
    DataFrame with columns close, volume, log_return.
    """
    rng = np.random.default_rng(seed)
    h = np.zeros(n)
    r = np.zeros(n)
    h[0] = omega / (1.0 - alpha - 0.5 * gamma - beta + 1e-12)
    z = rng.standard_normal(n)
    r[0] = np.sqrt(h[0]) * z[0]
    for t in range(1, n):
        indicator = 1.0 if r[t - 1] < 0 else 0.0
        h[t] = omega + (alpha + gamma * indicator) * r[t - 1] ** 2 + beta * h[t - 1]
        h[t] = max(h[t], 1e-12)
        r[t] = np.sqrt(h[t]) * z[t]
    close = 100.0 * np.exp(np.cumsum(r))
    volume = _make_volume(n, rng)
    return pd.DataFrame({'close': close, 'volume': volume, 'log_return': r})


def regime_series(
    n: int = 2000,
    params_state0: dict[str, float] | None = None,
    params_state1: dict[str, float] | None = None,
    transition_matrix: list[list[float]] | None = None,
    seed: int = 2,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Simulate a two-state Gaussian HMM regime-switching series.

    Parameters
    ----------
    n : number of bars
    params_state0 : dict with keys 'mu' and 'sigma' for state 0 (bull)
    params_state1 : dict with keys 'mu' and 'sigma' for state 1 (bear)
    transition_matrix : 2x2 row-stochastic transition matrix
    seed : RNG seed

    Returns
    -------
    (DataFrame with close, volume, log_return, state  columns, state array)
    Actually returns (df, states) where states is np.ndarray of 0/1.
    """
    if params_state0 is None:
        params_state0 = {'mu': 0.0005, 'sigma': 0.008}
    if params_state1 is None:
        params_state1 = {'mu': -0.001, 'sigma': 0.020}
    if transition_matrix is None:
        transition_matrix = [[0.97, 0.03], [0.05, 0.95]]
    rng = np.random.default_rng(seed)
    tm = np.array(transition_matrix)
    states = np.zeros(n, dtype=int)
    states[0] = 0
    for t in range(1, n):
        states[t] = rng.choice(2, p=tm[states[t - 1]])
    mus = np.where(states == 0, params_state0['mu'], params_state1['mu'])
    sigmas = np.where(states == 0, params_state0['sigma'], params_state1['sigma'])
    r = mus + sigmas * rng.standard_normal(n)
    close = 100.0 * np.exp(np.cumsum(r))
    volume = _make_volume(n, rng)
    df = pd.DataFrame({'close': close, 'volume': volume, 'log_return': r})
    return df, states


def cointegrated_pair(
    n: int = 1000,
    beta: float = 0.8,
    seed: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate two cointegrated price series.

    Spread = log(A) - beta * log(B) is mean-reverting (OU process).

    Returns
    -------
    (df_a, df_b) each with columns close, volume, log_return.
    """
    rng = np.random.default_rng(seed)
    # Common factor
    common = np.cumsum(rng.standard_normal(n) * 0.01)
    # Mean-reverting spread
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = 0.9 * spread[t - 1] + rng.standard_normal() * 0.005
    log_a = common + spread
    log_b = common / beta + rng.standard_normal(n) * 0.002
    close_a = 100.0 * np.exp(log_a - log_a[0])
    close_b = 100.0 * np.exp(log_b - log_b[0])
    r_a = np.diff(log_a, prepend=log_a[0])
    r_b = np.diff(log_b, prepend=log_b[0])
    vol_a = _make_volume(n, rng)
    vol_b = _make_volume(n, rng)
    df_a = pd.DataFrame({'close': close_a, 'volume': vol_a, 'log_return': r_a})
    df_b = pd.DataFrame({'close': close_b, 'volume': vol_b, 'log_return': r_b})
    return df_a, df_b


def uncorrelated_pair(
    n: int = 1000,
    seed: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate two independent (non-cointegrated) random walks.

    Returns
    -------
    (df_a, df_b) each with columns close, volume, log_return.
    """
    rng = np.random.default_rng(seed)
    r_a = rng.standard_normal(n) * 0.01
    r_b = rng.standard_normal(n) * 0.01
    close_a = 100.0 * np.exp(np.cumsum(r_a))
    close_b = 100.0 * np.exp(np.cumsum(r_b))
    vol_a = _make_volume(n, rng)
    vol_b = _make_volume(n, rng)
    df_a = pd.DataFrame({'close': close_a, 'volume': vol_a, 'log_return': r_a})
    df_b = pd.DataFrame({'close': close_b, 'volume': vol_b, 'log_return': r_b})
    return df_a, df_b
