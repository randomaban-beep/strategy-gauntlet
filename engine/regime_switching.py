"""
Two-state Gaussian HMM regime-switching model via hmmlearn.

Multiple random restarts are used to avoid local optima; the best
log-likelihood is selected.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from hmmlearn import hmm

import config

logging.getLogger("hmmlearn").setLevel(logging.ERROR)


@dataclass
class HMMFit:
    """Result from fitting a 2-state Gaussian HMM."""
    model: hmm.GaussianHMM
    log_likelihood: float
    aic: float
    bic: float
    means: np.ndarray   # shape (2,)
    stds: np.ndarray    # shape (2,)
    state_means: list[float]
    state_stds: list[float]


def fit_hmm(
    returns: pd.Series,
    n_components: int = 2,
    n_restarts: int = config.HMM_RESTARTS,
    seed: int = config.HMM_SEED,
) -> HMMFit:
    """Fit a 2-state Gaussian HMM with multiple restarts.

    Parameters
    ----------
    returns : log-return series
    n_components : number of hidden states (default 2)
    n_restarts : number of random restarts
    seed : base random seed

    Returns
    -------
    HMMFit dataclass with the best-likelihood model
    """
    X = returns.values.reshape(-1, 1)
    best_model = None
    best_ll = -np.inf

    for restart in range(n_restarts):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model = hmm.GaussianHMM(
                n_components=n_components,
                covariance_type='diag',
                n_iter=200,
                random_state=seed + restart,
                tol=1e-4,
            )
            try:
                model.fit(X)
                ll = model.score(X)
                if ll > best_ll:
                    best_ll = ll
                    best_model = model
            except Exception:
                continue

    if best_model is None:
        raise RuntimeError('HMM fitting failed on all restarts')

    n = len(returns)
    # Number of free parameters: n_components*(n_components-1) transition probs
    # + n_components means + n_components variances + (n_components-1) initial probs
    n_params = n_components ** 2 - 1 + 2 * n_components
    aic = -2 * best_ll * n + 2 * n_params
    bic = -2 * best_ll * n + n_params * np.log(n)

    means = best_model.means_.flatten()
    stds = np.sqrt(best_model.covars_.flatten())
    # Sort by mean (state 0 = higher mean)
    order = np.argsort(means)[::-1]
    means = means[order]
    stds = stds[order]

    return HMMFit(
        model=best_model,
        log_likelihood=best_ll * n,  # total log-likelihood
        aic=aic,
        bic=bic,
        means=means,
        stds=stds,
        state_means=list(means),
        state_stds=list(stds),
    )


def simulate_hmm_paths(
    fit: HMMFit,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate paths from a fitted 2-state Gaussian HMM.

    Parameters
    ----------
    fit : HMMFit from fit_hmm
    n_steps : number of steps per path
    n_paths : number of paths
    rng : seeded numpy Generator

    Returns
    -------
    np.ndarray of shape (n_paths, n_steps) with simulated log returns
    """
    model = fit.model
    n_states = model.n_components
    # Transition matrix
    A = model.transmat_
    means = model.means_.flatten()
    stds = np.sqrt(model.covars_.flatten())
    # Stationary distribution for initial state
    try:
        eigvals, eigvecs = np.linalg.eig(A.T)
        stationary = np.real(eigvecs[:, np.argmax(np.real(eigvals))])
        stationary = np.abs(stationary) / np.abs(stationary).sum()
    except Exception:
        stationary = np.ones(n_states) / n_states

    paths = np.zeros((n_paths, n_steps))
    for i in range(n_paths):
        state = rng.choice(n_states, p=stationary)
        for t in range(n_steps):
            paths[i, t] = means[state] + stds[state] * rng.standard_normal()
            state = rng.choice(n_states, p=A[state])
    return paths
