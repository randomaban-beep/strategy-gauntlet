"""
Hand-written bivariate Student-t copula for joint tail-dependent path simulation.

No external copula library is used; implemented directly with NumPy and SciPy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import special, stats


@dataclass
class StudentTCopulaFit:
    """Fitted parameters for a bivariate Student-t copula + marginals."""
    rho: float
    df: float
    mean_a: float
    std_a: float
    mean_b: float
    std_b: float
    marginal_a: np.ndarray
    marginal_b: np.ndarray


def _pseudo_observations(x: np.ndarray) -> np.ndarray:
    """Convert 1-D sample to uniform pseudo-observations in (0, 1) via ranks."""
    n = len(x)
    ranks = stats.rankdata(x, method="average")
    return ranks / (n + 1.0)


def _bivariate_t_copula_loglik(u1: np.ndarray, u2: np.ndarray, rho: float, df: float) -> float:
    """Evaluate log-likelihood of bivariate Student-t copula at (u1, u2)."""
    rho = float(np.clip(rho, -0.999, 0.999))
    x1 = stats.t.ppf(np.clip(u1, 1e-6, 1.0 - 1e-6), df=df)
    x2 = stats.t.ppf(np.clip(u2, 1e-6, 1.0 - 1e-6), df=df)

    one_minus_rho2 = 1.0 - rho ** 2
    quad = (x1 ** 2 - 2.0 * rho * x1 * x2 + x2 ** 2) / (df * one_minus_rho2)

    log_biv = (
        special.gammaln(0.5 * (df + 2.0))
        - special.gammaln(0.5 * df)
        - np.log(np.pi * df * np.sqrt(one_minus_rho2))
        - 0.5 * (df + 2.0) * np.log1p(quad)
    )
    log_marg = stats.t.logpdf(x1, df=df) + stats.t.logpdf(x2, df=df)
    return float(np.sum(log_biv - log_marg))


def fit_student_t_copula(
    returns_a: pd.Series | np.ndarray,
    returns_b: pd.Series | np.ndarray,
    df_grid: tuple[float, ...] = (3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 25.0),
) -> StudentTCopulaFit:
    """Fit a bivariate Student-t copula via rank correlation and profile likelihood over degrees of freedom."""
    ra = np.asarray(returns_a, dtype=float)
    rb = np.asarray(returns_b, dtype=float)
    n = min(len(ra), len(rb))
    ra = ra[:n]
    rb = rb[:n]

    u1 = _pseudo_observations(ra)
    u2 = _pseudo_observations(rb)

    # Estimate correlation via Kendall's tau relationship: rho = sin(pi * tau / 2)
    tau, _ = stats.kendalltau(ra, rb)
    if np.isnan(tau):
        rho = 0.0
    else:
        rho = float(np.sin(0.5 * np.pi * tau))
    rho = float(np.clip(rho, -0.995, 0.995))

    best_df = 5.0
    best_ll = -np.inf
    for nu in df_grid:
        ll = _bivariate_t_copula_loglik(u1, u2, rho, nu)
        if ll > best_ll:
            best_ll = ll
            best_df = float(nu)

    return StudentTCopulaFit(
        rho=rho,
        df=best_df,
        mean_a=float(np.mean(ra)),
        std_a=max(float(np.std(ra, ddof=1)), 1e-8),
        mean_b=float(np.mean(rb)),
        std_b=max(float(np.std(rb, ddof=1)), 1e-8),
        marginal_a=np.sort(ra),
        marginal_b=np.sort(rb),
    )


def simulate_student_t_copula_uniforms(
    rho: float,
    df: float,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate (n_samples, 2) uniform draws from a bivariate Student-t copula."""
    rho = float(np.clip(rho, -0.999, 0.999))
    z1 = rng.standard_normal(n_samples)
    z2 = rho * z1 + np.sqrt(max(1.0 - rho ** 2, 1e-12)) * rng.standard_normal(n_samples)

    chi2 = rng.chisquare(df=df, size=n_samples)
    scale = np.sqrt(chi2 / df)
    t1 = z1 / scale
    t2 = z2 / scale

    u1 = stats.t.cdf(t1, df=df)
    u2 = stats.t.cdf(t2, df=df)
    return np.column_stack([u1, u2])


def simulate_gaussian_copula_uniforms(
    rho: float,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate (n_samples, 2) uniform draws from a bivariate Gaussian copula."""
    rho = float(np.clip(rho, -0.999, 0.999))
    z1 = rng.standard_normal(n_samples)
    z2 = rho * z1 + np.sqrt(max(1.0 - rho ** 2, 1e-12)) * rng.standard_normal(n_samples)
    u1 = stats.norm.cdf(z1)
    u2 = stats.norm.cdf(z2)
    return np.column_stack([u1, u2])


def empirical_lower_tail_dependence(u: np.ndarray, quantile: float = 0.02) -> float:
    """Compute empirical lower-tail dependence P(U2 <= q | U1 <= q)."""
    mask1 = u[:, 0] <= quantile
    denom = int(np.sum(mask1))
    if denom == 0:
        return 0.0
    joint = int(np.sum(mask1 & (u[:, 1] <= quantile)))
    return float(joint / denom)


def simulate_copula_pair_paths(
    fit: StudentTCopulaFit,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator,
    error_correction_speed: float = 0.12,
    hedge_ratio: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate joint return paths `(paths_a, paths_b)` of shape `(n_paths, n_steps)`
    from a fitted bivariate Student-t copula.

    Uses inverse-normal / empirical scale mapping of Student-t copula innovations.
    If `hedge_ratio` is provided, applies a cointegrating error-correction term
    so joint shocks exhibit Student-t tail dependence while the spread mean-reverts.
    """
    total_draws = n_paths * n_steps
    u = simulate_student_t_copula_uniforms(fit.rho, fit.df, total_draws, rng)
    # Map uniforms to standard normal quantiles (preserving Student-t tail dependence)
    z1 = stats.norm.ppf(np.clip(u[:, 0], 1e-6, 1.0 - 1e-6)).reshape(n_paths, n_steps)
    z2 = stats.norm.ppf(np.clip(u[:, 1], 1e-6, 1.0 - 1e-6)).reshape(n_paths, n_steps)

    eps_a = fit.mean_a + fit.std_a * z1
    eps_b = fit.mean_b + fit.std_b * z2

    if hedge_ratio is None or error_correction_speed <= 0.0:
        return eps_a, eps_b

    paths_a = np.zeros((n_paths, n_steps))
    paths_b = eps_b.copy()
    for p in range(n_paths):
        log_pa = 0.0
        log_pb = 0.0
        for t in range(n_steps):
            spread_err = log_pa - hedge_ratio * log_pb
            ra = eps_a[p, t] - error_correction_speed * spread_err
            rb = paths_b[p, t]
            paths_a[p, t] = ra
            log_pa += ra
            log_pb += rb

    return paths_a, paths_b
