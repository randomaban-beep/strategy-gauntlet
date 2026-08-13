"""
Bailey & Lopez de Prado Deflated Sharpe Ratio (DSR).

Reference: Bailey, D.H. and Lopez de Prado, M. (2014).
"The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest
Overfitting and Non-Normality". Journal of Portfolio Management.

Implementation uses:
  - Observed Sharpe Ratio (SR)
  - Expected maximum SR under n_trials independent tests
  - Probabilistic Sharpe Ratio adjusted for skewness and kurtosis
  - DSR = P(SR* > SR_benchmark)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

import config


@dataclass
class DSRResult:
    """Result from the Deflated Sharpe Ratio calculation."""
    observed_sharpe: float          # annualized SR of the OOS returns
    expected_max_sharpe: float      # E[max SR] under n_trials trials
    dsr_probability: float          # P(SR* > expected_max_SR), in [0, 1]
    passes_threshold: bool          # dsr_probability >= dsr_threshold


def _annualized_sharpe(returns: np.ndarray, ann: int = config.ANNUALIZATION) -> float:
    """Compute annualized Sharpe ratio from a daily return array."""
    if len(returns) < 2:
        return 0.0
    std = float(np.std(returns, ddof=1))
    if std < 1e-12 or not np.isfinite(std):
        return 0.0
    return float(np.mean(returns) / std * np.sqrt(ann))


def _expected_max_sharpe(
    n_trials: int,
    trial_sharpes: np.ndarray | None = None,
    default_sr_std: float = 1.0,
) -> float:
    """Expected maximum Sharpe Ratio under n_trials independent trials.

    If trial_sharpes is provided, use the empirical variance.
    Otherwise assume SR ~ N(0, default_sr_std^2).

    Uses the approximation from Bailey & Lopez de Prado (2014) eq. (8):
        E[max SR] ≈ sigma_SR * ((1 - euler_gamma) * z(1 - 1/n) + euler_gamma * z(1 - 1/(n*e)))
    where euler_gamma ≈ 0.5772.

    For simplicity we use the tighter approximation:
        E[max SR] = sigma_SR * norm.ppf(1 - 1/n_trials)
    which is exact for large n_trials.

    Parameters
    ----------
    n_trials : number of independent strategy trials
    trial_sharpes : array of observed Sharpe ratios across trials (optional)
    default_sr_std : assumed std of SR distribution if trial_sharpes not given

    Returns
    -------
    Expected maximum Sharpe Ratio (float)
    """
    if n_trials <= 1:
        return 0.0  # No inflation adjustment for a single trial

    if trial_sharpes is not None and len(trial_sharpes) > 1:
        sigma_sr = float(np.std(trial_sharpes, ddof=1))
    else:
        sigma_sr = default_sr_std

    if sigma_sr <= 0:
        sigma_sr = default_sr_std

    # E[max of n_trials N(0,sigma^2) draws]
    # Using Euler-Mascheroni approximation
    gamma_em = 0.5772156649
    if n_trials >= 2:
        z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
        z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
        e_max = sigma_sr * ((1.0 - gamma_em) * z1 + gamma_em * z2)
    else:
        e_max = 0.0

    return float(e_max)


def probabilistic_sharpe_ratio(
    returns: np.ndarray,
    sr_benchmark: float,
    ann: int = config.ANNUALIZATION,
) -> float:
    """Probabilistic Sharpe Ratio: P(SR* > sr_benchmark).

    Adjusts for skewness, kurtosis and sample size.
    From Bailey & Lopez de Prado (2014), eq. (5):

        PSR(SR*) = Phi{ (SR - SR*) * sqrt(T-1) / sqrt(1 - skew*SR + (kurt-1)/4 * SR^2) }

    Parameters
    ----------
    returns : array of strategy returns
    sr_benchmark : benchmark Sharpe ratio (here: expected max SR)
    ann : annualization factor

    Returns
    -------
    Probability in [0, 1]
    """
    T = len(returns)
    if T < 4:
        return 0.0

    sr_hat = _annualized_sharpe(returns, ann)
    sr_daily = sr_hat / np.sqrt(ann)  # work in daily units for skew/kurt

    skew = float(stats.skew(returns))
    kurt = float(stats.kurtosis(returns, fisher=True))  # excess kurtosis

    denominator_sq = 1.0 - skew * sr_daily + (kurt - 1.0) / 4.0 * sr_daily ** 2
    if denominator_sq <= 0:
        denominator_sq = 1e-6  # numerical guard

    z = (sr_hat - sr_benchmark) * np.sqrt(T - 1) / np.sqrt(denominator_sq * ann)
    return float(stats.norm.cdf(z))


def deflated_sharpe_ratio(
    returns: np.ndarray,
    n_trials: int,
    trial_sharpes: np.ndarray | None = None,
    dsr_threshold: float = config.DSR_THRESHOLD,
    ann: int = config.ANNUALIZATION,
) -> DSRResult:
    """Compute the Deflated Sharpe Ratio.

    Parameters
    ----------
    returns : OOS return array (daily log returns)
    n_trials : number of strategy trials attempted (for expected max SR calculation)
    trial_sharpes : array of Sharpe ratios from all trials (optional; used for
        empirical sigma_SR). If None, a default sigma=1.0 is used.
    dsr_threshold : minimum DSR probability to pass
    ann : annualization factor

    Returns
    -------
    DSRResult dataclass
    """
    returns = np.asarray(returns, dtype=float)
    observed_sr = _annualized_sharpe(returns, ann)
    expected_max_sr = _expected_max_sharpe(n_trials, trial_sharpes)
    psr = probabilistic_sharpe_ratio(returns, expected_max_sr, ann)

    return DSRResult(
        observed_sharpe=observed_sr,
        expected_max_sharpe=expected_max_sr,
        dsr_probability=psr,
        passes_threshold=psr >= dsr_threshold,
    )


def score_strategy(
    oos_returns: np.ndarray,
    n_trials: int,
    trial_sharpes: np.ndarray | None = None,
    dsr_threshold: float = config.DSR_THRESHOLD,
    ann: int = config.ANNUALIZATION,
) -> DSRResult:
    """Convenience wrapper: compute DSR from OOS return stream.

    Uses the OOS return series from walk-forward, computes Sharpe, skew,
    kurtosis, and the DSR. Approves only if dsr_probability >= dsr_threshold.

    Parameters
    ----------
    oos_returns : concatenated OOS log return series
    n_trials : number of strategy attempts so far
    trial_sharpes : optional array of Sharpe ratios from previous trials
    dsr_threshold : minimum DSR probability to pass
    ann : annualization factor

    Returns
    -------
    DSRResult
    """
    return deflated_sharpe_ratio(
        returns=oos_returns,
        n_trials=n_trials,
        trial_sharpes=trial_sharpes,
        dsr_threshold=dsr_threshold,
        ann=ann,
    )
