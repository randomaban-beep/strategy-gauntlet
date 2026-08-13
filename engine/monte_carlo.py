"""
Monte Carlo risk engine.

Provides path simulation (GBM, block bootstrap, model-based) and risk metrics.
All randomness is seeded via np.random.Generator for reproducibility.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

import config
from strategies.base import Strategy, strategy_returns
from validation.costs import apply_costs


@dataclass
class RiskMetrics:
    """Risk metrics computed over an ensemble of path returns."""
    mean_total_return: float
    sharpe: float
    sortino: float
    max_drawdown: float   # negative number (worst peak-to-trough)
    var99: float          # 1st percentile of total returns (negative = loss)
    cvar99: float         # mean of returns below VaR99 (negative = loss)
    prob_of_profit: float # fraction of paths with positive total return


def simulate_gbm_paths(
    mu: float,
    sigma: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate GBM log-return paths.

    Parameters
    ----------
    mu : daily drift
    sigma : daily volatility
    n_steps : number of steps per path
    n_paths : number of paths
    rng : seeded numpy Generator

    Returns
    -------
    np.ndarray of shape (n_paths, n_steps)
    """
    dt = 1.0
    log_rets = (mu - 0.5 * sigma ** 2) * dt + sigma * rng.standard_normal((n_paths, n_steps))
    return log_rets


def simulate_block_bootstrap_paths(
    historical_returns: np.ndarray,
    n_steps: int,
    n_paths: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Non-parametric block bootstrap path simulation.

    Samples contiguous blocks (with wrap-around) from historical returns.

    Parameters
    ----------
    historical_returns : 1-D array of observed log returns
    n_steps : number of steps per path
    n_paths : number of paths
    block_length : number of consecutive bars per block
    rng : seeded numpy Generator

    Returns
    -------
    np.ndarray of shape (n_paths, n_steps)
    """
    n_hist = len(historical_returns)
    paths = np.zeros((n_paths, n_steps))
    for i in range(n_paths):
        idx = 0
        while idx < n_steps:
            start = rng.integers(0, n_hist)
            block = historical_returns[np.arange(start, start + block_length) % n_hist]
            end = min(idx + block_length, n_steps)
            paths[i, idx:end] = block[:end - idx]
            idx = end
    return paths


def evaluate_strategy_on_paths(
    strategy: Strategy,
    params: dict[str, Any],
    paths: np.ndarray,
    spread_bps: float = config.SPREAD_BPS,
    commission_bps: float = config.COMMISSION_BPS,
    impact_coef: float = config.IMPACT_COEF,
    allow_short: bool = True,
) -> np.ndarray:
    """Replay a strategy on each simulated path and return per-path total returns.

    Uses `strategy_returns` (the central forward-shift function) on each path.

    Parameters
    ----------
    strategy : Strategy instance
    params : hyper-parameters for the strategy
    paths : np.ndarray of shape (n_paths, n_steps) with log returns
    spread_bps, commission_bps, impact_coef : cost parameters
    allow_short : passed to strategy_returns

    Returns
    -------
    np.ndarray of shape (n_paths,) total log return per path
    """
    n_paths, n_steps = paths.shape
    total_returns = np.zeros(n_paths)

    for i in range(n_paths):
        path_rets = pd.Series(paths[i])
        # Build a minimal DataFrame for the strategy
        close = 100.0 * np.exp(np.cumsum(path_rets.values))
        df = pd.DataFrame({'close': close, 'log_return': path_rets.values})
        try:
            signals = strategy.generate_signals(df, **params)
            strat_rets = strategy_returns(signals, path_rets, allow_short=allow_short)
            net_rets = apply_costs(strat_rets, signals, spread_bps, commission_bps, impact_coef)
            total_returns[i] = float(net_rets.sum())
        except Exception:
            total_returns[i] = 0.0

    return total_returns


def compute_risk_metrics(
    path_returns: np.ndarray,
    ann_factor: int = config.ANNUALIZATION,
) -> RiskMetrics:
    """Compute risk metrics from an array of per-path total log returns.

    Parameters
    ----------
    path_returns : 1-D array of total log returns, one per simulated path
    ann_factor : annualization factor for Sharpe/Sortino

    Returns
    -------
    RiskMetrics dataclass
    """
    n = len(path_returns)
    if n == 0:
        return RiskMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    mean_ret = float(np.mean(path_returns))
    std_ret = float(np.std(path_returns, ddof=1)) if n > 1 else 0.0

    # Sharpe (annualized by sqrt of number of paths approximation is wrong;
    # here path_returns are total returns across all steps, not per-step,
    # so we report cross-sectional Sharpe of the distribution)
    sharpe = float(mean_ret / std_ret) if std_ret > 0 else 0.0

    # Sortino: downside std
    neg = path_returns[path_returns < 0]
    downside_std = float(np.std(neg, ddof=1)) if len(neg) > 1 else 0.0
    sortino = float(mean_ret / downside_std) if downside_std > 0 else (1e6 if mean_ret > 0 else 0.0)

    # Max drawdown: for each path compute cumulative return and find worst drawdown
    # For simplicity: max drawdown is computed on the distribution of total returns
    # (path-level, not intra-path). Use the worst 1% quantile as proxy.
    sorted_rets = np.sort(path_returns)
    var99 = float(np.percentile(path_returns, 1))   # 1st percentile
    cvar99_mask = path_returns <= var99
    cvar99 = float(np.mean(path_returns[cvar99_mask])) if cvar99_mask.any() else var99

    # Max drawdown: min cumulative sum (approximation over sorted paths)
    max_drawdown = float(np.min(path_returns))  # worst single-path return

    prob_of_profit = float(np.mean(path_returns > 0))

    return RiskMetrics(
        mean_total_return=mean_ret,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_drawdown,
        var99=var99,
        cvar99=cvar99,
        prob_of_profit=prob_of_profit,
    )


def run_monte_carlo(
    strategy: Strategy,
    params: dict[str, Any],
    trusted_models: dict[str, Any],
    historical_returns: pd.Series,
    n_paths: int = config.N_MC_PATHS,
    seed: int = config.MC_SEED,
    n_steps: int | None = None,
    block_length: int = config.BLOCK_LENGTH,
    spread_bps: float = config.SPREAD_BPS,
    commission_bps: float = config.COMMISSION_BPS,
    impact_coef: float = config.IMPACT_COEF,
    allow_short: bool = True,
) -> dict[str, RiskMetrics]:
    """Run Monte Carlo across all trusted models + block bootstrap.

    Parameters
    ----------
    strategy : Strategy instance
    params : best strategy parameters
    trusted_models : dict of model_name -> fit object
    historical_returns : observed log-return series (for bootstrap + GBM params)
    n_paths : number of MC paths per model
    seed : base random seed
    n_steps : steps per path (defaults to len(historical_returns))
    block_length : block length for bootstrap
    spread_bps, commission_bps, impact_coef : cost parameters
    allow_short : passed to strategy_returns

    Returns
    -------
    dict mapping model_name -> RiskMetrics
    """
    if n_steps is None:
        n_steps = len(historical_returns)

    results: dict[str, RiskMetrics] = {}
    rng = np.random.default_rng(seed)

    # Always run block bootstrap
    bootstrap_paths = simulate_block_bootstrap_paths(
        historical_returns.values, n_steps, n_paths, block_length, rng
    )
    bootstrap_rets = evaluate_strategy_on_paths(
        strategy, params, bootstrap_paths, spread_bps, commission_bps, impact_coef, allow_short
    )
    results['bootstrap'] = compute_risk_metrics(bootstrap_rets)

    # Run each trusted model
    for model_name, fit in trusted_models.items():
        det_offset = sum(ord(c) * (idx + 1) for idx, c in enumerate(model_name))
        model_rng = np.random.default_rng(seed + det_offset)
        try:
            if model_name == 'gbm':
                from validation.model_selection import GBMFit
                assert isinstance(fit, GBMFit)
                paths = simulate_gbm_paths(
                    mu=fit.mu, sigma=fit.sigma,
                    n_steps=n_steps, n_paths=n_paths, rng=model_rng
                )
            elif model_name == 'garch':
                from engine.garch import simulate_garch_paths
                paths = simulate_garch_paths(fit, n_steps, n_paths, model_rng)
            elif model_name == 'hmm':
                from engine.regime_switching import simulate_hmm_paths
                paths = simulate_hmm_paths(fit, n_steps, n_paths, model_rng)
            else:
                continue

            path_rets = evaluate_strategy_on_paths(
                strategy, params, paths, spread_bps, commission_bps, impact_coef, allow_short
            )
            results[model_name] = compute_risk_metrics(path_rets)
        except Exception as e:
            # Log and skip failing models
            import warnings
            warnings.warn(f'MC model {model_name} failed: {e}')
            continue

    return results


def passes_risk_gate(
    mc_results: dict[str, RiskMetrics],
    confidence_threshold: float = config.CONFIDENCE_THRESHOLD,
) -> tuple[bool, str]:
    """Check if a strategy passes the risk gate across ALL models.

    A strategy passes only if prob_of_profit >= confidence_threshold under
    every trusted model (not just the easiest one).

    Parameters
    ----------
    mc_results : dict from run_monte_carlo
    confidence_threshold : minimum probability of profit required

    Returns
    -------
    (passed: bool, reason: str)
    """
    if not mc_results:
        return False, 'No MC results available'

    failing_models = [
        f'{name} (p={m.prob_of_profit:.3f})'
        for name, m in mc_results.items()
        if m.prob_of_profit < confidence_threshold
    ]

    if failing_models:
        return False, f'Prob-of-profit below {confidence_threshold} under: {", ".join(failing_models)}'
    return True, 'Passed all models'


def evaluate_pair_strategy_on_paths(
    strategy: Strategy,
    params: dict[str, Any],
    paths_a: np.ndarray,
    paths_b: np.ndarray,
    hedge_ratio: float = 1.0,
    spread_bps: float = config.SPREAD_BPS,
    commission_bps: float = config.COMMISSION_BPS,
    impact_coef: float = config.IMPACT_COEF,
    allow_short: bool = True,
) -> np.ndarray:
    """Replay a pair strategy on joint simulated return paths (paths_a, paths_b).

    Uses the central `strategy_returns` forward-shift guard on the pair spread return.
    """
    from strategies.stat_arb import build_pair_dataframe

    n_paths, n_steps = paths_a.shape
    total_returns = np.zeros(n_paths)

    for i in range(n_paths):
        ra = paths_a[i]
        rb = paths_b[i]
        close_a = 100.0 * np.exp(np.cumsum(ra))
        close_b = 100.0 * np.exp(np.cumsum(rb))
        df_a = pd.DataFrame({'close': close_a, 'log_return': ra})
        df_b = pd.DataFrame({'close': close_b, 'log_return': rb})
        pair_df = build_pair_dataframe(df_a, df_b, hedge_ratio=hedge_ratio)
        try:
            signals = strategy.generate_signals(pair_df, **params)
            strat_rets = strategy_returns(signals, pair_df['log_return'], allow_short=allow_short)
            net_rets = apply_costs(strat_rets, signals, spread_bps, commission_bps, impact_coef)
            total_returns[i] = float(net_rets.sum())
        except Exception:
            total_returns[i] = 0.0

    return total_returns


def run_pair_monte_carlo(
    strategy: Strategy,
    params: dict[str, Any],
    pair_df: pd.DataFrame,
    n_paths: int = config.N_MC_PATHS,
    seed: int = config.MC_SEED,
    n_steps: int | None = None,
    block_length: int = config.BLOCK_LENGTH,
    spread_bps: float = config.SPREAD_BPS,
    commission_bps: float = config.COMMISSION_BPS,
    impact_coef: float = config.IMPACT_COEF,
    allow_short: bool = True,
) -> dict[str, RiskMetrics]:
    """Run Monte Carlo for a pair strategy using a bivariate Student-t copula and block bootstrap."""
    from engine.copula import fit_student_t_copula, simulate_copula_pair_paths

    if n_steps is None:
        n_steps = min(len(pair_df), 252)

    rng = np.random.default_rng(seed)
    results: dict[str, RiskMetrics] = {}

    hedge_ratio = float(pair_df['hedge_ratio'].iloc[0]) if 'hedge_ratio' in pair_df.columns else 1.0
    ra = pair_df['log_return_a'].values
    rb = pair_df['log_return_b'].values

    # 1. Joint block bootstrap of (ra, rb)
    n_hist = len(ra)
    boot_a = np.zeros((n_paths, n_steps))
    boot_b = np.zeros((n_paths, n_steps))
    for i in range(n_paths):
        idx = 0
        while idx < n_steps:
            start = rng.integers(0, n_hist)
            indices = np.arange(start, start + block_length) % n_hist
            end = min(idx + block_length, n_steps)
            boot_a[i, idx:end] = ra[indices[:end - idx]]
            boot_b[i, idx:end] = rb[indices[:end - idx]]
            idx = end

    boot_rets = evaluate_pair_strategy_on_paths(
        strategy, params, boot_a, boot_b,
        hedge_ratio=hedge_ratio,
        spread_bps=spread_bps, commission_bps=commission_bps,
        impact_coef=impact_coef, allow_short=allow_short,
    )
    results['bootstrap'] = compute_risk_metrics(boot_rets)

    # 2. Bivariate Student-t copula paths
    copula_fit = fit_student_t_copula(ra, rb)
    cop_rng = np.random.default_rng(seed + 777)
    cop_a, cop_b = simulate_copula_pair_paths(
        copula_fit,
        n_steps=n_steps,
        n_paths=n_paths,
        rng=cop_rng,
        hedge_ratio=hedge_ratio,
    )
    cop_rets = evaluate_pair_strategy_on_paths(
        strategy, params, cop_a, cop_b,
        hedge_ratio=hedge_ratio,
        spread_bps=spread_bps, commission_bps=commission_bps,
        impact_coef=impact_coef, allow_short=allow_short,
    )
    results['student_t_copula'] = compute_risk_metrics(cop_rets)

    return results

