"""
Rolling/expanding walk-forward validation with embargo.

For each fold:
  - Train window: select best params by Sharpe (after costs)
  - Embargo: gap between train end and test start
  - Test window: evaluate best params out-of-sample
Result: concatenated OOS returns, per-fold params/Sharpe, aggregate OOS Sharpe.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

import config
from strategies.base import Strategy, strategy_returns
from validation.costs import apply_costs


@dataclass
class FoldResult:
    """Result for a single walk-forward fold."""
    fold_idx: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    best_params: dict[str, Any]
    train_sharpe: float
    test_sharpe: float
    test_returns: pd.Series


@dataclass
class WalkForwardResult:
    """Aggregated walk-forward result."""
    oos_returns: pd.Series  # concatenated OOS returns
    folds: list[FoldResult] = field(default_factory=list)
    aggregate_sharpe: float = 0.0
    total_return: float = 0.0


def _sharpe(returns: pd.Series, ann: int = config.ANNUALIZATION) -> float:
    """Annualized Sharpe ratio; returns 0 if std is zero."""
    if len(returns) < 2:
        return 0.0
    std = returns.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return float(returns.mean() / std * np.sqrt(ann))


def _evaluate_params(
    df: pd.DataFrame,
    strategy: Strategy,
    params: dict[str, Any],
    spread_bps: float,
    commission_bps: float,
    impact_coef: float,
    allow_short: bool = True,
) -> float:
    """Evaluate a parameter set on df; return annualized Sharpe after costs."""
    try:
        signals = strategy.generate_signals(df, **params)
        raw_rets = strategy_returns(signals, df['log_return'], allow_short=allow_short)
        net_rets = apply_costs(raw_rets, signals, spread_bps, commission_bps, impact_coef)
        return _sharpe(net_rets)
    except Exception:
        return -np.inf


def walk_forward(
    df: pd.DataFrame,
    strategy: Strategy,
    train_bars: int = config.MIN_TRAIN_BARS,
    test_bars: int = config.MIN_TEST_BARS,
    embargo_bars: int = config.EMBARGO_BARS,
    spread_bps: float = config.SPREAD_BPS,
    commission_bps: float = config.COMMISSION_BPS,
    impact_coef: float = config.IMPACT_COEF,
    allow_short: bool = True,
    expanding: bool = False,
) -> WalkForwardResult:
    """Run rolling (or expanding) walk-forward validation.

    Parameters
    ----------
    df : full price DataFrame (must have log_return column)
    strategy : Strategy instance
    train_bars : size of the training window (rolling) or min train size (expanding)
    test_bars : size of the test window
    embargo_bars : bars to skip between train end and test start
    spread_bps, commission_bps, impact_coef : cost parameters
    allow_short : passed to strategy_returns
    expanding : if True, training window grows from train_bars to current fold start

    Returns
    -------
    WalkForwardResult

    Raises
    ------
    ValueError : if the data is too short to form at least 2 folds
    """
    n = len(df)
    min_needed = train_bars + embargo_bars + test_bars
    if n < min_needed:
        raise ValueError(
            f'Data too short ({n} bars) for even one fold '
            f'(need {min_needed} = {train_bars} train + {embargo_bars} embargo + {test_bars} test).'
        )

    # Build fold indices
    folds_indices: list[tuple[int, int, int, int]] = []
    test_start = train_bars + embargo_bars
    while test_start + test_bars <= n:
        train_end = test_start - embargo_bars  # exclusive
        train_start = 0 if expanding else max(0, train_end - train_bars)
        test_end = test_start + test_bars
        folds_indices.append((train_start, train_end, test_start, test_end))
        test_start += test_bars

    if len(folds_indices) < 2:
        raise ValueError(
            f'Only {len(folds_indices)} fold(s) possible with {n} bars '
            f'(need at least 2). Reduce train_bars or test_bars, or use more data.'
        )

    param_combos = strategy.all_param_combinations()
    fold_results: list[FoldResult] = []
    oos_series_list: list[pd.Series] = []

    for fold_idx, (ts, te, ss, se) in enumerate(folds_indices):
        train_df = df.iloc[ts:te]
        test_df = df.iloc[ss:se]

        # Select best params on train window
        best_params = param_combos[0]
        best_train_sharpe = -np.inf
        for params in param_combos:
            sh = _evaluate_params(
                train_df, strategy, params,
                spread_bps, commission_bps, impact_coef, allow_short
            )
            if sh > best_train_sharpe:
                best_train_sharpe = sh
                best_params = params

        # Evaluate on test window
        test_signals = strategy.generate_signals(test_df, **best_params)
        test_raw = strategy_returns(test_signals, test_df['log_return'], allow_short=allow_short)
        test_net = apply_costs(test_raw, test_signals, spread_bps, commission_bps, impact_coef)
        test_sh = _sharpe(test_net)

        fold_results.append(FoldResult(
            fold_idx=fold_idx,
            train_start=ts,
            train_end=te,
            test_start=ss,
            test_end=se,
            best_params=best_params,
            train_sharpe=best_train_sharpe,
            test_sharpe=test_sh,
            test_returns=test_net,
        ))
        oos_series_list.append(test_net)

    oos_returns = pd.concat(oos_series_list).reset_index(drop=True)
    agg_sharpe = _sharpe(oos_returns)
    total_return = float(oos_returns.sum())

    return WalkForwardResult(
        oos_returns=oos_returns,
        folds=fold_results,
        aggregate_sharpe=agg_sharpe,
        total_return=total_return,
    )
