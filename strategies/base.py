"""
Base classes and utilities for all trading strategies.

Convention
----------
Signals : +1 long, 0 flat, -1 short.
Look-ahead guard : The one-bar forward shift from signal[t] -> return[t+1]
happens ONLY in `strategy_returns`. No other module may shift returns.
"""
from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
import pandas as pd


@dataclass
class ParamSpec:
    """Specification of a single hyper-parameter grid dimension."""
    name: str
    values: list[Any]


def param_grid_combinations(specs: list[ParamSpec]) -> list[dict[str, Any]]:
    """Return the Cartesian product of all ParamSpec values.

    Parameters
    ----------
    specs : list of ParamSpec

    Returns
    -------
    List of dicts mapping parameter name -> value.
    """
    if not specs:
        return [{}]
    keys = [s.name for s in specs]
    value_lists = [s.values for s in specs]
    return [
        dict(zip(keys, combo))
        for combo in itertools.product(*value_lists)
    ]


class Strategy(ABC):
    """Abstract base class for all trading strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique strategy identifier."""
        ...

    @abstractmethod
    def param_grid(self) -> list[ParamSpec]:
        """Return the hyper-parameter grid specs for this strategy."""
        ...

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame, **params: Any) -> pd.Series:
        """Generate a signal series from price data.

        Parameters
        ----------
        df : DataFrame with at minimum columns 'close' and 'log_return'.
            Signal at bar t may only use information available at bar t.
        **params : strategy hyper-parameters.

        Returns
        -------
        pd.Series of int in {-1, 0, +1}, same index as df.
        """
        ...

    def all_param_combinations(self) -> list[dict[str, Any]]:
        """Return all parameter combinations for grid search."""
        return param_grid_combinations(self.param_grid())


def strategy_returns(
    signals: pd.Series,
    log_returns: pd.Series,
    allow_short: bool = True,
) -> pd.Series:
    """Compute strategy log returns with a one-bar forward shift.

    This is the SINGLE PLACE where the forward shift is applied.
    signal[t] * log_return[t+1], so position at t earns the next bar's return.

    Parameters
    ----------
    signals : pd.Series of {-1, 0, +1}
    log_returns : pd.Series of log returns (same index as signals)
    allow_short : if False, replace -1 signals with 0.

    Returns
    -------
    pd.Series of strategy log returns, same index as inputs.
    First element is always 0 (no return before the first signal).
    """
    sig = signals.copy()
    if not allow_short:
        sig = sig.clip(lower=0)
    # Shift signals forward by 1: position at t earns return at t+1
    shifted = sig.shift(1).fillna(0)
    return (shifted * log_returns).rename('strategy_return')
