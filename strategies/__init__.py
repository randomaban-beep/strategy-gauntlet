"""
Trading strategy candidates and central forward-shift return calculation.
"""
from __future__ import annotations

from strategies.base import ParamSpec, Strategy, param_grid_combinations, strategy_returns
from strategies.ma_crossover import MACrossover
from strategies.mean_reversion import MeanReversion
from strategies.stat_arb import CointegrationResult, StatArb, build_pair_dataframe, engle_granger_coint

__all__ = [
    "ParamSpec",
    "Strategy",
    "param_grid_combinations",
    "strategy_returns",
    "MACrossover",
    "MeanReversion",
    "StatArb",
    "CointegrationResult",
    "build_pair_dataframe",
    "engle_granger_coint",
]
