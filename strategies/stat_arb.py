"""
Statistical arbitrage (pairs trading) strategy using Engle-Granger cointegration.

Workflow:
1. Test cointegration between price_a and price_b via statsmodels `coint` (Engle-Granger).
2. Estimate hedge ratio via OLS (`price_a = intercept + hedge_ratio * price_b`).
3. Compute `spread = price_a - hedge_ratio * price_b`.
4. Generate entry/exit signals from the rolling z-score of the spread.
5. Only treat a pair as valid if cointegration is statistically significant.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

from strategies.base import ParamSpec, Strategy


@dataclass
class CointegrationResult:
    """Result from Engle-Granger cointegration test and OLS hedge ratio fit."""
    is_cointegrated: bool
    pvalue: float
    t_stat: float
    hedge_ratio: float
    intercept: float
    spread: pd.Series


def engle_granger_coint(
    price_a: pd.Series,
    price_b: pd.Series,
    significance: float = 0.05,
) -> CointegrationResult:
    """Run Engle-Granger cointegration test and OLS hedge ratio regression.

    Parameters
    ----------
    price_a : price series of leg A
    price_b : price series of leg B
    significance : p-value threshold to declare cointegration significant

    Returns
    -------
    CointegrationResult
    """
    a = pd.Series(price_a).astype(float).reset_index(drop=True)
    b = pd.Series(price_b).astype(float).reset_index(drop=True)

    if len(a) < 30 or len(b) < 30 or a.std() < 1e-12 or b.std() < 1e-12:
        zeros = pd.Series(0.0, index=a.index)
        return CointegrationResult(
            is_cointegrated=False,
            pvalue=1.0,
            t_stat=0.0,
            hedge_ratio=1.0,
            intercept=0.0,
            spread=zeros,
        )

    # OLS hedge ratio: price_a = intercept + hedge_ratio * price_b
    X = sm.add_constant(b.values)
    ols_res = sm.OLS(a.values, X).fit()
    intercept = float(ols_res.params[0])
    hedge_ratio = float(ols_res.params[1])

    try:
        t_stat, pvalue, _ = coint(a.values, b.values, maxlag=5)
        t_stat = float(t_stat)
        pvalue = float(pvalue)
    except Exception:
        t_stat = 0.0
        pvalue = 1.0

    spread = (a - hedge_ratio * b).rename("spread")
    is_coint = bool(pvalue < significance)

    return CointegrationResult(
        is_cointegrated=is_coint,
        pvalue=pvalue,
        t_stat=t_stat,
        hedge_ratio=hedge_ratio,
        intercept=intercept,
        spread=spread,
    )


def build_pair_dataframe(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    hedge_ratio: float | None = None,
    calib_bars: int = 200,
) -> pd.DataFrame:
    """Combine two asset DataFrames into a pair DataFrame suitable for StatArb and walk_forward.

    The `log_return` column represents the capital-normalized spread return at bar t:
        (P_{A,t-1} * r_{A,t} - h * P_{B,t-1} * r_{B,t}) / (P_{A,t-1} + |h| * P_{B,t-1})
    so that `strategy_returns(signals, df['log_return'])` applies the central 1-bar forward shift.
    """
    n = min(len(df_a), len(df_b))
    close_a = df_a["close"].iloc[:n].reset_index(drop=True).astype(float)
    close_b = df_b["close"].iloc[:n].reset_index(drop=True).astype(float)
    ret_a = df_a["log_return"].iloc[:n].reset_index(drop=True).astype(float)
    ret_b = df_b["log_return"].iloc[:n].reset_index(drop=True).astype(float)

    if hedge_ratio is None:
        c_len = min(calib_bars, n)
        coint_res = engle_granger_coint(close_a.iloc[:c_len], close_b.iloc[:c_len])
        hedge_ratio = coint_res.hedge_ratio

    spread = close_a - hedge_ratio * close_b

    # Previous bar prices for dollar-weighted spread return (causal at bar t)
    prev_a = close_a.shift(1).bfill()
    prev_b = close_b.shift(1).bfill()
    gross_notional = (prev_a + abs(hedge_ratio) * prev_b).clip(lower=1e-12)
    pair_return = (prev_a * ret_a - hedge_ratio * prev_b * ret_b) / gross_notional

    vol_a = df_a["volume"].iloc[:n].reset_index(drop=True) if "volume" in df_a.columns else pd.Series(1e6, index= range(n))
    vol_b = df_b["volume"].iloc[:n].reset_index(drop=True) if "volume" in df_b.columns else pd.Series(1e6, index=range(n))

    out = pd.DataFrame({
        "close": spread,
        "close_a": close_a,
        "close_b": close_b,
        "volume": 0.5 * (vol_a + vol_b),
        "log_return": pair_return,
        "log_return_a": ret_a,
        "log_return_b": ret_b,
        "hedge_ratio": hedge_ratio,
    })
    return out


class StatArb(Strategy):
    """Engle-Granger cointegration pairs-trading strategy.

    Only generates active signals if the pair is significantly cointegrated.
    """

    def __init__(
        self,
        significance: float = 0.05,
        hedge_ratio: float | None = None,
        is_cointegrated: bool | None = None,
        calib_bars: int = 200,
    ) -> None:
        self.significance = significance
        self.hedge_ratio = hedge_ratio
        self.is_cointegrated = is_cointegrated
        self.calib_bars = calib_bars

    @property
    def name(self) -> str:
        return "stat_arb"

    def param_grid(self) -> list[ParamSpec]:
        return [
            ParamSpec("window", [15, 20, 30]),
            ParamSpec("entry_threshold", [1.5, 2.0]),
            ParamSpec("exit_threshold", [0.0, 0.5]),
            ParamSpec("allow_short", [True]),
        ]

    def generate_signals(
        self,
        df: pd.DataFrame,
        window: int = 20,
        entry_threshold: float = 1.5,
        exit_threshold: float = 0.5,
        allow_short: bool = True,
        **_: Any,
    ) -> pd.Series:
        """Compute z-score entry/exit signals on the spread `price_a - hedge_ratio * price_b`.

        If the pair is not significantly cointegrated, returns all-zero signals.
        """
        n = len(df)
        if n == 0:
            return pd.Series([], dtype=int, name="signal")

        if self.is_cointegrated is False:
            return pd.Series(0, index=df.index, dtype=int, name="signal")

        if "close_a" in df.columns and "close_b" in df.columns:
            price_a = df["close_a"]
            price_b = df["close_b"]
            if self.is_cointegrated is None:
                c_len = min(self.calib_bars, n)
                coint_res = engle_granger_coint(
                    price_a.iloc[:c_len],
                    price_b.iloc[:c_len],
                    significance=self.significance,
                )
                if not coint_res.is_cointegrated:
                    return pd.Series(0, index=df.index, dtype=int, name="signal")
                hr = self.hedge_ratio if self.hedge_ratio is not None else coint_res.hedge_ratio
            else:
                if self.hedge_ratio is not None:
                    hr = self.hedge_ratio
                elif "hedge_ratio" in df.columns:
                    hr = float(df["hedge_ratio"].iloc[0])
                else:
                    c_len = min(self.calib_bars, n)
                    hr = engle_granger_coint(price_a.iloc[:c_len], price_b.iloc[:c_len]).hedge_ratio
            spread = price_a - hr * price_b
        else:
            spread = df["close"]

        roll_mean = spread.rolling(window, min_periods=window).mean()
        roll_std = spread.rolling(window, min_periods=window).std().replace(0, np.nan)
        z = (spread - roll_mean) / roll_std

        signals = np.zeros(n, dtype=int)
        position = 0

        for i in range(n):
            zi = z.iloc[i]
            if np.isnan(zi):
                signals[i] = 0
                continue
            if position == 0:
                if zi < -entry_threshold:
                    position = 1
                elif allow_short and zi > entry_threshold:
                    position = -1
            elif position == 1:
                if zi >= -exit_threshold:
                    position = 0
            elif position == -1:
                if zi <= exit_threshold:
                    position = 0
            signals[i] = position

        return pd.Series(signals, index=df.index, dtype=int, name="signal")
