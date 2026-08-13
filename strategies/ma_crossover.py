"""
Moving-average crossover strategy.

Long when fast MA > slow MA, short when fast MA < slow MA (if allow_short).
Signals are computed using only past and current prices (no look-ahead).
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from strategies.base import ParamSpec, Strategy


class MACrossover(Strategy):
    """Simple dual moving-average crossover strategy.

    Parameters
    ----------
    fast_window : lookback for the fast (short) moving average
    slow_window : lookback for the slow (long) moving average; must be > fast
    allow_short : if False signals are clipped to {0, +1}
    """

    @property
    def name(self) -> str:
        return 'ma_crossover'

    def param_grid(self) -> list[ParamSpec]:
        return [
            ParamSpec('fast_window', [5, 10, 20]),
            ParamSpec('slow_window', [30, 50, 100, 200]),
            ParamSpec('allow_short', [True, False]),
        ]

    def generate_signals(  # type: ignore[override]
        self,
        df: pd.DataFrame,
        fast_window: int = 10,
        slow_window: int = 50,
        allow_short: bool = True,
        **_: Any,
    ) -> pd.Series:
        """Return +1 when fast MA > slow MA, -1 when below (or 0 if not allow_short).

        NaN periods (warm-up) are filled with 0.
        fast_window must be < slow_window; if not, signals are all 0.
        """
        if fast_window >= slow_window:
            return pd.Series(0, index=df.index, dtype=int, name='signal')

        fast_ma = df['close'].rolling(fast_window, min_periods=fast_window).mean()
        slow_ma = df['close'].rolling(slow_window, min_periods=slow_window).mean()

        raw = pd.Series(0, index=df.index, dtype=int)
        valid = fast_ma.notna() & slow_ma.notna()
        raw[valid & (fast_ma > slow_ma)] = 1
        if allow_short:
            raw[valid & (fast_ma < slow_ma)] = -1

        return raw.rename('signal')
