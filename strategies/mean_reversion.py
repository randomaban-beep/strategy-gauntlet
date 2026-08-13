"""
Rolling z-score mean-reversion strategy.

Enters long when z-score < -entry_threshold (price cheap relative to MA),
enters short when z-score > +entry_threshold (price expensive),
exits when |z-score| < exit_threshold.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from strategies.base import ParamSpec, Strategy


class MeanReversion(Strategy):
    """Rolling z-score mean-reversion strategy.

    Parameters
    ----------
    window : lookback window for rolling mean and std
    entry_threshold : |z| must exceed this to open a position
    exit_threshold : |z| must fall below this to close the position
    allow_short : if False, only long entries are taken
    """

    @property
    def name(self) -> str:
        return 'mean_reversion'

    def param_grid(self) -> list[ParamSpec]:
        return [
            ParamSpec('window', [20, 40, 60]),
            ParamSpec('entry_threshold', [1.5, 2.0, 2.5]),
            ParamSpec('exit_threshold', [0.0, 0.5, 1.0]),
            ParamSpec('allow_short', [True, False]),
        ]

    def generate_signals(  # type: ignore[override]
        self,
        df: pd.DataFrame,
        window: int = 20,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
        allow_short: bool = True,
        **_: Any,
    ) -> pd.Series:
        """Return position signals using z-score mean-reversion logic.

        Uses a stateful loop to carry positions through the exit zone.
        No future data is used: only data up to and including bar t.
        """
        roll_mean = df['close'].rolling(window, min_periods=window).mean()
        roll_std = df['close'].rolling(window, min_periods=window).std()

        # Avoid division by zero
        roll_std = roll_std.replace(0, np.nan)
        z = (df['close'] - roll_mean) / roll_std

        n = len(df)
        signals = np.zeros(n, dtype=int)
        position = 0

        for i in range(n):
            if np.isnan(z.iloc[i]):
                signals[i] = 0
                continue
            zi = z.iloc[i]
            if position == 0:
                if zi < -entry_threshold:
                    position = 1  # price cheap: go long
                elif allow_short and zi > entry_threshold:
                    position = -1  # price expensive: go short
            elif position == 1:
                if zi >= -exit_threshold:
                    position = 0  # exit long
            elif position == -1:
                if zi <= exit_threshold:
                    position = 0  # exit short
            signals[i] = position

        return pd.Series(signals, index=df.index, dtype=int, name='signal')
