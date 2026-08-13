"""
Tests for strategies/base.py, ma_crossover.py, and mean_reversion.py.

Includes:
- Signals are in {-1, 0, 1}
- No look-ahead test: truncating data at t gives same signals as full series
- A deliberately look-ahead strategy is caught by the same test
- strategy_returns shifts by one bar
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.synthetic import gbm_series
from strategies.base import (
    ParamSpec,
    Strategy,
    param_grid_combinations,
    strategy_returns,
)
from strategies.ma_crossover import MACrossover
from strategies.mean_reversion import MeanReversion


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_df(n: int = 500, seed: int = 0) -> pd.DataFrame:
    return gbm_series(n=n, seed=seed)


def passes_no_lookahead(
    strategy: Strategy, df: pd.DataFrame, params: dict, cutoff: int
) -> bool:
    """Return True if signals on df[:cutoff] match full-series signals[:cutoff]."""
    sig_full = strategy.generate_signals(df, **params)
    sig_trunc = strategy.generate_signals(df.iloc[:cutoff], **params)
    return sig_full.iloc[:cutoff].equals(sig_trunc)


# ---------------------------------------------------------------------------
# Deliberately look-ahead strategy (negative control)
# ---------------------------------------------------------------------------

class _LookAheadStrategy(Strategy):
    """Strategy that cheats by using future returns — for testing only."""

    @property
    def name(self) -> str:
        return 'lookahead'

    def param_grid(self) -> list[ParamSpec]:
        return []

    def generate_signals(self, df: pd.DataFrame, **params) -> pd.Series:
        # Use the NEXT bar's return to decide signal — this is look-ahead!
        future = df['log_return'].shift(-1).fillna(0)
        return (future > 0).astype(int).rename('signal')


# ---------------------------------------------------------------------------
# param_grid_combinations
# ---------------------------------------------------------------------------

class TestParamGrid:
    def test_empty_specs(self):
        assert param_grid_combinations([]) == [{}]

    def test_single_spec(self):
        specs = [ParamSpec('a', [1, 2, 3])]
        combos = param_grid_combinations(specs)
        assert len(combos) == 3
        assert combos[0] == {'a': 1}

    def test_two_specs(self):
        specs = [ParamSpec('a', [1, 2]), ParamSpec('b', ['x', 'y'])]
        combos = param_grid_combinations(specs)
        assert len(combos) == 4
        assert {'a': 1, 'b': 'x'} in combos
        assert {'a': 2, 'b': 'y'} in combos


# ---------------------------------------------------------------------------
# strategy_returns
# ---------------------------------------------------------------------------

class TestStrategyReturns:
    def test_forward_shift(self):
        """strategy_returns applies a one-bar forward shift.

        signal[t-1] determines the position that earns log_return[t].
        shifted = [NaN->0, sig[0], sig[1], sig[2], sig[3]]
        With signals=[1,1,1,1,1] and log_rets=[0.1,0.2,0.3,0.4,0.5]:
          result = [0*0.1, 1*0.2, 1*0.3, 1*0.4, 1*0.5] = [0, 0.2, 0.3, 0.4, 0.5]
        """
        signals = pd.Series([1, 1, 1, 1, 1])
        log_rets = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])
        sr = strategy_returns(signals, log_rets)
        # shifted signals: [0, 1, 1, 1, 1] (first is 0 due to shift)
        # result[0] = 0 * 0.1 = 0.0
        # result[1] = 1 * 0.2 = 0.2  (sig[0]=1 earns return[1]=0.2)
        expected = pd.Series([0.0, 0.2, 0.3, 0.4, 0.5], name='strategy_return')
        pd.testing.assert_series_equal(sr, expected)

    def test_forward_shift_mixed(self):
        """Verify the forward shift with a mixed signal pattern."""
        # Position switches at t=2: sig=[1,-1,-1,1]
        # shifted=[0,1,-1,-1], ret=[0.1,0.2,0.3,0.4]
        # result=[0.0, 0.2, -0.3, -0.4]
        signals = pd.Series([1, -1, -1, 1])
        log_rets = pd.Series([0.1, 0.2, 0.3, 0.4])
        sr = strategy_returns(signals, log_rets)
        expected = pd.Series([0.0, 0.2, -0.3, -0.4], name='strategy_return')
        pd.testing.assert_series_equal(sr, expected)


    def test_no_short_clips_minus_one(self):
        signals = pd.Series([-1, -1, 1, 1])
        log_rets = pd.Series([0.1, 0.2, 0.3, 0.4])
        sr = strategy_returns(signals, log_rets, allow_short=False)
        assert (sr >= 0).all()

    def test_zero_signal_zero_return(self):
        signals = pd.Series([0, 0, 0, 0])
        log_rets = pd.Series([0.5, -0.3, 0.2, -0.1])
        sr = strategy_returns(signals, log_rets)
        assert (sr == 0).all()


# ---------------------------------------------------------------------------
# MACrossover
# ---------------------------------------------------------------------------

class TestMACrossover:
    def setup_method(self):
        self.strat = MACrossover()
        self.df = make_df(n=500)

    def test_signals_in_valid_set(self):
        sig = self.strat.generate_signals(self.df, fast_window=10, slow_window=50)
        assert set(sig.unique()).issubset({-1, 0, 1})

    def test_fast_ge_slow_returns_zero(self):
        sig = self.strat.generate_signals(self.df, fast_window=50, slow_window=10)
        assert (sig == 0).all()

    def test_no_short_no_minus_one(self):
        sig = self.strat.generate_signals(
            self.df, fast_window=10, slow_window=50, allow_short=False
        )
        assert -1 not in sig.values

    def test_no_lookahead_passes(self):
        cutoff = 300
        params = {'fast_window': 10, 'slow_window': 50, 'allow_short': True}
        assert passes_no_lookahead(self.strat, self.df, params, cutoff)

    def test_param_grid_has_combinations(self):
        combos = self.strat.all_param_combinations()
        # 3 fast * 4 slow * 2 allow_short = 24
        assert len(combos) == 24
        # fast < slow should still be in grid (filtering is strategy's job)
        for c in combos:
            assert 'fast_window' in c and 'slow_window' in c


# ---------------------------------------------------------------------------
# MeanReversion
# ---------------------------------------------------------------------------

class TestMeanReversion:
    def setup_method(self):
        self.strat = MeanReversion()
        self.df = make_df(n=500)

    def test_signals_in_valid_set(self):
        sig = self.strat.generate_signals(
            self.df, window=20, entry_threshold=2.0, exit_threshold=0.5
        )
        assert set(sig.unique()).issubset({-1, 0, 1})

    def test_no_short(self):
        sig = self.strat.generate_signals(
            self.df, window=20, entry_threshold=2.0,
            exit_threshold=0.5, allow_short=False
        )
        assert -1 not in sig.values

    def test_no_lookahead_passes(self):
        cutoff = 300
        params = {
            'window': 20, 'entry_threshold': 2.0,
            'exit_threshold': 0.5, 'allow_short': True,
        }
        assert passes_no_lookahead(self.strat, self.df, params, cutoff)

    def test_warm_up_is_zero(self):
        """Signals during warm-up period (< window bars) should be 0."""
        sig = self.strat.generate_signals(
            self.df, window=20, entry_threshold=2.0, exit_threshold=0.5
        )
        assert (sig.iloc[:19] == 0).all()


# ---------------------------------------------------------------------------
# Look-ahead detection
# ---------------------------------------------------------------------------

class TestLookAheadDetection:
    def test_lookahead_strategy_fails_check(self):
        """A deliberately look-ahead strategy should fail the no-lookahead check."""
        df = make_df(n=200)
        strat = _LookAheadStrategy()
        cutoff = 100
        # For a look-ahead strategy, signals on truncated data will differ
        # from the full series signals at the cutoff point (or just before)
        sig_full = strat.generate_signals(df)
        sig_trunc = strat.generate_signals(df.iloc[:cutoff])
        # At cutoff-1, full series uses return[cutoff] but truncated cannot
        # So they should differ at the boundary
        same = sig_full.iloc[:cutoff].equals(sig_trunc)
        assert not same, 'Look-ahead strategy should fail the no-look-ahead check'

    def test_ma_crossover_no_lookahead_multiple_cutoffs(self):
        """MA crossover should pass no-lookahead check at several cutoffs."""
        df = make_df(n=500)
        strat = MACrossover()
        params = {'fast_window': 10, 'slow_window': 50, 'allow_short': True}
        for cutoff in [100, 200, 350, 450]:
            assert passes_no_lookahead(strat, df, params, cutoff), (
                f'MA crossover failed no-lookahead check at cutoff={cutoff}'
            )
