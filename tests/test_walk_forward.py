"""
Tests for validation/costs.py and validation/walk_forward.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from data.synthetic import gbm_series, garch_series
from strategies.ma_crossover import MACrossover
from validation.costs import apply_costs
from validation.walk_forward import walk_forward, WalkForwardResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_trend_series(n: int = 2000, seed: int = 99) -> pd.DataFrame:
    """GBM with strong positive drift so MA strategy has an edge."""
    return gbm_series(n=n, mu=0.30, sigma=0.10, seed=seed)


def make_noise_series(n: int = 2000, seed: int = 77) -> pd.DataFrame:
    """GBM with near-zero drift (pure noise)."""
    return gbm_series(n=n, mu=0.0, sigma=0.20, seed=seed)


# ---------------------------------------------------------------------------
# Costs tests
# ---------------------------------------------------------------------------

class TestApplyCosts:
    def test_costs_are_non_negative(self):
        """Net returns should be <= gross returns (costs always reduce returns)."""
        returns = pd.Series([0.01, -0.02, 0.03, 0.0, 0.01])
        positions = pd.Series([1, 1, 0, -1, 0])
        net = apply_costs(returns, positions)
        assert (net <= returns + 1e-12).all(), 'Costs should never be negative'

    def test_zero_turnover_no_cost(self):
        """Constant position should incur no cost after the first bar."""
        returns = pd.Series([0.01, 0.01, 0.01, 0.01])
        positions = pd.Series([1, 1, 1, 1])
        # Only bar 0 has a trade (from 0 -> 1)
        net = apply_costs(returns, positions)
        # bars 1-3 should have returns equal to gross
        np.testing.assert_allclose(net.iloc[1:].values, returns.iloc[1:].values, atol=1e-10)

    def test_higher_costs_lower_returns(self):
        """Increasing spread_bps should monotonically decrease net returns."""
        returns = pd.Series([0.01] * 100)
        positions = pd.Series(np.tile([1, -1], 50))  # high turnover
        net_low = apply_costs(returns, positions, spread_bps=1.0).sum()
        net_high = apply_costs(returns, positions, spread_bps=50.0).sum()
        assert net_high < net_low

    def test_costs_shape_preserved(self):
        returns = pd.Series([0.1, -0.2, 0.3])
        positions = pd.Series([1, 0, -1])
        net = apply_costs(returns, positions)
        assert len(net) == len(returns)


# ---------------------------------------------------------------------------
# Walk-forward tests
# ---------------------------------------------------------------------------

class TestWalkForward:
    def test_noise_series_oos_sharpe_near_zero(self):
        """Pure-noise series should produce near-zero OOS Sharpe."""
        df = make_noise_series(n=2000, seed=77)
        strat = MACrossover()
        result = walk_forward(
            df, strat,
            train_bars=252, test_bars=63, embargo_bars=10,
        )
        # Should have at least 2 folds and aggregate OOS Sharpe should be low
        assert len(result.folds) >= 2
        # No strong threshold – just assert it's not clearly positive
        assert result.aggregate_sharpe < 2.0

    def test_trend_series_passes_threshold(self):
        """A series with strong trend should produce positive OOS Sharpe."""
        df = make_trend_series(n=2000, seed=99)
        strat = MACrossover()
        result = walk_forward(
            df, strat,
            train_bars=252, test_bars=63, embargo_bars=10,
        )
        assert result.aggregate_sharpe > 0.0
        assert result.total_return > 0.0

    def test_no_overlap_between_train_and_test(self):
        """Train and test windows must not overlap; embargo gap is enforced."""
        df = make_noise_series(n=2000)
        strat = MACrossover()
        embargo = 10
        result = walk_forward(
            df, strat,
            train_bars=252, test_bars=63, embargo_bars=embargo,
        )
        for fold in result.folds:
            # Train end <= test start - embargo
            assert fold.train_end <= fold.test_start - embargo, (
                f'Fold {fold.fold_idx}: train_end={fold.train_end}, '
                f'test_start={fold.test_start}, embargo={embargo}'
            )
            # No overlap at all
            assert fold.train_end <= fold.test_start

    def test_embargo_gap_equals_configured_value(self):
        """The gap between train_end and test_start must equal embargo_bars."""
        df = make_noise_series(n=2000)
        strat = MACrossover()
        embargo = 15
        result = walk_forward(
            df, strat,
            train_bars=252, test_bars=63, embargo_bars=embargo,
        )
        for fold in result.folds:
            gap = fold.test_start - fold.train_end
            assert gap == embargo, (
                f'Fold {fold.fold_idx}: expected embargo gap={embargo}, got {gap}'
            )

    def test_higher_costs_lower_oos_returns(self):
        """Higher costs should reduce OOS total return."""
        df = make_trend_series(n=2000)
        strat = MACrossover()
        result_low = walk_forward(
            df, strat, train_bars=252, test_bars=63, embargo_bars=10,
            spread_bps=1.0,
        )
        result_high = walk_forward(
            df, strat, train_bars=252, test_bars=63, embargo_bars=10,
            spread_bps=100.0,
        )
        assert result_high.total_return < result_low.total_return

    def test_too_short_raises_error(self):
        """Data too short for even 2 folds should raise ValueError."""
        df = make_noise_series(n=350)  # only 1 fold fits (needs 388 for 2 folds: 252+10+63+63)
        strat = MACrossover()
        with pytest.raises(ValueError):
            walk_forward(df, strat, train_bars=252, test_bars=63, embargo_bars=10)

    def test_oos_returns_length_matches_folds(self):
        """Concatenated OOS returns should have length = sum of test windows."""
        df = make_noise_series(n=2000)
        strat = MACrossover()
        result = walk_forward(
            df, strat, train_bars=252, test_bars=63, embargo_bars=10,
        )
        expected_len = sum(f.test_end - f.test_start for f in result.folds)
        assert len(result.oos_returns) == expected_len

    def test_at_least_two_folds(self):
        """Should produce at least 2 folds when data is sufficient."""
        df = make_noise_series(n=2000)
        strat = MACrossover()
        result = walk_forward(
            df, strat, train_bars=252, test_bars=63, embargo_bars=10,
        )
        assert len(result.folds) >= 2
