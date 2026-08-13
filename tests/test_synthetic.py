"""
Tests for data/synthetic.py: reproducibility, seed sensitivity,
GARCH autocorrelation, and cointegration properties.
"""
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox

from data.synthetic import (
    gbm_series,
    garch_series,
    regime_series,
    cointegrated_pair,
    uncorrelated_pair,
)


class TestReproducibility:
    def test_gbm_same_seed(self):
        df1 = gbm_series(seed=42)
        df2 = gbm_series(seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_gbm_different_seed(self):
        df1 = gbm_series(seed=0)
        df2 = gbm_series(seed=1)
        assert not df1['close'].equals(df2['close'])

    def test_garch_same_seed(self):
        df1 = garch_series(seed=10)
        df2 = garch_series(seed=10)
        pd.testing.assert_frame_equal(df1, df2)

    def test_garch_different_seed(self):
        df1 = garch_series(seed=0)
        df2 = garch_series(seed=5)
        assert not df1['log_return'].equals(df2['log_return'])

    def test_regime_same_seed(self):
        df1, s1 = regime_series(seed=20)
        df2, s2 = regime_series(seed=20)
        pd.testing.assert_frame_equal(df1, df2)
        np.testing.assert_array_equal(s1, s2)

    def test_regime_different_seed(self):
        df1, _ = regime_series(seed=0)
        df2, _ = regime_series(seed=9)
        assert not df1['close'].equals(df2['close'])

    def test_cointegrated_same_seed(self):
        a1, b1 = cointegrated_pair(seed=30)
        a2, b2 = cointegrated_pair(seed=30)
        pd.testing.assert_frame_equal(a1, a2)
        pd.testing.assert_frame_equal(b1, b2)

    def test_uncorrelated_same_seed(self):
        a1, b1 = uncorrelated_pair(seed=40)
        a2, b2 = uncorrelated_pair(seed=40)
        pd.testing.assert_frame_equal(a1, a2)
        pd.testing.assert_frame_equal(b1, b2)


class TestColumns:
    @pytest.mark.parametrize('fn,kwargs', [
        (gbm_series, {'n': 200}),
        (garch_series, {'n': 200}),
    ])
    def test_required_columns(self, fn, kwargs):
        df = fn(**kwargs)
        assert set(['close', 'volume', 'log_return']).issubset(df.columns)
        assert len(df) == kwargs['n']

    def test_regime_columns(self):
        df, states = regime_series(n=200)
        assert set(['close', 'volume', 'log_return']).issubset(df.columns)
        assert len(df) == 200
        assert len(states) == 200
        assert set(states).issubset({0, 1})


class TestGARCHProperties:
    def test_autocorrelation_in_squared_returns(self):
        """GARCH returns should show significant autocorrelation in squared returns."""
        df = garch_series(n=3000, seed=1)
        sq_rets = df['log_return'] ** 2
        lb = acorr_ljungbox(sq_rets, lags=[10], return_df=True)
        # p-value should be small (autocorrelation present)
        p_val = lb['lb_pvalue'].iloc[0]
        assert p_val < 0.05, f'Expected autocorrelation in squared returns, p={p_val:.4f}'

    def test_garch_returns_have_fat_tails(self):
        """GARCH returns should have excess kurtosis > 0."""
        from scipy.stats import kurtosis
        df = garch_series(n=3000, seed=1)
        k = kurtosis(df['log_return'], fisher=True)
        assert k > 0, f'Expected excess kurtosis > 0, got {k:.4f}'


class TestCointegration:
    def test_cointegrated_pair_passes_adf(self):
        """Spread of cointegrated pair should be stationary (ADF rejects unit root)."""
        df_a, df_b = cointegrated_pair(n=1500, seed=3)
        import statsmodels.api as sm
        y = np.log(df_a['close'].values)
        x = np.log(df_b['close'].values)
        x_with_const = sm.add_constant(x)
        res = sm.OLS(y, x_with_const).fit()
        spread = res.resid
        adf_stat, p_val, _, _, _, _ = adfuller(spread, maxlag=5)
        assert p_val < 0.05, (
            f'Cointegrated pair spread should be stationary, ADF p={p_val:.4f}'
        )

    def test_uncorrelated_pair_fails_adf(self):
        """Spread of uncorrelated pair should be non-stationary (ADF fails to reject)."""
        df_a, df_b = uncorrelated_pair(n=1500, seed=4)
        import statsmodels.api as sm
        y = np.log(df_a['close'].values)
        x = np.log(df_b['close'].values)
        x_with_const = sm.add_constant(x)
        res = sm.OLS(y, x_with_const).fit()
        spread = res.resid
        adf_stat, p_val, _, _, _, _ = adfuller(spread, maxlag=5)
        # For a random walk, ADF should fail to reject (p > 0.10)
        assert p_val > 0.05, (
            f'Uncorrelated pair spread should be non-stationary, ADF p={p_val:.4f}'
        )
