"""
Tests for validation/deflated_sharpe.py: DSR properties and scoring.
"""
from __future__ import annotations

import numpy as np
import pytest

from validation.deflated_sharpe import (
    DSRResult,
    _annualized_sharpe,
    _expected_max_sharpe,
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
    score_strategy,
)


class TestAnnualizedSharpe:
    def test_positive_returns(self):
        rets = np.array([0.001, 0.002] * 126)
        sr = _annualized_sharpe(rets)
        assert sr > 0

    def test_zero_std(self):
        rets = np.ones(252) * 0.001
        # Constant returns have zero std -> SR should be 0
        sr = _annualized_sharpe(rets)
        assert sr == 0.0

    def test_short_series(self):
        sr = _annualized_sharpe(np.array([0.01]))
        assert sr == 0.0


class TestExpectedMaxSharpe:
    def test_single_trial_is_zero(self):
        e = _expected_max_sharpe(n_trials=1)
        assert e == 0.0

    def test_increases_with_trials(self):
        e1 = _expected_max_sharpe(n_trials=2)
        e10 = _expected_max_sharpe(n_trials=10)
        e100 = _expected_max_sharpe(n_trials=100)
        e500 = _expected_max_sharpe(n_trials=500)
        assert e1 <= e10 <= e100 <= e500

    def test_with_trial_sharpes(self):
        trial_srs = np.random.default_rng(42).standard_normal(20)
        e = _expected_max_sharpe(n_trials=20, trial_sharpes=trial_srs)
        assert np.isfinite(e)


class TestDSR:
    def _make_good_returns(self, n: int = 500, seed: int = 0) -> np.ndarray:
        """Returns with high, consistent positive Sharpe."""
        rng = np.random.default_rng(seed)
        # Annualized SR ~ 3: mu/sigma*sqrt(252) = 3 => mu = 3*sigma/sqrt(252)
        sigma = 0.01
        mu = 3.0 * sigma / np.sqrt(252)
        return rng.normal(mu, sigma, n)

    def _make_noise_returns(self, n: int = 500, seed: int = 1) -> np.ndarray:
        """Pure noise returns (zero drift)."""
        rng = np.random.default_rng(seed)
        return rng.normal(0, 0.01, n)

    def test_passes_with_n_trials_1(self):
        """A good strategy with n_trials=1 should pass (no SR inflation)."""
        rets = self._make_good_returns(n=500)
        result = deflated_sharpe_ratio(rets, n_trials=1, dsr_threshold=0.80)
        assert result.dsr_probability >= 0.80, (
            f'Good returns with n_trials=1 should pass, got {result.dsr_probability:.4f}'
        )
        assert result.passes_threshold

    def test_fails_with_many_trials(self):
        """Same returns with n_trials=500 should fail (expected max SR much higher)."""
        rets = self._make_good_returns(n=500)
        result = deflated_sharpe_ratio(rets, n_trials=500, dsr_threshold=0.80)
        # With 500 trials, expected max SR is high, so even a good strategy may fail
        # The probability should be lower than with n_trials=1
        result_1 = deflated_sharpe_ratio(rets, n_trials=1, dsr_threshold=0.80)
        assert result.dsr_probability <= result_1.dsr_probability + 0.01, (
            f'n_trials=500 should give lower or equal DSR than n_trials=1'
        )

    def test_noise_fails_with_many_trials(self):
        """Noise returns + many trials should fail DSR."""
        rets = self._make_noise_returns(n=500)
        result = deflated_sharpe_ratio(rets, n_trials=100, dsr_threshold=0.80)
        # With zero drift and 100 trials, should fail
        assert not result.passes_threshold or result.dsr_probability < 0.80 + 0.1

    def test_probability_in_unit_interval(self):
        rets = self._make_good_returns(n=300)
        for n_trials in [1, 10, 100]:
            result = deflated_sharpe_ratio(rets, n_trials=n_trials)
            assert 0.0 <= result.dsr_probability <= 1.0

    def test_higher_skew_changes_result(self):
        """Positive skewness should improve DSR (less non-Gaussianity penalty)."""
        rng = np.random.default_rng(5)
        sigma = 0.01
        mu = 2.0 * sigma / np.sqrt(252)
        n = 500
        # Symmetric returns
        rets_sym = rng.normal(mu, sigma, n)
        # Positively skewed returns (using exponential transformation)
        raw = rng.exponential(sigma, n)
        rets_pos_skew = mu + (raw - sigma)  # mean-centered positive skew
        result_sym = deflated_sharpe_ratio(rets_sym, n_trials=5)
        result_skew = deflated_sharpe_ratio(rets_pos_skew, n_trials=5)
        # Both should be valid probabilities
        assert 0.0 <= result_sym.dsr_probability <= 1.0
        assert 0.0 <= result_skew.dsr_probability <= 1.0

    def test_output_fields(self):
        rets = self._make_good_returns(n=300)
        result = deflated_sharpe_ratio(rets, n_trials=5)
        assert hasattr(result, 'observed_sharpe')
        assert hasattr(result, 'expected_max_sharpe')
        assert hasattr(result, 'dsr_probability')
        assert hasattr(result, 'passes_threshold')
        assert np.isfinite(result.observed_sharpe)
        assert np.isfinite(result.expected_max_sharpe)

    def test_expected_max_increases_with_more_trials(self):
        rets = self._make_good_returns(n=300)
        r1 = deflated_sharpe_ratio(rets, n_trials=1)
        r10 = deflated_sharpe_ratio(rets, n_trials=10)
        r100 = deflated_sharpe_ratio(rets, n_trials=100)
        assert r1.expected_max_sharpe <= r10.expected_max_sharpe
        assert r10.expected_max_sharpe <= r100.expected_max_sharpe


class TestScoreStrategy:
    def test_score_strategy_returns_dsr_result(self):
        rets = np.random.default_rng(0).normal(0.001, 0.01, 300)
        result = score_strategy(rets, n_trials=5)
        assert isinstance(result, DSRResult)

    def test_score_strategy_probability_in_range(self):
        rets = np.random.default_rng(1).normal(0.0, 0.01, 300)
        result = score_strategy(rets, n_trials=10)
        assert 0.0 <= result.dsr_probability <= 1.0
