"""
Tests for engine/monte_carlo.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.synthetic import gbm_series
from engine.monte_carlo import (
    RiskMetrics,
    compute_risk_metrics,
    evaluate_strategy_on_paths,
    passes_risk_gate,
    run_monte_carlo,
    simulate_block_bootstrap_paths,
    simulate_gbm_paths,
)
from strategies.ma_crossover import MACrossover
from validation.model_selection import GBMFit


# ---------------------------------------------------------------------------
# simulate_gbm_paths
# ---------------------------------------------------------------------------

class TestSimulateGBMPaths:
    def test_shape(self):
        rng = np.random.default_rng(0)
        paths = simulate_gbm_paths(0.0, 0.01, n_steps=100, n_paths=50, rng=rng)
        assert paths.shape == (50, 100)

    def test_reproducible(self):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        p1 = simulate_gbm_paths(0.0, 0.01, 50, 20, rng1)
        p2 = simulate_gbm_paths(0.0, 0.01, 50, 20, rng2)
        np.testing.assert_array_equal(p1, p2)

    def test_different_seed_different_paths(self):
        rng1 = np.random.default_rng(1)
        rng2 = np.random.default_rng(2)
        p1 = simulate_gbm_paths(0.0, 0.01, 50, 20, rng1)
        p2 = simulate_gbm_paths(0.0, 0.01, 50, 20, rng2)
        assert not np.allclose(p1, p2)


# ---------------------------------------------------------------------------
# simulate_block_bootstrap_paths
# ---------------------------------------------------------------------------

class TestBlockBootstrap:
    def test_shape(self):
        hist = np.random.default_rng(5).standard_normal(500) * 0.01
        rng = np.random.default_rng(5)
        paths = simulate_block_bootstrap_paths(hist, n_steps=100, n_paths=30, block_length=20, rng=rng)
        assert paths.shape == (30, 100)

    def test_reproducible(self):
        hist = np.random.default_rng(5).standard_normal(500) * 0.01
        rng1 = np.random.default_rng(99)
        rng2 = np.random.default_rng(99)
        p1 = simulate_block_bootstrap_paths(hist, 100, 30, 20, rng1)
        p2 = simulate_block_bootstrap_paths(hist, 100, 30, 20, rng2)
        np.testing.assert_array_equal(p1, p2)

    def test_values_from_historical(self):
        """All values in bootstrapped paths should come from historical returns."""
        hist = np.array([0.01, 0.02, -0.01, 0.03, -0.02])
        rng = np.random.default_rng(7)
        paths = simulate_block_bootstrap_paths(hist, n_steps=20, n_paths=10, block_length=2, rng=rng)
        # Every value must be from hist
        for val in paths.flatten():
            assert any(np.isclose(val, h) for h in hist), f'{val} not in hist'


# ---------------------------------------------------------------------------
# compute_risk_metrics
# ---------------------------------------------------------------------------

class TestComputeRiskMetrics:
    def test_returns_risk_metrics_instance(self):
        arr = np.random.default_rng(0).standard_normal(100) * 0.01
        m = compute_risk_metrics(arr)
        assert isinstance(m, RiskMetrics)

    def test_prob_of_profit_in_range(self):
        arr = np.random.default_rng(1).standard_normal(200)
        m = compute_risk_metrics(arr)
        assert 0.0 <= m.prob_of_profit <= 1.0

    def test_cvar_leq_var_for_losing_strategy(self):
        """CVaR99 <= VaR99 <= 0 for a strongly negative strategy."""
        arr = -np.abs(np.random.default_rng(2).standard_normal(500)) - 0.1
        m = compute_risk_metrics(arr)
        assert m.cvar99 <= m.var99 + 1e-9
        assert m.var99 <= 0.0 + 1e-9

    def test_positive_drift_high_prob_of_profit(self):
        """A strongly positive drift series should give high prob of profit."""
        # All paths have positive total returns
        arr = np.abs(np.random.default_rng(3).standard_normal(500)) + 0.5
        m = compute_risk_metrics(arr)
        assert m.prob_of_profit > 0.8

    def test_same_seed_same_metrics(self):
        arr1 = np.random.default_rng(10).standard_normal(100)
        arr2 = np.random.default_rng(10).standard_normal(100)
        m1 = compute_risk_metrics(arr1)
        m2 = compute_risk_metrics(arr2)
        assert m1.mean_total_return == m2.mean_total_return

    def test_empty_array(self):
        m = compute_risk_metrics(np.array([]))
        assert m.prob_of_profit == 0.0


# ---------------------------------------------------------------------------
# passes_risk_gate
# ---------------------------------------------------------------------------

class TestRiskGate:
    def test_passes_when_all_above_threshold(self):
        mc = {
            'gbm': RiskMetrics(0.1, 1.0, 1.5, -0.05, -0.10, -0.12, 0.70),
            'bootstrap': RiskMetrics(0.1, 0.9, 1.2, -0.06, -0.11, -0.13, 0.65),
        }
        passed, reason = passes_risk_gate(mc, confidence_threshold=0.60)
        assert passed

    def test_fails_when_one_below_threshold(self):
        mc = {
            'gbm': RiskMetrics(0.1, 1.0, 1.5, -0.05, -0.10, -0.12, 0.70),
            'garch': RiskMetrics(-0.05, -0.5, -0.8, -0.20, -0.25, -0.30, 0.40),
        }
        passed, reason = passes_risk_gate(mc, confidence_threshold=0.60)
        assert not passed
        assert 'garch' in reason

    def test_empty_results_fails(self):
        passed, reason = passes_risk_gate({}, confidence_threshold=0.60)
        assert not passed


# ---------------------------------------------------------------------------
# run_monte_carlo (integration)
# ---------------------------------------------------------------------------

class TestRunMonteCarlo:
    def setup_method(self):
        self.df = gbm_series(n=500, mu=0.05, sigma=0.15, seed=0)
        self.strat = MACrossover()
        self.params = {'fast_window': 10, 'slow_window': 50, 'allow_short': True}
        self.hist_rets = self.df['log_return']
        self.gbm_fit = GBMFit(
            mu=float(self.hist_rets.mean()),
            sigma=float(self.hist_rets.std()),
            aic=0.0, bic=0.0,
        )

    def test_returns_dict_with_bootstrap(self):
        results = run_monte_carlo(
            self.strat, self.params,
            trusted_models={'gbm': self.gbm_fit},
            historical_returns=self.hist_rets,
            n_paths=50, seed=42, n_steps=100,
        )
        assert 'bootstrap' in results
        assert 'gbm' in results

    def test_reproducible_with_same_seed(self):
        kwargs = dict(
            strategy=self.strat,
            params=self.params,
            trusted_models={'gbm': self.gbm_fit},
            historical_returns=self.hist_rets,
            n_paths=50, seed=42, n_steps=100,
        )
        r1 = run_monte_carlo(**kwargs)
        r2 = run_monte_carlo(**kwargs)
        assert r1['bootstrap'].prob_of_profit == r2['bootstrap'].prob_of_profit
        assert r1['gbm'].prob_of_profit == r2['gbm'].prob_of_profit

    def test_risk_gate_rejects_when_one_fails(self):
        """Gate must reject if any model's prob_of_profit is below threshold."""
        mc = {
            'gbm': RiskMetrics(0.1, 1.0, 1.5, -0.05, -0.10, -0.12, 0.80),
            'bootstrap': RiskMetrics(-0.1, -1.0, -1.5, -0.30, -0.35, -0.40, 0.30),  # fails
        }
        passed, _ = passes_risk_gate(mc, confidence_threshold=0.60)
        assert not passed
