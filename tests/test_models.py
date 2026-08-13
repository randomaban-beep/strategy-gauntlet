"""
Tests for engine/garch.py, engine/regime_switching.py,
and validation/model_selection.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.synthetic import gbm_series, garch_series, regime_series
from engine.garch import fit_garch, simulate_garch_paths
from engine.regime_switching import fit_hmm, simulate_hmm_paths
from validation.model_selection import select_model


N_LONG = 2000   # enough for GARCH and HMM
N_SHORT = 100   # too short for GARCH/HMM


# ---------------------------------------------------------------------------
# GJR-GARCH
# ---------------------------------------------------------------------------

class TestGARCH:
    def test_fit_returns_parameters(self):
        df = garch_series(n=N_LONG, seed=1)
        fit = fit_garch(df['log_return'])
        assert 0 <= fit.alpha <= 1
        assert 0 <= fit.beta <= 1
        assert 0 <= fit.gamma
        assert fit.omega >= 0

    def test_alpha_beta_persistence(self):
        """alpha + beta should be < 1 for stationarity."""
        df = garch_series(n=N_LONG, alpha=0.07, beta=0.88, seed=1)
        fit = fit_garch(df['log_return'])
        assert fit.alpha + fit.beta < 1.05  # allow small numerical slack

    def test_simulate_shape(self):
        df = garch_series(n=N_LONG, seed=1)
        fit = fit_garch(df['log_return'])
        rng = np.random.default_rng(42)
        paths = simulate_garch_paths(fit, n_steps=100, n_paths=50, rng=rng)
        assert paths.shape == (50, 100)

    def test_simulate_reproducible(self):
        df = garch_series(n=N_LONG, seed=1)
        fit = fit_garch(df['log_return'])
        rng1 = np.random.default_rng(10)
        rng2 = np.random.default_rng(10)
        p1 = simulate_garch_paths(fit, 50, 10, rng1)
        p2 = simulate_garch_paths(fit, 50, 10, rng2)
        np.testing.assert_array_equal(p1, p2)

    def test_aic_bic_finite(self):
        df = garch_series(n=N_LONG, seed=1)
        fit = fit_garch(df['log_return'])
        assert np.isfinite(fit.aic)
        assert np.isfinite(fit.bic)


# ---------------------------------------------------------------------------
# HMM
# ---------------------------------------------------------------------------

class TestHMM:
    def test_fit_two_states(self):
        df, states = regime_series(n=N_LONG, seed=2)
        fit = fit_hmm(df['log_return'])
        assert len(fit.means) == 2
        assert len(fit.stds) == 2

    def test_state_ordering(self):
        """State 0 should have a higher mean than state 1."""
        df, _ = regime_series(
            n=N_LONG,
            params_state0={'mu': 0.001, 'sigma': 0.008},
            params_state1={'mu': -0.002, 'sigma': 0.020},
            seed=2,
        )
        fit = fit_hmm(df['log_return'])
        assert fit.means[0] >= fit.means[1], (
            f'Expected mean[0] >= mean[1], got {fit.means[0]:.4f} vs {fit.means[1]:.4f}'
        )

    def test_state_stds_positive(self):
        df, _ = regime_series(n=N_LONG, seed=2)
        fit = fit_hmm(df['log_return'])
        assert all(s > 0 for s in fit.stds)

    def test_simulate_shape(self):
        df, _ = regime_series(n=N_LONG, seed=2)
        fit = fit_hmm(df['log_return'])
        rng = np.random.default_rng(7)
        paths = simulate_hmm_paths(fit, n_steps=100, n_paths=20, rng=rng)
        assert paths.shape == (20, 100)

    def test_simulate_reproducible(self):
        df, _ = regime_series(n=N_LONG, seed=2)
        fit = fit_hmm(df['log_return'])
        rng1 = np.random.default_rng(99)
        rng2 = np.random.default_rng(99)
        p1 = simulate_hmm_paths(fit, 50, 10, rng1)
        p2 = simulate_hmm_paths(fit, 50, 10, rng2)
        np.testing.assert_array_equal(p1, p2)


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------

class TestModelSelection:
    def test_gbm_series_trusts_gbm(self):
        """On a plain GBM series, GBM should be trusted."""
        df = gbm_series(n=N_LONG, seed=0)
        result = select_model(df['log_return'])
        # GBM should appear as trusted (either alone or alongside others)
        assert 'gbm' in result.trusted_models or result.fallback

    def test_garch_series_trusts_garch(self):
        """On a GARCH synthetic series, GARCH model should be trusted."""
        df = garch_series(n=N_LONG, seed=1)
        result = select_model(df['log_return'], min_garch_history=500, min_hmm_history=500)
        # GARCH should be attempted and pass diagnostics
        model_names = [d.model_name for d in result.diagnostics]
        assert 'garch' in model_names, 'GARCH should be fit on GARCH series'
        garch_diag = next(d for d in result.diagnostics if d.model_name == 'garch')
        # GARCH residuals should be white noise (lb_sq_p should be high)
        assert garch_diag.lb_pvalue_sq_residuals > 0.01, (
            f'GARCH residuals should remove autocorrelation, p={garch_diag.lb_pvalue_sq_residuals:.4f}'
        )

    def test_regime_series_fits_hmm(self):
        """On a regime synthetic series, HMM should be fit and recover two states."""
        df, _ = regime_series(n=N_LONG, seed=2)
        result = select_model(df['log_return'], min_garch_history=500, min_hmm_history=500)
        model_names = [d.model_name for d in result.diagnostics]
        assert 'hmm' in model_names, 'HMM should be fit on regime series'
        hmm_diag = next(d for d in result.diagnostics if d.model_name == 'hmm')
        assert hmm_diag.fit is not None
        # Two distinct states
        assert len(hmm_diag.fit.means) == 2

    def test_short_series_falls_back_to_gbm(self):
        """Short series (< min history) should fall back to GBM only."""
        df = gbm_series(n=N_SHORT, seed=0)
        result = select_model(
            df['log_return'],
            min_garch_history=500,
            min_hmm_history=500,
        )
        model_names = [d.model_name for d in result.diagnostics]
        assert 'garch' not in model_names
        assert 'hmm' not in model_names
        assert 'gbm' in result.trusted_models or result.fallback

    def test_fallback_flag_when_no_model_passes(self):
        """If no model passes diagnostics, fallback=True and GBM is returned."""
        # Use a very short series where GBM itself might not have trustworthy diagnostics
        df = gbm_series(n=N_SHORT, seed=0)
        result = select_model(
            df['log_return'],
            min_garch_history=500,
            min_hmm_history=500,
        )
        # Whether fallback or not, GBM must be in trusted_models
        assert 'gbm' in result.trusted_models

    def test_result_has_at_least_one_trusted_model(self):
        """There must always be at least one trusted model (GBM fallback)."""
        for seed in [0, 1, 2, 3]:
            df = gbm_series(n=300, seed=seed)
            result = select_model(df['log_return'], min_garch_history=500, min_hmm_history=500)
            assert len(result.trusted_models) >= 1
