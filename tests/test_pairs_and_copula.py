"""
Tests for strategies/stat_arb.py and engine/copula.py (Batch 8: Pairs trading).
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from data.synthetic import cointegrated_pair, uncorrelated_pair
from engine.copula import (
    empirical_lower_tail_dependence,
    fit_student_t_copula,
    simulate_copula_pair_paths,
    simulate_gaussian_copula_uniforms,
    simulate_student_t_copula_uniforms,
)
from engine.monte_carlo import evaluate_pair_strategy_on_paths, run_pair_monte_carlo
from main import main
from pipeline.runner import run_pipeline
from pipeline.state import PipelineState
from strategies.stat_arb import StatArb, build_pair_dataframe, engle_granger_coint


class TestStatArbCointegration:
    def test_cointegrated_pair_accepted_and_uncorrelated_rejected(self):
        """Cointegrated synthetic pair is accepted; uncorrelated pair is rejected."""
        df_a_coint, df_b_coint = cointegrated_pair(n=1000, beta=0.8, seed=3)
        res_coint = engle_granger_coint(df_a_coint["close"], df_b_coint["close"])
        assert res_coint.is_cointegrated
        assert res_coint.pvalue < 0.05
        assert res_coint.hedge_ratio > 0.0

        pair_coint_df = build_pair_dataframe(df_a_coint, df_b_coint, hedge_ratio=res_coint.hedge_ratio)
        strat_coint = StatArb(hedge_ratio=res_coint.hedge_ratio, is_cointegrated=res_coint.is_cointegrated)
        sig_coint = strat_coint.generate_signals(pair_coint_df, window=20, entry_threshold=1.5, exit_threshold=0.5)
        assert set(sig_coint.unique()).issubset({-1, 0, 1})
        assert (sig_coint != 0).any(), "Valid cointegrated pair should generate non-zero signals"

        # Uncorrelated pair (negative control)
        df_a_unc, df_b_unc = uncorrelated_pair(n=1000, seed=4)
        res_unc = engle_granger_coint(df_a_unc["close"], df_b_unc["close"])
        assert not res_unc.is_cointegrated
        assert res_unc.pvalue >= 0.05

        pair_unc_df = build_pair_dataframe(df_a_unc, df_b_unc, hedge_ratio=res_unc.hedge_ratio)
        strat_unc = StatArb(hedge_ratio=res_unc.hedge_ratio, is_cointegrated=res_unc.is_cointegrated)
        sig_unc = strat_unc.generate_signals(pair_unc_df, window=20, entry_threshold=1.5, exit_threshold=0.5)
        assert (sig_unc == 0).all(), "Rejected uncorrelated pair must produce all-zero signals"

    def test_stat_arb_no_lookahead(self):
        """Truncating pair data at cutoff >= calib_bars produces identical signals."""
        df_a, df_b = cointegrated_pair(n=600, beta=0.8, seed=3)
        pair_df = build_pair_dataframe(df_a, df_b)
        strat = StatArb(calib_bars=200)
        cutoff = 350
        sig_full = strat.generate_signals(pair_df, window=20, entry_threshold=1.5, exit_threshold=0.5)
        sig_trunc = strat.generate_signals(pair_df.iloc[:cutoff], window=20, entry_threshold=1.5, exit_threshold=0.5)
        pd.testing.assert_series_equal(sig_full.iloc[:cutoff], sig_trunc)

    def test_pipeline_accepts_cointegrated_and_rejects_uncorrelated(self):
        """In the full pipeline, cointegrated pair passes while uncorrelated pair is rejected."""
        # 1. Cointegrated pair in pipeline
        state_coint = PipelineState(
            tickers=[],
            pairs=[("COINT_A", "COINT_B")],
            confidence_threshold=0.55,
            dsr_threshold=0.65,
            seed=3,
            synthetic=True,
            pure_noise=False,
            n_mc_paths=80,
        )
        run_pipeline(state_coint)
        assert len(state_coint.attempts_log) == 1
        att_coint = state_coint.attempts_log[0]
        assert att_coint.approved, f"Expected cointegrated pair to be approved, rejected at {att_coint.rejection_stage}: {att_coint.rejection_reason}"

        # 2. Uncorrelated pair in pipeline
        state_unc = PipelineState(
            tickers=[],
            pairs=[("UNC_A", "UNC_B")],
            confidence_threshold=0.55,
            dsr_threshold=0.65,
            seed=4,
            synthetic=True,
            pure_noise=True,
            n_mc_paths=80,
        )
        run_pipeline(state_unc)
        assert len(state_unc.attempts_log) == 1
        att_unc = state_unc.attempts_log[0]
        assert not att_unc.approved
        assert att_unc.rejection_stage is not None


class TestBivariateStudentTCopula:
    def test_copula_reproduces_input_correlation_approximately(self):
        """Simulated Student-t copula draws reproduce input correlation within tolerance."""
        target_rho = 0.72
        rng = np.random.default_rng(42)
        u = simulate_student_t_copula_uniforms(rho=target_rho, df=5.0, n_samples=25_000, rng=rng)
        # Convert uniform marginals via Kendall's tau -> implied rho = sin(pi * tau / 2)
        from scipy import stats
        tau, _ = stats.kendalltau(u[:, 0], u[:, 1])
        implied_rho = float(np.sin(0.5 * np.pi * tau))
        assert abs(implied_rho - target_rho) < 0.04, f"Expected ~{target_rho}, got {implied_rho:.4f}"

    def test_tail_dependence_greater_than_gaussian_copula(self):
        """Student-t copula (low df) exhibits strictly stronger lower-tail dependence than Gaussian copula."""
        rho = 0.50
        n_samples = 80_000
        rng_t = np.random.default_rng(100)
        rng_g = np.random.default_rng(100)

        u_t = simulate_student_t_copula_uniforms(rho=rho, df=3.0, n_samples=n_samples, rng=rng_t)
        u_g = simulate_gaussian_copula_uniforms(rho=rho, n_samples=n_samples, rng=rng_g)

        lambda_t = empirical_lower_tail_dependence(u_t, quantile=0.02)
        lambda_g = empirical_lower_tail_dependence(u_g, quantile=0.02)

        assert lambda_t > lambda_g + 0.05, (
            f"Expected Student-t tail dependence ({lambda_t:.4f}) > Gaussian ({lambda_g:.4f})"
        )

    def test_fit_and_evaluate_pair_strategy_on_paths(self):
        """fit_student_t_copula and evaluate_pair_strategy_on_paths run deterministically."""
        df_a, df_b = cointegrated_pair(n=600, beta=0.8, seed=10)
        pair_df = build_pair_dataframe(df_a, df_b)
        fit = fit_student_t_copula(pair_df["log_return_a"], pair_df["log_return_b"])
        assert -1.0 <= fit.rho <= 1.0
        assert fit.df >= 3.0

        rng = np.random.default_rng(55)
        paths_a, paths_b = simulate_copula_pair_paths(
            fit, n_steps=120, n_paths=30, rng=rng, hedge_ratio=float(pair_df["hedge_ratio"].iloc[0])
        )
        strat = StatArb(hedge_ratio=float(pair_df["hedge_ratio"].iloc[0]), is_cointegrated=True)
        params = {"window": 20, "entry_threshold": 1.5, "exit_threshold": 0.5, "allow_short": True}
        tot_rets = evaluate_pair_strategy_on_paths(
            strat, params, paths_a, paths_b, hedge_ratio=float(pair_df["hedge_ratio"].iloc[0])
        )
        assert tot_rets.shape == (30,)

        mc = run_pair_monte_carlo(strat, params, pair_df, n_paths=30, seed=55, n_steps=120)
        assert "student_t_copula" in mc
        assert "bootstrap" in mc

    def test_cli_with_pairs(self, tmp_path: Path):
        """Verify `python main.py --synthetic --pairs A,B` runs end-to-end."""
        out_file = tmp_path / "pairs_report.xlsx"
        rc = main([
            "--synthetic",
            "--tickers", "SYN_TREND_A",
            "--pairs", "A,B",
            "--output", str(out_file),
        ])
        assert rc == 0
        assert out_file.exists()
