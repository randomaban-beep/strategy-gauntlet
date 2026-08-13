"""
Tests for allocation/hrp.py: Hierarchical Risk Parity.
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import pytest

from allocation.hrp import HRPResult, allocate_hrp


class TestHRPProperties:
    def test_single_strategy_weight_one(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"strat_a": rng.normal(0.0, 0.01, 200)})
        res = allocate_hrp(df)
        assert isinstance(res, HRPResult)
        assert res.weights == {"strat_a": 1.0}

    def test_weights_sum_to_one_and_non_negative(self):
        rng = np.random.default_rng(1)
        df = pd.DataFrame({
            f"s_{i}": rng.normal(0.0, 0.01 * (i + 1), 300)
            for i in range(5)
        })
        res = allocate_hrp(df)
        assert len(res.weights) == 5
        assert all(w >= 0.0 for w in res.weights.values())
        assert np.isclose(sum(res.weights.values()), 1.0, atol=1e-9)

    def test_lower_variance_gets_higher_weight(self):
        rng = np.random.default_rng(2)
        df = pd.DataFrame({
            "low_vol": rng.normal(0.0, 0.005, 500),
            "high_vol": rng.normal(0.0, 0.030, 500),
        })
        res = allocate_hrp(df)
        assert res.weights["low_vol"] > res.weights["high_vol"]

    def test_identical_streams_get_equal_weights(self):
        rng = np.random.default_rng(3)
        stream = rng.normal(0.0005, 0.01, 300)
        df = pd.DataFrame({"a": stream, "b": stream.copy()})
        res = allocate_hrp(df)
        assert np.isclose(res.weights["a"], 0.5, atol=1e-6)
        assert np.isclose(res.weights["b"], 0.5, atol=1e-6)

    @pytest.mark.parametrize("n_strats", list(range(2, 11)))
    def test_works_for_2_to_10_strategies(self, n_strats: int):
        rng = np.random.default_rng(10 + n_strats)
        df = pd.DataFrame({
            f"s_{i}": rng.normal(0.0, 0.01 * (1.0 + 0.2 * i), 250)
            for i in range(n_strats)
        })
        res = allocate_hrp(df)
        assert len(res.weights) == n_strats
        assert all(w > 0.0 for w in res.weights.values())
        assert np.isclose(sum(res.weights.values()), 1.0, atol=1e-9)

    def test_deterministic(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            f"s_{i}": rng.normal(0.0, 0.01 * (i + 1), 250)
            for i in range(6)
        })
        res1 = allocate_hrp(df)
        res2 = allocate_hrp(df)
        assert res1.weights == res2.weights
        assert res1.ordered_assets == res2.ordered_assets

    def test_constant_and_nan_columns_dropped_with_warning(self):
        rng = np.random.default_rng(99)
        df = pd.DataFrame({
            "good_1": rng.normal(0.0, 0.01, 200),
            "good_2": rng.normal(0.0, 0.02, 200),
            "const_col": np.ones(200) * 0.001,
            "nan_col": np.full(200, np.nan),
        })
        with pytest.warns(UserWarning):
            res = allocate_hrp(df)
        assert "const_col" in res.dropped_assets
        assert "nan_col" in res.dropped_assets
        assert set(res.weights.keys()) == {"good_1", "good_2"}
        assert np.isclose(sum(res.weights.values()), 1.0, atol=1e-9)
