"""
Market data ingestion, local caching, and seeded synthetic generators.
"""
from __future__ import annotations

from data.fetch import enrich_dataframe, fetch_ticker, load_universe
from data.synthetic import cointegrated_pair, garch_series, gbm_series, regime_series, uncorrelated_pair

__all__ = [
    "enrich_dataframe",
    "fetch_ticker",
    "load_universe",
    "gbm_series",
    "garch_series",
    "regime_series",
    "cointegrated_pair",
    "uncorrelated_pair",
]
