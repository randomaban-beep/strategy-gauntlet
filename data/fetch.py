"""
Data fetching, local caching, minimum-history filtering, and feature enrichment.

Supports both live yfinance downloads (with cache in `cache/`) and offline
synthetic data generation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

import config
from data.synthetic import garch_series, gbm_series, regime_series

CACHE_DIR = Path("cache")
MIN_HISTORY_BARS = config.MIN_TRAIN_BARS + config.EMBARGO_BARS + 2 * config.MIN_TEST_BARS


def enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure close, volume, log_return, rolling_vol_20, and rolling_drift_20 exist."""
    out = df.copy()
    if "close" not in out.columns and "Close" in out.columns:
        out["close"] = out["Close"]
    if "volume" not in out.columns:
        if "Volume" in out.columns:
            out["volume"] = out["Volume"]
        else:
            out["volume"] = 1_000_000.0

    out["close"] = out["close"].astype(float)
    out["volume"] = out["volume"].astype(float)

    if "log_return" not in out.columns:
        log_p = np.log(out["close"].clip(lower=1e-12))
        out["log_return"] = log_p.diff().fillna(0.0)

    out["rolling_vol_20"] = (
        out["log_return"].rolling(20, min_periods=1).std().fillna(0.0)
        * np.sqrt(config.ANNUALIZATION)
    )
    out["rolling_drift_20"] = (
        out["log_return"].rolling(20, min_periods=1).mean().fillna(0.0)
        * config.ANNUALIZATION
    )
    return out.reset_index(drop=True)


def fetch_ticker(
    ticker: str,
    period: str = "5y",
    cache_dir: Path = CACHE_DIR,
    use_cache: bool = True,
    min_bars: int = MIN_HISTORY_BARS,
) -> pd.DataFrame | None:
    """Download or load cached data for a single ticker.

    Returns None if download fails or history is shorter than `min_bars`.
    Never raises on network/ticker errors so a single bad ticker does not kill a batch.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_name = ticker.replace("/", "_").replace("^", "_").upper()
    cache_file = cache_dir / f"{safe_name}.csv"

    df: pd.DataFrame | None = None
    if use_cache and cache_file.exists():
        try:
            df = pd.read_csv(cache_file)
        except Exception:
            df = None

    if df is None:
        try:
            import yfinance as yf
            hist = yf.Ticker(ticker).history(period=period, auto_adjust=True)
            if hist is None or hist.empty:
                return None
            df = pd.DataFrame({
                "close": hist["Close"].values,
                "volume": hist["Volume"].values if "Volume" in hist.columns else np.ones(len(hist)) * 1e6,
            })
            if use_cache:
                df.to_csv(cache_file, index=False)
        except Exception:
            return None

    if len(df) < min_bars:
        return None

    return enrich_dataframe(df)


def build_synthetic_universe(
    tickers: list[str] | None = None,
    n_bars: int = 800,
    seed: int = config.PIPELINE_SEED,
    pure_noise: bool = False,
) -> dict[str, pd.DataFrame]:
    """Generate a deterministic synthetic market dataset keyed by ticker name.

    If `pure_noise` is True, all tickers are zero-drift random walks.
    Otherwise, a mix of strong trend, moderate trend, GARCH, and regime-switching
    series is generated so the gauntlet has both passing and failing candidates.
    """
    if tickers is None:
        tickers = ["SYN_TREND_A", "SYN_TREND_B", "SYN_GARCH", "SYN_REGIME", "SYN_NOISE"]

    data: dict[str, pd.DataFrame] = {}
    for idx, ticker in enumerate(tickers):
        t_seed = seed + idx * 17
        if pure_noise:
            df = gbm_series(n=n_bars, mu=0.0, sigma=0.20, seed=t_seed)
        else:
            mode = idx % 5
            if mode == 0:
                df = gbm_series(n=n_bars, mu=0.45, sigma=0.10, seed=t_seed)
            elif mode == 1:
                df = gbm_series(n=n_bars, mu=0.40, sigma=0.11, seed=t_seed)
            elif mode == 2:
                df = garch_series(n=n_bars, seed=t_seed)
            elif mode == 3:
                df, _ = regime_series(n=n_bars, seed=t_seed)
            else:
                df = gbm_series(n=n_bars, mu=0.0, sigma=0.22, seed=t_seed)
        data[ticker] = enrich_dataframe(df)
    return data


def load_universe(
    tickers: list[str],
    synthetic: bool = False,
    n_bars: int = 800,
    seed: int = config.PIPELINE_SEED,
    pure_noise: bool = False,
    min_bars: int = MIN_HISTORY_BARS,
    use_cache: bool = True,
) -> dict[str, pd.DataFrame]:
    """Load and enrich market data for a list of tickers (synthetic or live)."""
    if synthetic:
        raw = build_synthetic_universe(tickers=tickers, n_bars=n_bars, seed=seed, pure_noise=pure_noise)
        return {k: v for k, v in raw.items() if len(v) >= min_bars}

    out: dict[str, pd.DataFrame] = {}
    for t in tickers:
        df = fetch_ticker(t, use_cache=use_cache, min_bars=min_bars)
        if df is not None:
            out[t] = df
    return out
