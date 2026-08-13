"""
Shared pipeline state and candidate attempt dataclasses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

import config
from allocation.hrp import HRPResult
from engine.monte_carlo import RiskMetrics
from strategies.base import Strategy
from validation.deflated_sharpe import DSRResult
from validation.model_selection import ModelSelectionResult
from validation.walk_forward import WalkForwardResult


@dataclass
class BacktestResult:
    """In-sample frictionless backtest summary."""
    total_return: float
    sharpe: float
    best_params: dict[str, Any]


@dataclass
class StrategyAttempt:
    """Record of a single candidate's journey through the gauntlet."""
    ticker: str
    strategy_name: str
    strategy: Strategy | None = None
    params: dict[str, Any] = field(default_factory=dict)
    backtest_result: BacktestResult | None = None
    validation_result: WalkForwardResult | None = None
    model_selection_result: ModelSelectionResult | None = None
    trusted_models: dict[str, Any] = field(default_factory=dict)
    risk_metrics: dict[str, RiskMetrics] = field(default_factory=dict)
    scoring_result: DSRResult | None = None
    approved: bool = False
    rejection_stage: str | None = None
    rejection_reason: str | None = None
    is_pair: bool = False
    pair_tickers: tuple[str, str] | None = None


@dataclass
class PipelineState:
    """Flat shared state read and written by all pipeline steps."""
    tickers: list[str] = field(default_factory=list)
    pairs: list[tuple[str, str]] = field(default_factory=list)
    risk_tolerance: str = config.DEFAULT_RISK_TOLERANCE
    max_attempts: int = config.MAX_ATTEMPTS
    confidence_threshold: float = config.CONFIDENCE_THRESHOLD
    dsr_threshold: float = config.DSR_THRESHOLD
    seed: int = config.PIPELINE_SEED
    synthetic: bool = False
    pure_noise: bool = False
    n_mc_paths: int = config.N_MC_PATHS

    market_data: dict[str, pd.DataFrame] = field(default_factory=dict)
    strategy_queue: list[tuple[str, Strategy]] = field(default_factory=list)
    current_candidate: StrategyAttempt | None = None
    attempt_counter: int = 0
    attempts_log: list[StrategyAttempt] = field(default_factory=list)
    approved_strategies: list[StrategyAttempt] = field(default_factory=list)
    portfolio_result: HRPResult | None = None
    portfolio_status: str = "not_run"
    logs: list[str] = field(default_factory=list)

    def log(self, message: str) -> None:
        """Append a timestamp-free deterministic log line."""
        self.logs.append(message)
