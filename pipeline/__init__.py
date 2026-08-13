"""
Autonomous strategy gauntlet pipeline state, modular steps, and runner loop.
"""
from __future__ import annotations

from pipeline.runner import run_pipeline
from pipeline.state import BacktestResult, PipelineState, StrategyAttempt

__all__ = [
    "PipelineState",
    "StrategyAttempt",
    "BacktestResult",
    "run_pipeline",
]
