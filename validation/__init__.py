"""
Statistical validation: transaction costs, walk-forward, model selection, and Deflated Sharpe.
"""
from __future__ import annotations

from validation.costs import apply_costs
from validation.deflated_sharpe import DSRResult, deflated_sharpe_ratio, score_strategy
from validation.model_selection import ModelDiagnostics, ModelSelectionResult, select_model
from validation.walk_forward import FoldResult, WalkForwardResult, walk_forward

__all__ = [
    "apply_costs",
    "walk_forward",
    "FoldResult",
    "WalkForwardResult",
    "select_model",
    "ModelDiagnostics",
    "ModelSelectionResult",
    "deflated_sharpe_ratio",
    "score_strategy",
    "DSRResult",
]
