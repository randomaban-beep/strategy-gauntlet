"""
Pipeline runner orchestrating the strategy-gauntlet loop.

Uses a plain deterministic Python loop over PipelineState (no LangGraph).
"""
from __future__ import annotations

import config
from pipeline.state import PipelineState
from pipeline.steps import (
    step_backtest,
    step_data,
    step_model_selection,
    step_portfolio,
    step_quant,
    step_risk,
    step_scoring,
    step_validation,
)


def run_pipeline(state: PipelineState) -> PipelineState:
    """Run the full strategy-gauntlet pipeline on `state` and return it.

    1. Executes `step_data` to load/filter market data and populate `strategy_queue`.
    2. Iterates while candidates remain and `attempt_counter < max_attempts`:
       - `step_quant` pops next candidate
       - `step_backtest` -> `step_validation` -> `step_model_selection`
         -> `step_risk` -> `step_scoring`
       - Any failure records `rejection_stage` and `rejection_reason` on the candidate
         and moves immediately to the next candidate.
    3. Executes `step_portfolio` (HRP over approved strategies).
    """
    if state.risk_tolerance in config.RISK_PRESETS:
        preset = config.RISK_PRESETS[state.risk_tolerance]
        # Apply preset defaults only if user did not explicitly override
        if state.confidence_threshold is None:
            state.confidence_threshold = preset["confidence_threshold"]
        if state.dsr_threshold is None:
            state.dsr_threshold = preset["dsr_threshold"]

    step_data(state)

    while state.strategy_queue and state.attempt_counter < state.max_attempts:
        has_next = step_quant(state)
        if not has_next or state.current_candidate is None:
            break

        cand = state.current_candidate
        try:
            if step_backtest(state):
                if step_validation(state):
                    if step_model_selection(state):
                        if step_risk(state):
                            step_scoring(state)
        except Exception as exc:
            if not cand.rejection_stage:
                cand.rejection_stage = "error"
                cand.rejection_reason = str(exc)

        state.attempts_log.append(cand)
        state.current_candidate = None

    step_portfolio(state)
    return state
