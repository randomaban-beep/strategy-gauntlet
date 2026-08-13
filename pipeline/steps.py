"""
Pipeline step functions operating on shared PipelineState.

Steps:
- data: load/filter ticker data and build the (ticker, strategy) candidate queue
- quant: pop the next candidate and increment attempt counter
- backtest: cheap frictionless in-sample gate (reject non-positive total return)
- validation: walk-forward validation with transaction costs
- model_selection: fit return models (GBM, GJR-GARCH, HMM) and select trusted ones
- risk: Monte Carlo simulation across all trusted models + block bootstrap
- scoring: Deflated Sharpe Ratio (DSR) gate using attempt_counter as n_trials
- portfolio: HRP allocation across approved strategies' OOS return streams
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from allocation.hrp import allocate_hrp
from data.fetch import MIN_HISTORY_BARS, enrich_dataframe, load_universe
from data.synthetic import cointegrated_pair, uncorrelated_pair
from engine.copula import fit_student_t_copula
from engine.monte_carlo import passes_risk_gate, run_monte_carlo, run_pair_monte_carlo
from pipeline.state import BacktestResult, PipelineState, StrategyAttempt
from strategies.base import Strategy, strategy_returns
from strategies.ma_crossover import MACrossover
from strategies.mean_reversion import MeanReversion
from strategies.stat_arb import StatArb, build_pair_dataframe, engle_granger_coint
from validation.deflated_sharpe import score_strategy
from validation.model_selection import select_model
from validation.walk_forward import walk_forward


def _in_sample_sharpe(rets: pd.Series, ann: int = config.ANNUALIZATION) -> float:
    if len(rets) < 2:
        return 0.0
    std = float(rets.std(ddof=1))
    if std < 1e-12 or np.isnan(std):
        return 0.0
    return float(rets.mean() / std * np.sqrt(ann))


def step_data(state: PipelineState) -> None:
    """Populate market_data and build the candidate (ticker, strategy) queue (including pairs)."""
    if not state.market_data:
        if not state.tickers and not state.pairs:
            state.tickers = ["SYN_TREND_A", "SYN_TREND_B", "SYN_GARCH", "SYN_REGIME", "SYN_NOISE"]
        if state.tickers:
            state.market_data = load_universe(
                tickers=state.tickers,
                synthetic=state.synthetic,
                seed=state.seed,
                pure_noise=state.pure_noise,
                min_bars=MIN_HISTORY_BARS,
            )
    else:
        filtered: dict[str, pd.DataFrame] = {}
        for t, df in state.market_data.items():
            if len(df) >= MIN_HISTORY_BARS:
                filtered[t] = enrich_dataframe(df) if "close_a" not in df.columns else df
            else:
                state.log(f"Skipping {t}: insufficient history ({len(df)} < {MIN_HISTORY_BARS})")
        state.market_data = filtered

    single_Asset_tickers = [t for t, df in state.market_data.items() if "close_a" not in df.columns]
    strategies: list[Strategy] = [MACrossover(), MeanReversion()]
    queue: list[tuple[str, Strategy]] = []
    for ticker in single_Asset_tickers:
        for strat in strategies:
            queue.append((ticker, strat))

    # Wire pairs into data step
    for p_idx, (tk_a, tk_b) in enumerate(state.pairs):
        pair_key = f"{tk_a}-{tk_b}"
        if pair_key not in state.market_data:
            if tk_a not in state.market_data or tk_b not in state.market_data:
                if state.synthetic:
                    if state.pure_noise:
                        df_a, df_b = uncorrelated_pair(n=800, seed=state.seed + 300 + p_idx)
                    else:
                        df_a, df_b = cointegrated_pair(n=800, beta=0.8, seed=state.seed + 300 + p_idx)
                    state.market_data[tk_a] = enrich_dataframe(df_a)
                    state.market_data[tk_b] = enrich_dataframe(df_b)
                else:
                    loaded = load_universe(
                        tickers=[tk_a, tk_b],
                        synthetic=False,
                        seed=state.seed,
                        min_bars=MIN_HISTORY_BARS,
                    )
                    state.market_data.update(loaded)

            if tk_a in state.market_data and tk_b in state.market_data:
                df_a = state.market_data[tk_a]
                df_b = state.market_data[tk_b]
                if len(df_a) >= MIN_HISTORY_BARS and len(df_b) >= MIN_HISTORY_BARS:
                    state.market_data[pair_key] = build_pair_dataframe(df_a, df_b)
                else:
                    state.log(f"Skipping pair {pair_key}: insufficient history")
                    continue
            else:
                state.log(f"Skipping pair {pair_key}: missing ticker data")
                continue

        pair_df = state.market_data[pair_key]
        # Pre-check cointegration on initial calibration window
        c_len = min(200, len(pair_df))
        coint_res = engle_granger_coint(
            pair_df["close_a"].iloc[:c_len],
            pair_df["close_b"].iloc[:c_len],
        )
        queue.append((
            pair_key,
            StatArb(
                hedge_ratio=coint_res.hedge_ratio,
                is_cointegrated=coint_res.is_cointegrated,
            ),
        ))

    state.strategy_queue = queue
    state.log(f"Data step complete: {len(state.market_data)} series loaded, {len(queue)} candidates queued.")


def step_quant(state: PipelineState) -> bool:
    """Pop the next candidate from the queue and increment attempt_counter."""
    if not state.strategy_queue or state.attempt_counter >= state.max_attempts:
        state.current_candidate = None
        return False

    ticker, strat = state.strategy_queue.pop(0)
    state.attempt_counter += 1
    is_pair = strat.name == "stat_arb" or ("-" in ticker and "close_a" in state.market_data.get(ticker, pd.DataFrame()).columns)
    pair_tickers = tuple(ticker.split("-", 1)) if is_pair and "-" in ticker else None
    state.current_candidate = StrategyAttempt(
        ticker=ticker,
        strategy_name=strat.name,
        strategy=strat,
        is_pair=is_pair,
        pair_tickers=pair_tickers,  # type: ignore[arg-type]
    )
    state.log(f"Attempt #{state.attempt_counter}: testing {strat.name} on {ticker}")
    return True


def step_backtest(state: PipelineState) -> bool:
    """Run frictionless in-sample backtest across parameter grid; reject non-positive return."""
    cand = state.current_candidate
    if cand is None or cand.strategy is None:
        return False

    df = state.market_data[cand.ticker]
    strat = cand.strategy

    if isinstance(strat, StatArb) and strat.is_cointegrated is False:
        cand.backtest_result = BacktestResult(total_return=0.0, sharpe=0.0, best_params={})
        cand.rejection_stage = "backtest"
        cand.rejection_reason = "Pair failed Engle-Granger cointegration significance check"
        state.log(f"Rejected {cand.ticker}/{cand.strategy_name} at backtest: {cand.rejection_reason}")
        return False

    combos = strat.all_param_combinations()

    best_params = combos[0]
    best_sharpe = -np.inf
    best_total_ret = -np.inf

    for params in combos:
        allow_short = bool(params.get("allow_short", True))
        signals = strat.generate_signals(df, **params)
        rets = strategy_returns(signals, df["log_return"], allow_short=allow_short)
        tot_ret = float(rets.sum())
        sh = _in_sample_sharpe(rets)
        if sh > best_sharpe or (sh == best_sharpe and tot_ret > best_total_ret):
            best_sharpe = sh
            best_total_ret = tot_ret
            best_params = params

    cand.params = best_params
    cand.backtest_result = BacktestResult(
        total_return=best_total_ret,
        sharpe=best_sharpe,
        best_params=best_params,
    )

    if best_total_ret <= 0.0:
        cand.rejection_stage = "backtest"
        cand.rejection_reason = f"Non-positive in-sample total return ({best_total_ret:.4f})"
        state.log(f"Rejected {cand.ticker}/{cand.strategy_name} at backtest: {cand.rejection_reason}")
        return False

    return True


def step_validation(state: PipelineState) -> bool:
    """Run walk-forward validation with transaction costs."""
    cand = state.current_candidate
    if cand is None or cand.strategy is None:
        return False

    df = state.market_data[cand.ticker]
    try:
        wf_res = walk_forward(df, cand.strategy)
    except Exception as exc:
        cand.rejection_stage = "validation"
        cand.rejection_reason = f"Walk-forward error: {exc}"
        state.log(f"Rejected {cand.ticker}/{cand.strategy_name} at validation: {cand.rejection_reason}")
        return False

    cand.validation_result = wf_res
    if wf_res.folds:
        cand.params = wf_res.folds[-1].best_params

    if wf_res.total_return <= 0.0 or wf_res.aggregate_sharpe <= 0.0:
        cand.rejection_stage = "validation"
        cand.rejection_reason = (
            f"OOS walk-forward failed (total_return={wf_res.total_return:.4f}, "
            f"sharpe={wf_res.aggregate_sharpe:.3f})"
        )
        state.log(f"Rejected {cand.ticker}/{cand.strategy_name} at validation: {cand.rejection_reason}")
        return False

    return True


def step_model_selection(state: PipelineState) -> bool:
    """Fit GBM, GJR-GARCH, and HMM (or Student-t copula for pairs) and select trusted models."""
    cand = state.current_candidate
    if cand is None:
        return False

    df = state.market_data[cand.ticker]
    if cand.is_pair and "log_return_a" in df.columns and "log_return_b" in df.columns:
        cop_fit = fit_student_t_copula(df["log_return_a"], df["log_return_b"])
        cand.trusted_models = {"student_t_copula": cop_fit}
        return True

    ms_res = select_model(df["log_return"])
    cand.model_selection_result = ms_res
    cand.trusted_models = ms_res.trusted_models
    return True


def step_risk(state: PipelineState) -> bool:
    """Run Monte Carlo stress test across all trusted models (or Student-t copula for pairs)."""
    cand = state.current_candidate
    if cand is None or cand.strategy is None:
        return False

    df = state.market_data[cand.ticker]
    allow_short = bool(cand.params.get("allow_short", True))

    if cand.is_pair and "log_return_a" in df.columns and "log_return_b" in df.columns:
        mc_res = run_pair_monte_carlo(
            strategy=cand.strategy,
            params=cand.params,
            pair_df=df,
            n_paths=state.n_mc_paths,
            seed=state.seed + state.attempt_counter,
            n_steps=min(len(df), 252),
            allow_short=allow_short,
        )
    else:
        mc_res = run_monte_carlo(
            strategy=cand.strategy,
            params=cand.params,
            trusted_models=cand.trusted_models,
            historical_returns=df["log_return"],
            n_paths=state.n_mc_paths,
            seed=state.seed + state.attempt_counter,
            n_steps=min(len(df), 252),
            allow_short=allow_short,
        )
    cand.risk_metrics = mc_res

    passed, reason = passes_risk_gate(mc_res, confidence_threshold=state.confidence_threshold)
    if not passed:
        cand.rejection_stage = "risk"
        cand.rejection_reason = reason
        state.log(f"Rejected {cand.ticker}/{cand.strategy_name} at risk: {reason}")
        return False

    return True


def step_scoring(state: PipelineState) -> bool:
    """Apply the Deflated Sharpe Ratio (DSR) gate using n_trials = attempt_counter."""
    cand = state.current_candidate
    if cand is None or cand.validation_result is None:
        return False

    oos_rets = cand.validation_result.oos_returns.values
    past_sharpes = [
        a.validation_result.aggregate_sharpe
        for a in state.attempts_log
        if a.validation_result is not None
    ]
    past_sharpes.append(cand.validation_result.aggregate_sharpe)
    trial_sharpes = np.array(past_sharpes, dtype=float) if len(past_sharpes) > 1 else None

    dsr_res = score_strategy(
        oos_returns=oos_rets,
        n_trials=state.attempt_counter,
        trial_sharpes=trial_sharpes,
        dsr_threshold=state.dsr_threshold,
    )
    cand.scoring_result = dsr_res

    if not dsr_res.passes_threshold:
        cand.rejection_stage = "scoring"
        cand.rejection_reason = (
            f"DSR probability {dsr_res.dsr_probability:.3f} < threshold {state.dsr_threshold:.2f} "
            f"(obs SR={dsr_res.observed_sharpe:.2f}, E[max SR]={dsr_res.expected_max_sharpe:.2f})"
        )
        state.log(f"Rejected {cand.ticker}/{cand.strategy_name} at scoring: {cand.rejection_reason}")
        return False

    cand.approved = True
    state.approved_strategies.append(cand)
    state.log(
        f"APPROVED {cand.ticker}/{cand.strategy_name} "
        f"(OOS SR={dsr_res.observed_sharpe:.2f}, DSR={dsr_res.dsr_probability:.3f})"
    )
    return True


def step_portfolio(state: PipelineState) -> None:
    """Allocate capital across approved strategies using HRP (requires >= 2 approved)."""
    if len(state.approved_strategies) < 2:
        state.portfolio_result = None
        state.portfolio_status = "insufficient strategies"
        state.log(
            f"Portfolio allocation skipped: insufficient strategies "
            f"({len(state.approved_strategies)} approved, need >= 2)."
        )
        return

    oos_dict: dict[str, pd.Series] = {}
    for idx, att in enumerate(state.approved_strategies):
        if att.validation_result is not None:
            key = f"{att.ticker}_{att.strategy_name}"
            if key in oos_dict:
                key = f"{key}_{idx}"
            oos_dict[key] = att.validation_result.oos_returns.reset_index(drop=True)

    oos_df = pd.DataFrame(oos_dict).dropna(how="any")
    hrp_res = allocate_hrp(oos_df)
    state.portfolio_result = hrp_res
    if len(hrp_res.weights) >= 2:
        state.portfolio_status = "allocated"
        state.log(f"Portfolio allocated across {len(hrp_res.weights)} strategies via HRP.")
    else:
        state.portfolio_status = "insufficient strategies"
        state.log("Portfolio allocation resulted in < 2 non-degenerate strategies.")


# Aliases matching specification names
data = step_data
quant = step_quant
backtest = step_backtest
validation = step_validation
model_selection = step_model_selection
risk = step_risk
scoring = step_scoring
portfolio = step_portfolio
