"""
Command-line entry point for strategy-gauntlet.
"""
from __future__ import annotations

import argparse
from typing import Sequence

import config
from pipeline.runner import run_pipeline
from pipeline.state import PipelineState
from reporting.excel_report import write_excel_report


def _parse_tickers(raw: str | Sequence[str] | None, synthetic: bool) -> list[str]:
    if not raw:
        if synthetic:
            return ["SYN_TREND_A", "SYN_TREND_B", "SYN_GARCH", "SYN_REGIME", "SYN_NOISE"]
        return ["SPY", "QQQ", "IWM", "TLT", "GLD"]
    if isinstance(raw, str):
        items = [t.strip() for t in raw.split(",") if t.strip()]
    else:
        items = []
        for token in raw:
            items.extend(t.strip() for t in token.split(",") if t.strip())
    return items


def _parse_pairs(raw: str | Sequence[str] | None) -> list[tuple[str, str]]:
    if not raw:
        return []
    tokens: list[str] = []
    if isinstance(raw, str):
        tokens = [t.strip() for t in raw.replace(";", " ").split() if t.strip()]
    else:
        for item in raw:
            tokens.extend(t.strip() for t in item.replace(";", " ").split() if t.strip())

    pairs: list[tuple[str, str]] = []
    for tok in tokens:
        sep = ":" if ":" in tok else ("-" if "-" in tok else ",")
        parts = [p.strip() for p in tok.split(sep) if p.strip()]
        if len(parts) == 2:
            pairs.append((parts[0], parts[1]))
        elif len(parts) > 2 and len(parts) % 2 == 0:
            for i in range(0, len(parts), 2):
                pairs.append((parts[i], parts[i + 1]))
    return pairs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="strategy-gauntlet",
        description=(
            "Research pipeline that stress-tests trading strategies with walk-forward validation, "
            "Monte Carlo risk checks, and Deflated Sharpe correction, then allocates capital with HRP."
        ),
    )
    parser.add_argument(
        "--tickers",
        nargs="*",
        default=None,
        help="Tickers to evaluate (space- or comma-separated).",
    )
    parser.add_argument(
        "--pairs",
        nargs="*",
        default=None,
        help="Pairs to evaluate for statistical arbitrage, e.g. --pairs A,B or --pairs AAPL,MSFT.",
    )
    parser.add_argument(
        "--risk-tolerance",
        choices=list(config.RISK_PRESETS.keys()),
        default=config.DEFAULT_RISK_TOLERANCE,
        help="Risk tolerance preset (conservative, moderate, aggressive).",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=config.MAX_ATTEMPTS,
        help="Maximum number of strategy candidates to evaluate.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=None,
        help="Minimum probability of profit across all trusted Monte Carlo models.",
    )
    parser.add_argument(
        "--dsr-threshold",
        type=float,
        default=None,
        help="Minimum Deflated Sharpe Ratio probability to approve a strategy.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.PIPELINE_SEED,
        help="Master random seed for reproducibility.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use offline synthetic market data instead of live yfinance downloads.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="run_report.xlsx",
        help="Output Excel report file path (default: run_report.xlsx).",
    )
    return parser


def print_summary(state: PipelineState, output_path: str) -> None:
    print("=" * 72)
    print("STRATEGY GAUNTLET -- RUN SUMMARY")
    print("=" * 72)
    print(f"Mode            : {'Synthetic (offline)' if state.synthetic else 'Live (yfinance)'}")
    print(f"Risk tolerance  : {state.risk_tolerance} (confidence>={state.confidence_threshold:.2f}, DSR>={state.dsr_threshold:.2f})")
    print(f"Total attempts  : {len(state.attempts_log)} / {state.max_attempts}")
    print(f"Approved        : {len(state.approved_strategies)}")
    print("-" * 72)
    for idx, att in enumerate(state.attempts_log, start=1):
        status = "APPROVED" if att.approved else f"REJECTED ({att.rejection_stage})"
        oos_sr = f"{att.validation_result.aggregate_sharpe:.2f}" if att.validation_result else "N/A"
        dsr_p = f"{att.scoring_result.dsr_probability:.2f}" if att.scoring_result else "N/A"
        reason = f" -- {att.rejection_reason}" if not att.approved and att.rejection_reason else ""
        print(f"  [{idx:02d}] {att.ticker:<14} | {att.strategy_name:<15} | {status:<22} | OOS SR={oos_sr} | DSR={dsr_p}{reason}")

    print("-" * 72)
    print(f"Portfolio status: {state.portfolio_status}")
    if state.portfolio_result and state.portfolio_result.weights:
        for k, w in state.portfolio_result.weights.items():
            print(f"  - {k:<30}: {w:.4f}")
    print(f"Excel report    : {output_path}")
    print("=" * 72)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    preset = config.RISK_PRESETS.get(args.risk_tolerance, config.RISK_PRESETS[config.DEFAULT_RISK_TOLERANCE])
    conf = args.confidence if args.confidence is not None else preset["confidence_threshold"]
    dsr_thresh = args.dsr_threshold if args.dsr_threshold is not None else preset["dsr_threshold"]

    tickers = _parse_tickers(args.tickers, synthetic=args.synthetic)
    pairs = _parse_pairs(args.pairs)

    state = PipelineState(
        tickers=tickers,
        pairs=pairs,
        risk_tolerance=args.risk_tolerance,
        max_attempts=args.max_attempts,
        confidence_threshold=conf,
        dsr_threshold=dsr_thresh,
        seed=args.seed,
        synthetic=args.synthetic,
        n_mc_paths=150 if args.synthetic else config.N_MC_PATHS,
    )

    run_pipeline(state)
    write_excel_report(state, args.output)
    print_summary(state, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
