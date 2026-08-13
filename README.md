# strategy-gauntlet

Research pipeline that stress-tests trading strategies with walk-forward validation, Monte Carlo risk checks, and Deflated Sharpe correction, then allocates capital with HRP.

[![tests](https://github.com/randomaban-beep/strategy-gauntlet/actions/workflows/tests.yml/badge.svg)](https://github.com/randomaban-beep/strategy-gauntlet/actions/workflows/tests.yml)

---

## Layered Validation Design (The Gauntlet)

Most backtests fail out of sample because of look-ahead bias, ignored transaction costs, parameter overfitting, or multiple-testing selection bias. `strategy-gauntlet` runs each `(asset, strategy)` candidate through a deterministic sequence of progressively stricter statistical gates:

```text
[Candidate Queue: (Ticker / Pair, Strategy)]
         │
         ▼
┌────────────────────────────────────────────────────────┐
│ Stage 1: Frictionless In-Sample Backtest               │
│  • Grid search with strict 1-bar forward shift guard   │
│  • Rejects if in-sample total log return <= 0          │
└──────────────────────────┬─────────────────────────────┘
                           ▼
┌────────────────────────────────────────────────────────┐
│ Stage 2: Walk-Forward Validation + Transaction Costs   │
│  • Rolling/expanding folds with configurable embargo   │
│  • Charges half-spread + commission + sqrt impact      │
│  • Rejects if concatenated OOS return or Sharpe <= 0   │
└──────────────────────────┬─────────────────────────────┘
                           ▼
┌────────────────────────────────────────────────────────┐
│ Stage 3: Return-Distribution Model Selection           │
│  • Fits GBM, GJR-GARCH(1,1), and 2-state Gaussian HMM  │
│  • Trusts only models passing Ljung-Box on residuals   │
│    and squared residuals (falls back to GBM if needed) │
└──────────────────────────┬─────────────────────────────┘
                           ▼
┌────────────────────────────────────────────────────────┐
│ Stage 4: Multi-Model Monte Carlo Stress Gate           │
│  • Replays strategy across synthetic paths from ALL    │
│    trusted parametric models + block bootstrap         │
│  • Rejects if Prob(Profit) < confidence under ANY      │
└──────────────────────────┬─────────────────────────────┘
                           ▼
┌────────────────────────────────────────────────────────┐
│ Stage 5: Deflated Sharpe Ratio (DSR) Gate              │
│  • Adjusts OOS Sharpe for skewness, kurtosis, sample   │
│    length, and expected max SR across N trials         │
│  • Rejects if DSR probability < dsr_threshold          │
└──────────────────────────┬─────────────────────────────┘
                           ▼
┌────────────────────────────────────────────────────────┐
│ Stage 6: Hierarchical Risk Parity (HRP) Allocation     │
│  • Clusters approved OOS return streams via correlation│
│    distance, quasi-diagonalizes, and allocates         │
│    inversely to cluster variance without matrix inv.   │
└────────────────────────────────────────────────────────┘
```

| Stage | Module | Primary Filter Criterion |
| :--- | :--- | :--- |
| **1. Naive Backtest** | `strategies/base.py`, `pipeline/steps.py` | In-sample frictionless return `> 0` (strict 1-bar forward shift) |
| **2. Walk-Forward** | `validation/walk_forward.py`, `validation/costs.py` | Net-of-cost OOS Sharpe `> 0` across embargoed folds |
| **3. Model Selection** | `validation/model_selection.py`, `engine/garch.py`, `engine/regime_switching.py` | Ljung-Box white-noise diagnostics on standardized & squared residuals |
| **4. Monte Carlo Risk** | `engine/monte_carlo.py`, `engine/copula.py` | `P(Profit) >= confidence` across **every** trusted model + bootstrap |
| **5. Deflated Sharpe** | `validation/deflated_sharpe.py` | Bailey & Lopez de Prado `DSR >= dsr_threshold` given `N` trials |
| **6. HRP Portfolio** | `allocation/hrp.py` | Tree-clustered inverse-variance weights across `>= 2` approved strategies |

---

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Example Commands

### 1. Run Offline on Synthetic Data (Reproducible)
```bash
python main.py --synthetic --output sample_run_report.xlsx
```

### 2. Custom Risk Tolerance, Thresholds, and Seed
```bash
python main.py --synthetic --risk-tolerance conservative --confidence 0.65 --dsr-threshold 0.85 --seed 42
```

### 3. Evaluate Specific Tickers (Live `yfinance` with Local Cache)
```bash
python main.py --tickers SPY,QQQ,TLT,GLD --risk-tolerance moderate --max-attempts 20
```

### 4. Pairs Trading (Statistical Arbitrage with Student-t Copula Stress)
```bash
python main.py --synthetic --pairs A,B
```

---

## Project Layout

```text
strategy-gauntlet/
├── .github/workflows/tests.yml   # CI workflow running pytest on Python 3.11
├── config.py                     # Global constants, risk presets, seeds, and thresholds
├── main.py                       # CLI entry point
├── requirements.txt              # Python package dependencies
├── sample_run_report.xlsx        # Example multi-sheet Excel output from --synthetic
├── data/
│   ├── fetch.py                  # yfinance downloader, CSV cache, and feature enrichment
│   └── synthetic.py              # Seeded GBM, GJR-GARCH, HMM regime, and cointegrated generators
├── strategies/
│   ├── base.py                   # Strategy ABC, ParamSpec, and central strategy_returns shift
│   ├── ma_crossover.py           # Fast/slow moving-average crossover strategy
│   ├── mean_reversion.py         # Rolling z-score mean-reversion strategy
│   └── stat_arb.py               # Engle-Granger cointegrated pairs trading strategy
├── validation/
│   ├── costs.py                  # Half-spread + commission + square-root impact model
│   ├── walk_forward.py           # Rolling/expanding walk-forward with embargo gap
│   ├── model_selection.py        # GBM vs GJR-GARCH vs HMM residual diagnostic selector
│   └── deflated_sharpe.py        # Bailey & Lopez de Prado Deflated Sharpe Ratio
├── engine/
│   ├── garch.py                  # GJR-GARCH(1,1) fitting and path simulation
│   ├── regime_switching.py       # 2-state Gaussian HMM fitting and simulation
│   ├── copula.py                 # Hand-written bivariate Student-t copula for tail dependence
│   └── monte_carlo.py            # Path replay, block bootstrap, VaR/CVaR, and risk gate
├── allocation/
│   └── hrp.py                    # Lopez de Prado Hierarchical Risk Parity allocator
├── pipeline/
│   ├── state.py                  # PipelineState and StrategyAttempt dataclasses
│   ├── steps.py                  # Modular step functions reading/writing shared state
│   └── runner.py                 # Deterministic bounded execution loop
├── reporting/
│   └── excel_report.py           # Formatted 3-sheet openpyxl workbook writer
└── tests/                        # Offline unit and end-to-end pytest suite
```

---

## Running Tests

All tests run 100% offline using seeded synthetic datasets:

```bash
pytest -q
```

---

## Limitations

1. **No Survivorship or Point-in-Time Handling in `yfinance`**: Yahoo Finance data reflects current tickers and retroactive corporate adjustments. It does not provide delisted equities or point-in-time universe snapshots, so live equity universe runs are subject to survivorship bias.
2. **Approximate Transaction Cost Model**: `validation/costs.py` applies a static half-spread, fixed commission (in bps), and a stylized square-root turnover penalty (`impact_coef * sqrt(|delta|)`). It does not model intraday order-book depth, time-varying bid-ask spreads, or borrow fees for short positions.
3. **GARCH and HMM Require Long History**: Reliable maximum-likelihood estimation of GJR-GARCH(1,1) and 2-state Gaussian HMMs requires at least ~500 daily bars (`MIN_GARCH_HISTORY = 500`, `MIN_HMM_HISTORY = 500`). Shorter histories automatically fall back to constant-volatility GBM.
4. **Live Fetching Must Be Smoke-Tested Locally**: Network calls to `yfinance` depend on Yahoo's public endpoints and rate limits. While all pipeline logic is verified offline via `--synthetic` and `pytest`, live ticker fetching should be smoke-tested locally with `cache/` enabled.
