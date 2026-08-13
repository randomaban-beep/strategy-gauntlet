"""Global configuration defaults for strategy-gauntlet."""

# Annualization
ANNUALIZATION = 252

# Risk tolerance presets: maps name -> {'confidence_threshold': float, 'dsr_threshold': float}
RISK_PRESETS = {
    'conservative': {'confidence_threshold': 0.70, 'dsr_threshold': 0.90},
    'moderate':     {'confidence_threshold': 0.60, 'dsr_threshold': 0.80},
    'aggressive':   {'confidence_threshold': 0.50, 'dsr_threshold': 0.70},
}
DEFAULT_RISK_TOLERANCE = 'moderate'

# Walk-forward
EMBARGO_BARS = 10
MIN_TRAIN_BARS = 252
MIN_TEST_BARS = 63

# Monte Carlo
N_MC_PATHS = 500
MC_SEED = 42
BLOCK_LENGTH = 20

# Costs (bps)
SPREAD_BPS = 5.0
COMMISSION_BPS = 1.0
IMPACT_COEF = 0.1

# DSR / scoring
DSR_THRESHOLD = 0.80
CONFIDENCE_THRESHOLD = 0.60

# Pipeline
MAX_ATTEMPTS = 50
PIPELINE_SEED = 0

# Model selection diagnostics
LJUNG_BOX_LAGS = 10
DIAGNOSTICS_SIGNIFICANCE = 0.05
MIN_GARCH_HISTORY = 500
MIN_HMM_HISTORY = 500

# HMM
HMM_RESTARTS = 5
HMM_SEED = 7
