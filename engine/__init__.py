"""
Simulation and risk engines: GJR-GARCH, Gaussian HMM, Student-t copula, and Monte Carlo.
"""
from __future__ import annotations

from engine.copula import fit_student_t_copula, simulate_copula_pair_paths
from engine.garch import fit_garch, simulate_garch_paths
from engine.monte_carlo import compute_risk_metrics, passes_risk_gate, run_monte_carlo, run_pair_monte_carlo
from engine.regime_switching import fit_hmm, simulate_hmm_paths

__all__ = [
    "fit_garch",
    "simulate_garch_paths",
    "fit_hmm",
    "simulate_hmm_paths",
    "fit_student_t_copula",
    "simulate_copula_pair_paths",
    "compute_risk_metrics",
    "passes_risk_gate",
    "run_monte_carlo",
    "run_pair_monte_carlo",
]
