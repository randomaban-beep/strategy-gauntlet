"""
Transaction cost model.

Charges on position changes: half-spread + commission + square-root market impact.
Costs are always non-negative.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def apply_costs(
    returns: pd.Series,
    positions: pd.Series,
    spread_bps: float = 5.0,
    commission_bps: float = 1.0,
    impact_coef: float = 0.1,
) -> pd.Series:
    """Subtract transaction costs from a return series.

    Costs are charged on POSITION CHANGES (trades). A position change of
    delta = |pos[t] - pos[t-1]| triggers:

        cost[t] = delta * (0.5 * spread_bps + commission_bps) / 10_000
                + impact_coef * sqrt(|delta|) / 10_000

    The first bar always has delta = |pos[0]|.

    Parameters
    ----------
    returns : pd.Series of strategy log returns (BEFORE costs)
    positions : pd.Series of position values (same index); may be {-1, 0, +1}
        or fractional.
    spread_bps : half-spread in basis points (total round-trip = 2x)
    commission_bps : one-way commission in basis points
    impact_coef : coefficient for sqrt-impact model (in bps / sqrt(shares))

    Returns
    -------
    pd.Series of net-of-cost log returns (always returns - cost, so net <= gross).
    """
    pos = positions.values.astype(float)
    prev_pos = np.zeros(len(pos))
    prev_pos[1:] = pos[:-1]
    delta = np.abs(pos - prev_pos)

    # Basis-point components
    linear_cost = delta * (0.5 * spread_bps + commission_bps) / 10_000.0
    impact_cost = impact_coef * np.sqrt(delta) / 10_000.0
    total_cost = linear_cost + impact_cost  # always >= 0

    return (returns - total_cost).rename(returns.name)
