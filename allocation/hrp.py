"""
Hierarchical Risk Parity (HRP) portfolio allocation.

Reference: Lopez de Prado, M. (2016).
"Building Diversified Portfolios that Outperform Out-of-Sample".
Journal of Portfolio Management.

Pipeline:
1. Correlation matrix -> distance matrix: d(i, j) = sqrt(0.5 * (1 - rho_ij))
2. Scipy hierarchical linkage (single linkage on pairwise Euclidean distances of d)
3. Quasi-diagonalization (leaf ordering from dendrogram)
4. Recursive bisection allocating inversely to cluster variance.
No matrix inversion is used.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform


@dataclass
class HRPResult:
    """Result from Hierarchical Risk Parity allocation."""
    weights: dict[str, float]
    ordered_assets: list[str] = field(default_factory=list)
    dropped_assets: list[str] = field(default_factory=list)


def _cluster_variance(cov: pd.DataFrame, items: list[str]) -> float:
    """Compute inverse-variance portfolio variance for a cluster of items.

    Uses only diagonal elements to form intra-cluster weights (no matrix inversion).
    """
    sub_cov = cov.loc[items, items].values
    diag = np.diag(sub_cov)
    inv_diag = 1.0 / np.maximum(diag, 1e-12)
    w = inv_diag / inv_diag.sum()
    return float(w @ sub_cov @ w)


def allocate_hrp(returns_df: pd.DataFrame) -> HRPResult:
    """Compute Hierarchical Risk Parity weights for a DataFrame of asset/strategy returns.

    Parameters
    ----------
    returns_df : pd.DataFrame where each column is a strategy/asset return stream.

    Returns
    -------
    HRPResult with weights summing to 1.0 (or empty if no valid columns remain).
    """
    if returns_df.empty or returns_df.shape[1] == 0:
        return HRPResult(weights={}, ordered_assets=[], dropped_assets=[])

    dropped: list[str] = []
    valid_cols: list[str] = []

    for col in returns_df.columns:
        series = returns_df[col]
        if series.isna().all():
            warnings.warn(f"Column '{col}' is all NaN; dropping from HRP allocation.")
            dropped.append(str(col))
            continue
        clean = series.dropna()
        if len(clean) < 2 or clean.std(ddof=1) < 1e-12:
            warnings.warn(f"Column '{col}' is constant or has insufficient data; dropping from HRP.")
            dropped.append(str(col))
            continue
        valid_cols.append(col)

    if not valid_cols:
        return HRPResult(weights={}, ordered_assets=[], dropped_assets=dropped)

    df = returns_df[valid_cols].dropna(how="any")
    # Re-check after aligned dropna
    still_valid: list[str] = []
    for col in valid_cols:
        if len(df[col]) < 2 or df[col].std(ddof=1) < 1e-12:
            warnings.warn(f"Column '{col}' became constant after alignment; dropping from HRP.")
            dropped.append(str(col))
        else:
            still_valid.append(col)

    valid_cols = still_valid
    if not valid_cols:
        return HRPResult(weights={}, ordered_assets=[], dropped_assets=dropped)

    df = df[valid_cols]
    n = len(valid_cols)

    # Edge case: 1 strategy
    if n == 1:
        col = str(valid_cols[0])
        return HRPResult(
            weights={col: 1.0},
            ordered_assets=[col],
            dropped_assets=dropped,
        )

    cov = df.cov()
    corr_mat = df.corr().clip(-1.0, 1.0).fillna(0.0).to_numpy(dtype=float, copy=True)
    np.fill_diagonal(corr_mat, 1.0)

    # Distance matrix: d_ij = sqrt(0.5 * (1 - corr_ij))
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr_mat), 0.0, 1.0))
    np.fill_diagonal(dist, 0.0)
    # Ensure symmetry
    dist = 0.5 * (dist + dist.T)

    if n == 2:
        ordered_indices = [0, 1]
    else:
        # Pairwise Euclidean distance of distance matrix columns (or condensed dist directly)
        condensed = squareform(dist, checks=False)
        link = linkage(condensed, method="single")
        ordered_indices = list(leaves_list(link))

    ordered_assets = [str(valid_cols[i]) for i in ordered_indices]

    # Recursive bisection
    weights = pd.Series(1.0, index=ordered_assets, dtype=float)
    clusters: list[list[str]] = [ordered_assets]

    while clusters:
        next_clusters: list[list[str]] = []
        for cluster in clusters:
            if len(cluster) > 1:
                mid = len(cluster) // 2
                left = cluster[:mid]
                right = cluster[mid:]
                var_left = _cluster_variance(cov, left)
                var_right = _cluster_variance(cov, right)
                denom = var_left + var_right
                alpha = 1.0 - var_left / denom if denom > 0 else 0.5
                weights[left] *= alpha
                weights[right] *= (1.0 - alpha)
                if len(left) > 1:
                    next_clusters.append(left)
                if len(right) > 1:
                    next_clusters.append(right)
        clusters = next_clusters

    # Normalize to avoid any floating-point drift
    total = float(weights.sum())
    if total > 0:
        weights = weights / total

    weight_dict = {str(k): float(v) for k, v in weights.items()}
    return HRPResult(
        weights=weight_dict,
        ordered_assets=ordered_assets,
        dropped_assets=dropped,
    )
