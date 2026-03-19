"""Composite scoring functions combining multiple feature domains.

Three market-state scores are implemented here, each on a defined scale:

    compression_score       0 – 100   (higher = more energy stored)
    settlement_pressure_score  -100 – +100  (sign = anticipated direction)
    liquidity_fragility_index  0 – 100   (higher = more fragile book)

Plus the original cross-feature interaction:

    oi_funding_interact     (oi_z_24h × |funding_z|)

Percentile-rank inputs
-----------------------
Several scores require knowing how extreme a current value is relative to
its own history.  Pass a ``*_history`` keyword argument (numpy array or
list of past observations) to get an exact empirical percentile rank.

When no history is supplied, a sigmoid approximation is used for z-score
inputs (``1 / (1 + e^{-x})``), mapping z=0 → 0.50, z=2 → 0.88, z=3 → 0.95.
For non-z-score inputs (vacuum_dist, convexity, kyle_lambda_ratio) the
neutral fallback 0.50 is used, which effectively zeroes out that weight
in the final score until history is available.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np


# ── Shared utility ─────────────────────────────────────────────────────────────


def _pct_rank(
    value: float,
    history: Optional[Union[np.ndarray, List[float]]],
    *,
    fallback: Optional[float] = None,
) -> float:
    """Empirical percentile rank of *value* within *history*.

    Parameters
    ----------
    value:
        The observation to rank.
    history:
        Reference distribution.  NaN entries are silently dropped.
    fallback:
        Value returned when history is absent or too short (< 2 non-NaN
        entries).  When ``None``, falls back to the sigmoid of *value*
        (appropriate for z-score inputs).
    """
    if np.isnan(value):
        return float("nan")

    if history is not None:
        arr = np.asarray(history, dtype=float)
        arr = arr[~np.isnan(arr)]
        if len(arr) >= 2:
            return float(np.mean(arr <= value))

    if fallback is not None:
        return float(fallback)

    # Sigmoid: natural mapping for z-score inputs when no history available
    return float(1.0 / (1.0 + np.exp(-np.clip(float(value), -500, 500))))


# ── Original cross-feature interaction ────────────────────────────────────────


def compute_composite_features(
    funding_feats: Dict[str, Any],
    oi_feats: Dict[str, float],
) -> Dict[str, float]:
    """Cross-feature interaction: OI z-score × |funding z-score|.

    Returns
    -------
    Dict with key ``oi_funding_interact``.
    High magnitude → OI and funding are both stretched simultaneously
    (long-short spring tension); sign tracks OI direction.
    """
    nan = float("nan")
    oi_z   = float(oi_feats.get("oi_z_24h", nan))
    fund_z = float(funding_feats.get("funding_z", nan))

    interact = nan if (np.isnan(oi_z) or np.isnan(fund_z)) else oi_z * abs(fund_z)
    return {"oi_funding_interact": interact}


# ── Compression Score (0 – 100) ───────────────────────────────────────────────


def compression_score(
    rv_pct: float,
    bb_width_pct: float,
    oi_z_7d: float,
    range_hours: float,
) -> float:
    """Measure how tightly a symbol is coiled — a proxy for stored energy.

    Parameters
    ----------
    rv_pct:
        Annualised realised volatility over 24 h, in percent
        (from ``compute_volatility_features()["rv_pct"]``).
        The formula scores 0 when rv_pct ≥ 20 and 100 when rv_pct = 0.
    bb_width_pct:
        Bollinger Band width (20-bar, ±2 σ) as % of midline
        (from ``compute_volatility_features()["bb_width_pct"]``).
        The formula scores 0 when bb_width_pct ≥ 20.
    oi_z_7d:
        7-day OI z-score (use ``oi_z_24h`` from ``compute_oi_features()``).
        OI building during compression amplifies the score.
    range_hours:
        Consecutive compressed bars
        (from ``compute_volatility_features()["range_hours"]``).
        12 hours maps to the full 100 % duration factor.

    Returns
    -------
    float
        Score in [0, 100].  NaN when any required input is NaN.

    Weights
    -------
    vol_compression   30 %   (rv_pct below 20)
    range_compression 25 %   (bb_width below 20)
    oi_buildup        25 %   (OI increasing during compression)
    duration_factor   20 %   (longer compression → more energy stored)
    """
    inputs = [rv_pct, bb_width_pct, oi_z_7d, range_hours]
    if any(np.isnan(x) for x in inputs):
        return float("nan")

    vol_compression   = max(0.0, (20.0 - rv_pct)       / 20.0) * 100.0
    range_compression = max(0.0, (20.0 - bb_width_pct) / 20.0) * 100.0
    oi_buildup        = min(100.0, max(0.0, oi_z_7d * 30.0))
    duration_factor   = min(100.0, range_hours / 12.0 * 100.0)

    return (
        0.30 * vol_compression
        + 0.25 * range_compression
        + 0.25 * oi_buildup
        + 0.20 * duration_factor
    )


# ── Settlement Pressure Score (-100 – +100) ───────────────────────────────────


def settlement_pressure_score(
    funding_z: float,
    oi_z_7d: float,
    vacuum_dist_squeeze_dir: float,
    thin_pct: float,
    minutes_to_settle: float,
    *,
    funding_z_history: Optional[Union[np.ndarray, List[float]]] = None,
    oi_z_history: Optional[Union[np.ndarray, List[float]]] = None,
    vacuum_history: Optional[Union[np.ndarray, List[float]]] = None,
) -> float:
    """Estimate the probability and direction of a settlement-driven price move.

    The score is negative when longs are likely to be squeezed (positive
    funding → longs pay → they have incentive to close → price falls) and
    positive when shorts are squeezed (negative funding).

    Parameters
    ----------
    funding_z:
        Robust z-score of the current funding rate
        (``compute_funding_features()["funding_z"]``).
    oi_z_7d:
        7-day OI z-score.  High OI during extreme funding → more to unwind.
    vacuum_dist_squeeze_dir:
        Vacuum distance (bps) in the direction of the anticipated squeeze.
        Pass ``vacuum_dist_ask`` when funding_z < 0 (shorts squeezed upward)
        or ``vacuum_dist_bid`` when funding_z > 0 (longs squeezed downward).
    thin_pct:
        Percentile rank of current book thinness (already 0-1).
    minutes_to_settle:
        Minutes until the next settlement (from
        ``compute_funding_features()["minutes_to_settlement"]``).
    funding_z_history, oi_z_history, vacuum_history:
        Optional trailing arrays for empirical percentile ranking.  When
        absent, sigmoid / neutral fallbacks are used (see module docstring).

    Returns
    -------
    float
        Score in [-100, +100].  NaN when funding_z is NaN.
        Positive → upward squeeze pressure.
        Negative → downward squeeze pressure.

    Weights
    -------
    funding_intensity   30 %  (how extreme is the current rate?)
    oi_intensity        25 %  (how much open interest to unwind?)
    path_openness       20 %  (how easy is it to move in squeeze direction?)
    book_fragility      15 %  (thin book amplifies the move)
    timing              10 %  (Gaussian peak at settlement moment)
    """
    if np.isnan(funding_z):
        return float("nan")

    # Direction: opposite to who pays (positive funding → longs pay → bearish)
    direction = -float(np.sign(funding_z)) if funding_z != 0 else 0.0

    funding_intensity = _pct_rank(abs(funding_z), funding_z_history)
    oi_intensity      = _pct_rank(oi_z_7d,        oi_z_history)
    path_openness     = _pct_rank(vacuum_dist_squeeze_dir, vacuum_history, fallback=0.5)
    book_fragility    = float(thin_pct) if not np.isnan(thin_pct) else 0.5

    # Gaussian timing weight: peaks at 0 min, half-width ≈ 15 min
    timing = float(np.exp(-0.5 * (minutes_to_settle / 15.0) ** 2))

    raw = (
        0.30 * funding_intensity
        + 0.25 * oi_intensity
        + 0.20 * path_openness
        + 0.15 * book_fragility
        + 0.10 * timing
    )

    return direction * raw * 100.0


# ── Liquidity Fragility Index (0 – 100) ───────────────────────────────────────


def liquidity_fragility_index(
    thin_pct: float,
    spread_z: float,
    convexity: float,
    resilience_5s: float = float("nan"),
    kyle_lambda_ratio: float = float("nan"),
    *,
    spread_z_history: Optional[Union[np.ndarray, List[float]]] = None,
    convexity_history: Optional[Union[np.ndarray, List[float]]] = None,
    kyle_history: Optional[Union[np.ndarray, List[float]]] = None,
) -> float:
    """Estimate how susceptible the current book is to a disorderly move.

    Parameters
    ----------
    thin_pct:
        Percentile rank of book thinness (already 0-1, from
        ``compute_orderbook_features()["thin_pct"]``).
    spread_z:
        Robust z-score of the current bid-ask spread vs. its history
        (compute externally: ``robust_z(spread_bps, spread_bps_history)``).
    convexity:
        Raw convexity ratio (from
        ``compute_orderbook_features()["convexity"]``).
        High → makers have pulled from top levels → fragile front.
    resilience_5s:
        *Optional.*  How quickly the book recovers after a trade, measured
        as a score in [0, 1] over a 5-second window (1 = instant recovery).
        Requires streaming trade/book data.  Omit for a 4-component LFI.
    kyle_lambda_ratio:
        *Optional.*  Kyle's λ ratio — price impact per unit of volume,
        normalised to its recent history (> 1 = higher impact than usual).
        Requires trade data.  Omit for a 4-component LFI.
    spread_z_history, convexity_history, kyle_history:
        Optional trailing arrays for empirical percentile ranking.

    Returns
    -------
    float
        Score in [0, 100].  NaN if the three core inputs are all NaN.

    Weights (always-available components add to 100 %)
    -------
    thin_pct             30 %  (overall book depth vs. history)
    spread_z             25 %  (wide spread = fragile)
    convexity            20 %  (makers pulled from top = fragile front)
    1 − resilience_5s   15 %  (slow recovery = fragile; skipped if NaN)
    kyle_lambda_ratio    10 %  (high price impact = fragile; skipped if NaN)
    """
    # Compute percentile ranks for scored components
    p_thin     = float(thin_pct)  if not np.isnan(thin_pct)  else float("nan")
    p_spread   = _pct_rank(spread_z,   spread_z_history)
    p_convex   = _pct_rank(convexity,  convexity_history, fallback=0.5)

    # Optional components
    has_resil  = not np.isnan(resilience_5s)
    has_kyle   = not np.isnan(kyle_lambda_ratio)
    p_resil    = float(1.0 - resilience_5s) if has_resil else float("nan")
    p_kyle     = _pct_rank(kyle_lambda_ratio, kyle_history, fallback=0.5) if has_kyle else float("nan")

    # Base 3-component score; all three must be available
    if any(np.isnan(x) for x in [p_thin, p_spread, p_convex]):
        return float("nan")

    # Adjust weights if optional components are absent
    if has_resil and has_kyle:
        score = (
            0.30 * p_thin
            + 0.25 * p_spread
            + 0.20 * p_convex
            + 0.15 * p_resil
            + 0.10 * p_kyle
        )
    elif has_resil:
        # Redistribute kyle weight to thin_pct
        score = (
            0.40 * p_thin
            + 0.25 * p_spread
            + 0.20 * p_convex
            + 0.15 * p_resil
        )
    elif has_kyle:
        # Redistribute resilience weight proportionally
        score = (
            0.38 * p_thin
            + 0.29 * p_spread
            + 0.23 * p_convex
            + 0.10 * p_kyle
        )
    else:
        # 3-component mode: redistribute optional weights proportionally
        score = (
            0.40 * p_thin
            + 0.33 * p_spread
            + 0.27 * p_convex
        )

    return float(score * 100.0)
