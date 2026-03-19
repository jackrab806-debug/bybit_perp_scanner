"""Orderbook snapshot feature computation.

Input contract
--------------
Bybit v5 snapshot (from ``BybitRestClient.get_orderbook`` or the WS feed):
    bids : List[List[str]]  – [["price", "size"], ...], sorted highest→lowest
    asks : List[List[str]]  – [["price", "size"], ...], sorted lowest→highest

All functions accept either string or numeric price/size pairs.

Key concept — vacuum distance
------------------------------
The "vacuum" in the bid (ask) book is the number of basis points away from
mid you need to reach before you can fill *target_q_usdt* of resting
liquidity.  A large vacuum means thin liquidity → the price can move far
before hitting meaningful support/resistance.

    vacuum_imbalance > 0  →  ask side is thinner  →  bullish pressure
    vacuum_imbalance < 0  →  bid side is thinner   →  bearish pressure
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


# ── Public helpers (importable standalone) ────────────────────────────────────


def vacuum_dist_bid(
    bids_sorted: List[Tuple[float, float]],
    mid: float,
    target_q_usdt: float,
) -> float:
    """Basis points from mid needed to fill *target_q_usdt* on the bid side.

    Iterates bids from best (highest price) downward, accumulating USDT
    notional.  Returns 9999 when the entire visible book is shallower than
    *target_q_usdt* (extreme vacuum).

    Parameters
    ----------
    bids_sorted:
        [(price, size), ...] sorted highest→lowest.
    mid:
        Midpoint price.
    target_q_usdt:
        Target notional to fill (in quote currency, typically USDT).
    """
    cumulative = 0.0
    for price, size in bids_sorted:  # highest to lowest
        cumulative += price * size
        if cumulative >= target_q_usdt:
            return (mid - price) / mid * 10_000  # bps
    return 9_999.0  # extreme vacuum: not enough depth in entire book


def vacuum_dist_ask(
    asks_sorted: List[Tuple[float, float]],
    mid: float,
    target_q_usdt: float,
) -> float:
    """Basis points from mid needed to fill *target_q_usdt* on the ask side.

    Iterates asks from best (lowest price) upward, accumulating USDT
    notional.  Returns 9999 when the entire visible book is insufficient.

    Parameters
    ----------
    asks_sorted:
        [(price, size), ...] sorted lowest→highest.
    mid:
        Midpoint price.
    target_q_usdt:
        Target notional to fill (in quote currency, typically USDT).
    """
    cumulative = 0.0
    for price, size in asks_sorted:  # lowest to highest
        cumulative += price * size
        if cumulative >= target_q_usdt:
            return (price - mid) / mid * 10_000  # bps
    return 9_999.0  # extreme vacuum: not enough depth in entire book


def thin_pct(
    snapshot: Dict[str, Any],
    history: Union[np.ndarray, List[float]],
) -> float:
    """Percentile rank of current book thinness vs. a historical reference.

    Thinness is defined as ``1 / (depth_bid_usdt + depth_ask_usdt)``.  A
    thinner book (less total resting liquidity) produces a higher thinness
    value, so a high percentile rank means the book is currently more
    vacuous than usual.

    Parameters
    ----------
    snapshot:
        Raw Bybit v5 orderbook dict with ``"b"`` / ``"a"`` lists.
    history:
        Array of previously observed thinness values — typically the last
        7 days of per-snapshot values (e.g. 168 hourly samples).  NaN
        entries are silently dropped.

    Returns
    -------
    float
        Percentile rank in [0, 1].  ``> 0.90`` signals an unusual vacuum.
        ``nan`` if the book is empty or history has fewer than 2 values.
    """
    raw_bids = snapshot.get("b", [])
    raw_asks = snapshot.get("a", [])
    if not raw_bids or not raw_asks:
        return float("nan")

    bids = [(float(p), float(s)) for p, s in raw_bids]
    asks = [(float(p), float(s)) for p, s in raw_asks]
    total_usdt = sum(p * s for p, s in bids) + sum(p * s for p, s in asks)
    if total_usdt == 0:
        return float("nan")

    thinness = 1.0 / total_usdt

    arr = np.asarray(history, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        return float("nan")

    # Fraction of history that is <= current thinness → high = unusually thin
    return float(np.mean(arr <= thinness))


# ── Main feature function ─────────────────────────────────────────────────────


def compute_orderbook_features(
    snapshot: Dict[str, Any],
    target_q_usdt: float = 100_000,
    depth_band_pct: float = 1.0,
    history: Optional[Union[np.ndarray, List[float]]] = None,
) -> Dict[str, float]:
    """Compute orderbook snapshot features.

    Parameters
    ----------
    snapshot:
        Raw Bybit v5 orderbook dict with keys ``"b"`` (bids) and ``"a"``
        (asks), each a list of ``[price, size]`` string pairs.
    target_q_usdt:
        Notional threshold for vacuum-distance calculation (default $100k).
    depth_band_pct:
        Width of the symmetric depth band around mid, as a percentage
        (default 1.0 → look at bids/asks within ±1 % of mid).
    history:
        Optional array of past thinness values for ``thin_pct`` computation.
        When ``None``, ``thin_pct`` is omitted from the output.

    Returns
    -------
    Dict[str, float]
        Feature name → value.  NaN for features that cannot be computed
        (e.g. empty book).

    Features
    --------
    mid_price
        (best_bid + best_ask) / 2.
    spread_bps
        Bid-ask spread in basis points.
    depth_bid_usdt, depth_ask_usdt
        Total USDT notional resting in bids / asks visible in the snapshot.
    depth_ratio
        depth_bid_usdt / depth_ask_usdt.  > 1 → more bid-side liquidity.
    depth_imbalance
        (bid − ask) / (bid + ask).  Signed: +1 = all bids, −1 = all asks.
    vacuum_dist_bid, vacuum_dist_ask
        Bps from mid to fill *target_q_usdt* on each side.  9999 = extreme.
    vacuum_imbalance
        (vac_ask − vac_bid) / (vac_ask + vac_bid).
        +1 = ask side totally empty → upward pressure.
        −1 = bid side totally empty → downward pressure.
    depth_band_bid_usdt, depth_band_ask_usdt
        USDT depth within *depth_band_pct* % of mid on each side.
    depth_band_imbalance
        Signed imbalance within the depth band.
    inner_depth_10
        Raw size (contracts) summed across the top-10 bid levels and
        top-10 ask levels.  Measures how much contract volume sits at
        the very front of the book.
    convexity
        ``sum(size levels 11–50) / (sum(size levels 1–10) + ε)``.
        High value → outer book is thick relative to inner book → makers
        have pulled from the top, leaving a fragile front.
    thin_pct  *(only present when* ``history`` *is provided)*
        Percentile rank of current thinness vs. historical values.
        ``> 0.90`` signals an unusual vacuum in total resting liquidity.
    """
    nan = float("nan")
    out: Dict[str, float] = {
        "mid_price": nan,
        "spread_bps": nan,
        "depth_bid_usdt": nan,
        "depth_ask_usdt": nan,
        "depth_ratio": nan,
        "depth_imbalance": nan,
        "vacuum_dist_bid": nan,
        "vacuum_dist_ask": nan,
        "vacuum_imbalance": nan,
        "depth_band_bid_usdt": nan,
        "depth_band_ask_usdt": nan,
        "depth_band_imbalance": nan,
        "inner_depth_10": nan,
        "convexity": nan,
    }

    raw_bids = snapshot.get("b", [])
    raw_asks = snapshot.get("a", [])

    if not raw_bids or not raw_asks:
        return out

    # Parse to (float, float) tuples
    bids: List[Tuple[float, float]] = [(float(p), float(s)) for p, s in raw_bids]
    asks: List[Tuple[float, float]] = [(float(p), float(s)) for p, s in raw_asks]

    best_bid = bids[0][0]   # highest bid
    best_ask = asks[0][0]   # lowest ask

    if best_bid <= 0 or best_ask <= 0 or best_ask <= best_bid:
        return out

    mid = (best_bid + best_ask) / 2.0
    out["mid_price"] = mid
    out["spread_bps"] = (best_ask - best_bid) / mid * 10_000

    # ── Total depth ───────────────────────────────────────────────────────────
    bid_usdt = sum(p * s for p, s in bids)
    ask_usdt = sum(p * s for p, s in asks)
    out["depth_bid_usdt"] = bid_usdt
    out["depth_ask_usdt"] = ask_usdt

    total_usdt = bid_usdt + ask_usdt
    if total_usdt > 0:
        out["depth_ratio"] = bid_usdt / ask_usdt if ask_usdt > 0 else 9_999.0
        out["depth_imbalance"] = (bid_usdt - ask_usdt) / total_usdt

    # ── Vacuum distances ──────────────────────────────────────────────────────
    vd_bid = vacuum_dist_bid(bids, mid, target_q_usdt)
    vd_ask = vacuum_dist_ask(asks, mid, target_q_usdt)
    out["vacuum_dist_bid"] = vd_bid
    out["vacuum_dist_ask"] = vd_ask

    vac_sum = vd_bid + vd_ask
    if vac_sum > 0:
        out["vacuum_imbalance"] = (vd_ask - vd_bid) / vac_sum

    # ── Depth-band imbalance (within ±depth_band_pct % of mid) ───────────────
    band = depth_band_pct / 100.0
    band_bid = sum(p * s for p, s in bids if p >= mid * (1.0 - band))
    band_ask = sum(p * s for p, s in asks if p <= mid * (1.0 + band))
    out["depth_band_bid_usdt"] = band_bid
    out["depth_band_ask_usdt"] = band_ask

    band_total = band_bid + band_ask
    if band_total > 0:
        out["depth_band_imbalance"] = (band_bid - band_ask) / band_total

    # ── Inner depth & convexity (raw contract size, not USDT) ─────────────────
    _INNER = 10   # top levels that count as "inner"
    _OUTER_END = 50

    inner_bid = sum(s for _, s in bids[:_INNER])
    inner_ask = sum(s for _, s in asks[:_INNER])
    inner = inner_bid + inner_ask
    out["inner_depth_10"] = inner

    outer_bid = sum(s for _, s in bids[_INNER:_OUTER_END])
    outer_ask = sum(s for _, s in asks[_INNER:_OUTER_END])
    outer = outer_bid + outer_ask
    out["convexity"] = outer / (inner + 1e-10)

    # ── Thin percentile (only when history provided) ───────────────────────────
    if history is not None:
        out["thin_pct"] = thin_pct(snapshot, history)

    return out
