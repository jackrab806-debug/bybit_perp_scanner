"""Price and volume flow feature computation.

Input contract
--------------
DataFrame with columns:
    timestamp : datetime64[ns, UTC], sorted ascending
    symbol    : str
    open      : float64
    high      : float64
    low       : float64
    close     : float64
    volume    : float64   (base-asset contracts)
    turnover  : float64   (USDT notional)

Data source: 1-hour klines from the backfill.  1 period = 1 h.

Taker-buy proxy
---------------
Without trade-level data we infer buying/selling pressure from bar structure:

    taker_proxy = (close − low) / (high − low)

    0.0  →  closed at the low  →  sellers dominated
    1.0  →  closed at the high →  buyers dominated
    0.5  →  neutral (or doji)

The per-bar delta (signed pressure × volume):

    delta = (2 × taker_proxy − 1) × volume

Summing deltas over N bars gives a cumulative volume delta (CVD) proxy.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .utils import robust_z

# ── Period constants (1 period = 1 h) ─────────────────────────────────────────
_P1H = 1
_P4H = 4
_P24H = 24
_ATR_PERIOD = 14

# ── Public API ─────────────────────────────────────────────────────────────────


def compute_flow_features(
    df: pd.DataFrame,
    z_window: int = 168,
) -> Dict[str, float]:
    """Compute price and volume flow features from OHLCV klines.

    Parameters
    ----------
    df:
        1-hour OHLCV klines for a single symbol, sorted ascending.
        Required columns: ``open``, ``high``, ``low``, ``close``, ``volume``.
    z_window:
        Trailing window (in 1h periods) used as reference distribution for
        robust z-scores (default 168 = 7 days).

    Returns
    -------
    Dict[str, float]
        Feature name → value.  NaN when insufficient history.

    Features
    --------
    return_1h, return_4h, return_24h
        Simple percentage price return over 1 / 4 / 24 bars.
    atr_pct
        14-period Average True Range expressed as % of current close.
        Proxy for realised short-term volatility.
    vol_z
        Robust z-score of the current bar's volume relative to the trailing
        ``z_window`` bars.
    vol_spike_ratio
        Current volume divided by the EWM(span=24) volume.  > 2 = spike.
    taker_proxy
        (close − low) / (high − low) for the current bar.  0 = all sell,
        1 = all buy.  0.5 on a doji (high == low).
    taker_proxy_ema
        EWM(span=6) of the per-bar taker_proxy — smoothed directional
        pressure over the last ~6 hours.
    cvd_ratio_24h
        Cumulative volume delta over the last 24 bars, normalised by total
        volume → range roughly −1 to +1.
        Positive = net buying pressure; negative = net selling.
    cvd_z
        Robust z-score of the current bar's signed delta relative to the
        trailing ``z_window`` bar deltas.
    price_accel
        Current 1h return minus the trailing mean 1h return (excess
        momentum).  Positive = accelerating upward vs. recent average.
    """
    nan = float("nan")
    out: Dict[str, float] = {
        "return_1h": nan,
        "return_4h": nan,
        "return_24h": nan,
        "atr_pct": nan,
        "vol_z": nan,
        "vol_spike_ratio": nan,
        "taker_proxy": nan,
        "taker_proxy_ema": nan,
        "cvd_ratio_24h": nan,
        "cvd_z": nan,
        "price_accel": nan,
    }

    if df is None or df.empty:
        return out

    close = df["close"].to_numpy(dtype=float)
    high  = df["high"].to_numpy(dtype=float)
    low   = df["low"].to_numpy(dtype=float)
    opn   = df["open"].to_numpy(dtype=float)
    vol   = df["volume"].to_numpy(dtype=float)
    n     = len(close)

    cur_close = close[-1]
    cur_vol   = vol[-1]

    # ── Price returns ─────────────────────────────────────────────────────────
    def _ret(lag: int) -> float:
        if n <= lag:
            return nan
        base = close[-(1 + lag)]
        if base == 0 or np.isnan(base):
            return nan
        return float((cur_close - base) / base * 100.0)

    out["return_1h"]  = _ret(_P1H)
    out["return_4h"]  = _ret(_P4H)
    out["return_24h"] = _ret(_P24H)

    # ── ATR (14-period) ───────────────────────────────────────────────────────
    if n >= _ATR_PERIOD + 1:
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        tr = np.maximum.reduce([
            high - low,
            np.abs(high - prev_close),
            np.abs(low  - prev_close),
        ])
        atr = float(pd.Series(tr).ewm(span=_ATR_PERIOD, adjust=False).mean().iloc[-1])
        if cur_close > 0:
            out["atr_pct"] = atr / cur_close * 100.0

    # ── Volume features ───────────────────────────────────────────────────────
    ref_vol = vol[-z_window:] if n >= 2 else vol
    out["vol_z"] = robust_z(cur_vol, ref_vol)

    ema_vol = float(pd.Series(vol).ewm(span=_P24H, adjust=False).mean().iloc[-1])
    if ema_vol > 0:
        out["vol_spike_ratio"] = float(cur_vol / ema_vol)

    # ── Taker-buy proxy ───────────────────────────────────────────────────────
    hl_range = high - low
    # Avoid division by zero on doji bars (high == low)
    with np.errstate(invalid="ignore", divide="ignore"):
        taker = np.where(hl_range > 0, (close - low) / hl_range, 0.5)

    out["taker_proxy"] = float(taker[-1])
    out["taker_proxy_ema"] = float(
        pd.Series(taker).ewm(span=6, adjust=False).mean().iloc[-1]
    )

    # ── CVD proxy ─────────────────────────────────────────────────────────────
    # Per-bar signed delta: (2×taker - 1) × volume
    delta = (2.0 * taker - 1.0) * vol   # range: [-vol, +vol]

    cur_delta = delta[-1]
    ref_delta = delta[-z_window:] if n >= 2 else delta
    out["cvd_z"] = robust_z(cur_delta, ref_delta)

    if n >= _P24H:
        window_delta = delta[-_P24H:]
        window_vol   = vol[-_P24H:]
        total_vol = window_vol.sum()
        if total_vol > 0:
            out["cvd_ratio_24h"] = float(window_delta.sum() / total_vol)

    # ── Price acceleration ────────────────────────────────────────────────────
    # Compare current 1h return against the trailing mean 1h return
    if n >= _P24H + 1:
        returns_1h = (close[1:] - close[:-1]) / np.where(close[:-1] > 0, close[:-1], np.nan) * 100.0
        mean_ret = float(np.nanmean(returns_1h[-_P24H:]))
        cur_ret = out["return_1h"]
        if not np.isnan(cur_ret):
            out["price_accel"] = cur_ret - mean_ret

    return out
