"""Volatility feature computation from 1-hour OHLCV klines.

Input contract
--------------
DataFrame with columns:
    timestamp : datetime64[ns, UTC], sorted ascending
    symbol    : str
    open      : float64
    high      : float64
    low       : float64
    close     : float64
    volume    : float64
    turnover  : float64

Output values used by composite scoring
-----------------------------------------
rv_pct
    Annualised realised volatility over a 24-hour window, expressed as a
    percentage.  Typical BTC values: 20–80 %.  Used in compression_score
    with a threshold of 20 (vol_compression = 0 when rv_pct ≥ 20).

bb_width_pct
    Bollinger Band width (20-bar, ±2 σ) expressed as a percentage of the
    midline.  Typical values: 2–15 %.  Used in compression_score with the
    same threshold of 20.

range_hours
    Number of consecutive 1-hour bars (counting backwards from the latest)
    that are "compressed" — defined as a bar whose high-low range is below
    the rolling median range over the trailing ``z_window`` bars.  Used in
    compression_score; a value of 12 maps to a 100 % duration factor.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .utils import robust_z

# ── Annualisation constant (1h bars) ──────────────────────────────────────────
_PERIODS_PER_YEAR = 24 * 365   # 8 760

# ── Public API ─────────────────────────────────────────────────────────────────


def compute_volatility_features(
    df: pd.DataFrame,
    z_window: int = 168,
) -> Dict[str, float]:
    """Compute volatility features from 1-hour OHLCV klines.

    Parameters
    ----------
    df:
        1-hour OHLCV klines for a single symbol, sorted ascending.
        Required columns: ``high``, ``low``, ``close``.
    z_window:
        Trailing window (in 1h periods) for z-score reference distributions
        and the dynamic compression threshold (default 168 = 7 days).

    Returns
    -------
    Dict[str, float]
        Feature name → value.  NaN when insufficient history.

    Features
    --------
    rv_pct
        Annualised realised volatility over the last 24 bars, in percent.
        ``std(log_returns[-24:]) × √8760 × 100``.
    rv_pct_7d
        Same, over the last 168 bars (7 days).
    bb_width_pct
        20-bar Bollinger Band width (upper − lower) as a percentage of the
        midline.  ``4 × rolling_std(close, 20) / rolling_mean(close, 20) × 100``.
    range_24h_pct
        Simple high-low range over the last 24 bars as a percentage of the
        current close.  Fast proxy for short-term price excursion.
    range_hours
        Count of consecutive "compressed" bars ending at the latest bar.
        A bar is compressed when its ``(high − low) / close`` is below the
        rolling median of the same measure over the trailing ``z_window``
        bars.  Proxy for how long the current coiling has been building.
    rv_z
        Robust z-score of the current ``rv_pct`` (24 h) relative to its
        own trailing ``z_window`` history.
    bb_z
        Robust z-score of the current ``bb_width_pct`` relative to its
        own trailing ``z_window`` history.
    """
    nan = float("nan")
    out: Dict[str, float] = {
        "rv_pct": nan,
        "rv_pct_7d": nan,
        "bb_width_pct": nan,
        "range_24h_pct": nan,
        "range_hours": nan,
        "rv_z": nan,
        "bb_z": nan,
    }

    if df is None or df.empty:
        return out

    close = df["close"].to_numpy(dtype=float)
    high  = df["high"].to_numpy(dtype=float)
    low   = df["low"].to_numpy(dtype=float)
    n     = len(close)

    # ── Log returns ───────────────────────────────────────────────────────────
    with np.errstate(invalid="ignore", divide="ignore"):
        log_ret = np.where(
            close[:-1] > 0,
            np.log(close[1:] / close[:-1]),
            np.nan,
        )

    # ── Realised volatility (annualised, %) ───────────────────────────────────
    _RV24 = 24
    _RV7D = 168
    ann = np.sqrt(_PERIODS_PER_YEAR) * 100.0

    if len(log_ret) >= _RV24:
        rv_24h = float(np.nanstd(log_ret[-_RV24:]) * ann)
        out["rv_pct"] = rv_24h

        # Rolling rv for z-score (compute rv at each bar, trailing 24h window)
        if n >= _RV24 + 1 and z_window >= 2:
            rv_series = np.array([
                np.nanstd(log_ret[max(0, i - _RV24):i]) * ann
                for i in range(_RV24, len(log_ret) + 1)
            ])
            ref_rv = rv_series[-z_window:] if len(rv_series) >= z_window else rv_series
            out["rv_z"] = robust_z(rv_24h, ref_rv)

    if len(log_ret) >= _RV7D:
        out["rv_pct_7d"] = float(np.nanstd(log_ret[-_RV7D:]) * ann)

    # ── Bollinger Band width (20-bar) ─────────────────────────────────────────
    _BB = 20
    if n >= _BB:
        series = pd.Series(close)
        bb_mid = float(series.rolling(_BB).mean().iloc[-1])
        bb_std = float(series.rolling(_BB).std(ddof=1).iloc[-1])

        if bb_mid > 0 and not np.isnan(bb_std):
            bb_w = 4.0 * bb_std / bb_mid * 100.0  # 2σ above + 2σ below
            out["bb_width_pct"] = bb_w

            # Rolling bb_width for z-score
            if n >= _BB + 1 and z_window >= 2:
                roll_mean = series.rolling(_BB).mean()
                roll_std  = series.rolling(_BB).std(ddof=1)
                bb_series = (4.0 * roll_std / roll_mean * 100.0).dropna().to_numpy()
                ref_bb = bb_series[-z_window:] if len(bb_series) >= z_window else bb_series
                out["bb_z"] = robust_z(bb_w, ref_bb)

    # ── 24-hour high-low range ────────────────────────────────────────────────
    _R24 = 24
    if n >= _R24 and close[-1] > 0:
        hi24 = np.max(high[-_R24:])
        lo24 = np.min(low[-_R24:])
        out["range_24h_pct"] = float((hi24 - lo24) / close[-1] * 100.0)

    # ── Compression duration (range_hours) ────────────────────────────────────
    with np.errstate(invalid="ignore", divide="ignore"):
        bar_range_pct = np.where(close > 0, (high - low) / close * 100.0, np.nan)

    ref_window = bar_range_pct[-z_window:] if n >= z_window else bar_range_pct
    median_range = float(np.nanmedian(ref_window))

    count = 0
    for rng in reversed(bar_range_pct):
        if not np.isnan(rng) and rng < median_range:
            count += 1
        else:
            break
    out["range_hours"] = float(count)

    return out
