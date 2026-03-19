"""Open-interest feature computation.

Input contract
--------------
DataFrame with columns:
    timestamp    : datetime64[ns, UTC], sorted ascending
    symbol       : str
    open_interest: float64

OI is snapshotted every 1 hour on Bybit, so 1 period = 1 h.

Period shortcuts used throughout:
    1  period  = 1 h
    4  periods = 4 h
    24 periods = 24 h (1 day)
    168 periods = 7 days
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
_P7D = 168

# ── Public API ─────────────────────────────────────────────────────────────────


def compute_oi_features(
    df: pd.DataFrame,
    z_window: int = 168,
) -> Dict[str, float]:
    """Compute open-interest features from historical data.

    Parameters
    ----------
    df:
        Historical open interest for a single symbol, sorted ascending.
        Required columns: ``timestamp``, ``open_interest``.
    z_window:
        Number of trailing 1h periods used as the reference distribution
        for robust z-scores (default 168 = 7 days).

    Returns
    -------
    Dict[str, float]
        Feature name → value.  Any feature that cannot be computed due to
        insufficient history is returned as ``float("nan")``.

    Features
    --------
    oi_current
        Latest raw open interest (in contracts).
    oi_delta_1h, oi_delta_4h, oi_delta_24h
        Absolute change in OI over 1 h / 4 h / 24 h.
    oi_pct_1h, oi_pct_4h, oi_pct_24h
        Percentage change in OI over 1 h / 4 h / 24 h.
    oi_z_1h
        Robust z-score of the current 1h pct-change relative to the
        trailing ``z_window`` 1h pct-changes.
    oi_z_24h
        Robust z-score of the current 24h pct-change relative to the
        trailing ``z_window`` 24h pct-changes (computed on non-overlapping
        daily periods).
    oi_ema_ratio
        current OI / EWM(span=24) — measures deviation from the medium-term
        trend.  Values above 1 mean OI is elevated vs. its recent average.
    oi_trend
        Normalised linear slope of OI over the last ``z_window`` periods,
        expressed as % change per period.  Positive = rising trend.
    """
    nan = float("nan")
    out: Dict[str, float] = {
        "oi_current": nan,
        "oi_delta_1h": nan,
        "oi_delta_4h": nan,
        "oi_delta_24h": nan,
        "oi_pct_1h": nan,
        "oi_pct_4h": nan,
        "oi_pct_24h": nan,
        "oi_z_1h": nan,
        "oi_z_24h": nan,
        "oi_ema_ratio": nan,
        "oi_trend": nan,
    }

    if df is None or df.empty:
        return out

    oi = df["open_interest"].to_numpy(dtype=float)
    n = len(oi)

    current = oi[-1]
    out["oi_current"] = float(current)

    # ── Absolute deltas ───────────────────────────────────────────────────────
    if n > _P1H:
        out["oi_delta_1h"] = float(current - oi[-(1 + _P1H)])
    if n > _P4H:
        out["oi_delta_4h"] = float(current - oi[-(1 + _P4H)])
    if n > _P24H:
        out["oi_delta_24h"] = float(current - oi[-(1 + _P24H)])

    # ── Percentage changes ────────────────────────────────────────────────────
    def _pct(lag: int) -> float:
        if n <= lag:
            return nan
        base = oi[-(1 + lag)]
        if base == 0 or np.isnan(base):
            return nan
        return float((current - base) / abs(base) * 100.0)

    out["oi_pct_1h"] = _pct(_P1H)
    out["oi_pct_4h"] = _pct(_P4H)
    out["oi_pct_24h"] = _pct(_P24H)

    # ── Robust z-scores ───────────────────────────────────────────────────────
    # z of 1h pct changes
    pct_changes_1h = _rolling_pct(oi, lag=_P1H)
    if len(pct_changes_1h) >= 2:
        current_pct_1h = out["oi_pct_1h"]
        if not np.isnan(current_pct_1h):
            ref = pct_changes_1h[-z_window:]
            out["oi_z_1h"] = robust_z(current_pct_1h, ref)

    # z of 24h pct changes (non-overlapping daily steps)
    pct_changes_24h = _rolling_pct(oi, lag=_P24H)
    if len(pct_changes_24h) >= 2:
        current_pct_24h = out["oi_pct_24h"]
        if not np.isnan(current_pct_24h):
            ref = pct_changes_24h[-z_window:]
            out["oi_z_24h"] = robust_z(current_pct_24h, ref)

    # ── EMA ratio ─────────────────────────────────────────────────────────────
    series = pd.Series(oi)
    ema24 = float(series.ewm(span=_P24H, adjust=False).mean().iloc[-1])
    if ema24 != 0 and not np.isnan(ema24):
        out["oi_ema_ratio"] = float(current / ema24)

    # ── Trend slope ───────────────────────────────────────────────────────────
    window_oi = oi[-z_window:] if n >= z_window else oi
    if len(window_oi) >= 2:
        out["oi_trend"] = _normalised_slope(window_oi)

    return out


# ── Helpers ───────────────────────────────────────────────────────────────────


def _rolling_pct(oi: np.ndarray, lag: int) -> np.ndarray:
    """Return array of (oi[i] - oi[i-lag]) / |oi[i-lag]| * 100 for all valid i."""
    if len(oi) <= lag:
        return np.array([], dtype=float)
    base = oi[:-lag]
    curr = oi[lag:]
    with np.errstate(invalid="ignore", divide="ignore"):
        pct = np.where(base != 0, (curr - base) / np.abs(base) * 100.0, np.nan)
    return pct


def _normalised_slope(arr: np.ndarray) -> float:
    """Linear regression slope normalised to the mean of the series.

    Result is expressed as % change per period.  Robust to scale differences
    across symbols (BTC OI in tens of thousands vs small-cap OI in hundreds).
    """
    clean = arr[~np.isnan(arr)]
    if len(clean) < 2:
        return float("nan")
    mean_val = np.mean(clean)
    if mean_val == 0:
        return float("nan")
    x = np.arange(len(clean), dtype=float)
    slope = np.polyfit(x, clean, 1)[0]
    return float(slope / abs(mean_val) * 100.0)
