"""Funding-rate feature computation.

Input contract
--------------
DataFrame with columns:
    timestamp   : datetime64[ns, UTC], sorted ascending
    symbol      : str
    funding_rate: float64

Funding is settled every 8 hours on Bybit, so 1 period = 8 h.

Period shortcuts used throughout:
    3 periods  = 24 h
    9 periods  = 72 h (3 days)
    21 periods = 7 days
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .utils import robust_z

# ── Period constants (1 period = 8 h) ────────────────────────────────────────
_P24H = 3
_P72H = 9
_P7D = 21

# ── Public API ────────────────────────────────────────────────────────────────


def compute_funding_features(
    df: pd.DataFrame,
    z_window: int = 90,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Compute funding-rate features from historical data.

    Parameters
    ----------
    df:
        Historical funding rates for a single symbol, sorted ascending.
        Required columns: ``timestamp``, ``funding_rate``.
    z_window:
        Number of trailing periods used as the reference distribution for
        the robust z-score (default 90 ≈ 30 days).
    now:
        Reference wall-clock time for settlement-phase features.
        Defaults to ``datetime.now(timezone.utc)`` when ``None``.

    Returns
    -------
    Dict[str, Any]
        Feature name → value (float or str).  Numeric features that cannot
        be computed due to insufficient history are returned as
        ``float("nan")``.

    Features
    --------
    funding_current
        The most recent funding rate.
    funding_cum_24h
        Sum of the last 3 periods (= 24 h of cumulative funding).
    funding_cum_72h
        Sum of the last 9 periods (= 72 h).
    funding_cum_7d
        Sum of the last 21 periods (= 7 days).
    funding_z
        Robust z-score of the current rate relative to the trailing
        ``z_window`` periods.
    funding_ema_fast
        EWM mean with span=6 (≈ 2-day half-life).
    funding_ema_slow
        EWM mean with span=21 (≈ 7-day half-life).
    funding_ema_signal
        funding_ema_fast − funding_ema_slow (momentum direction).
    funding_streak
        Number of consecutive same-sign periods ending at the latest bar.
        Positive = consecutive positive funding; negative = consecutive
        negative funding.  Zero when the latest rate is exactly 0.
    minutes_to_settlement
        Minutes until the next 00:00 / 08:00 / 16:00 UTC settlement.
        Continuous range 0–480; 0 at the exact settlement moment.
    settlement_phase
        String label for the current market phase relative to settlement:
        ``"FAR"``        — more than 60 min until settlement
        ``"APPROACH"``   — 15–60 min until settlement
        ``"IMMINENT"``   — 0–15 min until settlement
        ``"POST_SETTLE"``— 0–15 min after the last settlement
    """
    nan = float("nan")
    out: Dict[str, Any] = {
        "funding_current": nan,
        "funding_cum_24h": nan,
        "funding_cum_72h": nan,
        "funding_cum_7d": nan,
        "funding_z": nan,
        "funding_ema_fast": nan,
        "funding_ema_slow": nan,
        "funding_ema_signal": nan,
        "funding_streak": nan,
        "minutes_to_settlement": nan,
        "settlement_phase": nan,
    }

    # ── Settlement timing (always computable, no history needed) ─────────────
    ref = now if now is not None else datetime.now(timezone.utc)
    mins = _minutes_to_settlement(ref)
    out["minutes_to_settlement"] = mins
    out["settlement_phase"] = _settlement_phase(mins)

    if df is None or df.empty:
        return out

    rates = df["funding_rate"].to_numpy(dtype=float)
    n = len(rates)

    # ── Current rate ─────────────────────────────────────────────────────────
    current = rates[-1]
    out["funding_current"] = float(current)

    # ── Cumulative sums ───────────────────────────────────────────────────────
    if n >= _P24H:
        out["funding_cum_24h"] = float(rates[-_P24H:].sum())
    if n >= _P72H:
        out["funding_cum_72h"] = float(rates[-_P72H:].sum())
    if n >= _P7D:
        out["funding_cum_7d"] = float(rates[-_P7D:].sum())

    # ── Robust z-score ────────────────────────────────────────────────────────
    window_vals = rates[-z_window:] if n >= 2 else rates
    out["funding_z"] = robust_z(current, window_vals)

    # ── EWM signals ───────────────────────────────────────────────────────────
    series = pd.Series(rates)
    ema_fast = float(series.ewm(span=6, adjust=False).mean().iloc[-1])
    ema_slow = float(series.ewm(span=21, adjust=False).mean().iloc[-1])
    out["funding_ema_fast"] = ema_fast
    out["funding_ema_slow"] = ema_slow
    out["funding_ema_signal"] = ema_fast - ema_slow

    # ── Streak ────────────────────────────────────────────────────────────────
    out["funding_streak"] = float(_streak(rates))

    return out


# ── Helpers ───────────────────────────────────────────────────────────────────


def _minutes_to_settlement(now: datetime) -> float:
    """Minutes until the next 00:00 / 08:00 / 16:00 UTC settlement.

    Returns 0.0 at the exact settlement moment; up to ~480.0 just after one.
    Works by computing how far into the current 8-hour period we are and
    subtracting from the period length.
    """
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    total_sec = now.hour * 3600 + now.minute * 60 + now.second + now.microsecond / 1e6
    period_sec = 8 * 3600  # 480 minutes
    into_period = total_sec % period_sec
    to_next_sec = (period_sec - into_period) % period_sec  # 0 at settlement moment
    return to_next_sec / 60.0


def _settlement_phase(minutes: float) -> str:
    """Classify the market phase relative to the upcoming settlement.

    Boundaries (minutes to next settlement):
        POST_SETTLE : > 465  (i.e. within 15 min after the last settlement)
        IMMINENT    : ≤ 15   (includes 0 = at settlement moment)
        APPROACH    : 15 < x ≤ 60
        FAR         : everything else (60 < x ≤ 465)
    """
    if minutes >= 465.0:    # 480 - 15 = 465 → within 15 min after last settle
        return "POST_SETTLE"
    if minutes <= 15.0:
        return "IMMINENT"
    if minutes <= 60.0:
        return "APPROACH"
    return "FAR"


def _streak(rates: np.ndarray) -> int:
    """Count consecutive same-sign periods from the end of the array.

    Returns a signed integer: positive for a positive-funding streak,
    negative for a negative-funding streak, 0 if the latest rate is 0.
    """
    if len(rates) == 0:
        return 0
    last = rates[-1]
    if last == 0:
        return 0
    sign = int(np.sign(last))
    count = 0
    for r in reversed(rates):
        if np.sign(r) == sign:
            count += 1
        else:
            break
    return sign * count
