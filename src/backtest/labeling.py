"""Triple-barrier labeling for detected events.

Given an Event and a price series (any bar resolution), computes:
  - entry price at the first bar on or after event.timestamp
  - TP barrier at entry * (1 + direction * 2.0 * sigma)
  - SL barrier at entry * (1 - direction * 1.5 * sigma)
  - outcome: TP | SL | TIMEOUT (within max_bars)
  - time_to_outcome_s: seconds from entry to barrier touch
  - mfe_15m / mae_15m: max favorable / adverse excursion within 15 minutes
  - mfe_60m / mae_60m: same for 60 minutes
  - time_to_mfe_s: seconds from entry to max favorable price (within 60 min)

All excursions are expressed as fractions of the entry price.

Usage
-----
    from src.backtest.labeling import label_event, label_events

    # Single event
    labeled = label_event(event, price_series, sigma=0.002)

    # Batch
    labeled_list = label_events(events, price_map, sigma_map)

Notes
-----
``max_bars`` is a *bar count*, not a time duration.  If your price_series
has 1-second resolution, max_bars=300 ≈ 5 minutes.  If it has 1-minute
resolution, max_bars=300 ≈ 5 hours.  Pass an appropriate value for your
data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..events.definitions import Event

# ── Default barrier parameters ─────────────────────────────────────────────────

_TP_MULT  = 2.0     # take-profit multiple of sigma
_SL_MULT  = 1.5     # stop-loss multiple of sigma
_MAX_BARS = 300     # vertical barrier: 300 bars ≈ 5 min @ 1s resolution

# ── Excursion windows ─────────────────────────────────────────────────────────

_15M_S = 15 * 60   # 900 seconds
_60M_S = 60 * 60   # 3 600 seconds


# ── Helpers ────────────────────────────────────────────────────────────────────


def _direction_int(direction_str: str) -> int:
    """Convert "LONG"/"SHORT"/"NEUTRAL" to +1 / -1 / 0."""
    return 1 if direction_str == "LONG" else -1 if direction_str == "SHORT" else 0


# ── LabeledEvent ──────────────────────────────────────────────────────────────


@dataclass
class LabeledEvent:
    """
    Triple-barrier outcome for a single detected (or baseline) event.

    Attributes
    ----------
    event_id, timestamp, symbol, event_type :
        Copied from the source Event.
    direction : int
        +1 LONG, -1 SHORT, 0 NEUTRAL.
    score : float
        Event score at detection time.
    entry_price : float
        Mid-price at the first bar on or after event.timestamp.
    sigma : float
        Realized vol fraction used to size the barriers.
    tp_barrier, sl_barrier : float
        Absolute price levels of the take-profit / stop-loss barriers.
    hit_tp, hit_sl : bool
        Whether the respective barrier was touched first.
    outcome : str
        "TP" | "SL" | "TIMEOUT".
    time_to_outcome_s : float | None
        Seconds from entry bar to barrier touch, or None on TIMEOUT.
    mfe_15m, mae_15m : float
        Max favorable / adverse excursion within 15 min, as fraction of entry.
    mfe_60m, mae_60m : float
        Same for 60 minutes.
    time_to_mfe_s : float
        Seconds from entry to max favorable price within the 60-min window.
    """

    # Identity
    event_id:   str
    timestamp:  datetime
    symbol:     str
    event_type: str
    direction:  int     # +1, -1, 0
    score:      float

    # Entry and barriers
    entry_price: float
    sigma:       float
    tp_barrier:  float
    sl_barrier:  float

    # Outcome
    hit_tp:            bool
    hit_sl:            bool
    outcome:           str    # "TP" | "SL" | "TIMEOUT"
    time_to_outcome_s: Optional[float]

    # Excursions (fraction of entry)
    mfe_15m: float
    mae_15m: float
    mfe_60m: float
    mae_60m: float
    time_to_mfe_s: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id":          self.event_id,
            "timestamp":         self.timestamp.isoformat(),
            "symbol":            self.symbol,
            "event_type":        self.event_type,
            "direction":         self.direction,
            "score":             self.score,
            "entry_price":       self.entry_price,
            "sigma":             self.sigma,
            "tp_barrier":        self.tp_barrier,
            "sl_barrier":        self.sl_barrier,
            "hit_tp":            self.hit_tp,
            "hit_sl":            self.hit_sl,
            "outcome":           self.outcome,
            "time_to_outcome_s": self.time_to_outcome_s,
            "mfe_15m":           self.mfe_15m,
            "mae_15m":           self.mae_15m,
            "mfe_60m":           self.mfe_60m,
            "mae_60m":           self.mae_60m,
            "time_to_mfe_s":     self.time_to_mfe_s,
        }


# ── Core labeling function ─────────────────────────────────────────────────────


def label_event(
    event: Event,
    price_series: pd.Series,
    sigma: float,
    max_bars: int = _MAX_BARS,
) -> Optional[LabeledEvent]:
    """
    Apply triple-barrier labeling to a single Event.

    Parameters
    ----------
    event :
        Event from ``src.events.definitions``.
    price_series :
        ``pd.Series`` with a UTC-aware ``DatetimeIndex`` (sorted ascending)
        and float values representing the mid-price at each bar.
    sigma :
        Realized volatility fraction for this trade window
        (e.g., 0.003 for 0.3%).  Used to size the TP/SL barriers.
    max_bars :
        Maximum number of bars to look forward (vertical barrier).

    Returns
    -------
    LabeledEvent or None if the price series has no data on or after
    ``event.timestamp``, or if ``event.direction`` is NEUTRAL.
    """
    direction = _direction_int(event.direction)
    if direction == 0:
        return None

    # ── Locate entry bar ──────────────────────────────────────────────────────
    ts = event.timestamp
    # Normalise tz
    if ts.tzinfo is not None:
        idx_utc = price_series.index.tz_convert("UTC") if price_series.index.tz is not None \
                  else price_series.index.tz_localize("UTC")
    else:
        idx_utc = price_series.index.tz_localize("UTC") if price_series.index.tz is None \
                  else price_series.index.tz_convert("UTC")
        ts = ts.replace(tzinfo=__import__("datetime").timezone.utc)

    after_mask = idx_utc >= ts
    if not after_mask.any():
        return None

    entry_pos   = int(np.argmax(after_mask))
    entry_idx   = price_series.index[entry_pos]
    entry       = float(price_series.iloc[entry_pos])
    t0_s        = entry_idx.timestamp()

    tp_barrier  = entry * (1.0 + direction * _TP_MULT * sigma)
    sl_barrier  = entry * (1.0 - direction * _SL_MULT * sigma)

    # ── Triple-barrier outcome ────────────────────────────────────────────────
    window = price_series.iloc[entry_pos : entry_pos + max_bars]

    hit_tp = False
    hit_sl = False
    time_to_outcome_s: Optional[float] = None

    for bar_ts, price in zip(window.index, window.values):
        elapsed = bar_ts.timestamp() - t0_s
        if direction == 1:
            if price >= tp_barrier:
                hit_tp = True
                time_to_outcome_s = elapsed
                break
            if price <= sl_barrier:
                hit_sl = True
                time_to_outcome_s = elapsed
                break
        else:  # direction == -1
            if price <= tp_barrier:
                hit_tp = True
                time_to_outcome_s = elapsed
                break
            if price >= sl_barrier:
                hit_sl = True
                time_to_outcome_s = elapsed
                break

    outcome = "TP" if hit_tp else "SL" if hit_sl else "TIMEOUT"

    # ── Excursions ────────────────────────────────────────────────────────────

    def _excursions(seconds: float) -> Tuple[float, float, float]:
        cutoff  = entry_idx + pd.Timedelta(seconds=seconds)
        w       = price_series.loc[entry_idx:cutoff]
        if w.empty:
            return 0.0, 0.0, 0.0
        # moves[i] > 0 = favorable for the trade direction
        moves     = (w.values - entry) / entry * direction
        mfe       = float(max(float(moves.max()), 0.0))
        mae       = float(max(float((-moves).max()), 0.0))
        mfe_pos   = int(np.argmax(moves))
        t_to_mfe  = w.index[mfe_pos].timestamp() - t0_s
        return mfe, mae, t_to_mfe

    mfe_15m, mae_15m, _        = _excursions(_15M_S)
    mfe_60m, mae_60m, t_to_mfe = _excursions(_60M_S)

    event_type_str = (
        event.event_type.value
        if hasattr(event.event_type, "value")
        else str(event.event_type)
    )

    return LabeledEvent(
        event_id=event.event_id,
        timestamp=event.timestamp,
        symbol=event.symbol,
        event_type=event_type_str,
        direction=direction,
        score=event.score,
        entry_price=entry,
        sigma=sigma,
        tp_barrier=tp_barrier,
        sl_barrier=sl_barrier,
        hit_tp=hit_tp,
        hit_sl=hit_sl,
        outcome=outcome,
        time_to_outcome_s=time_to_outcome_s,
        mfe_15m=mfe_15m,
        mae_15m=mae_15m,
        mfe_60m=mfe_60m,
        mae_60m=mae_60m,
        time_to_mfe_s=t_to_mfe,
    )


# ── Batch helper ──────────────────────────────────────────────────────────────


def label_events(
    events:    List[Event],
    price_map: Dict[str, pd.Series],
    sigma_map: Dict[str, float],
    max_bars:  int = _MAX_BARS,
) -> List[LabeledEvent]:
    """
    Batch-label a list of Events.

    Parameters
    ----------
    events :
        List of Event objects from ``src.events.definitions``.
    price_map :
        Dict mapping symbol → price Series (UTC DatetimeIndex, mid-prices).
    sigma_map :
        Dict mapping symbol → realized vol fraction.
        Defaults to 0.002 (0.2%) for symbols not in the map.
    max_bars :
        Vertical barrier bar count (passed to label_event).

    Returns
    -------
    List of LabeledEvent (NEUTRAL events and symbols without price data
    are silently skipped).
    """
    results: List[LabeledEvent] = []
    for ev in events:
        series = price_map.get(ev.symbol)
        if series is None or series.empty:
            continue
        sigma   = sigma_map.get(ev.symbol, 0.002)
        labeled = label_event(ev, series, sigma, max_bars)
        if labeled is not None:
            results.append(labeled)
    return results
