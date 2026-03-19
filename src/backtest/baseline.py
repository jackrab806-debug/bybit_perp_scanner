"""Matched random baseline generator for event validation.

For each detected event, draws N random timestamps from the price history
whose market conditions match the event's context:
  - Same symbol
  - Same day of week
  - Same hour of day (±2 hours)
  - Similar realized volatility (within 20%)
  - NOT within 60 minutes of any real event

Applies the same triple-barrier labeling to each baseline sample so that
event and baseline LabeledEvent lists are directly comparable.

Usage
-----
    from src.backtest.baseline import BaselineSampler, create_baseline

    sampler  = BaselineSampler(price_map, sigma_map, seed=42)
    baselines = sampler.sample(event, all_event_timestamps, n_samples=10)

    # Or convenience wrapper (creates a fresh sampler per call):
    baselines = create_baseline(event, price_map, all_event_timestamps,
                                sigma_map, n_samples=10)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..events.definitions import Event
from .labeling import LabeledEvent, label_event

# ── Matching parameters ───────────────────────────────────────────────────────

_EXCLUDE_MIN  = 60     # exclude ± this many minutes around real events
_HOUR_WINDOW  = 2      # match hour of day ± 2 hours
_RV_TOL       = 0.20   # require |candidate_rv - event_rv| / event_rv ≤ 0.20
_RV_WIN_MIN   = 60     # rolling window (minutes) to compute local RV
_MAX_ATTEMPTS = 5_000  # stop trying after this many candidate draws


# ── Rolling realized volatility ───────────────────────────────────────────────


def _rolling_rv(price_series: pd.Series, window_min: int = _RV_WIN_MIN) -> pd.Series:
    """
    Annualized realized volatility computed from a price series.

    Parameters
    ----------
    price_series :
        DatetimeIndex price series (any resolution, UTC-aware).
    window_min :
        Lookback window in minutes.

    Returns
    -------
    pd.Series aligned to ``price_series.index``.
    """
    log_ret = np.log(price_series / price_series.shift(1)).dropna()
    if len(log_ret) < 2:
        return pd.Series(index=price_series.index, dtype=float)

    # Detect median bar frequency in seconds
    dt_ns          = np.diff(log_ret.index.asi8)  # nanoseconds
    median_freq_s  = float(np.median(dt_ns) / 1e9) if len(dt_ns) > 0 else 60.0
    median_freq_s  = max(median_freq_s, 1.0)

    bars_per_window = max(2, int(window_min * 60 / median_freq_s))
    bars_per_year   = max(1, int(365 * 24 * 3600 / median_freq_s))

    rv = (
        log_ret
        .rolling(bars_per_window, min_periods=2)
        .std()
        .mul(math.sqrt(bars_per_year))  # annualize
    )
    return rv.reindex(price_series.index)


# ── BaselineSample container ──────────────────────────────────────────────────


@dataclass
class BaselineSample:
    """One matched random sample for a given reference event."""

    ref_event_id: str
    sample_idx:   int
    sample_ts:    datetime
    labeled:      LabeledEvent


# ── BaselineSampler ───────────────────────────────────────────────────────────


class BaselineSampler:
    """
    Draws matched random baseline samples from pre-loaded price series.

    Pre-computes rolling RV once per symbol on construction, then reuses
    it across all ``sample()`` calls for that symbol.

    Parameters
    ----------
    price_map :
        Dict symbol → DatetimeIndex price Series (UTC-aware, sorted asc).
    sigma_map :
        Dict symbol → realized vol fraction used for triple-barrier sizing.
    rv_window_min :
        Lookback window (minutes) for computing local rolling RV.
    seed :
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        price_map:    Dict[str, pd.Series],
        sigma_map:    Dict[str, float],
        rv_window_min: int = _RV_WIN_MIN,
        seed:          Optional[int] = None,
    ) -> None:
        self._price_map = price_map
        self._sigma_map = sigma_map
        self._rng       = random.Random(seed)

        # Pre-compute rolling RV for each symbol
        self._rv_map: Dict[str, pd.Series] = {
            sym: _rolling_rv(series, rv_window_min)
            for sym, series in price_map.items()
        }

    # ── Public API ─────────────────────────────────────────────────────────────

    def sample(
        self,
        event:                Event,
        all_event_timestamps: Sequence[datetime],
        n_samples:            int = 10,
    ) -> List[BaselineSample]:
        """
        Draw up to ``n_samples`` matched baseline observations for one Event.

        Matching criteria (applied in order):
          1. Same symbol
          2. Timestamp is strictly before event.timestamp (no look-ahead)
          3. Same day of week
          4. Same hour of day ± 2 hours
          5. Local rolling RV within 20% of the event's RV
          6. NOT within 60 minutes of any real event timestamp

        Returns fewer than ``n_samples`` if the history is too short or
        too few matching candidates exist.
        """
        sym    = event.symbol
        series = self._price_map.get(sym)
        if series is None or series.empty:
            return []

        sigma   = self._sigma_map.get(sym, 0.002)
        rv_s    = self._rv_map.get(sym, pd.Series(dtype=float))

        # --- Event context ---
        ev_ts   = event.timestamp
        if ev_ts.tzinfo is None:
            ev_ts = ev_ts.replace(tzinfo=timezone.utc)
        ev_dow  = ev_ts.weekday()
        ev_hour = ev_ts.hour

        # Event RV: prefer the stored feature, else use sigma_map value
        event_rv = event.features.get("rv_pct")
        event_rv = (float(event_rv) / 100.0) if event_rv is not None else sigma

        # --- Build exclusion array (vectorised) ---
        excl_ts_s = self._build_excl_array(all_event_timestamps)

        # --- Candidate index: bars before event, matching day + hour ---
        idx = series.index
        before = idx[idx < ev_ts]
        if before.empty:
            return []

        candidates = self._filter_day_hour(before, ev_dow, ev_hour)
        if candidates.empty:
            return []

        # --- Optionally narrow by RV ---
        candidates = self._filter_rv(
            candidates, rv_s, event_rv, _RV_TOL
        )
        if candidates.empty:
            return []

        # --- Draw samples ---
        candidate_list = candidates.tolist()
        self._rng.shuffle(candidate_list)

        results: List[BaselineSample] = []
        for ts in candidate_list:
            if len(results) >= n_samples:
                break
            if _MAX_ATTEMPTS and len(results) + _MAX_ATTEMPTS < len(candidate_list):
                pass  # (checked implicitly by list length)
            if self._is_excluded(ts, excl_ts_s):
                continue

            dummy   = _make_dummy_event(event, ts.to_pydatetime())
            labeled = label_event(dummy, series, sigma)
            if labeled is None:
                continue

            results.append(BaselineSample(
                ref_event_id=event.event_id,
                sample_idx=len(results),
                sample_ts=ts.to_pydatetime(),
                labeled=labeled,
            ))

        return results

    # ── Internal helpers ───────────────────────────────────────────────────────

    @staticmethod
    def build_excl_array(all_event_timestamps: Sequence[datetime]) -> np.ndarray:
        """Vectorised exclusion set (public alias for external use)."""
        return BaselineSampler._build_excl_array(all_event_timestamps)

    @staticmethod
    def _build_excl_array(all_event_timestamps: Sequence[datetime]) -> np.ndarray:
        ts_list = []
        for dt in all_event_timestamps:
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            ts_list.append(dt.timestamp())
        return np.array(ts_list, dtype=float)

    @staticmethod
    def _is_excluded(
        ts: pd.Timestamp,
        excl_ts_s: np.ndarray,
        window_s: float = _EXCLUDE_MIN * 60,
    ) -> bool:
        if excl_ts_s.size == 0:
            return False
        t_s = ts.timestamp()
        return bool(np.any(np.abs(excl_ts_s - t_s) <= window_s))

    @staticmethod
    def _filter_day_hour(
        idx:      pd.DatetimeIndex,
        ev_dow:   int,
        ev_hour:  int,
        hw:       int = _HOUR_WINDOW,
    ) -> pd.DatetimeIndex:
        """Keep only bars matching day-of-week and hour ± hw."""
        dow_match  = idx.day_of_week == ev_dow
        # Hour distance (wraps around midnight)
        fwd = (idx.hour - ev_hour) % 24
        bwd = (ev_hour - idx.hour) % 24
        hr_match = (fwd <= hw) | (bwd <= hw)
        return idx[dow_match & hr_match]

    @staticmethod
    def _filter_rv(
        candidates: pd.DatetimeIndex,
        rv_s:       pd.Series,
        event_rv:   float,
        tol:        float,
    ) -> pd.DatetimeIndex:
        """Narrow candidates to those whose rolling RV is within tol of event_rv."""
        if rv_s.empty or math.isnan(event_rv) or event_rv <= 0:
            return candidates
        rv_at = rv_s.reindex(candidates)
        rv_lo = event_rv * (1.0 - tol)
        rv_hi = event_rv * (1.0 + tol)
        valid = rv_at[(rv_at >= rv_lo) & (rv_at <= rv_hi)].dropna()
        # Fall back to unfiltered if RV filter leaves nothing (sparse series)
        return valid.index if not valid.empty else candidates


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_dummy_event(template: Event, ts: datetime) -> Event:
    """Create a dummy Event at ``ts`` with the same type/direction as ``template``."""
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return Event(
        event_id=f"baseline_{template.event_id}_{int(ts.timestamp())}",
        timestamp=ts,
        symbol=template.symbol,
        event_type=template.event_type,
        direction=template.direction,
        score=template.score,
        features=template.features,
    )


# ── Convenience wrapper ────────────────────────────────────────────────────────


def create_baseline(
    event:                Event,
    price_map:            Dict[str, pd.Series],
    all_event_timestamps: Sequence[datetime],
    sigma_map:            Dict[str, float],
    n_samples:            int = 10,
    seed:                 Optional[int] = None,
) -> List[LabeledEvent]:
    """
    Convenience wrapper: creates a ``BaselineSampler`` and returns the
    ``LabeledEvent`` list for a single event.

    Parameters
    ----------
    event :
        The real detected Event to match against.
    price_map :
        Dict symbol → price Series.
    all_event_timestamps :
        All real event timestamps (used to build the exclusion zone).
    sigma_map :
        Dict symbol → realized vol fraction for barrier sizing.
    n_samples :
        How many baseline samples to draw.
    seed :
        RNG seed.

    Returns
    -------
    List of up to ``n_samples`` LabeledEvent objects.
    """
    sampler = BaselineSampler(price_map, sigma_map, seed=seed)
    samples = sampler.sample(event, all_event_timestamps, n_samples)
    return [s.labeled for s in samples]
