"""Event Detection and Alerting for the Bybit Pressure Scanner.

Four event types are detected in real-time and logged to SQLite + CSV,
with optional webhook dispatch:

  COMPRESSION_SQUEEZE_SETUP  — vol compression + funding/OI alignment near settlement
  FUNDING_SQUEEZE_SETUP      — strong funding bias + OI + thin book near settlement
  VACUUM_BREAK               — book thins severely + spread spikes + flow confirms
  CASCADE_ACTIVE             — active liquidation cascade following a squeeze setup

Cooldown rules (per symbol):
  • 30 minutes between the same event type
  • 10 minutes between ANY two events

Integration
-----------
Pass EventDetector (and optionally AlertManager) to PressureScanner:

    from src.events.definitions import EventDetector, AlertManager
    detector = EventDetector()
    alerts   = AlertManager(webhook_url="https://hooks.example.com/...")
    scanner  = PressureScanner(symbols_by_tier, rest,
                               event_detector=detector,
                               alert_manager=alerts)

Batch replay
------------
    from src.events.definitions import batch_replay
    records = [...]   # list of feature-snapshot dicts
    events  = batch_replay(records)
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import math
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from statistics import median
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import time as _time_mod

# Lazy import to avoid circular deps — resolved at first AlertManager instantiation
_AnalysisAgent: Any = None

def _get_analysis_agent_class() -> Any:
    global _AnalysisAgent
    if _AnalysisAgent is None:
        try:
            from src.agents.analysis_agent import AnalysisAgent as _AA
            _AnalysisAgent = _AA
        except Exception as exc:
            logger.warning("AnalysisAgent unavailable: %s", exc)
            _AnalysisAgent = False  # sentinel: don't retry
    return _AnalysisAgent if _AnalysisAgent else None

import aiohttp

if TYPE_CHECKING:
    from ..scanner.pressure_scanner import SymbolState

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────

_ROOT       = Path(__file__).resolve().parent.parent.parent
_DB_PATH    = _ROOT / "data" / "events.db"
_ALERTS_DIR = _ROOT / "data" / "alerts"

# ── Cooldown constants ─────────────────────────────────────────────────────────

_COOLDOWN_SAME = timedelta(minutes=30)
_COOLDOWN_ANY  = timedelta(minutes=10)

# ── Thresholds — COMPRESSION_SQUEEZE_SETUP ────────────────────────────────────

_CS_MIN_CS       = 70.0
_CS_MIN_FUND_Z   = 2.0
_CS_MIN_OI_Z     = 1.5
_CS_MIN_THIN     = 0.80
_CS_MIN_SPS      = 65.0

# ── Thresholds — FUNDING_SQUEEZE_SETUP ────────────────────────────────────────

_FS_MIN_FUND_Z   = 2.0
_FS_MIN_OI_Z     = 1.5
_FS_MIN_THIN     = 0.75
_FS_MIN_SPS      = 60.0

# ── Thresholds — VACUUM_BREAK ─────────────────────────────────────────────────

_VB_MIN_THIN         = 0.97    # top 3% (was 0.90 — top 10% is just noise)
_VB_SPREAD_SIGMA     = 3.0     # spread_z > median + 3*MAD (was 1.5 — too permissive)
_VB_MIN_PRICE_ACCEL  = 0.005   # |price_accel| > 0.5 % / h (was 0.2 — too loose)
_VB_MIN_TAKER_TAIL   = 0.10    # taker_proxy < 0.10 (selling) or > 0.90 (was 0.15 — stricter)
_VB_MIN_CVD_RATIO    = 0.60    # |cvd_ratio_24h| > 0.60 (was 0.40 — stronger imbalance)
_VB_MIN_THIN_HISTORY = 60      # require ≥60 thinness data points before firing
_VB_MIN_THIN_ACCEL   = 0.05    # recent thin_pct must be ≥5 pp above older window

# ── Thresholds — CASCADE_ACTIVE ────────────────────────────────────────────────

_CA_MIN_OI_Z_1H     = 2.5      # |oi_z_1h| > 2.5  (OI spike ≈ mass liquidation)
_CA_MIN_PRICE_ACCEL = 0.003    # |price_accel| > 0.3 % / h
_CA_MIN_VAC_BPS     = 1000.0   # vacuum_dist in move direction > 1000 bps (thin book)
_CA_SQUEEZE_WINDOW  = timedelta(minutes=30)

# ── Thresholds — VOLUME_EXPLOSION (realtime 5min vs 1h baseline) ──────────────

_VE_RT_WINDOW       = 300    # 5 min in seconds
_VE_MIN_HISTORY     = 600    # need >= 10 min of 1s data before checking
_VE_VOL_RATIO_T1    = 3.0    # Tier 1: 5min vol >= 3x the 1h per-5min average
_VE_VOL_RATIO_T2    = 3.5    # Tier 2: higher bar — mid-caps have spikier vol
_VE_VOL_RATIO_T3    = 4.0    # Tier 3: micro-caps have naturally spiky vol
_VE_MOVE_PCT_T1     = 0.5    # Tier 1: |5min price move| >= 0.5%
_VE_MOVE_PCT_T2     = 1.5    # Tier 2: |5min price move| >= 1.5%
_VE_MOVE_PCT_T3     = 2.0    # Tier 3: micro-caps move 1% all the time, need 2%
_VE_MIN_TURNOVER    = 2_000_000  # skip coins with < $2M 24h turnover
_VE_COOLDOWN        = timedelta(minutes=10)   # fast re-arm
_VE_RT_EVAL_INTERVAL = 5.0   # seconds between realtime checks per symbol

# ── Thresholds — OI_SURGE ────────────────────────────────────────────────────

_OIS_PCT_1H     = 10.0                  # OI +10% in 1 hour
_OIS_PCT_4H     = 20.0                  # OI +20% in 4 hours
_OIS_MIN_OI_USD = 1_000_000             # minimum $1M OI in USD
_OIS_COOLDOWN   = timedelta(minutes=60) # 1h cooldown per symbol


# ── Helpers ────────────────────────────────────────────────────────────────────


def _mad(values: List[float]) -> float:
    """Median Absolute Deviation."""
    if len(values) < 2:
        return 0.0
    m = median(values)
    return median([abs(v - m) for v in values])


def _direction_from_funding(fund_z: float) -> str:
    """Map funding z-score sign to expected price-move direction.

    Positive funding → longs pay → crowded longs → bearish unwind → SHORT.
    Negative funding → shorts pay → crowded shorts → bullish squeeze → LONG.
    """
    if math.isnan(fund_z) or fund_z == 0:
        return "NEUTRAL"
    return "SHORT" if fund_z > 0 else "LONG"


# ── Event types ────────────────────────────────────────────────────────────────


class EventType(str, Enum):
    COMPRESSION_SQUEEZE_SETUP = "COMPRESSION_SQUEEZE_SETUP"
    FUNDING_SQUEEZE_SETUP     = "FUNDING_SQUEEZE_SETUP"
    VACUUM_BREAK              = "VACUUM_BREAK"
    CASCADE_ACTIVE            = "CASCADE_ACTIVE"
    VOLUME_EXPLOSION          = "VOLUME_EXPLOSION"
    OI_SURGE                  = "OI_SURGE"


# ── Event dataclass ────────────────────────────────────────────────────────────


@dataclass
class Event:
    event_id:   str
    timestamp:  datetime
    symbol:     str
    event_type: EventType
    direction:  str            # "LONG" | "SHORT" | "NEUTRAL"
    score:      float          # 0–100 severity / confidence
    features:   Dict[str, Any]  # full feature snapshot at detection time

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id":   self.event_id,
            "timestamp":  self.timestamp.isoformat(),
            "symbol":     self.symbol,
            "event_type": self.event_type.value,
            "direction":  self.direction,
            "score":      self.score,
            "features":   self.features,
        }


# ── Event Detector ─────────────────────────────────────────────────────────────


class EventDetector:
    """
    Stateful, per-symbol event detector.

    Call ``evaluate(symbol, state)`` after every composite-score update.
    Returns a (usually empty) list of newly fired Events.

    Not thread-safe — intended for use inside a single asyncio event loop.
    """

    _WARMUP_SECONDS = 600  # suppress events for 10 min after startup

    def __init__(self) -> None:
        # symbol -> EventType -> last-fired datetime
        self._last_same:   Dict[str, Dict[EventType, datetime]] = {}
        # symbol -> last any-event datetime
        self._last_any:    Dict[str, datetime] = {}
        # symbol -> last squeeze-setup datetime (prerequisite for CASCADE_ACTIVE)
        self._last_squeeze: Dict[str, datetime] = {}
        self._warmup_until: float = __import__("time").monotonic() + self._WARMUP_SECONDS
        # per-symbol evaluation throttle: skip if called more frequently than _eval_interval
        self._last_eval_ts: Dict[str, float] = {}
        self._eval_interval: float = 60.0  # seconds
        # separate throttle for realtime explosion checks (5s)
        self._last_rt_eval_ts: Dict[str, float] = {}

    # ── Cooldown helpers ───────────────────────────────────────────────────────

    def _can_fire(
        self,
        symbol: str,
        etype: EventType,
        now: datetime,
        *,
        same_cooldown: Optional[timedelta] = None,
    ) -> bool:
        if symbol in self._last_any:
            if now - self._last_any[symbol] < _COOLDOWN_ANY:
                return False
        cd = same_cooldown if same_cooldown is not None else _COOLDOWN_SAME
        sym_map = self._last_same.get(symbol, {})
        if etype in sym_map:
            if now - sym_map[etype] < cd:
                return False
        return True

    def _record(self, symbol: str, etype: EventType, now: datetime) -> None:
        self._last_any[symbol] = now
        self._last_same.setdefault(symbol, {})[etype] = now
        if etype in (EventType.COMPRESSION_SQUEEZE_SETUP, EventType.FUNDING_SQUEEZE_SETUP):
            self._last_squeeze[symbol] = now

    def _active_squeeze_within_30m(self, symbol: str, now: datetime) -> bool:
        ts = self._last_squeeze.get(symbol)
        return ts is not None and (now - ts) < _CA_SQUEEZE_WINDOW

    # ── Main entry point ───────────────────────────────────────────────────────

    def evaluate(self, symbol: str, state: "SymbolState") -> List[Event]:
        """
        Evaluate all four event conditions against the current SymbolState.

        Called at the end of ``PressureScanner._recompute_composite_scores``,
        after rolling histories have been updated.

        Returns a list of newly fired Events (usually empty).
        """
        import time as _time
        now_m = _time.monotonic()
        if now_m < self._warmup_until:
            return []
        if now_m - self._last_eval_ts.get(symbol, -self._eval_interval) < self._eval_interval:
            return []
        self._last_eval_ts[symbol] = now_m

        now = datetime.now(timezone.utc)
        events: List[Event] = []

        # (event_type, checker, same_type_cooldown override or None for default 30min)
        # Note: VOLUME_EXPLOSION is handled separately via evaluate_realtime()
        # which runs from the collector's 1s snapshot loop with its own 5s throttle.
        checkers = [
            (EventType.COMPRESSION_SQUEEZE_SETUP, self._check_compression_squeeze, None),
            (EventType.FUNDING_SQUEEZE_SETUP,     self._check_funding_squeeze,     None),
            # VACUUM_BREAK disabled — 0% hit rate on 63 events (pure noise)
            (EventType.CASCADE_ACTIVE,            self._check_cascade_active,       None),
            (EventType.OI_SURGE,                  self._check_oi_surge,             _OIS_COOLDOWN),
        ]
        for etype, checker, cooldown in checkers:
            if not self._can_fire(symbol, etype, now, same_cooldown=cooldown):
                continue
            event = checker(symbol, state, now)
            if event is not None:
                self._record(symbol, etype, now)
                events.append(event)

        return events

    # ── Realtime evaluation (called from collector 1s loop) ────────────────────

    def evaluate_realtime(self, symbol: str, state: "SymbolState") -> List[Event]:
        """
        Fast-path evaluation for realtime VOLUME_EXPLOSION only.

        Called every second from the collector's snapshot loop.
        Uses a separate 5s throttle so it doesn't compete with the
        main 60s evaluate() throttle.

        Requires state.rt_vol_1s and state.rt_mid_1s to be populated.
        """
        import time as _time
        now_m = _time.monotonic()
        if now_m < self._warmup_until:
            return []
        if now_m - self._last_rt_eval_ts.get(symbol, -_VE_RT_EVAL_INTERVAL) < _VE_RT_EVAL_INTERVAL:
            return []
        self._last_rt_eval_ts[symbol] = now_m

        now = datetime.now(timezone.utc)
        if not self._can_fire(symbol, EventType.VOLUME_EXPLOSION, now,
                              same_cooldown=_VE_COOLDOWN):
            return []

        event = self._check_volume_explosion(symbol, state, now)
        if event is not None:
            self._record(symbol, EventType.VOLUME_EXPLOSION, now)
            return [event]
        return []

    # ── Feature snapshot helper ────────────────────────────────────────────────

    @staticmethod
    def _snapshot(state: "SymbolState") -> Dict[str, Any]:
        """Collect a flat feature-snapshot dict for logging."""
        ff  = state.funding_feats or {}
        oi  = state.oi_feats      or {}
        vol = state.vol_feats     or {}
        fl  = state.flow_feats    or {}
        ob  = state.ob_feats      or {}
        return {
            "compression":           state.compression,
            "sps":                   state.sps,
            "lfi":                   state.lfi,
            "rank":                  state.rank,
            "funding_z":             ff.get("funding_z"),
            "funding_current":       ff.get("current"),
            "cum_24h":               ff.get("cum_24h"),
            "settlement_phase":      ff.get("settlement_phase"),
            "minutes_to_settlement": ff.get("minutes_to_settlement"),
            "oi_z_24h":              oi.get("oi_z_24h"),
            "oi_z_1h":               oi.get("oi_z_1h"),
            "oi_delta_pct_1h":       oi.get("oi_delta_pct_1h"),
            "rv_pct":                vol.get("rv_pct"),
            "bb_width_pct":          vol.get("bb_width_pct"),
            "range_hours":           vol.get("range_hours"),
            "cvd_ratio_24h":         fl.get("cvd_ratio_24h"),
            "taker_proxy":           fl.get("taker_proxy"),
            "price_accel":           fl.get("price_accel"),
            "thin_pct":              ob.get("thin_pct"),
            "spread_bps":            ob.get("spread_bps"),
            "vacuum_dist_bid":       ob.get("vacuum_dist_bid"),
            "vacuum_dist_ask":       ob.get("vacuum_dist_ask"),
            "depth_bid_usdt":        ob.get("depth_bid_usdt"),
            "depth_ask_usdt":        ob.get("depth_ask_usdt"),
            "convexity":             ob.get("convexity"),
            "tier":                  state.tier,
        }

    # ── Condition checkers ─────────────────────────────────────────────────────

    def _check_compression_squeeze(
        self, symbol: str, state: "SymbolState", now: datetime
    ) -> Optional[Event]:
        """
        COMPRESSION_SQUEEZE_SETUP fires when volatility is compressed,
        funding is extreme, OI is elevated, the book is thin, settlement
        is approaching, and settlement-pressure-score is already high.

        Conditions:
            compression_score  > 70
            |funding_z|        > 2.0
            oi_z_24h           > 1.5   (proxy for oi_z_7d)
            thin_pct           > 0.80
            settlement_phase  in {APPROACH, IMMINENT}
            |sps|              > 65
        """
        ff    = state.funding_feats or {}
        oi    = state.oi_feats      or {}
        ob    = state.ob_feats      or {}

        cs     = state.compression
        sps    = state.sps
        fund_z = ff.get("funding_z", math.nan)
        oi_z   = oi.get("oi_z_24h",  math.nan)
        thin   = ob.get("thin_pct",  math.nan)
        phase  = ff.get("settlement_phase", "FAR")

        has_nan = any(math.isnan(x) for x in (cs, sps, fund_z, oi_z, thin))

        logger.debug(
            "%s CS check: cs=%.1f (need>%.0f), fund_z=%.2f (need |z|>%.1f), "
            "oi_z=%.2f (need>%.1f), thin=%.3f (need>%.2f), "
            "phase=%s (need APPROACH/IMMINENT), sps=%.1f (need |sps|>%.0f), has_nan=%s",
            symbol, cs, _CS_MIN_CS, fund_z, _CS_MIN_FUND_Z,
            oi_z, _CS_MIN_OI_Z, thin, _CS_MIN_THIN,
            phase, sps, _CS_MIN_SPS, has_nan,
        )

        if has_nan:
            return None

        if not (
            cs     > _CS_MIN_CS
            and abs(fund_z) > _CS_MIN_FUND_Z
            and oi_z   > _CS_MIN_OI_Z
            and thin   > _CS_MIN_THIN
            and phase in ("APPROACH", "IMMINENT")
            and abs(sps) > _CS_MIN_SPS
        ):
            return None

        return Event(
            event_id=uuid.uuid4().hex,
            timestamp=now,
            symbol=symbol,
            event_type=EventType.COMPRESSION_SQUEEZE_SETUP,
            direction=_direction_from_funding(fund_z),
            score=float(state.rank if not math.isnan(state.rank) else cs),
            features=self._snapshot(state),
        )

    def _check_funding_squeeze(
        self, symbol: str, state: "SymbolState", now: datetime
    ) -> Optional[Event]:
        """
        FUNDING_SQUEEZE_SETUP fires when funding bias is extreme, OI is
        elevated, and book is thin.

        Gates (hard requirements):
            (|funding_z| > 2.0  OR  |funding_current| > 0.001)
            oi_z_24h  > 1.5   (skipped if NaN)
            thin_pct  > 0.75  (skipped if NaN)

        Bonuses (add to score, do NOT gate):
            settlement_phase IMMINENT +20, APPROACH +10
            |funding_z| > 3.0  → +10
            thin_pct > 0.90    → +10
        """
        ff    = state.funding_feats or {}
        oi    = state.oi_feats      or {}
        ob    = state.ob_feats      or {}

        fund_z   = ff.get("funding_z",        math.nan)
        fund_cur = ff.get("funding_current",   math.nan)
        oi_z     = oi.get("oi_z_24h",          math.nan)
        thin     = ob.get("thin_pct",          math.nan)
        phase    = ff.get("settlement_phase",  "FAR")
        sps      = state.sps

        # Funding gate: z-score OR raw rate
        fund_z_pass   = not math.isnan(fund_z) and abs(fund_z) > _FS_MIN_FUND_Z
        fund_raw_pass = not math.isnan(fund_cur) and abs(fund_cur) > 0.001

        logger.debug(
            "%s FS check: fund_z=%.2f fund_cur=%.6f (z_pass=%s raw_pass=%s), "
            "oi_z=%.2f (need>%.1f), phase=%s, "
            "thin=%.3f (need>%.2f), sps=%.1f",
            symbol,
            fund_z if not math.isnan(fund_z) else 0.0,
            fund_cur if not math.isnan(fund_cur) else 0.0,
            fund_z_pass, fund_raw_pass,
            oi_z if not math.isnan(oi_z) else 0.0, _FS_MIN_OI_Z,
            phase,
            thin if not math.isnan(thin) else 0.0, _FS_MIN_THIN,
            sps if not math.isnan(sps) else 0.0,
        )

        if not (fund_z_pass or fund_raw_pass):
            return None

        # Require at least one of oi_z or thin_pct to have real data
        if math.isnan(oi_z) and math.isnan(thin):
            return None

        # OI gate (skip if data missing)
        if not math.isnan(oi_z) and oi_z < _FS_MIN_OI_Z:
            return None

        # Thin gate (skip if data missing)
        if not math.isnan(thin) and thin < _FS_MIN_THIN:
            return None

        # ── Score: base + bonuses ───────────────────────────────────────
        score = 40.0  # base for passing all gates

        # Phase bonus (not a gate)
        if phase == "IMMINENT":
            score += 20.0
        elif phase == "APPROACH":
            score += 10.0

        # Extreme funding bonus
        if not math.isnan(fund_z) and abs(fund_z) > 3.0:
            score += 10.0

        # Very thin book bonus
        if not math.isnan(thin) and thin > 0.90:
            score += 10.0

        # SPS contribution (bonus, not gate)
        if not math.isnan(sps) and abs(sps) > 50:
            score += min(20.0, (abs(sps) - 50) * 0.4)

        score = min(100.0, score)

        if score < 60.0:
            return None

        return Event(
            event_id=uuid.uuid4().hex,
            timestamp=now,
            symbol=symbol,
            event_type=EventType.FUNDING_SQUEEZE_SETUP,
            direction=_direction_from_funding(fund_z),
            score=round(score, 1),
            features=self._snapshot(state),
        )

    def _check_vacuum_break(
        self, symbol: str, state: "SymbolState", now: datetime
    ) -> Optional[Event]:
        """
        VACUUM_BREAK fires when the book is extremely thin, the bid/ask
        spread has just spiked above its recent distribution, and flow
        confirms directional pressure.

        Conditions:
            thin_pct              > 0.97   (top 3% — was 0.90)
            thinness_history len  ≥ 60     (suppress startup noise)
            recent thin_pct       ≥ 5 pp above older window (book thinning, not just thin)
            spread_z              > median(history) + 3.0 * MAD(history)  (was 1.5)
            cur_sz                ≥ 2× baseline median (absolute floor)
            price_accel proxy     |price_accel| > 0.005   (0.5 % / h — was 0.2)
            flow proxy            taker_proxy < 0.10 or > 0.90 (was 0.15)
                                  OR |cvd_ratio_24h| > 0.60 (was 0.40)
            Both price_break AND flow_confirm required (was OR)
        Note: spread_z_history already contains the current tick's value
        (it is appended before _recompute_composite_scores returns).
        """
        ob = state.ob_feats  or {}
        fl = state.flow_feats or {}

        thin        = ob.get("thin_pct",      math.nan)
        price_accel = fl.get("price_accel",   math.nan)
        taker_p     = fl.get("taker_proxy",   math.nan)
        cvd_ratio   = fl.get("cvd_ratio_24h", math.nan)

        if math.isnan(thin) or not (thin > _VB_MIN_THIN):
            return None

        # Require sufficient thinness history to avoid false positives at startup
        thin_hist = getattr(state, "thinness_history", None)
        if thin_hist is None or len(thin_hist) < _VB_MIN_THIN_HISTORY:
            return None

        # Post-reconnect cooldown: WS snapshot after reconnect looks thin until
        # the book rebuilds — suppress VACUUM_BREAK for 120 s after any reconnect
        ws_reconnect_ts = getattr(state, "last_ws_reconnect_ts", None)
        if ws_reconnect_ts is not None and (now - ws_reconnect_ts).total_seconds() < 120:
            return None

        # Require that the book *thinned out recently* — not just "is thin"
        thin_list = list(thin_hist)
        if len(thin_list) >= 20:
            recent = thin_list[-5:]
            older  = thin_list[-20:-10]
            if older:
                avg_recent = sum(recent) / len(recent)
                avg_older  = sum(older) / len(older)
                if avg_recent - avg_older < _VB_MIN_THIN_ACCEL * avg_older:
                    return None

        # Spread-z spike: compare latest spread_z against history baseline
        sz_hist = list(state.spread_z_history)
        if len(sz_hist) < 20:
            return None

        cur_sz   = sz_hist[-1]
        baseline = sz_hist[:-1]
        bl_med   = median(baseline)
        bl_mad   = _mad(baseline)
        spike_threshold = bl_med + _VB_SPREAD_SIGMA * max(bl_mad, 1e-10)

        if not (cur_sz > spike_threshold):
            return None

        # Require spread is absolutely high, not just relatively high
        if bl_med > 0 and cur_sz < bl_med * 2.0:
            return None

        # Price-break proxy: large 1-hour price acceleration
        price_break = (
            not math.isnan(price_accel)
            and abs(price_accel) > _VB_MIN_PRICE_ACCEL
        )

        # Flow proxy: aggressive one-sided taker activity or CVD imbalance
        flow_confirm = (
            (not math.isnan(taker_p)
             and (taker_p < _VB_MIN_TAKER_TAIL or taker_p > 1.0 - _VB_MIN_TAKER_TAIL))
            or (not math.isnan(cvd_ratio) and abs(cvd_ratio) > _VB_MIN_CVD_RATIO)
        )

        if not (price_break and flow_confirm):
            return None

        # ── Quality score (0–100) ─────────────────────────────────────────────
        # Rewards events where every signal is at its extreme, not just over threshold.

        quality = 0.0

        # Thinness extremity (0–25): 0.97→0, 1.00→15, ≥1.02→25
        quality += min(25.0, (thin - 0.97) * 500.0)

        # Spread-spike magnitude (0–25): each sigma above baseline = 5 pts
        if bl_mad > 0:
            sigma_above = (cur_sz - bl_med) / bl_mad
            quality += min(25.0, sigma_above * 5.0)

        # Price-move strength (0–25): 0.5%/h→12.5, 1%/h→25
        quality += min(25.0, abs(price_accel) * 2500.0)

        # Flow imbalance strength (0–25): |cvd_ratio| 0.6→15, 1.0→25
        if not math.isnan(cvd_ratio):
            quality += min(25.0, abs(cvd_ratio) * 25.0)

        if quality < 65.0:
            return None

        # ── Direction: whichever OB side is thinner drives the vacuum ─────────
        vac_bid = ob.get("vacuum_dist_bid", 0.0) or 0.0
        vac_ask = ob.get("vacuum_dist_ask", 0.0) or 0.0
        direction = "LONG" if vac_ask > vac_bid else "SHORT"

        snapshot = self._snapshot(state)
        snapshot["quality_score"] = round(quality, 1)
        return Event(
            event_id=uuid.uuid4().hex,
            timestamp=now,
            symbol=symbol,
            event_type=EventType.VACUUM_BREAK,
            direction=direction,
            score=round(quality, 1),
            features=snapshot,
        )

    def _check_cascade_active(
        self, symbol: str, state: "SymbolState", now: datetime
    ) -> Optional[Event]:
        """
        CASCADE_ACTIVE fires when a squeeze setup has already been detected
        within the last 30 minutes, OI is spiking (indicating mass liquidation),
        price is accelerating strongly, and the book on that side is near-empty.

        Conditions:
            active squeeze event within 30 min          (prerequisite)
            |oi_z_1h|     > 2.5   (rapid OI drop ≈ liquidations proxy)
            |price_accel| > 0.003  (0.3 % / h directional move proxy)
            vacuum_dist in move direction > 1000 bps   (depth resistance proxy)
        """
        if not self._active_squeeze_within_30m(symbol, now):
            return None

        oi = state.oi_feats   or {}
        fl = state.flow_feats or {}
        ob = state.ob_feats   or {}

        oi_z_1h     = oi.get("oi_z_1h",     math.nan)
        price_accel = fl.get("price_accel",  math.nan)

        if any(math.isnan(x) for x in (oi_z_1h, price_accel)):
            return None

        if not (abs(oi_z_1h) > _CA_MIN_OI_Z_1H):
            return None

        if not (abs(price_accel) > _CA_MIN_PRICE_ACCEL):
            return None

        # Direction and vacuum check
        if price_accel > 0:
            vac = ob.get("vacuum_dist_ask", math.nan)  # price up → asks must be thin
            direction = "LONG"
        else:
            vac = ob.get("vacuum_dist_bid", math.nan)  # price down → bids must be thin
            direction = "SHORT"

        if math.isnan(vac) or not (vac > _CA_MIN_VAC_BPS):
            return None

        rank = state.rank
        return Event(
            event_id=uuid.uuid4().hex,
            timestamp=now,
            symbol=symbol,
            event_type=EventType.CASCADE_ACTIVE,
            direction=direction,
            score=float(rank if not math.isnan(rank) else 50.0),
            features=self._snapshot(state),
        )

    def _check_volume_explosion(
        self, symbol: str, state: "SymbolState", now: datetime
    ) -> Optional[Event]:
        """
        VOLUME_EXPLOSION fires when a 5-minute window shows an extreme
        spike in volume and price move vs. the trailing 1h baseline.

        Data source: state.rt_vol_1s and state.rt_mid_1s (1-second
        snapshots populated by DataCollector._snapshot_loop).

        Conditions (ALL three required):
            5min volume   >= 3x the 1h per-5min average
            |5min move|   >= 0.5% (T1) or 1.0% (T2/T3)
            OI increasing (new positions opening, not just closes)

        Quality bonuses:
            Vacuum EMPTY in move direction
            Funding contra (short squeeze in progress)
        """
        rt_vol = getattr(state, "rt_vol_1s", None)
        rt_mid = getattr(state, "rt_mid_1s", None)

        if rt_vol is None or rt_mid is None:
            return None
        if len(rt_vol) < _VE_MIN_HISTORY or len(rt_mid) < _VE_MIN_HISTORY:
            return None

        vol_list = list(rt_vol)
        mid_list = list(rt_mid)

        # === 1. VOLUME: 5min vs 1h average ===
        vol_5m = sum(vol_list[-_VE_RT_WINDOW:])
        vol_total = sum(vol_list)
        n_windows = len(vol_list) / _VE_RT_WINDOW
        avg_5m = vol_total / n_windows if n_windows > 0 else 0.0

        if avg_5m <= 0:
            return None

        vol_ratio = vol_5m / avg_5m

        # Tier-specific volume ratio thresholds
        if state.tier == 1:
            vol_threshold = _VE_VOL_RATIO_T1
        elif state.tier == 2:
            vol_threshold = _VE_VOL_RATIO_T2
        else:
            vol_threshold = _VE_VOL_RATIO_T3

        if vol_ratio < vol_threshold:
            return None

        # === 1b. MINIMUM TURNOVER: skip micro-caps with < $2M 24h turnover ===
        turnover_24h = getattr(state, "turnover_24h", None)
        if turnover_24h is not None and turnover_24h < _VE_MIN_TURNOVER:
            return None

        # === 2. PRICE MOVE: 5min ===
        price_now = mid_list[-1]
        price_5m_ago = mid_list[-min(_VE_RT_WINDOW, len(mid_list))]

        if price_5m_ago <= 0:
            return None

        move_pct = (price_now - price_5m_ago) / price_5m_ago * 100.0

        if state.tier == 1:
            move_threshold = _VE_MOVE_PCT_T1
        elif state.tier == 2:
            move_threshold = _VE_MOVE_PCT_T2
        else:
            move_threshold = _VE_MOVE_PCT_T3
        if abs(move_pct) < move_threshold:
            return None

        # === 3. OI INCREASING (new positions, not just closures) ===
        oi = state.oi_feats or {}
        oi_delta_1h = oi.get("oi_delta_pct_1h", math.nan)
        # Require OI flat or increasing — reject negative (= positions closing)
        if not math.isnan(oi_delta_1h) and oi_delta_1h < -0.5:
            return None

        # === Direction ===
        direction = "LONG" if move_pct > 0 else "SHORT"

        # === Quality score (0-100) ===
        quality = 0.0

        # Volume excess (0-30): each 1x above threshold = 10 pts
        quality += min(30.0, (vol_ratio - vol_threshold) * 10.0)

        # Price move excess (0-30): each 0.5% above threshold = 10 pts
        quality += min(30.0, (abs(move_pct) - move_threshold) * 20.0)

        # OI contribution (0-15): growing OI = stronger signal
        if not math.isnan(oi_delta_1h) and oi_delta_1h > 0:
            quality += min(15.0, oi_delta_1h * 5.0)

        # === Quality bonuses ===
        ob = state.ob_feats or {}
        ff = state.funding_feats or {}

        # Vacuum EMPTY in move direction (+10)
        if direction == "LONG":
            vac = ob.get("vacuum_dist_ask", 0.0) or 0.0
        else:
            vac = ob.get("vacuum_dist_bid", 0.0) or 0.0
        vac_empty = vac >= 9999
        if vac_empty:
            quality += 10.0

        # Funding contra — short squeeze in progress (+15)
        funding_current = ff.get("funding_current", math.nan)
        funding_contra = False
        if not math.isnan(funding_current):
            if direction == "LONG" and funding_current < -0.0001:
                # Negative funding + price rising = short squeeze
                funding_contra = True
                quality += 15.0
            elif direction == "SHORT" and funding_current > 0.0001:
                # Positive funding + price falling = long squeeze
                funding_contra = True
                quality += 15.0

        quality = min(100.0, quality)

        # Build feature snapshot
        snapshot = self._snapshot(state)
        snapshot.update({
            "vol_ratio_5m":    round(vol_ratio, 2),
            "move_pct_5m":     round(move_pct, 3),
            "oi_delta_pct_1h": round(oi_delta_1h, 3) if not math.isnan(oi_delta_1h) else None,
            "price":           price_now,
            "vac_empty":       vac_empty,
            "funding_contra":  funding_contra,
            "quality_score":   round(quality, 1),
        })

        return Event(
            event_id=uuid.uuid4().hex,
            timestamp=now,
            symbol=symbol,
            event_type=EventType.VOLUME_EXPLOSION,
            direction=direction,
            score=round(quality, 1),
            features=snapshot,
        )

    # ── OI_SURGE ────────────────────────────────────────────────────────────────

    def _check_oi_surge(
        self, symbol: str, state: "SymbolState", now: datetime
    ) -> Optional[Event]:
        """Detect explosive OI increase — massive new positions being built.

        OI surging = future liquidation fuel.  Fires regardless of funding;
        catches moves like NAORIS (+71% OI, neutral funding, +30% price).
        """
        oi = state.oi_feats or {}
        ob = state.ob_feats or {}
        ff = state.funding_feats or {}

        # Current OI in contracts
        oi_current = oi.get("oi_current")
        if oi_current is None or oi_current <= 0:
            return None

        # Mid price for USD conversion
        mid_price = ob.get("mid_price")
        if mid_price is None or mid_price <= 0:
            return None

        oi_usd = oi_current * mid_price
        if oi_usd < _OIS_MIN_OI_USD:
            return None

        # Thick book = OI build is absorbed without fragility → skip
        thin = ob.get("thin_pct", 0.5) or 0.5
        if thin < 0.70:
            return None

        # OI percentage changes (computed by compute_oi_features)
        oi_pct_1h = oi.get("oi_pct_1h", math.nan)
        oi_pct_4h = oi.get("oi_pct_4h", math.nan)

        # Check thresholds — pick strongest timeframe
        surge_detected = False
        surge_tf = ""
        surge_pct = 0.0

        if not math.isnan(oi_pct_1h) and oi_pct_1h >= _OIS_PCT_1H:
            surge_detected = True
            surge_tf = "1h"
            surge_pct = oi_pct_1h

        if not math.isnan(oi_pct_4h) and oi_pct_4h >= _OIS_PCT_4H:
            if not surge_detected or oi_pct_4h > surge_pct:
                surge_detected = True
                surge_tf = "4h"
                surge_pct = oi_pct_4h

        if not surge_detected:
            return None

        # Direction from funding bias
        funding_cur = ff.get("funding_current", 0) or 0
        fund_z = ff.get("funding_z", 0) or 0

        if funding_cur < -0.0003:
            direction = "LONG"    # shorts building → potential squeeze UP
        elif funding_cur > 0.0003:
            direction = "SHORT"   # longs building → potential squeeze DN
        else:
            # Neutral funding — use price direction
            price_accel = (state.flow_feats or {}).get("price_accel", 0) or 0
            if price_accel > 0.001:
                direction = "LONG"
            elif price_accel < -0.001:
                direction = "SHORT"
            else:
                direction = "LONG"  # default

        # Quality score (0-100)
        quality = 0.0

        # Surge magnitude (0-40)
        if surge_tf == "1h":
            quality += min(40.0, surge_pct * 2.0)   # 10% → 20, 20% → 40
        else:
            quality += min(40.0, surge_pct * 1.0)   # 20% → 20, 40% → 40

        # Thin book bonus (0-20)
        thin = ob.get("thin_pct", 0.5) or 0.5
        if thin > 0.90:
            quality += 20.0
        elif thin > 0.80:
            quality += 10.0

        # Funding alignment (0-15)
        if abs(funding_cur) > 0.001:
            quality += 15.0
        elif abs(funding_cur) > 0.0005:
            quality += 8.0

        # OI z-score bonus (0-15) — high z = unusual surge
        oi_z_1h = oi.get("oi_z_1h", 0) or 0
        if abs(oi_z_1h) > 3.0:
            quality += 15.0
        elif abs(oi_z_1h) > 2.0:
            quality += 8.0

        # Vacuum bonus (0-10)
        if direction == "LONG":
            vac = ob.get("vacuum_dist_ask", 0) or 0
        else:
            vac = ob.get("vacuum_dist_bid", 0) or 0
        if vac >= 9999:
            quality += 10.0

        quality = min(100.0, quality)
        if quality < 50:
            return None

        snapshot = self._snapshot(state)
        snapshot.update({
            "oi_surge_pct":   round(surge_pct, 1),
            "surge_timeframe": surge_tf,
            "oi_usd":          round(oi_usd),
            "oi_pct_1h":       round(oi_pct_1h, 2) if not math.isnan(oi_pct_1h) else None,
            "oi_pct_4h":       round(oi_pct_4h, 2) if not math.isnan(oi_pct_4h) else None,
        })

        return Event(
            event_id=uuid.uuid4().hex,
            timestamp=now,
            symbol=symbol,
            event_type=EventType.OI_SURGE,
            direction=direction,
            score=round(quality, 1),
            features=snapshot,
        )


# ── Alert Manager ──────────────────────────────────────────────────────────────


class AlertManager:
    """
    Persists Events to:
      • SQLite  — data/events.db  (queryable history)
      • CSV     — data/alerts/YYYY-MM-DD.csv  (rolling daily log)
      • Webhook — optional async POST for real-time notification

    Usage (async context required for webhook dispatch):

        alert_mgr = AlertManager(webhook_url="https://...")
        await alert_mgr.handle(event)
        await alert_mgr.close()
    """

    _CSV_FIELDS = [
        "event_id", "timestamp", "symbol", "event_type", "direction", "score",
        "compression", "sps", "lfi", "rank",
        "funding_z", "settlement_phase", "minutes_to_settlement",
        "oi_z_24h", "oi_z_1h", "thin_pct", "spread_bps",
        "vacuum_dist_bid", "vacuum_dist_ask", "price_accel", "tier",
    ]

    def __init__(
        self,
        db_path:         Path = _DB_PATH,
        alerts_dir:      Path = _ALERTS_DIR,
        webhook_url:     Optional[str] = None,
        webhook_timeout: float = 5.0,
        global_max:      int   = 15,
        global_window:   float = 600.0,
        telegram_token:  Optional[str] = None,
        telegram_chat_id: Optional[str] = None,
    ) -> None:
        self._db_path        = db_path
        self._alerts_dir     = alerts_dir
        self._webhook_url    = webhook_url
        self._webhook_timeout = webhook_timeout
        self._tg_token       = telegram_token
        self._tg_chat_id     = telegram_chat_id
        self._session: Optional[aiohttp.ClientSession] = None
        # Global rate limit: at most global_max dispatched events per global_window seconds
        self._global_events: List[tuple] = []  # list of (monotonic_ts, Event)
        self._global_max    = global_max
        self._global_window = global_window

        # Digest system: buffer all events, send summary every 30 min
        self._event_history: List[Event] = []  # all events (pruned to 30 min)
        self._digest_task: Optional[asyncio.Task] = None

        # VE dedup: max 1 Telegram VE alert per symbol per 30 min
        self._last_ve_alert: Dict[str, datetime] = {}

        # OI_SURGE rate limit: max 3 individual Telegram alerts per 2h
        self._oi_surge_tg_times: List[float] = []  # monotonic timestamps

        # BTC regime: callable returning (pct_4h, price) or None; set by scanner
        self._btc_price_fn: Optional[Any] = None  # Callable[[], Optional[Tuple[float,float]]]

        # AI Analysis Agent — initialized lazily on first high-score event
        _cls = _get_analysis_agent_class()
        self._analysis_agent: Optional[Any] = _cls() if _cls else None

        db_path.parent.mkdir(parents=True, exist_ok=True)
        alerts_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

        if self._tg_token and self._tg_chat_id:
            logger.info("Telegram alerts enabled (chat_id=%s)", self._tg_chat_id)

    # ── SQLite ─────────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    event_id      TEXT PRIMARY KEY,
                    timestamp     TEXT NOT NULL,
                    symbol        TEXT NOT NULL,
                    event_type    TEXT NOT NULL,
                    direction     TEXT NOT NULL,
                    score         REAL NOT NULL,
                    features_json TEXT NOT NULL
                )
            """)
            con.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_symbol ON events(symbol)"
            )
            con.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_ts ON events(timestamp)"
            )
            con.commit()

    def _log_sqlite(self, event: Event) -> None:
        try:
            with sqlite3.connect(self._db_path) as con:
                con.execute(
                    "INSERT OR IGNORE INTO events VALUES (?,?,?,?,?,?,?)",
                    (
                        event.event_id,
                        event.timestamp.isoformat(),
                        event.symbol,
                        event.event_type.value,
                        event.direction,
                        event.score,
                        json.dumps(event.features, default=str),
                    ),
                )
                con.commit()
        except Exception as exc:
            logger.error("SQLite log failed for %s: %s", event.event_id, exc)

    # ── CSV ────────────────────────────────────────────────────────────────────

    def _log_csv(self, event: Event) -> None:
        date_str = event.timestamp.strftime("%Y-%m-%d")
        csv_path = self._alerts_dir / f"{date_str}.csv"
        write_header = not csv_path.exists()
        feats = event.features
        row = {
            "event_id":               event.event_id,
            "timestamp":              event.timestamp.isoformat(),
            "symbol":                 event.symbol,
            "event_type":             event.event_type.value,
            "direction":              event.direction,
            "score":                  round(event.score, 4),
            "compression":            feats.get("compression"),
            "sps":                    feats.get("sps"),
            "lfi":                    feats.get("lfi"),
            "rank":                   feats.get("rank"),
            "funding_z":              feats.get("funding_z"),
            "settlement_phase":       feats.get("settlement_phase"),
            "minutes_to_settlement":  feats.get("minutes_to_settlement"),
            "oi_z_24h":               feats.get("oi_z_24h"),
            "oi_z_1h":               feats.get("oi_z_1h"),
            "thin_pct":               feats.get("thin_pct"),
            "spread_bps":             feats.get("spread_bps"),
            "vacuum_dist_bid":        feats.get("vacuum_dist_bid"),
            "vacuum_dist_ask":        feats.get("vacuum_dist_ask"),
            "price_accel":            feats.get("price_accel"),
            "tier":                   feats.get("tier"),
        }
        try:
            with open(csv_path, "a", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=self._CSV_FIELDS)
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
        except Exception as exc:
            logger.error("CSV log failed for %s: %s", event.event_id, exc)

    # ── Webhook ────────────────────────────────────────────────────────────────

    async def _dispatch_webhook(self, event: Event) -> None:
        if not self._webhook_url:
            return
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        try:
            timeout = aiohttp.ClientTimeout(total=self._webhook_timeout)
            async with self._session.post(
                self._webhook_url,
                json=event.to_dict(),
                timeout=timeout,
            ) as resp:
                if resp.status >= 400:
                    logger.warning(
                        "Webhook HTTP %d for event %s (%s %s)",
                        resp.status, event.event_id, event.symbol,
                        event.event_type.value,
                    )
        except Exception as exc:
            logger.warning("Webhook dispatch failed: %s", exc)

    # ── Telegram ───────────────────────────────────────────────────────────────

    async def _dispatch_telegram(self, text: str, parse_mode: str = "HTML") -> None:
        if not self._tg_token or not self._tg_chat_id:
            return
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        url = f"https://api.telegram.org/bot{self._tg_token}/sendMessage"
        timeout = aiohttp.ClientTimeout(total=30)
        payload = {
            "chat_id": self._tg_chat_id,
            "text": text,
            "parse_mode": parse_mode,
        }

        for attempt in range(2):
            try:
                async with self._session.post(
                    url, json=payload, timeout=timeout,
                ) as resp:
                    if resp.status < 400:
                        return  # success
                    body = await resp.text()
                    logger.warning(
                        "Telegram HTTP %d (attempt %d): %s",
                        resp.status, attempt + 1, body[:300],
                    )
            except Exception as exc:
                logger.warning(
                    "Telegram dispatch failed (attempt %d): %r",
                    attempt + 1, exc,
                )
            if attempt == 0:
                await asyncio.sleep(3)

    def _format_event_telegram(self, event: Event) -> str:
        """Format an Event as a Telegram message."""
        f = event.features
        dir_arrow = "UP" if event.direction == "LONG" else "DN"

        lines = [f"<b>{event.event_type.value}</b> {dir_arrow}"]
        lines.append(f"<b>{event.symbol}</b> | Score: {event.score:.0f}")

        if event.event_type == EventType.VOLUME_EXPLOSION:
            vol_r = f.get("vol_ratio_5m", "?")
            move = f.get("move_pct_5m", "?")
            oi_d = f.get("oi_delta_pct_1h")
            oi_str = f"+{oi_d:.1f}%" if oi_d else "N/A"
            vac = "EMPTY" if f.get("vac_empty") else "ok"
            fc = " [SQUEEZE]" if f.get("funding_contra") else ""
            lines.append(f"Vol: {vol_r}x | Move: {move:+.1f}% (5m) | OI: {oi_str}")
            lines.append(f"Vac: {vac}{fc}")
        elif event.event_type == EventType.OI_SURGE:
            surge_pct = f.get("oi_surge_pct", 0)
            surge_tf = f.get("surge_timeframe", "?")
            oi_usd = f.get("oi_usd", 0)
            funding = f.get("funding_current", 0) or 0
            thin = f.get("thin_pct", 0) or 0
            oi_usd_str = f"${oi_usd / 1e6:.1f}M" if oi_usd else "?"
            fund_str = f"{funding * 100:.3f}%" if funding else "neutral"
            lines.append(f"OI: +{surge_pct:.1f}% ({surge_tf}) | {oi_usd_str}")
            lines.append(f"Funding: {fund_str} | Thin: {thin:.2f}")
        else:
            rank = f.get("rank")
            thin = f.get("thin_pct")
            phase = f.get("settlement_phase", "?")
            if rank is not None:
                lines.append(f"Rank: {rank:.1f} | Phase: {phase}")
            if thin is not None:
                lines.append(f"Thin: {thin:.2f}")

        return "\n".join(lines)

    async def send_settlement_scan(self, text: str) -> None:
        """Send a pre-formatted settlement scan message to Telegram."""
        await self._dispatch_telegram(text)

    # ── Digest summary ──────────────────────────────────────────────────────

    def _prune_event_history(self) -> None:
        """Remove events older than 2 hours (matches digest interval)."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=2)
        self._event_history = [
            ev for ev in self._event_history if ev.timestamp >= cutoff
        ]

    def _format_digest(self) -> Optional[str]:
        """Build a Telegram digest summary of active coins in the last 30 min.

        Returns None if no coins are active. Shows max 10 coins.
        """
        self._prune_event_history()
        if not self._event_history:
            return None

        # BTC regime header
        btc_str = ""
        if self._btc_price_fn is not None:
            try:
                result = self._btc_price_fn()
                if result is not None:
                    pct_4h, _ = result
                    if pct_4h > 1.0:
                        emoji = "\U0001f7e2"   # green circle
                    elif pct_4h < -1.0:
                        emoji = "\U0001f534"   # red circle
                    else:
                        emoji = "\u27a1\ufe0f"  # right arrow
                    btc_str = f" | BTC: {pct_4h:+.1f}% (4h) {emoji}"
            except Exception:
                pass

        # Group by symbol
        by_sym: Dict[str, List[Event]] = {}
        for ev in self._event_history:
            by_sym.setdefault(ev.symbol, []).append(ev)

        # Build rows: symbol, direction, max_score, counts per type, unique types
        _TYPE_ABBR = {
            EventType.FUNDING_SQUEEZE_SETUP:     "FS",
            EventType.COMPRESSION_SQUEEZE_SETUP: "CS",
            EventType.VACUUM_BREAK:              "VB",
            EventType.VOLUME_EXPLOSION:          "VE",
            EventType.CASCADE_ACTIVE:            "CA",
            EventType.OI_SURGE:                  "OI",
        }
        rows = []
        for sym, events in by_sym.items():
            # Filter events for digest inclusion:
            #  FS: score >= 75 AND phase APPROACH/IMMINENT only
            #  VB: score >= 80
            #  Others: score >= 50
            digest_events = []
            for ev in events:
                if ev.event_type == EventType.FUNDING_SQUEEZE_SETUP:
                    phase = (ev.features or {}).get("settlement_phase", "FAR")
                    if ev.score < 75 or phase not in ("APPROACH", "IMMINENT"):
                        continue
                elif ev.event_type == EventType.VACUUM_BREAK:
                    if ev.score < 80:
                        continue
                else:
                    if ev.score < 50:
                        continue
                digest_events.append(ev)

            if not digest_events:
                continue

            # Only show coins with 2+ event types, OR strong CA/VE alone
            unique_evt_types = {e.event_type for e in digest_events}
            if len(unique_evt_types) < 2:
                # Allow strong CA or VE alone (high hit rates)
                has_strong = any(
                    e.event_type in (EventType.CASCADE_ACTIVE, EventType.VOLUME_EXPLOSION)
                    and e.score >= 60
                    for e in digest_events
                )
                if not has_strong:
                    continue

            max_score = max(ev.score for ev in digest_events)
            # Direction from highest-scored event
            best_ev = max(digest_events, key=lambda e: e.score)
            direction = best_ev.direction

            # Count per type
            type_counts: Dict[str, int] = {}
            unique_types = set()
            for ev in digest_events:
                abbr = _TYPE_ABBR.get(ev.event_type, ev.event_type.value[:2])
                type_counts[abbr] = type_counts.get(abbr, 0) + 1
                unique_types.add(abbr)

            # Status emoji by unique type count
            n_unique = len(unique_types)
            if n_unique >= 4:
                status = "\xf0\x9f\x94\xb4".encode().decode()  # red circle
            elif n_unique >= 3:
                status = "\xf0\x9f\x9f\xa0".encode().decode()  # orange circle
            elif n_unique >= 2:
                status = "\xf0\x9f\x9f\xa1".encode().decode()  # yellow circle
            else:
                status = "\xf0\x9f\x9f\xa2".encode().decode()  # green circle

            counts_str = " ".join(f"{k}:{v}" for k, v in sorted(type_counts.items()))
            dir_arrow = "UP" if direction == "LONG" else "DN"

            rows.append((n_unique, max_score, sym, dir_arrow, max_score, counts_str, status))

        # Sort: unique types desc, then score desc; cap at 10
        rows.sort(key=lambda r: (-r[0], -r[1]))
        total = len(rows)
        if total == 0:
            return None  # Don't send empty digests
        rows = rows[:10]

        lines = [f"<b>DIGEST (2h)</b>{btc_str}", ""]
        for _, _, sym, dir_arrow, score, counts, status in rows:
            lines.append(f"{status} <b>{sym}</b> {dir_arrow} | {score:.0f} | {counts}")

        suffix = f" (top 10 of {total})" if total > 10 else ""
        lines.append(f"\n{total} active coin(s){suffix}")
        return "\n".join(lines)

    async def _digest_loop(self) -> None:
        """Send a digest summary every 2 hours."""
        while True:
            await asyncio.sleep(7200)  # 2 hours
            try:
                msg = self._format_digest()
                if msg:
                    await self._dispatch_telegram(msg)
                    logger.info("Digest sent (%d active coins)", msg.count("\n") - 2)
            except Exception as exc:
                logger.error("Digest loop error: %s", exc)

    def start_digest_loop(self) -> None:
        """Start the background digest loop. Call from an async context."""
        if self._digest_task is None or self._digest_task.done():
            self._digest_task = asyncio.ensure_future(self._digest_loop())
            logger.info("Digest loop started (every 2h)")

    def stop_digest_loop(self) -> None:
        """Cancel the digest loop."""
        if self._digest_task and not self._digest_task.done():
            self._digest_task.cancel()

    # ── Public API ─────────────────────────────────────────────────────────────

    async def handle(self, event: Event) -> None:
        """Persist event to SQLite+CSV. No individual Telegram — UnifiedReport handles that."""

        self._log_sqlite(event)
        self._log_csv(event)

        # Buffer for UnifiedReport queries
        self._event_history.append(event)
        self._prune_event_history()

        logger.info(
            "[EVENT] %-12s  %-30s  %s  dir=%-5s  score=%.1f",
            event.symbol,
            event.event_type.value,
            event.timestamp.strftime("%H:%M:%S"),
            event.direction,
            event.score,
        )

        await self._dispatch_webhook(event)

    async def close(self) -> None:
        """Close the underlying aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()


# ── Batch replay ───────────────────────────────────────────────────────────────


def batch_replay(
    feature_records: List[Dict[str, Any]],
    *,
    alert_manager: Optional[AlertManager] = None,
) -> List[Event]:
    """
    Replay a list of pre-computed feature snapshots through a fresh
    EventDetector.  Useful for back-testing thresholds against historical data.

    Each record must be a dict with keys matching SymbolState attributes:
        symbol, tier, compression, sps, lfi, rank,
        funding_feats, oi_feats, vol_feats, flow_feats, ob_feats,
        spread_z_history  (list[float]),
        spread_bps_history (list[float]).

    Missing keys fall back to empty dicts / NaN as appropriate.

    If ``alert_manager`` is provided the events are written to SQLite + CSV,
    but webhook dispatch is skipped (no running event loop assumed).

    Returns all fired events in order.
    """
    _DEFAULTS: Dict[str, Any] = {
        "funding_feats":    {},
        "oi_feats":         {},
        "vol_feats":        {},
        "flow_feats":       {},
        "ob_feats":         {},
        "spread_z_history":  [],
        "spread_bps_history": [],
        "compression":      math.nan,
        "sps":              math.nan,
        "lfi":              math.nan,
        "rank":             math.nan,
        "tier":             3,
    }

    detector = EventDetector()
    events: List[Event] = []

    for record in feature_records:
        state = SimpleNamespace(**{**_DEFAULTS, **record})
        sym   = getattr(state, "symbol", "UNKNOWN")
        fired = detector.evaluate(sym, state)

        if fired and alert_manager is not None:
            for ev in fired:
                alert_manager._log_sqlite(ev)
                alert_manager._log_csv(ev)

        events.extend(fired)

    return events
