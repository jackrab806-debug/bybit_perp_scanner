"""Pressure Scanner — real-time per-symbol pressure_rank display.

Combines live orderbook (WebSocket), historical parquet data, and the
feature library into a ranked table that refreshes every second.

Usage
-----
    python -m src.scanner.pressure_scanner
    python -m src.scanner.pressure_scanner --config configs/symbols.yaml
    python -m src.scanner.pressure_scanner --symbols BTCUSDT ETHUSDT
    python -m src.scanner.pressure_scanner --log-level DEBUG

pressure_rank formula
---------------------
    base = (
        0.35 * abs(sps)  / 100
        + 0.30 * compression_score / 100
        + 0.20 * lfi     / 100
        + 0.15 * tier_multiplier(liq_tier)
    )
    pressure_rank = base * 100

    tier_multiplier: T1 = 0.4, T2 = 0.8, T3 = 1.0

Flags
-----
    HOT      : pressure_rank > 75
    CRITICAL : pressure_rank > 85  AND  settlement_phase == "IMMINENT"
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from ..bybit.rest import BybitRestClient
from ..bybit.ws import BybitWebSocketClient, LocalOrderbook
from ..features import (
    compute_flow_features,
    compute_funding_features,
    compute_oi_features,
    compute_orderbook_features,
    compute_volatility_features,
    compression_score,
    liquidity_fragility_index,
    settlement_pressure_score,
)
from ..features.utils import robust_z
from ..events.definitions import AlertManager, EventDetector

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

_ROOT = Path(__file__).resolve().parent.parent.parent
_DATA_DIR = _ROOT / "data" / "raw"
_CONFIG_DEFAULT = _ROOT / "configs" / "symbols.yaml"

# ── Thresholds ────────────────────────────────────────────────────────────────

HOT_THRESHOLD = 75.0
CRITICAL_THRESHOLD = 85.0

# ── Poll intervals (seconds) ──────────────────────────────────────────────────

DISPLAY_INTERVAL = 1.0         # terminal refresh
DISPLAY_INTERVAL_PRESETTL = 0.5  # faster refresh in pre-settlement mode
OI_POLL_INTERVAL = 60.0        # fetch latest OI
FUNDING_POLL_INTERVAL = 300.0  # fetch latest funding rate (5 min)
KLINES_POLL_INTERVAL = 3600.0  # append latest completed 1 h kline

# ── History deque length (168 = 7 days of hourly samples) ─────────────────────

_HIST_LEN = 168

# ── Tier multipliers ──────────────────────────────────────────────────────────

_TIER_MULT: Dict[int, float] = {1: 0.4, 2: 0.8, 3: 1.0}

# ── ANSI helpers ──────────────────────────────────────────────────────────────

_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_RED    = "\033[31m"
_YELLOW = "\033[33m"
_CYAN   = "\033[36m"
_BG_RED = "\033[41m"


def _clr(text: str, *codes: str) -> str:
    return "".join(codes) + text + _RESET


# ── Scoring helpers ───────────────────────────────────────────────────────────


def _tier_multiplier(tier: int) -> float:
    return _TIER_MULT.get(tier, 1.0)


def _pressure_rank(sps: float, cs: float, lfi: float, tier: int) -> float:
    """Combine composite scores into a single 0–100 rank."""
    if any(np.isnan(x) for x in (sps, cs, lfi)):
        return float("nan")
    base = (
        0.35 * abs(sps) / 100.0
        + 0.30 * cs / 100.0
        + 0.20 * lfi / 100.0
        + 0.15 * _tier_multiplier(tier)
    )
    return float(base * 100.0)


# ── Per-symbol state ──────────────────────────────────────────────────────────


@dataclass
class SymbolState:
    symbol: str
    tier: int

    # ── Historical DataFrames (parquet + incremental REST rows) ───────────────
    klines_df:  Optional[pd.DataFrame] = None
    funding_df: Optional[pd.DataFrame] = None
    oi_df:      Optional[pd.DataFrame] = None

    # ── Rolling history deques ────────────────────────────────────────────────
    # thin_pct: raw thinness = 1 / total_usdt
    thinness_history:  Deque[float] = field(default_factory=lambda: deque(maxlen=_HIST_LEN))

    # SPS funding_intensity: history of abs(funding_z) for _pct_rank
    funding_abs_z_history: Deque[float] = field(default_factory=lambda: deque(maxlen=_HIST_LEN))

    # SPS oi_intensity: history of oi_z_24h
    oi_z_history:      Deque[float] = field(default_factory=lambda: deque(maxlen=_HIST_LEN))

    # SPS path_openness: history of raw vacuum_dist in squeeze direction
    vacuum_history:    Deque[float] = field(default_factory=lambda: deque(maxlen=_HIST_LEN))

    # LFI spread_z: raw spread_bps history (to compute robust_z each tick)
    spread_bps_history: Deque[float] = field(default_factory=lambda: deque(maxlen=_HIST_LEN))

    # LFI p_spread: history of past spread_z values (for _pct_rank inside LFI)
    spread_z_history:  Deque[float] = field(default_factory=lambda: deque(maxlen=_HIST_LEN))

    # LFI p_convex: history of raw convexity values
    convexity_history: Deque[float] = field(default_factory=lambda: deque(maxlen=_HIST_LEN))

    # ── Latest computed features ──────────────────────────────────────────────
    funding_feats: Dict[str, Any]   = field(default_factory=dict)
    oi_feats:      Dict[str, float] = field(default_factory=dict)
    vol_feats:     Dict[str, float] = field(default_factory=dict)
    flow_feats:    Dict[str, float] = field(default_factory=dict)
    ob_feats:      Dict[str, float] = field(default_factory=dict)

    # ── Composite scores ──────────────────────────────────────────────────────
    compression: float = float("nan")
    sps:         float = float("nan")
    lfi:         float = float("nan")
    rank:        float = float("nan")

    # ── Realtime 1-second histories (populated by DataCollector only) ────────
    # Trade volume in USDT per second (maxlen=3600 = 1h)
    rt_vol_1s:  Deque[float] = field(default_factory=lambda: deque(maxlen=3600))
    # Mid price per second
    rt_mid_1s:  Deque[float] = field(default_factory=lambda: deque(maxlen=3600))

    # ── 24h turnover (populated once at startup from bulk tickers) ─────────
    turnover_24h: Optional[float] = None

    # ── Staleness timestamps (monotonic) ──────────────────────────────────────
    last_ob_ts:   float = 0.0
    last_oi_ts:   float = 0.0
    last_fund_ts: float = 0.0

    # ── WS reconnect timestamp (UTC) — set before evaluate() call ─────────────
    last_ws_reconnect_ts: Optional[datetime] = None


# ── Scanner ───────────────────────────────────────────────────────────────────


class PressureScanner:
    """
    Real-time pressure scanner for Bybit USDT perpetuals.

    Data flow
    ---------
    Startup  →  load parquet files  →  compute initial features
    WebSocket→  live OB updates  →  recompute composite scores each tick
    REST (60s)  →  append new OI row  →  recompute OI / scores
    REST (5min) →  append new funding row  →  recompute funding / scores
    REST (1h)   →  append new kline row  →  recompute vol / flow / scores
    Display (1s)→  clear screen, render sorted table
    """

    def __init__(
        self,
        symbols_by_tier: Dict[int, List[str]],
        rest: BybitRestClient,
        *,
        event_detector: Optional[EventDetector] = None,
        alert_manager:  Optional[AlertManager]  = None,
    ) -> None:
        self._rest           = rest
        self._event_detector = event_detector
        self._alert_manager  = alert_manager
        self._states: Dict[str, SymbolState] = {}
        all_symbols: List[str] = []

        for tier, syms in sorted(symbols_by_tier.items()):
            for sym in syms:
                self._states[sym] = SymbolState(symbol=sym, tier=tier)
                all_symbols.append(sym)

        self._ws = BybitWebSocketClient(
            symbols=all_symbols,
            orderbook_depth=200,
            on_orderbook=self._on_orderbook,
        )
        self._running = False

    # ── Startup ───────────────────────────────────────────────────────────────

    def _load_parquets(self) -> None:
        """Load historical parquet files into each SymbolState."""
        for sym, state in self._states.items():
            for attr, subdir in (
                ("klines_df",  "klines"),
                ("funding_df", "funding"),
                ("oi_df",      "oi"),
            ):
                path = _DATA_DIR / subdir / f"{sym}.parquet"
                if path.exists():
                    try:
                        setattr(state, attr, pd.read_parquet(path))
                    except Exception as exc:
                        logger.warning("Failed to load %s for %s: %s", subdir, sym, exc)

        logger.info("Parquet data loaded for %d symbols", len(self._states))

    def _compute_initial_features(self) -> None:
        """Compute all features that don't require live OB data."""
        for state in self._states.values():
            self._recompute_kline_features(state)
            self._recompute_oi_features(state)
            self._recompute_funding_features(state)
        logger.info("Initial features computed for %d symbols", len(self._states))

    # ── Per-domain feature recomputation ─────────────────────────────────────

    def _recompute_kline_features(self, state: SymbolState) -> None:
        if state.klines_df is not None and not state.klines_df.empty:
            state.vol_feats  = compute_volatility_features(state.klines_df)
            state.flow_feats = compute_flow_features(state.klines_df)

    def _recompute_oi_features(self, state: SymbolState) -> None:
        if state.oi_df is not None and not state.oi_df.empty:
            state.oi_feats = compute_oi_features(state.oi_df)

    def _recompute_funding_features(self, state: SymbolState) -> None:
        now = datetime.now(timezone.utc)
        state.funding_feats = compute_funding_features(state.funding_df, now=now)

    def _recompute_composite_scores(self, state: SymbolState) -> None:
        """Recompute compression_score, SPS, LFI, and pressure_rank from current features."""
        nan = float("nan")
        rv      = state.vol_feats.get("rv_pct",       nan)
        bb_w    = state.vol_feats.get("bb_width_pct", nan)
        oi_z    = state.oi_feats.get("oi_z_24h",      nan)
        rng_h   = state.vol_feats.get("range_hours",  nan)
        fund_z  = state.funding_feats.get("funding_z", nan)
        mins    = state.funding_feats.get("minutes_to_settlement", nan)
        thin_p  = state.ob_feats.get("thin_pct", nan)
        vac_ask = state.ob_feats.get("vacuum_dist_ask", nan)
        vac_bid = state.ob_feats.get("vacuum_dist_bid", nan)
        spread_bps = state.ob_feats.get("spread_bps", nan)
        convexity  = state.ob_feats.get("convexity",  nan)

        # ── Compression score ─────────────────────────────────────────────────
        state.compression = compression_score(rv, bb_w, oi_z, rng_h)

        # ── Vacuum in the direction of the anticipated squeeze ─────────────────
        if not np.isnan(fund_z) and fund_z != 0:
            # Positive funding → longs pay → bearish squeeze → price moves down → bid thins
            vac_sq = vac_bid if fund_z > 0 else vac_ask
        else:
            vac_sq = nan

        # ── History snapshots (built BEFORE computing scores so first call uses
        #    an empty list and falls back to sigmoid/neutral) ───────────────────
        fz_hist  = list(state.funding_abs_z_history) or None
        oz_hist  = list(state.oi_z_history)           or None
        vac_hist = list(state.vacuum_history)          or None
        sz_hist  = list(state.spread_z_history)        or None
        cx_hist  = list(state.convexity_history)       or None

        # ── Spread z-score (compute externally, as LFI docstring requires) ────
        if not np.isnan(spread_bps) and len(state.spread_bps_history) >= 2:
            spread_z = robust_z(spread_bps, list(state.spread_bps_history))
        else:
            spread_z = nan

        # ── Settlement pressure score ─────────────────────────────────────────
        state.sps = settlement_pressure_score(
            funding_z=fund_z,
            oi_z_7d=oi_z,
            vacuum_dist_squeeze_dir=vac_sq,
            thin_pct=thin_p,
            minutes_to_settle=mins,
            funding_z_history=fz_hist,
            oi_z_history=oz_hist,
            vacuum_history=vac_hist,
        )

        # ── Liquidity Fragility Index ─────────────────────────────────────────
        state.lfi = liquidity_fragility_index(
            thin_pct=thin_p,
            spread_z=spread_z,
            convexity=convexity,
            spread_z_history=sz_hist,
            convexity_history=cx_hist,
        )

        # ── Update rolling histories ──────────────────────────────────────────
        if not np.isnan(fund_z):
            state.funding_abs_z_history.append(abs(fund_z))
        if not np.isnan(oi_z):
            state.oi_z_history.append(oi_z)
        if not np.isnan(vac_sq):
            state.vacuum_history.append(vac_sq)
        if not np.isnan(spread_bps):
            state.spread_bps_history.append(spread_bps)
        if not np.isnan(spread_z):
            state.spread_z_history.append(spread_z)
        if not np.isnan(convexity):
            state.convexity_history.append(convexity)

        # ── Pressure rank ─────────────────────────────────────────────────────
        state.rank = _pressure_rank(state.sps, state.compression, state.lfi, state.tier)

        # ── Event detection ────────────────────────────────────────────────────
        if self._event_detector is not None:
            # Propagate WS reconnect time so _check_vacuum_break can gate on it
            state.last_ws_reconnect_ts = self._ws.last_reconnect_ts
            events = self._event_detector.evaluate(state.symbol, state)
            if events and self._alert_manager is not None:
                for ev in events:
                    asyncio.ensure_future(self._alert_manager.handle(ev))

    # ── WebSocket callback ────────────────────────────────────────────────────

    def _on_orderbook(self, symbol: str, book: LocalOrderbook) -> None:
        state = self._states.get(symbol)
        if state is None:
            return

        bids = book.get_sorted_bids(200)
        asks = book.get_sorted_asks(200)
        if not bids or not asks:
            return

        # Build thinness history for thin_pct computation
        thinness_hist = list(state.thinness_history) if len(state.thinness_history) >= 2 else None

        # Bybit-format snapshot dict
        snapshot = {
            "b": [[str(p), str(s)] for p, s in bids],
            "a": [[str(p), str(s)] for p, s in asks],
        }

        state.ob_feats = compute_orderbook_features(snapshot, history=thinness_hist)
        state.last_ob_ts = time.monotonic()

        # Update thinness history (1 / total_usdt)
        bid_u = state.ob_feats.get("depth_bid_usdt", float("nan"))
        ask_u = state.ob_feats.get("depth_ask_usdt", float("nan"))
        if not (np.isnan(bid_u) or np.isnan(ask_u)) and (bid_u + ask_u) > 0:
            state.thinness_history.append(1.0 / (bid_u + ask_u))

        self._recompute_composite_scores(state)

    # ── REST polling tasks ────────────────────────────────────────────────────

    async def _poll_oi(self) -> None:
        """Append the latest OI value to each symbol's oi_df every 60 s."""
        while self._running:
            await asyncio.sleep(OI_POLL_INTERVAL)
            for sym, state in self._states.items():
                try:
                    cur, _ = await self._rest.get_oi_last_prev(sym)
                    if cur is None:
                        continue
                    new_row = pd.DataFrame([{
                        "timestamp":    pd.Timestamp.now(tz="UTC"),
                        "symbol":       sym,
                        "open_interest": float(cur),
                    }])
                    if state.oi_df is None or state.oi_df.empty:
                        state.oi_df = new_row
                    else:
                        state.oi_df = pd.concat([state.oi_df, new_row], ignore_index=True)
                    self._recompute_oi_features(state)
                    self._recompute_composite_scores(state)
                    state.last_oi_ts = time.monotonic()
                except Exception as exc:
                    logger.debug("OI poll error for %s: %s", sym, exc)

    async def _poll_funding(self) -> None:
        """Append the latest funding rate to each symbol's funding_df every 5 min."""
        while self._running:
            await asyncio.sleep(FUNDING_POLL_INTERVAL)
            for sym, state in self._states.items():
                try:
                    rate = await self._rest.get_latest_funding(sym)
                    if rate is None:
                        continue
                    new_row = pd.DataFrame([{
                        "timestamp":    pd.Timestamp.now(tz="UTC"),
                        "symbol":       sym,
                        "funding_rate": float(rate),
                    }])
                    if state.funding_df is None or state.funding_df.empty:
                        state.funding_df = new_row
                    else:
                        state.funding_df = pd.concat([state.funding_df, new_row], ignore_index=True)
                    self._recompute_funding_features(state)
                    self._recompute_composite_scores(state)
                    state.last_fund_ts = time.monotonic()
                except Exception as exc:
                    logger.debug("Funding poll error for %s: %s", sym, exc)

    async def _poll_klines(self) -> None:
        """Append the latest completed 1 h kline to each symbol's klines_df every hour."""
        while self._running:
            await asyncio.sleep(KLINES_POLL_INTERVAL)
            for sym, state in self._states.items():
                try:
                    result = await self._rest._get(
                        "/v5/market/kline",
                        {"category": "linear", "symbol": sym, "interval": "60", "limit": 2},
                    )
                    klines = result.get("list", [])
                    if len(klines) < 2:
                        continue
                    # index 0 = current (incomplete) bar; index 1 = last completed bar
                    k = klines[1]
                    ts = pd.Timestamp(int(k[0]), unit="ms", tz="UTC")
                    new_row = pd.DataFrame([{
                        "timestamp": ts,
                        "symbol":   sym,
                        "open":     float(k[1]),
                        "high":     float(k[2]),
                        "low":      float(k[3]),
                        "close":    float(k[4]),
                        "volume":   float(k[5]),
                        "turnover": float(k[6]),
                    }])
                    if state.klines_df is None or state.klines_df.empty:
                        state.klines_df = new_row
                    elif ts not in state.klines_df["timestamp"].values:
                        state.klines_df = pd.concat(
                            [state.klines_df, new_row], ignore_index=True
                        )
                    self._recompute_kline_features(state)
                    self._recompute_composite_scores(state)
                except Exception as exc:
                    logger.debug("Klines poll error for %s: %s", sym, exc)

    # ── Display ───────────────────────────────────────────────────────────────

    @staticmethod
    def _minutes_to_settlement() -> float:
        """Minutes until the next 00:00 / 08:00 / 16:00 UTC settlement."""
        now = datetime.now(timezone.utc)
        total_sec = (
            now.hour * 3600 + now.minute * 60 + now.second + now.microsecond / 1e6
        )
        period_sec = 8 * 3600
        into_period = total_sec % period_sec
        mins = ((period_sec - into_period) % period_sec) / 60.0
        # At the exact settlement instant, treat it as just-settled (full period ahead)
        return mins if mins >= 0.1 else 480.0

    def _render(self, mins_to_settle: float) -> str:
        pre_settle = mins_to_settle <= 60.0
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        n = len(self._states)

        sorted_states = sorted(
            self._states.values(),
            key=lambda s: s.rank if not np.isnan(s.rank) else -1.0,
            reverse=True,
        )

        lines: List[str] = []

        # ── Header ────────────────────────────────────────────────────────────
        header = f"  BYBIT PERP PRESSURE SCANNER  |  {now_str}  |  {n} symbols"
        if pre_settle:
            mm = int(mins_to_settle)
            ss = int((mins_to_settle - mm) * 60)
            header += f"  |  {_clr(f'PRE-SETTLEMENT  {mm:02d}:{ss:02d}', _BOLD, _YELLOW)}"
        lines.append(_clr(header, _BOLD))
        lines.append("")

        # ── Column headers ────────────────────────────────────────────────────
        lines.append(
            _clr(
                f"  {'Symbol':<12} {'T':>2}  {'Rank':>6}  {'Phase':<10}  "
                f"{'SPS':>7}  {'CS':>6}  {'LFI':>6}  Flags",
                _BOLD,
            )
        )
        lines.append("  " + "─" * 74)

        # ── Rows ──────────────────────────────────────────────────────────────
        show_n = 10 if pre_settle else len(sorted_states)

        for idx, state in enumerate(sorted_states[:show_n]):
            rank = state.rank
            sps  = state.sps
            cs   = state.compression
            lfi  = state.lfi
            phase = state.funding_feats.get("settlement_phase", "?")

            rank_s = f"{rank:6.1f}" if not np.isnan(rank) else "   NaN"
            sps_s  = f"{sps:+7.1f}" if not np.isnan(sps)  else "    NaN"
            cs_s   = f"{cs:6.1f}"   if not np.isnan(cs)   else "   NaN"
            lfi_s  = f"{lfi:6.1f}"  if not np.isnan(lfi)  else "   NaN"

            is_critical = (
                not np.isnan(rank)
                and rank > CRITICAL_THRESHOLD
                and phase == "IMMINENT"
            )
            is_hot = not np.isnan(rank) and rank > HOT_THRESHOLD

            if is_critical:
                rank_s = _clr(rank_s, _BOLD, _BG_RED)
                flag   = _clr(" CRITICAL", _BOLD, _RED)
            elif is_hot:
                rank_s = _clr(rank_s, _BOLD, _YELLOW)
                flag   = _clr(" HOT", _YELLOW)
            elif idx < 5:
                rank_s = _clr(rank_s, _CYAN)
                flag   = ""
            else:
                flag   = ""

            row = (
                f"  {state.symbol:<12} {state.tier:>2}  {rank_s}  {phase:<10}  "
                f"{sps_s}  {cs_s}  {lfi_s}{flag}"
            )
            lines.append(row)

        # ── Pre-settlement: note hidden symbols ───────────────────────────────
        if pre_settle and len(sorted_states) > show_n:
            hidden = len(sorted_states) - show_n
            lines.append(f"\n  … {hidden} more symbols hidden (showing top {show_n} in pre-settlement mode)")

        # ── Footer ────────────────────────────────────────────────────────────
        lines.append("")
        lines.append(
            f"  Thresholds:  HOT > {HOT_THRESHOLD:.0f}  |  "
            f"CRITICAL > {CRITICAL_THRESHOLD:.0f} + IMMINENT"
        )

        return "\n".join(lines)

    async def _display_loop(self) -> None:
        """Refresh the terminal at DISPLAY_INTERVAL seconds."""
        while self._running:
            mins = self._minutes_to_settlement()
            pre_settle = mins <= 60.0
            text = self._render(mins)
            sys.stdout.write("\033[2J\033[H")
            sys.stdout.write(text + "\n")
            sys.stdout.flush()
            interval = DISPLAY_INTERVAL_PRESETTL if pre_settle else DISPLAY_INTERVAL
            await asyncio.sleep(interval)

    # ── Run ───────────────────────────────────────────────────────────────────

    async def _fetch_turnover(self) -> None:
        """Populate turnover_24h on each SymbolState from bulk tickers."""
        try:
            tickers = await self._rest.get_bulk_tickers()
            for sym, state in self._states.items():
                data = tickers.get(sym)
                if data:
                    state.turnover_24h = float(data.get("turnover24h", 0))
            logger.info("Loaded turnover_24h for %d symbols", len(tickers))
        except Exception as exc:
            logger.warning("Failed to fetch turnover data: %s", exc)

    async def _refresh_universe(self) -> int:
        """Fetch all USDT linear tickers and add symbols with >$1M turnover.

        Returns the number of newly added symbols.
        """
        try:
            tickers = await self._rest.get_bulk_tickers()
        except Exception as exc:
            logger.warning("Universe refresh failed (ticker fetch): %s", exc)
            return 0

        _MIN_TURNOVER = 1_000_000
        new_syms: List[str] = []

        for sym, data in tickers.items():
            if not sym.endswith("USDT"):
                continue
            if sym in self._states:
                # Update turnover for existing symbols while we're here
                self._states[sym].turnover_24h = float(data.get("turnover24h", 0))
                continue
            turnover = float(data.get("turnover24h", 0))
            if turnover < _MIN_TURNOVER:
                continue
            # Assign tier based on turnover
            if turnover >= 500_000_000:
                tier = 1
            elif turnover >= 50_000_000:
                tier = 2
            else:
                tier = 3
            self._states[sym] = SymbolState(symbol=sym, tier=tier)
            self._states[sym].turnover_24h = turnover
            new_syms.append(sym)

        if not new_syms:
            logger.info("Universe refresh: no new symbols (total %d)", len(self._states))
            return 0

        # Load parquet data for new symbols (if backfilled)
        for sym in new_syms:
            state = self._states[sym]
            for attr, subdir in (
                ("klines_df", "klines"),
                ("funding_df", "funding"),
                ("oi_df", "oi"),
            ):
                path = _DATA_DIR / subdir / f"{sym}.parquet"
                if path.exists():
                    try:
                        setattr(state, attr, pd.read_parquet(path))
                    except Exception:
                        pass
            self._recompute_kline_features(state)
            self._recompute_oi_features(state)
            self._recompute_funding_features(state)

        # Subscribe to WS streams for new symbols
        extra = (
            [f"publicTrade.{s}" for s in new_syms]
            + [f"liquidation.{s}" for s in new_syms]
        )
        await self._ws.add_symbols(new_syms, extra_topics=extra)

        logger.info(
            "Universe refresh: added %d symbols (total %d): %s",
            len(new_syms), len(self._states),
            ", ".join(new_syms[:10]) + ("..." if len(new_syms) > 10 else ""),
        )
        return len(new_syms)

    async def run(self) -> None:
        """Start the WS, polling tasks, and display loop; run until cancelled."""
        self._running = True
        self._load_parquets()
        self._compute_initial_features()
        await self._fetch_turnover()

        tasks = [
            asyncio.create_task(self._ws.run(),         name="ws"),
            asyncio.create_task(self._poll_oi(),        name="oi-poll"),
            asyncio.create_task(self._poll_funding(),   name="fund-poll"),
            asyncio.create_task(self._poll_klines(),    name="kline-poll"),
            asyncio.create_task(self._display_loop(),   name="display"),
        ]

        try:
            await asyncio.gather(*tasks)
        except (asyncio.CancelledError, KeyboardInterrupt):
            pass
        finally:
            self._running = False
            self._ws.stop()
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


# ── CLI ───────────────────────────────────────────────────────────────────────


def _load_symbols_yaml(path: Path) -> Dict[int, List[str]]:
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    result: Dict[int, List[str]] = {}
    for tier in (1, 2, 3):
        key = f"tier_{tier}"
        if key in cfg:
            result[tier] = list(cfg[key])
    if not result:
        raise ValueError(f"No tier_1/tier_2/tier_3 keys found in {path}")
    return result


async def _main(args: argparse.Namespace) -> None:
    if args.symbols:
        symbols_by_tier: Dict[int, List[str]] = {3: list(args.symbols)}
    elif args.config.exists():
        symbols_by_tier = _load_symbols_yaml(args.config)
    else:
        print(f"Config not found: {args.config}", file=sys.stderr)
        print("Run backfill first:  python -m src.bybit.backfill", file=sys.stderr)
        sys.exit(1)

    async with BybitRestClient() as rest:
        scanner = PressureScanner(symbols_by_tier, rest)
        await scanner.run()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bybit Perpetuals Pressure Scanner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_CONFIG_DEFAULT,
        help="Path to symbols.yaml (generated by backfill)",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        metavar="SYM",
        help="Override symbol list, e.g. --symbols BTCUSDT ETHUSDT",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )

    try:
        asyncio.run(_main(args))
    except KeyboardInterrupt:
        print("\nScanner stopped.")


if __name__ == "__main__":
    main()
