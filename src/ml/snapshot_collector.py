"""Hourly Feature Snapshot Collector.

Every hour, captures a full feature vector for ALL tracked coins
and saves to SQLite. After enough data accumulates (7+ days),
this feeds the improved ML model.

Key difference from event-based data: we capture ALL coins,
including the 95% that are NOT moving. This gives the model
negative examples to learn from.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from datetime import datetime, timezone
from functools import partial
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..scanner.pressure_scanner import PressureScanner, SymbolState

logger = logging.getLogger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS ml_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    symbol TEXT NOT NULL,

    -- Price
    mid_price REAL,

    -- Funding
    funding_rate REAL,
    funding_z REAL,

    -- Open Interest
    oi_current REAL,
    oi_usd REAL,
    oi_change_1h_pct REAL,
    oi_change_4h_pct REAL,
    oi_z_24h REAL,

    -- Orderbook
    thin_pct REAL,
    depth_bid_usdt REAL,
    depth_ask_usdt REAL,
    vacuum_dist_bid REAL,
    vacuum_dist_ask REAL,
    spread_bps REAL,
    imbalance REAL,

    -- Volatility
    bb_width_pct REAL,
    rv_pct REAL,

    -- Flow
    cvd_ratio_24h REAL,
    taker_proxy REAL,
    price_accel REAL,

    -- Composite
    compression REAL,
    sps REAL,
    lfi REAL,
    rank REAL,
    convexity REAL,

    -- Derived
    oi_to_depth_ratio REAL,
    funding_x_oi REAL,
    vacuum_asymmetry REAL,

    -- BTC context
    btc_change_1h REAL,
    btc_change_4h REAL,

    -- Time
    hour_utc INTEGER,
    mins_to_settlement INTEGER,
    is_weekend INTEGER,

    -- Events active (from events table, last 2h)
    has_fs INTEGER DEFAULT 0,
    has_vb INTEGER DEFAULT 0,
    has_ve INTEGER DEFAULT 0,
    has_ca INTEGER DEFAULT 0,
    has_oi INTEGER DEFAULT 0,
    num_event_types_2h INTEGER DEFAULT 0,
    max_event_score REAL DEFAULT 0,

    -- Label (filled LATER by SnapshotLabeler)
    move_1h_pct REAL,
    move_2h_pct REAL,
    move_4h_pct REAL,
    max_move_4h_pct REAL,
    abs_max_move_4h REAL,
    label_filled INTEGER DEFAULT 0,

    UNIQUE(timestamp, symbol)
)
"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_snap_ts ON ml_snapshots(timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_snap_symbol ON ml_snapshots(symbol)",
    "CREATE INDEX IF NOT EXISTS idx_snap_unfilled ON ml_snapshots(label_filled) WHERE label_filled = 0",
]

_INSERT_SQL = """
INSERT OR IGNORE INTO ml_snapshots (
    timestamp, symbol, mid_price,
    funding_rate, funding_z,
    oi_current, oi_usd, oi_change_1h_pct, oi_change_4h_pct, oi_z_24h,
    thin_pct, depth_bid_usdt, depth_ask_usdt,
    vacuum_dist_bid, vacuum_dist_ask, spread_bps, imbalance,
    bb_width_pct, rv_pct,
    cvd_ratio_24h, taker_proxy, price_accel,
    compression, sps, lfi, rank, convexity,
    oi_to_depth_ratio, funding_x_oi, vacuum_asymmetry,
    btc_change_1h, btc_change_4h,
    hour_utc, mins_to_settlement, is_weekend,
    has_fs, has_vb, has_ve, has_ca, has_oi,
    num_event_types_2h, max_event_score
) VALUES (
    :timestamp, :symbol, :mid_price,
    :funding_rate, :funding_z,
    :oi_current, :oi_usd, :oi_change_1h_pct, :oi_change_4h_pct, :oi_z_24h,
    :thin_pct, :depth_bid_usdt, :depth_ask_usdt,
    :vacuum_dist_bid, :vacuum_dist_ask, :spread_bps, :imbalance,
    :bb_width_pct, :rv_pct,
    :cvd_ratio_24h, :taker_proxy, :price_accel,
    :compression, :sps, :lfi, :rank, :convexity,
    :oi_to_depth_ratio, :funding_x_oi, :vacuum_asymmetry,
    :btc_change_1h, :btc_change_4h,
    :hour_utc, :mins_to_settlement, :is_weekend,
    :has_fs, :has_vb, :has_ve, :has_ca, :has_oi,
    :num_event_types_2h, :max_event_score
)
"""


class SnapshotCollector:
    """Collects hourly feature snapshots for all coins."""

    def __init__(self, scanner: "PressureScanner", db_path: str = "data/events.db") -> None:
        self.scanner = scanner
        self.db_path = db_path
        self._running = False

    def initialize(self) -> None:
        """Create snapshots table (sync — call before event loop)."""
        with sqlite3.connect(self.db_path) as con:
            con.execute(_CREATE_TABLE)
            for idx_sql in _CREATE_INDEXES:
                con.execute(idx_sql)
            con.commit()
        logger.info("SnapshotCollector: ml_snapshots table ready")

    async def run_loop(self, interval_seconds: int = 3600) -> None:
        """Capture snapshots every hour."""
        self._running = True
        logger.info("SnapshotCollector started (interval=%ds)", interval_seconds)

        while self._running:
            try:
                loop = asyncio.get_event_loop()
                n = await loop.run_in_executor(None, self._capture_all)
                logger.info("SnapshotCollector: captured %d snapshots", n)
            except Exception:
                logger.error("SnapshotCollector error", exc_info=True)

            await asyncio.sleep(interval_seconds)

    def _capture_all(self) -> int:
        """Capture snapshot for all coins (runs in executor thread)."""
        states: Dict[str, Any] = getattr(self.scanner, "_states", {})
        if not states:
            return 0

        now = datetime.now(timezone.utc)
        ts_str = now.strftime("%Y-%m-%d %H:00:00")  # Round to hour

        # BTC context
        btc_1h, btc_4h = self._get_btc_context(states)

        # Recent events from DB
        events_by_sym = self._get_recent_events()

        rows: List[Dict[str, Any]] = []
        for symbol, state in states.items():
            row = self._extract_features(
                symbol, state, now, btc_1h, btc_4h,
                events_by_sym.get(symbol, []),
            )
            if row:
                row["timestamp"] = ts_str
                rows.append(row)

        if not rows:
            return 0

        inserted = 0
        with sqlite3.connect(self.db_path) as con:
            for row in rows:
                try:
                    con.execute(_INSERT_SQL, row)
                    inserted += 1
                except Exception:
                    pass  # UNIQUE constraint = already captured this hour
            con.commit()

        return inserted

    # ── Feature extraction ────────────────────────────────────────────────────

    def _extract_features(
        self,
        symbol: str,
        state: "SymbolState",
        now: datetime,
        btc_1h: Optional[float],
        btc_4h: Optional[float],
        recent_events: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Extract full feature vector from SymbolState."""
        row: Dict[str, Any] = {"symbol": symbol}

        ff = state.funding_feats or {}
        oi = state.oi_feats or {}
        vol = state.vol_feats or {}
        fl = state.flow_feats or {}
        ob = state.ob_feats or {}

        # Price
        mid = ob.get("mid_price")
        row["mid_price"] = mid

        # Funding
        row["funding_rate"] = ff.get("funding_current")
        row["funding_z"] = ff.get("funding_z")

        # OI
        oi_cur = oi.get("oi_current")
        oi_usd = (oi_cur * mid) if (oi_cur and mid) else None
        row["oi_current"] = oi_cur
        row["oi_usd"] = oi_usd
        row["oi_change_1h_pct"] = oi.get("oi_pct_1h")
        row["oi_change_4h_pct"] = oi.get("oi_pct_4h")
        row["oi_z_24h"] = oi.get("oi_z_24h")

        # Orderbook
        d_bid = ob.get("depth_bid_usdt")
        d_ask = ob.get("depth_ask_usdt")
        v_bid = ob.get("vacuum_dist_bid")
        v_ask = ob.get("vacuum_dist_ask")
        row["thin_pct"] = ob.get("thin_pct")
        row["depth_bid_usdt"] = d_bid
        row["depth_ask_usdt"] = d_ask
        row["vacuum_dist_bid"] = v_bid
        row["vacuum_dist_ask"] = v_ask
        row["spread_bps"] = ob.get("spread_bps")
        row["imbalance"] = ob.get("depth_band_imbalance")
        row["convexity"] = ob.get("convexity")

        # Volatility
        row["bb_width_pct"] = vol.get("bb_width_pct")
        row["rv_pct"] = vol.get("rv_pct")

        # Flow
        row["cvd_ratio_24h"] = fl.get("cvd_ratio_24h")
        row["taker_proxy"] = fl.get("taker_proxy")
        row["price_accel"] = fl.get("price_accel")

        # Composite scores
        row["compression"] = _safe(state.compression)
        row["sps"] = _safe(state.sps)
        row["lfi"] = _safe(state.lfi)
        row["rank"] = _safe(state.rank)

        # Derived
        row["oi_to_depth_ratio"] = None
        if oi_usd and d_ask and d_ask > 0:
            row["oi_to_depth_ratio"] = oi_usd / d_ask

        fz = ff.get("funding_z")
        row["funding_x_oi"] = None
        if fz is not None and row["oi_z_24h"] is not None:
            row["funding_x_oi"] = abs(fz) * abs(row["oi_z_24h"])

        row["vacuum_asymmetry"] = None
        if v_bid is not None and v_ask is not None:
            s = v_bid + v_ask
            if s > 1e-6:
                row["vacuum_asymmetry"] = (v_ask - v_bid) / s

        # BTC context
        row["btc_change_1h"] = btc_1h
        row["btc_change_4h"] = btc_4h

        # Time
        row["hour_utc"] = now.hour
        cur_mins = now.hour * 60 + now.minute
        row["mins_to_settlement"] = min(
            (s * 60 - cur_mins) % 1440 for s in (0, 8, 16, 24)
        )
        row["is_weekend"] = 1 if now.weekday() >= 5 else 0

        # Events
        event_types: set = set()
        max_score = 0.0
        for ev in recent_events:
            et = ev.get("event_type", "")
            event_types.add(et)
            sc = ev.get("score") or 0
            if sc > max_score:
                max_score = sc

        row["has_fs"] = 1 if "FUNDING_SQUEEZE_SETUP" in event_types else 0
        row["has_vb"] = 1 if "VACUUM_BREAK" in event_types else 0
        row["has_ve"] = 1 if "VOLUME_EXPLOSION" in event_types else 0
        row["has_ca"] = 1 if "CASCADE_ACTIVE" in event_types else 0
        row["has_oi"] = 1 if "OI_SURGE" in event_types else 0
        row["num_event_types_2h"] = len(event_types)
        row["max_event_score"] = max_score

        return row

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_btc_context(
        self, states: Dict[str, Any]
    ) -> tuple:
        """Get BTC 1h and 4h price changes from klines."""
        btc = states.get("BTCUSDT")
        if btc is None or btc.klines_df is None or btc.klines_df.empty:
            return None, None

        df = btc.klines_df
        close_now = float(df["close"].iloc[-1])
        btc_1h = None
        btc_4h = None

        if len(df) >= 2:
            c1 = float(df["close"].iloc[-2])
            if c1 > 0:
                btc_1h = (close_now - c1) / c1 * 100.0

        if len(df) >= 5:
            c4 = float(df["close"].iloc[-5])
            if c4 > 0:
                btc_4h = (close_now - c4) / c4 * 100.0

        return btc_1h, btc_4h

    def _get_recent_events(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get events from last 2h grouped by symbol (sync)."""
        by_sym: Dict[str, List[Dict[str, Any]]] = {}
        try:
            with sqlite3.connect(self.db_path) as con:
                con.row_factory = sqlite3.Row
                cur = con.execute("""
                    SELECT symbol, event_type, score, direction
                    FROM events
                    WHERE timestamp > datetime('now', '-2 hours')
                """)
                for row in cur.fetchall():
                    sym = row["symbol"]
                    by_sym.setdefault(sym, []).append(dict(row))
        except Exception as exc:
            logger.debug("Event fetch error: %s", exc)
        return by_sym

    def stop(self) -> None:
        self._running = False


def _safe(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
        return None if f != f else f  # NaN → None
    except (TypeError, ValueError):
        return None
