"""
Outcome Tracker — Measures what actually happened after each scanner event.

Runs every 30 minutes via asyncio task. For events older than 4 hours that
haven't been evaluated yet, fetches 1h klines from Bybit and calculates
price moves in the predicted direction.

Outcome labels:
  STRONG_HIT  — price moved >= 10% in predicted direction within 4h
  HIT         — price moved >= 5%
  PARTIAL     — price moved >= 3%
  MISS        — price moved < 3% or moved against prediction

Results land in the 'outcomes' table in events.db, feeding the
ObductionAgent and ReflectionStore.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional

import aiohttp

logger = logging.getLogger(__name__)

# ── Thresholds ─────────────────────────────────────────────────────────────────

HIT_THRESHOLD_PCT    = 5.0    # direction-adjusted move for a HIT
STRONG_HIT_PCT       = 10.0   # STRONG_HIT
PARTIAL_HIT_PCT      = 3.0    # PARTIAL (noted but not promoted to HIT)
MIN_SCORE_TO_TRACK   = 50.0   # only track events with score >= this
EVAL_WINDOW_HOURS    = 4      # wait at least this long before evaluating
MAX_LOOKBACK_DAYS    = 14     # don't evaluate events older than this

# ── DB helpers ─────────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS outcomes (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol            TEXT    NOT NULL,
    event_type        TEXT    NOT NULL,
    event_score       REAL,
    event_direction   TEXT,
    event_timestamp   TEXT    NOT NULL,

    price_at_event    REAL,
    price_1h          REAL,
    price_2h          REAL,
    price_4h          REAL,

    move_1h_pct       REAL,
    move_2h_pct       REAL,
    move_4h_pct       REAL,
    max_favorable_pct REAL,
    max_adverse_pct   REAL,

    outcome           TEXT,
    outcome_notes     TEXT,
    evaluated_at      TEXT    NOT NULL,

    UNIQUE(symbol, event_type, event_timestamp)
);
CREATE INDEX IF NOT EXISTS idx_outcomes_symbol  ON outcomes(symbol, event_type);
CREATE INDEX IF NOT EXISTS idx_outcomes_outcome ON outcomes(outcome);
CREATE INDEX IF NOT EXISTS idx_outcomes_evts    ON outcomes(event_timestamp);
"""


class OutcomeTracker:
    """Async outcome tracker — safe to run as a long-lived asyncio task."""

    def __init__(self, db_path: str) -> None:
        self._db_path  = db_path
        self._running  = False
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ot-db")

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def initialize(self) -> None:
        """Create the outcomes table (blocking, call once at startup)."""
        with sqlite3.connect(self._db_path) as con:
            for stmt in _SCHEMA.split(";"):
                s = stmt.strip()
                if s:
                    con.execute(s)
            con.commit()
        logger.info("OutcomeTracker: outcomes table ready")

    async def run_loop(self, interval_minutes: int = 30) -> None:
        """Main loop — evaluate pending outcomes every N minutes."""
        self._running = True
        logger.info("OutcomeTracker started (interval=%dmin)", interval_minutes)
        while self._running:
            try:
                await self._evaluate_pending()
            except Exception:
                logger.exception("OutcomeTracker loop error")
            await asyncio.sleep(interval_minutes * 60)

    def stop(self) -> None:
        self._running = False

    # ── Evaluation ─────────────────────────────────────────────────────────────

    async def _evaluate_pending(self) -> None:
        """Fetch unevaluated events and evaluate them."""
        pending = await self._get_pending_events()
        if not pending:
            logger.debug("OutcomeTracker: no pending evaluations")
            return

        logger.info("OutcomeTracker: evaluating %d events", len(pending))
        for row in pending:
            try:
                await self._evaluate_single(row)
                await asyncio.sleep(0.35)   # Bybit rate limit courtesy
            except Exception:
                logger.exception("OutcomeTracker: error evaluating %s", row.get("symbol"))

    async def _get_pending_events(self) -> list:
        """Return events older than EVAL_WINDOW_HOURS not yet in outcomes."""
        min_age_iso  = (datetime.now(timezone.utc) - timedelta(hours=EVAL_WINDOW_HOURS)).isoformat()
        max_age_iso  = (datetime.now(timezone.utc) - timedelta(days=MAX_LOOKBACK_DAYS)).isoformat()

        def _query():
            with sqlite3.connect(self._db_path) as con:
                con.row_factory = sqlite3.Row
                rows = con.execute("""
                    SELECT e.symbol, e.event_type, e.timestamp, e.score, e.direction
                    FROM events e
                    LEFT JOIN outcomes o
                        ON  e.symbol      = o.symbol
                        AND e.event_type  = o.event_type
                        AND e.timestamp   = o.event_timestamp
                    WHERE o.id IS NULL
                      AND e.score       >= ?
                      AND e.timestamp    < ?
                      AND e.timestamp    > ?
                    ORDER BY e.timestamp ASC
                    LIMIT 60
                """, (MIN_SCORE_TO_TRACK, min_age_iso, max_age_iso)).fetchall()
            return [dict(r) for r in rows]

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, _query)

    async def _evaluate_single(self, event: dict) -> None:
        symbol    = event["symbol"]
        direction = event.get("direction", "LONG")
        ts_str    = event["timestamp"]

        # Parse timestamp → milliseconds
        event_ts_ms = _parse_ts_ms(ts_str)
        if event_ts_ms is None:
            logger.warning("OutcomeTracker: cannot parse timestamp %r for %s", ts_str, symbol)
            return

        prices = await self._fetch_prices(symbol, event_ts_ms)
        p0 = prices.get("at_event")
        if not p0 or p0 <= 0:
            return

        p1h = prices.get("1h")
        p2h = prices.get("2h")
        p4h = prices.get("4h")

        def _move(p: Optional[float]) -> Optional[float]:
            if p is None:
                return None
            raw = (p - p0) / p0 * 100.0
            return -raw if direction in ("SHORT", "DN") else raw

        m1h = _move(p1h)
        m2h = _move(p2h)
        m4h = _move(p4h)

        moves = [m for m in (m1h, m2h, m4h) if m is not None]
        max_fav = max(moves) if moves else 0.0
        max_adv = min(moves) if moves else 0.0

        if max_fav >= STRONG_HIT_PCT:
            outcome = "STRONG_HIT"
        elif max_fav >= HIT_THRESHOLD_PCT:
            outcome = "HIT"
        elif max_fav >= PARTIAL_HIT_PCT:
            outcome = "PARTIAL"
        else:
            outcome = "MISS"

        notes = f"max_fav={max_fav:+.1f}% max_adv={max_adv:+.1f}%"

        def _save():
            with sqlite3.connect(self._db_path) as con:
                con.execute("""
                    INSERT OR IGNORE INTO outcomes
                      (symbol, event_type, event_score, event_direction, event_timestamp,
                       price_at_event, price_1h, price_2h, price_4h,
                       move_1h_pct, move_2h_pct, move_4h_pct,
                       max_favorable_pct, max_adverse_pct,
                       outcome, outcome_notes, evaluated_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    symbol, event["event_type"], event.get("score"), direction, ts_str,
                    p0, p1h, p2h, p4h,
                    m1h, m2h, m4h,
                    max_fav, max_adv,
                    outcome, notes,
                    datetime.now(timezone.utc).isoformat(),
                ))
                con.commit()

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, _save)
        logger.info(
            "OutcomeTracker: %s %-24s → %-10s (%s)",
            symbol, event["event_type"], outcome, notes,
        )

    # ── Bybit price fetch ──────────────────────────────────────────────────────

    async def _fetch_prices(self, symbol: str, event_ts_ms: int) -> Dict[str, Optional[float]]:
        """Fetch 1h klines starting at event_ts_ms from Bybit."""
        url    = "https://api.bybit.com/v5/market/kline"
        params = {
            "category": "linear",
            "symbol":   symbol,
            "interval": "60",
            "start":    event_ts_ms,
            "limit":    6,
        }
        try:
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, params=params) as resp:
                    if resp.status != 200:
                        logger.debug("OutcomeTracker: Bybit %d for %s", resp.status, symbol)
                        return {}
                    data = await resp.json()

            klines = data.get("result", {}).get("list", [])
            if not klines:
                return {}
            klines = list(reversed(klines))   # Bybit returns newest-first

            # [startTime, open, high, low, close, volume, turnover]
            def _close(i: int) -> Optional[float]:
                if i < len(klines):
                    try:
                        return float(klines[i][4])
                    except (IndexError, ValueError):
                        return None
                return None

            return {
                "at_event": _close(0),
                "1h":       _close(1),
                "2h":       _close(2),
                "4h":       _close(4),
            }
        except Exception:
            logger.debug("OutcomeTracker: price fetch error for %s", symbol, exc_info=True)
            return {}


# ── Timestamp helper ──────────────────────────────────────────────────────────

def _parse_ts_ms(ts_str: str) -> Optional[int]:
    """Parse an ISO8601 timestamp string to milliseconds epoch."""
    if not ts_str:
        return None
    clean = ts_str.replace("+00:00", "").replace("Z", "")
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
    ):
        try:
            dt = datetime.strptime(clean, fmt).replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000)
        except ValueError:
            continue
    return None
