"""Label snapshots with actual price moves.

Runs every hour. For snapshots older than 4h that haven't been
labeled yet, fetches the actual price move and writes the label.

Uses aiohttp (already a project dependency) for Bybit API calls.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)

# Process max 500 snapshots per cycle, with 0.2s between API calls
_BATCH_LIMIT = 500
_API_DELAY = 0.2


class SnapshotLabeler:
    """Labels ml_snapshots with actual price outcomes."""

    def __init__(self, db_path: str = "data/events.db") -> None:
        self.db_path = db_path
        self._running = False

    async def run_loop(self, interval_seconds: int = 3600) -> None:
        """Label unfilled snapshots every hour."""
        self._running = True
        logger.info("SnapshotLabeler started")

        while self._running:
            try:
                n = await self._label_pending()
                if n > 0:
                    logger.info("SnapshotLabeler: labeled %d snapshots", n)
            except Exception:
                logger.error("SnapshotLabeler error", exc_info=True)

            await asyncio.sleep(interval_seconds)

    async def _label_pending(self) -> int:
        """Find and label unlabeled snapshots older than 5h."""
        # Must be at least 5h old to ensure 4h of price data exists
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=5)).strftime(
            "%Y-%m-%d %H:00:00"
        )

        with sqlite3.connect(self.db_path) as con:
            con.row_factory = sqlite3.Row
            rows = con.execute(
                """
                SELECT id, symbol, timestamp, mid_price
                FROM ml_snapshots
                WHERE label_filled = 0
                AND timestamp < ?
                ORDER BY timestamp
                LIMIT ?
                """,
                (cutoff, _BATCH_LIMIT),
            ).fetchall()

        if not rows:
            return 0

        labeled = 0
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for row in rows:
                try:
                    moves = await self._fetch_price_moves(
                        session, row["symbol"], row["timestamp"], row["mid_price"]
                    )
                    if moves:
                        self._update_label(row["id"], moves)
                        labeled += 1
                    await asyncio.sleep(_API_DELAY)
                except Exception:
                    logger.debug(
                        "Label error %s/%s",
                        row["symbol"], row["timestamp"],
                        exc_info=True,
                    )

        return labeled

    async def _fetch_price_moves(
        self,
        session: aiohttp.ClientSession,
        symbol: str,
        timestamp_str: str,
        price_at: Optional[float],
    ) -> Optional[Dict[str, Optional[float]]]:
        """Fetch price at 1h, 2h, 4h after snapshot."""
        if not price_at or price_at == 0:
            return None

        try:
            dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:00:00")
            dt = dt.replace(tzinfo=timezone.utc)
            ts_ms = int(dt.timestamp() * 1000)
        except ValueError:
            return None

        url = "https://api.bybit.com/v5/market/kline"
        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": "60",
            "start": ts_ms,
            "limit": 6,
        }

        async with session.get(url, params=params) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()

        klines = data.get("result", {}).get("list", [])
        if not klines:
            return None

        # Bybit returns newest-first
        klines = list(reversed(klines))

        def _close(i: int) -> Optional[float]:
            if i < len(klines):
                try:
                    return float(klines[i][4])
                except (IndexError, ValueError):
                    return None
            return None

        p1 = _close(1)
        p2 = _close(2)
        p4 = _close(4)

        move_1h = ((p1 - price_at) / price_at * 100) if p1 else None
        move_2h = ((p2 - price_at) / price_at * 100) if p2 else None
        move_4h = ((p4 - price_at) / price_at * 100) if p4 else None

        # Max absolute move across all available future prices
        closes = [_close(i) for i in range(1, min(len(klines), 5))]
        moves = [((c - price_at) / price_at * 100) for c in closes if c]
        max_move = max(moves, key=abs) if moves else None
        abs_max = abs(max_move) if max_move is not None else None

        return {
            "move_1h": move_1h,
            "move_2h": move_2h,
            "move_4h": move_4h,
            "max_move": max_move,
            "abs_max": abs_max,
        }

    def _update_label(self, snapshot_id: int, moves: Dict[str, Optional[float]]) -> None:
        """Write label to database (sync)."""
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                """
                UPDATE ml_snapshots SET
                    move_1h_pct = ?,
                    move_2h_pct = ?,
                    move_4h_pct = ?,
                    max_move_4h_pct = ?,
                    abs_max_move_4h = ?,
                    label_filled = 1
                WHERE id = ?
                """,
                (
                    moves["move_1h"],
                    moves["move_2h"],
                    moves["move_4h"],
                    moves["max_move"],
                    moves["abs_max"],
                    snapshot_id,
                ),
            )
            con.commit()

    def stop(self) -> None:
        self._running = False
