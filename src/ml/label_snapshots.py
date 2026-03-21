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

# Process max 5000 snapshots per cycle, 10 parallel API calls
_BATCH_LIMIT = 5000
_API_DELAY = 0.05
_MAX_CONCURRENT = 10


class SnapshotLabeler:
    """Labels ml_snapshots with actual price outcomes."""

    def __init__(self, db_path: str = "data/events.db") -> None:
        self.db_path = db_path
        self._running = False

    async def run_loop(self, interval_seconds: int = 1800) -> None:
        """Label unfilled snapshots every 30 min."""
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

        # Group rows by (symbol, timestamp) to deduplicate API calls
        groups: Dict[tuple, List[sqlite3.Row]] = {}
        for row in rows:
            key = (row["symbol"], row["timestamp"])
            groups.setdefault(key, []).append(row)

        labeled = 0
        sem = asyncio.Semaphore(_MAX_CONCURRENT)
        timeout = aiohttp.ClientTimeout(total=10)

        async with aiohttp.ClientSession(timeout=timeout) as session:

            async def _process_group(
                key: tuple, group_rows: List[sqlite3.Row]
            ) -> int:
                async with sem:
                    symbol, timestamp_str = key
                    mid_price = group_rows[0]["mid_price"]
                    try:
                        moves = await self._fetch_price_moves(
                            session, symbol, timestamp_str, mid_price
                        )
                        if moves:
                            self._update_labels_batch(
                                [r["id"] for r in group_rows], moves
                            )
                            return len(group_rows)
                    except Exception:
                        logger.debug(
                            "Label error %s/%s", symbol, timestamp_str,
                            exc_info=True,
                        )
                    await asyncio.sleep(_API_DELAY)
                    return 0

            tasks = [
                _process_group(key, group_rows)
                for key, group_rows in groups.items()
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, int):
                    labeled += r

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

    def _update_labels_batch(
        self, snapshot_ids: List[int], moves: Dict[str, Optional[float]]
    ) -> None:
        """Write label to database for multiple IDs sharing same moves (sync)."""
        params = (
            moves["move_1h"],
            moves["move_2h"],
            moves["move_4h"],
            moves["max_move"],
            moves["abs_max"],
        )
        with sqlite3.connect(self.db_path) as con:
            for sid in snapshot_ids:
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
                    params + (sid,),
                )
            con.commit()

    def stop(self) -> None:
        self._running = False
