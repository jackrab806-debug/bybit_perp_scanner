"""
Reflection Store — Persistent learnings from the Obduction Agent.

Reads the 'reflections' table and formats recent learnings as context
that the Analysis Agent injects into Claude prompts, making each
analysis aware of past performance patterns.

Also provides per-event-type hit-rate statistics so the Analysis Agent
can cite historical accuracy in its output.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class ReflectionStore:
    """Thread-safe async reader for the reflections and outcomes tables."""

    def __init__(self, db_path: str) -> None:
        self._db_path  = db_path
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rs-db")

    async def get_recent_learnings(self, limit: int = 5) -> str:
        """
        Return the N most recent learnings formatted for prompt injection.

        Example output:
            <past_learnings>
            - LEARNING: OI_SURGE score>80 → 72% hit rate, prioritise over VB.
            - LEARNING: FS in APPROACH phase outperforms FAR 3x.
            </past_learnings>

        Returns empty string if no learnings exist yet.
        """
        def _query() -> list:
            try:
                with sqlite3.connect(self._db_path) as con:
                    rows = con.execute("""
                        SELECT learning FROM reflections
                        ORDER BY timestamp DESC
                        LIMIT ?
                    """, (limit,)).fetchall()
                    return [r[0] for r in rows if r[0]]
            except sqlite3.OperationalError:
                return []   # table doesn't exist yet

        loop = asyncio.get_running_loop()
        learnings = await loop.run_in_executor(self._executor, _query)

        if not learnings:
            return ""

        lines = ["<past_learnings>"]
        for l in learnings:
            lines.append(f"- {l}")
        lines.append("</past_learnings>")
        return "\n".join(lines)

    async def get_outcome_stats(
        self, event_type: str, score_min: float = 0.0
    ) -> Optional[Dict]:
        """
        Return hit-rate statistics for a specific event type + score range.

        Used by Analysis Agent to say e.g. "historically, OI_SURGE score>75
        has a 68% hit rate with avg +7.2% favorable move".

        Returns None if insufficient data (< 5 outcomes).
        """
        def _query() -> Optional[Dict]:
            try:
                with sqlite3.connect(self._db_path) as con:
                    con.row_factory = sqlite3.Row
                    row = con.execute("""
                        SELECT
                            COUNT(*)  as total,
                            SUM(CASE WHEN outcome IN ('HIT','STRONG_HIT')
                                     THEN 1 ELSE 0 END) as hits,
                            AVG(max_favorable_pct) as avg_fav,
                            AVG(max_adverse_pct)   as avg_adv
                        FROM outcomes
                        WHERE event_type  = ?
                          AND event_score >= ?
                    """, (event_type, score_min)).fetchone()

                    if row is None or row["total"] < 5:
                        return None

                    return {
                        "event_type": event_type,
                        "score_min":  score_min,
                        "total":      row["total"],
                        "hits":       row["hits"],
                        "hit_rate":   round(row["hits"] / row["total"] * 100, 1),
                        "avg_fav":    round(row["avg_fav"] or 0, 1),
                        "avg_adv":    round(row["avg_adv"] or 0, 1),
                    }
            except sqlite3.OperationalError:
                return None

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, _query)
