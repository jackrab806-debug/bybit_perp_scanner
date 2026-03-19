"""
Obduction Agent — Post-settlement performance analysis and pattern discovery.

Runs on schedule: 02:00, 10:00, 18:00 UTC (2 hours after each 8h settlement).
Queries outcome stats, calls Claude to find patterns, writes reflections to DB,
and sends a concise 📊 report to Telegram.

"Obduction" = inference to the best explanation — reverse-engineering WHY
certain signals led to big moves and others didn't.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)

MODEL            = "claude-sonnet-4-5"
OBDUCTION_HOURS  = (2, 10, 18)       # UTC hours to run
MIN_OUTCOMES     = 10                  # skip if fewer than this many outcomes in last 24h
MAX_REFLECTIONS  = 20                  # cap stored reflections

_REFLECTIONS_SCHEMA = """
CREATE TABLE IF NOT EXISTS reflections (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp     TEXT    NOT NULL,
    learning      TEXT    NOT NULL,
    full_analysis TEXT,
    stats_json    TEXT,
    applied       INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_reflections_ts ON reflections(timestamp DESC);
"""


class ObductionAgent:
    """Analyzes hit/miss patterns, writes learnings, sends Telegram report."""

    def __init__(self, db_path: str) -> None:
        self._db_path  = db_path
        self._api_key  = os.environ.get("ANTHROPIC_API_KEY", "")
        self._tg_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self._tg_chat  = os.environ.get("TELEGRAM_CHAT_ID", "")
        self._running  = False
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ob-db")

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def initialize(self) -> None:
        """Create reflections table if needed (blocking, call once)."""
        with sqlite3.connect(self._db_path) as con:
            for stmt in _REFLECTIONS_SCHEMA.split(";"):
                s = stmt.strip()
                if s:
                    con.execute(s)
            con.commit()
        logger.info("ObductionAgent: reflections table ready")

    async def run_scheduled(self) -> None:
        """Run obduction at 02:00, 10:00, 18:00 UTC forever."""
        self._running = True
        logger.info("ObductionAgent scheduler started (runs at %s UTC)", OBDUCTION_HOURS)
        while self._running:
            now        = datetime.now(timezone.utc)
            next_run   = _next_scheduled_time(now, OBDUCTION_HOURS)
            sleep_secs = max(60.0, (next_run - now).total_seconds())
            logger.debug(
                "ObductionAgent: next run at %s UTC (%.0f min)",
                next_run.strftime("%H:%M"), sleep_secs / 60,
            )
            await asyncio.sleep(sleep_secs)
            try:
                await self.run_obduction()
            except Exception:
                logger.exception("ObductionAgent: run_obduction error")

    def stop(self) -> None:
        self._running = False

    # ── Main analysis ──────────────────────────────────────────────────────────

    async def run_obduction(self) -> Optional[str]:
        """Full analysis cycle: gather → analyse → save → telegram."""
        stats = await self._gather_stats()
        if stats is None:
            logger.info("ObductionAgent: insufficient outcome data, skipping")
            return None

        analysis = await self._analyse_with_claude(stats)

        if analysis:
            await self._save_reflection(analysis, stats)
            await self._send_telegram(analysis, stats)
            logger.info("ObductionAgent: cycle complete (%d chars)", len(analysis))

        return analysis

    # ── Stats gathering ────────────────────────────────────────────────────────

    async def _gather_stats(self) -> Optional[Dict[str, Any]]:
        """Read outcome stats from SQLite. Returns None if too few rows."""

        def _query() -> Optional[Dict[str, Any]]:
            with sqlite3.connect(self._db_path) as con:
                con.row_factory = sqlite3.Row

                # 24h aggregate
                counts = {
                    row["outcome"]: row["cnt"]
                    for row in con.execute("""
                        SELECT outcome, COUNT(*) as cnt
                        FROM outcomes
                        WHERE evaluated_at > datetime('now','-24 hours')
                        GROUP BY outcome
                    """)
                }
                total_24h = sum(counts.values())
                if total_24h < MIN_OUTCOMES:
                    return None

                # Per-type breakdown (7 days)
                per_type: List[Dict] = [
                    dict(r) for r in con.execute("""
                        SELECT event_type, outcome,
                               COUNT(*)            as cnt,
                               AVG(event_score)    as avg_score,
                               AVG(max_favorable_pct) as avg_fav,
                               AVG(max_adverse_pct)   as avg_adv
                        FROM outcomes
                        WHERE evaluated_at > datetime('now','-7 days')
                        GROUP BY event_type, outcome
                        ORDER BY event_type, outcome
                    """)
                ]

                # Score-band hit rates (7 days)
                bands: List[Dict] = [
                    dict(r) for r in con.execute("""
                        SELECT
                            CASE
                                WHEN event_score >= 80 THEN '80+'
                                WHEN event_score >= 70 THEN '70-79'
                                WHEN event_score >= 60 THEN '60-69'
                                ELSE '50-59'
                            END                     as band,
                            COUNT(*)                as total,
                            SUM(CASE WHEN outcome IN ('HIT','STRONG_HIT')
                                     THEN 1 ELSE 0 END) as hits,
                            AVG(max_favorable_pct)  as avg_fav
                        FROM outcomes
                        WHERE evaluated_at > datetime('now','-7 days')
                        GROUP BY band
                        ORDER BY band DESC
                    """)
                ]

                # Best 5 (24h)
                best = [
                    dict(r) for r in con.execute("""
                        SELECT symbol, event_type, event_score,
                               event_direction, max_favorable_pct, outcome
                        FROM outcomes
                        WHERE evaluated_at > datetime('now','-24 hours')
                        ORDER BY max_favorable_pct DESC
                        LIMIT 5
                    """)
                ]

                # Worst 5 (24h)
                worst = [
                    dict(r) for r in con.execute("""
                        SELECT symbol, event_type, event_score,
                               event_direction, max_favorable_pct, outcome
                        FROM outcomes
                        WHERE evaluated_at > datetime('now','-24 hours')
                        ORDER BY max_favorable_pct ASC
                        LIMIT 5
                    """)
                ]

                # All-time summary
                row = con.execute("""
                    SELECT COUNT(*) as total,
                           SUM(CASE WHEN outcome IN ('HIT','STRONG_HIT')
                                    THEN 1 ELSE 0 END) as hits,
                           AVG(max_favorable_pct) as avg_fav
                    FROM outcomes
                """).fetchone()
                alltime = dict(row) if row else {}

                return {
                    "outcome_counts_24h": counts,
                    "total_24h":          total_24h,
                    "per_type_7d":        per_type,
                    "score_bands_7d":     bands,
                    "best_24h":           best,
                    "worst_24h":          worst,
                    "alltime":            alltime,
                }

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, _query)

    # ── Claude analysis ────────────────────────────────────────────────────────

    async def _analyse_with_claude(self, stats: Dict[str, Any]) -> str:
        """Call Claude to analyse patterns; fall back to basic report on failure."""
        if not self._api_key:
            return _basic_report(stats)

        prompt = _build_obduction_prompt(stats)
        try:
            timeout = aiohttp.ClientTimeout(total=25)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key":         self._api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type":      "application/json",
                    },
                    json={
                        "model":      MODEL,
                        "max_tokens": 350,
                        "messages":   [{"role": "user", "content": prompt}],
                    },
                ) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        logger.error("ObductionAgent: Claude HTTP %d: %s", resp.status, body[:200])
                        return _basic_report(stats)
                    data = await resp.json()
                    text = (data.get("content") or [{}])[0].get("text", "").strip()
                    return text or _basic_report(stats)

        except Exception:
            logger.exception("ObductionAgent: Claude API error")
            return _basic_report(stats)

    # ── Reflection persistence ─────────────────────────────────────────────────

    async def _save_reflection(self, analysis: str, stats: Dict[str, Any]) -> None:
        """Extract LEARNING line and persist to reflections table."""
        learning = ""
        for line in analysis.splitlines():
            if "LEARNING:" in line.upper():
                learning = line.strip()
                break
        if not learning:
            # Use first non-empty line as fallback learning
            for line in analysis.splitlines():
                if line.strip():
                    learning = line.strip()[:200]
                    break

        def _save():
            with sqlite3.connect(self._db_path) as con:
                con.execute("""
                    INSERT INTO reflections (timestamp, learning, full_analysis, stats_json)
                    VALUES (?, ?, ?, ?)
                """, (
                    datetime.now(timezone.utc).isoformat(),
                    learning,
                    analysis,
                    json.dumps(stats.get("outcome_counts_24h", {})),
                ))
                # Prune old reflections beyond cap
                con.execute("""
                    DELETE FROM reflections
                    WHERE id NOT IN (
                        SELECT id FROM reflections
                        ORDER BY timestamp DESC
                        LIMIT ?
                    )
                """, (MAX_REFLECTIONS,))
                con.commit()

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, _save)
        logger.info("ObductionAgent: reflection saved: %s", learning[:80])

    # ── Telegram report ────────────────────────────────────────────────────────

    async def _send_telegram(self, analysis: str, stats: Dict[str, Any]) -> None:
        if not self._tg_token or not self._tg_chat:
            return

        counts    = stats.get("outcome_counts_24h", {})
        total     = stats.get("total_24h", 0)
        hits      = counts.get("HIT", 0) + counts.get("STRONG_HIT", 0)
        hit_rate  = f"{hits/total*100:.0f}%" if total else "N/A"

        msg = (
            f"\U0001f4ca <b>OBDUCTION</b>\n"
            f"24h: {hits}/{total} hits ({hit_rate})\n\n"
            f"{analysis}"
        )

        try:
            timeout = aiohttp.ClientTimeout(total=15)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"https://api.telegram.org/bot{self._tg_token}/sendMessage",
                    json={
                        "chat_id":    self._tg_chat,
                        "text":       msg,
                        "parse_mode": "HTML",
                    },
                ) as resp:
                    if resp.status >= 400:
                        body = await resp.text()
                        logger.warning("ObductionAgent: Telegram HTTP %d: %s", resp.status, body[:100])
        except Exception:
            logger.exception("ObductionAgent: Telegram send error")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _next_scheduled_time(now: datetime, hours: tuple) -> datetime:
    """Return the next datetime matching one of the given UTC hours."""
    for h in hours:
        candidate = now.replace(hour=h, minute=0, second=0, microsecond=0)
        if candidate > now:
            return candidate
    # All hours passed today — first slot tomorrow
    tomorrow = now.date() + timedelta(days=1)
    return datetime(tomorrow.year, tomorrow.month, tomorrow.day,
                    hours[0], 0, 0, tzinfo=timezone.utc)


def _basic_report(stats: Dict[str, Any]) -> str:
    """Fallback plain-text report when Claude is unavailable."""
    counts   = stats.get("outcome_counts_24h", {})
    total    = stats.get("total_24h", 0)
    hits     = counts.get("HIT", 0) + counts.get("STRONG_HIT", 0)
    rate_str = f"{hits/total*100:.0f}%" if total else "N/A"

    best = stats.get("best_24h", [{}])
    top  = best[0] if best else {}

    bands = stats.get("score_bands_7d", [])
    band_str = " | ".join(
        f"{b['band']}: {b['hits']}/{b['total']}"
        for b in bands
    )

    return (
        f"24h results: {hits}/{total} ({rate_str}) hits.\n"
        f"Score bands (7d): {band_str or 'N/A'}\n"
        f"Best: {top.get('symbol','?')} {top.get('event_type','?')} "
        f"{top.get('max_favorable_pct',0):+.1f}%"
    )


def _build_obduction_prompt(stats: Dict[str, Any]) -> str:
    return f"""<role>
You are a performance analyst for a crypto perpetual futures scanner.
Analyze hit/miss data and extract actionable patterns — concise, data-driven.
</role>

<data>
{json.dumps(stats, indent=2, default=str)}
</data>

<task>
Produce a 4-part report (total <= 150 words):

1. SCORECARD (1 sentence): Overall 24h hit rate + best event type.

2. KEY PATTERN (2 sentences): What distinguishes HITs from MISSes?
   Focus on score bands, event types, or direction bias.

3. ACTIONABLE (1 sentence): One specific threshold or filter change
   that would improve performance based on this data.

4. LEARNING (1 sentence, starts with "LEARNING:"):
   A concrete rule for future analysis. Example format:
   "LEARNING: OI_SURGE score>80 → 72% hit rate, prioritize over VB."

Only reference numbers present in the data. No hedging.
</task>"""
