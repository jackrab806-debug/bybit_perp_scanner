"""
Settlement Scheduler.

Runs SettlementScanner automatically ~60 min before each settlement.
Prints rankings to terminal and optionally sends webhook alerts.

Can also be run standalone for a one-shot scan.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, TYPE_CHECKING

from .scanner import (
    SettlementScanner,
    format_rankings_table,
    format_telegram_message,
)

if TYPE_CHECKING:
    from ..scanner.pressure_scanner import PressureScanner
    from ..events.definitions import AlertManager

logger = logging.getLogger(__name__)

# Bybit settlement hours (UTC)
SETTLEMENT_HOURS = [0, 8, 16]

# How far before settlement to start scanning (minutes)
PRE_SCAN_MINUTES = 60

# How often to re-scan during the pre-settlement window (minutes)
UPDATE_INTERVAL_MINUTES = 10


class SettlementScheduler:
    """
    Runs settlement scan automatically and re-scans every 10 min
    during the last hour before each settlement.
    """

    def __init__(
        self,
        pressure_scanner: PressureScanner,
        alert_manager: Optional[AlertManager] = None,
    ) -> None:
        self.scanner = SettlementScanner(pressure_scanner)
        self.alert_manager = alert_manager
        self._running = False
        self._tg_sent_for: Optional[datetime] = None  # track which settlement got TG

    async def run_forever(self) -> None:
        """Main loop -- runs alongside the collector."""
        self._running = True
        logger.info("SettlementScheduler started")

        while self._running:
            now = datetime.now(timezone.utc)
            next_settle = _next_settlement(now)
            secs_to_settle = (next_settle - now).total_seconds()
            secs_to_scan = secs_to_settle - (PRE_SCAN_MINUTES * 60)

            if secs_to_scan > 60:
                # Not yet in the scan window -- sleep until ~30s before it
                sleep_for = max(30, secs_to_scan - 30)
                logger.debug(
                    "Next settlement: %s UTC (%d min). Sleeping %d min.",
                    next_settle.strftime("%H:%M"),
                    secs_to_settle // 60,
                    sleep_for // 60,
                )
                await asyncio.sleep(sleep_for)
                continue

            if secs_to_settle > 0:
                # We are in the T-60 to T-0 window
                mins_left = secs_to_settle / 60
                logger.info(
                    "PRE-SETTLEMENT SCAN -- %s UTC in %d min",
                    next_settle.strftime("%H:%M"),
                    int(mins_left),
                )
                # Send Telegram only ONCE per settlement at T-30min
                send_tg = (
                    mins_left <= 30
                    and self._tg_sent_for != next_settle
                )
                await self._run_scan(send_telegram=send_tg)
                if send_tg:
                    self._tg_sent_for = next_settle
                await asyncio.sleep(UPDATE_INTERVAL_MINUTES * 60)
            else:
                # Settlement just passed -- wait 60s then loop
                await asyncio.sleep(60)

    async def run_once(self) -> None:
        """Run a single scan immediately (for manual use / testing)."""
        await self._run_scan()

    async def _run_scan(self, send_telegram: bool = True) -> None:
        """Run scan and produce output."""
        try:
            rankings = await self.scanner.compute_rankings()

            if not rankings:
                logger.warning("No rankings produced -- missing data?")
                return

            # Terminal output
            table = format_rankings_table(
                rankings, top_n=10, n_total=self.scanner.n_symbols,
            )
            print("\n" + table)

            # Webhook / Telegram (if configured and allowed)
            if send_telegram and self.alert_manager is not None:
                msg = format_telegram_message(rankings, top_n=10)
                if hasattr(self.alert_manager, "send_settlement_scan"):
                    await self.alert_manager.send_settlement_scan(msg)
                elif hasattr(self.alert_manager, "_webhook_url") and self.alert_manager._webhook_url:
                    logger.debug("Webhook configured but send_settlement_scan not available")

            # Log top 10
            for r in rankings[:10]:
                logger.info(
                    "Settlement target #%d: %s %s score=%.0f",
                    r["rank"], r["symbol"], r["direction"], r["score"],
                )

        except Exception as e:
            logger.error("Settlement scan error: %s", e, exc_info=True)

    def stop(self) -> None:
        self._running = False


def _next_settlement(now: datetime) -> datetime:
    """Compute the next 00:00/08:00/16:00 UTC settlement time."""
    today = now.date()
    for hour in SETTLEMENT_HOURS:
        settle = datetime(
            today.year, today.month, today.day,
            hour, 0, 0, tzinfo=timezone.utc,
        )
        if settle > now:
            return settle
    tomorrow = today + timedelta(days=1)
    return datetime(
        tomorrow.year, tomorrow.month, tomorrow.day,
        0, 0, 0, tzinfo=timezone.utc,
    )
