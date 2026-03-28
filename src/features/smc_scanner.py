"""SMC Scanner — Runs SMC detection on fragile coins.

Scans coins with active events (score >= 60) in last 2h.
When an SMC setup is found, sends Telegram alert via AlertManager.
All setups saved to smc_setups table for outcome analysis.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import time as _time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .klines_5m import KlineCollector5m
from .market_structure import detect_smc_setup, SMCSetup

if TYPE_CHECKING:
    from ..events.definitions import AlertManager
    from ..ml.predictor import FragilityPredictor

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent.parent

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS smc_setups (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    symbol TEXT,
    direction TEXT,
    sweep_level REAL,
    displacement_pct REAL,
    has_fvg INTEGER,
    fvg_top REAL,
    fvg_bottom REAL,
    entry_price REAL,
    stop_loss REAL,
    take_profit REAL,
    risk_reward REAL,
    confidence REAL,
    ml_probability REAL,
    outcome_1h_pct REAL,
    outcome_4h_pct REAL,
    hit_target INTEGER,
    hit_stop INTEGER
)
"""


class SMCScanner:
    """Scans fragile coins for SMC entry setups on 5m klines."""

    def __init__(
        self,
        kline_collector: KlineCollector5m,
        scanner: Any = None,
        predictor: Optional["FragilityPredictor"] = None,
        alert_manager: Optional["AlertManager"] = None,
        db_path: str = "data/events.db",
    ) -> None:
        self.klines = kline_collector
        self.scanner = scanner
        self.predictor = predictor
        self.alert_manager = alert_manager
        self.db_path = str(_ROOT / db_path) if not db_path.startswith("/") else db_path
        self._running = False
        self._recent_setups: Dict[str, float] = {}  # symbol -> monotonic ts
        self._cooldown = 3600  # 1h per symbol

    def initialize(self) -> None:
        with sqlite3.connect(self.db_path) as con:
            con.execute(_CREATE_TABLE)
            con.commit()
        logger.info("SMCScanner: smc_setups table ready")

    async def run_loop(self, interval_seconds: int = 60) -> None:
        self._running = True
        logger.info("SMCScanner started (every %ds)", interval_seconds)
        while self._running:
            try:
                await self._scan()
            except Exception:
                logger.error("SMCScanner error", exc_info=True)
            await asyncio.sleep(interval_seconds)

    async def _scan(self) -> None:
        active = self._get_active_symbols()
        if not active:
            return

        self.klines.set_active_symbols(active)

        now = _time.monotonic()
        for symbol in active:
            if now - self._recent_setups.get(symbol, 0) < self._cooldown:
                continue

            kline_data = self.klines.get_klines(symbol, count=100)
            if len(kline_data) < 30:
                continue

            setup = detect_smc_setup(symbol, kline_data)
            if not setup or setup.confidence < 50:
                continue

            # ML context
            ml_prob = self._get_ml_prob(symbol)

            await self._send_alert(setup, ml_prob)
            self._save_setup(setup, ml_prob)
            self._recent_setups[symbol] = _time.monotonic()

    # ── Helpers ────────────────────────────────────────────────────────────

    def _get_active_symbols(self) -> set:
        symbols: set = set()
        try:
            with sqlite3.connect(self.db_path) as con:
                rows = con.execute("""
                    SELECT DISTINCT symbol FROM events
                    WHERE timestamp > datetime('now', '-2 hours')
                      AND score >= 60
                """).fetchall()
            for r in rows:
                symbols.add(r[0])
        except Exception:
            logger.debug("Active symbols query error", exc_info=True)
        return symbols

    def _get_ml_prob(self, symbol: str) -> Optional[float]:
        if not self.predictor or not self.predictor.is_loaded or not self.scanner:
            return None
        states = getattr(self.scanner, "_states", {})
        state = states.get(symbol)
        if not state:
            return None
        try:
            events = self._get_recent_events(symbol)
            features = self.predictor.build_features_from_state(state, events)
            return self.predictor.predict(features)
        except Exception:
            return None

    def _get_recent_events(self, symbol: str) -> List[Dict[str, Any]]:
        try:
            with sqlite3.connect(self.db_path) as con:
                con.row_factory = sqlite3.Row
                rows = con.execute("""
                    SELECT event_type, score, direction FROM events
                    WHERE symbol = ? AND timestamp > datetime('now', '-2 hours')
                """, (symbol,)).fetchall()
            return [dict(r) for r in rows]
        except Exception:
            return []

    async def _send_alert(self, setup: SMCSetup, ml_prob: Optional[float]) -> None:
        """Save to DB only — UnifiedReport handles Telegram."""
        self._save_setup(setup, ml_prob)
        logger.info(
            "SMC setup saved: %s %s entry=%.6g RR=1:%.1f conf=%d",
            setup.symbol, setup.direction, setup.entry_price,
            setup.risk_reward, setup.confidence,
        )

    def _save_setup(self, setup: SMCSetup, ml_prob: Optional[float]) -> None:
        try:
            with sqlite3.connect(self.db_path) as con:
                con.execute("""
                    INSERT INTO smc_setups (
                        timestamp, symbol, direction, sweep_level,
                        displacement_pct, has_fvg, fvg_top, fvg_bottom,
                        entry_price, stop_loss, take_profit,
                        risk_reward, confidence, ml_probability
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    datetime.now(timezone.utc).isoformat(),
                    setup.symbol, setup.direction, setup.sweep_level,
                    setup.displacement_size_pct,
                    1 if setup.fvg else 0,
                    setup.fvg.top if setup.fvg else None,
                    setup.fvg.bottom if setup.fvg else None,
                    setup.entry_price, setup.stop_loss, setup.take_profit,
                    setup.risk_reward, setup.confidence, ml_prob,
                ))
                con.commit()
        except Exception:
            logger.error("SMC save error", exc_info=True)

    def stop(self) -> None:
        self._running = False
