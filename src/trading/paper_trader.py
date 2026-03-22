"""Paper Trading Engine — Simulated trades based on scanner signals.

Takes hypothetical positions when ML probability + event signals align.
Tracks P&L, win rate, and entry quality for learning.

No real money. No exchange API. Uses live mid-price from SymbolState.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..events.definitions import AlertManager
    from ..ml.predictor import FragilityPredictor
    from ..scanner.pressure_scanner import PressureScanner

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class TradeStatus(str, Enum):
    OPEN = "OPEN"
    CLOSED_TP = "CLOSED_TP"
    CLOSED_SL = "CLOSED_SL"
    CLOSED_TIME = "CLOSED_TIME"
    CLOSED_MANUAL = "CLOSED_MANUAL"


@dataclass
class PaperTrade:
    id: str
    symbol: str
    direction: str              # LONG or SHORT
    entry_price: float
    entry_time: str

    position_size_usd: float = 1000.0

    # Risk management
    stop_loss_pct: float = 3.0
    take_profit_pct: float = 8.0
    max_hold_hours: int = 8

    # Entry signals
    ml_probability: float = 0.0
    event_types: str = ""
    num_event_types: int = 0
    event_score: float = 0.0
    entry_trigger: str = ""

    # Market context at entry
    funding_rate: float = 0.0
    oi_change_pct: float = 0.0
    thin_pct: float = 0.0
    btc_change_4h: float = 0.0
    bb_width_pct: float = 0.0

    # Tracking (updated in real-time)
    current_price: float = 0.0
    unrealized_pnl_pct: float = 0.0
    max_favorable_pct: float = 0.0
    max_adverse_pct: float = 0.0

    # Exit
    exit_price: float = 0.0
    exit_time: str = ""
    exit_reason: str = ""
    realized_pnl_pct: float = 0.0
    realized_pnl_usd: float = 0.0
    hold_duration_minutes: int = 0

    status: str = "OPEN"


# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS paper_trades (
    id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL,
    entry_price REAL,
    entry_time TEXT,
    position_size_usd REAL,
    stop_loss_pct REAL,
    take_profit_pct REAL,
    max_hold_hours INTEGER,
    ml_probability REAL,
    event_types TEXT,
    num_event_types INTEGER,
    event_score REAL,
    entry_trigger TEXT,
    funding_rate REAL,
    oi_change_pct REAL,
    thin_pct REAL,
    btc_change_4h REAL,
    bb_width_pct REAL,
    max_favorable_pct REAL,
    max_adverse_pct REAL,
    exit_price REAL,
    exit_time TEXT,
    exit_reason TEXT,
    realized_pnl_pct REAL,
    realized_pnl_usd REAL,
    hold_duration_minutes INTEGER,
    status TEXT DEFAULT 'OPEN',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS paper_daily_stats (
    date TEXT PRIMARY KEY,
    trades_opened INTEGER,
    trades_closed INTEGER,
    wins INTEGER,
    losses INTEGER,
    total_pnl_pct REAL,
    total_pnl_usd REAL,
    avg_win_pct REAL,
    avg_loss_pct REAL,
    best_trade_pct REAL,
    worst_trade_pct REAL,
    avg_hold_minutes REAL,
    win_rate REAL
);
"""

_UPSERT_SQL = """
INSERT OR REPLACE INTO paper_trades (
    id, symbol, direction, entry_price, entry_time,
    position_size_usd, stop_loss_pct, take_profit_pct, max_hold_hours,
    ml_probability, event_types, num_event_types, event_score, entry_trigger,
    funding_rate, oi_change_pct, thin_pct, btc_change_4h, bb_width_pct,
    max_favorable_pct, max_adverse_pct,
    exit_price, exit_time, exit_reason,
    realized_pnl_pct, realized_pnl_usd, hold_duration_minutes,
    status
) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
"""


# ---------------------------------------------------------------------------
# PaperTrader
# ---------------------------------------------------------------------------

class PaperTrader:
    """Manages hypothetical trades based on scanner signals."""

    def __init__(
        self,
        scanner: "PressureScanner",
        predictor: Optional["FragilityPredictor"] = None,
        alert_manager: Optional["AlertManager"] = None,
        db_path: str = "data/events.db",
    ) -> None:
        self.scanner = scanner
        self.predictor = predictor
        self.alert_manager = alert_manager
        self.db_path = str(_ROOT / db_path) if not db_path.startswith("/") else db_path

        self.open_trades: Dict[str, PaperTrade] = {}   # symbol -> trade
        self.max_open_trades = 5
        self._running = False
        self._trade_counter = 0

        # Entry criteria
        self.min_ml_probability = 0.50
        self.min_event_types = 2
        self.min_event_score = 65

        # High-conviction shortcut
        self.high_conviction_prob = 0.70

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Create tables and restore open trades."""
        with sqlite3.connect(self.db_path) as con:
            con.executescript(_CREATE_SQL)
        self._restore_open_trades()
        logger.info("PaperTrader initialized (%d open trades)", len(self.open_trades))

    async def run_loop(self, interval_seconds: int = 60) -> None:
        """Main loop: update, check exits, check entries every 60 s."""
        self._running = True
        logger.info("PaperTrader started (every %ds)", interval_seconds)

        while self._running:
            try:
                self._update_open_trades()
                await self._check_exits()
                await self._check_entries()
            except Exception:
                logger.error("PaperTrader cycle error", exc_info=True)

            await asyncio.sleep(interval_seconds)

    def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------
    # State access helpers
    # ------------------------------------------------------------------

    def _get_states(self) -> Dict[str, Any]:
        return getattr(self.scanner, "_states", {})

    def _mid_price(self, state: Any) -> Optional[float]:
        ob = getattr(state, "ob_feats", None) or {}
        mid = ob.get("mid_price")
        if mid and mid > 0:
            return float(mid)
        return None

    # ------------------------------------------------------------------
    # Entry logic
    # ------------------------------------------------------------------

    async def _check_entries(self) -> None:
        if len(self.open_trades) >= self.max_open_trades:
            return

        states = self._get_states()
        if not states:
            return

        recent_events = self._get_recent_events()

        for symbol, events in recent_events.items():
            if symbol in self.open_trades:
                continue
            if symbol in ("BTCUSDT", "ETHUSDT"):
                continue

            state = states.get(symbol)
            if not state:
                continue

            mid = self._mid_price(state)
            if not mid:
                continue

            # ML prediction
            ml_prob = 0.0
            features: Optional[Dict[str, float]] = None
            if self.predictor and self.predictor.is_loaded:
                features = self.predictor.build_features_from_state(state, events)
                ml_prob = self.predictor.predict(features) or 0.0

            event_types = list({e.get("event_type", "") for e in events})
            num_types = len(event_types)
            max_score = max((e.get("score", 0) or 0 for e in events), default=0)

            should_enter = False
            trigger = ""

            # Condition A: ML high conviction
            if ml_prob >= self.high_conviction_prob and num_types >= 1:
                should_enter = True
                trigger = f"ML_HIGH ({ml_prob:.0%})"

            # Condition B: Multiple event types + decent ML + score
            elif (ml_prob >= self.min_ml_probability
                  and num_types >= self.min_event_types
                  and max_score >= self.min_event_score):
                should_enter = True
                trigger = f"MULTI_SIGNAL ({num_types} types)"

            # Condition C: Volume explosion with ML confirmation
            elif ("VOLUME_EXPLOSION" in event_types
                  and ml_prob >= 0.40
                  and max_score >= 50):
                should_enter = True
                trigger = f"VE_CONFIRMED (ML {ml_prob:.0%})"

            if should_enter:
                await self._open_trade(
                    symbol, state, mid, events, event_types,
                    num_types, max_score, ml_prob, trigger, features,
                )

            if len(self.open_trades) >= self.max_open_trades:
                break

    async def _open_trade(
        self, symbol: str, state: Any, mid: float,
        events: List[Dict], event_types: List[str],
        num_types: int, max_score: float, ml_prob: float,
        trigger: str, features: Optional[Dict[str, float]],
    ) -> None:
        # Direction from funding
        ff = getattr(state, "funding_feats", None) or {}
        funding = float(ff.get("funding_current", 0) or 0)

        if funding < -0.0003:
            direction = "LONG"
        elif funding > 0.0003:
            direction = "SHORT"
        else:
            direction = "LONG"
            for ev in events:
                d = ev.get("direction", "")
                if d in ("LONG", "UP"):
                    direction = "LONG"
                    break
                elif d in ("SHORT", "DN"):
                    direction = "SHORT"
                    break

        # Dynamic SL/TP
        if ml_prob >= 0.70:
            sl, tp = 2.5, 10.0
        elif num_types >= 3:
            sl, tp = 3.0, 10.0
        else:
            sl, tp = 3.0, 8.0

        self._trade_counter += 1
        trade_id = (
            f"PT_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}"
            f"_{self._trade_counter}"
        )

        oi_change = 0.0
        thin = 0.0
        bb = 0.0
        btc_4h = 0.0
        if features:
            oi_change = features.get("oi_change_1h_pct", 0) or 0
            thin = features.get("thin_pct", 0) or 0
            bb = features.get("bb_width_pct", 0) or 0
            btc_4h = features.get("btc_change_4h", 0) or 0

        trade = PaperTrade(
            id=trade_id,
            symbol=symbol,
            direction=direction,
            entry_price=mid,
            entry_time=datetime.now(timezone.utc).isoformat(),
            stop_loss_pct=sl,
            take_profit_pct=tp,
            ml_probability=ml_prob,
            event_types=",".join(event_types),
            num_event_types=num_types,
            event_score=max_score,
            entry_trigger=trigger,
            funding_rate=funding,
            oi_change_pct=oi_change,
            thin_pct=thin,
            btc_change_4h=btc_4h,
            bb_width_pct=bb,
            current_price=mid,
        )

        self.open_trades[symbol] = trade
        self._save_trade(trade)

        _short = {
            "FUNDING_SQUEEZE_SETUP": "FS",
            "VACUUM_BREAK": "VB",
            "VOLUME_EXPLOSION": "VE",
            "CASCADE_ACTIVE": "CA",
            "OI_SURGE": "OI",
            "COMPRESSION_SQUEEZE_SETUP": "CS",
        }
        evt_short = ",".join(_short.get(t, t) for t in event_types)

        msg = (
            f"\U0001f4dd PAPER TRADE OPENED\n"
            f"{'🟢' if direction == 'LONG' else '🔴'} {symbol} {direction}\n"
            f"Entry: {mid}\n"
            f"ML: {ml_prob:.0%} | {trigger}\n"
            f"Events: {evt_short}\n"
            f"SL: {sl}% | TP: {tp}%\n"
            f"Open trades: {len(self.open_trades)}/{self.max_open_trades}"
        )
        await self._send_telegram(msg)

        logger.info(
            "Paper trade opened: %s %s @ %s (ML=%.2f, trigger=%s)",
            symbol, direction, mid, ml_prob, trigger,
        )

    # ------------------------------------------------------------------
    # Price update
    # ------------------------------------------------------------------

    def _update_open_trades(self) -> None:
        states = self._get_states()

        for symbol, trade in self.open_trades.items():
            state = states.get(symbol)
            if not state:
                continue

            mid = self._mid_price(state)
            if not mid:
                continue

            trade.current_price = mid

            if trade.direction == "LONG":
                pnl_pct = ((mid - trade.entry_price) / trade.entry_price) * 100
            else:
                pnl_pct = ((trade.entry_price - mid) / trade.entry_price) * 100

            trade.unrealized_pnl_pct = pnl_pct

            if pnl_pct > trade.max_favorable_pct:
                trade.max_favorable_pct = pnl_pct
            if pnl_pct < trade.max_adverse_pct:
                trade.max_adverse_pct = pnl_pct

    # ------------------------------------------------------------------
    # Exit logic
    # ------------------------------------------------------------------

    async def _check_exits(self) -> None:
        to_close: List[tuple] = []

        for symbol, trade in self.open_trades.items():
            reason = None

            if trade.unrealized_pnl_pct >= trade.take_profit_pct:
                reason = TradeStatus.CLOSED_TP
            elif trade.unrealized_pnl_pct <= -trade.stop_loss_pct:
                reason = TradeStatus.CLOSED_SL
            else:
                try:
                    entry_dt = datetime.fromisoformat(trade.entry_time)
                    held_min = (datetime.now(timezone.utc) - entry_dt).total_seconds() / 60
                    if held_min > trade.max_hold_hours * 60:
                        reason = TradeStatus.CLOSED_TIME
                except Exception:
                    pass

            if reason:
                to_close.append((symbol, reason))

        for symbol, reason in to_close:
            await self._close_trade(symbol, reason)

    async def _close_trade(self, symbol: str, reason: TradeStatus) -> None:
        trade = self.open_trades.get(symbol)
        if not trade:
            return

        now = datetime.now(timezone.utc)

        trade.exit_price = trade.current_price
        trade.exit_time = now.isoformat()
        trade.exit_reason = reason.value
        trade.realized_pnl_pct = trade.unrealized_pnl_pct
        trade.realized_pnl_usd = trade.position_size_usd * (trade.realized_pnl_pct / 100)
        trade.status = reason.value

        try:
            entry_dt = datetime.fromisoformat(trade.entry_time)
            trade.hold_duration_minutes = int((now - entry_dt).total_seconds() / 60)
        except Exception:
            trade.hold_duration_minutes = 0

        self._save_trade(trade)
        del self.open_trades[symbol]

        _reason_text = {
            "CLOSED_TP": "\U0001f3af Take Profit",
            "CLOSED_SL": "\U0001f6d1 Stop Loss",
            "CLOSED_TIME": "\u23f0 Time Expired",
        }
        pnl_emoji = "\u2705" if trade.realized_pnl_pct > 0 else "\u274c"

        msg = (
            f"{pnl_emoji} PAPER TRADE CLOSED\n"
            f"{trade.symbol} {trade.direction}\n"
            f"Entry: {trade.entry_price} \u2192 Exit: {trade.exit_price}\n"
            f"P&L: {trade.realized_pnl_pct:+.1f}% (${trade.realized_pnl_usd:+.1f})\n"
            f"Reason: {_reason_text.get(reason.value, reason.value)}\n"
            f"Held: {trade.hold_duration_minutes}min\n"
            f"Max fav: {trade.max_favorable_pct:+.1f}% | "
            f"Max adv: {trade.max_adverse_pct:+.1f}%\n"
            f"Trigger: {trade.entry_trigger}"
        )
        await self._send_telegram(msg)

        logger.info(
            "Paper trade closed: %s %s P&L=%+.1f%%",
            symbol, reason.value, trade.realized_pnl_pct,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_trade(self, t: PaperTrade) -> None:
        with sqlite3.connect(self.db_path) as con:
            con.execute(_UPSERT_SQL, (
                t.id, t.symbol, t.direction,
                t.entry_price, t.entry_time,
                t.position_size_usd, t.stop_loss_pct,
                t.take_profit_pct, t.max_hold_hours,
                t.ml_probability, t.event_types,
                t.num_event_types, t.event_score, t.entry_trigger,
                t.funding_rate, t.oi_change_pct,
                t.thin_pct, t.btc_change_4h, t.bb_width_pct,
                t.max_favorable_pct, t.max_adverse_pct,
                t.exit_price, t.exit_time, t.exit_reason,
                t.realized_pnl_pct, t.realized_pnl_usd,
                t.hold_duration_minutes, t.status,
            ))
            con.commit()

    def _restore_open_trades(self) -> None:
        with sqlite3.connect(self.db_path) as con:
            con.row_factory = sqlite3.Row
            rows = con.execute(
                "SELECT * FROM paper_trades WHERE status = 'OPEN'"
            ).fetchall()

        for row in rows:
            trade = PaperTrade(
                id=row["id"],
                symbol=row["symbol"],
                direction=row["direction"],
                entry_price=row["entry_price"],
                entry_time=row["entry_time"],
                position_size_usd=row["position_size_usd"],
                stop_loss_pct=row["stop_loss_pct"],
                take_profit_pct=row["take_profit_pct"],
                max_hold_hours=row["max_hold_hours"],
                ml_probability=row["ml_probability"],
                event_types=row["event_types"] or "",
                num_event_types=row["num_event_types"],
                event_score=row["event_score"],
                entry_trigger=row["entry_trigger"] or "",
                funding_rate=row["funding_rate"] or 0,
                oi_change_pct=row["oi_change_pct"] or 0,
                thin_pct=row["thin_pct"] or 0,
                btc_change_4h=row["btc_change_4h"] or 0,
                bb_width_pct=row["bb_width_pct"] or 0,
                max_favorable_pct=row["max_favorable_pct"] or 0,
                max_adverse_pct=row["max_adverse_pct"] or 0,
                status="OPEN",
            )
            self.open_trades[trade.symbol] = trade

        if rows:
            logger.info("Restored %d open paper trades", len(rows))

    # ------------------------------------------------------------------
    # Recent events
    # ------------------------------------------------------------------

    def _get_recent_events(self) -> Dict[str, List[Dict[str, Any]]]:
        """Events from last 2h grouped by symbol."""
        by_symbol: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        try:
            with sqlite3.connect(self.db_path) as con:
                con.row_factory = sqlite3.Row
                rows = con.execute("""
                    SELECT symbol, event_type, score, direction
                    FROM events
                    WHERE timestamp > datetime('now', '-2 hours')
                      AND score >= 50
                """).fetchall()
            for row in rows:
                by_symbol[row["symbol"]].append(dict(row))
        except Exception:
            logger.error("Event fetch error", exc_info=True)
        return by_symbol

    # ------------------------------------------------------------------
    # Telegram
    # ------------------------------------------------------------------

    async def _send_telegram(self, msg: str) -> None:
        if self.alert_manager:
            try:
                await self.alert_manager._dispatch_telegram(msg, parse_mode="")
            except Exception:
                logger.debug("PaperTrader telegram error", exc_info=True)


# ---------------------------------------------------------------------------
# Daily report
# ---------------------------------------------------------------------------

class PaperTradeReporter:
    """Generate daily paper trading summary."""

    def __init__(self, db_path: str = "data/events.db") -> None:
        self.db_path = str(_ROOT / db_path) if not db_path.startswith("/") else db_path

    def generate_daily_report(self) -> str:
        with sqlite3.connect(self.db_path) as con:
            con.row_factory = sqlite3.Row

            closed = [
                dict(r) for r in con.execute("""
                    SELECT * FROM paper_trades
                    WHERE status != 'OPEN'
                      AND exit_time > datetime('now', '-24 hours')
                    ORDER BY exit_time DESC
                """).fetchall()
            ]

            open_trades = [
                dict(r) for r in con.execute(
                    "SELECT * FROM paper_trades WHERE status = 'OPEN'"
                ).fetchall()
            ]

            alltime = dict(con.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN realized_pnl_pct > 0 THEN 1 ELSE 0 END) as wins,
                    ROUND(AVG(realized_pnl_pct), 2) as avg_pnl,
                    ROUND(SUM(realized_pnl_usd), 2) as total_pnl_usd,
                    ROUND(AVG(CASE WHEN realized_pnl_pct > 0
                              THEN realized_pnl_pct END), 2) as avg_win,
                    ROUND(AVG(CASE WHEN realized_pnl_pct <= 0
                              THEN realized_pnl_pct END), 2) as avg_loss
                FROM paper_trades
                WHERE status != 'OPEN'
            """).fetchone())

        if not closed and not open_trades:
            return ""

        total = alltime.get("total", 0) or 0
        wins = alltime.get("wins", 0) or 0
        win_rate = (wins / total * 100) if total > 0 else 0

        lines = ["\U0001f4c8 PAPER TRADE REPORT (24h)\n"]
        lines.append(f"All-time: {wins}/{total} wins ({win_rate:.0f}%)")
        lines.append(f"Total P&L: ${alltime.get('total_pnl_usd', 0) or 0:+.1f}")
        lines.append(
            f"Avg win: {alltime.get('avg_win', 0) or 0:+.1f}% | "
            f"Avg loss: {alltime.get('avg_loss', 0) or 0:.1f}%"
        )

        if closed:
            lines.append(f"\nClosed today: {len(closed)}")
            for t in closed[:5]:
                emoji = "\u2705" if t["realized_pnl_pct"] > 0 else "\u274c"
                lines.append(
                    f"  {emoji} {t['symbol']} {t['direction']} "
                    f"{t['realized_pnl_pct']:+.1f}% ({t['entry_trigger']})"
                )

        if open_trades:
            lines.append(f"\nOpen: {len(open_trades)}")
            for t in open_trades:
                lines.append(
                    f"  {t['symbol']} {t['direction']} "
                    f"MFE: {t.get('max_favorable_pct', 0) or 0:+.1f}%"
                )

        return "\n".join(lines)
