"""Unified Scanner Report — ONE alert type, every 4 hours, max 5 coins.

Strict data-driven filtering (3 weeks, 5500+ outcomes):
  TIER 1: 4+ event types in 4h → always show (every historical case = big move)
  TIER 2: 3 types + ML >= 50% → show (46% win rate)
  TIER 3: 2 types + ML >= 70% + has VE or CA → show
  Else: filtered out.

VACUUM_BREAK excluded from counting (0% hit rate on 63 events).
Single event types alone excluded (mostly noise).
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..events.definitions import AlertManager
    from ..ml.predictor import FragilityPredictor

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent.parent
REPORT_INTERVAL = 14400  # 4 hours
MAX_COINS = 5

# VB excluded — 0% hit rate
_VALID_TYPES = {
    "FUNDING_SQUEEZE_SETUP", "VOLUME_EXPLOSION",
    "CASCADE_ACTIVE", "OI_SURGE",
}

_SHORT = {
    "FUNDING_SQUEEZE_SETUP": "FS", "VOLUME_EXPLOSION": "VE",
    "CASCADE_ACTIVE": "CA", "OI_SURGE": "OI",
}


class UnifiedReport:
    def __init__(
        self,
        scanner: Any,
        predictor: Optional["FragilityPredictor"] = None,
        alert_manager: Optional["AlertManager"] = None,
        db_path: str = "data/events.db",
    ) -> None:
        self.scanner = scanner
        self.predictor = predictor
        self.alert_manager = alert_manager
        self.db_path = str(_ROOT / db_path) if not db_path.startswith("/") else db_path
        self._running = False

    async def run_loop(self) -> None:
        self._running = True
        logger.info("UnifiedReport started (every %ds)", REPORT_INTERVAL)
        while self._running:
            try:
                await self._generate_and_send()
            except Exception:
                logger.error("UnifiedReport error", exc_info=True)
            await asyncio.sleep(REPORT_INTERVAL)

    async def _generate_and_send(self) -> None:
        candidates = self._gather_candidates()
        if not candidates:
            logger.debug("UnifiedReport: 0 coins passed filters")
            return
        now = datetime.now(timezone.utc)
        msg = self._format(candidates, now)
        if msg and self.alert_manager:
            await self.alert_manager._dispatch_telegram(msg, parse_mode="")
            logger.info("UnifiedReport sent (%d coins)", len(candidates))

    # ------------------------------------------------------------------
    # Candidate gathering
    # ------------------------------------------------------------------

    def _gather_candidates(self) -> List[Dict[str, Any]]:
        states = self._get_states()
        if not states:
            return []

        events_by_sym = self._query_events()
        smc_by_sym = self._query_smc()

        # Settlement countdown
        now = datetime.now(timezone.utc)
        cur = now.hour * 60 + now.minute
        mins_to_settle = min((s * 60 - cur) % 1440 for s in (0, 8, 16, 24))

        candidates: List[Dict[str, Any]] = []

        for sym, events in events_by_sym.items():
            state = states.get(sym)
            if not state:
                continue

            types_set = {e["event_type"] for e in events if e["event_type"] in _VALID_TYPES}
            num_types = len(types_set)
            max_score = max((e["score"] for e in events), default=0)

            # ML prediction
            ml_prob = 0.0
            if self.predictor and self.predictor.is_loaded:
                try:
                    feats = self.predictor.build_features_from_state(state, events)
                    ml_prob = self.predictor.predict(feats) or 0.0
                except Exception:
                    pass

            # ── Strict filter ──
            tier = None
            if num_types >= 4:
                tier = 1
            elif num_types >= 3 and ml_prob >= 0.50:
                tier = 2
            elif (num_types >= 2 and ml_prob >= 0.70
                  and types_set & {"VOLUME_EXPLOSION", "CASCADE_ACTIVE"}):
                tier = 3

            if tier is None:
                continue

            coin = self._build_coin(
                sym, state, events, sorted(types_set), num_types,
                max_score, ml_prob, tier, smc_by_sym.get(sym), mins_to_settle,
            )
            candidates.append(coin)

        candidates.sort(key=lambda c: (c["tier"], -c["num_types"], -c["max_score"]))
        return candidates[:MAX_COINS]

    # ------------------------------------------------------------------
    # DB queries
    # ------------------------------------------------------------------

    def _query_events(self) -> Dict[str, List[Dict]]:
        by_sym: Dict[str, List[Dict]] = {}
        try:
            with sqlite3.connect(self.db_path) as con:
                con.row_factory = sqlite3.Row
                rows = con.execute("""
                    SELECT symbol, event_type, score, direction
                    FROM events
                    WHERE timestamp > datetime('now', '-4 hours')
                      AND score >= 50
                      AND event_type != 'VACUUM_BREAK'
                """).fetchall()
            for r in rows:
                by_sym.setdefault(r["symbol"], []).append(dict(r))
        except Exception:
            logger.error("Event query error", exc_info=True)
        return by_sym

    def _query_smc(self) -> Dict[str, Dict]:
        by_sym: Dict[str, Dict] = {}
        try:
            with sqlite3.connect(self.db_path) as con:
                con.row_factory = sqlite3.Row
                rows = con.execute("""
                    SELECT symbol, direction, entry_price, stop_loss,
                           take_profit, risk_reward, sweep_level,
                           displacement_pct, has_fvg
                    FROM smc_setups
                    WHERE timestamp > datetime('now', '-4 hours')
                      AND confidence >= 60
                    ORDER BY confidence DESC
                """).fetchall()
            for r in rows:
                sym = r["symbol"]
                if sym not in by_sym:
                    by_sym[sym] = dict(r)
        except Exception:
            pass  # table may not exist yet
        return by_sym

    # ------------------------------------------------------------------
    # Build coin data
    # ------------------------------------------------------------------

    def _build_coin(
        self, symbol, state, events, event_types, num_types,
        max_score, ml_prob, tier, smc, mins_to_settle,
    ) -> Dict[str, Any]:
        ff = getattr(state, "funding_feats", None) or {}
        oi = getattr(state, "oi_feats", None) or {}
        ob = getattr(state, "ob_feats", None) or {}
        vol = getattr(state, "vol_feats", None) or {}

        fr = float(ff.get("funding_current", 0) or 0)
        afr = abs(fr)
        if afr > 0.005:
            fr_label = "\u26a0\ufe0f extreme"
        elif afr > 0.001:
            fr_label = "high"
        elif afr > 0.0005:
            fr_label = "elevated"
        else:
            fr_label = "neutral"

        if fr < -0.0001:
            fr_side = "shorts paying"
        elif fr > 0.0001:
            fr_side = "longs paying"
        else:
            fr_side = "balanced"

        oi_1h = float(oi.get("oi_pct_1h", 0) or 0)
        oi_cur = float(oi.get("oi_current", 0) or 0)
        mid = float(ob.get("mid_price", 0) or 0)
        oi_usd = oi_cur * mid

        thin = float(ob.get("thin_pct", 0) or 0)
        d_ask = float(ob.get("depth_ask_usdt", 0) or 0)
        vac_a = float(ob.get("vacuum_dist_ask", 0) or 0)
        vac_b = float(ob.get("vacuum_dist_bid", 0) or 0)
        vacuum = "EMPTY" if (vac_a > 5000 or vac_b > 5000) else ("THIN" if thin > 0.90 else "ok")

        bb = float(vol.get("bb_width_pct", 0) or 0)

        # Volume ratio from rt_vol_1s
        vol_ratio = 1.0
        rt = getattr(state, "rt_vol_1s", None)
        if rt and len(rt) > 300:
            recent = list(rt)
            r_sum = sum(recent[-300:])
            bl_n = len(recent) - 300
            if bl_n > 0:
                bl_sum = sum(recent[:-300]) / (bl_n / 300)
                vol_ratio = r_sum / bl_sum if bl_sum > 0 else 1.0

        # Direction from event majority
        dirs = [e.get("direction", "") for e in events]
        ups = sum(1 for d in dirs if d in ("LONG", "UP"))
        dns = sum(1 for d in dirs if d in ("SHORT", "DN"))
        direction = "LONG" if ups >= dns else "SHORT"

        return {
            "symbol": symbol, "direction": direction,
            "event_types": event_types, "num_types": num_types,
            "max_score": max_score, "ml_prob": ml_prob, "tier": tier,
            "fr": fr, "fr_label": fr_label, "fr_side": fr_side,
            "oi_1h": oi_1h, "oi_usd": oi_usd,
            "thin": thin, "vacuum": vacuum, "depth_ask": d_ask,
            "bb": bb, "vol_ratio": vol_ratio, "price": mid,
            "smc": smc, "mins_to_settle": mins_to_settle,
        }

    # ------------------------------------------------------------------
    # Format
    # ------------------------------------------------------------------

    def _format(self, candidates: List[Dict], now: datetime) -> str:
        lines: List[str] = []

        # Header
        btc_str = ""
        states = self._get_states()
        btc = states.get("BTCUSDT")
        if btc:
            ob = getattr(btc, "ob_feats", None) or {}
            p = ob.get("mid_price", 0)
            if p:
                btc_str = f" | BTC: ${p:,.0f}"

        lines.append(
            f"\U0001f4ca SCANNER REPORT | {now:%H:%M} UTC{btc_str}"
        )

        # Settlement warning
        settle = candidates[0]["mins_to_settle"]
        if settle <= 60:
            s_hour = min(
                (s for s in (0, 8, 16, 24) if (s * 60 - (now.hour * 60 + now.minute)) % 1440 == settle),
                default=0,
            ) % 24
            lines.append(f"\u26a0\ufe0f Settlement {s_hour:02d}:00 in {settle}min")

        lines.append("")

        _TIER_LABEL = {1: "\U0001f534 TIER 1", 2: "\U0001f7e1 TIER 2", 3: "\U0001f7e2 TIER 3"}

        for i, c in enumerate(candidates, 1):
            de = "\U0001f7e2" if c["direction"] == "LONG" else "\U0001f534"
            ts = "+".join(_SHORT.get(t, t[:2]) for t in c["event_types"])

            lines.append(f"\u2501\u2501\u2501 {i}. {c['symbol']} \u2501\u2501\u2501")
            lines.append(
                f"{de} {c['direction']} | {ts} ({c['num_types']}/4) | "
                f"Score: {c['max_score']:.0f} | {_TIER_LABEL[c['tier']]}"
            )
            lines.append("")

            # Funding
            lines.append(
                f"Funding: {c['fr']*100:+.3f}% {c['fr_label']} ({c['fr_side']})"
            )

            # OI
            oi_parts = [f"OI: {c['oi_1h']:+.0f}% (1h)"]
            if c["oi_usd"] > 0:
                oi_parts.append(f"${c['oi_usd']/1e6:.1f}M total")
            lines.append(" | ".join(oi_parts))

            # Book
            book = []
            if c["thin"] > 0.80:
                book.append(f"Thin: {c['thin']:.2f}")
            if c["vacuum"] != "ok":
                book.append(c["vacuum"])
            if c["depth_ask"] > 0:
                book.append(f"Depth: ${c['depth_ask']/1e3:.0f}K")
            lines.append(f"Book: {' | '.join(book) if book else 'normal'}")

            # BB + Vol
            extra = []
            if c["bb"] > 0:
                tag = " (compressed)" if c["bb"] < 3 else ""
                extra.append(f"BB: {c['bb']:.1f}%{tag}")
            if c["vol_ratio"] > 1.5:
                extra.append(f"Vol: {c['vol_ratio']:.1f}x")
            if extra:
                lines.append(" | ".join(extra))

            # SMC
            smc = c.get("smc")
            if smc and smc.get("entry_price"):
                lines.append("")
                lines.append(
                    f"SMC: Sweep {smc['sweep_level']:.6g} "
                    f"\u2192 Disp {smc['displacement_pct']:.1f}%"
                )
                lines.append(
                    f"     Entry {smc['entry_price']:.6g} | "
                    f"SL {smc['stop_loss']:.6g} | "
                    f"TP {smc['take_profit']:.6g}"
                )
                lines.append(f"     R:R 1:{smc['risk_reward']:.1f}")

            lines.append(f"ML: {c['ml_prob']:.0%}")
            lines.append("")

        lines.append(
            f"Scanned {len(states)} coins | "
            f"{len(candidates)} passed filters"
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------

    def _get_states(self):
        return getattr(self.scanner, "_states", {})

    def stop(self) -> None:
        self._running = False
