"""ML-ranked Conviction Digest — periodic Telegram summary of top-probability coins.

Uses FragilityPredictor to score all active symbols and sends a ranked digest
every 2 hours via the existing AlertManager's Telegram dispatch.
"""

from __future__ import annotations

import asyncio
import logging
import math
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..events.definitions import AlertManager
    from ..scanner.pressure_scanner import SymbolState

from .predictor import FragilityPredictor

logger = logging.getLogger(__name__)

_SHORT = {
    "FUNDING_SQUEEZE_SETUP": "FS",
    "VACUUM_BREAK": "VB",
    "VOLUME_EXPLOSION": "VE",
    "CASCADE_ACTIVE": "CA",
    "OI_SURGE": "OI",
    "COMPRESSION_SQUEEZE_SETUP": "CS",
}


class ConvictionDigest:
    """Score all coins with ML model and send top-N as Telegram digest."""

    def __init__(
        self,
        predictor: FragilityPredictor,
        alert_manager: "AlertManager",
        states_fn: Callable[[], Dict[str, "SymbolState"]],
        recent_events_fn: Optional[Callable[[str], List[Dict[str, Any]]]] = None,
        btc_price_fn: Optional[Callable[[], Optional[tuple]]] = None,
        interval_s: int = 7200,  # 2 hours
        top_n: int = 5,
        min_prob: float = 0.60,
    ) -> None:
        self.predictor = predictor
        self.alert_manager = alert_manager
        self._states_fn = states_fn
        self._recent_events_fn = recent_events_fn
        self._btc_price_fn = btc_price_fn
        self.interval_s = interval_s
        self.top_n = top_n
        self.min_prob = min_prob
        self._task: Optional[asyncio.Task] = None

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.ensure_future(self._loop())
            logger.info("ConvictionDigest started (every %ds)", self.interval_s)

    def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(self.interval_s)
            try:
                msg = self._build_digest()
                if msg:
                    await self.alert_manager._dispatch_telegram(msg)
                    logger.info("Conviction digest sent")
            except Exception as exc:
                logger.error("Conviction digest error: %r", exc)

    def _build_digest(self) -> Optional[str]:
        if not self.predictor.is_loaded:
            return None

        states = self._states_fn()
        if not states:
            return None

        # Score all symbols
        scored: List[tuple] = []  # (prob, symbol, state, features, events)
        for sym, state in states.items():
            if state.rank is None or state.rank < 10:
                continue

            recent: List[Dict[str, Any]] = []
            if self._recent_events_fn:
                try:
                    recent = self._recent_events_fn(sym)
                except Exception:
                    pass

            features = self.predictor.build_features_from_state(state, recent)
            prob = self.predictor.predict(features)
            if prob is not None and prob >= self.min_prob:
                scored.append((prob, sym, state, features, recent))

        if not scored:
            return None

        scored.sort(key=lambda x: -x[0])
        top = scored[: self.top_n]

        now = datetime.now(timezone.utc)

        # BTC regime header
        btc_str = ""
        if self._btc_price_fn:
            try:
                result = self._btc_price_fn()
                if result:
                    pct, price = result
                    arrow = "\U0001f7e2" if pct >= 0 else "\U0001f534"
                    btc_str = f" | BTC: {pct:+.1f}% {arrow}"
            except Exception:
                pass

        lines = [
            f"\U0001f3af <b>HIGH CONVICTION</b> | {now:%H:%M} UTC{btc_str}",
            "",
        ]

        for i, (prob, sym, state, feats, evts) in enumerate(top, 1):
            fund_z = feats.get("funding_z", 0)
            direction = "DN" if fund_z > 0 else "UP"
            dir_dot = "\U0001f7e2" if direction == "UP" else "\U0001f534"

            # Event types
            etypes = list({_SHORT.get(e.get("event_type", ""), e.get("event_type", ""))
                          for e in evts})
            n_types = len(etypes) if etypes else 0
            evt_str = "+".join(sorted(etypes)) if etypes else "-"
            type_str = f" ({n_types} types)" if n_types >= 2 else ""

            # Line 1: symbol, direction, probability, events
            lines.append(
                f"{i}. <b>{sym}</b> {dir_dot} {direction}"
                f" | {prob:.0%} | {evt_str}{type_str}"
            )

            # Line 2: key data
            ff = getattr(state, "funding_feats", None) or {}
            oi = getattr(state, "oi_feats", None) or {}
            ob = getattr(state, "ob_feats", None) or {}

            fr = ff.get("funding_current", 0) or 0
            fr_pct = fr * 100

            oi_1h = oi.get("oi_delta_pct_1h", 0) or 0

            thin = ob.get("thin_pct", 0) or 0
            if thin > 0.95:
                book_str = "EMPTY"
            elif thin > 0.85:
                book_str = f"Thin: {thin:.2f}"
            else:
                book_str = "ok"

            lines.append(
                f"   FR: {fr_pct:+.2f}%"
                f" | OI: {oi_1h:+.0f}% 1h"
                f" | Book: {book_str}"
            )

        total_above = sum(1 for p, *_ in scored if p >= self.predictor.threshold)
        lines.append(f"\n{len(scored)} coins \u2265 {self.min_prob:.0%}"
                     f" | {total_above} above threshold")

        return "\n".join(lines)
