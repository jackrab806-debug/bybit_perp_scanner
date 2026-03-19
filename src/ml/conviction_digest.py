"""ML-ranked Conviction Digest — periodic Telegram summary of top-probability coins.

Uses FragilityPredictor to score all active symbols and sends a ranked digest
every 2 hours via the existing AlertManager's Telegram dispatch.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..events.definitions import AlertManager
    from ..scanner.pressure_scanner import SymbolState

from .predictor import FragilityPredictor

logger = logging.getLogger(__name__)


class ConvictionDigest:
    """Score all coins with ML model and send top-N as Telegram digest."""

    def __init__(
        self,
        predictor: FragilityPredictor,
        alert_manager: "AlertManager",
        states_fn: Callable[[], Dict[str, "SymbolState"]],
        recent_events_fn: Optional[Callable[[str], List[Dict[str, Any]]]] = None,
        interval_s: int = 7200,  # 2 hours
        top_n: int = 5,
        min_prob: float = 0.60,
    ) -> None:
        self.predictor = predictor
        self.alert_manager = alert_manager
        self._states_fn = states_fn
        self._recent_events_fn = recent_events_fn
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
                logger.error("Conviction digest error: %s", exc)

    def _build_digest(self) -> Optional[str]:
        if not self.predictor.is_loaded:
            return None

        states = self._states_fn()
        if not states:
            return None

        # Score all symbols
        scored: List[tuple] = []  # (prob, symbol, state, features)
        for sym, state in states.items():
            if state.rank is None or state.rank < 10:
                continue  # skip very inactive coins

            recent = []
            if self._recent_events_fn:
                try:
                    recent = self._recent_events_fn(sym)
                except Exception:
                    pass

            features = self.predictor.build_features_from_state(state, recent)
            prob = self.predictor.predict(features)
            if prob is not None and prob >= self.min_prob:
                scored.append((prob, sym, state, features))

        if not scored:
            return None

        # Sort by probability descending
        scored.sort(key=lambda x: -x[0])
        top = scored[:self.top_n]

        # Format message
        now = datetime.now(timezone.utc)
        thr = self.predictor.threshold

        lines = [
            f"<b>\U0001f52e ML CONVICTION ({now:%H:%M} UTC)</b>",
            f"Model: AUC={self.predictor.metrics.get('auc', 0):.3f} | thr={thr:.2f}",
            "",
        ]

        for i, (prob, sym, state, feats) in enumerate(top, 1):
            # Conviction level
            if prob >= thr * 1.5:
                badge = "\U0001f534"  # red = high conviction
            elif prob >= thr:
                badge = "\U0001f7e0"  # orange = above threshold
            else:
                badge = "\u26aa"     # white = watch

            # Direction from funding
            fund_z = feats.get("funding_z", 0)
            direction = "DN" if fund_z > 0 else "UP"

            # Key signal summary
            signals = []
            if feats.get("num_event_types_2h", 0) >= 2:
                signals.append(f"{int(feats['num_event_types_2h'])}types")
            if feats.get("thin_pct", 0) > 0.85:
                signals.append("thin")
            if abs(fund_z) > 2.0:
                signals.append(f"fz={fund_z:+.1f}")
            if feats.get("compression", 0) > 70:
                signals.append(f"cs={feats['compression']:.0f}")

            sig_str = " | ".join(signals) if signals else "-"

            lines.append(
                f"{badge} {i}. <b>{sym}</b> {direction} "
                f"P={prob:.1%} | rank={state.rank:.0f} | {sig_str}"
            )

        total_above = sum(1 for p, *_ in scored if p >= thr)
        lines.append(f"\n{len(scored)} coins scored | {total_above} above threshold")

        return "\n".join(lines)
