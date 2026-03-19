"""
Analysis Agent — AI-powered event interpretation.

Called for high-score individual Telegram alerts (CA, VE, OI_SURGE with score >= 70).
Uses Claude to produce a short, actionable 2-3 sentence interpretation that gets
appended to the Telegram message.

Cost estimate: ~$0.05-0.15/day at 5-15 calls × Sonnet pricing.

Rate limiting:
  - Only events with score >= ANALYSIS_SCORE_THRESHOLD (default 70)
  - Max MAX_ANALYSES_PER_HOUR per rolling hour
  - 30-minute cache per symbol (no repeat analysis)
  - Hard timeout of 12s so slow API never blocks the alert pipeline

Reflection context:
  - Pulls recent learnings from ReflectionStore (written by ObductionAgent)
  - Includes historical hit-rate for this event type in the prompt
  - Makes each analysis aware of what patterns have worked in the past
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────

ANALYSIS_SCORE_THRESHOLD = 70       # minimum event score to trigger analysis
MAX_ANALYSES_PER_HOUR    = 12       # rate cap — controls API spend
SYMBOL_CACHE_TTL         = 1800     # seconds before re-analyzing same symbol (30 min)
API_TIMEOUT              = 12.0     # seconds before giving up on Claude
MODEL                    = "claude-sonnet-4-5"


# ── Agent ──────────────────────────────────────────────────────────────────────

class AnalysisAgent:
    """
    Interprets scanner events using Claude API.
    All methods are async and never raise — degraded output returns None.
    """

    def __init__(self) -> None:
        self._api_key: str = os.environ.get("ANTHROPIC_API_KEY", "")
        self._hour_timestamps: list[float] = []        # monotonic ts of recent calls
        self._cache: Dict[str, tuple[str, float]] = {} # symbol -> (text, mono_ts)

        # Reflection store — lazy init to avoid import issues at startup
        self._reflections: Optional[Any] = None
        self._db_path = os.environ.get("EVENTS_DB_PATH", "data/events.db")

        if self._api_key:
            logger.info(
                "AnalysisAgent enabled (model=%s, threshold=%d, cap=%d/h)",
                MODEL, ANALYSIS_SCORE_THRESHOLD, MAX_ANALYSES_PER_HOUR,
            )
        else:
            logger.warning("ANTHROPIC_API_KEY not set — AnalysisAgent disabled")

    def _get_reflection_store(self) -> Optional[Any]:
        """Lazy-init the ReflectionStore to avoid circular import issues."""
        if self._reflections is None:
            try:
                from src.agents.reflection_store import ReflectionStore
                self._reflections = ReflectionStore(self._db_path)
            except Exception as exc:
                logger.debug("ReflectionStore unavailable: %s", exc)
                self._reflections = False   # sentinel: don't retry
        return self._reflections if self._reflections else None

    # ── Public ────────────────────────────────────────────────────────────────

    async def analyze_event(self, event: Any, btc_context: str = "") -> Optional[str]:
        """
        Return a 2-3 sentence AI analysis for a high-score event, or None.

        Silently returns None when:
          - API key missing
          - Score below threshold
          - Rate limited (cap reached for this hour)
          - Symbol was analyzed within the last 30 min (cache hit)
          - API call fails or times out
        """
        if not self._api_key:
            return None

        score = getattr(event, "score", 0) or 0
        if score < ANALYSIS_SCORE_THRESHOLD:
            return None

        now = time.monotonic()

        # Rate limit — rolling 1-hour window
        self._hour_timestamps = [t for t in self._hour_timestamps if now - t < 3600]
        if len(self._hour_timestamps) >= MAX_ANALYSES_PER_HOUR:
            logger.debug(
                "AnalysisAgent: rate limited (%d/%d calls this hour)",
                len(self._hour_timestamps), MAX_ANALYSES_PER_HOUR,
            )
            return None

        # Symbol cache
        cached_text, cached_ts = self._cache.get(event.symbol, (None, 0.0))
        if cached_text and (now - cached_ts) < SYMBOL_CACHE_TTL:
            logger.debug("AnalysisAgent: cache hit for %s", event.symbol)
            return cached_text

        # Gather reflection context (non-blocking — empty string on failure)
        learnings    = ""
        hist_stats   = None
        store = self._get_reflection_store()
        if store is not None:
            try:
                learnings, hist_stats = await asyncio.gather(
                    store.get_recent_learnings(limit=4),
                    store.get_outcome_stats(
                        str(getattr(event, "event_type", "")),
                        score_min=max(0.0, score - 15.0),
                    ),
                    return_exceptions=True,
                )
                if isinstance(learnings, Exception):
                    learnings = ""
                if isinstance(hist_stats, Exception):
                    hist_stats = None
            except Exception:
                learnings  = ""
                hist_stats = None

        prompt = self._build_prompt(event, btc_context, learnings, hist_stats)
        t0 = time.monotonic()

        try:
            analysis = await asyncio.wait_for(
                self._call_claude(prompt),
                timeout=API_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "AnalysisAgent: timeout after %.0fs for %s",
                API_TIMEOUT, event.symbol,
            )
            return None
        except Exception as exc:
            logger.error("AnalysisAgent: unexpected error for %s: %r", event.symbol, exc)
            return None

        elapsed = time.monotonic() - t0

        if analysis:
            self._hour_timestamps.append(now)
            self._cache[event.symbol] = (analysis, now)
            logger.info(
                "AnalysisAgent: analyzed %s %s score=%.0f in %.1fs (%d/h used)",
                event.symbol,
                getattr(event, "event_type", "?"),
                score,
                elapsed,
                len(self._hour_timestamps),
            )

        return analysis

    # ── Prompt ────────────────────────────────────────────────────────────────

    def _build_prompt(
        self,
        event: Any,
        btc_context: str,
        learnings: str = "",
        hist_stats: Optional[Dict] = None,
    ) -> str:
        symbol      = getattr(event, "symbol", "UNKNOWN")
        event_type  = getattr(event, "event_type", "UNKNOWN")
        score       = getattr(event, "score", 0)
        direction   = getattr(event, "direction", "UNKNOWN")
        features    = getattr(event, "features", {}) or {}

        # Pull the most useful numbers from features
        funding_cur  = features.get("funding_current")
        funding_z    = features.get("funding_z")
        oi_z         = features.get("oi_z_24h") or features.get("oi_z_1h")
        thin         = features.get("thin_pct")
        vac_ask      = features.get("vacuum_dist_ask")
        vac_bid      = features.get("vacuum_dist_bid")
        spread_bps   = features.get("spread_bps")
        rank         = features.get("rank")
        phase        = features.get("settlement_phase", "N/A")
        minutes_set  = features.get("minutes_to_settlement")
        vol_ratio    = features.get("vol_ratio_5m")
        move_pct     = features.get("move_pct_5m")
        oi_surge_pct = features.get("oi_surge_pct")
        surge_tf     = features.get("surge_timeframe")
        oi_usd       = features.get("oi_usd")

        def _fmt(val: Any, fmt: str, suffix: str = "") -> str:
            if val is None:
                return "N/A"
            try:
                return format(float(val), fmt) + suffix
            except (TypeError, ValueError):
                return str(val)

        fr_str  = _fmt(funding_cur, "+.4f", "%") if funding_cur is not None else "neutral"
        frz_str = _fmt(funding_z, "+.1f")
        oiz_str = _fmt(oi_z, "+.1f")
        thn_str = _fmt(thin, ".2f")

        vac_str = "N/A"
        if vac_ask is not None and vac_ask >= 9000:
            vac_str = "EMPTY ask-side (no depth)"
        elif vac_bid is not None and vac_bid >= 9000:
            vac_str = "EMPTY bid-side (no depth)"
        elif vac_ask is not None:
            vac_str = f"{vac_ask:.0f} bps above mid"

        oi_usd_str = "N/A"
        if oi_usd:
            oi_usd_str = f"${oi_usd/1e6:.1f}M" if oi_usd >= 1e6 else f"${oi_usd/1e3:.0f}K"

        event_detail_lines = []
        if oi_surge_pct is not None:
            event_detail_lines.append(f"OI surge: +{oi_surge_pct:.1f}% ({surge_tf}) | Total OI: {oi_usd_str}")
        if vol_ratio is not None:
            event_detail_lines.append(f"Vol spike: {vol_ratio}x | Price move (5m): {_fmt(move_pct, '+.2f', '%')}")
        if rank is not None:
            event_detail_lines.append(f"Composite rank: {rank:.1f}")
        event_detail = "\n".join(event_detail_lines) if event_detail_lines else "N/A"

        settle_str = "N/A"
        if minutes_set is not None:
            settle_str = f"{phase} ({minutes_set:.0f} min to settlement)"

        # Historical accuracy block (from ReflectionStore)
        hist_block = ""
        if hist_stats and hist_stats.get("total", 0) >= 5:
            hist_block = (
                f"\n<historical_accuracy>\n"
                f"Past {event_type} events score>={hist_stats['score_min']:.0f}: "
                f"{hist_stats['hits']}/{hist_stats['total']} hits "
                f"({hist_stats['hit_rate']}%) | "
                f"avg favorable: {hist_stats['avg_fav']:+.1f}% | "
                f"avg adverse: {hist_stats['avg_adv']:+.1f}%\n"
                f"</historical_accuracy>"
            )

        # Learnings block (from ObductionAgent)
        learn_block = f"\n{learnings}" if learnings else ""

        return (
            f"<role>\n"
            f"You are a concise crypto derivatives analyst. "
            f"Output goes directly into a Telegram alert.\n"
            f"</role>\n"
            f"\n"
            f"<scanner_event>\n"
            f"Symbol:        {symbol}\n"
            f"Event:         {event_type}\n"
            f"Score:         {score:.0f} / 100\n"
            f"Direction:     {direction}\n"
            f"BTC (4h):      {btc_context or 'N/A'}\n"
            f"\n"
            f"Funding rate:  {fr_str}  (z-score: {frz_str})\n"
            f"OI z-score:    {oiz_str}\n"
            f"Book thinness: {thn_str}  (0=liquid, 1=empty)\n"
            f"Vacuum:        {vac_str}\n"
            f"Spread:        {_fmt(spread_bps, '.1f', ' bps')}\n"
            f"Settlement:    {settle_str}\n"
            f"\n"
            f"{event_detail}\n"
            f"</scanner_event>"
            f"{hist_block}"
            f"{learn_block}"
            f"\n\n"
            f"<task>\n"
            f"Write 2-3 sentences explaining:\n"
            f"1. WHY this symbol is fragile right now (mechanism: funding, OI build, thin book)\n"
            f"2. MOST LIKELY move in the next 1-4 hours\n"
            f"3. ONE concrete entry confirmation to watch\n"
            f"\n"
            f"Rules: reference actual numbers, under 55 words, no disclaimers, "
            f"no bullet points, flowing sentences, English only.\n"
            f"If historical accuracy is provided, factor it into your confidence.\n"
            f"</task>"
        )

    # ── API call ──────────────────────────────────────────────────────────────

    async def _call_claude(self, prompt: str) -> Optional[str]:
        """POST to Anthropic Messages API using aiohttp."""
        import aiohttp

        headers = {
            "x-api-key":         self._api_key,
            "anthropic-version": "2023-06-01",
            "content-type":      "application/json",
        }
        payload = {
            "model":      MODEL,
            "max_tokens": 160,
            "messages":   [{"role": "user", "content": prompt}],
        }

        try:
            timeout = aiohttp.ClientTimeout(total=API_TIMEOUT)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=payload,
                ) as resp:
                    body = await resp.text()
                    if resp.status != 200:
                        logger.error(
                            "AnalysisAgent: Claude API HTTP %d: %s",
                            resp.status, body[:300],
                        )
                        return None
                    data = json.loads(body)
                    text = (data.get("content") or [{}])[0].get("text", "")
                    return text.strip() or None

        except Exception as exc:
            logger.error("AnalysisAgent: _call_claude error: %r", exc)
            return None
