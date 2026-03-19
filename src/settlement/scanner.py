"""
Funding Settlement Scanner.

Runs ~60 min before each funding settlement (00:00, 08:00, 16:00 UTC).
Ranks all symbols by squeeze potential and outputs top 10 with direction.

Squeeze Potential = (Funding Pressure) / (Book Resistance)

Where:
  Funding Pressure = |funding_rate| * open_interest_usd
  Book Resistance  = depth in squeeze direction (within 1% of mid)

Higher ratio = thinner book absorbing more pressure = bigger potential move.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np
import yaml

if TYPE_CHECKING:
    from ..bybit.rest import BybitRestClient
    from ..scanner.pressure_scanner import PressureScanner, SymbolState

logger = logging.getLogger(__name__)

# ── Tier thresholds (24h turnover in USD) ────────────────────────────────────

_TIER1_TURNOVER = 1_000_000_000   # > $1B
_TIER2_TURNOVER = 50_000_000      # > $50M
_MIN_TURNOVER   = 1_000_000       # > $1M (filter floor)

_CONFIG_FALLBACK = Path(__file__).resolve().parent.parent.parent / "configs" / "symbols.yaml"


async def discover_symbols(rest: BybitRestClient) -> Dict[int, List[str]]:
    """
    Fetch all USDT perpetual tickers from Bybit and classify into tiers.

    Returns Dict[tier, List[symbol]] with:
      Tier 1: turnover24h > $1B
      Tier 2: turnover24h > $50M
      Tier 3: turnover24h > $1M

    Falls back to configs/symbols.yaml if the API call fails.
    """
    try:
        tickers = await rest.get_bulk_tickers()
    except Exception as exc:
        logger.warning("Failed to fetch tickers: %s — falling back to yaml", exc)
        return _load_yaml_fallback()

    if not tickers:
        logger.warning("Empty ticker response — falling back to yaml")
        return _load_yaml_fallback()

    # Filter USDT perps with sufficient volume
    result: Dict[int, List[str]] = {1: [], 2: [], 3: []}

    for sym, data in tickers.items():
        if not sym.endswith("USDT"):
            continue
        turnover = data.get("turnover24h", 0)
        if turnover < _MIN_TURNOVER:
            continue

        if turnover >= _TIER1_TURNOVER:
            result[1].append(sym)
        elif turnover >= _TIER2_TURNOVER:
            result[2].append(sym)
        else:
            result[3].append(sym)

    # Sort each tier alphabetically
    for tier in result:
        result[tier].sort()

    total = sum(len(v) for v in result.values())
    if total == 0:
        logger.warning("No symbols passed turnover filter — falling back to yaml")
        return _load_yaml_fallback()

    logger.info(
        "Scanning %d symbols (%d tier1, %d tier2, %d tier3)",
        total, len(result[1]), len(result[2]), len(result[3]),
    )
    return result


def _load_yaml_fallback() -> Dict[int, List[str]]:
    """Load symbols from configs/symbols.yaml as fallback."""
    if not _CONFIG_FALLBACK.exists():
        logger.error("Fallback config not found: %s", _CONFIG_FALLBACK)
        return {}
    with open(_CONFIG_FALLBACK) as fh:
        cfg = yaml.safe_load(fh)
    result: Dict[int, List[str]] = {}
    for tier in (1, 2, 3):
        key = f"tier_{tier}"
        if key in cfg:
            result[tier] = list(cfg[key])
    logger.info("Loaded %d symbols from yaml fallback", sum(len(v) for v in result.values()))
    return result


class SettlementScanner:
    """Computes pre-settlement squeeze rankings for all symbols."""

    def __init__(self, pressure_scanner: PressureScanner) -> None:
        self.scanner = pressure_scanner

    @property
    def n_symbols(self) -> int:
        """Total number of symbols being tracked."""
        return len(self.scanner._states)

    async def compute_rankings(self) -> List[Dict[str, Any]]:
        """
        Compute squeeze potential for all symbols and return sorted list.

        Returns list of dicts with keys: rank, symbol, direction,
        funding_rate, funding_rate_pct, oi_usd, funding_pressure,
        book_depth_squeeze_side, squeeze_ratio, vacuum_bps, mark_price,
        tier, score.
        """
        results = []

        for symbol, state in self.scanner._states.items():
            result = self._compute_symbol(symbol, state)
            if result is not None:
                results.append(result)

        # Sort by squeeze_ratio descending
        results.sort(key=lambda r: r["squeeze_ratio"], reverse=True)

        # Assign ranks and normalize scores
        if results:
            max_ratio = results[0]["squeeze_ratio"]
            for i, r in enumerate(results):
                r["rank"] = i + 1
                r["score"] = round(
                    (r["squeeze_ratio"] / max_ratio) * 100, 1
                ) if max_ratio > 0 else 0.0

        return results

    def _compute_symbol(self, symbol: str, state: SymbolState) -> Optional[Dict[str, Any]]:
        """Compute squeeze potential for one symbol."""

        # === 1. FUNDING RATE (decimal, e.g. -0.0003 = -0.03%) ===
        funding_rate = state.funding_feats.get("funding_current", float("nan"))
        if math.isnan(funding_rate) or funding_rate == 0:
            return None

        # === 2. MID PRICE ===
        mid_price = state.ob_feats.get("mid_price", float("nan"))
        if math.isnan(mid_price) or mid_price <= 0:
            return None

        # === 3. OPEN INTEREST (USD) ===
        # oi_current is in contracts; multiply by mid_price for USD value
        oi_contracts = state.oi_feats.get("oi_current", float("nan"))
        if math.isnan(oi_contracts) or oi_contracts <= 0:
            return None
        oi_usd = oi_contracts * mid_price

        # === 4. MINIMUM OI FILTER ===
        if oi_usd < 2_000_000:
            return None

        # === 5. FUNDING PRESSURE ===
        funding_pressure = abs(funding_rate) * oi_usd

        # === 6. SQUEEZE DIRECTION ===
        # Negative funding = shorts pay = short squeeze (price UP)
        # Positive funding = longs pay = long squeeze (price DOWN)
        if funding_rate < 0:
            direction = "SHORT_SQUEEZE"
            squeeze_side = "ask"  # shorts liquidating = forced buying through asks
        else:
            direction = "LONG_SQUEEZE"
            squeeze_side = "bid"  # longs liquidating = forced selling through bids

        # === 7. BOOK DEPTH in squeeze direction (USD within ~1%) ===
        if squeeze_side == "ask":
            depth = state.ob_feats.get("depth_ask_usdt", float("nan"))
        else:
            depth = state.ob_feats.get("depth_bid_usdt", float("nan"))

        if math.isnan(depth) or depth <= 0:
            # Fallback: assume 0.1% of OI as depth so symbol still ranks (lower)
            depth = oi_usd * 0.001

        # === 8. SQUEEZE RATIO ===
        squeeze_ratio = funding_pressure / depth

        # === 9. VACUUM distance in squeeze direction (bps) ===
        if squeeze_side == "ask":
            vacuum_bps = state.ob_feats.get("vacuum_dist_ask", float("nan"))
        else:
            vacuum_bps = state.ob_feats.get("vacuum_dist_bid", float("nan"))

        return {
            "symbol": symbol,
            "direction": direction,
            "funding_rate": funding_rate,
            "funding_rate_pct": f"{funding_rate * 100:.4f}%",
            "oi_usd": round(oi_usd),
            "funding_pressure": round(funding_pressure, 2),
            "book_depth_squeeze_side": round(depth, 2),
            "squeeze_ratio": round(squeeze_ratio, 4),
            "vacuum_bps": vacuum_bps if not math.isnan(vacuum_bps) else None,
            "mark_price": mid_price,
            "tier": state.tier,
            "score": 0.0,  # filled after sorting
        }


def _next_settlement(now: datetime) -> datetime:
    """Compute the next 00:00/08:00/16:00 UTC settlement time."""
    today = now.date()
    for hour in (0, 8, 16):
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


def _fmt_vacuum(vac: Optional[float]) -> str:
    """Format vacuum bps: 9999 = EMPTY (extreme), None = N/A."""
    if vac is None:
        return "N/A"
    if vac >= 9999:
        return "EMPTY"
    return f"{vac:.0f}"


def format_rankings_table(
    rankings: List[Dict[str, Any]],
    top_n: int = 10,
    n_total: int = 0,
) -> str:
    """Format top N as a text table for terminal output."""
    now = datetime.now(timezone.utc)
    next_settle = _next_settlement(now)
    mins_to = int((next_settle - now).total_seconds() / 60)

    lines = []
    lines.append("=" * 80)
    universe_str = f"  |  {n_total} symbols" if n_total else ""
    lines.append(
        f"  PRE-SETTLEMENT SCAN  |  {next_settle.strftime('%Y-%m-%d %H:%M')} UTC  "
        f"|  T-{mins_to} min{universe_str}"
    )
    lines.append("=" * 80)
    lines.append(
        f"  {'#':<3} {'Symbol':<13} {'Dir':<6} {'FR%':<10} "
        f"{'OI':>8} {'Depth':>8} {'Score':>6} {'Vac':>6}"
    )
    lines.append("  " + "-" * 76)

    for r in rankings[:top_n]:
        dir_label = ">> UP" if r["direction"] == "SHORT_SQUEEZE" else "v  DN"

        # Format OI
        oi = r["oi_usd"]
        if oi >= 1_000_000_000:
            oi_str = f"{oi / 1e9:.1f}B"
        elif oi >= 1_000_000:
            oi_str = f"{oi / 1e6:.1f}M"
        else:
            oi_str = f"{oi / 1e3:.0f}K"

        # Format depth
        depth = r["book_depth_squeeze_side"]
        if depth >= 1_000_000:
            depth_str = f"{depth / 1e6:.1f}M"
        elif depth >= 1_000:
            depth_str = f"{depth / 1e3:.0f}K"
        else:
            depth_str = f"{depth:.0f}"

        # Vacuum
        vac_str = _fmt_vacuum(r.get("vacuum_bps"))

        lines.append(
            f"  {r['rank']:<3} {r['symbol']:<13} {dir_label:<6} "
            f"{r['funding_rate_pct']:<10} "
            f"{oi_str:>8} {depth_str:>8} "
            f"{r['score']:>6.1f} {vac_str:>6}"
        )

    lines.append("=" * 80)
    lines.append("")
    lines.append("  Dir: >> UP = shorts squeezed (price may rise)")
    lines.append("       v  DN = longs squeezed (price may fall)")
    lines.append("  Score = normalized squeeze potential (100 = highest)")
    lines.append("  Vac = vacuum dist in squeeze dir (bps); EMPTY = no depth found")
    lines.append("")

    return "\n".join(lines)


def format_telegram_message(rankings: List[Dict[str, Any]], top_n: int = 10) -> str:
    """Format for Telegram (HTML parse mode)."""
    now = datetime.now(timezone.utc)
    next_settle = _next_settlement(now)
    mins_to = int((next_settle - now).total_seconds() / 60)

    lines = []
    lines.append(
        f"<b>SETTLEMENT SCAN</b> | {next_settle.strftime('%H:%M')} UTC ({mins_to}min)"
    )
    lines.append("")

    for r in rankings[:top_n]:
        dir_text = "UP" if r["direction"] == "SHORT_SQUEEZE" else "DN"

        oi = r["oi_usd"]
        oi_str = f"{oi / 1e6:.0f}M" if oi >= 1e6 else f"{oi / 1e3:.0f}K"

        lines.append(
            f"{r['rank']}. <b>{r['symbol']}</b> {dir_text} "
            f"| FR: {r['funding_rate_pct']} "
            f"| OI: {oi_str} "
            f"| Score: {r['score']:.0f}"
        )

    lines.append("")
    lines.append("Ratio = pressure/depth -- higher = more explosive")

    return "\n".join(lines)
