"""Liquidity Map — orderbook structure, liquidation clusters, and path resistance.

Public API
----------
estimate_liquidation_levels(current_price, oi_value_usdt, side)
    -> List[Tuple[float, float, int]]   # (liq_price, volume_usdt, dominant_leverage)
    Estimate where liquidations cluster for "long" or "short" positions.

path_resistance(book, current_price, target_price, cascade_volume=nan)
    -> Dict[str, Any]
    Compute USDT depth and resistance classification between two prices.

display_liquidity_map(symbol, book, funding_feats, oi_feats, sps, console=None)
    Render a full liquidity map for one symbol to a Rich console.

CLI
---
    python -m src.scanner.liquidity_map BTCUSDT
    python -m src.scanner.liquidity_map SOLUSDT --band-bps 10 --levels 12
"""

from __future__ import annotations

import argparse
import asyncio
import math
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rich.console import Console
from rich.text import Text

from ..bybit.rest import BybitRestClient
from ..bybit.ws import LocalOrderbook
from ..features import (
    compute_funding_features,
    compute_oi_features,
    compute_orderbook_features,
    settlement_pressure_score,
)

# ── Leverage distribution for liquidation estimation ─────────────────────────
# Weights represent the assumed fraction of open interest at each leverage level.
# Mid-cap assumption (tier-2): average 10x, range 5x–25x.
_LEVERAGE_DIST: Dict[int, float] = {5: 0.15, 10: 0.35, 15: 0.25, 20: 0.15, 25: 0.10}

# ── Display / classification constants ───────────────────────────────────────
_MAX_BAR_CHARS = 12          # max length of depth bar in display
_VACUUM_RATIO = 0.30         # depth < median * ratio → vacuum zone
_WALL_RATIO   = 2.0          # depth > median * ratio → wall (squeeze target)
_CLUSTER_PCT  = 0.2          # cluster liq levels within 0.2 % of each other

# path_resistance thresholds
_OPEN_PATH = 0.30
_CONTESTED = 0.70            # > 0.30 and ≤ 0.70 → CONTESTED, > 0.70 → BLOCKED


# ── Formatting helpers ────────────────────────────────────────────────────────


def _fmt_usdt(v: float) -> str:
    """Format a USDT notional into a compact string ($1.2M, $450K, $8.3K)."""
    if not math.isfinite(v) or v == 0:
        return "$0"
    if v >= 1_000_000:
        return f"${v / 1_000_000:.1f}M"
    if v >= 1_000:
        return f"${v / 1_000:.1f}K"
    return f"${v:.0f}"


def _fmt_price(p: float) -> str:
    """Format a price with up to 5 significant figures."""
    if p >= 10_000:
        return f"${p:,.0f}"
    if p >= 1_000:
        return f"${p:,.1f}"
    if p >= 100:
        return f"${p:.2f}"
    return f"${p:.3f}"


def _bar_str(depth: float, max_depth: float, max_chars: int = _MAX_BAR_CHARS) -> str:
    """Return a bar of '|' characters proportional to depth."""
    if max_depth <= 0 or depth <= 0:
        return ""
    n = max(1, round(depth / max_depth * max_chars))
    return "|" * min(n, max_chars)


# ── Core functions ────────────────────────────────────────────────────────────


def estimate_liquidation_levels(
    current_price: float,
    oi_value_usdt: float,
    side: str,
) -> List[Tuple[float, float, int]]:
    """Estimate liquidation price clusters for a position side.

    Parameters
    ----------
    current_price:
        Current mid price, used as a proxy for the average entry price.
    oi_value_usdt:
        Estimated USDT notional for this side (e.g. half of total OI value).
    side:
        ``"long"`` or ``"short"``.

    Returns
    -------
    List of ``(liq_price, volume_usdt, dominant_leverage)`` tuples, sorted by
    liq_price descending (longs) or ascending (shorts).  Nearby levels (within
    ``_CLUSTER_PCT`` %) are merged into a single cluster.
    """
    if current_price <= 0 or oi_value_usdt <= 0 or side not in ("long", "short"):
        return []

    raw: List[Tuple[float, float, int]] = []
    for leverage, fraction in _LEVERAGE_DIST.items():
        volume = oi_value_usdt * fraction
        if side == "long":
            liq_price = current_price * (1.0 - 1.0 / leverage)
        else:
            liq_price = current_price * (1.0 + 1.0 / leverage)
        raw.append((liq_price, volume, leverage))

    # Sort by price so we can merge adjacent levels
    raw.sort(key=lambda x: x[0])

    clusters: List[Tuple[float, float, int]] = []
    for price, vol, lev in raw:
        if clusters:
            prev_price, prev_vol, prev_lev = clusters[-1]
            if abs(price - prev_price) / prev_price * 100 <= _CLUSTER_PCT:
                # Merge: volume-weighted average price, dominant leverage by volume
                total_vol = prev_vol + vol
                merged_price = (prev_price * prev_vol + price * vol) / total_vol
                dominant_lev = lev if vol > prev_vol else prev_lev
                clusters[-1] = (merged_price, total_vol, dominant_lev)
                continue
        clusters.append((price, vol, lev))

    # Sort: longs descending (highest liq price near current → most dangerous first)
    #        shorts ascending (lowest liq price near current → most dangerous first)
    clusters.sort(key=lambda x: x[0], reverse=(side == "long"))
    return clusters


def path_resistance(
    book: LocalOrderbook,
    current_price: float,
    target_price: float,
    cascade_volume: float = float("nan"),
) -> Dict[str, Any]:
    """Compute resting depth on the path between current and target price.

    Parameters
    ----------
    book:
        Live ``LocalOrderbook`` instance.
    current_price:
        Mid price at the time of computation.
    target_price:
        Price level we are measuring resistance toward.
    cascade_volume:
        Estimated USDT notional of the forced liquidation flow (from
        ``estimate_liquidation_levels``).  When provided, a
        ``resistance_ratio`` and ``label`` are included in the result.

    Returns
    -------
    Dict with keys:

    depth_usdt:
        Total USDT resting depth between current and target price.
    distance_pct:
        Percentage distance from current to target price.
    resistance_ratio:
        ``depth_usdt / cascade_volume`` (only when cascade_volume is given).
    label:
        ``"OPEN PATH"``, ``"CONTESTED"``, or ``"BLOCKED"`` (requires cascade_volume).
    """
    going_up = target_price > current_price
    distance_pct = abs(target_price - current_price) / current_price * 100.0

    depth_usdt = 0.0
    if going_up:
        for price, size in book.get_sorted_asks(500):
            if price > target_price:
                break
            if price > current_price:
                depth_usdt += price * size
    else:
        for price, size in book.get_sorted_bids(500):
            if price < target_price:
                break
            if price < current_price:
                depth_usdt += price * size

    result: Dict[str, Any] = {
        "depth_usdt":   depth_usdt,
        "distance_pct": distance_pct,
        "resistance_ratio": float("nan"),
        "label": None,
    }

    if not math.isnan(cascade_volume) and cascade_volume > 0:
        ratio = depth_usdt / cascade_volume
        result["resistance_ratio"] = ratio
        if ratio < _OPEN_PATH:
            result["label"] = "OPEN PATH"
        elif ratio <= _CONTESTED:
            result["label"] = "CONTESTED"
        else:
            result["label"] = "BLOCKED"

    return result


# ── LiquidityMap display ──────────────────────────────────────────────────────


class LiquidityMap:
    """Renders a terminal liquidity map for one symbol using Rich.

    Parameters
    ----------
    symbol:
        Instrument name (e.g. ``"SOLUSDT"``).
    book:
        Live ``LocalOrderbook`` instance.
    funding_feats, oi_feats:
        Feature dicts from ``compute_funding_features`` / ``compute_oi_features``.
    sps:
        Settlement Pressure Score in [-100, +100].
    band_bps:
        Width of each price band in basis points (default 10).
    num_levels:
        Number of price bands to show on each side (default 10).
    """

    def __init__(
        self,
        symbol: str,
        book: LocalOrderbook,
        funding_feats: Dict[str, Any],
        oi_feats: Dict[str, float],
        sps: float,
        *,
        band_bps: float = 10.0,
        num_levels: int = 10,
    ) -> None:
        self.symbol = symbol
        self.book = book
        self.funding_feats = funding_feats
        self.oi_feats = oi_feats
        self.sps = sps
        self.band_bps = band_bps
        self.num_levels = num_levels

        bb = book.best_bid()
        ba = book.best_ask()
        self.mid = (bb + ba) / 2.0 if bb and ba else 0.0
        self.spread_bps = (ba - bb) / self.mid * 10_000 if self.mid > 0 else 0.0

    # ── Internal helpers ───────────────────────────────────────────────────

    def _aggregate_bands(
        self,
        levels: List[Tuple[float, float]],
        above: bool,
    ) -> List[Tuple[float, float]]:
        """Aggregate individual price levels into bands of `band_bps` width.

        Returns a list of (band_midprice, total_usdt_depth), nearest levels first.
        """
        if not levels or self.mid <= 0:
            return []

        band_size = self.mid * self.band_bps / 10_000
        bands: Dict[float, float] = {}
        for price, size in levels:
            idx = round((price - self.mid) / band_size)
            band_price = self.mid + idx * band_size
            bands[band_price] = bands.get(band_price, 0.0) + price * size

        if above:
            sorted_bands = sorted(
                [(p, d) for p, d in bands.items() if p > self.mid],
                key=lambda x: x[0],
            )[: self.num_levels]
        else:
            sorted_bands = sorted(
                [(p, d) for p, d in bands.items() if p <= self.mid],
                key=lambda x: x[0],
                reverse=True,
            )[: self.num_levels]

        return sorted_bands

    def _classify_bands(
        self, bands: List[Tuple[float, float]]
    ) -> Tuple[float, List[str]]:
        """Return (max_depth, list_of_tags) where tag is 'vacuum', 'wall', or ''."""
        if not bands:
            return 0.0, []
        depths = [d for _, d in bands]
        max_depth = max(depths)
        median_d = float(np.median(depths))
        tags = []
        for _, d in bands:
            if d < median_d * _VACUUM_RATIO:
                tags.append("vacuum")
            elif d > median_d * _WALL_RATIO:
                tags.append("wall")
            else:
                tags.append("")
        return max_depth, tags

    def _find_squeeze_target(
        self, bands: List[Tuple[float, float]], tags: List[str]
    ) -> Optional[Tuple[float, float]]:
        """Find the first 'wall' band after a series of vacuum zones."""
        seen_vacuum = False
        for (price, depth), tag in zip(bands, tags):
            if tag == "vacuum":
                seen_vacuum = True
            elif seen_vacuum and tag == "wall":
                return price, depth
        return None

    def _liq_clusters_in_range(
        self,
        clusters: List[Tuple[float, float, int]],
        lo: float,
        hi: float,
    ) -> List[Tuple[float, float, int]]:
        """Return clusters whose price falls within [lo, hi]."""
        return [(p, v, lev) for p, v, lev in clusters if lo <= p <= hi]

    # ── Render ─────────────────────────────────────────────────────────────

    def render(self, console: Optional[Console] = None) -> None:
        """Print the full liquidity map to *console* (or a default Console)."""
        if console is None:
            console = Console()

        nan = float("nan")
        fund_z  = self.funding_feats.get("funding_z",           nan)
        oi_z    = self.oi_feats.get("oi_z_24h",                 nan)
        oi_cur  = self.oi_feats.get("oi_current",               nan)
        phase   = self.funding_feats.get("settlement_phase",    "?")
        mins    = self.funding_feats.get("minutes_to_settlement", nan)
        mid     = self.mid
        sps     = self.sps

        # ── Liquidation cluster estimates ─────────────────────────────────
        # Rough OI value: oi_current (contracts) × price; split 50/50 long/short
        oi_usdt = oi_cur * mid if (not math.isnan(oi_cur) and mid > 0) else 0.0
        long_oi  = oi_usdt * 0.50
        short_oi = oi_usdt * 0.50

        liq_longs  = estimate_liquidation_levels(mid, long_oi,  "long")
        liq_shorts = estimate_liquidation_levels(mid, short_oi, "short")

        # ── Aggregate orderbook into bands ───────────────────────────────
        ask_bands = self._aggregate_bands(self.book.get_sorted_asks(500), above=True)
        bid_bands = self._aggregate_bands(self.book.get_sorted_bids(500), above=False)

        ask_max, ask_tags = self._classify_bands(ask_bands)
        bid_max, bid_tags = self._classify_bands(bid_bands)
        global_max = max(ask_max, bid_max, 1.0)

        # Squeeze direction from SPS
        squeeze_down = not math.isnan(sps) and sps < 0
        squeeze_up   = not math.isnan(sps) and sps > 0

        # Find squeeze targets
        squeeze_ask_target = self._find_squeeze_target(ask_bands, ask_tags)
        squeeze_bid_target = self._find_squeeze_target(bid_bands, bid_tags)

        # ── Path resistance for the primary squeeze direction ────────────
        path_info: Optional[Dict[str, Any]] = None
        cascade_vol = 0.0
        path_direction = ""
        target_price_for_path = mid

        if squeeze_down and bid_bands:
            # Look for the first significant wall below current price
            target_band = (
                squeeze_bid_target[0] if squeeze_bid_target else bid_bands[-1][0]
            )
            cascade_vol = sum(v for _, v, _ in liq_longs)  # longs get liquidated going down
            path_info = path_resistance(self.book, mid, target_band, cascade_vol)
            path_direction = "DOWN"
            target_price_for_path = target_band
        elif squeeze_up and ask_bands:
            target_band = (
                squeeze_ask_target[0] if squeeze_ask_target else ask_bands[-1][0]
            )
            cascade_vol = sum(v for _, v, _ in liq_shorts)  # shorts get liquidated going up
            path_info = path_resistance(self.book, mid, target_band, cascade_vol)
            path_direction = "UP"
            target_price_for_path = target_band

        # ── Build display ─────────────────────────────────────────────────
        console.print()

        # Title
        console.print(
            Text(f"=== LIQUIDITY MAP: {self.symbol} ===", style="bold cyan")
        )

        # Info row
        phase_str = phase
        if not math.isnan(mins):
            phase_str += f" ({int(mins)} min to settle)"
        spread_str = f"{self.spread_bps:.1f}" if not math.isnan(self.spread_bps) else "?"
        info = Text()
        info.append(f"Price: {_fmt_price(mid)}  |  ")
        info.append(f"Spread: {spread_str} bps  |  Phase: ")
        phase_color = "red bold" if phase == "IMMINENT" else (
            "yellow" if phase == "APPROACH" else "white"
        )
        info.append(phase_str, style=phase_color)
        console.print(info)

        # Score row
        sps_str   = f"{sps:+.1f}" if not math.isnan(sps) else "N/A"
        sps_dir   = "DOWN" if squeeze_down else ("UP" if squeeze_up else "NEUTRAL")
        fund_z_str = f"{fund_z:+.1f}" if not math.isnan(fund_z) else "N/A"
        oi_z_str   = f"{oi_z:+.1f}"  if not math.isnan(oi_z)   else "N/A"
        score_line = Text()
        score_line.append(f"SPS: ")
        sps_style = "red bold" if squeeze_down else ("green bold" if squeeze_up else "white")
        score_line.append(f"{sps_str} ({sps_dir})", style=sps_style)
        score_line.append(f"  |  Funding Z: {fund_z_str}  |  OI Z: {oi_z_str}")
        console.print(score_line)

        console.print()

        # Column header
        header = Text()
        header.append("ASK (resistance above)", style="bold red")
        header.append("          ")
        header.append("BID (support below)", style="bold green")
        console.print(header)
        console.print("─" * 60)

        # ── ASK rows (descending: furthest first → nearest to current last) ──
        ask_liq_map = {
            round(p / (mid * 0.001)) * (mid * 0.001): (v, lev)
            for p, v, lev in liq_shorts
        }

        for (price, depth), tag in reversed(list(zip(ask_bands, ask_tags))):
            bar    = _bar_str(depth, global_max)
            amount = _fmt_usdt(depth)
            row = Text()

            # Check for liquidation cluster near this band
            nearby_liq = self._liq_clusters_in_range(
                liq_shorts, price - mid * self.band_bps / 10_000, price + mid * self.band_bps / 10_000
            )
            liq_ann = ""
            if nearby_liq:
                _, lv, dom_lev = nearby_liq[0]
                liq_ann = f" [LIQ] Est {_fmt_usdt(lv)} ({dom_lev}x shorts)"

            is_squeeze_target = (
                squeeze_ask_target is not None
                and abs(price - squeeze_ask_target[0]) < 1e-8
            )

            row.append(f"  {_fmt_price(price):<12}", style="red")
            row.append(f"{bar:<{_MAX_BAR_CHARS + 2}}", style="dim red")
            row.append(f"{amount:<8}", style="red")

            if liq_ann:
                row.append(liq_ann, style="yellow")
            elif tag == "vacuum":
                row.append("<-- VACUUM ZONE", style="dim yellow")
            elif is_squeeze_target:
                row.append("<-- SQUEEZE TARGET (wall)", style="bold magenta")

            console.print(row)

        # ── Current price separator ────────────────────────────────────────
        cur_line = Text()
        cur_line.append(f"--- CURRENT: {_fmt_price(mid)} ---", style="bold white")
        console.print(cur_line)

        # ── BID rows (descending: nearest to current first → furthest last) ─
        for (price, depth), tag in zip(bid_bands, bid_tags):
            bar    = _bar_str(depth, global_max)
            amount = _fmt_usdt(depth)
            row = Text()

            nearby_liq = self._liq_clusters_in_range(
                liq_longs, price - mid * self.band_bps / 10_000, price + mid * self.band_bps / 10_000
            )
            liq_ann = ""
            if nearby_liq:
                _, lv, dom_lev = nearby_liq[0]
                liq_ann = f" [LIQ] Est {_fmt_usdt(lv)} ({dom_lev}x longs)"

            is_squeeze_target = (
                squeeze_bid_target is not None
                and abs(price - squeeze_bid_target[0]) < 1e-8
            )

            row.append(f"  {_fmt_price(price):<12}", style="green")
            row.append(f"{bar:<{_MAX_BAR_CHARS + 2}}", style="dim green")
            row.append(f"{amount:<8}", style="green")

            if liq_ann:
                row.append(liq_ann, style="yellow")
            elif tag == "vacuum":
                row.append("<-- VACUUM ZONE", style="dim yellow")
            elif is_squeeze_target:
                row.append("<-- SQUEEZE TARGET (wall)", style="bold magenta")

            console.print(row)

        console.print("─" * 60)

        # ── Path resistance summary ────────────────────────────────────────
        if path_info is not None and path_direction:
            depth_str = _fmt_usdt(path_info["depth_usdt"])
            dist_str  = f"{path_info['distance_pct']:.2f}%"
            label     = path_info.get("label") or "N/A"
            label_style = (
                "bold green" if label == "OPEN PATH" else
                "yellow"     if label == "CONTESTED"  else
                "bold red"
            )
            path_line = Text()
            path_line.append(f"Path {path_direction}: resistance = {depth_str} across {dist_str} = ")
            path_line.append(label, style=label_style)
            console.print(path_line)

            if cascade_vol > 0:
                cascade_str = _fmt_usdt(cascade_vol)
                console.print(
                    f"Estimated cascade: {cascade_str} forced {'selling' if squeeze_down else 'buying'}"
                    f" vs {depth_str} resistance"
                )

            target_str = _fmt_price(target_price_for_path)
            dist_pct = path_info["distance_pct"]
            sign = "-" if squeeze_down else "+"
            console.print(
                f"Expected magnitude: {sign}{dist_pct:.2f}% to wall at {target_str}"
            )

        console.print()


# ── Convenience wrapper ───────────────────────────────────────────────────────


def display_liquidity_map(
    symbol: str,
    book: LocalOrderbook,
    funding_feats: Dict[str, Any],
    oi_feats: Dict[str, float],
    sps: float,
    *,
    console: Optional[Console] = None,
    band_bps: float = 10.0,
    num_levels: int = 10,
) -> None:
    """Render the liquidity map for *symbol* to a Rich console.

    This is the primary entry point when calling from the pressure scanner or
    other modules that have already computed features.
    """
    lmap = LiquidityMap(
        symbol, book, funding_feats, oi_feats, sps,
        band_bps=band_bps, num_levels=num_levels,
    )
    lmap.render(console)


# ── CLI ───────────────────────────────────────────────────────────────────────


async def _fetch_and_render(args: argparse.Namespace) -> None:
    """Fetch live data for one symbol and render its liquidity map."""
    symbol = args.symbol.upper()
    console = Console()

    async with BybitRestClient() as rest:
        console.print(f"Fetching data for [bold]{symbol}[/bold]…")

        # Orderbook snapshot (REST fallback — WS not needed for one-shot display)
        ob_snap = await rest.get_orderbook(symbol, limit=200)
        book = LocalOrderbook()
        book.on_snapshot(ob_snap)

        if not book.bids or not book.asks:
            console.print(f"[red]Empty orderbook for {symbol}. Is the symbol valid?[/red]")
            return

        mid = (book.best_bid() + book.best_ask()) / 2.0  # type: ignore[operator]

        # OI: fetch last 200 hourly rows for feature computation
        oi_result = await rest._get(
            "/v5/market/open-interest",
            {"category": "linear", "symbol": symbol, "intervalTime": "1h", "limit": 200},
        )
        import pandas as pd
        oi_rows = oi_result.get("list", [])
        if oi_rows:
            oi_df = pd.DataFrame([{
                "timestamp":    pd.Timestamp(int(r["timestamp"]), unit="ms", tz="UTC"),
                "symbol":       symbol,
                "open_interest": float(r["openInterest"]),
            } for r in reversed(oi_rows)])  # API returns newest-first; reverse to ascending
        else:
            oi_df = None

        # Funding: fetch last 90 rows for feature computation
        fund_result = await rest._get(
            "/v5/market/funding/history",
            {"category": "linear", "symbol": symbol, "limit": 90},
        )
        fund_rows = fund_result.get("list", [])
        if fund_rows:
            fund_df = pd.DataFrame([{
                "timestamp":   pd.Timestamp(int(r["fundingRateTimestamp"]), unit="ms", tz="UTC"),
                "symbol":      symbol,
                "funding_rate": float(r["fundingRate"]),
            } for r in reversed(fund_rows)])
        else:
            fund_df = None

        # Compute features
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        funding_feats = compute_funding_features(fund_df, now=now)
        oi_feats      = compute_oi_features(oi_df) if oi_df is not None else {}

        fund_z = funding_feats.get("funding_z", float("nan"))
        oi_z   = oi_feats.get("oi_z_24h", float("nan"))
        mins   = funding_feats.get("minutes_to_settlement", float("nan"))

        ob_feats = compute_orderbook_features(ob_snap)
        thin_p   = ob_feats.get("thin_pct", float("nan"))

        # Vacuum in squeeze direction
        vac_ask = ob_feats.get("vacuum_dist_ask", float("nan"))
        vac_bid = ob_feats.get("vacuum_dist_bid", float("nan"))
        vac_sq  = vac_bid if (not math.isnan(fund_z) and fund_z > 0) else vac_ask

        sps = settlement_pressure_score(
            funding_z=fund_z,
            oi_z_7d=oi_z,
            vacuum_dist_squeeze_dir=vac_sq,
            thin_pct=thin_p,
            minutes_to_settle=mins,
        )

    lmap = LiquidityMap(
        symbol, book, funding_feats, oi_feats, sps,
        band_bps=args.band_bps, num_levels=args.levels,
    )
    lmap.render(console)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bybit Perpetuals Liquidity Map",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("symbol", help="Symbol to display, e.g. BTCUSDT")
    parser.add_argument(
        "--band-bps",
        type=float,
        default=10.0,
        help="Price band width in basis points",
    )
    parser.add_argument(
        "--levels",
        type=int,
        default=10,
        help="Number of price bands to show on each side",
    )
    args = parser.parse_args()

    try:
        asyncio.run(_fetch_and_render(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
