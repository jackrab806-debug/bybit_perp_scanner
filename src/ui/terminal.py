"""Live Dashboard — interactive terminal combining pressure scanner + liquidity map.

Layout (full screen, split 50/50 vertically)
---------------------------------------------
+----------------------------------------------------------+
|  PRESSURE SCANNER            Next settle: 00:00 (23m)   |
|  #  Symbol     Tier  Rank  Dir   SPS  Comp  LFI    FZ  |
|  1  SUIUSDT    T2    87   DOWN  -78    82    71    3.1  |
|  2  DOGEUSDT   T2    74   DOWN  -65    68    63    2.8  |
|  ...                                                     |
|  Filter: All tiers  |  Auto-follow: top rank             |
+----------------------------------------------------------+
|  LIQUIDITY MAP: SUIUSDT (auto: top rank)                 |
|  Price: $3.42  Spread: 4.2 bps  IMMINENT (12 min)       |
|  SPS: -78 (DOWN)  |  FZ: +3.1                           |
|  ASK                              BID                    |
|  $3.48   ||||     $42K            $3.36  ||    $12K      |
|  $3.46   |||      $18K            $3.34  |    $4K    <-V |
|  $3.44   ||       $8K             $3.32  |    $3K    <-V |
|  ─── $3.42 ───                    $3.30  [LIQ] $800K    |
|                                   $3.28  ||||||  $38K <-W|
|  Path DOWN: OPEN PATH ($4K vs $800K)                     |
|  Expected: -4.1% to wall at $3.28                        |
+----------------------------------------------------------+

Keyboard controls
-----------------
  Up / Down   move selection in scanner
  Enter       lock selection (stop auto-following top rank)
  Esc         return to auto-follow mode
  T           cycle tier filter: All → T2 only → T3 only → All
  R           force re-compute all features
  Q / Ctrl-C  quit
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..bybit.rest import BybitRestClient
from ..scanner.liquidity_map import (
    _bar_str,
    _fmt_price,
    _fmt_usdt,
    estimate_liquidation_levels,
    path_resistance,
)
from ..scanner.pressure_scanner import (
    CRITICAL_THRESHOLD,
    DISPLAY_INTERVAL,
    DISPLAY_INTERVAL_PRESETTL,
    HOT_THRESHOLD,
    PressureScanner,
    SymbolState,
)

# ── Optional keyboard support (Unix only) ─────────────────────────────────────

try:
    import termios
    import tty
    _HAS_TERMIOS = True
except ImportError:
    _HAS_TERMIOS = False

logger = logging.getLogger(__name__)

_CONFIG_DEFAULT = Path(__file__).resolve().parent.parent.parent / "configs" / "symbols.yaml"

# ── Display constants ─────────────────────────────────────────────────────────

_ASK_COL_W   = 28   # fixed character width for the ask column in the OB view
_MAX_BARS    = 6    # max bar chars for compact OB display
_COMPACT_LEVELS   = 8
_COMPACT_BAND_BPS_FALLBACK = 2.0


# ── Dashboard ─────────────────────────────────────────────────────────────────


class Dashboard(PressureScanner):
    """
    Full-screen interactive terminal dashboard.

    Inherits the complete data pipeline from ``PressureScanner`` (WebSocket,
    REST polling, ``SymbolState`` management, composite score computation).
    Overrides only the display layer to use Rich ``Live`` and adds keyboard
    input handling.
    """

    def __init__(
        self,
        symbols_by_tier: Dict[int, List[str]],
        rest: BybitRestClient,
    ) -> None:
        super().__init__(symbols_by_tier, rest)

        # ── UI state ──────────────────────────────────────────────────────────
        self._selected_sym: Optional[str] = None   # None → auto-follow top rank
        self._locked: bool = False
        self._tier_filter: str = "all"             # "all" | "t2" | "t3"

    # ── Selection helpers ─────────────────────────────────────────────────────

    def _filtered_sorted(self) -> List[SymbolState]:
        states = list(self._states.values())
        if self._tier_filter == "t2":
            states = [s for s in states if s.tier == 2]
        elif self._tier_filter == "t3":
            states = [s for s in states if s.tier == 3]
        return sorted(
            states,
            key=lambda s: s.rank if not math.isnan(s.rank) else -1.0,
            reverse=True,
        )

    def _current_symbol(self) -> Optional[str]:
        """Return the symbol shown in the liquidity panel."""
        if self._locked and self._selected_sym:
            return self._selected_sym
        ranked = self._filtered_sorted()
        return ranked[0].symbol if ranked else None

    def _selection_index(self) -> int:
        """Index of the selected symbol in the current filtered+sorted list."""
        ranked = self._filtered_sorted()
        sym = self._selected_sym or self._current_symbol()
        syms = [s.symbol for s in ranked]
        return syms.index(sym) if sym in syms else 0

    def _move_selection(self, delta: int) -> None:
        ranked = self._filtered_sorted()
        if not ranked:
            return
        idx = self._selection_index()
        new_idx = max(0, min(len(ranked) - 1, idx + delta))
        self._selected_sym = ranked[new_idx].symbol

    def _lock_selection(self) -> None:
        if self._selected_sym is None:
            self._selected_sym = self._current_symbol()
        self._locked = True

    def _unlock_selection(self) -> None:
        self._locked = False
        self._selected_sym = None

    def _toggle_tier_filter(self) -> None:
        cycle = ["all", "t2", "t3"]
        self._tier_filter = cycle[(cycle.index(self._tier_filter) + 1) % 3]

    def _force_refresh(self) -> None:
        """Re-compute all feature domains from the in-memory DataFrames."""
        for state in self._states.values():
            self._recompute_kline_features(state)
            self._recompute_oi_features(state)
            self._recompute_funding_features(state)
            if state.ob_feats:
                self._recompute_composite_scores(state)

    # ── Keyboard loop ─────────────────────────────────────────────────────────

    async def _handle_raw_key(self, raw: bytes) -> None:
        if raw in (b"q", b"Q", b"\x03"):     # Q or Ctrl-C
            self._running = False
        elif raw == b"\x1b[A":               # Up arrow
            self._move_selection(-1)
        elif raw == b"\x1b[B":               # Down arrow
            self._move_selection(1)
        elif raw in (b"\r", b"\n"):          # Enter → lock
            self._lock_selection()
        elif raw == b"\x1b":                 # Esc → unlock / auto-follow
            self._unlock_selection()
        elif raw in (b"r", b"R"):            # R → force refresh
            self._force_refresh()
        elif raw in (b"t", b"T"):            # T → cycle tier filter
            self._toggle_tier_filter()

    async def _keyboard_loop(self) -> None:
        if not _HAS_TERMIOS:
            logger.warning("Keyboard input not supported on this platform (no termios).")
            return

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        tty.setraw(fd)

        loop = asyncio.get_event_loop()
        key_queue: asyncio.Queue[bytes] = asyncio.Queue()

        def _on_readable() -> None:
            try:
                data = os.read(fd, 16)
                key_queue.put_nowait(data)
            except OSError:
                pass

        loop.add_reader(fd, _on_readable)
        try:
            while self._running:
                try:
                    raw = await asyncio.wait_for(key_queue.get(), timeout=0.2)
                    await self._handle_raw_key(raw)
                except asyncio.TimeoutError:
                    pass
        finally:
            loop.remove_reader(fd)
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    # ── Scanner panel ─────────────────────────────────────────────────────────

    def _render_scanner_panel(self, mins_to_settle: float) -> Panel:
        pre_settle = mins_to_settle <= 60.0
        mm = int(mins_to_settle)
        ss = int((mins_to_settle - mm) * 60)

        title = Text()
        title.append("PRESSURE SCANNER")
        title.append(f"  |  Next settle: {mm:02d}:{ss:02d}")
        if pre_settle:
            title.append("  (PRE-SETTLEMENT)", style="bold yellow")

        tier_label = {"all": "All tiers", "t2": "T2 only", "t3": "T3 only"}[
            self._tier_filter
        ]
        cur_sym = self._current_symbol()
        footer = Text()
        footer.append(f"Filter: {tier_label}")
        if self._locked and cur_sym:
            footer.append(f"  |  Locked: {cur_sym}", style="bold cyan")
        else:
            footer.append("  |  Auto-follow: top rank", style="dim")
        footer.append("   Up/Down select  Enter lock  Esc auto  T tier  R refresh  Q quit",
                      style="dim")

        table = Table(box=None, show_header=True, header_style="bold", padding=(0, 1))
        table.add_column("#",     width=3,  justify="right")
        table.add_column("Symbol",width=12)
        table.add_column("Tier",  width=4)
        table.add_column("Rank",  width=6,  justify="right")
        table.add_column("Dir",   width=5)
        table.add_column("SPS",   width=6,  justify="right")
        table.add_column("Comp",  width=6,  justify="right")
        table.add_column("LFI",   width=6,  justify="right")
        table.add_column("FZ",    width=6,  justify="right")

        for i, state in enumerate(self._filtered_sorted()):
            is_selected = state.symbol == cur_sym
            phase = state.funding_feats.get("settlement_phase", "")
            is_crit = (
                not math.isnan(state.rank)
                and state.rank > CRITICAL_THRESHOLD
                and phase == "IMMINENT"
            )
            is_hot = not math.isnan(state.rank) and state.rank > HOT_THRESHOLD

            sps = state.sps
            fz  = state.funding_feats.get("funding_z", float("nan"))

            rank_s = f"{state.rank:.0f}"       if not math.isnan(state.rank)       else "?"
            sps_s  = f"{sps:+.0f}"             if not math.isnan(sps)              else "?"
            comp_s = f"{state.compression:.0f}" if not math.isnan(state.compression) else "?"
            lfi_s  = f"{state.lfi:.0f}"        if not math.isnan(state.lfi)        else "?"
            fz_s   = f"{fz:+.1f}"             if not math.isnan(fz)               else "?"
            dir_s  = (
                "DOWN" if not math.isnan(sps) and sps < 0 else
                "UP"   if not math.isnan(sps) and sps > 0 else "NEUT"
            )

            if is_selected:
                row_style = "on dark_blue"
            elif is_crit:
                row_style = "bold red"
            elif is_hot:
                row_style = "bold yellow"
            elif i < 5:
                row_style = "cyan"
            else:
                row_style = ""

            table.add_row(
                str(i + 1), state.symbol, f"T{state.tier}",
                rank_s, dir_s, sps_s, comp_s, lfi_s, fz_s,
                style=row_style,
            )

        return Panel(table, title=title, subtitle=footer)

    # ── Liquidity panel ───────────────────────────────────────────────────────

    def _render_liquidity_panel(self, sym: Optional[str]) -> Panel:
        """Compact two-column orderbook with liquidation annotations."""
        auto_str = "" if self._locked else " (auto: top rank)"
        title = f"LIQUIDITY MAP: {sym or '—'}{auto_str}"

        if sym is None:
            return Panel("[dim]No symbol selected.[/dim]", title=title)

        state = self._states.get(sym)
        book  = self._ws.orderbooks.get(sym)

        if state is None or book is None or not book.bids or not book.asks:
            return Panel("[dim]Waiting for live orderbook data…[/dim]", title=title)

        bb = book.best_bid()
        ba = book.best_ask()
        if bb is None or ba is None:
            return Panel("[dim]Empty orderbook.[/dim]", title=title)

        mid        = (bb + ba) / 2.0
        spread_bps = (ba - bb) / mid * 10_000
        fund_z     = state.funding_feats.get("funding_z",              float("nan"))
        phase      = state.funding_feats.get("settlement_phase",       "?")
        mins       = state.funding_feats.get("minutes_to_settlement",  float("nan"))
        sps        = state.sps

        squeeze_down = not math.isnan(sps) and sps < 0
        squeeze_up   = not math.isnan(sps) and sps > 0

        # ── Aggregate into compact bands ──────────────────────────────────────
        raw_asks = book.get_sorted_asks(200)
        raw_bids = book.get_sorted_bids(200)

        ask_range = (raw_asks[-1][0] - raw_asks[0][0]) if len(raw_asks) >= 2 else 0
        bid_range = (raw_bids[0][0] - raw_bids[-1][0]) if len(raw_bids) >= 2 else 0
        ob_range  = max(ask_range, bid_range)

        if ob_range > 0 and mid > 0:
            band_size = ob_range / _COMPACT_LEVELS
        else:
            band_size = mid * _COMPACT_BAND_BPS_FALLBACK / 10_000

        def aggregate(levels: List[Tuple[float, float]], above: bool) -> List[Tuple[float, float]]:
            bands: Dict[float, float] = {}
            for price, size in levels:
                idx = round((price - mid) / band_size)
                bp  = mid + idx * band_size
                bands[bp] = bands.get(bp, 0.0) + price * size
            if above:
                return sorted(
                    [(p, d) for p, d in bands.items() if p > mid]
                )[:_COMPACT_LEVELS]
            return sorted(
                [(p, d) for p, d in bands.items() if p <= mid], reverse=True
            )[:_COMPACT_LEVELS]

        ask_bands = aggregate(raw_asks, above=True)
        bid_bands = aggregate(raw_bids, above=False)

        all_d  = [d for _, d in ask_bands + bid_bands]
        med_d  = float(np.median(all_d)) if all_d else 1.0
        max_d  = max(all_d) if all_d else 1.0

        def classify(d: float) -> str:
            if d < med_d * 0.30: return "vacuum"
            if d > med_d * 2.00: return "wall"
            return ""

        ask_tags = [classify(d) for _, d in ask_bands]
        bid_tags = [classify(d) for _, d in bid_bands]

        # ── Liquidation clusters ──────────────────────────────────────────────
        oi_cur  = state.oi_feats.get("oi_current", float("nan"))
        oi_usdt = oi_cur * mid if not math.isnan(oi_cur) and mid > 0 else 0.0
        liq_longs  = estimate_liquidation_levels(mid, oi_usdt * 0.5, "long")  if oi_usdt > 0 else []
        liq_shorts = estimate_liquidation_levels(mid, oi_usdt * 0.5, "short") if oi_usdt > 0 else []

        def liq_near(clusters: List[Tuple[float, float, int]], price: float) -> List[Tuple[float, float, int]]:
            return [(p, v, lev) for p, v, lev in clusters if abs(p - price) <= band_size * 1.5]

        # ── Squeeze target & path resistance ─────────────────────────────────
        def first_wall(bands: List[Tuple[float, float]], tags: List[str]) -> Optional[Tuple[float, float]]:
            seen_vacuum = False
            for (price, depth), tag in zip(bands, tags):
                if tag == "vacuum":
                    seen_vacuum = True
                elif seen_vacuum and tag == "wall":
                    return price, depth
            return None

        bid_wall = first_wall(bid_bands, bid_tags)
        ask_wall = first_wall(ask_bands, ask_tags)

        path_line: Optional[Text] = None
        if squeeze_down and bid_wall:
            cascade = sum(v for _, v, _ in liq_longs)
            pr = path_resistance(
                book, mid, bid_wall[0],
                cascade_volume=cascade if cascade > 0 else float("nan"),
            )
            label_style = (
                "bold green" if pr["label"] == "OPEN PATH" else
                "yellow"     if pr["label"] == "CONTESTED"  else "bold red"
            )
            path_line = Text(" Path DOWN: ")
            path_line.append(pr["label"] or "N/A", style=label_style)
            path_line.append(
                f" ({_fmt_usdt(pr['depth_usdt'])} vs {_fmt_usdt(cascade)})\n"
                f" Expected: -{pr['distance_pct']:.1f}% to wall at {_fmt_price(bid_wall[0])}"
            )
        elif squeeze_up and ask_wall:
            cascade = sum(v for _, v, _ in liq_shorts)
            pr = path_resistance(
                book, mid, ask_wall[0],
                cascade_volume=cascade if cascade > 0 else float("nan"),
            )
            label_style = (
                "bold green" if pr["label"] == "OPEN PATH" else
                "yellow"     if pr["label"] == "CONTESTED"  else "bold red"
            )
            path_line = Text(" Path UP: ")
            path_line.append(pr["label"] or "N/A", style=label_style)
            path_line.append(
                f" ({_fmt_usdt(pr['depth_usdt'])} vs {_fmt_usdt(cascade)})\n"
                f" Expected: +{pr['distance_pct']:.1f}% to wall at {_fmt_price(ask_wall[0])}"
            )

        # ── Build content as Text ─────────────────────────────────────────────
        content = Text()

        # Info row
        phase_style = (
            "bold red" if phase == "IMMINENT" else
            "yellow"   if phase == "APPROACH" else "white"
        )
        content.append(f" Price: {_fmt_price(mid)}   Spread: {spread_bps:.1f} bps   ")
        content.append(phase, style=phase_style)
        if not math.isnan(mins):
            content.append(f" ({int(mins)} min)")
        content.append("\n")

        # Score row
        sps_str   = f"{sps:+.1f}" if not math.isnan(sps) else "N/A"
        sps_dir   = "DOWN" if squeeze_down else ("UP" if squeeze_up else "NEUTRAL")
        sps_style = "red bold" if squeeze_down else ("green bold" if squeeze_up else "white")
        fz_s      = f"{fund_z:+.1f}" if not math.isnan(fund_z) else "N/A"
        content.append(" SPS: ")
        content.append(f"{sps_str} ({sps_dir})", style=sps_style)
        content.append(f"   FZ: {fz_s}\n")

        # Column headers
        ask_hdr = " ASK"
        bid_hdr = " BID"
        content.append(f"{ask_hdr:<{_ASK_COL_W}}", style="bold red")
        content.append(bid_hdr, style="bold green")
        content.append("\n")

        # ── Orderbook rows (two-column: ask left, bid right) ──────────────────
        n_ask  = len(ask_bands)
        n_bid  = len(bid_bands)
        n_rows = max(n_ask, n_bid) + 1   # +1 for the separator row

        for i in range(n_rows):
            # ASK column: index n_ask-1-i (furthest ask at row 0 → nearest at row n_ask-1)
            ask_idx = n_ask - 1 - i
            if 0 <= ask_idx < n_ask:
                price, depth = ask_bands[ask_idx]
                bar = _bar_str(depth, max_d, _MAX_BARS)
                amt = _fmt_usdt(depth)
                cell = f" {_fmt_price(price):<10} {bar:<{_MAX_BARS + 1}}{amt}"
                content.append(f"{cell:<{_ASK_COL_W}}", style="red")
            elif ask_idx < 0:
                # Separator row
                sep = f" ─── {_fmt_price(mid)} ───"
                content.append(f"{sep:<{_ASK_COL_W}}", style="bold white")
            else:
                content.append(" " * _ASK_COL_W)

            # BID column: bids[i] (nearest at row 0, furthest at bottom)
            if i < n_bid:
                price, depth = bid_bands[i]
                tag = bid_tags[i]
                bar = _bar_str(depth, max_d, _MAX_BARS)
                amt = _fmt_usdt(depth)
                cell = f" {_fmt_price(price):<10} {bar:<{_MAX_BARS + 1}}{amt}"
                content.append(cell, style="green")

                # Annotations
                nearby_liq    = liq_near(liq_longs, price)
                is_bid_wall   = bid_wall is not None and abs(price - bid_wall[0]) < 1e-8

                if nearby_liq:
                    _, lv, lev = nearby_liq[0]
                    content.append(f" [LIQ] {_fmt_usdt(lv)} ({lev}x)", style="yellow")
                elif is_bid_wall and squeeze_down:
                    content.append(" <-- WALL/TARGET", style="bold magenta")
                elif tag == "vacuum":
                    content.append(" <-- VACUUM", style="dim yellow")

            content.append("\n")

        # Path resistance summary
        if path_line is not None:
            content.append("\n")
            content.append_text(path_line)

        return Panel(content, title=title)

    # ── Layout ────────────────────────────────────────────────────────────────

    def _render(self, mins_to_settle: float) -> Layout:
        sym = self._current_symbol()
        layout = Layout()
        layout.split_column(
            Layout(name="scanner",   ratio=1),
            Layout(name="liquidity", ratio=1),
        )
        layout["scanner"].update(self._render_scanner_panel(mins_to_settle))
        layout["liquidity"].update(self._render_liquidity_panel(sym))
        return layout

    # ── Display loop (overrides PressureScanner's ANSI version) ──────────────

    async def _display_loop(self) -> None:
        console = Console()
        with Live(console=console, refresh_per_second=4, screen=True) as live:
            while self._running:
                mins = self._minutes_to_settlement()
                live.update(self._render(mins))
                await asyncio.sleep(
                    DISPLAY_INTERVAL_PRESETTL if mins <= 60.0 else DISPLAY_INTERVAL
                )

    # ── Run (override: adds keyboard task) ────────────────────────────────────

    async def run(self) -> None:
        self._running = True
        self._load_parquets()
        self._compute_initial_features()

        tasks = [
            asyncio.create_task(self._ws.run(),        name="ws"),
            asyncio.create_task(self._poll_oi(),       name="oi-poll"),
            asyncio.create_task(self._poll_funding(),  name="fund-poll"),
            asyncio.create_task(self._poll_klines(),   name="kline-poll"),
            asyncio.create_task(self._display_loop(),  name="display"),
            asyncio.create_task(self._keyboard_loop(), name="keyboard"),
        ]

        try:
            await asyncio.gather(*tasks)
        except (asyncio.CancelledError, KeyboardInterrupt):
            pass
        finally:
            self._running = False
            self._ws.stop()
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


# ── CLI ───────────────────────────────────────────────────────────────────────


def _load_symbols_yaml(path: Path) -> Dict[int, List[str]]:
    import yaml
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    result: Dict[int, List[str]] = {}
    for tier in (1, 2, 3):
        key = f"tier_{tier}"
        if key in cfg:
            result[tier] = list(cfg[key])
    if not result:
        raise ValueError(f"No tier_1/tier_2/tier_3 keys found in {path}")
    return result


async def _main(args: argparse.Namespace) -> None:
    if args.symbols:
        symbols_by_tier: Dict[int, List[str]] = {3: list(args.symbols)}
    elif args.config.exists():
        symbols_by_tier = _load_symbols_yaml(args.config)
    else:
        print(f"Config not found: {args.config}", file=sys.stderr)
        print("Run backfill first:  python -m src.bybit.backfill", file=sys.stderr)
        sys.exit(1)

    async with BybitRestClient() as rest:
        dashboard = Dashboard(symbols_by_tier, rest)
        await dashboard.run()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bybit Perpetuals Live Dashboard",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_CONFIG_DEFAULT,
        help="Path to symbols.yaml (generated by backfill)",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        metavar="SYM",
        help="Override symbol list, e.g. --symbols BTCUSDT ETHUSDT",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )

    try:
        asyncio.run(_main(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
