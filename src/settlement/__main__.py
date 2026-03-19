"""
Run a settlement scan manually, without waiting for the next settlement.

Symbols are discovered dynamically from Bybit tickers API (turnover > $1M).
Use --symbols to override with a specific list.

Usage:
    python -m src.settlement
    python -m src.settlement --symbols BTCUSDT ETHUSDT SOLUSDT
    python -m src.settlement --wait 15   # wait 15s for OB data (default: 30)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from typing import Dict, List

from ..bybit.rest import BybitRestClient
from ..scanner.pressure_scanner import PressureScanner
from .scanner import discover_symbols
from .scheduler import SettlementScheduler

logger = logging.getLogger(__name__)


async def _main(args: argparse.Namespace) -> None:
    async with BybitRestClient() as rest:
        # Resolve symbol universe
        if args.symbols:
            symbols_by_tier: Dict[int, List[str]] = {3: list(args.symbols)}
            n_syms = len(args.symbols)
            print(f"Using {n_syms} manual symbols (all tier 3)")
        else:
            print("Discovering symbols from Bybit ...")
            symbols_by_tier = await discover_symbols(rest)
            if not symbols_by_tier:
                print("No symbols found. Check network / API.", file=sys.stderr)
                sys.exit(1)
            n_syms = sum(len(v) for v in symbols_by_tier.values())
            t1 = len(symbols_by_tier.get(1, []))
            t2 = len(symbols_by_tier.get(2, []))
            t3 = len(symbols_by_tier.get(3, []))
            print(f"Scanning {n_syms} symbols ({t1} tier1, {t2} tier2, {t3} tier3)")

        scanner = PressureScanner(symbols_by_tier, rest)
        scanner._running = True
        scanner._load_parquets()
        scanner._compute_initial_features()

        # Start WS to get live orderbook data
        ws_task = asyncio.create_task(scanner._ws.run(), name="ws")

        # Also kick off REST polls for funding + OI immediately
        # (parquet data may be stale or absent for dynamically discovered symbols)
        poll_tasks = [
            asyncio.create_task(_quick_poll_funding(rest, scanner)),
            asyncio.create_task(_quick_poll_oi(rest, scanner)),
        ]

        wait = args.wait
        print(f"Collecting live data ... ({wait}s)")
        await asyncio.sleep(wait)

        # Cancel polls
        for t in poll_tasks:
            t.cancel()

        # Run scan
        scheduler = SettlementScheduler(scanner)
        await scheduler.run_once()

        # Cleanup
        scanner._running = False
        scanner._ws.stop()
        ws_task.cancel()
        try:
            await asyncio.gather(ws_task, *poll_tasks, return_exceptions=True)
        except asyncio.CancelledError:
            pass


async def _quick_poll_funding(rest: BybitRestClient, scanner: PressureScanner) -> None:
    """Fetch latest funding rate for all symbols once at startup."""
    import pandas as pd
    from datetime import datetime, timezone

    for sym, state in scanner._states.items():
        try:
            rate = await rest.get_latest_funding(sym)
            if rate is None:
                continue
            new_row = pd.DataFrame([{
                "timestamp": pd.Timestamp.now(tz="UTC"),
                "symbol": sym,
                "funding_rate": float(rate),
            }])
            if state.funding_df is None or state.funding_df.empty:
                state.funding_df = new_row
            else:
                state.funding_df = pd.concat([state.funding_df, new_row], ignore_index=True)
            scanner._recompute_funding_features(state)
        except Exception:
            pass
        await asyncio.sleep(0.05)


async def _quick_poll_oi(rest: BybitRestClient, scanner: PressureScanner) -> None:
    """Fetch latest OI for all symbols once at startup."""
    import pandas as pd

    for sym, state in scanner._states.items():
        try:
            cur, _ = await rest.get_oi_last_prev(sym)
            if cur is None:
                continue
            new_row = pd.DataFrame([{
                "timestamp": pd.Timestamp.now(tz="UTC"),
                "symbol": sym,
                "open_interest": float(cur),
            }])
            if state.oi_df is None or state.oi_df.empty:
                state.oi_df = new_row
            else:
                state.oi_df = pd.concat([state.oi_df, new_row], ignore_index=True)
            scanner._recompute_oi_features(state)
        except Exception:
            pass
        await asyncio.sleep(0.05)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m src.settlement",
        description="Run a one-shot pre-settlement squeeze scan",
    )
    parser.add_argument(
        "--symbols", nargs="+", metavar="SYM",
        help="Override: scan only these symbols (assigned to tier 3)",
    )
    parser.add_argument(
        "--wait", type=int, default=30,
        help="Seconds to wait for WS orderbook data before scanning (default: 30)",
    )
    parser.add_argument(
        "--log-level", default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )

    try:
        asyncio.run(_main(args))
    except KeyboardInterrupt:
        print("\nScan cancelled.")


if __name__ == "__main__":
    main()
