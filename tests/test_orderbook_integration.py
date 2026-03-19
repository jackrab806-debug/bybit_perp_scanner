"""
Orderbook integration tests — REST vs WebSocket comparison, stress test,
and multi-symbol validation.

Run:
    python -m tests.test_orderbook_integration          # all tests
    python -m tests.test_orderbook_integration --quick   # skip long-running tests
"""

import asyncio
import argparse
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List

sys.path.insert(0, ".")

from src.bybit.rest import BybitRestClient
from src.bybit.ws import BybitWebSocketClient, LocalOrderbook


# ── Helpers ───────────────────────────────────────────────────────────────────

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
SHUTDOWN_TIMEOUT = 5


async def shutdown_ws(ws_client: BybitWebSocketClient, ws_task: asyncio.Task) -> None:
    """Stop a WS client and cancel its task with a hard timeout."""
    ws_client.stop()
    ws_task.cancel()
    try:
        await asyncio.wait_for(asyncio.shield(ws_task), timeout=SHUTDOWN_TIMEOUT)
    except (asyncio.CancelledError, asyncio.TimeoutError):
        pass


@dataclass
class TestResult:
    name: str
    passed: bool
    details: List[str] = field(default_factory=list)


def banner(title: str) -> None:
    sep = "=" * 70
    print(f"\n{sep}\n  {title}\n{sep}")


# ── Test 1-3: REST vs WS orderbook comparison ────────────────────────────────


async def test_rest_vs_ws(client: BybitRestClient) -> TestResult:
    """
    1. Fetch a REST orderbook snapshot for BTCUSDT.
    2. Start WS client and let it build a local book for ~10 s.
    3. Compare best bid/ask within one tick.
    """
    banner("TEST 1-3: REST vs WS orderbook comparison (BTCUSDT)")
    result = TestResult(name="REST vs WS comparison", passed=False)

    tick_size = await client.get_tick_size("BTCUSDT")
    result.details.append(f"Tick size: {tick_size}")

    # --- REST snapshot ---
    rest_book = await client.get_orderbook("BTCUSDT", limit=50)
    rest_bids = rest_book.get("b", [])
    rest_asks = rest_book.get("a", [])

    if not rest_bids or not rest_asks:
        result.details.append("REST orderbook empty — cannot proceed")
        return result

    rest_best_bid = max(float(p) for p, _ in rest_bids)
    rest_best_ask = min(float(p) for p, _ in rest_asks)
    result.details.append(f"REST best bid: {rest_best_bid}")
    result.details.append(f"REST best ask: {rest_best_ask}")

    # --- WS local book ---
    ws_ready = asyncio.Event()
    ws_update_count = 0

    def on_ob(symbol: str, book: LocalOrderbook) -> None:
        nonlocal ws_update_count
        ws_update_count += 1
        if ws_update_count >= 2:
            ws_ready.set()

    ws_client = BybitWebSocketClient(
        symbols=["BTCUSDT"],
        orderbook_depth=50,
        on_orderbook=on_ob,
    )
    ws_task = asyncio.create_task(ws_client.run())

    try:
        # Wait up to 15 s for initial data, then let it accumulate
        await asyncio.wait_for(ws_ready.wait(), timeout=15)
        print("  WS connected — accumulating data for 10 s …")
        await asyncio.sleep(10)

        book = ws_client.orderbooks.get("BTCUSDT")
        if book is None or not book.bids or not book.asks:
            result.details.append("WS orderbook empty after 10 s")
            return result

        ws_best_bid = book.best_bid()
        ws_best_ask = book.best_ask()
        result.details.append(f"WS  best bid: {ws_best_bid}")
        result.details.append(f"WS  best ask: {ws_best_ask}")

        # Take a fresh REST snapshot right before comparison
        rest_book2 = await client.get_orderbook("BTCUSDT", limit=50)
        rest_best_bid2 = max(float(p) for p, _ in rest_book2.get("b", []))
        rest_best_ask2 = min(float(p) for p, _ in rest_book2.get("a", []))
        result.details.append(f"REST best bid (fresh): {rest_best_bid2}")
        result.details.append(f"REST best ask (fresh): {rest_best_ask2}")

        # Allow a generous tolerance of 5 ticks to account for network latency
        tolerance = tick_size * 5
        bid_diff = abs(ws_best_bid - rest_best_bid2)
        ask_diff = abs(ws_best_ask - rest_best_ask2)

        bid_ok = bid_diff <= tolerance
        ask_ok = ask_diff <= tolerance
        result.details.append(
            f"Bid diff: {bid_diff:.4f} (tolerance {tolerance:.4f}) → "
            f"{'MATCH' if bid_ok else 'MISMATCH'}"
        )
        result.details.append(
            f"Ask diff: {ask_diff:.4f} (tolerance {tolerance:.4f}) → "
            f"{'MATCH' if ask_ok else 'MISMATCH'}"
        )
        result.details.append(
            f"Snapshots: {book.snapshot_count}, "
            f"Deltas: {book.delta_count}, "
            f"Deletes: {book.delete_count}"
        )
        result.passed = bid_ok and ask_ok

    finally:
        await shutdown_ws(ws_client, ws_task)

    return result


# ── Test 4: 10-minute stress test ────────────────────────────────────────────


async def test_stress(client: BybitRestClient, duration_s: int = 600) -> TestResult:
    """
    Run WS for *duration_s* seconds and log:
      - Snapshots vs deltas received
      - Size-0 deletes handled
      - Best bid/ask every 30 s (must never be None or crossed)
      - Reconnect count (should be 0)
    """
    banner(f"TEST 4: Stress test — BTCUSDT for {duration_s // 60} min")
    result = TestResult(name="Stress test", passed=True)

    ws_client = BybitWebSocketClient(
        symbols=["BTCUSDT"],
        orderbook_depth=50,
    )
    ws_task = asyncio.create_task(ws_client.run())

    # Wait for initial data
    await asyncio.sleep(5)

    start = time.monotonic()
    log_interval = 30
    next_log = start + log_interval
    crossed_count = 0
    none_count = 0
    sample_count = 0

    try:
        while time.monotonic() - start < duration_s:
            await asyncio.sleep(1)
            now = time.monotonic()

            if now >= next_log:
                sample_count += 1
                book = ws_client.orderbooks.get("BTCUSDT")
                if book is None:
                    none_count += 1
                    elapsed = int(now - start)
                    msg = f"  [{elapsed:>4d}s]  book=None"
                    print(msg)
                    result.details.append(msg)
                    next_log = now + log_interval
                    continue

                bb = book.best_bid()
                ba = book.best_ask()
                elapsed = int(now - start)

                if bb is None or ba is None:
                    none_count += 1
                    msg = (
                        f"  [{elapsed:>4d}s]  bid={bb}  ask={ba}  "
                        f"snaps={book.snapshot_count}  deltas={book.delta_count}  "
                        f"deletes={book.delete_count}  ⚠ None detected"
                    )
                elif bb >= ba:
                    crossed_count += 1
                    msg = (
                        f"  [{elapsed:>4d}s]  bid={bb}  ask={ba}  "
                        f"snaps={book.snapshot_count}  deltas={book.delta_count}  "
                        f"deletes={book.delete_count}  ⚠ CROSSED"
                    )
                else:
                    msg = (
                        f"  [{elapsed:>4d}s]  bid={bb}  ask={ba}  "
                        f"snaps={book.snapshot_count}  deltas={book.delta_count}  "
                        f"deletes={book.delete_count}"
                    )
                print(msg)
                result.details.append(msg)
                next_log = now + log_interval

    finally:
        await shutdown_ws(ws_client, ws_task)

    book = ws_client.orderbooks.get("BTCUSDT")
    total_snaps = book.snapshot_count if book else 0
    total_deltas = book.delta_count if book else 0
    total_deletes = book.delete_count if book else 0
    reconnects = ws_client.reconnect_count

    summary_lines = [
        f"Total snapshots : {total_snaps}",
        f"Total deltas    : {total_deltas}",
        f"Total deletes   : {total_deletes}",
        f"Reconnects      : {reconnects}",
        f"Crossed samples : {crossed_count}/{sample_count}",
        f"None samples    : {none_count}/{sample_count}",
    ]
    for line in summary_lines:
        print(f"  {line}")
        result.details.append(line)

    if crossed_count > 0:
        result.passed = False
        result.details.append("FAIL: crossed book detected")
    if none_count > 0:
        result.passed = False
        result.details.append("FAIL: None bid/ask detected")
    if reconnects > 0:
        result.passed = False
        result.details.append(f"FAIL: {reconnects} reconnects during test")

    return result


# ── Test 5: Multi-symbol test ────────────────────────────────────────────────


async def test_multi_symbol(duration_s: int = 300) -> TestResult:
    """
    Subscribe to 5 symbols for *duration_s* seconds.
    Validate:
      - All 5 have valid orderbooks at all times.
      - No crossed books (bid >= ask).
      - Log message rate per symbol.
    """
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "SUIUSDT"]
    banner(
        f"TEST 5: Multi-symbol test — {', '.join(symbols)} for {duration_s // 60} min"
    )
    result = TestResult(name="Multi-symbol test", passed=True)

    msg_counts: Dict[str, int] = {s: 0 for s in symbols}

    def on_ob(symbol: str, book: LocalOrderbook) -> None:
        if symbol in msg_counts:
            msg_counts[symbol] += 1

    ws_client = BybitWebSocketClient(
        symbols=symbols,
        orderbook_depth=50,
        on_orderbook=on_ob,
    )
    ws_task = asyncio.create_task(ws_client.run())

    await asyncio.sleep(8)

    start = time.monotonic()
    log_interval = 30
    next_log = start + log_interval
    crossed_events: Dict[str, int] = {s: 0 for s in symbols}
    missing_events: Dict[str, int] = {s: 0 for s in symbols}
    sample_count = 0

    try:
        while time.monotonic() - start < duration_s:
            await asyncio.sleep(1)
            now = time.monotonic()

            if now >= next_log:
                sample_count += 1
                elapsed = int(now - start)
                print(f"  [{elapsed:>4d}s]")

                for sym in symbols:
                    book = ws_client.orderbooks.get(sym)
                    if book is None or not book.bids or not book.asks:
                        missing_events[sym] += 1
                        rate = msg_counts[sym] / max(1, elapsed)
                        print(f"    {sym:>10s}  ⚠ EMPTY  msgs={msg_counts[sym]}  rate={rate:.1f}/s")
                        continue

                    bb = book.best_bid()
                    ba = book.best_ask()

                    if bb is None or ba is None:
                        missing_events[sym] += 1
                        status = "⚠ None"
                    elif bb >= ba:
                        crossed_events[sym] += 1
                        status = "⚠ CROSSED"
                    else:
                        status = "OK"

                    rate = msg_counts[sym] / max(1, elapsed)
                    print(
                        f"    {sym:>10s}  bid={bb}  ask={ba}  "
                        f"msgs={msg_counts[sym]}  rate={rate:.1f}/s  {status}"
                    )

                next_log = now + log_interval

    finally:
        await shutdown_ws(ws_client, ws_task)

    total_elapsed = time.monotonic() - start
    print("\n  ── Per-symbol summary ──")
    for sym in symbols:
        book = ws_client.orderbooks.get(sym)
        snaps = book.snapshot_count if book else 0
        deltas = book.delta_count if book else 0
        deletes = book.delete_count if book else 0
        rate = msg_counts[sym] / max(1, total_elapsed)
        line = (
            f"    {sym:>10s}  snaps={snaps}  deltas={deltas}  "
            f"deletes={deletes}  rate={rate:.1f}/s  "
            f"crossed={crossed_events[sym]}  missing={missing_events[sym]}"
        )
        print(line)
        result.details.append(line)

    reconnects = ws_client.reconnect_count
    result.details.append(f"Reconnects: {reconnects}")
    print(f"  Reconnects: {reconnects}")

    for sym in symbols:
        if crossed_events[sym] > 0:
            result.passed = False
            result.details.append(f"FAIL: {sym} had {crossed_events[sym]} crossed samples")
        if missing_events[sym] > 0:
            result.passed = False
            result.details.append(f"FAIL: {sym} had {missing_events[sym]} missing/None samples")
    if reconnects > 0:
        result.passed = False
        result.details.append(f"FAIL: {reconnects} reconnects")

    return result


# ── Main ──────────────────────────────────────────────────────────────────────


async def main(quick: bool = False) -> None:
    results: List[TestResult] = []

    async with BybitRestClient() as client:
        # Test 1-3: REST vs WS
        results.append(await test_rest_vs_ws(client))

        # Test 4: Stress test
        stress_duration = 60 if quick else 600
        results.append(await test_stress(client, duration_s=stress_duration))

    # Test 5: Multi-symbol (no REST needed)
    multi_duration = 60 if quick else 300
    results.append(await test_multi_symbol(duration_s=multi_duration))

    # ── Final summary ─────────────────────────────────────────────────
    banner("SUMMARY")
    all_passed = True
    for r in results:
        status = PASS if r.passed else FAIL
        print(f"  {status}  {r.name}")
        for d in r.details:
            print(f"         {d}")
        if not r.passed:
            all_passed = False

    print()
    if all_passed:
        print(f"  Overall: {PASS} — all tests passed")
    else:
        print(f"  Overall: {FAIL} — one or more tests failed")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orderbook integration tests")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run stress/multi tests for 1 min instead of 10/5 min",
    )
    args = parser.parse_args()
    asyncio.run(main(quick=args.quick))
