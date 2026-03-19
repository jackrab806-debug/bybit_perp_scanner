"""Historical data backfill for Bybit USDT perpetuals.

Downloads funding rates, open interest, and OHLCV klines for the
top-50 symbols by 24h turnover and saves them as Parquet files.

Usage:
    python -m src.bybit.backfill           # normal run
    python -m src.bybit.backfill --force   # overwrite existing files
    python -m src.bybit.backfill --days 30 # shorter lookback

Output:
    configs/symbols.yaml
    data/raw/funding/{SYMBOL}.parquet
    data/raw/oi/{SYMBOL}.parquet
    data/raw/klines/{SYMBOL}.parquet
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

from .rest import BybitRestClient, BybitAPIError

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw"
CONFIGS_DIR = ROOT / "configs"

# ── Parameters ────────────────────────────────────────────────────────────────

TOP_N = 250
TIER_1_THRESHOLD = 500_000_000  # $500 M
TIER_2_THRESHOLD = 50_000_000   # $50 M
TIER_3_THRESHOLD = 1_000_000    # $1 M

DEFAULT_LOOKBACK_DAYS = 90
DEFAULT_1M_LOOKBACK_DAYS = 180  # 6 months for 1-minute klines
KLINES_INTERVAL = "60"          # 1-hour candles
KLINES_1M_INTERVAL = "1"        # 1-minute candles
OI_INTERVAL = "1h"
MAX_CONCURRENT = 5              # parallel symbol downloads


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _ms_to_str(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


# ── Universe selection ────────────────────────────────────────────────────────

async def build_universe(client: BybitRestClient) -> Dict[str, Any]:
    """Fetch top-50 USDT perpetuals by 24h turnover, classify into tiers,
    save configs/symbols.yaml, and return the universe dict."""

    log.info("Fetching tickers for universe selection …")
    data = await client._get("/v5/market/tickers", {"category": "linear"})

    tickers: List[Dict[str, Any]] = []
    for t in data.get("list", []):
        sym = t.get("symbol", "")
        if not sym.endswith("USDT"):
            continue
        try:
            turnover = float(t.get("turnover24h") or 0)
        except (ValueError, TypeError):
            continue
        tickers.append({"symbol": sym, "turnover24h": turnover})

    tickers.sort(key=lambda x: x["turnover24h"], reverse=True)
    top = tickers[:TOP_N]

    tier_1, tier_2, tier_3 = [], [], []
    for t in top:
        v = t["turnover24h"]
        if v >= TIER_1_THRESHOLD:
            tier_1.append(t["symbol"])
        elif v >= TIER_2_THRESHOLD:
            tier_2.append(t["symbol"])
        else:
            tier_3.append(t["symbol"])

    log.info(
        "Universe: %d symbols  |  T1=%d (>$500M)  T2=%d (>$50M)  T3=%d (>$5M)",
        len(top), len(tier_1), len(tier_2), len(tier_3),
    )
    if top:
        log.info("  Top symbol: %s  ($%.0f turnover)", top[0]["symbol"], top[0]["turnover24h"])

    universe: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tier_1": tier_1,
        "tier_2": tier_2,
        "tier_3": tier_3,
    }

    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    out = CONFIGS_DIR / "symbols.yaml"
    with open(out, "w") as fh:
        yaml.dump(universe, fh, default_flow_style=False, allow_unicode=True, sort_keys=False)
    log.info("Saved universe → %s", out.relative_to(ROOT))

    return universe


# ── Historical data fetchers ──────────────────────────────────────────────────

async def _fetch_funding(
    client: BybitRestClient,
    symbol: str,
    start_ms: int,
    end_ms: int,
) -> pd.DataFrame:
    """Paginate through /v5/market/funding/history and return a DataFrame."""

    records: List[Dict] = []
    cursor: Optional[str] = None
    page = 0

    while True:
        params: Dict[str, Any] = {
            "category": "linear",
            "symbol": symbol,
            "startTime": str(start_ms),
            "endTime": str(end_ms),
            "limit": 200,
        }
        if cursor:
            params["cursor"] = cursor

        data = await client._get("/v5/market/funding/history", params)
        batch: List[Dict] = data.get("list", [])
        records.extend(batch)
        cursor = data.get("nextPageCursor") or ""
        page += 1

        log.info(
            "  [funding] %-12s  page %2d  +%3d rows  (total %d)",
            symbol, page, len(batch), len(records),
        )

        if not batch or not cursor:
            break
        await asyncio.sleep(0.05)

    if not records:
        return pd.DataFrame(columns=["timestamp", "symbol", "funding_rate"])

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(
        df["fundingRateTimestamp"].astype("int64"), unit="ms", utc=True
    )
    df["symbol"] = symbol
    df["funding_rate"] = df["fundingRate"].astype("float64")
    return (
        df[["timestamp", "symbol", "funding_rate"]]
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )


async def _fetch_oi(
    client: BybitRestClient,
    symbol: str,
    start_ms: int,
    end_ms: int,
) -> pd.DataFrame:
    """Paginate through /v5/market/open-interest (1h) and return a DataFrame."""

    records: List[Dict] = []
    cursor: Optional[str] = None
    page = 0

    while True:
        params: Dict[str, Any] = {
            "category": "linear",
            "symbol": symbol,
            "intervalTime": OI_INTERVAL,
            "startTime": str(start_ms),
            "endTime": str(end_ms),
            "limit": 200,
        }
        if cursor:
            params["cursor"] = cursor

        data = await client._get("/v5/market/open-interest", params)
        batch: List[Dict] = data.get("list", [])
        records.extend(batch)
        cursor = data.get("nextPageCursor") or ""
        page += 1

        log.info(
            "  [oi]      %-12s  page %2d  +%3d rows  (total %d)",
            symbol, page, len(batch), len(records),
        )

        if not batch or not cursor:
            break
        await asyncio.sleep(0.05)

    if not records:
        return pd.DataFrame(columns=["timestamp", "symbol", "open_interest"])

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(
        df["timestamp"].astype("int64"), unit="ms", utc=True
    )
    df["symbol"] = symbol
    df["open_interest"] = df["openInterest"].astype("float64")
    return (
        df[["timestamp", "symbol", "open_interest"]]
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )


async def _fetch_klines(
    client: BybitRestClient,
    symbol: str,
    start_ms: int,
    end_ms: int,
) -> pd.DataFrame:
    """Paginate through /v5/market/kline (1h) backwards and return a DataFrame.

    Bybit returns newest-first up to 1000 bars per request. We page backwards
    by setting end = earliest_received_timestamp - 1ms each iteration.
    """

    all_rows: List[List[str]] = []
    current_end = end_ms
    page = 0

    while current_end > start_ms:
        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": KLINES_INTERVAL,
            "start": str(start_ms),
            "end": str(current_end),
            "limit": 1000,
        }

        data = await client._get("/v5/market/kline", params)
        batch: List[List[str]] = data.get("list", [])

        if not batch:
            break

        all_rows.extend(batch)
        page += 1
        earliest_ms = int(batch[-1][0])

        log.info(
            "  [klines]  %-12s  page %2d  +%4d bars  (total %d, back to %s)",
            symbol, page, len(batch), len(all_rows), _ms_to_str(earliest_ms),
        )

        if earliest_ms <= start_ms or len(batch) < 1000:
            break

        current_end = earliest_ms - 1
        await asyncio.sleep(0.05)

    if not all_rows:
        return pd.DataFrame(
            columns=["timestamp", "symbol", "open", "high", "low", "close", "volume", "turnover"]
        )

    # Bybit kline row: [startTime, open, high, low, close, volume, turnover]
    df = pd.DataFrame(all_rows, columns=["ts", "open", "high", "low", "close", "volume", "turnover"])
    df["timestamp"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms", utc=True)
    df["symbol"] = symbol
    for col in ["open", "high", "low", "close", "volume", "turnover"]:
        df[col] = df[col].astype("float64")

    return (
        df[["timestamp", "symbol", "open", "high", "low", "close", "volume", "turnover"]]
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )


async def _fetch_klines_1m(
    client: BybitRestClient,
    symbol: str,
    start_ms: int,
    end_ms: int,
) -> pd.DataFrame:
    """Paginate through /v5/market/kline (1m) backwards and return a DataFrame.

    Same logic as _fetch_klines but with 1-minute interval.
    ~130 pages per 90 days, ~260 pages per 180 days.
    """
    all_rows: List[List[str]] = []
    current_end = end_ms
    page = 0

    while current_end > start_ms:
        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": KLINES_1M_INTERVAL,
            "start": str(start_ms),
            "end": str(current_end),
            "limit": 1000,
        }

        data = await client._get("/v5/market/kline", params)
        batch: List[List[str]] = data.get("list", [])

        if not batch:
            break

        all_rows.extend(batch)
        page += 1
        earliest_ms = int(batch[-1][0])

        if page % 25 == 0 or len(batch) < 1000:
            log.info(
                "  [klines1m] %-12s  page %3d  total %6d bars  (back to %s)",
                symbol, page, len(all_rows), _ms_to_str(earliest_ms),
            )

        if earliest_ms <= start_ms or len(batch) < 1000:
            break

        current_end = earliest_ms - 1
        await asyncio.sleep(0.02)

    if not all_rows:
        return pd.DataFrame(
            columns=["timestamp", "symbol", "open", "high", "low", "close", "volume", "turnover"]
        )

    df = pd.DataFrame(all_rows, columns=["ts", "open", "high", "low", "close", "volume", "turnover"])
    df["timestamp"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms", utc=True)
    df["symbol"] = symbol
    for col in ["open", "high", "low", "close", "volume", "turnover"]:
        df[col] = df[col].astype("float64")

    return (
        df[["timestamp", "symbol", "open", "high", "low", "close", "volume", "turnover"]]
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )


# ── Save helper ───────────────────────────────────────────────────────────────

def _save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow")


# ── Per-symbol backfill task ──────────────────────────────────────────────────

async def _backfill_symbol(
    client: BybitRestClient,
    symbol: str,
    start_ms: int,
    end_ms: int,
    force: bool,
    sem: asyncio.Semaphore,
    idx: int,
    total: int,
    *,
    start_1m_ms: Optional[int] = None,
    skip_1m: bool = False,
) -> None:
    async with sem:
        log.info("▶ [%d/%d] %s", idx, total, symbol)

        # ── Funding ──────────────────────────────────────────────────────────
        out = DATA_DIR / "funding" / f"{symbol}.parquet"
        if force or not out.exists():
            try:
                df = await _fetch_funding(client, symbol, start_ms, end_ms)
                _save_parquet(df, out)
                log.info("  ✓ funding  %-12s  %d rows → %s", symbol, len(df), out.relative_to(ROOT))
            except Exception as exc:
                log.error("  ✗ funding  %-12s  %s", symbol, exc)
        else:
            log.info("  ↷ funding  %-12s  exists, skipping", symbol)

        # ── Open Interest ─────────────────────────────────────────────────────
        out = DATA_DIR / "oi" / f"{symbol}.parquet"
        if force or not out.exists():
            try:
                df = await _fetch_oi(client, symbol, start_ms, end_ms)
                _save_parquet(df, out)
                log.info("  ✓ oi       %-12s  %d rows → %s", symbol, len(df), out.relative_to(ROOT))
            except Exception as exc:
                log.error("  ✗ oi       %-12s  %s", symbol, exc)
        else:
            log.info("  ↷ oi       %-12s  exists, skipping", symbol)

        # ── Klines (1h) ──────────────────────────────────────────────────────
        out = DATA_DIR / "klines" / f"{symbol}.parquet"
        if force or not out.exists():
            try:
                df = await _fetch_klines(client, symbol, start_ms, end_ms)
                _save_parquet(df, out)
                log.info("  ✓ klines   %-12s  %d rows → %s", symbol, len(df), out.relative_to(ROOT))
            except Exception as exc:
                log.error("  ✗ klines   %-12s  %s", symbol, exc)
        else:
            log.info("  ↷ klines   %-12s  exists, skipping", symbol)

        # ── Klines (1m) ──────────────────────────────────────────────────────
        if not skip_1m:
            out_1m = DATA_DIR / "klines_1m" / f"{symbol}.parquet"
            s1m = start_1m_ms if start_1m_ms is not None else start_ms
            if force or not out_1m.exists():
                try:
                    df = await _fetch_klines_1m(client, symbol, s1m, end_ms)
                    _save_parquet(df, out_1m)
                    log.info("  ✓ klines1m %-12s  %d rows → %s", symbol, len(df), out_1m.relative_to(ROOT))
                except Exception as exc:
                    log.error("  ✗ klines1m %-12s  %s", symbol, exc)
            else:
                log.info("  ↷ klines1m %-12s  exists, skipping", symbol)

        log.info("◀ [%d/%d] %s  done", idx, total, symbol)


# ── Main ─────────────────────────────────────────────────────────────────────

async def run(
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    force: bool = False,
    skip_1m: bool = False,
    lookback_1m_days: int = DEFAULT_1M_LOOKBACK_DAYS,
) -> None:
    end_ms = _now_ms()
    start_ms = end_ms - lookback_days * 24 * 3600 * 1000
    start_1m_ms = end_ms - lookback_1m_days * 24 * 3600 * 1000

    log.info(
        "Backfill  %s → %s  (%d days, 1m: %s%d days)",
        _ms_to_str(start_ms), _ms_to_str(end_ms), lookback_days,
        "skip" if skip_1m else "", lookback_1m_days,
    )

    async with BybitRestClient() as client:
        universe = await build_universe(client)
        symbols = universe["tier_1"] + universe["tier_2"] + universe["tier_3"]
        log.info("Downloading %d symbols (max %d concurrent) …", len(symbols), MAX_CONCURRENT)

        sem = asyncio.Semaphore(MAX_CONCURRENT)
        tasks = [
            _backfill_symbol(
                client, sym, start_ms, end_ms, force, sem, i + 1, len(symbols),
                start_1m_ms=start_1m_ms, skip_1m=skip_1m,
            )
            for i, sym in enumerate(symbols)
        ]
        await asyncio.gather(*tasks)

    log.info("All done. Files in %s/", DATA_DIR.relative_to(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Bybit historical data backfill")
    parser.add_argument(
        "--days", type=int, default=DEFAULT_LOOKBACK_DAYS,
        help=f"Lookback period in days for 1h data (default: {DEFAULT_LOOKBACK_DAYS})",
    )
    parser.add_argument(
        "--days-1m", type=int, default=DEFAULT_1M_LOOKBACK_DAYS,
        help=f"Lookback period in days for 1m klines (default: {DEFAULT_1M_LOOKBACK_DAYS})",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing Parquet files",
    )
    parser.add_argument(
        "--skip-1m", action="store_true",
        help="Skip 1-minute kline download (faster for testing)",
    )
    args = parser.parse_args()
    asyncio.run(run(
        lookback_days=args.days,
        force=args.force,
        skip_1m=args.skip_1m,
        lookback_1m_days=args.days_1m,
    ))


if __name__ == "__main__":
    main()
