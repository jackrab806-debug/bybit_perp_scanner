"""Download historical data from Bybit for backtesting.

Downloads 1h klines, 5m klines (90d), funding, and OI for top symbols.
Saves to data/historical/{symbol}/*.parquet. Skips existing files.

Usage:
    python -m src.backtest.download_history --days 365 --symbols 150
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List

import aiohttp
import pandas as pd

logger = logging.getLogger(__name__)

_BASE = "https://api.bybit.com"
_ROOT = Path(__file__).resolve().parent.parent.parent
HIST_DIR = _ROOT / "data" / "historical"


async def get_top_symbols(
    session: aiohttp.ClientSession, limit: int = 150, min_turnover: float = 5e6,
) -> List[str]:
    async with session.get(
        f"{_BASE}/v5/market/tickers", params={"category": "linear"}
    ) as resp:
        data = await resp.json()
    tickers = data.get("result", {}).get("list", [])
    pairs = []
    for t in tickers:
        try:
            tv = float(t.get("turnover24h", 0))
            if tv >= min_turnover and t["symbol"].endswith("USDT"):
                pairs.append((t["symbol"], tv))
        except (ValueError, KeyError):
            pass
    pairs.sort(key=lambda x: -x[1])
    return [s for s, _ in pairs[:limit]]


async def _paginate_klines(
    session: aiohttp.ClientSession, symbol: str, interval: str,
    start_ms: int, end_ms: int,
) -> pd.DataFrame:
    rows: list = []
    cur_end = end_ms
    while cur_end > start_ms:
        try:
            async with session.get(
                f"{_BASE}/v5/market/kline",
                params={
                    "category": "linear", "symbol": symbol,
                    "interval": interval, "end": cur_end, "limit": 1000,
                },
            ) as resp:
                data = await resp.json()
            kl = data.get("result", {}).get("list", [])
            if not kl:
                break
            rows.extend(kl)
            oldest = int(kl[-1][0])
            if oldest >= cur_end:
                break
            cur_end = oldest
            await asyncio.sleep(0.15)
        except Exception as e:
            logger.warning("Kline %s %s: %s", symbol, interval, e)
            break
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=[
        "timestamp", "open", "high", "low", "close", "volume", "turnover",
    ])
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    return df[(df["timestamp"] >= start_ms) & (df["timestamp"] <= end_ms)]


async def _paginate_funding(
    session: aiohttp.ClientSession, symbol: str, start_ms: int, end_ms: int,
) -> pd.DataFrame:
    rows: list = []
    cur_end = end_ms
    while cur_end > start_ms:
        try:
            async with session.get(
                f"{_BASE}/v5/market/funding/history",
                params={
                    "category": "linear", "symbol": symbol,
                    "endTime": cur_end, "limit": 200,
                },
            ) as resp:
                data = await resp.json()
            recs = data.get("result", {}).get("list", [])
            if not recs:
                break
            rows.extend(recs)
            oldest = int(recs[-1]["fundingRateTimestamp"])
            if oldest >= cur_end:
                break
            cur_end = oldest
            await asyncio.sleep(0.15)
        except Exception as e:
            logger.warning("Funding %s: %s", symbol, e)
            break
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_numeric(df["fundingRateTimestamp"])
    df["funding_rate"] = pd.to_numeric(df["fundingRate"])
    return df[["timestamp", "funding_rate"]].drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


async def _paginate_oi(
    session: aiohttp.ClientSession, symbol: str, start_ms: int, end_ms: int,
    interval: str = "4h",
) -> pd.DataFrame:
    rows: list = []
    cur_end = end_ms
    while cur_end > start_ms:
        try:
            async with session.get(
                f"{_BASE}/v5/market/open-interest",
                params={
                    "category": "linear", "symbol": symbol,
                    "intervalTime": interval, "endTime": cur_end, "limit": 200,
                },
            ) as resp:
                data = await resp.json()
            recs = data.get("result", {}).get("list", [])
            if not recs:
                break
            rows.extend(recs)
            oldest = int(recs[-1]["timestamp"])
            if oldest >= cur_end:
                break
            cur_end = oldest
            await asyncio.sleep(0.15)
        except Exception as e:
            logger.warning("OI %s: %s", symbol, e)
            break
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_numeric(df["timestamp"])
    df["oi"] = pd.to_numeric(df["openInterest"])
    return df[["timestamp", "oi"]].drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


async def download_symbol(
    session: aiohttp.ClientSession, symbol: str, days: int = 365,
) -> None:
    sym_dir = HIST_DIR / symbol
    sym_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    end_ms = int(now.timestamp() * 1000)
    start_ms = int((now - timedelta(days=days)).timestamp() * 1000)

    # 1h klines
    p = sym_dir / "klines_1h.parquet"
    if not p.exists():
        df = await _paginate_klines(session, symbol, "60", start_ms, end_ms)
        if not df.empty:
            df.to_parquet(p)
            logger.info("  %s 1h: %d rows", symbol, len(df))

    # 5m klines (90 days only)
    p = sym_dir / "klines_5m.parquet"
    if not p.exists():
        start_5m = int((now - timedelta(days=90)).timestamp() * 1000)
        df = await _paginate_klines(session, symbol, "5", start_5m, end_ms)
        if not df.empty:
            df.to_parquet(p)
            logger.info("  %s 5m: %d rows", symbol, len(df))

    # Funding
    p = sym_dir / "funding.parquet"
    if not p.exists():
        df = await _paginate_funding(session, symbol, start_ms, end_ms)
        if not df.empty:
            df.to_parquet(p)
            logger.info("  %s funding: %d rows", symbol, len(df))

    # OI
    p = sym_dir / "oi.parquet"
    if not p.exists():
        df = await _paginate_oi(session, symbol, start_ms, end_ms)
        if not df.empty:
            df.to_parquet(p)
            logger.info("  %s OI: %d rows", symbol, len(df))


async def download_all(days: int = 365, n_symbols: int = 150) -> None:
    HIST_DIR.mkdir(parents=True, exist_ok=True)
    timeout = aiohttp.ClientTimeout(total=15)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        symbols = await get_top_symbols(session, limit=n_symbols)
        logger.info("Downloading %d symbols (%d days)", len(symbols), days)

        for i, sym in enumerate(symbols):
            try:
                await download_symbol(session, sym, days)
                logger.info("[%d/%d] %s done", i + 1, len(symbols), sym)
            except Exception as e:
                logger.error("Failed %s: %s", sym, e)
            await asyncio.sleep(0.3)

    logger.info("Download complete")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=365)
    p.add_argument("--symbols", type=int, default=150)
    args = p.parse_args()
    asyncio.run(download_all(days=args.days, n_symbols=args.symbols))
