"""5-Minute Kline Collector.

Fetches and maintains rolling 5m kline history for tracked symbols.
Only collects for symbols with active events (saves bandwidth).
Uses aiohttp (already a project dependency).
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Set

import aiohttp

logger = logging.getLogger(__name__)

KLINE_HISTORY = 200  # 200 x 5min = ~16 hours


@dataclass
class Kline:
    """Single 5-minute candle."""
    timestamp: int      # Unix ms
    open: float
    high: float
    low: float
    close: float
    volume: float
    turnover: float

    @property
    def body_size(self) -> float:
        return abs(self.close - self.open)

    @property
    def range_size(self) -> float:
        return self.high - self.low

    @property
    def body_ratio(self) -> float:
        """Body as fraction of total range. >0.6 = impulsive."""
        return self.body_size / self.range_size if self.range_size > 0 else 0

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        return self.close < self.open


class KlineCollector5m:
    """Maintains 5m kline history for active symbols via Bybit REST."""

    def __init__(self) -> None:
        self.klines: Dict[str, deque] = {}
        self._active_symbols: Set[str] = set()
        self._running = False

    def set_active_symbols(self, symbols: Set[str]) -> None:
        self._active_symbols = symbols
        for s in symbols:
            if s not in self.klines:
                self.klines[s] = deque(maxlen=KLINE_HISTORY)

    async def run_loop(self, interval_seconds: int = 300) -> None:
        self._running = True
        logger.info("KlineCollector5m started")
        while self._running:
            symbols = list(self._active_symbols)
            if symbols:
                await self._fetch_batch(symbols)
            await asyncio.sleep(interval_seconds)

    async def _fetch_batch(self, symbols: List[str]) -> None:
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for symbol in symbols:
                try:
                    async with session.get(
                        "https://api.bybit.com/v5/market/kline",
                        params={
                            "category": "linear",
                            "symbol": symbol,
                            "interval": "5",
                            "limit": 50,
                        },
                    ) as resp:
                        if resp.status != 200:
                            continue
                        data = await resp.json()

                    kline_list = data.get("result", {}).get("list", [])
                    if not kline_list:
                        continue

                    # Bybit returns newest first — reverse
                    kline_list.reverse()

                    existing_ts: set = set()
                    dq = self.klines.setdefault(symbol, deque(maxlen=KLINE_HISTORY))
                    existing_ts = {k.timestamp for k in dq}

                    for k in kline_list:
                        ts = int(k[0])
                        if ts not in existing_ts:
                            dq.append(Kline(
                                timestamp=ts,
                                open=float(k[1]),
                                high=float(k[2]),
                                low=float(k[3]),
                                close=float(k[4]),
                                volume=float(k[5]),
                                turnover=float(k[6]),
                            ))

                    await asyncio.sleep(0.1)
                except Exception:
                    logger.debug("Kline fetch error %s", symbol, exc_info=True)

    def get_klines(self, symbol: str, count: int = 50) -> List[Kline]:
        dq = self.klines.get(symbol)
        if not dq:
            return []
        kl = list(dq)
        return kl[-count:] if len(kl) >= count else kl

    def stop(self) -> None:
        self._running = False
