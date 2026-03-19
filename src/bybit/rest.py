"""Bybit v5 REST client with token-bucket rate limiting."""

import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple

import aiohttp


class BybitAPIError(Exception):
    """Raised when the Bybit API returns a non-zero retCode or HTTP error."""


class TokenBucket:
    """
    Async token-bucket rate limiter.

    Capacity  : 600 tokens  (full burst allowance)
    Refill rate: 120 tokens/second  (600 / 5 s window)
    """

    capacity: int = 600
    refill_rate: float = 120.0  # tokens per second

    def __init__(self) -> None:
        self._tokens: float = float(self.capacity)
        self._last_refill: float = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> None:
        """
        Consume *tokens* from the bucket.

        Blocks until the requested number of tokens are available.
        Never allows the bucket to exceed *capacity*.
        """
        async with self._lock:
            while True:
                now = time.monotonic()
                elapsed = now - self._last_refill
                self._tokens = min(
                    float(self.capacity),
                    self._tokens + elapsed * self.refill_rate,
                )
                self._last_refill = now

                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return

                # Sleep for the exact time needed to accumulate the deficit.
                deficit = tokens - self._tokens
                wait = deficit / self.refill_rate
                await asyncio.sleep(wait)


class BybitRestClient:
    """
    Async Bybit v5 public REST client with integrated rate limiting.

    Usage (single shared instance):
        async with BybitRestClient() as client:
            symbols = await client.list_usdt_linear_symbols()
    """

    BASE_URL = "https://api.bybit.com"
    TIMEOUT = aiohttp.ClientTimeout(total=10)

    def __init__(self) -> None:
        self._rate_limiter = TokenBucket()
        self._session: Optional[aiohttp.ClientSession] = None

    # ------------------------------------------------------------------
    # Context-manager helpers
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "BybitRestClient":
        self._session = aiohttp.ClientSession(
            headers={"User-Agent": "bybit-perp-scanner/2.0"},
            timeout=self.TIMEOUT,
        )
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _get(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Rate-limited GET request.

        Returns the ``result`` payload from a successful Bybit response.
        Raises :class:`BybitAPIError` on HTTP or API-level failures.
        """
        await self._rate_limiter.acquire()

        if self._session is None:
            raise RuntimeError("Client not started – use 'async with BybitRestClient()'")

        url = f"{self.BASE_URL}{endpoint}"
        try:
            async with self._session.get(url, params=params) as resp:
                resp.raise_for_status()
                data: Dict[str, Any] = await resp.json()
        except aiohttp.ClientResponseError as exc:
            raise BybitAPIError(f"HTTP {exc.status} from {url}") from exc
        except aiohttp.ClientError as exc:
            raise BybitAPIError(f"Request failed: {exc}") from exc

        if data.get("retCode") != 0:
            raise BybitAPIError(f"API error: {data.get('retMsg', 'unknown')}")

        return data.get("result", {})

    # ------------------------------------------------------------------
    # Public API methods
    # ------------------------------------------------------------------

    async def list_usdt_linear_symbols(self) -> List[str]:
        """Return sorted list of active USDT-margined perpetual symbols."""
        try:
            data = await self._get(
                "/v5/market/instruments-info", {"category": "linear"}
            )
            return sorted(
                inst["symbol"]
                for inst in data.get("list", [])
                if inst.get("quoteCoin") == "USDT"
                and inst.get("status") == "Trading"
            )
        except BybitAPIError as exc:
            print(f"Error fetching symbols: {exc}")
            return []

    async def get_bulk_tickers(self) -> Dict[str, Dict[str, Any]]:
        """
        Return ticker data keyed by symbol.

        Each value contains: ``turnover24h``, ``markPrice``, ``indexPrice``.
        """
        try:
            data = await self._get("/v5/market/tickers", {"category": "linear"})
            tickers: Dict[str, Dict[str, Any]] = {}
            for t in data.get("list", []):
                sym = t.get("symbol")
                if sym:
                    tickers[sym] = {
                        "turnover24h": float(t.get("turnover24h", 0)),
                        "markPrice": float(t["markPrice"]) if t.get("markPrice") else None,
                        "indexPrice": float(t["indexPrice"]) if t.get("indexPrice") else None,
                    }
            return tickers
        except BybitAPIError as exc:
            print(f"Error fetching bulk tickers: {exc}")
            return {}

    async def get_1h_quote_volume(self, symbol: str) -> Optional[float]:
        """Sum turnover of the last four 15-minute klines to get 1-hour volume."""
        try:
            data = await self._get(
                "/v5/market/kline",
                {
                    "category": "linear",
                    "symbol": symbol,
                    "interval": "15",
                    "limit": 4,
                },
            )
            klines = data.get("list", [])
            if len(klines) < 4:
                return None
            return sum(float(k[6]) for k in klines if len(k) > 6)
        except (BybitAPIError, ValueError, IndexError) as exc:
            print(f"Error fetching 1h volume for {symbol}: {exc}")
            return None

    async def get_oi_last_prev(
        self, symbol: str
    ) -> Tuple[Optional[float], Optional[float]]:
        """Return (current_oi, previous_oi) from the last two hourly OI entries."""
        try:
            data = await self._get(
                "/v5/market/open-interest",
                {
                    "category": "linear",
                    "symbol": symbol,
                    "intervalTime": "1h",
                    "limit": 2,
                },
            )
            oi_list = data.get("list", [])
            if not oi_list:
                return None, None
            current = float(oi_list[0].get("openInterest", 0))
            previous = float(oi_list[1].get("openInterest", 0)) if len(oi_list) >= 2 else None
            return current, previous
        except (BybitAPIError, ValueError, IndexError) as exc:
            print(f"Error fetching OI for {symbol}: {exc}")
            return None, None

    async def get_orderbook(
        self, symbol: str, limit: int = 50
    ) -> Dict[str, Any]:
        """
        Return the current orderbook snapshot for *symbol*.

        Returns a dict with ``b`` (bids) and ``a`` (asks),
        each a list of ``[price_str, size_str]`` pairs.
        """
        data = await self._get(
            "/v5/market/orderbook",
            {"category": "linear", "symbol": symbol, "limit": limit},
        )
        return data

    async def get_tick_size(self, symbol: str) -> float:
        """Return the tick size (minimum price increment) for *symbol*."""
        data = await self._get(
            "/v5/market/instruments-info",
            {"category": "linear", "symbol": symbol},
        )
        inst_list = data.get("list", [])
        if not inst_list:
            raise BybitAPIError(f"No instrument info for {symbol}")
        price_filter = inst_list[0].get("priceFilter", {})
        return float(price_filter["tickSize"])

    async def get_latest_funding(self, symbol: str) -> Optional[float]:
        """
        Return the most recent funding rate.

        Returns ``None`` for symbols without funding history (404 treated as
        missing data, not an error).
        """
        await self._rate_limiter.acquire()

        if self._session is None:
            raise RuntimeError("Client not started – use 'async with BybitRestClient()'")

        url = f"{self.BASE_URL}/v5/market/funding/history"
        params = {"category": "linear", "symbol": symbol, "limit": 1}
        try:
            async with self._session.get(url, params=params) as resp:
                if resp.status == 404:
                    return None
                resp.raise_for_status()
                data = await resp.json()
        except aiohttp.ClientError as exc:
            print(f"Error fetching funding for {symbol}: {exc}")
            return None

        if data.get("retCode") != 0:
            return None

        funding_list = data.get("result", {}).get("list", [])
        if not funding_list:
            return None
        raw = funding_list[0].get("fundingRate")
        return float(raw) if raw else None
