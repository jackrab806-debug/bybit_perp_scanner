"""Bybit v5 public WebSocket client with local orderbook management."""

import asyncio
import contextlib
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WS_URL = "wss://stream.bybit.com/v5/public/linear"

PING_INTERVAL = 20    # send ping every N seconds
NO_MSG_TIMEOUT = 30   # force reconnect if silent for N seconds
BATCH_SIZE = 10       # max topics per subscribe message
MAX_BACKOFF = 30      # exponential backoff ceiling (seconds)

# ---------------------------------------------------------------------------
# Local orderbook
# ---------------------------------------------------------------------------


class LocalOrderbook:
    """
    Maintains a live, consistent local copy of one symbol's order book.

    Bybit sends:
    - A full **snapshot** (type="snapshot") on first subscription.
    - Incremental **deltas** (type="delta") afterwards.

    A size of 0 in a delta means "delete that price level".
    """

    def __init__(self) -> None:
        self.bids: Dict[float, float] = {}   # price → size
        self.asks: Dict[float, float] = {}   # price → size
        self.snapshot_count: int = 0
        self.delta_count: int = 0
        self.delete_count: int = 0

    def on_snapshot(self, data: Dict[str, Any]) -> None:
        """Replace the entire book with snapshot data."""
        self.bids = {float(p): float(s) for p, s in data["b"]}
        self.asks = {float(p): float(s) for p, s in data["a"]}
        self.snapshot_count += 1

    def on_delta(self, data: Dict[str, Any]) -> None:
        """Apply an incremental update to the local book."""
        for p, s in data["b"]:
            price, size = float(p), float(s)
            if size == 0:
                self.bids.pop(price, None)   # DELETE
                self.delete_count += 1
            else:
                self.bids[price] = size      # UPDATE / INSERT

        for p, s in data["a"]:
            price, size = float(p), float(s)
            if size == 0:
                self.asks.pop(price, None)
                self.delete_count += 1
            else:
                self.asks[price] = size
        self.delta_count += 1

    def get_sorted_bids(self, n: int = 50) -> List[tuple]:
        """Return up to *n* bid levels sorted best (highest) first."""
        return sorted(self.bids.items(), reverse=True)[:n]

    def get_sorted_asks(self, n: int = 50) -> List[tuple]:
        """Return up to *n* ask levels sorted best (lowest) first."""
        return sorted(self.asks.items())[:n]

    def best_bid(self) -> Optional[float]:
        return max(self.bids, default=None)

    def best_ask(self) -> Optional[float]:
        return min(self.asks, default=None)


# ---------------------------------------------------------------------------
# Callback type aliases
# ---------------------------------------------------------------------------

OrderbookCallback = Callable[[str, LocalOrderbook], None]
MessageCallback = Callable[[str, Dict[str, Any]], None]

# ---------------------------------------------------------------------------
# WebSocket client
# ---------------------------------------------------------------------------


class BybitWebSocketClient:
    """
    Persistent WebSocket connection to Bybit's public linear stream.

    Features
    --------
    - Local orderbook (snapshot + delta) per subscribed symbol.
    - Batch subscription (10 topics per message).
    - Ping every 20 s to keep the connection alive.
    - Force-reconnect if no message received for 30 s.
    - Exponential backoff on reconnect: 1 s, 2 s, 4 s, 8 s … capped at 30 s.

    Parameters
    ----------
    symbols:
        List of symbol names to subscribe to (e.g. ``["BTCUSDT", "ETHUSDT"]``).
    extra_topics:
        Additional raw topic strings appended to the subscription list,
        e.g. ``["tickers.BTCUSDT", "publicTrade.ETHUSDT"]``.
    orderbook_depth:
        Depth level for orderbook subscriptions (1, 50, 200, 500).
    on_orderbook:
        Called with ``(symbol, LocalOrderbook)`` on every orderbook update.
    on_message:
        Called with ``(topic, raw_message_dict)`` for all non-orderbook topics
        and unrecognised messages (excluding ping/pong/subscription acks).
    """

    def __init__(
        self,
        symbols: List[str],
        *,
        extra_topics: Optional[List[str]] = None,
        orderbook_depth: int = 50,
        on_orderbook: Optional[OrderbookCallback] = None,
        on_message: Optional[MessageCallback] = None,
    ) -> None:
        self._symbols = list(symbols)
        self._extra_topics = extra_topics or []
        self._depth = orderbook_depth
        self._on_orderbook = on_orderbook
        self._on_message = on_message

        self.orderbooks: Dict[str, LocalOrderbook] = {}
        self.reconnect_count: int = 0
        self.last_reconnect_ts: Optional[datetime] = None  # UTC; set on every reconnect

        self._running = False
        self._received_message = False   # used for backoff reset logic
        self._active_ws: Optional[aiohttp.ClientWebSocketResponse] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_topics(self) -> List[str]:
        topics = [f"orderbook.{self._depth}.{sym}" for sym in self._symbols]
        topics.extend(self._extra_topics)
        return topics

    async def _subscribe(self, ws: aiohttp.ClientWebSocketResponse, topics: List[str]) -> None:
        """Send subscription requests in batches of BATCH_SIZE."""
        for i in range(0, len(topics), BATCH_SIZE):
            batch = topics[i : i + BATCH_SIZE]
            payload = json.dumps({"op": "subscribe", "args": batch})
            await ws.send_str(payload)
            logger.debug("Subscribed batch (%d topics): %s", len(batch), batch)

    async def _ping_loop(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        """Send a ping every PING_INTERVAL seconds."""
        while True:
            await asyncio.sleep(PING_INTERVAL)
            try:
                await ws.send_str(json.dumps({"op": "ping"}))
                logger.debug("Ping sent")
            except Exception:
                break  # connection gone; exit so the task can be cancelled

    def _handle_raw(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Non-JSON message received: %r", raw)
            return

        # Swallow op-level responses (ping/pong, subscription acks)
        if msg.get("op") in ("ping", "pong", "subscribe"):
            return

        topic: str = msg.get("topic", "")
        data: Dict[str, Any] = msg.get("data", {})
        msg_type: str = msg.get("type", "")

        if topic.startswith("orderbook."):
            # topic format: "orderbook.<depth>.<symbol>"
            parts = topic.split(".", 2)
            if len(parts) == 3:
                symbol = parts[2]
                book = self.orderbooks.setdefault(symbol, LocalOrderbook())
                if msg_type == "snapshot":
                    book.on_snapshot(data)
                elif msg_type == "delta":
                    book.on_delta(data)
                if self._on_orderbook:
                    self._on_orderbook(symbol, book)
        else:
            if self._on_message:
                self._on_message(topic, msg)

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def _connect_once(self) -> bool:
        """
        Open one WebSocket connection and process messages until it drops.

        Returns ``True`` if at least one application message was received
        (used to decide whether to reset the backoff counter).
        """
        topics = self._build_topics()
        received_any = False
        ping_task: Optional[asyncio.Task] = None

        async with aiohttp.ClientSession() as session:
            try:
                ws: aiohttp.ClientWebSocketResponse = await session.ws_connect(
                    WS_URL,
                    heartbeat=None,   # we handle keepalive ourselves
                )
            except Exception as exc:
                logger.error("Failed to connect: %s", exc)
                return False

            logger.info("WebSocket connected to %s", WS_URL)
            self._active_ws = ws

            try:
                await self._subscribe(ws, topics)
                ping_task = asyncio.create_task(self._ping_loop(ws))

                while True:
                    try:
                        msg = await asyncio.wait_for(
                            ws.receive(), timeout=NO_MSG_TIMEOUT
                        )
                    except asyncio.TimeoutError:
                        logger.warning(
                            "No message for %ds – forcing reconnect", NO_MSG_TIMEOUT
                        )
                        break

                    if msg.type == aiohttp.WSMsgType.TEXT:
                        self._handle_raw(msg.data)
                        received_any = True
                    elif msg.type == aiohttp.WSMsgType.BINARY:
                        pass  # Bybit does not use binary frames
                    elif msg.type in (
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.CLOSE,
                        aiohttp.WSMsgType.ERROR,
                    ):
                        logger.info("WebSocket closed (type=%s)", msg.type)
                        break

            finally:
                self._active_ws = None
                if ping_task:
                    ping_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await ping_task
                with contextlib.suppress(BaseException):
                    await asyncio.wait_for(ws.close(), timeout=2)

        return received_any

    async def run(self) -> None:
        """
        Run forever, reconnecting with exponential backoff.

        Call :meth:`stop` (or cancel the coroutine) to terminate.
        """
        self._running = True
        attempt = 0

        while self._running:
            received_any = await self._connect_once()

            if not self._running:
                break

            # Reset backoff after a connection that actually got messages
            if received_any:
                attempt = 0

            self.reconnect_count += 1
            self.last_reconnect_ts = datetime.now(timezone.utc)
            delay = min(2 ** attempt, MAX_BACKOFF)
            logger.info(
                "Reconnecting in %.1fs (attempt %d)…", delay, attempt + 1
            )
            await asyncio.sleep(delay)
            attempt += 1

    async def add_symbols(self, new_symbols: List[str], extra_topics: Optional[List[str]] = None) -> None:
        """Add symbols and optionally extra topics to the live connection.

        Updates internal lists (so reconnects pick them up) and subscribes
        on the current connection if one is active.
        """
        added = [s for s in new_symbols if s not in self._symbols]
        if not added and not extra_topics:
            return

        new_topics: List[str] = []
        for s in added:
            self._symbols.append(s)
            new_topics.append(f"orderbook.{self._depth}.{s}")

        if extra_topics:
            for t in extra_topics:
                if t not in self._extra_topics:
                    self._extra_topics.append(t)
                    new_topics.append(t)

        if new_topics and self._active_ws and not self._active_ws.closed:
            await self._subscribe(self._active_ws, new_topics)
            logger.info("Dynamically subscribed %d new topics (%d symbols)", len(new_topics), len(added))

    def stop(self) -> None:
        """Signal the run loop to exit after the current connection drops."""
        self._running = False
