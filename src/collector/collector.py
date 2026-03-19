"""Headless 24/7 data collector for Bybit USDT perpetuals.

Collects three streams per symbol and writes them to daily Parquet files:

  ob_snapshots.parquet   — 1-second orderbook feature snapshots
                           (depth-50 book: best bid/ask, depths, spread,
                            imbalance, thin_pct, vacuum_dist)
  trade_aggs.parquet     — 1-second trade aggregates
                           (buy/sell volume, CVD, VWAP, sweep_score)
  liquidations.parquet   — per-minute liquidation aggregates
                           (long/short liq volume, counts, max size)

Events detected by EventDetector are written to SQLite via AlertManager
(shared with the dashboard at data/events.db).

Output layout
-------------
  data/collected/
    YYYY-MM-DD/
      ob_snapshots.parquet
      trade_aggs.parquet
      liquidations.parquet

Usage
-----
  python -m src.collector                         # uses configs/symbols.yaml
  python -m src.collector --symbols BTCUSDT ETHUSDT
  python -m src.collector --flush-interval 300    # flush every 5 min (default)
  python -m src.collector --log-level DEBUG
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
import time
from collections import deque, defaultdict
from datetime import datetime, date, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from ..bybit.rest import BybitRestClient
from ..bybit.ws import BybitWebSocketClient
from ..events.definitions import AlertManager, EventDetector
from ..scanner.pressure_scanner import PressureScanner, SymbolState
from ..settlement.scheduler import SettlementScheduler

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────

_ROOT        = Path(__file__).resolve().parent.parent.parent
_CONFIG_DEF  = _ROOT / "configs" / "symbols.yaml"
_COLLECT_DIR = _ROOT / "data" / "collected"

# ── Timing ────────────────────────────────────────────────────────────────────

_SNAPSHOT_INTERVAL  = 1.0    # seconds between OB + trade snapshots
_FLUSH_INTERVAL     = 300    # seconds between Parquet flushes (5 min)
_OB_STALE_THRESH    = 5.0    # skip OB snapshot if book not updated in this many seconds
_TRADE_SIZE_HISTORY = 1_000  # rolling trade sizes kept for sweep-score computation
_MAX_BUF_ROWS       = 50_000 # force flush if buffer exceeds this many rows

# ── Pyarrow schemas ───────────────────────────────────────────────────────────

_OB_SCHEMA = pa.schema([
    ("timestamp",        pa.timestamp("us", tz="UTC")),
    ("symbol",           pa.string()),
    ("bid1_price",       pa.float64()),
    ("bid1_size",        pa.float64()),
    ("ask1_price",       pa.float64()),
    ("ask1_size",        pa.float64()),
    ("mid_price",        pa.float64()),
    ("spread_bps",       pa.float64()),
    ("bid_depth_usdt",   pa.float64()),
    ("ask_depth_usdt",   pa.float64()),
    ("depth_imbalance",  pa.float64()),
    ("thin_pct",         pa.float64()),
    ("vacuum_dist_bid",  pa.float64()),
    ("vacuum_dist_ask",  pa.float64()),
])

_TRADE_SCHEMA = pa.schema([
    ("timestamp",       pa.timestamp("us", tz="UTC")),
    ("symbol",          pa.string()),
    ("buy_vol",         pa.float64()),
    ("sell_vol",        pa.float64()),
    ("cvd",             pa.float64()),
    ("trade_count",     pa.int32()),
    ("vwap",            pa.float64()),
    ("sweep_score",     pa.float64()),
    ("total_vol_usdt",  pa.float64()),
])

_LIQ_SCHEMA = pa.schema([
    ("timestamp",          pa.timestamp("us", tz="UTC")),
    ("symbol",             pa.string()),
    ("long_liq_vol_usdt",  pa.float64()),   # "Buy" side = long liq = forced sell
    ("short_liq_vol_usdt", pa.float64()),   # "Sell" side = short liq = forced buy
    ("long_liq_count",     pa.int32()),
    ("short_liq_count",    pa.int32()),
    ("max_single_liq_usdt",pa.float64()),
])


# ── DataCollector ─────────────────────────────────────────────────────────────


class DataCollector(PressureScanner):
    """
    Headless 24/7 data collector.

    Inherits the full feature-computation and event-detection pipeline from
    ``PressureScanner``.  Overrides the WebSocket construction to add trade
    and liquidation topic subscriptions, removes the display loop, and adds:

      * _snapshot_loop  — 1 s timer: snapshot OB features + trade aggregates
      * _liq_agg_loop   — 60 s timer: aggregate and record liquidation bars
      * _flush_loop     — periodic Parquet writer flush

    Parameters
    ----------
    symbols_by_tier :
        Dict[tier_int, List[symbol_str]] passed to PressureScanner.
    rest :
        Shared BybitRestClient instance.
    output_dir :
        Root directory for daily Parquet output (default: data/collected/).
    event_detector :
        Optional EventDetector for real-time event logging.
    alert_manager :
        Optional AlertManager (SQLite + CSV + webhook).
    flush_interval :
        Seconds between Parquet flushes to disk.
    ob_depth :
        Orderbook depth for WS subscription (50 recommended for collection).
    """

    def __init__(
        self,
        symbols_by_tier:  Dict[int, List[str]],
        rest:             BybitRestClient,
        *,
        output_dir:       Path = _COLLECT_DIR,
        event_detector:   Optional[EventDetector] = None,
        alert_manager:    Optional[AlertManager]  = None,
        flush_interval:   int  = _FLUSH_INTERVAL,
        ob_depth:         int  = 50,
    ) -> None:
        super().__init__(
            symbols_by_tier,
            rest,
            event_detector=event_detector,
            alert_manager=alert_manager,
        )

        self._output_dir    = Path(output_dir)
        self._flush_interval = flush_interval
        self._ob_depth      = ob_depth

        all_syms = list(self._states.keys())

        # ── Replace base-class WS with one that also subscribes to
        #    publicTrade and liquidation topics ─────────────────────────────
        extra_topics = (
            [f"publicTrade.{s}" for s in all_syms]
            + [f"liquidation.{s}" for s in all_syms]
        )
        self._ws = BybitWebSocketClient(
            symbols=all_syms,
            extra_topics=extra_topics,
            orderbook_depth=ob_depth,
            on_orderbook=self._on_orderbook,
            on_message=self._on_raw_message,
        )

        # ── Trade buffers (drained every second in _snapshot_loop) ─────────
        self._trade_buf: Dict[str, List[Dict[str, Any]]] = {s: [] for s in all_syms}

        # ── Rolling trade sizes in USDT for sweep-score computation ─────────
        self._trade_sizes: Dict[str, Deque[float]] = {
            s: deque(maxlen=_TRADE_SIZE_HISTORY) for s in all_syms
        }

        # ── Raw liq buffer (drained every minute in _liq_agg_loop) ─────────
        self._liq_raw_buf: Dict[str, List[Dict[str, Any]]] = {s: [] for s in all_syms}

        # ── In-memory Parquet row buffers ────────────────────────────────────
        self._ob_buf:    List[Dict[str, Any]] = []
        self._trade_buf_rows: List[Dict[str, Any]] = []
        self._liq_rows:  List[Dict[str, Any]] = []

        # ── Parquet writers (one per stream per day) ─────────────────────────
        self._current_day: Optional[date] = None
        self._writers: Dict[str, Optional[pq.ParquetWriter]] = {
            "ob":    None,
            "trade": None,
            "liq":   None,
        }

    # ── Universe refresh (extends base to init collector buffers) ────────────

    async def _refresh_universe(self) -> int:
        n = await super()._refresh_universe()
        if n > 0:
            # Init trade/liq buffers for any new symbols
            for sym in self._states:
                if sym not in self._trade_buf:
                    self._trade_buf[sym] = []
                if sym not in self._trade_sizes:
                    self._trade_sizes[sym] = deque(maxlen=_TRADE_SIZE_HISTORY)
                if sym not in self._liq_raw_buf:
                    self._liq_raw_buf[sym] = []
        return n

    async def _universe_refresh_loop(self) -> None:
        """Refresh the symbol universe every 8 hours."""
        while self._running:
            await asyncio.sleep(8 * 3600)
            try:
                added = await self._refresh_universe()
                if added > 0:
                    logger.info("Universe refresh loop: added %d symbols", added)
            except Exception as exc:
                logger.error("Universe refresh loop error: %s", exc)

    # ── WebSocket message handler ──────────────────────────────────────────────

    def _on_raw_message(self, topic: str, msg: Dict[str, Any]) -> None:
        """Handle non-orderbook WebSocket messages (trades + liquidations)."""
        if topic.startswith("publicTrade."):
            sym = topic[len("publicTrade."):]
            if sym not in self._states:
                return
            for trade in msg.get("data", []):
                v    = float(trade.get("v", 0))
                p    = float(trade.get("p", 0))
                size_usdt = v * p
                self._trade_buf[sym].append(trade)
                if size_usdt > 0:
                    self._trade_sizes[sym].append(size_usdt)

        elif topic.startswith("liquidation."):
            sym = topic[len("liquidation."):]
            if sym not in self._states:
                return
            data = msg.get("data", {})
            if isinstance(data, dict) and data:
                self._liq_raw_buf[sym].append({
                    "side":      data.get("side", ""),
                    "size_usdt": float(data.get("size", 0)) * float(data.get("price", 0)),
                    "ts_ms":     data.get("updatedTime", int(time.time() * 1000)),
                })

    # ── Feature-row builders ───────────────────────────────────────────────────

    def _make_ob_row(
        self,
        sym:   str,
        state: SymbolState,
        ts:    pd.Timestamp,
    ) -> Optional[Dict[str, Any]]:
        """Build one OB snapshot row from the current state."""
        if time.monotonic() - state.last_ob_ts > _OB_STALE_THRESH:
            return None  # book too stale

        book = self._ws.orderbooks.get(sym)
        if book is None:
            return None

        bid1 = book.best_bid()
        ask1 = book.best_ask()
        if bid1 is None or ask1 is None:
            return None

        mid = (bid1 + ask1) / 2.0

        ob = state.ob_feats
        return {
            "timestamp":       ts,
            "symbol":          sym,
            "bid1_price":      bid1,
            "bid1_size":       book.bids.get(bid1, float("nan")),
            "ask1_price":      ask1,
            "ask1_size":       book.asks.get(ask1, float("nan")),
            "mid_price":       mid,
            "spread_bps":      ob.get("spread_bps",      float("nan")),
            "bid_depth_usdt":  ob.get("depth_bid_usdt",  float("nan")),
            "ask_depth_usdt":  ob.get("depth_ask_usdt",  float("nan")),
            "depth_imbalance": ob.get("depth_band_imbalance", float("nan")),
            "thin_pct":        ob.get("thin_pct",        float("nan")),
            "vacuum_dist_bid": ob.get("vacuum_dist_bid", float("nan")),
            "vacuum_dist_ask": ob.get("vacuum_dist_ask", float("nan")),
        }

    def _make_trade_row(
        self,
        sym:    str,
        ts:     pd.Timestamp,
        trades: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Aggregate one second's worth of trades into a single row."""
        buy_vol  = 0.0
        sell_vol = 0.0
        pv_sum   = 0.0   # price × volume (for VWAP)
        vol_sum  = 0.0   # total volume in base units
        usdt_sum = 0.0   # total volume in USDT

        for t in trades:
            v    = float(t.get("v", 0))
            p    = float(t.get("p", 0))
            side = t.get("S", "")
            if side == "Buy":
                buy_vol  += v
            else:
                sell_vol += v
            pv_sum   += p * v
            vol_sum  += v
            usdt_sum += p * v

        vwap = pv_sum / vol_sum if vol_sum > 0 else float("nan")

        # Sweep score: fraction of USDT volume in "large" trades
        sweep_score = self._compute_sweep_score(sym, trades)

        return {
            "timestamp":      ts,
            "symbol":         sym,
            "buy_vol":        buy_vol,
            "sell_vol":       sell_vol,
            "cvd":            buy_vol - sell_vol,
            "trade_count":    len(trades),
            "vwap":           vwap,
            "sweep_score":    sweep_score,
            "total_vol_usdt": usdt_sum,
        }

    def _compute_sweep_score(
        self,
        sym:    str,
        trades: List[Dict[str, Any]],
    ) -> float:
        """
        Fraction of USDT volume from "sweeping" (large) trades.

        A trade is classified as a sweep if its USDT size exceeds the
        90th-percentile of the rolling size history for that symbol.
        Falls back to 3× mean if history < 10 trades.
        """
        if not trades:
            return 0.0

        sizes      = [float(t.get("v", 0)) * float(t.get("p", 0)) for t in trades]
        total_usdt = sum(sizes)
        if total_usdt == 0:
            return 0.0

        history = list(self._trade_sizes[sym])
        if len(history) >= 10:
            threshold = float(np.percentile(history, 90))
        elif history:
            threshold = float(np.mean(history)) * 3.0
        else:
            return 0.0

        sweep_usdt = sum(s for s in sizes if s > threshold)
        return sweep_usdt / total_usdt

    # ── Async loops ────────────────────────────────────────────────────────────

    async def _snapshot_loop(self) -> None:
        """Every second: snapshot OB features and aggregate trades."""
        logger.info("Snapshot loop started (1 s cadence, %d symbols)", len(self._states))
        while self._running:
            t0  = asyncio.get_event_loop().time()
            now = datetime.now(timezone.utc)
            ts  = pd.Timestamp(now)

            # Day rollover check
            today = now.date()
            if self._current_day != today:
                await self._day_rollover(today)

            for sym, state in self._states.items():
                # OB snapshot
                ob_row = self._make_ob_row(sym, state, ts)
                if ob_row is not None:
                    self._ob_buf.append(ob_row)
                    # Realtime mid price history
                    state.rt_mid_1s.append(ob_row["mid_price"])

                # Trade aggregate (drain buffer)
                trades = self._trade_buf[sym]
                self._trade_buf[sym] = []
                trade_row = self._make_trade_row(sym, ts, trades)
                self._trade_buf_rows.append(trade_row)

                # Realtime trade volume history
                state.rt_vol_1s.append(trade_row["total_vol_usdt"])

                # Realtime volume explosion check (5s throttle inside)
                if self._event_detector is not None:
                    rt_events = self._event_detector.evaluate_realtime(sym, state)
                    if rt_events and self._alert_manager is not None:
                        for ev in rt_events:
                            asyncio.ensure_future(self._alert_manager.handle(ev))

            # Force flush if buffers are growing too large
            if len(self._ob_buf) > _MAX_BUF_ROWS:
                await asyncio.get_event_loop().run_in_executor(
                    None, self._do_flush
                )

            elapsed = asyncio.get_event_loop().time() - t0
            await asyncio.sleep(max(0.0, _SNAPSHOT_INTERVAL - elapsed))

    async def _liq_agg_loop(self) -> None:
        """Every minute: aggregate liquidation events into 1-min bars."""
        logger.info("Liquidation aggregation loop started (1 min cadence)")
        while self._running:
            # Align to the next minute boundary
            now       = time.time()
            next_min  = (int(now) // 60 + 1) * 60
            await asyncio.sleep(next_min - now)

            minute_ts = pd.Timestamp(
                datetime.utcfromtimestamp(int(now) // 60 * 60)
            ).tz_localize("UTC")

            for sym in self._states:
                raw = self._liq_raw_buf[sym]
                self._liq_raw_buf[sym] = []
                if not raw:
                    continue
                self._liq_rows.append(self._agg_liq_minute(sym, minute_ts, raw))

    async def _flush_loop(self) -> None:
        """Periodically flush in-memory row buffers to Parquet."""
        logger.info(
            "Flush loop started (%d s interval)", self._flush_interval
        )
        while self._running:
            await asyncio.sleep(self._flush_interval)
            await asyncio.get_event_loop().run_in_executor(None, self._do_flush)

    # ── Liquidation aggregation ────────────────────────────────────────────────

    @staticmethod
    def _agg_liq_minute(
        sym:       str,
        minute_ts: pd.Timestamp,
        raw:       List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        long_vol  = 0.0   # "Buy" side = long liq = forced sell
        short_vol = 0.0
        long_cnt  = 0
        short_cnt = 0
        max_size  = 0.0

        for ev in raw:
            s    = ev.get("side", "")
            usdt = ev.get("size_usdt", 0.0)
            if s == "Buy":
                long_vol += usdt
                long_cnt += 1
            elif s == "Sell":
                short_vol += usdt
                short_cnt += 1
            if usdt > max_size:
                max_size = usdt

        return {
            "timestamp":           minute_ts,
            "symbol":              sym,
            "long_liq_vol_usdt":   long_vol,
            "short_liq_vol_usdt":  short_vol,
            "long_liq_count":      long_cnt,
            "short_liq_count":     short_cnt,
            "max_single_liq_usdt": max_size,
        }

    # ── Parquet I/O ────────────────────────────────────────────────────────────

    def _open_writers(self, day: date) -> Dict[str, pq.ParquetWriter]:
        day_dir = self._output_dir / str(day)
        day_dir.mkdir(parents=True, exist_ok=True)
        return {
            "ob":    pq.ParquetWriter(str(day_dir / "ob_snapshots.parquet"),  _OB_SCHEMA),
            "trade": pq.ParquetWriter(str(day_dir / "trade_aggs.parquet"),    _TRADE_SCHEMA),
            "liq":   pq.ParquetWriter(str(day_dir / "liquidations.parquet"),  _LIQ_SCHEMA),
        }

    def _close_writers(self) -> None:
        for key, w in self._writers.items():
            if w is not None:
                try:
                    w.close()
                except Exception as exc:
                    logger.warning("Error closing %s writer: %s", key, exc)
        self._writers = {"ob": None, "trade": None, "liq": None}

    async def _day_rollover(self, new_day: date) -> None:
        """Flush remaining data, close today's writers, open tomorrow's."""
        if self._current_day is not None:
            logger.info(
                "Day rollover: %s → %s, flushing final data",
                self._current_day, new_day,
            )
            await asyncio.get_event_loop().run_in_executor(None, self._do_flush)
            self._close_writers()

        self._current_day = new_day
        self._writers     = self._open_writers(new_day)
        logger.info(
            "Opened Parquet writers for %s in %s",
            new_day, self._output_dir / str(new_day),
        )

    def _do_flush(self) -> None:
        """
        Synchronous flush: convert in-memory row lists to Arrow tables and
        write to the current day's ParquetWriters.  Called from a thread
        executor to avoid blocking the event loop.
        """
        # ── OB snapshots ──────────────────────────────────────────────────────
        if self._ob_buf and self._writers.get("ob"):
            try:
                df    = pd.DataFrame(self._ob_buf)
                table = _to_arrow(df, _OB_SCHEMA)
                self._writers["ob"].write_table(table)
                self._ob_buf.clear()
                logger.debug("Flushed %d OB rows", len(df))
            except Exception as exc:
                logger.error("OB flush failed: %s", exc)

        # ── Trade aggregates ──────────────────────────────────────────────────
        if self._trade_buf_rows and self._writers.get("trade"):
            try:
                df    = pd.DataFrame(self._trade_buf_rows)
                table = _to_arrow(df, _TRADE_SCHEMA)
                self._writers["trade"].write_table(table)
                self._trade_buf_rows.clear()
                logger.debug("Flushed %d trade rows", len(df))
            except Exception as exc:
                logger.error("Trade flush failed: %s", exc)

        # ── Liquidations ──────────────────────────────────────────────────────
        if self._liq_rows and self._writers.get("liq"):
            try:
                df    = pd.DataFrame(self._liq_rows)
                table = _to_arrow(df, _LIQ_SCHEMA)
                self._writers["liq"].write_table(table)
                self._liq_rows.clear()
                logger.debug("Flushed %d liq rows", len(df))
            except Exception as exc:
                logger.error("Liq flush failed: %s", exc)

    # ── run() override ─────────────────────────────────────────────────────────

    async def run(self) -> None:
        """
        Start all data-collection tasks and run until cancelled or SIGINT.

        Tasks started (no display loop):
          ws           — live WebSocket (OB + trades + liquidations)
          oi-poll      — REST OI every 60 s
          fund-poll    — REST funding every 5 min
          kline-poll   — REST klines every 1 h
          snapshot     — 1 s OB + trade snapshot loop
          liq-agg      — 1 min liquidation aggregation
          flush        — periodic Parquet flush
        """
        self._running = True
        self._load_parquets()
        self._compute_initial_features()
        await self._fetch_turnover()

        # Expand universe: add any USDT perps with >$1M turnover not in config
        await self._refresh_universe()

        # Open writers for today immediately
        await self._day_rollover(datetime.now(timezone.utc).date())

        # Settlement scheduler (pre-settlement squeeze rankings)
        self._settlement_scheduler = SettlementScheduler(
            self, alert_manager=self._alert_manager,
        )

        # ── Outcome tracker + Obduction agent (learning loop) ─────────────────
        _db_path = str(_ROOT / "data" / "events.db")
        try:
            from ..agents.outcome_tracker import OutcomeTracker
            from ..agents.obduction_agent import ObductionAgent
            _outcome_tracker = OutcomeTracker(db_path=_db_path)
            _outcome_tracker.initialize()
            _obduction_agent = ObductionAgent(db_path=_db_path)
            _obduction_agent.initialize()
            _learning_tasks = [
                asyncio.create_task(_outcome_tracker.run_loop(interval_minutes=30), name="outcome-tracker"),
                asyncio.create_task(_obduction_agent.run_scheduled(), name="obduction"),
            ]
            logger.info("Learning loop started (OutcomeTracker + ObductionAgent)")
        except Exception as _exc:
            logger.warning("Learning loop unavailable: %s", _exc)
            _learning_tasks = []

        # ── ML snapshot collector + labeler ───────────────────────────────────
        _ml_tasks: list = []
        try:
            from ..ml.snapshot_collector import SnapshotCollector
            from ..ml.label_snapshots import SnapshotLabeler
            _snap_collector = SnapshotCollector(scanner=self, db_path=_db_path)
            _snap_collector.initialize()
            _snap_labeler = SnapshotLabeler(db_path=_db_path)
            _ml_tasks = [
                asyncio.create_task(_snap_collector.run_loop(interval_seconds=3600), name="ml-snapshots"),
                asyncio.create_task(_snap_labeler.run_loop(interval_seconds=3600), name="ml-labeler"),
            ]
            logger.info("ML snapshot collection started (hourly)")
        except Exception as _exc:
            logger.warning("ML snapshot collection unavailable: %s", _exc)

        tasks = [
            asyncio.create_task(self._ws.run(),         name="ws"),
            asyncio.create_task(self._poll_oi(),        name="oi-poll"),
            asyncio.create_task(self._poll_funding(),   name="fund-poll"),
            asyncio.create_task(self._poll_klines(),    name="kline-poll"),
            asyncio.create_task(self._snapshot_loop(),  name="snapshot"),
            asyncio.create_task(self._liq_agg_loop(),   name="liq-agg"),
            asyncio.create_task(self._flush_loop(),     name="flush"),
            asyncio.create_task(self._settlement_scheduler.run_forever(), name="settlement"),
            asyncio.create_task(self._universe_refresh_loop(), name="universe-refresh"),
            *_learning_tasks,
            *_ml_tasks,
        ]

        logger.info(
            "DataCollector started: %d symbols, output=%s, flush=%ds",
            len(self._states), self._output_dir, self._flush_interval,
        )

        try:
            await asyncio.gather(*tasks)
        except (asyncio.CancelledError, KeyboardInterrupt):
            pass
        finally:
            self._running = False
            self._ws.stop()
            logger.info("Shutting down — flushing final data ...")
            self._do_flush()
            self._close_writers()
            if self._alert_manager is not None:
                await self._alert_manager.close()
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            logger.info("DataCollector stopped.")


# ── Arrow conversion helper ────────────────────────────────────────────────────


def _to_arrow(df: pd.DataFrame, schema: pa.Schema) -> pa.Table:
    """
    Convert a DataFrame to an Arrow Table using the supplied schema.

    Columns absent from ``df`` are filled with nulls.  The timestamp
    column is cast to UTC-aware microsecond precision.
    """
    # Ensure all schema columns are present
    for field in schema:
        if field.name not in df.columns:
            df[field.name] = None

    # Cast timestamp column
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Cast integer columns explicitly
    for field in schema:
        if pa.types.is_integer(field.type) and field.name in df.columns:
            df[field.name] = df[field.name].fillna(0).astype("int32")

    return pa.Table.from_pandas(df[list(f.name for f in schema)], schema=schema, safe=False)


# ── CLI helpers ────────────────────────────────────────────────────────────────


def _load_symbols_yaml(path: Path) -> Dict[int, List[str]]:
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


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.collector",
        description="Headless 24/7 Bybit data collector",
    )
    p.add_argument(
        "--config", type=Path, default=_CONFIG_DEF, metavar="PATH",
        help="Path to symbols.yaml (default: configs/symbols.yaml)",
    )
    p.add_argument(
        "--symbols", nargs="+", metavar="SYM",
        help="Override config: collect only these symbols (assigned to tier 3)",
    )
    p.add_argument(
        "--output-dir", type=Path, default=_COLLECT_DIR, metavar="DIR",
        help=f"Root directory for collected Parquet files (default: {_COLLECT_DIR})",
    )
    p.add_argument(
        "--flush-interval", type=int, default=_FLUSH_INTERVAL, metavar="SECS",
        help=f"Seconds between Parquet flushes (default: {_FLUSH_INTERVAL})",
    )
    p.add_argument(
        "--no-events", action="store_true",
        help="Disable event detection (no SQLite writes)",
    )
    p.add_argument(
        "--webhook-url", default=None, metavar="URL",
        help="Webhook URL for event notifications",
    )
    p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    return p


async def _async_main(args: argparse.Namespace) -> None:
    # ── Load .env ─────────────────────────────────────────────────────────────
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")

    # ── Symbol universe ───────────────────────────────────────────────────────
    if args.symbols:
        symbols_by_tier: Dict[int, List[str]] = {3: list(args.symbols)}
    elif args.config.exists():
        symbols_by_tier = _load_symbols_yaml(args.config)
    else:
        print(f"Config not found: {args.config}", file=sys.stderr)
        print("Run backfill first:  python -m src.bybit.backfill", file=sys.stderr)
        sys.exit(1)

    n_syms = sum(len(v) for v in symbols_by_tier.values())
    logger.info("Loaded %d symbols across %d tiers", n_syms, len(symbols_by_tier))

    # ── REST client ───────────────────────────────────────────────────────────
    rest = BybitRestClient()
    await rest.__aenter__()  # start aiohttp session

    # ── Event detection ───────────────────────────────────────────────────────
    event_detector: Optional[EventDetector] = None
    alert_manager:  Optional[AlertManager]  = None

    if not args.no_events:
        import os
        tg_token   = os.environ.get("TELEGRAM_BOT_TOKEN")
        tg_chat_id = os.environ.get("TELEGRAM_CHAT_ID")

        event_detector = EventDetector()
        alert_manager  = AlertManager(
            webhook_url=args.webhook_url,
            telegram_token=tg_token,
            telegram_chat_id=tg_chat_id,
        )
        logger.info("Event detection enabled (SQLite: %s)", _ROOT / "data" / "events.db")
        alert_manager.start_digest_loop()

    # ── Collector ─────────────────────────────────────────────────────────────
    collector = DataCollector(
        symbols_by_tier=symbols_by_tier,
        rest=rest,
        output_dir=args.output_dir,
        event_detector=event_detector,
        alert_manager=alert_manager,
        flush_interval=args.flush_interval,
    )

    # Wire BTC regime into digest
    if alert_manager is not None:
        def _btc_4h_change():
            """Return (pct_change_4h, current_price) for BTCUSDT or None."""
            st = collector._states.get("BTCUSDT")
            if st is None or st.klines_df is None or st.klines_df.empty:
                return None
            df = st.klines_df
            if len(df) < 4:
                return None
            close_now = float(df["close"].iloc[-1])
            close_4h  = float(df["close"].iloc[-4])
            if close_4h <= 0:
                return None
            pct = (close_now - close_4h) / close_4h * 100.0
            return (pct, close_now)
        alert_manager._btc_price_fn = _btc_4h_change

    # ── ML Conviction Digest ─────────────────────────────────────────────────
    conviction_digest = None
    if alert_manager is not None:
        try:
            from src.ml.predictor import FragilityPredictor
            from src.ml.conviction_digest import ConvictionDigest

            predictor = FragilityPredictor()
            if predictor.load_model():
                def _get_recent_events(sym: str):
                    """Get recent events for a symbol as dicts."""
                    return [
                        {
                            "event_type": ev.event_type.value,
                            "score": ev.score,
                            "direction": ev.direction,
                        }
                        for ev in alert_manager._event_history
                        if ev.symbol == sym
                    ]

                conviction_digest = ConvictionDigest(
                    predictor=predictor,
                    alert_manager=alert_manager,
                    states_fn=lambda: collector._states,
                    recent_events_fn=_get_recent_events,
                )
                conviction_digest.start()
                logger.info("ML ConvictionDigest enabled (AUC=%.3f)",
                            predictor.metrics.get("auc", 0))
            else:
                logger.info("No ML model found — conviction digest disabled")
        except Exception as exc:
            logger.warning("ML conviction digest init failed: %s", exc)

    # Graceful shutdown on SIGTERM (for systemd / Docker)
    loop = asyncio.get_event_loop()

    def _handle_sigterm() -> None:
        logger.info("SIGTERM received — initiating graceful shutdown ...")
        for task in asyncio.all_tasks(loop):
            task.cancel()

    try:
        loop.add_signal_handler(signal.SIGTERM, _handle_sigterm)
    except (NotImplementedError, RuntimeError):
        pass  # Windows / environments without SIGTERM support

    try:
        await collector.run()
    finally:
        if conviction_digest is not None:
            conviction_digest.stop()
        if alert_manager is not None:
            alert_manager.stop_digest_loop()
        await rest.__aexit__(None, None, None)


def main() -> None:
    parser = _build_arg_parser()
    args   = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        asyncio.run(_async_main(args))
    except KeyboardInterrupt:
        pass
