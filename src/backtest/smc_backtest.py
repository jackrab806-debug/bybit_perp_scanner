"""Backtest SMC entry detection on historical 5m klines.

For each symbol with 5m data:
1. Slide a 100-candle window across the history
2. Run sweep -> displacement -> FVG detection
3. Simulate trades forward (TP/SL/timeout)
4. Report win rate, profit factor, total PnL

Usage:
    python -m src.backtest.smc_backtest [--min-rr 2.0]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from src.features.klines_5m import Kline
from src.features.market_structure import (
    detect_swing_points, detect_sweep, detect_displacement,
    detect_fvg, find_target,
)

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent.parent
HIST_DIR = _ROOT / "data" / "historical"


@dataclass
class BacktestResult:
    symbol: str
    total: int
    wins: int
    losses: int
    timeouts: int
    win_rate: float
    avg_rr: float
    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: float
    total_pnl_pct: float


def _load_klines(symbol: str) -> List[Kline]:
    p = HIST_DIR / symbol / "klines_5m.parquet"
    if not p.exists():
        return []
    df = pd.read_parquet(p).sort_values("timestamp")
    return [
        Kline(
            timestamp=int(r["timestamp"]),
            open=float(r["open"]), high=float(r["high"]),
            low=float(r["low"]), close=float(r["close"]),
            volume=float(r["volume"]), turnover=float(r["turnover"]),
        )
        for _, r in df.iterrows()
    ]


def _simulate(
    klines: List[Kline], entry_idx: int, entry: float,
    sl: float, tp: float, direction: str, max_candles: int = 96,
) -> dict:
    """Simulate trade forward. 96 candles = 8h on 5m."""
    end = min(entry_idx + max_candles, len(klines))
    for i in range(entry_idx + 1, end):
        c = klines[i]
        if direction == "LONG":
            if c.low <= sl:
                return {"outcome": "LOSS", "pnl": (sl - entry) / entry * 100,
                        "candles": i - entry_idx}
            if c.high >= tp:
                return {"outcome": "WIN", "pnl": (tp - entry) / entry * 100,
                        "candles": i - entry_idx}
        else:
            if c.high >= sl:
                return {"outcome": "LOSS", "pnl": (entry - sl) / entry * 100,
                        "candles": i - entry_idx}
            if c.low <= tp:
                return {"outcome": "WIN", "pnl": (entry - tp) / entry * 100,
                        "candles": i - entry_idx}
    last = klines[end - 1]
    pnl = ((last.close - entry) / entry * 100 if direction == "LONG"
           else (entry - last.close) / entry * 100)
    return {"outcome": "TIMEOUT", "pnl": pnl, "candles": max_candles}


def backtest_symbol(symbol: str, min_rr: float = 2.0) -> Optional[BacktestResult]:
    klines = _load_klines(symbol)
    if len(klines) < 200:
        return None

    trades: list = []
    win_size = 100
    step = 12  # 1h steps
    i = win_size

    while i < len(klines) - 96:
        window = klines[i - win_size: i]
        try:
            swings = detect_swing_points(window, lookback=3)
            if len(swings) < 3:
                i += step
                continue
            sweep = detect_sweep(window, swings, lookback_candles=6)
            if not sweep:
                i += step
                continue
            disp_idx = detect_displacement(window, sweep)
            if disp_idx is None:
                i += step
                continue
            disp_candle = window[disp_idx]
            fvg = detect_fvg(window, disp_idx, sweep["type"])
            tgt = find_target(swings, window[-1].close, sweep["type"])
            if not tgt:
                i += step
                continue

            if sweep["type"] == "LONG":
                entry = fvg.midpoint if fvg else disp_candle.close
                sl = sweep["sweep_level"] * 0.998
            else:
                entry = fvg.midpoint if fvg else disp_candle.close
                sl = sweep["sweep_level"] * 1.002

            risk = abs(entry - sl)
            reward = abs(tgt - entry)
            rr = reward / risk if risk > 0 else 0

            if rr >= min_rr:
                res = _simulate(klines, i, entry, sl, tgt, sweep["type"])
                res["rr"] = rr
                res["fvg"] = fvg is not None
                res["dir"] = sweep["type"]
                trades.append(res)
                i += 48  # skip 4h after trade
                continue
        except Exception:
            pass
        i += step

    if len(trades) < 5:
        return None

    wins = [t for t in trades if t["outcome"] == "WIN"]
    losses = [t for t in trades if t["outcome"] == "LOSS"]
    timeouts = [t for t in trades if t["outcome"] == "TIMEOUT"]

    win_pnl = sum(t["pnl"] for t in wins) if wins else 0
    loss_pnl = abs(sum(t["pnl"] for t in losses)) if losses else 0.001

    return BacktestResult(
        symbol=symbol,
        total=len(trades),
        wins=len(wins),
        losses=len(losses),
        timeouts=len(timeouts),
        win_rate=len(wins) / len(trades) * 100,
        avg_rr=float(np.mean([t["rr"] for t in trades])),
        avg_win_pct=float(np.mean([t["pnl"] for t in wins])) if wins else 0,
        avg_loss_pct=float(np.mean([t["pnl"] for t in losses])) if losses else 0,
        profit_factor=win_pnl / loss_pnl,
        total_pnl_pct=sum(t["pnl"] for t in trades),
    )


def run_full_backtest(min_rr: float = 2.0) -> None:
    symbols = sorted(
        d.name for d in HIST_DIR.iterdir()
        if d.is_dir() and (d / "klines_5m.parquet").exists()
    )
    print(f"Backtesting {len(symbols)} symbols (min R:R={min_rr})")
    print("=" * 70)

    results: List[BacktestResult] = []
    for i, sym in enumerate(symbols):
        r = backtest_symbol(sym, min_rr)
        if r:
            results.append(r)
            print(
                f"[{i+1}/{len(symbols)}] {sym}: "
                f"{r.wins}W/{r.losses}L/{r.timeouts}T "
                f"({r.win_rate:.0f}%) PF={r.profit_factor:.2f} "
                f"PnL={r.total_pnl_pct:+.1f}%"
            )

    if not results:
        print("No results with enough trades.")
        return

    total = sum(r.total for r in results)
    wins = sum(r.wins for r in results)
    losses = sum(r.losses for r in results)
    pnl = sum(r.total_pnl_pct for r in results)

    print("\n" + "=" * 70)
    print("AGGREGATE")
    print("=" * 70)
    print(f"Symbols: {len(results)} | Trades: {total}")
    print(f"Wins: {wins} | Losses: {losses} | WR: {wins/total*100:.1f}%")
    print(f"Avg PF: {np.mean([r.profit_factor for r in results]):.2f}")
    print(f"Total PnL: {pnl:+.1f}%")

    by_pnl = sorted(results, key=lambda r: -r.total_pnl_pct)
    print("\nTop 5:")
    for r in by_pnl[:5]:
        print(f"  {r.symbol}: WR={r.win_rate:.0f}% PF={r.profit_factor:.2f} PnL={r.total_pnl_pct:+.1f}%")
    print("Worst 5:")
    for r in by_pnl[-5:]:
        print(f"  {r.symbol}: WR={r.win_rate:.0f}% PF={r.profit_factor:.2f} PnL={r.total_pnl_pct:+.1f}%")


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("--min-rr", type=float, default=2.0)
    args = p.parse_args()
    run_full_backtest(min_rr=args.min_rr)
