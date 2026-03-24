"""Market Structure Detection — Swing highs/lows, sweeps, displacement, FVGs.

Identifies SMC/ICT entry setups on 5m klines:
1. Swing points (liquidity pools)
2. Liquidity sweeps (stop hunts)
3. Displacement candles (smart money confirmation)
4. Fair Value Gaps (entry zones)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

from .klines_5m import Kline

logger = logging.getLogger(__name__)


@dataclass
class SwingPoint:
    type: str           # "HIGH" or "LOW"
    price: float
    timestamp: int
    index: int
    strength: int
    swept: bool = False


@dataclass
class FVG:
    type: str           # "BULLISH" or "BEARISH"
    top: float
    bottom: float
    timestamp: int

    @property
    def midpoint(self) -> float:
        return (self.top + self.bottom) / 2

    @property
    def size_pct(self) -> float:
        return (self.top - self.bottom) / self.bottom * 100 if self.bottom > 0 else 0


@dataclass
class SMCSetup:
    symbol: str
    direction: str          # "LONG" or "SHORT"
    timestamp: int
    sweep_level: float
    sweep_candle_ts: int
    displacement_price: float
    displacement_size_pct: float
    fvg: Optional[FVG]
    target_level: float
    confidence: float       # 0-100
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward: float


# ── Detection functions ───────────────────────────────────────────────────────


def detect_swing_points(klines: List[Kline], lookback: int = 3) -> List[SwingPoint]:
    """Detect swing highs and lows. Requires `lookback` candles on each side."""
    swings: List[SwingPoint] = []
    if len(klines) < lookback * 2 + 1:
        return swings

    for i in range(lookback, len(klines) - lookback):
        candle = klines[i]

        # Swing high
        is_high = all(
            klines[i - j].high < candle.high and klines[i + j].high < candle.high
            for j in range(1, lookback + 1)
        )
        if is_high:
            swings.append(SwingPoint("HIGH", candle.high, candle.timestamp, i, lookback))

        # Swing low
        is_low = all(
            klines[i - j].low > candle.low and klines[i + j].low > candle.low
            for j in range(1, lookback + 1)
        )
        if is_low:
            swings.append(SwingPoint("LOW", candle.low, candle.timestamp, i, lookback))

    return swings


def detect_sweep(
    klines: List[Kline], swings: List[SwingPoint], lookback_candles: int = 6,
) -> Optional[dict]:
    """Detect liquidity sweep: wick beyond swing level, body rejects back."""
    if len(klines) < 3 or not swings:
        return None

    for i in range(max(0, len(klines) - lookback_candles), len(klines)):
        candle = klines[i]

        # Sweep of swing lows (LONG setup)
        for swing in swings:
            if swing.type != "LOW" or swing.swept or swing.index >= i:
                continue
            if candle.low < swing.price and candle.close > swing.price and candle.is_bullish:
                swing.swept = True
                return {
                    "type": "LONG", "sweep_level": swing.price,
                    "sweep_candle": candle, "sweep_index": i, "swing": swing,
                }

        # Sweep of swing highs (SHORT setup)
        for swing in swings:
            if swing.type != "HIGH" or swing.swept or swing.index >= i:
                continue
            if candle.high > swing.price and candle.close < swing.price and candle.is_bearish:
                swing.swept = True
                return {
                    "type": "SHORT", "sweep_level": swing.price,
                    "sweep_candle": candle, "sweep_index": i, "swing": swing,
                }

    return None


def detect_displacement(
    klines: List[Kline], sweep: dict,
    min_body_ratio: float = 0.60, min_move_pct: float = 0.3,
) -> Optional[int]:
    """Find displacement candle after sweep. Returns its index or None."""
    sweep_idx = sweep["sweep_index"]
    direction = sweep["type"]

    # Average volume of preceding 20 candles
    start = max(0, sweep_idx - 20)
    prev_vols = [k.volume for k in klines[start:sweep_idx]]
    avg_vol = sum(prev_vols) / len(prev_vols) if prev_vols else 0

    for i in range(sweep_idx, min(sweep_idx + 5, len(klines))):
        candle = klines[i]

        if direction == "LONG" and not candle.is_bullish:
            continue
        if direction == "SHORT" and not candle.is_bearish:
            continue
        if candle.body_ratio < min_body_ratio:
            continue

        move_pct = candle.body_size / candle.open * 100 if candle.open > 0 else 0
        if move_pct < min_move_pct:
            continue
        if avg_vol > 0 and candle.volume < avg_vol * 1.5:
            continue

        return i

    return None


def detect_fvg(klines: List[Kline], disp_idx: int, direction: str) -> Optional[FVG]:
    """Detect Fair Value Gap around displacement candle."""
    if disp_idx < 1 or disp_idx >= len(klines) - 1:
        return None

    prev = klines[disp_idx - 1]
    disp = klines[disp_idx]
    nxt = klines[disp_idx + 1]

    if direction == "LONG" and nxt.low > prev.high:
        return FVG("BULLISH", nxt.low, prev.high, disp.timestamp)
    if direction == "SHORT" and nxt.high < prev.low:
        return FVG("BEARISH", prev.low, nxt.high, disp.timestamp)

    return None


def find_target(
    swings: List[SwingPoint], current_price: float, direction: str,
) -> Optional[float]:
    """Find nearest unswept liquidity target."""
    if direction == "LONG":
        targets = [s.price for s in swings
                   if s.type == "HIGH" and not s.swept and s.price > current_price]
        return min(targets) if targets else None
    targets = [s.price for s in swings
               if s.type == "LOW" and not s.swept and s.price < current_price]
    return max(targets) if targets else None


# ── Full pipeline ─────────────────────────────────────────────────────────────


def detect_smc_setup(symbol: str, klines: List[Kline]) -> Optional[SMCSetup]:
    """Run full SMC detection: swings -> sweep -> displacement -> FVG -> target."""
    if len(klines) < 30:
        return None

    swings = detect_swing_points(klines, lookback=3)
    if len(swings) < 3:
        return None

    sweep = detect_sweep(klines, swings, lookback_candles=6)
    if not sweep:
        return None

    disp_idx = detect_displacement(klines, sweep)
    if disp_idx is None:
        return None

    disp_candle = klines[disp_idx]
    fvg = detect_fvg(klines, disp_idx, sweep["type"])

    current_price = klines[-1].close
    target = find_target(swings, current_price, sweep["type"])

    # Default target: 2x displacement move
    if not target:
        disp_move = abs(disp_candle.close - sweep["sweep_level"])
        if sweep["type"] == "LONG":
            target = disp_candle.close + disp_move
        else:
            target = disp_candle.close - disp_move

    # Entry / stop / TP
    if sweep["type"] == "LONG":
        entry = fvg.midpoint if fvg else disp_candle.close
        stop = sweep["sweep_level"] * 0.998
    else:
        entry = fvg.midpoint if fvg else disp_candle.close
        stop = sweep["sweep_level"] * 1.002
    tp = target

    risk = abs(entry - stop)
    reward = abs(tp - entry)
    rr = reward / risk if risk > 0 else 0

    # Confidence
    conf = 50  # base: sweep + displacement confirmed
    if fvg:
        conf += 20
    if rr >= 2.0:
        conf += 15
    if rr >= 3.0:
        conf += 5
    disp_pct = disp_candle.body_size / disp_candle.open * 100 if disp_candle.open > 0 else 0
    if disp_pct > 1.0:
        conf += 10
    conf = min(100, conf)

    return SMCSetup(
        symbol=symbol,
        direction=sweep["type"],
        timestamp=klines[-1].timestamp,
        sweep_level=sweep["sweep_level"],
        sweep_candle_ts=sweep["sweep_candle"].timestamp,
        displacement_price=disp_candle.close,
        displacement_size_pct=disp_pct,
        fvg=fvg,
        target_level=target,
        confidence=conf,
        entry_price=entry,
        stop_loss=stop,
        take_profit=tp,
        risk_reward=rr,
    )
