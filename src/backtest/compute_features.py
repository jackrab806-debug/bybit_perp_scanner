"""Compute ML features from historical data.

Takes raw klines/funding/OI parquet files and produces the same feature
set used in live ml_snapshots — enabling training on 1 year of data.

Features computable historically:
  bb_width_pct, rv_pct, compression, vol_ratio, price changes,
  funding_rate, funding_z, abs_funding, oi changes, oi_z_24h,
  hour_utc, mins_to_settlement, settlement_urgency, is_weekend,
  pressure_ratio, btc context.

Features NOT available historically (orderbook-based):
  thin_pct, depth, vacuum_dist, spread_z, imbalance — set to 0.

Usage:
    python -m src.backtest.compute_features
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent.parent
HIST_DIR = _ROOT / "data" / "historical"


def _bb_width(close: pd.Series, period: int = 20) -> pd.Series:
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    return ((sma + 2 * std) - (sma - 2 * std)) / sma * 100


def _rv(close: pd.Series, period: int = 20) -> pd.Series:
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(period).std() * np.sqrt(365 * 24) * 100


def _settlement_mins(hour: int, minute: int) -> int:
    cur = hour * 60 + minute
    return min((s * 60 - cur) % 1440 for s in (0, 8, 16, 24))


def compute_symbol(symbol: str) -> pd.DataFrame:
    """Compute features for one symbol from its historical parquet files."""
    sym_dir = HIST_DIR / symbol
    klines_path = sym_dir / "klines_1h.parquet"
    if not klines_path.exists():
        return pd.DataFrame()

    kl = pd.read_parquet(klines_path).sort_values("timestamp").reset_index(drop=True)
    if len(kl) < 30:
        return pd.DataFrame()

    df = pd.DataFrame()
    df["timestamp"] = kl["timestamp"]
    df["symbol"] = symbol
    df["mid_price"] = kl["close"]

    # ── Kline features ────────────────────────────────────────────────────
    df["bb_width_pct"] = _bb_width(kl["close"])
    df["rv_pct"] = _rv(kl["close"])
    bb_mean = df["bb_width_pct"].rolling(50).mean()
    df["compression"] = bb_mean / (df["bb_width_pct"] + 0.001)
    vol_avg = kl["volume"].rolling(24).mean()
    df["vol_ratio_5m"] = kl["volume"] / (vol_avg + 1)

    # ── Time features ─────────────────────────────────────────────────────
    dt = pd.to_datetime(kl["timestamp"], unit="ms", utc=True)
    df["hour_utc"] = dt.dt.hour
    df["is_weekend"] = (dt.dt.weekday >= 5).astype(int)
    df["mins_to_settlement"] = [_settlement_mins(t.hour, t.minute) for t in dt]
    df["settlement_urgency"] = 1.0 / (df["mins_to_settlement"] + 10)

    # ── Funding ───────────────────────────────────────────────────────────
    funding_path = sym_dir / "funding.parquet"
    if funding_path.exists():
        fund = pd.read_parquet(funding_path).sort_values("timestamp")
        df["funding_rate"] = np.nan
        # Vectorised merge_asof (much faster than row loop)
        merged = pd.merge_asof(
            df[["timestamp"]].assign(_idx=df.index),
            fund.rename(columns={"funding_rate": "_fr"}),
            on="timestamp", direction="backward",
        )
        df["funding_rate"] = merged["_fr"].values
        df["funding_rate"] = df["funding_rate"].ffill().fillna(0)
    else:
        df["funding_rate"] = 0.0

    df["abs_funding"] = df["funding_rate"].abs()
    roll_mean = df["funding_rate"].rolling(72, min_periods=10).mean()
    roll_std = df["funding_rate"].rolling(72, min_periods=10).std()
    df["funding_z"] = (df["funding_rate"] - roll_mean) / (roll_std + 1e-8)

    # ── OI ────────────────────────────────────────────────────────────────
    oi_path = sym_dir / "oi.parquet"
    if oi_path.exists():
        oi = pd.read_parquet(oi_path).sort_values("timestamp")
        merged = pd.merge_asof(
            df[["timestamp"]].assign(_idx=df.index),
            oi.rename(columns={"oi": "_oi"}),
            on="timestamp", direction="backward",
        )
        df["oi_current"] = merged["_oi"].values
        df["oi_current"] = df["oi_current"].ffill()
    else:
        df["oi_current"] = 0.0

    df["oi_usd"] = df["oi_current"] * df["mid_price"]
    df["oi_change_1h_pct"] = df["oi_current"].pct_change(1) * 100
    df["oi_change_4h_pct"] = df["oi_current"].pct_change(4) * 100
    oi_mean = df["oi_current"].rolling(24, min_periods=5).mean()
    oi_std = df["oi_current"].rolling(24, min_periods=5).std()
    df["oi_z_24h"] = (df["oi_current"] - oi_mean) / (oi_std + 1e-8)

    # ── Derived ───────────────────────────────────────────────────────────
    df["pressure_ratio"] = (
        df["abs_funding"] * df["oi_usd"]
        / (df["oi_usd"].rolling(24).mean() + 1)
    )

    # Features not available historically — defaults
    for col in [
        "thin_pct", "depth_ask_usdt", "depth_bid_usdt",
        "vacuum_dist_ask", "vacuum_dist_bid", "spread_z",
        "spread_bps", "imbalance", "convexity",
        "sps", "lfi", "rank",
        "vacuum_asymmetry", "oi_to_depth_ratio", "funding_x_oi",
        "cvd_ratio_24h", "taker_proxy", "price_accel",
    ]:
        df[col] = 0.0

    # Event features (none historically)
    for col in ["has_fs", "has_vb", "has_ve", "has_ca", "has_oi",
                "num_event_types_2h", "max_event_score",
                "event_intensity", "multi_signal"]:
        df[col] = 0

    # BTC context (filled by add_btc_context later)
    df["btc_change_1h"] = 0.0
    df["btc_change_4h"] = 0.0

    # ── Labels ────────────────────────────────────────────────────────────
    close = kl["close"]
    high = kl["high"]
    low = kl["low"]

    for h in (1, 2, 4):
        fut_hi = high.rolling(h).max().shift(-h)
        fut_lo = low.rolling(h).min().shift(-h)
        up = (fut_hi - close) / close * 100
        dn = (close - fut_lo) / close * 100
        df[f"move_{h}h_pct"] = up
        df[f"max_move_{h}h"] = pd.concat([up, dn], axis=1).max(axis=1)

    df["abs_max_move_4h"] = df["max_move_4h"]
    df["label"] = (df["abs_max_move_4h"] >= 5.0).astype(int)

    # Drop warmup NaNs
    df = df.dropna(subset=["bb_width_pct", "rv_pct"]).reset_index(drop=True)
    return df


def add_btc_context(df: pd.DataFrame) -> pd.DataFrame:
    """Merge BTC 1h/4h price changes into all rows."""
    btc_path = HIST_DIR / "BTCUSDT" / "klines_1h.parquet"
    if not btc_path.exists():
        return df
    btc = pd.read_parquet(btc_path).sort_values("timestamp")
    btc["btc_1h"] = btc["close"].pct_change(1) * 100
    btc["btc_4h"] = btc["close"].pct_change(4) * 100
    btc_lk = btc[["timestamp", "btc_1h", "btc_4h"]]

    df = pd.merge_asof(
        df.sort_values("timestamp"), btc_lk,
        on="timestamp", direction="backward",
    )
    df["btc_change_1h"] = df.pop("btc_1h").fillna(0)
    df["btc_change_4h"] = df.pop("btc_4h").fillna(0)
    return df


def compute_all(symbols: list | None = None) -> pd.DataFrame:
    if not symbols:
        symbols = sorted(d.name for d in HIST_DIR.iterdir() if d.is_dir())
    parts = []
    for i, sym in enumerate(symbols):
        try:
            df = compute_symbol(sym)
            if not df.empty:
                parts.append(df)
                logger.info(
                    "[%d/%d] %s: %d rows, %d hits (%.1f%%)",
                    i + 1, len(symbols), sym, len(df),
                    df["label"].sum(), df["label"].mean() * 100,
                )
        except Exception as e:
            logger.error("Feature error %s: %s", sym, e)
    if not parts:
        return pd.DataFrame()
    combined = pd.concat(parts, ignore_index=True)
    logger.info(
        "Total: %d rows, %d hits (%.1f%%)",
        len(combined), combined["label"].sum(), combined["label"].mean() * 100,
    )
    return combined


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = compute_all()
    if not df.empty:
        df = add_btc_context(df)
        out = HIST_DIR / "training_data.parquet"
        df.to_parquet(out)
        print(f"\nSaved to {out}")
        print(f"Shape: {df.shape}")
        print(f"Labels: {df['label'].value_counts().to_dict()}")
        print(f"Hit rate: {df['label'].mean()*100:.1f}%")
