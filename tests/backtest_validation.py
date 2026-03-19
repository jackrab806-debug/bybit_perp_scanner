"""Complete backtest validation pipeline.

1. Replay historical features through EventDetector (settlement-aware)
2. Label events with triple-barrier using 1m klines
3. Generate matched baselines
4. Run statistical validation (KS, bootstrap CI, Cohen's d, walk-forward)
5. Print PASS/FAIL summary per event type

Usage:
    python -u -m tests.backtest_validation
    python -u -m tests.backtest_validation --max-symbols 10
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import warnings
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

sys.path.insert(0, ".")

from src.events.definitions import Event, EventDetector, EventType
from src.backtest.labeling import LabeledEvent, label_event, label_events
from src.backtest.baseline import BaselineSampler
from src.backtest.validation import validate_event_type, ValidationResult
from src.features import (
    compute_flow_features,
    compute_funding_features,
    compute_oi_features,
    compute_volatility_features,
    compression_score,
    settlement_pressure_score,
    liquidity_fragility_index,
)
from src.features.utils import robust_z
from src.scanner.pressure_scanner import _pressure_rank

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"


def _p(msg: str) -> None:
    print(msg, flush=True)


# ── Constants ─────────────────────────────────────────────────────────────────

MIN_KLINE_ROWS = 48
BASELINE_PER_EVENT = 10
MAX_BARS_1M = 3600       # 60 hours of 1m bars as vertical barrier
MAX_BARS_1H = 60         # fallback for 1h labeling

SETTLEMENT_HOURS = (0, 8, 16)
APPROACH_MINUTES = 60     # sample every 5 min when < 60 min to settlement
APPROACH_STEP_MIN = 5


# ── Settlement-aware timestamp generation ─────────────────────────────────────

def _generate_replay_timestamps(klines_df: pd.DataFrame) -> List[pd.Timestamp]:
    """Generate replay timestamps: 1h everywhere, +5min during APPROACH phase.

    Settlements at 00:00, 08:00, 16:00 UTC — 3 per day.
    Within 60 minutes before each settlement, sample every 5 minutes.
    """
    if klines_df is None or len(klines_df) < MIN_KLINE_ROWS:
        return []

    ts_start = klines_df.iloc[MIN_KLINE_ROWS]["timestamp"]
    ts_end = klines_df.iloc[-1]["timestamp"]

    if hasattr(ts_start, "to_pydatetime"):
        ts_start = ts_start.to_pydatetime()
        ts_end = ts_end.to_pydatetime()
    if ts_start.tzinfo is None:
        ts_start = ts_start.replace(tzinfo=timezone.utc)
    if ts_end.tzinfo is None:
        ts_end = ts_end.replace(tzinfo=timezone.utc)

    timestamps: List[pd.Timestamp] = []
    seen = set()

    # Hourly grid
    cur = ts_start.replace(minute=0, second=0, microsecond=0)
    while cur <= ts_end:
        key = int(cur.timestamp())
        if key not in seen:
            timestamps.append(pd.Timestamp(cur))
            seen.add(key)
        cur += timedelta(hours=1)

    # 5-minute steps during APPROACH windows (60 min before settlement)
    day = ts_start.replace(hour=0, minute=0, second=0, microsecond=0)
    while day <= ts_end + timedelta(days=1):
        for sh in SETTLEMENT_HOURS:
            settle_time = day.replace(hour=sh)
            approach_start = settle_time - timedelta(minutes=APPROACH_MINUTES)
            t = approach_start
            while t < settle_time:
                if ts_start <= t <= ts_end:
                    key = int(t.timestamp())
                    if key not in seen:
                        timestamps.append(pd.Timestamp(t))
                        seen.add(key)
                t += timedelta(minutes=APPROACH_STEP_MIN)
        day += timedelta(days=1)

    timestamps.sort()
    return timestamps


# ── Deterministic kline-based OB proxies ──────────────────────────────────────

def _kline_ob_proxy(
    klines_df: pd.DataFrame,
    idx: int,
) -> Dict[str, float]:
    """Deterministic OB feature proxies from kline data (no RNG noise).

    thin_pct:  inverse of volume relative to 7-day median.
               Low volume ≈ thin book.
    spread_bps: (high - low) / close * 10000.
               Intrabar range as spread proxy.
    vacuum_dist: inverse of ATR percentile rank.
               Low ATR during compression = large vacuum outside range.
    convexity:  volume concentration ratio (current vs trailing mean).
    """
    row = klines_df.iloc[idx]
    close = float(row["close"])
    high = float(row["high"])
    low = float(row["low"])
    volume = float(row["volume"])

    if close <= 0:
        return _nan_ob()

    # 7-day lookback (168 hourly bars)
    lb_start = max(0, idx - 168)
    lookback = klines_df.iloc[lb_start:idx + 1]

    # thin_pct: 1 - (vol / median_7d_vol), percentile-ranked over 7d
    med_vol = lookback["volume"].median()
    if med_vol > 0:
        thin_raw = 1.0 - (volume / med_vol)
    else:
        thin_raw = 0.5
    thin_series = 1.0 - (lookback["volume"] / med_vol).clip(0, 2)
    thin_pct = float((thin_series <= thin_raw).mean())
    thin_pct = float(np.clip(thin_pct, 0.01, 0.99))

    # spread_bps: intrabar range
    spread_bps = (high - low) / close * 10_000

    # ATR (14-bar) for vacuum proxy
    atr_lb = max(0, idx - 14)
    atr_slice = klines_df.iloc[atr_lb:idx + 1]
    tr = atr_slice["high"] - atr_slice["low"]
    atr = tr.mean() if len(tr) > 0 else 0
    atr_pct = atr / close * 100 if close > 0 else 0

    # vacuum_dist: inverse ATR percentile (low ATR = high vacuum)
    atr_history = (lookback["high"] - lookback["low"]).div(lookback["close"]) * 100
    atr_rank = float((atr_history <= atr_pct).mean()) if len(atr_history) > 1 else 0.5
    vacuum_base = (1.0 - atr_rank) * 2000  # scale to bps
    vacuum_bid = float(np.clip(vacuum_base * 0.9, 10, 5000))
    vacuum_ask = float(np.clip(vacuum_base * 1.1, 10, 5000))

    # convexity: volume concentration (current bar vs mean)
    mean_vol = lookback["volume"].mean()
    convexity = (volume / mean_vol) if mean_vol > 0 else 1.0
    convexity = float(np.clip(convexity, 0.1, 5.0))

    depth_usdt = volume * close * 0.01

    return {
        "mid_price": close,
        "spread_bps": float(spread_bps),
        "thin_pct": thin_pct,
        "vacuum_dist_bid": vacuum_bid,
        "vacuum_dist_ask": vacuum_ask,
        "depth_bid_usdt": depth_usdt,
        "depth_ask_usdt": depth_usdt,
        "depth_ratio": 1.0,
        "depth_imbalance": 0.0,
        "vacuum_imbalance": (vacuum_ask - vacuum_bid) / max(vacuum_ask + vacuum_bid, 1),
        "convexity": convexity,
        "inner_depth_10": depth_usdt * 0.3,
    }


def _nan_ob() -> Dict[str, float]:
    nan = float("nan")
    return {
        "mid_price": nan, "spread_bps": nan, "thin_pct": nan,
        "vacuum_dist_bid": nan, "vacuum_dist_ask": nan,
        "depth_bid_usdt": nan, "depth_ask_usdt": nan,
        "depth_ratio": nan, "depth_imbalance": nan,
        "vacuum_imbalance": nan, "convexity": nan, "inner_depth_10": nan,
    }


# ── Feature replay engine (settlement-aware) ─────────────────────────────────

def replay_symbol(
    symbol: str,
    tier: int,
    klines_df: pd.DataFrame,
    oi_df: Optional[pd.DataFrame],
    funding_df: Optional[pd.DataFrame],
) -> List[Dict[str, Any]]:
    """Compute feature snapshots: 1h grid + 5-min near settlements.

    Optimization: approach-phase 5-min steps reuse cached kline-derived
    features (vol, flow, OI, OB) since 1h klines don't change between
    intra-hour steps. Only funding + composites are recomputed.
    """
    if klines_df is None or len(klines_df) < MIN_KLINE_ROWS:
        return []

    timestamps = _generate_replay_timestamps(klines_df)
    if not timestamps:
        return []

    records: List[Dict[str, Any]] = []
    spread_bps_hist: List[float] = []
    spread_z_hist: List[float] = []

    # Cache for kline-derived features (keyed by kline index)
    _cached_idx: int = -1
    _cached_vol: Dict[str, Any] = {}
    _cached_flow: Dict[str, Any] = {}
    _cached_oi: Dict[str, Any] = {}
    _cached_ob: Dict[str, float] = {}

    for ts in timestamps:
        k_mask = klines_df["timestamp"] <= ts
        n_rows = int(k_mask.sum())
        if n_rows < MIN_KLINE_ROWS:
            continue
        idx = n_rows - 1

        now = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        # Recompute kline features only when kline index advances
        if idx != _cached_idx:
            _cached_idx = idx
            # Feature functions only need ~14 days of history max
            k_slice = klines_df.iloc[max(0, idx - 336):idx + 1]
            oi_slice = oi_df[oi_df["timestamp"] <= ts].tail(336) if oi_df is not None else None

            _cached_vol = compute_volatility_features(k_slice)
            _cached_flow = compute_flow_features(k_slice)
            _cached_oi = compute_oi_features(oi_slice) if oi_slice is not None and len(oi_slice) >= 2 else {}
            _cached_ob = _kline_ob_proxy(klines_df, idx)

        # Always recompute funding (settlement_phase changes intra-hour)
        f_slice = funding_df[funding_df["timestamp"] <= ts].tail(200) if funding_df is not None else None
        funding_feats = compute_funding_features(
            f_slice if f_slice is not None and len(f_slice) >= 2 else None, now=now
        )

        ob_feats = _cached_ob
        rv = _cached_vol.get("rv_pct", math.nan)
        bb_w = _cached_vol.get("bb_width_pct", math.nan)
        oi_z = _cached_oi.get("oi_z_24h", math.nan)
        rng_h = _cached_vol.get("range_hours", math.nan)
        fund_z = funding_feats.get("funding_z", math.nan)
        mins = funding_feats.get("minutes_to_settlement", math.nan)
        thin_p = ob_feats.get("thin_pct", math.nan)

        cs = compression_score(rv, bb_w, oi_z, rng_h)

        if not np.isnan(fund_z) and fund_z != 0:
            vac_sq = ob_feats["vacuum_dist_bid"] if fund_z > 0 else ob_feats["vacuum_dist_ask"]
        else:
            vac_sq = math.nan

        sps = settlement_pressure_score(
            funding_z=fund_z, oi_z_7d=oi_z,
            vacuum_dist_squeeze_dir=vac_sq,
            thin_pct=thin_p, minutes_to_settle=mins,
        )

        spread_bps = ob_feats.get("spread_bps", math.nan)
        if not np.isnan(spread_bps) and idx != (_cached_idx - 1):
            spread_bps_hist.append(spread_bps)
        spread_z = robust_z(spread_bps, spread_bps_hist) if len(spread_bps_hist) >= 5 else math.nan
        if not np.isnan(spread_z):
            spread_z_hist.append(spread_z)

        convexity = ob_feats.get("convexity", math.nan)
        lfi = liquidity_fragility_index(
            thin_pct=thin_p, spread_z=spread_z, convexity=convexity,
        )
        rank = _pressure_rank(sps, cs, lfi, tier)

        records.append({
            "symbol": symbol,
            "tier": tier,
            "timestamp": now,
            "compression": cs,
            "sps": sps,
            "lfi": lfi,
            "rank": rank,
            "funding_feats": funding_feats,
            "oi_feats": _cached_oi,
            "vol_feats": _cached_vol,
            "flow_feats": _cached_flow,
            "ob_feats": ob_feats,
            "spread_z_history": list(spread_z_hist[-200:]),
            "spread_bps_history": list(spread_bps_hist[-200:]),
        })

    return records


# ── Historical replay with correct timestamps ────────────────────────────────

def historical_replay(records: List[Dict[str, Any]]) -> List[Event]:
    """Replay feature snapshots through EventDetector with historical timestamps."""
    from types import SimpleNamespace

    _DEFAULTS: Dict[str, Any] = {
        "funding_feats": {}, "oi_feats": {}, "vol_feats": {},
        "flow_feats": {}, "ob_feats": {},
        "spread_z_history": [], "spread_bps_history": [],
        "compression": math.nan, "sps": math.nan,
        "lfi": math.nan, "rank": math.nan, "tier": 3,
    }

    sorted_records = sorted(records, key=lambda r: r.get("timestamp", datetime.min))
    detector = EventDetector()
    all_events: List[Event] = []

    for rec in sorted_records:
        state = SimpleNamespace(**{**_DEFAULTS, **rec})
        sym = getattr(state, "symbol", "UNKNOWN")
        ts = rec.get("timestamp")
        if ts is None:
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        checkers = [
            (EventType.COMPRESSION_SQUEEZE_SETUP, detector._check_compression_squeeze),
            (EventType.FUNDING_SQUEEZE_SETUP,     detector._check_funding_squeeze),
            (EventType.VACUUM_BREAK,              detector._check_vacuum_break),
            (EventType.CASCADE_ACTIVE,            detector._check_cascade_active),
        ]
        for etype, checker in checkers:
            if not detector._can_fire(sym, etype, ts):
                continue
            event = checker(sym, state, ts)
            if event is not None:
                event.timestamp = ts
                detector._record(sym, etype, ts)
                all_events.append(event)

    return all_events


# ── Price series builder ──────────────────────────────────────────────────────

def build_price_map(symbols: List[str]) -> Dict[str, pd.Series]:
    """Build symbol -> close price Series. Prefers 1m data, falls back to 1h."""
    pm: Dict[str, pd.Series] = {}
    for sym in symbols:
        path_1m = DATA_DIR / "klines_1m" / f"{sym}.parquet"
        path_1h = DATA_DIR / "klines" / f"{sym}.parquet"
        path = path_1m if path_1m.exists() else path_1h
        if not path.exists():
            continue
        df = pd.read_parquet(path).sort_values("timestamp")
        series = df.set_index("timestamp")["close"]
        if series.index.tz is None:
            series.index = series.index.tz_localize("UTC")
        pm[sym] = series
    return pm


def build_sigma_map(price_map: Dict[str, pd.Series]) -> Dict[str, float]:
    """Compute per-symbol sigma (std of returns) for barrier sizing."""
    sm: Dict[str, float] = {}
    for sym, series in price_map.items():
        rets = series.pct_change().dropna()
        if len(rets) >= 60:
            sm[sym] = float(rets.tail(10080).std())  # last ~7 days
        else:
            sm[sym] = 0.005
    return sm


def _detect_resolution(price_map: Dict[str, pd.Series]) -> str:
    """Check if price_map uses 1m or 1h data."""
    for sym, series in price_map.items():
        if len(series) > 2:
            median_gap = series.index.to_series().diff().median()
            if median_gap < pd.Timedelta(minutes=5):
                return "1m"
            return "1h"
    return "1h"


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(max_symbols: int = 50) -> None:
    t0 = time.monotonic()

    import yaml
    cfg_path = ROOT / "configs" / "symbols.yaml"
    if not cfg_path.exists():
        _p("Error: configs/symbols.yaml not found. Run backfill first.")
        sys.exit(1)

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    symbols_by_tier: Dict[int, List[str]] = {}
    for tier in (1, 2, 3):
        key = f"tier_{tier}"
        if key in cfg:
            symbols_by_tier[tier] = cfg[key]

    all_symbols: List[str] = []
    sym_tier: Dict[str, int] = {}
    for tier, syms in sorted(symbols_by_tier.items()):
        for s in syms:
            if len(all_symbols) < max_symbols:
                all_symbols.append(s)
                sym_tier[s] = tier

    has_1m = (DATA_DIR / "klines_1m").exists() and any(
        (DATA_DIR / "klines_1m" / f"{s}.parquet").exists() for s in all_symbols
    )

    _p(f"{'=' * 70}")
    _p(f"  BACKTEST VALIDATION PIPELINE")
    _p(f"  Symbols: {len(all_symbols)}  |  1m data: {'YES' if has_1m else 'NO (using 1h)'}")
    _p(f"  OB proxies: deterministic (volume/range/ATR-based)")
    _p(f"  Replay: 1h grid + 5-min near settlement")
    _p(f"{'=' * 70}")

    # ── Step 1: Replay features ───────────────────────────────────────────────
    _p(f"\n[1/5] Replaying features through EventDetector …")
    all_records: List[Dict[str, Any]] = []

    for idx, sym in enumerate(all_symbols):
        k_path = DATA_DIR / "klines" / f"{sym}.parquet"
        o_path = DATA_DIR / "oi" / f"{sym}.parquet"
        f_path = DATA_DIR / "funding" / f"{sym}.parquet"

        kdf = pd.read_parquet(k_path).sort_values("timestamp") if k_path.exists() else None
        odf = pd.read_parquet(o_path).sort_values("timestamp") if o_path.exists() else None
        fdf = pd.read_parquet(f_path).sort_values("timestamp") if f_path.exists() else None

        records = replay_symbol(sym, sym_tier[sym], kdf, odf, fdf)
        all_records.extend(records)

        if (idx + 1) % 5 == 0 or idx == len(all_symbols) - 1:
            _p(f"  {idx + 1}/{len(all_symbols)} symbols done  ({len(all_records):,} snapshots)")

    _p(f"  Total feature snapshots: {len(all_records):,}")

    events = historical_replay(all_records)
    _p(f"  Events detected: {len(events)}")

    by_type: Dict[str, List[Event]] = defaultdict(list)
    for ev in events:
        by_type[ev.event_type.value if hasattr(ev.event_type, "value") else str(ev.event_type)].append(ev)

    for etype, evs in sorted(by_type.items()):
        syms = set(e.symbol for e in evs)
        _p(f"    {etype:<35s}  {len(evs):>4d} events  ({len(syms)} symbols)")

    if not events:
        _p("\n  No events detected — cannot proceed with validation.")
        _p("  This may indicate thresholds are too strict for the available data.")
        _p(f"\n{'=' * 70}")
        _p(f"  SUMMARY: NO EVENTS TO VALIDATE")
        _p(f"{'=' * 70}")
        return

    # ── Step 2: Build price map & label events ────────────────────────────────
    _p(f"\n[2/5] Labeling events with triple barrier …")
    price_map = build_price_map(all_symbols)
    sigma_map = build_sigma_map(price_map)
    resolution = _detect_resolution(price_map)
    max_bars = MAX_BARS_1M if resolution == "1m" else MAX_BARS_1H
    _p(f"  Price resolution: {resolution}  |  max_bars: {max_bars}")

    labeled_all = label_events(events, price_map, sigma_map, max_bars=max_bars)
    _p(f"  Labeled: {len(labeled_all)} / {len(events)} events")

    labeled_by_type: Dict[str, List[LabeledEvent]] = defaultdict(list)
    for le in labeled_all:
        labeled_by_type[le.event_type].append(le)

    for etype, les in sorted(labeled_by_type.items()):
        wins = sum(1 for e in les if e.hit_tp)
        wr = wins / len(les) * 100 if les else 0
        avg_mfe = np.mean([e.mfe_60m for e in les]) * 100 if les else 0
        _p(f"    {etype:<35s}  n={len(les):>4d}  win_rate={wr:.1f}%  avg_mfe_60m={avg_mfe:.3f}%")

    # ── Step 3: Generate baseline samples ─────────────────────────────────────
    _p(f"\n[3/5] Generating matched baseline samples ({BASELINE_PER_EVENT} per event) …")
    all_event_timestamps = [e.timestamp for e in events]
    sampler = BaselineSampler(price_map, sigma_map, seed=42)

    baseline_by_type: Dict[str, List[LabeledEvent]] = defaultdict(list)
    total_bl = 0

    for ev in events:
        etype_str = ev.event_type.value if hasattr(ev.event_type, "value") else str(ev.event_type)
        samples = sampler.sample(ev, all_event_timestamps, n_samples=BASELINE_PER_EVENT)
        for s in samples:
            baseline_by_type[etype_str].append(s.labeled)
            total_bl += 1

    _p(f"  Total baseline samples: {total_bl:,}")
    for etype, bls in sorted(baseline_by_type.items()):
        _p(f"    {etype:<35s}  {len(bls):>4d} baseline samples")

    # ── Step 4: Statistical validation ────────────────────────────────────────
    _p(f"\n[4/5] Running statistical validation …")
    results: Dict[str, ValidationResult] = {}

    for etype in sorted(set(list(labeled_by_type.keys()) + list(baseline_by_type.keys()))):
        ev_labeled = labeled_by_type.get(etype, [])
        bl_labeled = baseline_by_type.get(etype, [])

        if len(ev_labeled) < 2:
            _p(f"  {etype}: skipped (< 2 labeled events)")
            continue

        result = validate_event_type(ev_labeled, bl_labeled)
        results[etype] = result
        _p(f"\n  ── {etype} ──")
        _p(result.summary())

    # ── Step 5: Summary table ─────────────────────────────────────────────────
    elapsed = time.monotonic() - t0

    _p(f"\n{'=' * 70}")
    _p(f"  FINAL SUMMARY")
    _p(f"{'=' * 70}")

    header = (
        f"  {'Event type':<35s} {'Count':>5s} {'WinR':>6s} {'MFE60m':>8s} "
        f"{'KS p':>8s} {'Cohen d':>8s} {'All?':>6s}"
    )
    _p(header)
    _p(f"  {'─' * 78}")

    all_pass = True
    for etype in sorted(results.keys()):
        r = results[etype]
        les = labeled_by_type.get(etype, [])
        avg_mfe = np.mean([e.mfe_60m for e in les]) * 100 if les else 0

        ks_p = f"{r.ks_pvalue:.4f}" if not math.isnan(r.ks_pvalue) else "n/a"
        cd = f"{r.cohens_d:.3f}" if not math.isnan(r.cohens_d) else "n/a"
        pass_str = "\033[92mPASS\033[0m" if r.passes_all else "\033[91mFAIL\033[0m"

        _p(
            f"  {etype:<35s} {r.n_events:>5d} {r.win_rate * 100:>5.1f}% "
            f"{avg_mfe:>7.3f}% {ks_p:>8s} {cd:>8s} {pass_str}"
        )
        if not r.passes_all:
            all_pass = False

    for etype, evs in sorted(by_type.items()):
        if etype not in results:
            _p(f"  {etype:<35s} {len(evs):>5d}     — (too few labeled events)")

    _p(f"\n  Elapsed: {elapsed:.1f}s")
    overall = "\033[92mPASS\033[0m" if all_pass and results else "\033[91mFAIL\033[0m"
    _p(f"  Overall: {overall}")
    _p("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest validation pipeline")
    parser.add_argument(
        "--max-symbols", type=int, default=50,
        help="Max symbols to process (default: all 50)",
    )
    args = parser.parse_args()
    run(max_symbols=args.max_symbols)
