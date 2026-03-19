"""Extract training data from events.db for the fragility prediction model.

Each row = one event with its evaluated outcome.
Features = everything known at event time (no look-ahead).
Label = 1 if HIT or STRONG_HIT, else 0.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_DB = _ROOT / "data" / "events_vps.db"

# Features extracted from features_json (available for all event types)
_COMMON_FEATS = [
    "compression", "sps", "lfi", "rank",
    "funding_z", "funding_current", "cum_24h",
    "minutes_to_settlement",
    "oi_z_24h", "oi_z_1h",
    "rv_pct", "bb_width_pct", "range_hours",
    "cvd_ratio_24h", "taker_proxy", "price_accel",
    "thin_pct", "spread_bps",
    "vacuum_dist_bid", "vacuum_dist_ask",
    "depth_bid_usdt", "depth_ask_usdt",
    "convexity",
]

# Engineered feature column names (added during build)
_ENGINEERED = [
    "score",
    "is_fs", "is_vb", "is_ve", "is_ca", "is_oi",
    "dir_long",
    "tier",
    "hour_utc",
    "near_settlement",
    "vacuum_max",
    "depth_imbalance",
    "num_event_types_2h",
    "total_events_2h",
    "max_score_2h",
]


def extract_training_data(
    db_path: str | Path = _DEFAULT_DB,
) -> Tuple[pd.DataFrame, list[str]]:
    """Return (dataframe, feature_col_names) ready for model training."""

    conn = sqlite3.connect(str(db_path))

    # Join outcomes with event features
    rows = conn.execute("""
        SELECT
            o.symbol,
            o.event_type,
            o.event_score,
            o.event_direction,
            o.event_timestamp,
            o.move_4h_pct,
            o.max_favorable_pct,
            o.max_adverse_pct,
            o.outcome,
            e.features_json,
            e.score   AS e_score,
            e.direction AS e_dir
        FROM outcomes o
        JOIN events e
            ON o.symbol = e.symbol
           AND o.event_type = e.event_type
           AND o.event_timestamp = e.timestamp
        WHERE o.outcome IS NOT NULL
        ORDER BY o.event_timestamp
    """).fetchall()

    cols = [
        "symbol", "event_type", "event_score", "event_direction",
        "event_timestamp", "move_4h_pct", "max_favorable_pct",
        "max_adverse_pct", "outcome", "features_json", "e_score", "e_dir",
    ]
    df = pd.DataFrame(rows, columns=cols)
    conn.close()

    if df.empty:
        logger.error("No joined outcome+event rows found")
        return df, []

    logger.info("Loaded %d outcome rows", len(df))

    # ── Label ──
    # Include PARTIAL (3-5% move) for more signal; still useful as "actionable move"
    df["label"] = df["outcome"].isin(["HIT", "STRONG_HIT", "PARTIAL"]).astype(int)

    # ── Parse features_json ──
    def _parse(js: str) -> dict:
        try:
            return json.loads(js)
        except Exception:
            return {}

    parsed = df["features_json"].apply(_parse)

    for col in _COMMON_FEATS:
        df[col] = parsed.apply(lambda d, c=col: d.get(c))

    # Coerce to numeric, NaN for missing / "NaN" strings
    for col in _COMMON_FEATS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Engineered features ──
    df["score"] = pd.to_numeric(df["event_score"], errors="coerce").fillna(
        pd.to_numeric(df["e_score"], errors="coerce")
    ).fillna(50)

    df["is_fs"] = (df["event_type"] == "FUNDING_SQUEEZE_SETUP").astype(int)
    df["is_vb"] = (df["event_type"] == "VACUUM_BREAK").astype(int)
    df["is_ve"] = (df["event_type"] == "VOLUME_EXPLOSION").astype(int)
    df["is_ca"] = (df["event_type"] == "CASCADE_ACTIVE").astype(int)
    df["is_oi"] = (df["event_type"] == "OI_SURGE").astype(int)

    df["dir_long"] = df["event_direction"].apply(
        lambda x: 1 if x in ("LONG", "UP") else 0
    )

    # Tier from features_json
    df["tier"] = parsed.apply(lambda d: d.get("tier", 3))
    df["tier"] = pd.to_numeric(df["tier"], errors="coerce").fillna(3).astype(int)

    # Time features
    df["event_ts"] = pd.to_datetime(df["event_timestamp"], utc=True)
    df["hour_utc"] = df["event_ts"].dt.hour

    # Settlement proximity
    def _mins_to_settle(ts):
        cur = ts.hour * 60 + ts.minute
        return min((s * 60 - cur) % 1440 for s in (0, 480, 960, 1440))

    df["near_settlement"] = df["event_ts"].apply(
        lambda t: 1 if _mins_to_settle(t) <= 60 else 0
    )

    # Orderbook derived
    vb = df["vacuum_dist_bid"].fillna(0)
    va = df["vacuum_dist_ask"].fillna(0)
    df["vacuum_max"] = np.maximum(vb, va)

    db = df["depth_bid_usdt"].fillna(0)
    da = df["depth_ask_usdt"].fillna(0)
    df["depth_imbalance"] = (db - da) / (db + da + 1e-6)

    # ── Interaction features ──
    df["thin_x_funding"] = df["thin_pct"].fillna(0) * df["funding_z"].fillna(0).abs()
    df["thin_x_oi"] = df["thin_pct"].fillna(0) * df["oi_z_24h"].fillna(0).abs()
    df["compression_x_thin"] = df["compression"].fillna(0) * df["thin_pct"].fillna(0)
    df["funding_x_oi"] = df["funding_z"].fillna(0).abs() * df["oi_z_24h"].fillna(0).abs()
    df["bb_x_rv"] = df["bb_width_pct"].fillna(0) * df["rv_pct"].fillna(0)
    df["vacuum_asymmetry"] = (
        (df["vacuum_dist_ask"].fillna(0) - df["vacuum_dist_bid"].fillna(0))
        / (df["vacuum_dist_ask"].fillna(0) + df["vacuum_dist_bid"].fillna(0) + 1e-6)
    )

    # ── Multi-event features (most important) ──
    df = df.sort_values("event_ts").reset_index(drop=True)
    _add_multi_event_features(df)

    # ── Assemble final feature list ──
    _INTERACTIONS = [
        "thin_x_funding", "thin_x_oi", "compression_x_thin",
        "funding_x_oi", "bb_x_rv", "vacuum_asymmetry",
    ]
    feature_cols = _ENGINEERED + _COMMON_FEATS + _INTERACTIONS

    # Drop any column that is >80% NaN
    keep = []
    for c in feature_cols:
        if c in df.columns and df[c].notna().mean() > 0.20:
            keep.append(c)
        else:
            logger.debug("Dropping feature %s (%.0f%% missing)", c,
                         100 * (1 - df[c].notna().mean()) if c in df.columns else 100)
    feature_cols = keep

    # Fill remaining NaNs with 0
    for c in feature_cols:
        df[c] = df[c].fillna(0)

    logger.info(
        "Extracted %d samples, %d features, %d HITs (%.1f%%)",
        len(df), len(feature_cols), df["label"].sum(),
        100 * df["label"].mean(),
    )
    return df, feature_cols


def _add_multi_event_features(df: pd.DataFrame) -> None:
    """For each row, count events for the same symbol in the prior 2h window."""

    df["num_event_types_2h"] = 1
    df["total_events_2h"] = 1
    df["max_score_2h"] = df["score"]

    # Vectorised approach: for each symbol, use a rolling time window
    for sym in df["symbol"].unique():
        mask = df["symbol"] == sym
        idxs = df.index[mask]
        ts_arr = df.loc[idxs, "event_ts"].values          # datetime64
        et_arr = df.loc[idxs, "event_type"].values         # str
        sc_arr = df.loc[idxs, "score"].values.astype(float)

        for i, idx in enumerate(idxs):
            t = ts_arr[i]
            window_start = t - np.timedelta64(2, "h")
            in_window = (ts_arr >= window_start) & (ts_arr <= t)
            df.at[idx, "num_event_types_2h"] = len(set(et_arr[in_window]))
            df.at[idx, "total_events_2h"] = int(in_window.sum())
            df.at[idx, "max_score_2h"] = float(sc_arr[in_window].max())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-8s %(message)s")
    df, feats = extract_training_data()
    print(f"\nShape: {df.shape}")
    print(f"Label: {df['label'].value_counts().to_dict()}")
    print(f"Features ({len(feats)}): {feats}")
    out = _ROOT / "data" / "ml_training_data.csv"
    df.to_csv(out, index=False)
    print(f"Saved to {out}")
