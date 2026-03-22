"""Extract training data from ml_snapshots for the fragility prediction model.

Each row = one hourly snapshot with pre-computed features.
Label = 1 if abs_max_move_4h >= 5.0%, else 0.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Tuple

import pandas as pd

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_DB = _ROOT / "data" / "events.db"

# ── Raw columns stored in ml_snapshots ────────────────────────────────────────

_RAW_FEATURES = [
    # Funding
    "funding_rate", "funding_z",
    # OI
    "oi_change_1h_pct", "oi_change_4h_pct", "oi_z_24h",
    # Orderbook
    "thin_pct", "depth_bid_usdt", "depth_ask_usdt",
    "vacuum_dist_bid", "vacuum_dist_ask",
    "spread_bps", "imbalance", "convexity",
    # Volatility
    "bb_width_pct", "rv_pct",
    # Flow
    "cvd_ratio_24h", "taker_proxy", "price_accel",
    # Composite scores
    "compression", "sps", "lfi", "rank",
    # Pre-computed derived
    "oi_to_depth_ratio", "funding_x_oi", "vacuum_asymmetry",
    # BTC context
    "btc_change_1h", "btc_change_4h",
    # Time
    "hour_utc", "mins_to_settlement", "is_weekend",
    # Events
    "has_fs", "has_vb", "has_ve", "has_ca", "has_oi",
    "num_event_types_2h", "max_event_score",
    # Raw OI value (for log transform)
    "oi_usd",
    # New columns (NULL for old rows, populated going forward)
    "spread_z", "vol_ratio_5m",
]

# ── Computed interaction features (kept lean — only proven strong ones) ────────

_COMPUTED_FEATURES = [
    "abs_funding",
    "pressure_ratio",
    "settlement_urgency",
    "event_intensity",
    "multi_signal",
]

# Threshold for binary label
_MOVE_THRESHOLD_PCT = 5.0


def extract_training_data(
    db_path: str | Path = _DEFAULT_DB,
) -> Tuple[pd.DataFrame, list[str]]:
    """Return (dataframe, feature_col_names) ready for model training."""

    conn = sqlite3.connect(str(db_path))

    df = pd.read_sql_query(
        """
        SELECT *
        FROM ml_snapshots
        WHERE label_filled = 1
          AND abs_max_move_4h IS NOT NULL
        ORDER BY timestamp
        """,
        conn,
    )
    conn.close()

    if df.empty:
        logger.error("No labeled snapshots found in ml_snapshots")
        return df, []

    logger.info("Loaded %d labeled snapshots", len(df))

    # ── Label ──
    df["abs_max_move"] = df["abs_max_move_4h"].abs().fillna(0)
    df["label"] = (df["abs_max_move"] >= _MOVE_THRESHOLD_PCT).astype(int)

    # ── Coerce raw columns to numeric ──
    raw_available = [c for c in _RAW_FEATURES if c in df.columns]
    for c in raw_available:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Fill NaN in raw features with 0
    for c in raw_available:
        df[c] = df[c].fillna(0)

    # ── Computed interaction features ──
    _add_computed_features(df)

    # ── Assemble feature list ──
    feature_cols = raw_available + _COMPUTED_FEATURES

    # Drop columns that are >80% NaN (handles spread_z/vol_ratio_5m if still NULL)
    keep = []
    for c in feature_cols:
        if c not in df.columns:
            continue
        frac = df[c].notna().mean()
        if frac > 0.20:
            keep.append(c)
        else:
            logger.debug("Dropping feature %s (%.0f%% missing)", c, 100 * (1 - frac))
    feature_cols = keep

    # Fill remaining NaNs with 0
    for c in feature_cols:
        df[c] = df[c].fillna(0)

    logger.info(
        "Extracted %d samples, %d features, %d positive (%.1f%%)",
        len(df), len(feature_cols), df["label"].sum(),
        100 * df["label"].mean(),
    )
    return df, feature_cols


def _add_computed_features(df: pd.DataFrame) -> None:
    """Compute the strongest interaction features only."""

    df["abs_funding"] = df["funding_rate"].abs()

    # Pressure / resistance: funding pressure * OI / book depth
    df["pressure_ratio"] = (
        df["funding_rate"].abs() * df["oi_usd"]
        / (df["depth_ask_usdt"] + 1)
    )

    # Non-linear settlement proximity (strongest in last 30 min)
    df["settlement_urgency"] = 1.0 / (df["mins_to_settlement"] + 10)

    # Event intensity (weighted by empirical hit rates)
    df["event_intensity"] = (
        df["has_fs"] * 1.0
        + df["has_vb"] * 0.5
        + df["has_ve"] * 3.0
        + df["has_ca"] * 2.0
        + df["has_oi"] * 1.5
    )

    # Multi-signal flag
    df["multi_signal"] = (df["num_event_types_2h"] >= 2).astype(int)


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
