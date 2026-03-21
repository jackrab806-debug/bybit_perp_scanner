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

# Feature columns stored in ml_snapshots (all known at snapshot time)
_FEATURE_COLS = [
    "funding_rate", "funding_z",
    "oi_change_1h_pct", "oi_change_4h_pct", "oi_z_24h",
    "thin_pct", "depth_bid_usdt", "depth_ask_usdt",
    "vacuum_dist_bid", "vacuum_dist_ask",
    "spread_z", "imbalance", "bb_width_pct",
    "vol_ratio_5m", "oi_to_depth_ratio", "funding_x_oi",
    "btc_change_1h", "btc_change_4h",
    "hour_utc", "mins_to_settlement", "is_weekend",
    "has_fs", "has_vb", "has_ve", "has_ca", "has_oi",
    "num_event_types_2h", "max_event_score",
]

# Threshold for binary label (abs max move in 4h window)
_MOVE_THRESHOLD_PCT = 5.0


def extract_training_data(
    db_path: str | Path = _DEFAULT_DB,
) -> Tuple[pd.DataFrame, list[str]]:
    """Return (dataframe, feature_col_names) ready for model training."""

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

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
    df["label"] = (df["abs_max_move_4h"].abs() >= _MOVE_THRESHOLD_PCT).astype(int)

    # ── Validate feature columns exist ──
    feature_cols = [c for c in _FEATURE_COLS if c in df.columns]
    missing = set(_FEATURE_COLS) - set(feature_cols)
    if missing:
        logger.warning("Missing feature columns: %s", missing)

    # Coerce to numeric
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop columns that are >80% NaN
    keep = []
    for c in feature_cols:
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
