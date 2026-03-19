"""Live Fragility Predictor — scores coins using the trained model.

Loads data/fragility_model.pkl and provides predict(features) → probability.
Graceful degradation: returns None if model not loaded.
"""

from __future__ import annotations

import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..scanner.pressure_scanner import SymbolState

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = _ROOT / "data" / "fragility_model.pkl"


class FragilityPredictor:
    """Score coins using the trained ML model."""

    def __init__(self) -> None:
        self.model: Any = None
        self.model_type: Optional[str] = None
        self.feature_cols: List[str] = []
        self.metrics: Dict[str, Any] = {}
        self.threshold: float = 0.20
        self._loaded = False

    # ── Load / reload ─────────────────────────────────────────────────────────

    def load_model(self) -> bool:
        if not MODEL_PATH.exists():
            logger.warning("No model at %s", MODEL_PATH)
            return False
        try:
            with open(MODEL_PATH, "rb") as f:
                blob = pickle.load(f)
            self.model = blob["model"]
            self.model_type = blob["model_type"]
            self.feature_cols = blob["feature_cols"]
            self.metrics = blob.get("metrics", {})
            self.threshold = self.metrics.get("best_threshold", 0.20)
            self._loaded = True
            logger.info(
                "FragilityPredictor loaded (%s, thr=%.3f, AUC=%.4f)",
                self.model_type, self.threshold, self.metrics.get("auc", 0),
            )
            return True
        except Exception as exc:
            logger.error("Failed to load model: %s", exc)
            return False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict(self, features: Dict[str, float]) -> Optional[float]:
        """Return P(>5% move in 4h) for one coin, or None."""
        if not self._loaded:
            return None
        try:
            X = np.array(
                [[features.get(c, 0) for c in self.feature_cols]],
                dtype=np.float32,
            )
            if self.model_type == "lightgbm":
                return float(self.model.predict(X)[0])
            return float(self.model.predict_proba(X)[0][1])
        except Exception as exc:
            logger.debug("Predict error: %s", exc)
            return None

    def predict_batch(
        self, features_list: List[Dict[str, float]]
    ) -> List[float]:
        """Predict for multiple coins at once (more efficient)."""
        if not self._loaded or not features_list:
            return [0.0] * len(features_list)
        try:
            X = np.array(
                [[f.get(c, 0) for c in self.feature_cols] for f in features_list],
                dtype=np.float32,
            )
            if self.model_type == "lightgbm":
                probs = self.model.predict(X)
            else:
                probs = self.model.predict_proba(X)[:, 1]
            return [float(p) for p in probs]
        except Exception as exc:
            logger.error("Batch predict error: %s", exc)
            return [0.0] * len(features_list)

    # ── Feature builder from live state ───────────────────────────────────────

    def build_features_from_state(
        self,
        state: "SymbolState",
        recent_events: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, float]:
        """Map live SymbolState → feature dict matching training columns."""

        ff = state.funding_feats or {}
        oi = state.oi_feats or {}
        vol = state.vol_feats or {}
        fl = state.flow_feats or {}
        ob = state.ob_feats or {}

        now = datetime.now(timezone.utc)
        cur_mins = now.hour * 60 + now.minute
        mins_to_set = min((s * 60 - cur_mins) % 1440 for s in (0, 480, 960, 1440))

        evts = recent_events or []
        etypes = {e.get("event_type", "") for e in evts}

        feat: Dict[str, float] = {
            # Engineered
            "score": max((e.get("score", 0) for e in evts), default=0),
            "is_fs": 1 if "FUNDING_SQUEEZE_SETUP" in etypes else 0,
            "is_vb": 1 if "VACUUM_BREAK" in etypes else 0,
            "is_ve": 1 if "VOLUME_EXPLOSION" in etypes else 0,
            "is_ca": 1 if "CASCADE_ACTIVE" in etypes else 0,
            "is_oi": 1 if "OI_SURGE" in etypes else 0,
            "dir_long": 1 if (ff.get("funding_current") or 0) < 0 else 0,
            "tier": float(state.tier),
            "hour_utc": float(now.hour),
            "near_settlement": 1.0 if mins_to_set <= 60 else 0.0,
            "vacuum_max": max(
                ob.get("vacuum_dist_bid", 0) or 0,
                ob.get("vacuum_dist_ask", 0) or 0,
            ),
            "depth_imbalance": _safe_imbalance(
                ob.get("depth_bid_usdt", 0) or 0,
                ob.get("depth_ask_usdt", 0) or 0,
            ),
            "num_event_types_2h": float(len(etypes)),
            "total_events_2h": float(len(evts)),
            "max_score_2h": max((e.get("score", 0) for e in evts), default=0),
            # Common from state
            "compression": _nan0(state.compression),
            "sps": _nan0(state.sps),
            "lfi": _nan0(state.lfi),
            "rank": _nan0(state.rank),
            "funding_z": _nan0(ff.get("funding_z")),
            "funding_current": _nan0(ff.get("funding_current")),
            "cum_24h": _nan0(ff.get("cum_24h")),
            "minutes_to_settlement": float(mins_to_set),
            "oi_z_24h": _nan0(oi.get("oi_z_24h")),
            "oi_z_1h": _nan0(oi.get("oi_z_1h")),
            "rv_pct": _nan0(vol.get("rv_pct")),
            "bb_width_pct": _nan0(vol.get("bb_width_pct")),
            "range_hours": _nan0(vol.get("range_hours")),
            "cvd_ratio_24h": _nan0(fl.get("cvd_ratio_24h")),
            "taker_proxy": _nan0(fl.get("taker_proxy")),
            "price_accel": _nan0(fl.get("price_accel")),
            "thin_pct": _nan0(ob.get("thin_pct")),
            "spread_bps": _nan0(ob.get("spread_bps")),
            "vacuum_dist_bid": _nan0(ob.get("vacuum_dist_bid")),
            "vacuum_dist_ask": _nan0(ob.get("vacuum_dist_ask")),
            "depth_bid_usdt": _nan0(ob.get("depth_bid_usdt")),
            "depth_ask_usdt": _nan0(ob.get("depth_ask_usdt")),
            "convexity": _nan0(ob.get("convexity")),
            # Interaction features
            "thin_x_funding": _nan0(ob.get("thin_pct")) * abs(_nan0(ff.get("funding_z"))),
            "thin_x_oi": _nan0(ob.get("thin_pct")) * abs(_nan0(oi.get("oi_z_24h"))),
            "compression_x_thin": _nan0(state.compression) * _nan0(ob.get("thin_pct")),
            "funding_x_oi": abs(_nan0(ff.get("funding_z"))) * abs(_nan0(oi.get("oi_z_24h"))),
            "bb_x_rv": _nan0(vol.get("bb_width_pct")) * _nan0(vol.get("rv_pct")),
            "vacuum_asymmetry": _safe_imbalance(
                _nan0(ob.get("vacuum_dist_ask")),
                _nan0(ob.get("vacuum_dist_bid")),
            ),
        }
        return feat


def _nan0(v: Any) -> float:
    if v is None:
        return 0.0
    try:
        f = float(v)
        return 0.0 if f != f else f  # NaN check
    except (TypeError, ValueError):
        return 0.0


def _safe_imbalance(bid: float, ask: float) -> float:
    s = bid + ask
    return (bid - ask) / s if s > 1e-6 else 0.0
