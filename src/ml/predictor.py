"""Live Fragility Predictor — scores coins using the trained model.

Loads data/fragility_model.pkl and provides predict(features) -> probability.
Supports ensemble prediction (average of fold models).
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
    """Score coins using the trained ML model (ensemble)."""

    def __init__(self) -> None:
        self.model: Any = None
        self.fold_models: List[Any] = []
        self.reg_model: Any = None
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
            self.fold_models = blob.get("fold_models", [])
            self.reg_model = blob.get("reg_model")
            self.model_type = blob["model_type"]
            self.feature_cols = blob["feature_cols"]
            self.metrics = blob.get("metrics", {})
            self.threshold = self.metrics.get("best_threshold", 0.20)
            self._loaded = True
            logger.info(
                "FragilityPredictor loaded (%s, thr=%.3f, AUC=%.4f, CV=%.4f, %d folds)",
                self.model_type, self.threshold,
                self.metrics.get("auc", 0),
                self.metrics.get("cv_auc_mean", 0),
                len(self.fold_models),
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
        """Return P(>3% move in 4h). Ensemble average if fold models exist."""
        if not self._loaded:
            return None
        try:
            X = np.array(
                [[features.get(c, 0) for c in self.feature_cols]],
                dtype=np.float32,
            )

            if self.fold_models and self.model_type == "lightgbm":
                # Ensemble: average all fold models + final model
                preds = [m.predict(X)[0] for m in self.fold_models]
                preds.append(self.model.predict(X)[0])
                return float(np.mean(preds))

            if self.model_type == "lightgbm":
                return float(self.model.predict(X)[0])
            return float(self.model.predict_proba(X)[0][1])
        except Exception as exc:
            logger.debug("Predict error: %s", exc)
            return None

    def predict_move(self, features: Dict[str, float]) -> Optional[float]:
        """Return predicted move magnitude (%) from regression model."""
        if not self._loaded or self.reg_model is None:
            return None
        try:
            X = np.array(
                [[features.get(c, 0) for c in self.feature_cols]],
                dtype=np.float32,
            )
            return float(self.reg_model.predict(X)[0])
        except Exception:
            return None

    def predict_batch(
        self, features_list: List[Dict[str, float]]
    ) -> List[float]:
        """Predict for multiple coins at once."""
        if not self._loaded or not features_list:
            return [0.0] * len(features_list)
        try:
            X = np.array(
                [[f.get(c, 0) for c in self.feature_cols] for f in features_list],
                dtype=np.float32,
            )

            if self.fold_models and self.model_type == "lightgbm":
                all_preds = np.stack([m.predict(X) for m in self.fold_models]
                                    + [self.model.predict(X)])
                return [float(p) for p in all_preds.mean(axis=0)]

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
        """Map live SymbolState -> feature dict matching training columns."""

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

        # Raw values
        funding_rate = _nan0(ff.get("funding_current"))
        funding_z = _nan0(ff.get("funding_z"))
        oi_change_1h = _nan0(oi.get("oi_pct_1h"))
        oi_change_4h = _nan0(oi.get("oi_pct_4h"))
        oi_z_24h = _nan0(oi.get("oi_z_24h"))
        thin_pct = _nan0(ob.get("thin_pct"))
        d_bid = _nan0(ob.get("depth_bid_usdt"))
        d_ask = _nan0(ob.get("depth_ask_usdt"))
        v_bid = _nan0(ob.get("vacuum_dist_bid"))
        v_ask = _nan0(ob.get("vacuum_dist_ask"))
        bb_width = _nan0(vol.get("bb_width_pct"))
        oi_cur = _nan0(oi.get("oi_current"))
        mid = _nan0(ob.get("mid_price"))
        oi_usd = oi_cur * mid if (oi_cur > 0 and mid > 0) else 0.0

        has_fs = 1 if "FUNDING_SQUEEZE_SETUP" in etypes else 0
        has_vb = 1 if "VACUUM_BREAK" in etypes else 0
        has_ve = 1 if "VOLUME_EXPLOSION" in etypes else 0
        has_ca = 1 if "CASCADE_ACTIVE" in etypes else 0
        has_oi = 1 if "OI_SURGE" in etypes else 0
        n_types = float(len(etypes - {""}))

        feat: Dict[str, float] = {
            # Raw features (matching DB columns)
            "funding_rate": funding_rate,
            "funding_z": funding_z,
            "oi_change_1h_pct": oi_change_1h,
            "oi_change_4h_pct": oi_change_4h,
            "oi_z_24h": oi_z_24h,
            "thin_pct": thin_pct,
            "depth_bid_usdt": d_bid,
            "depth_ask_usdt": d_ask,
            "vacuum_dist_bid": v_bid,
            "vacuum_dist_ask": v_ask,
            "spread_bps": _nan0(ob.get("spread_bps")),
            "imbalance": _nan0(ob.get("depth_band_imbalance")),
            "convexity": _nan0(ob.get("convexity")),
            "bb_width_pct": bb_width,
            "rv_pct": _nan0(vol.get("rv_pct")),
            "cvd_ratio_24h": _nan0(fl.get("cvd_ratio_24h")),
            "taker_proxy": _nan0(fl.get("taker_proxy")),
            "price_accel": _nan0(fl.get("price_accel")),
            "compression": _nan0(state.compression),
            "sps": _nan0(state.sps),
            "lfi": _nan0(state.lfi),
            "rank": _nan0(state.rank),
            "oi_to_depth_ratio": oi_usd / (d_ask + 1) if d_ask >= 0 else 0,
            "funding_x_oi": abs(funding_z) * abs(oi_z_24h),
            "vacuum_asymmetry": _safe_imbalance(v_ask, v_bid),
            "btc_change_1h": 0.0,  # filled by caller if available
            "btc_change_4h": 0.0,
            "hour_utc": float(now.hour),
            "mins_to_settlement": float(mins_to_set),
            "is_weekend": 1.0 if now.weekday() >= 5 else 0.0,
            "has_fs": float(has_fs),
            "has_vb": float(has_vb),
            "has_ve": float(has_ve),
            "has_ca": float(has_ca),
            "has_oi": float(has_oi),
            "num_event_types_2h": n_types,
            "max_event_score": max((e.get("score", 0) or 0 for e in evts), default=0),
            "oi_usd": oi_usd,
            "spread_z": _nan0(ob.get("spread_z")),
            "vol_ratio_5m": 0.0,  # populated at runtime if available

            # Computed interaction features (top 5 only)
            "abs_funding": abs(funding_rate),
            "pressure_ratio": abs(funding_rate) * oi_usd / (d_ask + 1),
            "settlement_urgency": 1.0 / (mins_to_set + 10),
            "event_intensity": (
                has_fs * 1.0 + has_vb * 0.5 + has_ve * 3.0
                + has_ca * 2.0 + has_oi * 1.5
            ),
            "multi_signal": 1.0 if n_types >= 2 else 0.0,
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


def _safe_imbalance(a: float, b: float) -> float:
    s = a + b
    return (a - b) / s if s > 1e-6 else 0.0
