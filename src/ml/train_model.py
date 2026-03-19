"""Train the fragility prediction model.

Model: LightGBM (falls back to sklearn GradientBoosting).
Split: walk-forward 70/30 to prevent look-ahead bias.
Output: data/fragility_model.pkl + data/model_metrics.json
"""

from __future__ import annotations

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = _ROOT / "data" / "fragility_model.pkl"
METRICS_PATH = _ROOT / "data" / "model_metrics.json"


def train_model(db_path: str | Path | None = None) -> Optional[Dict[str, Any]]:
    """Full pipeline: extract → split → train → evaluate → save."""

    from src.ml.extract_features import extract_training_data

    kwargs = {} if db_path is None else {"db_path": db_path}
    df, feature_cols = extract_training_data(**kwargs)

    if len(df) < 100:
        logger.error("Not enough data: %d rows", len(df))
        return None

    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values.astype(np.float32)

    # Walk-forward split
    split = int(len(df) * 0.7)
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]

    logger.info("Train: %d (%.1f%% hit)  Val: %d (%.1f%% hit)",
                len(X_tr), 100 * y_tr.mean(), len(X_val), 100 * y_val.mean())

    # Train
    try:
        import lightgbm as lgb
        model = _train_lgb(X_tr, y_tr, X_val, y_val, feature_cols)
        model_type = "lightgbm"
    except (ImportError, OSError):
        logger.info("LightGBM unavailable — using sklearn")
        model = _train_sklearn(X_tr, y_tr, X_val, y_val)
        model_type = "sklearn"

    # Evaluate
    metrics = _evaluate(model, X_val, y_val, feature_cols, model_type)

    # Save
    blob = {
        "model": model,
        "model_type": model_type,
        "feature_cols": feature_cols,
        "metrics": metrics,
        "trained_at": datetime.utcnow().isoformat(),
        "train_samples": int(len(X_tr)),
        "val_samples": int(len(X_val)),
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(blob, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Model saved → %s  (AUC=%.4f)", MODEL_PATH, metrics["auc"])
    return metrics


# ── LightGBM ──────────────────────────────────────────────────────────────────


def _train_lgb(X_tr, y_tr, X_val, y_val, feature_cols):
    import lightgbm as lgb

    n_pos = float(y_tr.sum())
    scale = (len(y_tr) - n_pos) / max(n_pos, 1)

    params = {
        "objective": "binary",
        "metric": "auc",
        "is_unbalance": True,
        "learning_rate": 0.01,
        "num_leaves": 12,
        "max_depth": 4,
        "min_child_samples": 25,
        "subsample": 0.7,
        "subsample_freq": 1,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.5,
        "reg_lambda": 3.0,
        "min_gain_to_split": 0.05,
        "path_smooth": 1.0,
        "verbose": -1,
    }

    ds_tr = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_cols)
    ds_val = lgb.Dataset(X_val, label=y_val, feature_name=feature_cols)

    model = lgb.train(
        params, ds_tr,
        valid_sets=[ds_val],
        num_boost_round=500,
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)],
    )

    imp = dict(zip(feature_cols, model.feature_importance(importance_type="gain")))
    imp = dict(sorted(imp.items(), key=lambda kv: kv[1], reverse=True))
    logger.info("Feature importance (top-7): %s",
                {k: round(v, 1) for k, v in list(imp.items())[:7]})
    return model


# ── Sklearn fallback ──────────────────────────────────────────────────────────


def _train_sklearn(X_tr, y_tr, X_val, y_val):
    from sklearn.ensemble import GradientBoostingClassifier

    n_pos = float(y_tr.sum())
    w = np.where(y_tr == 1, (len(y_tr) - n_pos) / n_pos, 1.0)

    model = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        min_samples_leaf=20, subsample=0.8, random_state=42,
    )
    model.fit(X_tr, y_tr, sample_weight=w)
    return model


# ── Evaluation ────────────────────────────────────────────────────────────────


def _evaluate(model, X_val, y_val, feature_cols, model_type) -> Dict[str, Any]:
    from sklearn.metrics import roc_auc_score, precision_recall_curve

    y_prob = (model.predict(X_val) if model_type == "lightgbm"
              else model.predict_proba(X_val)[:, 1])

    auc = roc_auc_score(y_val, y_prob)

    prec, rec, thr = precision_recall_curve(y_val, y_prob)
    f1 = 2 * prec * rec / (prec + rec + 1e-10)
    best_i = int(np.argmax(f1))
    best_thr = float(thr[best_i]) if best_i < len(thr) else 0.5

    # Per-threshold analysis
    ta = {}
    for t in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50):
        yp = (y_prob >= t).astype(int)
        tp = int(((yp == 1) & (y_val == 1)).sum())
        fp = int(((yp == 1) & (y_val == 0)).sum())
        fn = int(((yp == 0) & (y_val == 1)).sum())
        p = tp / (tp + fp) if (tp + fp) else 0
        r = tp / (tp + fn) if (tp + fn) else 0
        ta[str(t)] = {
            "precision": round(p, 4), "recall": round(r, 4),
            "n_alerts": int(yp.sum()), "tp": tp, "fp": fp,
        }

    # Feature importance
    if model_type == "lightgbm":
        imp = dict(zip(feature_cols, model.feature_importance(importance_type="gain")))
    else:
        imp = dict(zip(feature_cols, model.feature_importances_))
    imp = {k: round(float(v), 2) for k, v in
           sorted(imp.items(), key=lambda kv: kv[1], reverse=True)}

    metrics = {
        "auc": round(auc, 4),
        "best_threshold": round(best_thr, 4),
        "best_f1": round(float(f1[best_i]), 4),
        "val_samples": int(len(y_val)),
        "val_hits": int(y_val.sum()),
        "threshold_analysis": ta,
        "feature_importance": imp,
    }

    logger.info("AUC=%.4f  best_thr=%.4f  best_F1=%.4f", auc, best_thr, f1[best_i])
    logger.info("Threshold analysis:")
    logger.info("  %-8s %8s %8s %8s %6s", "Thresh", "Prec", "Recall", "Alerts", "TP")
    for t, m in ta.items():
        logger.info("  %-8s %7.1f%% %7.1f%% %8d %6d",
                     t, 100 * m["precision"], 100 * m["recall"], m["n_alerts"], m["tp"])

    return metrics


if __name__ == "__main__":
    import argparse as _ap
    _p = _ap.ArgumentParser()
    _p.add_argument("--db", type=str, default=None)
    _args = _p.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-8s %(message)s")
    m = train_model(db_path=_args.db)
    if m:
        print(f"\nAUC: {m['auc']}")
        print(f"Model saved to {MODEL_PATH}")
