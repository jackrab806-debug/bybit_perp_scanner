"""Train the fragility prediction model.

Model: LightGBM ensemble (time-series cross-validation).
Optional: regression model for move magnitude prediction.
Output: data/fragility_model.pkl + data/model_metrics.json

Model protection: never overwrites a better existing model.
"""

from __future__ import annotations

import json
import logging
import pickle
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = _ROOT / "data" / "fragility_model.pkl"
METRICS_PATH = _ROOT / "data" / "model_metrics.json"


def train_model(db_path: str | Path | None = None) -> Optional[Dict[str, Any]]:
    """Full pipeline: extract -> CV train -> ensemble -> evaluate -> save."""

    from src.ml.extract_features import extract_training_data

    kwargs = {} if db_path is None else {"db_path": db_path}
    df, feature_cols = extract_training_data(**kwargs)

    if len(df) < 200:
        logger.error("Not enough data: %d rows (need 200+)", len(df))
        return None

    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values.astype(np.float32)

    logger.info("Total: %d samples, %d features, %.1f%% positive",
                len(X), len(feature_cols), 100 * y.mean())

    # ── Time-series cross-validation ──
    try:
        import lightgbm as lgb
        fold_models, fold_aucs = _train_cv(X, y, feature_cols)
        model_type = "lightgbm"
    except (ImportError, OSError):
        logger.info("LightGBM unavailable — using sklearn single-split")
        split = int(len(X) * 0.7)
        model = _train_sklearn(X[:split], y[:split], X[split:], y[split:])
        fold_models = [model]
        from sklearn.metrics import roc_auc_score
        fold_aucs = [roc_auc_score(y[split:], model.predict_proba(X[split:])[:, 1])]
        model_type = "sklearn"

    if not fold_models:
        logger.error("No valid fold models produced")
        return None

    cv_mean = float(np.mean(fold_aucs))
    cv_std = float(np.std(fold_aucs))
    logger.info("CV AUC: %.4f (+/- %.4f)  folds: %s",
                cv_mean, cv_std, [round(a, 4) for a in fold_aucs])

    # ── Final model: train on 85%, validate on last 15% ──
    final_split = int(len(X) * 0.85)
    X_tr, X_val = X[:final_split], X[final_split:]
    y_tr, y_val = y[:final_split], y[final_split:]

    logger.info("Final split: %d train (%.1f%% pos), %d val (%.1f%% pos)",
                len(X_tr), 100 * y_tr.mean(), len(X_val), 100 * y_val.mean())

    if model_type == "lightgbm":
        final_model = _train_lgb(X_tr, y_tr, X_val, y_val, feature_cols, seed=42)
    else:
        final_model = _train_sklearn(X_tr, y_tr, X_val, y_val)

    # ── Evaluate ──
    metrics = _evaluate(final_model, X_val, y_val, feature_cols, model_type)
    metrics["cv_auc_mean"] = round(cv_mean, 4)
    metrics["cv_auc_std"] = round(cv_std, 4)
    metrics["fold_aucs"] = [round(a, 4) for a in fold_aucs]
    metrics["n_folds"] = len(fold_models)
    metrics["label_threshold_pct"] = 3.0

    # ── Regression model (move magnitude) ──
    reg_model = None
    if model_type == "lightgbm" and "abs_max_move" in df.columns:
        try:
            y_reg = df["abs_max_move"].values.astype(np.float32)
            reg_model = _train_regression(
                X[:final_split], y_reg[:final_split],
                X[final_split:], y_reg[final_split:],
                feature_cols,
            )
            # Evaluate regression
            reg_preds = reg_model.predict(X_val)
            from sklearn.metrics import mean_absolute_error
            reg_mae = mean_absolute_error(y_reg[final_split:], reg_preds)
            metrics["regression_mae"] = round(reg_mae, 4)
            logger.info("Regression MAE: %.4f%%", reg_mae)
        except Exception as exc:
            logger.warning("Regression model failed: %s", exc)

    # ── Model protection: never downgrade ──
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_auc = metrics["auc"]
    existing_auc = 0.0

    if MODEL_PATH.exists():
        try:
            with open(MODEL_PATH, "rb") as f:
                old_blob = pickle.load(f)
            existing_auc = old_blob.get("metrics", {}).get("auc", 0)
        except Exception:
            pass

    if new_auc < existing_auc - 0.005:
        logger.warning(
            "Retrain SKIPPED: new AUC %.4f < existing %.4f (margin 0.005)",
            new_auc, existing_auc,
        )
        metrics["retrain_skipped"] = True
        metrics["existing_auc"] = existing_auc
        return metrics

    # Backup existing model
    if MODEL_PATH.exists():
        backup = MODEL_PATH.with_suffix(".backup.pkl")
        shutil.copy2(MODEL_PATH, backup)
        logger.info("Backed up existing model -> %s", backup)

    # ── Save ──
    blob = {
        "model": final_model,
        "fold_models": fold_models,
        "reg_model": reg_model,
        "model_type": model_type,
        "feature_cols": feature_cols,
        "metrics": metrics,
        "trained_at": datetime.utcnow().isoformat(),
        "train_samples": int(len(X_tr)),
        "val_samples": int(len(X_val)),
        "total_samples": int(len(X)),
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(blob, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Model saved -> %s  (AUC=%.4f, CV=%.4f, prev=%.4f)",
                MODEL_PATH, new_auc, cv_mean, existing_auc)
    return metrics


# ── Time-series CV ────────────────────────────────────────────────────────────


def _train_cv(
    X: np.ndarray, y: np.ndarray, feature_cols: List[str],
) -> tuple:
    """Time-series cross-validation. Returns (fold_models, fold_aucs)."""
    from sklearn.metrics import roc_auc_score

    n = len(X)
    fold_size = n // 5

    models = []
    aucs = []

    # 3 validation folds: train on past, validate on future
    for fold in range(3):
        train_end = (fold + 2) * fold_size
        val_start = train_end
        val_end = min((fold + 3) * fold_size, n)

        X_tr, y_tr = X[:train_end], y[:train_end]
        X_val, y_val = X[val_start:val_end], y[val_start:val_end]

        if len(X_val) < 100 or y_val.sum() < 5:
            logger.warning("Fold %d: not enough data (val=%d, pos=%d), skipping",
                          fold, len(X_val), int(y_val.sum()))
            continue

        model = _train_lgb(X_tr, y_tr, X_val, y_val, feature_cols, seed=fold * 42)

        y_pred = model.predict(X_val)
        auc = roc_auc_score(y_val, y_pred)
        aucs.append(auc)
        models.append(model)

        logger.info("Fold %d: AUC=%.4f (%d train, %d val, %.1f%% pos)",
                    fold, auc, len(X_tr), len(X_val), 100 * y_val.mean())

    return models, aucs


# ── LightGBM ──────────────────────────────────────────────────────────────────


def _train_lgb(X_tr, y_tr, X_val, y_val, feature_cols, seed=42):
    import lightgbm as lgb

    n_pos = float(y_tr.sum())
    n_neg = len(y_tr) - n_pos
    scale = n_neg / max(n_pos, 1)

    params = {
        "objective": "binary",
        "metric": "auc",
        "scale_pos_weight": scale,
        "learning_rate": 0.03,
        "num_leaves": 20,
        "max_depth": 5,
        "min_child_samples": 30,
        "subsample": 0.7,
        "subsample_freq": 1,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
        "min_gain_to_split": 0.02,
        "path_smooth": 1.0,
        "verbose": -1,
        "seed": seed,
    }

    ds_tr = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_cols)
    ds_val = lgb.Dataset(X_val, label=y_val, feature_name=feature_cols)

    model = lgb.train(
        params, ds_tr,
        valid_sets=[ds_val],
        num_boost_round=500,
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],
    )

    imp = dict(zip(feature_cols, model.feature_importance(importance_type="gain")))
    imp = dict(sorted(imp.items(), key=lambda kv: kv[1], reverse=True))
    logger.info("Feature importance (top-7): %s",
                {k: round(v, 1) for k, v in list(imp.items())[:7]})
    return model


# ── Regression ────────────────────────────────────────────────────────────────


def _train_regression(X_tr, y_tr, X_val, y_val, feature_cols):
    """Train regression model to predict move magnitude."""
    import lightgbm as lgb

    params = {
        "objective": "regression",
        "metric": "mae",
        "learning_rate": 0.03,
        "num_leaves": 20,
        "max_depth": 5,
        "min_child_samples": 30,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "verbose": -1,
    }

    ds_tr = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_cols)
    ds_val = lgb.Dataset(X_val, label=y_val, feature_name=feature_cols)

    model = lgb.train(
        params, ds_tr,
        valid_sets=[ds_val],
        num_boost_round=300,
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )
    return model


# ── Sklearn fallback ──────────────────────────────────────────────────────────


def _train_sklearn(X_tr, y_tr, X_val, y_val):
    from sklearn.ensemble import GradientBoostingClassifier

    n_pos = float(y_tr.sum())
    w = np.where(y_tr == 1, (len(y_tr) - n_pos) / max(n_pos, 1), 1.0)

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
        if m.get("cv_auc_mean"):
            print(f"CV AUC: {m['cv_auc_mean']} (+/- {m.get('cv_auc_std', 0)})")
        print(f"Model saved to {MODEL_PATH}")
