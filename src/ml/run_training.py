"""CLI for training the fragility model.

Usage:
    python -m src.ml.run_training [--db data/events.db] [--force]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the fragility prediction model")
    parser.add_argument(
        "--db", type=Path, default=None,
        help="Path to events database (default: data/events_vps.db)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Train even if existing model exists",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )

    from src.ml.train_model import train_model, MODEL_PATH

    if MODEL_PATH.exists() and not args.force:
        import pickle
        with open(MODEL_PATH, "rb") as f:
            blob = pickle.load(f)
        logger.info(
            "Existing model: AUC=%.4f, trained=%s, samples=%d",
            blob.get("metrics", {}).get("auc", 0),
            blob.get("trained_at", "?"),
            blob.get("train_samples", 0),
        )
        print("Model already exists. Use --force to retrain.")
        print(f"  AUC: {blob.get('metrics', {}).get('auc', 0):.4f}")
        print(f"  Threshold: {blob.get('metrics', {}).get('best_threshold', 0):.4f}")

    metrics = train_model(db_path=args.db)
    if metrics is None:
        print("Training failed — not enough data", file=sys.stderr)
        sys.exit(1)

    print(f"\nAUC: {metrics['auc']:.4f}")
    print(f"Best threshold: {metrics['best_threshold']:.4f}")
    print(f"Best F1: {metrics['best_f1']:.4f}")

    print("\nThreshold analysis:")
    print(f"  {'Thresh':<8} {'Prec':>8} {'Recall':>8} {'Alerts':>8} {'TP':>6}")
    for t, m in metrics["threshold_analysis"].items():
        print(f"  {t:<8} {100*m['precision']:>7.1f}% {100*m['recall']:>7.1f}% "
              f"{m['n_alerts']:>8} {m['tp']:>6}")

    print("\nTop-10 features:")
    for i, (feat, imp) in enumerate(list(metrics["feature_importance"].items())[:10], 1):
        print(f"  {i:2}. {feat:<25} {imp:.1f}")


if __name__ == "__main__":
    main()
