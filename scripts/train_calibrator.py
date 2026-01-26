#!/usr/bin/env python3
"""Script to train probability calibration model.

This script trains an isotonic regression calibrator on historical
prediction data if enough samples exist in the database.

Usage:
    python scripts/train_calibrator.py

The trained model is saved to models/calibrator.pkl and will be
automatically loaded by the trading bot on subsequent runs.
"""

import logging
import sys
from pathlib import Path

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.ml.calibrator import IsotonicCalibrator
from src.storage.db import Database, PredictionRepository

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

MIN_SAMPLES = 50  # Minimum samples needed for training


def main():
    """Train calibration model."""
    logger.info("=" * 50)
    logger.info("CALIBRATOR TRAINING SCRIPT")
    logger.info("=" * 50)

    # Initialize database
    db = Database(settings.db_path)
    try:
        db.connect()
    except Exception as e:
        logger.error(f"Could not connect to database: {e}")
        logger.info("Make sure to run the bot at least once to initialize the database.")
        return 1

    pred_repo = PredictionRepository(db)

    # Get training data
    logger.info("Loading prediction history...")
    data = pred_repo.get_training_data(min_samples=MIN_SAMPLES)

    if data is None:
        count = pred_repo.get_predictions_count()
        logger.warning(f"Not enough data for training.")
        logger.warning(f"Current predictions with outcomes: need {MIN_SAMPLES}, have fewer.")
        logger.warning(f"Total predictions in database: {count}")
        logger.info("")
        logger.info("To generate training data:")
        logger.info("1. Run the bot to make predictions")
        logger.info("2. After market settlement, update prediction outcomes")
        logger.info("3. Re-run this script when enough data exists")
        return 0

    p_raw, outcomes = data
    p_raw = np.array(p_raw)
    outcomes = np.array(outcomes)

    logger.info(f"Loaded {len(p_raw)} samples with outcomes")
    logger.info(f"  Positive outcomes: {outcomes.sum()} ({100*outcomes.mean():.1f}%)")
    logger.info(f"  Negative outcomes: {len(outcomes) - outcomes.sum()}")

    # Train calibrator
    logger.info("Training isotonic calibrator...")
    calibrator = IsotonicCalibrator()

    try:
        calibrator.fit(p_raw, outcomes)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1

    # Evaluate calibration
    logger.info("Evaluating calibration...")
    p_calibrated = np.array([calibrator.predict(p) for p in p_raw])

    # Brier score (lower is better)
    brier_raw = np.mean((p_raw - outcomes) ** 2)
    brier_cal = np.mean((p_calibrated - outcomes) ** 2)

    logger.info(f"  Brier score (raw):        {brier_raw:.4f}")
    logger.info(f"  Brier score (calibrated): {brier_cal:.4f}")
    logger.info(f"  Improvement: {100*(brier_raw - brier_cal)/brier_raw:.1f}%")

    # Save model
    model_path = Path("models/calibrator.pkl")
    logger.info(f"Saving model to {model_path}...")

    try:
        calibrator.save(model_path)
        logger.info("Model saved successfully!")
    except Exception as e:
        logger.error(f"Could not save model: {e}")
        return 1

    logger.info("")
    logger.info("Training complete. The calibrator will be used on next bot run.")
    logger.info("=" * 50)

    db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
