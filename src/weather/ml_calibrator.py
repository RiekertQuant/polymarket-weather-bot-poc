"""ML-based probability calibrator for weather forecasts."""

import logging
import pickle
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class CalibrationInput:
    """Input features for calibration model."""

    p_raw: float  # Raw probability from normal CDF
    forecast_temp: float  # Forecast temperature (Fahrenheit)
    threshold: float  # Threshold temperature (Fahrenheit)
    sigma: float = 3.6  # Uncertainty in Fahrenheit (~2°C)
    target_date: Optional[date] = None  # For seasonality features


class MLCalibrator:
    """ML-based probability calibrator.

    Uses a trained model to adjust raw probability estimates
    based on historical forecast accuracy patterns.
    """

    def __init__(self, model_path: Optional[Path] = None):
        """Initialize calibrator.

        Args:
            model_path: Path to trained model pickle file.
                       If None, looks in default location.
        """
        self.model = None
        self.model_path = model_path or Path("experiments/ml_calibration/calibrator.pkl")
        self._load_model()

    def _load_model(self) -> None:
        """Load the trained model from disk."""
        if not self.model_path.exists():
            logger.warning(f"Calibrator model not found at {self.model_path}")
            return

        try:
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
            logger.info(f"Loaded calibrator from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load calibrator: {e}")
            self.model = None

    def is_available(self) -> bool:
        """Check if calibrator model is loaded."""
        return self.model is not None

    def calibrate(self, input_data: CalibrationInput) -> float:
        """Calibrate a raw probability estimate.

        Args:
            input_data: CalibrationInput with features.

        Returns:
            Calibrated probability (0 to 1).
        """
        if not self.is_available():
            return input_data.p_raw

        try:
            features = self._build_features(input_data)
            proba = self.model.predict_proba([features])

            # Handle different model output shapes
            if proba.shape[1] > 1:
                return float(proba[0, 1])
            else:
                return float(proba[0, 0])

        except Exception as e:
            logger.warning(f"Calibration failed, using raw probability: {e}")
            return input_data.p_raw

    def _build_features(self, input_data: CalibrationInput) -> list:
        """Build feature vector for model.

        Args:
            input_data: CalibrationInput with raw values.

        Returns:
            List of features in expected order.
        """
        temp_diff = input_data.forecast_temp - input_data.threshold

        # Determine date features
        if input_data.target_date:
            month = input_data.target_date.month
            weekday = input_data.target_date.weekday()
        else:
            today = date.today()
            month = today.month
            weekday = today.weekday()

        return [
            input_data.p_raw,                           # 0: p_raw
            input_data.forecast_temp,                   # 1: forecast_temp
            input_data.threshold,                       # 2: threshold
            temp_diff,                                  # 3: temp_diff
            abs(temp_diff),                             # 4: abs_temp_diff
            1 if input_data.threshold < 20 else 0,     # 5: is_extreme_cold
            1 if input_data.threshold > 40 else 0,     # 6: is_warm
            month,                                      # 7: month
            weekday,                                    # 8: weekday
            input_data.sigma,                           # 9: sigma
        ]

    def predict(self, features: dict) -> float:
        """Predict calibrated probability from feature dict.

        This method is called by WeatherProbabilityEngine._apply_calibration().

        Args:
            features: Dict with keys: p_raw, city, forecast_temp, threshold, temp_diff

        Returns:
            Calibrated probability.
        """
        # Convert Celsius to Fahrenheit for the model
        forecast_temp_f = features["forecast_temp"] * 9/5 + 32
        threshold_f = features["threshold"] * 9/5 + 32

        input_data = CalibrationInput(
            p_raw=features["p_raw"],
            forecast_temp=forecast_temp_f,
            threshold=threshold_f,
        )

        return self.calibrate(input_data)
