"""Feature engineering for ML calibration."""

from dataclasses import dataclass
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class PredictionFeatures:
    """Features for a single prediction."""

    # Core features
    p_raw: float
    forecast_temp: float
    threshold: float
    temp_diff: float  # forecast_temp - threshold

    # City encoding (one-hot or label encoded)
    city: str

    # Temporal features
    target_date: date
    days_ahead: int

    # Optional metadata
    market_id: Optional[str] = None


class FeatureBuilder:
    """Builds feature vectors for ML models."""

    # Supported cities for one-hot encoding
    CITIES = ["New York City", "London", "Seoul"]

    def __init__(self):
        """Initialize feature builder."""
        self._city_to_idx = {city: i for i, city in enumerate(self.CITIES)}

    def build_features(self, pred: PredictionFeatures) -> np.ndarray:
        """Convert PredictionFeatures to numpy array.

        Args:
            pred: Prediction features.

        Returns:
            Numpy array of features.
        """
        # Numeric features
        numeric = [
            pred.p_raw,
            pred.forecast_temp,
            pred.threshold,
            pred.temp_diff,
            pred.days_ahead,
        ]

        # One-hot encode city
        city_onehot = [0.0] * len(self.CITIES)
        if pred.city in self._city_to_idx:
            city_onehot[self._city_to_idx[pred.city]] = 1.0

        return np.array(numeric + city_onehot)

    def build_dataframe(self, predictions: list[PredictionFeatures]) -> pd.DataFrame:
        """Convert list of predictions to DataFrame.

        Args:
            predictions: List of prediction features.

        Returns:
            DataFrame with all features.
        """
        records = []
        for pred in predictions:
            record = {
                "p_raw": pred.p_raw,
                "forecast_temp": pred.forecast_temp,
                "threshold": pred.threshold,
                "temp_diff": pred.temp_diff,
                "days_ahead": pred.days_ahead,
                "city": pred.city,
                "target_date": pred.target_date,
                "market_id": pred.market_id,
            }
            records.append(record)

        return pd.DataFrame(records)

    def get_feature_names(self) -> list[str]:
        """Get ordered list of feature names.

        Returns:
            List of feature names matching build_features output.
        """
        numeric_names = ["p_raw", "forecast_temp", "threshold", "temp_diff", "days_ahead"]
        city_names = [f"city_{city.replace(' ', '_')}" for city in self.CITIES]
        return numeric_names + city_names
