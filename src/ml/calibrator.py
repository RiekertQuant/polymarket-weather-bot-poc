"""Probability calibration models."""

import logging
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


class Calibrator(ABC):
    """Abstract base class for probability calibrators."""

    @abstractmethod
    def fit(self, p_raw: np.ndarray, y_true: np.ndarray) -> None:
        """Fit calibrator on raw probabilities and outcomes.

        Args:
            p_raw: Raw probability predictions.
            y_true: Binary outcomes (0 or 1).
        """
        pass

    @abstractmethod
    def predict(self, features: Union[float, np.ndarray, dict]) -> float:
        """Predict calibrated probability.

        Args:
            features: Raw probability or feature dict.

        Returns:
            Calibrated probability.
        """
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        """Save model to file."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "Calibrator":
        """Load model from file."""
        pass


class IsotonicCalibrator(Calibrator):
    """Isotonic regression calibrator.

    Simple and effective for probability calibration.
    Learns a monotonic mapping from raw to calibrated probabilities.
    """

    def __init__(self):
        """Initialize calibrator."""
        self._model: Optional[IsotonicRegression] = None
        self._is_fitted = False

    def fit(self, p_raw: np.ndarray, y_true: np.ndarray) -> None:
        """Fit isotonic regression.

        Args:
            p_raw: Raw probability predictions.
            y_true: Binary outcomes.
        """
        if len(p_raw) < 10:
            raise ValueError("Need at least 10 samples to fit calibrator")

        self._model = IsotonicRegression(out_of_bounds="clip")
        self._model.fit(p_raw, y_true)
        self._is_fitted = True
        logger.info(f"Fitted isotonic calibrator on {len(p_raw)} samples")

    def predict(self, features: Union[float, np.ndarray, dict]) -> float:
        """Predict calibrated probability.

        Args:
            features: Raw probability (float), array, or feature dict with 'p_raw'.

        Returns:
            Calibrated probability.
        """
        if not self._is_fitted or self._model is None:
            # Fall back to raw probability
            if isinstance(features, dict):
                return features.get("p_raw", 0.5)
            return float(features) if not isinstance(features, np.ndarray) else float(features[0])

        # Extract p_raw from features
        if isinstance(features, dict):
            p_raw = features.get("p_raw", 0.5)
        elif isinstance(features, np.ndarray):
            p_raw = features[0] if len(features) > 0 else 0.5
        else:
            p_raw = float(features)

        # Predict calibrated probability
        result = self._model.predict([p_raw])[0]
        return float(np.clip(result, 0.0, 1.0))

    def save(self, path: Path) -> None:
        """Save model to pickle file.

        Args:
            path: Output file path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self._model,
                "is_fitted": self._is_fitted,
            }, f)
        logger.info(f"Saved calibrator to {path}")

    @classmethod
    def load(cls, path: Path) -> "IsotonicCalibrator":
        """Load model from pickle file.

        Args:
            path: Model file path.

        Returns:
            Loaded calibrator.
        """
        calibrator = cls()
        if path.exists():
            with open(path, "rb") as f:
                data = pickle.load(f)
                calibrator._model = data.get("model")
                calibrator._is_fitted = data.get("is_fitted", False)
            logger.info(f"Loaded calibrator from {path}")
        else:
            logger.warning(f"Calibrator file not found: {path}")
        return calibrator


class LogisticCalibrator(Calibrator):
    """Platt scaling (logistic regression) calibrator.

    Alternative to isotonic regression, uses logistic regression
    to learn calibration mapping.
    """

    def __init__(self):
        """Initialize calibrator."""
        self._model: Optional[LogisticRegression] = None
        self._is_fitted = False

    def fit(self, p_raw: np.ndarray, y_true: np.ndarray) -> None:
        """Fit logistic regression calibrator.

        Args:
            p_raw: Raw probability predictions.
            y_true: Binary outcomes.
        """
        if len(p_raw) < 10:
            raise ValueError("Need at least 10 samples to fit calibrator")

        # Convert to logit space for features
        p_clipped = np.clip(p_raw, 0.001, 0.999)
        X = np.log(p_clipped / (1 - p_clipped)).reshape(-1, 1)

        self._model = LogisticRegression(solver="lbfgs")
        self._model.fit(X, y_true)
        self._is_fitted = True
        logger.info(f"Fitted logistic calibrator on {len(p_raw)} samples")

    def predict(self, features: Union[float, np.ndarray, dict]) -> float:
        """Predict calibrated probability.

        Args:
            features: Raw probability or feature dict.

        Returns:
            Calibrated probability.
        """
        if not self._is_fitted or self._model is None:
            if isinstance(features, dict):
                return features.get("p_raw", 0.5)
            return float(features) if not isinstance(features, np.ndarray) else float(features[0])

        if isinstance(features, dict):
            p_raw = features.get("p_raw", 0.5)
        elif isinstance(features, np.ndarray):
            p_raw = features[0] if len(features) > 0 else 0.5
        else:
            p_raw = float(features)

        p_clipped = np.clip(p_raw, 0.001, 0.999)
        X = np.log(p_clipped / (1 - p_clipped)).reshape(-1, 1)
        result = self._model.predict_proba(X)[0, 1]
        return float(np.clip(result, 0.0, 1.0))

    def save(self, path: Path) -> None:
        """Save model to pickle file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self._model,
                "is_fitted": self._is_fitted,
            }, f)

    @classmethod
    def load(cls, path: Path) -> "LogisticCalibrator":
        """Load model from pickle file."""
        calibrator = cls()
        if path.exists():
            with open(path, "rb") as f:
                data = pickle.load(f)
                calibrator._model = data.get("model")
                calibrator._is_fitted = data.get("is_fitted", False)
        return calibrator
