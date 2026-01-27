"""Weather probability calculations."""

import logging
from dataclasses import dataclass
from datetime import date
from typing import Optional

import numpy as np
from scipy import stats

from src.weather.open_meteo import OpenMeteoClient, DailyForecast

logger = logging.getLogger(__name__)


@dataclass
class ProbabilityResult:
    """Result of probability calculation."""

    p_raw: float  # Raw probability from normal distribution
    p_calibrated: float  # Calibrated probability (from ML model or same as raw)
    forecast_temp: float  # Forecasted temperature
    threshold: float  # Temperature threshold (lower bound for range)
    sigma: float  # Standard deviation used
    comparison: str  # ">=", "<=", or "range"
    threshold_upper: Optional[float] = None  # Upper bound for range markets


class WeatherProbabilityEngine:
    """Engine for calculating weather outcome probabilities."""

    def __init__(
        self,
        weather_client: Optional[OpenMeteoClient] = None,
        default_sigma: float = 2.0,
        calibrator=None,
    ):
        """Initialize probability engine.

        Args:
            weather_client: Weather API client.
            default_sigma: Default forecast uncertainty (standard deviation).
            calibrator: Optional calibration model for p_calibrated.
        """
        self.weather_client = weather_client or OpenMeteoClient()
        self.default_sigma = default_sigma
        self.calibrator = calibrator

        # Per-city sigma overrides (can be set from historical calibration)
        self._city_sigma: dict[str, float] = {}

    def get_sigma(self, city: str) -> float:
        """Get sigma for a city (calibrated or default).

        Args:
            city: City name.

        Returns:
            Standard deviation for forecast uncertainty.
        """
        return self._city_sigma.get(city, self.default_sigma)

    def set_city_sigma(self, city: str, sigma: float) -> None:
        """Set calibrated sigma for a city.

        Args:
            city: City name.
            sigma: Calibrated standard deviation.
        """
        self._city_sigma[city] = sigma

    def calculate_probability(
        self,
        city: str,
        threshold: float,
        target_date: date,
        comparison: str = ">=",
        forecast: Optional[DailyForecast] = None,
        threshold_upper: Optional[float] = None,
    ) -> Optional[ProbabilityResult]:
        """Calculate probability of temperature outcome.

        Uses a normal distribution centered on the forecast with uncertainty sigma.

        Args:
            city: City name.
            threshold: Temperature threshold in Celsius (lower bound for range).
            target_date: Target date for the forecast.
            comparison: ">=" for above, "<=" for below, "range" for between.
            forecast: Optional pre-fetched forecast.
            threshold_upper: Upper bound in Celsius (required for "range").

        Returns:
            ProbabilityResult with raw and calibrated probabilities.
        """
        # Get forecast if not provided
        if forecast is None:
            forecast = self.weather_client.get_forecast_for_date(city, target_date)

        if forecast is None:
            logger.warning(f"No forecast available for {city} on {target_date}")
            return None

        # Use max temperature for "highest temperature" markets
        forecast_temp = forecast.temperature_max
        sigma = self.get_sigma(city)

        # Calculate probability using normal CDF
        if comparison == "range" and threshold_upper is not None:
            # P(lower <= X <= upper) = CDF(upper) - CDF(lower)
            z_lower = (threshold - forecast_temp) / sigma
            z_upper = (threshold_upper - forecast_temp) / sigma
            p_raw = stats.norm.cdf(z_upper) - stats.norm.cdf(z_lower)
        elif comparison == "<=":
            # P(X <= threshold_upper)
            bound = threshold_upper if threshold_upper is not None else threshold
            z_score = (bound - forecast_temp) / sigma
            p_raw = stats.norm.cdf(z_score)
        elif comparison == "<":
            # P(X < threshold)
            z_score = (threshold - forecast_temp) / sigma
            p_raw = stats.norm.cdf(z_score)
        else:  # comparison == ">="
            # P(X >= threshold)
            z_score = (threshold - forecast_temp) / sigma
            p_raw = 1.0 - stats.norm.cdf(z_score)

        # Ensure probability is in valid range
        p_raw = float(np.clip(p_raw, 0.0, 1.0))

        # Apply calibration if available
        p_calibrated = self._apply_calibration(p_raw, city, forecast_temp, threshold)

        return ProbabilityResult(
            p_raw=p_raw,
            p_calibrated=p_calibrated,
            forecast_temp=forecast_temp,
            threshold=threshold if threshold is not None else 0.0,
            sigma=sigma,
            comparison=comparison,
            threshold_upper=threshold_upper,
        )

    def _apply_calibration(
        self,
        p_raw: float,
        city: str,
        forecast_temp: float,
        threshold: float,
    ) -> float:
        """Apply calibration model if available.

        Args:
            p_raw: Raw probability.
            city: City name.
            forecast_temp: Forecasted temperature.
            threshold: Temperature threshold.

        Returns:
            Calibrated probability.
        """
        if self.calibrator is None:
            return p_raw

        try:
            # Build features for calibrator
            features = {
                "p_raw": p_raw,
                "city": city,
                "forecast_temp": forecast_temp,
                "threshold": threshold,
                "temp_diff": forecast_temp - threshold,
            }
            return self.calibrator.predict(features)
        except Exception as e:
            logger.warning(f"Calibration failed, using raw probability: {e}")
            return p_raw

    def calculate_from_forecast_value(
        self,
        forecast_temp: float,
        threshold: float,
        sigma: float,
        comparison: str = ">=",
    ) -> float:
        """Calculate probability directly from forecast value.

        Useful for testing and when forecast is already known.

        Args:
            forecast_temp: Forecasted max temperature.
            threshold: Temperature threshold.
            sigma: Forecast uncertainty.
            comparison: ">=" or "<".

        Returns:
            Probability value.
        """
        z_score = (threshold - forecast_temp) / sigma

        if comparison == ">=":
            p = 1.0 - stats.norm.cdf(z_score)
        else:
            p = stats.norm.cdf(z_score)

        return float(np.clip(p, 0.0, 1.0))
