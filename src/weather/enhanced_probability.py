"""Enhanced weather probability calculations with all improvements.

Improvements included:
1. Increased base sigma (2.5°C vs 2.0°C)
2. Bias correction (+0.5°C for cold bias)
3. Dynamic sigma based on forecast horizon
4. Ensemble forecasting (NWS + Open-Meteo)
5. Extreme temperature band adjustments
6. Weather regime detection (stable vs transitional)
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats

from src.weather.open_meteo import OpenMeteoClient, DailyForecast

logger = logging.getLogger(__name__)


@dataclass
class EnsembleForecast:
    """Ensemble forecast from multiple sources."""

    mean_temp: float  # Weighted mean temperature
    sources: dict[str, float]  # Temperature from each source
    spread: float  # Spread between sources (measure of uncertainty)
    confidence: str  # "high", "medium", "low" based on agreement


@dataclass
class EnhancedProbabilityResult:
    """Enhanced result with additional metadata."""

    p_raw: float
    p_calibrated: float
    forecast_temp: float  # Final (bias-corrected, ensemble) temperature
    threshold: float
    sigma: float  # Dynamic sigma used
    comparison: str
    threshold_upper: Optional[float] = None

    # Enhanced metadata
    bias_correction: float = 0.0  # Bias correction applied
    days_ahead: int = 0  # Forecast horizon
    ensemble: Optional[EnsembleForecast] = None
    weather_regime: str = "unknown"  # "stable", "transitional", "unknown"
    band_type: str = "normal"  # "extreme_cold", "extreme_warm", "normal"


class EnhancedProbabilityEngine:
    """Enhanced probability engine with all improvements."""

    # Bias correction in Celsius (positive = forecasts run cold)
    DEFAULT_BIAS_CORRECTION = 0.5  # ~0.9°F, based on observed cold bias

    # Base sigma and horizon scaling
    BASE_SIGMA = 2.5  # Increased from 2.0°C
    SIGMA_PER_DAY = 0.3  # Additional uncertainty per day ahead

    # Extreme temperature thresholds (Celsius)
    EXTREME_COLD_C = -10.0  # ~14°F
    EXTREME_WARM_C = 10.0  # ~50°F (winter context)

    def __init__(
        self,
        primary_client=None,
        secondary_client=None,
        enable_ensemble: bool = True,
        enable_bias_correction: bool = True,
        enable_dynamic_sigma: bool = True,
        enable_regime_detection: bool = True,
        calibrator=None,
        bias_models_path: Optional[Path] = None,
    ):
        """Initialize enhanced probability engine.

        Args:
            primary_client: Primary weather client (NWS preferred).
            secondary_client: Secondary weather client (Open-Meteo).
            enable_ensemble: Whether to use ensemble forecasting.
            enable_bias_correction: Whether to apply bias correction.
            enable_dynamic_sigma: Whether to scale sigma by forecast horizon.
            enable_regime_detection: Whether to detect weather regimes.
            calibrator: Optional ML calibrator.
            bias_models_path: Path to bias correction models JSON.
        """
        self.primary_client = primary_client
        self.secondary_client = secondary_client or OpenMeteoClient()
        self.enable_ensemble = enable_ensemble
        self.enable_bias_correction = enable_bias_correction
        self.enable_dynamic_sigma = enable_dynamic_sigma
        self.enable_regime_detection = enable_regime_detection
        self.calibrator = calibrator

        # Load bias models if available
        self.bias_correction = self.DEFAULT_BIAS_CORRECTION
        if bias_models_path and bias_models_path.exists():
            self._load_bias_models(bias_models_path)

        # Per-city overrides
        self._city_sigma: dict[str, float] = {}
        self._city_bias: dict[str, float] = {}

    def _load_bias_models(self, path: Path) -> None:
        """Load bias correction models from file."""
        try:
            with open(path) as f:
                models = json.load(f)
            # Convert from Fahrenheit to Celsius
            bias_f = models.get("additive_bias", 0.0)
            self.bias_correction = bias_f * 5 / 9  # F to C
            logger.info(f"Loaded bias correction: {bias_f:.1f}°F ({self.bias_correction:.2f}°C)")
        except Exception as e:
            logger.warning(f"Failed to load bias models: {e}")

    def get_sigma(self, city: str, days_ahead: int = 1) -> float:
        """Get dynamic sigma based on city and forecast horizon.

        Args:
            city: City name.
            days_ahead: Days ahead for the forecast.

        Returns:
            Standard deviation for forecast uncertainty.
        """
        base = self._city_sigma.get(city, self.BASE_SIGMA)

        if self.enable_dynamic_sigma:
            # Uncertainty increases with forecast horizon
            # Day 1: base, Day 2: base + 0.3, Day 3: base + 0.6, etc.
            return base + (days_ahead - 1) * self.SIGMA_PER_DAY
        return base

    def get_bias_correction(self, city: str) -> float:
        """Get bias correction for a city.

        Args:
            city: City name.

        Returns:
            Bias correction in Celsius (positive = add to forecast).
        """
        if not self.enable_bias_correction:
            return 0.0
        return self._city_bias.get(city, self.bias_correction)

    def _get_ensemble_forecast(
        self,
        city: str,
        target_date: date,
    ) -> Optional[EnsembleForecast]:
        """Get ensemble forecast from multiple sources.

        Args:
            city: City name.
            target_date: Target date.

        Returns:
            EnsembleForecast with combined prediction.
        """
        sources = {}

        # Get primary forecast
        if self.primary_client:
            try:
                forecast = self.primary_client.get_forecast_for_date(city, target_date)
                if forecast:
                    sources["primary"] = forecast.temperature_max
            except Exception as e:
                logger.debug(f"Primary client failed: {e}")

        # Get secondary forecast
        if self.secondary_client:
            try:
                forecast = self.secondary_client.get_forecast_for_date(city, target_date)
                if forecast:
                    sources["secondary"] = forecast.temperature_max
            except Exception as e:
                logger.debug(f"Secondary client failed: {e}")

        if not sources:
            return None

        # Calculate ensemble statistics
        temps = list(sources.values())
        mean_temp = np.mean(temps)
        spread = max(temps) - min(temps) if len(temps) > 1 else 0.0

        # Determine confidence based on spread
        if spread < 1.0:  # Sources agree within 1°C
            confidence = "high"
        elif spread < 2.5:  # Sources agree within 2.5°C
            confidence = "medium"
        else:
            confidence = "low"

        return EnsembleForecast(
            mean_temp=float(mean_temp),
            sources=sources,
            spread=float(spread),
            confidence=confidence,
        )

    def _detect_weather_regime(
        self,
        city: str,
        target_date: date,
        forecast_temp: float,
    ) -> str:
        """Detect weather regime (stable vs transitional).

        Transitional weather (fronts, rapid changes) has higher uncertainty.

        Args:
            city: City name.
            target_date: Target date.
            forecast_temp: Current forecast temperature.

        Returns:
            "stable", "transitional", or "unknown"
        """
        if not self.enable_regime_detection:
            return "unknown"

        try:
            # Get multi-day forecast to detect transitions
            if self.primary_client:
                forecasts = self.primary_client.get_forecast(city, days=5)
            else:
                forecasts = self.secondary_client.get_forecast(city, days=5)

            if len(forecasts) < 3:
                return "unknown"

            # Find the target date in forecasts
            temps = []
            target_idx = -1
            for i, f in enumerate(forecasts):
                temps.append(f.temperature_max)
                if f.date == target_date:
                    target_idx = i

            if target_idx < 0 or len(temps) < 3:
                return "unknown"

            # Calculate day-to-day temperature changes
            changes = [abs(temps[i+1] - temps[i]) for i in range(len(temps)-1)]
            max_change = max(changes) if changes else 0

            # Large day-to-day swings indicate transitional weather
            if max_change > 5.0:  # >5°C swing (~9°F)
                return "transitional"
            elif max_change < 2.0:  # <2°C swing
                return "stable"
            else:
                return "moderate"

        except Exception as e:
            logger.debug(f"Regime detection failed: {e}")
            return "unknown"

    def _get_band_type(self, threshold: float, threshold_upper: Optional[float]) -> str:
        """Classify temperature band type.

        Args:
            threshold: Lower threshold (Celsius).
            threshold_upper: Upper threshold (Celsius).

        Returns:
            "extreme_cold", "extreme_warm", or "normal"
        """
        # Use the midpoint for range markets
        if threshold_upper is not None:
            midpoint = (threshold + threshold_upper) / 2
        else:
            midpoint = threshold

        if midpoint < self.EXTREME_COLD_C:
            return "extreme_cold"
        elif midpoint > self.EXTREME_WARM_C:
            return "extreme_warm"
        else:
            return "normal"

    def calculate_probability(
        self,
        city: str,
        threshold: float,
        target_date: date,
        comparison: str = ">=",
        forecast: Optional[DailyForecast] = None,
        threshold_upper: Optional[float] = None,
    ) -> Optional[EnhancedProbabilityResult]:
        """Calculate probability with all enhancements.

        Args:
            city: City name.
            threshold: Temperature threshold in Celsius.
            target_date: Target date for the forecast.
            comparison: ">=" for above, "<=" for below, "range" for between.
            forecast: Optional pre-fetched forecast.
            threshold_upper: Upper bound in Celsius (for "range").

        Returns:
            EnhancedProbabilityResult with all metadata.
        """
        # Calculate days ahead
        days_ahead = (target_date - date.today()).days
        if days_ahead < 0:
            days_ahead = 0

        # Get forecast
        ensemble = None
        if self.enable_ensemble and forecast is None:
            ensemble = self._get_ensemble_forecast(city, target_date)
            if ensemble:
                forecast_temp = ensemble.mean_temp
            elif self.primary_client:
                f = self.primary_client.get_forecast_for_date(city, target_date)
                forecast_temp = f.temperature_max if f else None
            else:
                f = self.secondary_client.get_forecast_for_date(city, target_date)
                forecast_temp = f.temperature_max if f else None
        elif forecast is not None:
            forecast_temp = forecast.temperature_max
        elif self.primary_client:
            f = self.primary_client.get_forecast_for_date(city, target_date)
            forecast_temp = f.temperature_max if f else None
        else:
            f = self.secondary_client.get_forecast_for_date(city, target_date)
            forecast_temp = f.temperature_max if f else None

        if forecast_temp is None:
            logger.warning(f"No forecast available for {city} on {target_date}")
            return None

        # Apply bias correction
        bias_correction = self.get_bias_correction(city)
        corrected_temp = forecast_temp + bias_correction

        # Get dynamic sigma
        sigma = self.get_sigma(city, days_ahead)

        # Adjust sigma for weather regime
        regime = self._detect_weather_regime(city, target_date, corrected_temp)
        if regime == "transitional":
            sigma *= 1.3  # 30% more uncertainty for transitional weather
        elif regime == "stable":
            sigma *= 0.9  # 10% less uncertainty for stable weather

        # Adjust sigma for ensemble disagreement
        if ensemble and ensemble.confidence == "low":
            sigma *= 1.2  # 20% more uncertainty when sources disagree

        # Get band type
        band_type = self._get_band_type(threshold, threshold_upper)

        # Additional uncertainty for extreme temperatures
        if band_type == "extreme_cold":
            sigma *= 1.15  # 15% more uncertainty for extreme cold
        elif band_type == "extreme_warm":
            sigma *= 1.1  # 10% more uncertainty for extreme warm

        # Calculate probability using normal CDF
        if comparison == "range" and threshold_upper is not None:
            z_lower = (threshold - corrected_temp) / sigma
            z_upper = (threshold_upper - corrected_temp) / sigma
            p_raw = stats.norm.cdf(z_upper) - stats.norm.cdf(z_lower)
        elif comparison == "<=":
            bound = threshold_upper if threshold_upper is not None else threshold
            z_score = (bound - corrected_temp) / sigma
            p_raw = stats.norm.cdf(z_score)
        elif comparison == "<":
            z_score = (threshold - corrected_temp) / sigma
            p_raw = stats.norm.cdf(z_score)
        else:  # comparison == ">="
            z_score = (threshold - corrected_temp) / sigma
            p_raw = 1.0 - stats.norm.cdf(z_score)

        p_raw = float(np.clip(p_raw, 0.0, 1.0))

        # Apply ML calibration if available
        p_calibrated = self._apply_calibration(
            p_raw, city, corrected_temp, threshold
        )

        return EnhancedProbabilityResult(
            p_raw=p_raw,
            p_calibrated=p_calibrated,
            forecast_temp=corrected_temp,
            threshold=threshold if threshold is not None else 0.0,
            sigma=sigma,
            comparison=comparison,
            threshold_upper=threshold_upper,
            bias_correction=bias_correction,
            days_ahead=days_ahead,
            ensemble=ensemble,
            weather_regime=regime,
            band_type=band_type,
        )

    def _apply_calibration(
        self,
        p_raw: float,
        city: str,
        forecast_temp: float,
        threshold: float,
    ) -> float:
        """Apply ML calibration if available."""
        if self.calibrator is None:
            return p_raw

        try:
            features = {
                "p_raw": p_raw,
                "city": city,
                "forecast_temp": forecast_temp,
                "threshold": threshold,
                "temp_diff": forecast_temp - threshold,
            }
            return self.calibrator.predict(features)
        except Exception as e:
            logger.warning(f"Calibration failed: {e}")
            return p_raw
