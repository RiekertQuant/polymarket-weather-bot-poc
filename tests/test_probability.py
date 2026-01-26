"""Tests for weather probability calculations."""

import pytest
import numpy as np
from datetime import date, timedelta

from src.weather.probability import WeatherProbabilityEngine, ProbabilityResult
from src.weather.open_meteo import MockOpenMeteoClient, DailyForecast


class TestProbabilityCalculation:
    """Tests for probability engine."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create mock weather client with known forecasts
        tomorrow = date.today() + timedelta(days=1)
        self.mock_forecasts = {
            "London": [
                DailyForecast(
                    date=tomorrow,
                    temperature_max=12.0,  # Forecast: 12°C high
                    temperature_min=5.0,
                    temperature_mean=8.5,
                )
            ],
            "New York City": [
                DailyForecast(
                    date=tomorrow,
                    temperature_max=8.0,  # Forecast: 8°C high
                    temperature_min=2.0,
                    temperature_mean=5.0,
                )
            ],
            "Seoul": [
                DailyForecast(
                    date=tomorrow,
                    temperature_max=3.0,  # Forecast: 3°C high
                    temperature_min=-2.0,
                    temperature_mean=0.5,
                )
            ],
        }
        self.weather_client = MockOpenMeteoClient(self.mock_forecasts)
        self.engine = WeatherProbabilityEngine(
            weather_client=self.weather_client,
            default_sigma=2.0,
        )
        self.tomorrow = tomorrow

    def test_probability_returns_valid_range(self):
        """Probability should be between 0 and 1."""
        result = self.engine.calculate_probability(
            city="London",
            threshold=9.0,
            target_date=self.tomorrow,
        )
        assert result is not None
        assert 0.0 <= result.p_raw <= 1.0
        assert 0.0 <= result.p_calibrated <= 1.0

    def test_high_probability_when_threshold_below_forecast(self):
        """P should be high when threshold is well below forecast."""
        # London forecast: 12°C, threshold: 9°C (3°C below)
        result = self.engine.calculate_probability(
            city="London",
            threshold=9.0,
            target_date=self.tomorrow,
        )
        assert result is not None
        # With sigma=2 and temp 3° above threshold, P should be high
        assert result.p_raw > 0.90

    def test_low_probability_when_threshold_above_forecast(self):
        """P should be low when threshold is well above forecast."""
        # London forecast: 12°C, threshold: 18°C (6°C above)
        result = self.engine.calculate_probability(
            city="London",
            threshold=18.0,
            target_date=self.tomorrow,
        )
        assert result is not None
        # With sigma=2 and temp 6° below threshold, P should be very low
        assert result.p_raw < 0.01

    def test_probability_around_50_when_threshold_equals_forecast(self):
        """P should be ~50% when threshold equals forecast."""
        # London forecast: 12°C, threshold: 12°C
        result = self.engine.calculate_probability(
            city="London",
            threshold=12.0,
            target_date=self.tomorrow,
        )
        assert result is not None
        # Should be close to 50%
        assert 0.45 <= result.p_raw <= 0.55

    def test_below_comparison(self):
        """Test P(temp < threshold)."""
        # London forecast: 12°C, threshold: 15°C
        # P(temp < 15) should be high
        result = self.engine.calculate_probability(
            city="London",
            threshold=15.0,
            target_date=self.tomorrow,
            comparison="<",
        )
        assert result is not None
        assert result.p_raw > 0.90

    def test_different_cities_different_probs(self):
        """Different cities should have different probabilities."""
        # Same threshold, different forecasts
        london = self.engine.calculate_probability(
            city="London",
            threshold=10.0,
            target_date=self.tomorrow,
        )
        nyc = self.engine.calculate_probability(
            city="New York City",
            threshold=10.0,
            target_date=self.tomorrow,
        )
        seoul = self.engine.calculate_probability(
            city="Seoul",
            threshold=10.0,
            target_date=self.tomorrow,
        )

        assert london is not None and nyc is not None and seoul is not None
        # London (12°C forecast) > NYC (8°C) > Seoul (3°C)
        assert london.p_raw > nyc.p_raw > seoul.p_raw

    def test_sigma_affects_probability(self):
        """Higher sigma should make probability closer to 50%."""
        engine_low_sigma = WeatherProbabilityEngine(
            weather_client=self.weather_client,
            default_sigma=1.0,
        )
        engine_high_sigma = WeatherProbabilityEngine(
            weather_client=self.weather_client,
            default_sigma=4.0,
        )

        # Threshold 2° above forecast
        low_sigma = engine_low_sigma.calculate_probability(
            city="London", threshold=14.0, target_date=self.tomorrow
        )
        high_sigma = engine_high_sigma.calculate_probability(
            city="London", threshold=14.0, target_date=self.tomorrow
        )

        assert low_sigma is not None and high_sigma is not None
        # High sigma should be closer to 0.5
        assert abs(high_sigma.p_raw - 0.5) < abs(low_sigma.p_raw - 0.5)

    def test_city_sigma_override(self):
        """Per-city sigma should be used when set."""
        self.engine.set_city_sigma("London", 1.0)

        assert self.engine.get_sigma("London") == 1.0
        assert self.engine.get_sigma("Seoul") == 2.0  # Default


class TestDirectProbabilityCalculation:
    """Tests for direct probability calculation without weather API."""

    def setup_method(self):
        """Set up test fixtures."""
        self.engine = WeatherProbabilityEngine()

    def test_calculate_from_forecast_value(self):
        """Test direct calculation from forecast value."""
        # Forecast: 10°C, threshold: 8°C, sigma: 2
        p = self.engine.calculate_from_forecast_value(
            forecast_temp=10.0,
            threshold=8.0,
            sigma=2.0,
        )
        # 1 sigma above, should be ~84%
        assert 0.83 <= p <= 0.85

    def test_calculate_below_threshold(self):
        """Test P(temp < threshold)."""
        # Forecast: 10°C, threshold: 12°C
        # P(temp < 12) with sigma=2 -> 1 sigma above -> ~84%
        p = self.engine.calculate_from_forecast_value(
            forecast_temp=10.0,
            threshold=12.0,
            sigma=2.0,
            comparison="<",
        )
        assert 0.83 <= p <= 0.85

    def test_extreme_values_clamped(self):
        """Extreme z-scores should still give valid probabilities."""
        # Very high threshold (10 sigma above forecast)
        p_low = self.engine.calculate_from_forecast_value(
            forecast_temp=10.0,
            threshold=30.0,
            sigma=2.0,
        )
        assert p_low >= 0.0

        # Very low threshold (10 sigma below forecast)
        p_high = self.engine.calculate_from_forecast_value(
            forecast_temp=10.0,
            threshold=-10.0,
            sigma=2.0,
        )
        assert p_high <= 1.0
