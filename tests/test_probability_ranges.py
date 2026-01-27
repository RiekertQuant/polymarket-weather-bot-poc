"""Tests for range and comparison probability calculations."""

import pytest
from datetime import date, timedelta

from src.weather.probability import WeatherProbabilityEngine
from src.weather.open_meteo import MockOpenMeteoClient, DailyForecast
from src.polymarket.real_client import fahrenheit_to_celsius, parse_outcome_temp


def _make_engine(forecast_max: float, sigma: float = 2.0) -> tuple:
    """Create an engine with a single NYC forecast at the given max temp."""
    tomorrow = date.today() + timedelta(days=1)
    forecasts = {
        "New York City": [
            DailyForecast(
                date=tomorrow,
                temperature_max=forecast_max,
                temperature_min=forecast_max - 5.0,
                temperature_mean=forecast_max - 2.5,
            )
        ]
    }
    client = MockOpenMeteoClient(forecasts)
    engine = WeatherProbabilityEngine(weather_client=client, default_sigma=sigma)
    return engine, tomorrow


class TestRangeProbability:
    """Tests for P(lower <= X <= upper) range calculations."""

    def test_forecast_centered_in_range(self):
        """Forecast centered in range should give highest probability."""
        engine, day = _make_engine(forecast_max=0.0, sigma=2.0)
        result = engine.calculate_probability(
            city="New York City",
            threshold=-2.0,
            target_date=day,
            comparison="range",
            threshold_upper=2.0,
        )
        assert result is not None
        # Range [-2, 2] centered on forecast=0, sigma=2 -> ~68%
        assert 0.60 < result.p_raw < 0.75

    def test_forecast_below_range(self):
        """Forecast well below range should give low probability."""
        engine, day = _make_engine(forecast_max=-10.0, sigma=2.0)
        result = engine.calculate_probability(
            city="New York City",
            threshold=0.0,
            target_date=day,
            comparison="range",
            threshold_upper=2.0,
        )
        assert result is not None
        assert result.p_raw < 0.01

    def test_forecast_above_range(self):
        """Forecast well above range should give low probability."""
        engine, day = _make_engine(forecast_max=15.0, sigma=2.0)
        result = engine.calculate_probability(
            city="New York City",
            threshold=0.0,
            target_date=day,
            comparison="range",
            threshold_upper=2.0,
        )
        assert result is not None
        assert result.p_raw < 0.01

    def test_narrow_range_near_forecast(self):
        """Narrow range near forecast should give moderate probability."""
        engine, day = _make_engine(forecast_max=0.0, sigma=2.0)
        result = engine.calculate_probability(
            city="New York City",
            threshold=-0.5,
            target_date=day,
            comparison="range",
            threshold_upper=0.5,
        )
        assert result is not None
        # Narrow 1°C band centered on forecast -> modest probability
        assert 0.15 < result.p_raw < 0.25

    def test_wide_range_covering_forecast(self):
        """Wide range covering forecast should give near-1.0 probability."""
        engine, day = _make_engine(forecast_max=0.0, sigma=2.0)
        result = engine.calculate_probability(
            city="New York City",
            threshold=-10.0,
            target_date=day,
            comparison="range",
            threshold_upper=10.0,
        )
        assert result is not None
        assert result.p_raw > 0.99


class TestLessEqualProbability:
    """Tests for P(X <= threshold) calculations."""

    def test_forecast_well_below_threshold(self):
        """Forecast well below threshold -> high probability."""
        engine, day = _make_engine(forecast_max=0.0, sigma=2.0)
        result = engine.calculate_probability(
            city="New York City",
            threshold=None,
            target_date=day,
            comparison="<=",
            threshold_upper=10.0,
        )
        assert result is not None
        assert result.p_raw > 0.99

    def test_forecast_at_threshold(self):
        """Forecast at threshold -> ~50%."""
        engine, day = _make_engine(forecast_max=5.0, sigma=2.0)
        result = engine.calculate_probability(
            city="New York City",
            threshold=None,
            target_date=day,
            comparison="<=",
            threshold_upper=5.0,
        )
        assert result is not None
        assert 0.45 < result.p_raw < 0.55

    def test_forecast_well_above_threshold(self):
        """Forecast well above threshold -> low probability."""
        engine, day = _make_engine(forecast_max=15.0, sigma=2.0)
        result = engine.calculate_probability(
            city="New York City",
            threshold=None,
            target_date=day,
            comparison="<=",
            threshold_upper=5.0,
        )
        assert result is not None
        assert result.p_raw < 0.01


class TestRangeProbabilitySumsToOne:
    """Contiguous ranges should sum to ~1.0, like a real market."""

    def test_nyc_market_ranges_sum_to_one(self):
        """Simulate a real NYC market with contiguous ranges summing to ~1.0.

        Ranges: <=15°F, 16-17, 18-19, 20-21, 22-23, 24-25, >=26°F
        """
        # Forecast about -3.3°C (~26°F), sigma ~1.1°C (~2°F)
        forecast_celsius = fahrenheit_to_celsius(21)
        engine, day = _make_engine(
            forecast_max=forecast_celsius, sigma=1.1
        )

        outcomes = [
            "15°F or below",
            "16-17°F",
            "18-19°F",
            "20-21°F",
            "22-23°F",
            "24-25°F",
            "26°F or higher",
        ]

        total = 0.0
        for outcome_str in outcomes:
            ot = parse_outcome_temp(outcome_str)
            assert ot is not None, f"Failed to parse: {outcome_str}"

            result = engine.calculate_probability(
                city="New York City",
                threshold=ot.lower_celsius,
                target_date=day,
                comparison=ot.comparison,
                threshold_upper=ot.upper_celsius,
            )
            assert result is not None, f"No result for: {outcome_str}"
            total += result.p_raw

        assert total == pytest.approx(1.0, abs=0.01)
