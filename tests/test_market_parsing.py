"""Tests for market title parsing."""

from datetime import date, timedelta

import pytest

from src.polymarket.parsing import (
    parse_market_title,
    normalize_city,
    extract_temperature,
    extract_date,
    detect_comparison,
    is_valid_weather_market,
)


class TestNormalizeCity:
    """Tests for city normalization."""

    def test_london_lowercase(self):
        assert normalize_city("london") == "London"

    def test_london_in_sentence(self):
        assert normalize_city("Will London hit 9C?") == "London"

    def test_new_york_full(self):
        assert normalize_city("New York City temperature") == "New York City"

    def test_nyc_abbreviation(self):
        assert normalize_city("NYC weather tomorrow") == "New York City"

    def test_seoul(self):
        assert normalize_city("Seoul temperature forecast") == "Seoul"

    def test_unknown_city(self):
        assert normalize_city("Tokyo weather") is None


class TestExtractTemperature:
    """Tests for temperature extraction."""

    def test_celsius_with_symbol(self):
        assert extract_temperature("hit 9°C tomorrow") == 9.0

    def test_celsius_without_symbol(self):
        assert extract_temperature("reach 15C") == 15.0

    def test_celsius_with_space(self):
        assert extract_temperature("exceed 5 °C") == 5.0

    def test_decimal_temperature(self):
        assert extract_temperature("reach 10.5°C") == 10.5

    def test_no_temperature(self):
        assert extract_temperature("will it rain?") is None


class TestExtractDate:
    """Tests for date extraction."""

    def test_tomorrow(self):
        expected = date.today() + timedelta(days=1)
        assert extract_date("temperature tomorrow") == expected

    def test_today(self):
        assert extract_date("temperature today") == date.today()

    def test_explicit_date(self):
        result = extract_date("temperature on 01/28/2025")
        assert result == date(2025, 1, 28)

    def test_no_date(self):
        assert extract_date("temperature forecast") is None


class TestDetectComparison:
    """Tests for comparison detection."""

    def test_hit_implies_gte(self):
        assert detect_comparison("Will London hit 9C?") == ">="

    def test_reach_implies_gte(self):
        assert detect_comparison("Will NYC reach 15C?") == ">="

    def test_below_implies_lt(self):
        assert detect_comparison("Will it stay below 5C?") == "<"

    def test_under_implies_lt(self):
        assert detect_comparison("Temperature under 10C?") == "<"


class TestParseMarketTitle:
    """Tests for full market title parsing."""

    def test_london_tomorrow(self):
        parsed = parse_market_title("Will London hit 9°C tomorrow?")
        assert parsed.city == "London"
        assert parsed.threshold_celsius == 9.0
        assert parsed.target_date == date.today() + timedelta(days=1)
        assert parsed.comparison == ">="

    def test_nyc_market(self):
        parsed = parse_market_title(
            "Will New York City reach 15°C tomorrow?",
            "Resolves YES if daily max temp is 15C or above."
        )
        assert parsed.city == "New York City"
        assert parsed.threshold_celsius == 15.0

    def test_seoul_market(self):
        parsed = parse_market_title("Will Seoul temperature exceed 5°C tomorrow?")
        assert parsed.city == "Seoul"
        assert parsed.threshold_celsius == 5.0

    def test_below_market(self):
        parsed = parse_market_title("Will London stay below 0°C tomorrow?")
        assert parsed.city == "London"
        assert parsed.threshold_celsius == 0.0
        assert parsed.comparison == "<"


class TestIsValidWeatherMarket:
    """Tests for market validation."""

    def test_valid_market(self):
        parsed = parse_market_title("Will London hit 9°C tomorrow?")
        assert is_valid_weather_market(parsed) is True

    def test_missing_city(self):
        parsed = parse_market_title("Will temperature hit 9°C tomorrow?")
        assert is_valid_weather_market(parsed) is False

    def test_missing_threshold(self):
        parsed = parse_market_title("Will London be warm tomorrow?")
        assert is_valid_weather_market(parsed) is False

    def test_missing_date(self):
        parsed = parse_market_title("Will London hit 9°C?")
        assert is_valid_weather_market(parsed) is False
