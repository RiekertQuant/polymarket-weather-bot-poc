"""Tests for pure parsing functions in real_client.py."""

import pytest
from datetime import date

from src.polymarket.real_client import (
    fahrenheit_to_celsius,
    parse_temp_range_title,
    OutcomeTemp,
    parse_outcome_temp,
    parse_event_date,
)


class TestFahrenheitToCelsius:
    """Tests for fahrenheit_to_celsius conversion."""

    def test_freezing_point(self):
        assert fahrenheit_to_celsius(32) == pytest.approx(0.0)

    def test_boiling_point(self):
        assert fahrenheit_to_celsius(212) == pytest.approx(100.0)

    def test_negative_fahrenheit(self):
        assert fahrenheit_to_celsius(0) == pytest.approx(-17.78, abs=0.01)

    def test_identity_at_minus_40(self):
        assert fahrenheit_to_celsius(-40) == pytest.approx(-40.0)


class TestParseTempRangeTitle:
    """Tests for parse_temp_range_title city extraction."""

    def test_nyc_abbreviation(self):
        result = parse_temp_range_title("Highest temperature in NYC on January 27?")
        assert result == "New York City"

    def test_new_york_full_name(self):
        result = parse_temp_range_title("Highest temperature in new york on January 27?")
        assert result == "New York City"

    def test_unrelated_title(self):
        result = parse_temp_range_title("Will it rain in London tomorrow?")
        assert result is None

    def test_empty_string(self):
        result = parse_temp_range_title("")
        assert result is None


class TestParseOutcomeTemp:
    """Tests for parse_outcome_temp outcome string parsing."""

    def test_or_below(self):
        result = parse_outcome_temp("15°F or below")
        assert result is not None
        assert result.comparison == "<="
        assert result.lower_celsius is None
        # 15 + 0.5 = 15.5°F -> celsius
        assert result.upper_celsius == pytest.approx(fahrenheit_to_celsius(15.5))

    def test_or_lower_synonym(self):
        result = parse_outcome_temp("15°F or lower")
        assert result is not None
        assert result.comparison == "<="
        assert result.upper_celsius == pytest.approx(fahrenheit_to_celsius(15.5))

    def test_or_higher(self):
        result = parse_outcome_temp("26°F or higher")
        assert result is not None
        assert result.comparison == ">="
        # 26 - 0.5 = 25.5°F -> celsius
        assert result.lower_celsius == pytest.approx(fahrenheit_to_celsius(25.5))
        assert result.upper_celsius is None

    def test_or_above_synonym(self):
        result = parse_outcome_temp("26°F or above")
        assert result is not None
        assert result.comparison == ">="
        assert result.lower_celsius == pytest.approx(fahrenheit_to_celsius(25.5))

    def test_range_20_21(self):
        result = parse_outcome_temp("20-21°F")
        assert result is not None
        assert result.comparison == "range"
        # 20 - 0.5 = 19.5°F, 21 + 0.5 = 21.5°F
        assert result.lower_celsius == pytest.approx(fahrenheit_to_celsius(19.5))
        assert result.upper_celsius == pytest.approx(fahrenheit_to_celsius(21.5))

    def test_range_16_17(self):
        result = parse_outcome_temp("16-17°F")
        assert result is not None
        assert result.comparison == "range"
        assert result.lower_celsius == pytest.approx(fahrenheit_to_celsius(15.5))
        assert result.upper_celsius == pytest.approx(fahrenheit_to_celsius(17.5))

    def test_unrelated_string(self):
        assert parse_outcome_temp("sunny skies") is None

    def test_empty_string(self):
        assert parse_outcome_temp("") is None

    def test_adjacent_ranges_share_boundaries(self):
        """Adjacent outcome ranges should share exact boundary values (continuity)."""
        r1 = parse_outcome_temp("18-19°F")  # upper at 19.5°F
        r2 = parse_outcome_temp("20-21°F")  # lower at 19.5°F
        assert r1 is not None and r2 is not None
        assert r1.upper_celsius == pytest.approx(r2.lower_celsius)


class TestParseEventDate:
    """Tests for parse_event_date title/end_date parsing."""

    def test_full_month_name(self):
        result = parse_event_date(
            "Highest temperature in NYC on January 27?",
            "2026-01-27T23:59:00Z",
        )
        assert result == date(2026, 1, 27)

    def test_abbreviated_month(self):
        result = parse_event_date(
            "Temperature in NYC on Jan 5?",
            "2026-01-05T23:59:00Z",
        )
        assert result == date(2026, 1, 5)

    def test_invalid_date_falls_back_to_end_date(self):
        # February 30 is invalid, should fall back to end_date
        result = parse_event_date(
            "Temperature on February 30?",
            "2026-02-28T23:59:00Z",
        )
        assert result == date(2026, 2, 28)

    def test_no_month_match_uses_end_date(self):
        result = parse_event_date(
            "What will the temperature be?",
            "2026-03-15T23:59:00Z",
        )
        assert result == date(2026, 3, 15)

    def test_invalid_end_date_returns_none(self):
        result = parse_event_date(
            "What will the temperature be?",
            "not-a-date",
        )
        assert result is None
