"""Tests for strategy filters."""

import pytest

from src.config import Settings
from src.strategy.filters import MarketFilter, FilterReason


class TestPriceFilters:
    """Tests for price-based filters."""

    def setup_method(self):
        """Set up test fixtures."""
        self.filter = MarketFilter()

    def test_skip_5050_range_middle(self):
        """Should skip prices in 50/50 range."""
        result = self.filter.check_price_range(0.50)
        assert not result.passed
        assert result.reason == FilterReason.PRICE_5050

    def test_skip_5050_range_lower_bound(self):
        """Should skip at lower bound of 50/50 range."""
        result = self.filter.check_price_range(0.40)
        assert not result.passed
        assert result.reason == FilterReason.PRICE_5050

    def test_skip_5050_range_upper_bound(self):
        """Should skip at upper bound of 50/50 range."""
        result = self.filter.check_price_range(0.60)
        assert not result.passed
        assert result.reason == FilterReason.PRICE_5050

    def test_accept_cheap_price(self):
        """Should accept cheap prices in [0.001, 0.10]."""
        result = self.filter.check_price_range(0.05)
        assert result.passed
        assert result.reason == FilterReason.PASSED

    def test_accept_price_at_min(self):
        """Should accept price at minimum."""
        result = self.filter.check_price_range(0.001)
        assert result.passed

    def test_accept_price_at_max(self):
        """Should accept price at maximum (0.10)."""
        result = self.filter.check_price_range(0.10)
        assert result.passed

    def test_reject_too_low_price(self):
        """Should reject prices below minimum."""
        result = self.filter.check_price_range(0.0005)
        assert not result.passed
        assert result.reason == FilterReason.PRICE_TOO_LOW

    def test_reject_expensive_price(self):
        """Should reject prices above maximum."""
        result = self.filter.check_price_range(0.15)
        assert not result.passed
        assert result.reason == FilterReason.PRICE_TOO_HIGH

    def test_reject_between_max_and_5050(self):
        """Prices between 0.10 and 0.40 should be rejected as too high."""
        result = self.filter.check_price_range(0.25)
        assert not result.passed
        assert result.reason == FilterReason.PRICE_TOO_HIGH


class TestEdgeFilters:
    """Tests for edge-based filters."""

    def setup_method(self):
        """Set up test fixtures."""
        self.filter = MarketFilter()

    def test_sufficient_edge_absolute(self):
        """Should accept when p_model >= 0.60 (absolute minimum)."""
        # Price: 0.05, p_model: 0.65
        # Required: max(0.60, 0.05 + 0.30) = max(0.60, 0.35) = 0.60
        result = self.filter.check_edge(p_model=0.65, price=0.05)
        assert result.passed

    def test_sufficient_edge_relative(self):
        """Should accept when p_model >= price + 0.30."""
        # Price: 0.08, p_model: 0.70
        # Required: max(0.60, 0.08 + 0.30) = max(0.60, 0.38) = 0.60
        result = self.filter.check_edge(p_model=0.70, price=0.08)
        assert result.passed

    def test_insufficient_edge(self):
        """Should reject when edge is too small."""
        # Price: 0.05, p_model: 0.50
        # Required: 0.60, but p_model is only 0.50
        result = self.filter.check_edge(p_model=0.50, price=0.05)
        assert not result.passed
        assert result.reason == FilterReason.INSUFFICIENT_EDGE

    def test_edge_at_boundary(self):
        """Should accept at exact boundary."""
        # Price: 0.05, p_model: 0.60 (exactly at minimum)
        result = self.filter.check_edge(p_model=0.60, price=0.05)
        assert result.passed

    def test_high_price_requires_higher_edge(self):
        """Higher price should require higher p_model."""
        # Price: 0.10, p_model: 0.60
        # Required: max(0.60, 0.10 + 0.30) = max(0.60, 0.40) = 0.60
        result = self.filter.check_edge(p_model=0.60, price=0.10)
        assert result.passed


class TestCombinedFilters:
    """Tests for combined filter application."""

    def setup_method(self):
        """Set up test fixtures."""
        self.filter = MarketFilter()

    def test_all_filters_pass(self):
        """Market passing all filters."""
        result = self.filter.apply_all_filters(
            price=0.05,
            p_model=0.70,
            city="London",
            active=True,
        )
        assert result.passed
        assert result.reason == FilterReason.PASSED

    def test_inactive_market_fails(self):
        """Inactive market should fail."""
        result = self.filter.apply_all_filters(
            price=0.05,
            p_model=0.70,
            city="London",
            active=False,
        )
        assert not result.passed
        assert result.reason == FilterReason.MARKET_INACTIVE

    def test_missing_city_fails(self):
        """Market without city should fail."""
        result = self.filter.apply_all_filters(
            price=0.05,
            p_model=0.70,
            city=None,
            active=True,
        )
        assert not result.passed
        assert result.reason == FilterReason.INVALID_CITY

    def test_5050_price_fails_first(self):
        """50/50 price should fail before edge check."""
        result = self.filter.apply_all_filters(
            price=0.50,
            p_model=0.90,  # Good edge, but price is 50/50
            city="London",
            active=True,
        )
        assert not result.passed
        assert result.reason == FilterReason.PRICE_5050


class TestCustomSettings:
    """Tests with custom settings."""

    def test_custom_price_range(self):
        """Custom price range should be respected."""
        settings = Settings(
            price_min=0.01,
            price_max=0.20,  # Wider range
        )
        filter = MarketFilter(settings)

        result = filter.check_price_range(0.15)
        assert result.passed

    def test_custom_edge_requirements(self):
        """Custom edge requirements should be respected."""
        settings = Settings(
            min_edge_absolute=0.70,  # Higher requirement
            min_edge_relative=0.40,
        )
        filter = MarketFilter(settings)

        # p_model=0.65 would pass default but not custom
        result = filter.check_edge(p_model=0.65, price=0.05)
        assert not result.passed
