"""Market filtering logic."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from src.config import Settings


class FilterReason(Enum):
    """Reasons for filtering out a market."""

    PASSED = "passed"
    PRICE_TOO_HIGH = "price_too_high"
    PRICE_TOO_LOW = "price_too_low"
    PRICE_5050 = "price_in_50_50_range"
    INSUFFICIENT_EDGE = "insufficient_edge"
    MISSING_DATA = "missing_data"
    MARKET_INACTIVE = "market_inactive"
    INVALID_CITY = "invalid_city"


@dataclass
class FilterResult:
    """Result of market filtering."""

    passed: bool
    reason: FilterReason
    details: Optional[str] = None


class MarketFilter:
    """Filters markets based on trading rules."""

    def __init__(self, settings: Optional[Settings] = None):
        """Initialize filter with settings.

        Args:
            settings: Trading settings. Uses defaults if not provided.
        """
        self.settings = settings or Settings()

    def check_price_range(self, price: float) -> FilterResult:
        """Check if price is in acceptable range.

        Args:
            price: Current YES price (0 to 1).

        Returns:
            FilterResult indicating pass/fail.
        """
        # Skip 50/50 range
        if self.settings.skip_price_min <= price <= self.settings.skip_price_max:
            return FilterResult(
                passed=False,
                reason=FilterReason.PRICE_5050,
                details=f"Price {price:.3f} in 50/50 range [{self.settings.skip_price_min}, {self.settings.skip_price_max}]",
            )

        # Check for cheap shares (what we want)
        if price < self.settings.price_min:
            return FilterResult(
                passed=False,
                reason=FilterReason.PRICE_TOO_LOW,
                details=f"Price {price:.3f} below minimum {self.settings.price_min}",
            )

        if price > self.settings.price_max:
            return FilterResult(
                passed=False,
                reason=FilterReason.PRICE_TOO_HIGH,
                details=f"Price {price:.3f} above maximum {self.settings.price_max}",
            )

        return FilterResult(passed=True, reason=FilterReason.PASSED)

    def check_edge(self, p_model: float, price: float) -> FilterResult:
        """Check if edge is sufficient.

        Edge requirement: p_model >= max(min_edge_absolute, price + min_edge_relative)

        Args:
            p_model: Model probability estimate.
            price: Current market price.

        Returns:
            FilterResult indicating pass/fail.
        """
        min_required = max(
            self.settings.min_edge_absolute,
            price + self.settings.min_edge_relative,
        )

        if p_model < min_required:
            return FilterResult(
                passed=False,
                reason=FilterReason.INSUFFICIENT_EDGE,
                details=f"p_model {p_model:.3f} < required {min_required:.3f}",
            )

        return FilterResult(passed=True, reason=FilterReason.PASSED)

    def check_market_valid(
        self,
        city: Optional[str],
        active: bool,
        has_price: bool,
    ) -> FilterResult:
        """Check basic market validity.

        Args:
            city: Parsed city name.
            active: Whether market is active.
            has_price: Whether price data is available.

        Returns:
            FilterResult indicating pass/fail.
        """
        if not active:
            return FilterResult(
                passed=False,
                reason=FilterReason.MARKET_INACTIVE,
            )

        if city is None:
            return FilterResult(
                passed=False,
                reason=FilterReason.INVALID_CITY,
                details="Could not parse city from market title",
            )

        if not has_price:
            return FilterResult(
                passed=False,
                reason=FilterReason.MISSING_DATA,
                details="No price data available",
            )

        return FilterResult(passed=True, reason=FilterReason.PASSED)

    def apply_all_filters(
        self,
        price: float,
        p_model: float,
        city: Optional[str] = None,
        active: bool = True,
    ) -> FilterResult:
        """Apply all filters to a market.

        Args:
            price: Current YES price.
            p_model: Model probability estimate.
            city: Parsed city name.
            active: Whether market is active.

        Returns:
            FilterResult from first failing filter, or passed.
        """
        # Check validity first
        validity = self.check_market_valid(city, active, has_price=True)
        if not validity.passed:
            return validity

        # Check price range
        price_check = self.check_price_range(price)
        if not price_check.passed:
            return price_check

        # Check edge
        edge_check = self.check_edge(p_model, price)
        if not edge_check.passed:
            return edge_check

        return FilterResult(passed=True, reason=FilterReason.PASSED)
