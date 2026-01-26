"""Base classes for Polymarket client."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from typing import Optional


@dataclass
class Market:
    """Represents a Polymarket market."""

    id: str
    title: str
    description: str
    end_date: date
    yes_price: float  # Current YES price (0 to 1)
    volume: float  # Total volume in USD
    active: bool = True

    # Parsed fields (populated by parsing module)
    city: Optional[str] = None
    threshold_celsius: Optional[float] = None
    target_date: Optional[date] = None


@dataclass
class OrderBook:
    """Simplified orderbook for a market."""

    market_id: str
    best_bid: float  # Best bid price
    best_ask: float  # Best ask price
    bid_size: float  # Size at best bid
    ask_size: float  # Size at best ask


class PolymarketClientBase(ABC):
    """Abstract base class for Polymarket clients."""

    @abstractmethod
    def get_weather_markets(self) -> list[Market]:
        """Fetch all weather-related markets.

        Returns:
            List of Market objects for weather temperature markets.
        """
        pass

    @abstractmethod
    def get_orderbook(self, market_id: str) -> Optional[OrderBook]:
        """Fetch orderbook for a specific market.

        Args:
            market_id: The market identifier.

        Returns:
            OrderBook if available, None otherwise.
        """
        pass

    @abstractmethod
    def get_market_by_id(self, market_id: str) -> Optional[Market]:
        """Fetch a specific market by ID.

        Args:
            market_id: The market identifier.

        Returns:
            Market if found, None otherwise.
        """
        pass
