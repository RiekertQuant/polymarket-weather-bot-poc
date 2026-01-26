"""Mock Polymarket client for testing with fixtures."""

import json
from datetime import date
from pathlib import Path
from typing import Optional

from src.polymarket.client_base import Market, OrderBook, PolymarketClientBase
from src.polymarket.parsing import parse_market_title


class MockPolymarketClient(PolymarketClientBase):
    """Mock client that loads market data from JSON fixtures."""

    def __init__(self, fixtures_path: Optional[Path] = None):
        """Initialize mock client.

        Args:
            fixtures_path: Path to fixtures directory. Defaults to tests/fixtures.
        """
        if fixtures_path is None:
            fixtures_path = Path(__file__).parent.parent.parent / "tests" / "fixtures"
        self.fixtures_path = fixtures_path
        self._markets: dict[str, Market] = {}
        self._orderbooks: dict[str, OrderBook] = {}
        self._load_fixtures()

    def _load_fixtures(self) -> None:
        """Load market and orderbook fixtures from JSON files."""
        markets_file = self.fixtures_path / "markets.json"
        orderbooks_file = self.fixtures_path / "orderbooks.json"

        if markets_file.exists():
            with open(markets_file) as f:
                markets_data = json.load(f)
                for m in markets_data:
                    market = Market(
                        id=m["id"],
                        title=m["title"],
                        description=m.get("description", ""),
                        end_date=date.fromisoformat(m["end_date"]),
                        yes_price=m["yes_price"],
                        volume=m.get("volume", 0.0),
                        active=m.get("active", True),
                    )
                    # Parse market title
                    parsed = parse_market_title(market.title, market.description)
                    market.city = parsed.city
                    market.threshold_celsius = parsed.threshold_celsius
                    market.target_date = parsed.target_date
                    self._markets[market.id] = market

        if orderbooks_file.exists():
            with open(orderbooks_file) as f:
                orderbooks_data = json.load(f)
                for ob in orderbooks_data:
                    self._orderbooks[ob["market_id"]] = OrderBook(
                        market_id=ob["market_id"],
                        best_bid=ob["best_bid"],
                        best_ask=ob["best_ask"],
                        bid_size=ob.get("bid_size", 100.0),
                        ask_size=ob.get("ask_size", 100.0),
                    )

    def get_weather_markets(self) -> list[Market]:
        """Get all weather markets from fixtures.

        Returns:
            List of markets that have valid weather parsing.
        """
        return [
            m for m in self._markets.values()
            if m.active and m.city is not None
        ]

    def get_orderbook(self, market_id: str) -> Optional[OrderBook]:
        """Get orderbook for a market.

        Args:
            market_id: Market identifier.

        Returns:
            OrderBook if found.
        """
        return self._orderbooks.get(market_id)

    def get_market_by_id(self, market_id: str) -> Optional[Market]:
        """Get a specific market by ID.

        Args:
            market_id: Market identifier.

        Returns:
            Market if found.
        """
        return self._markets.get(market_id)

    def add_market(self, market: Market) -> None:
        """Add a market programmatically (for testing).

        Args:
            market: Market to add.
        """
        self._markets[market.id] = market

    def add_orderbook(self, orderbook: OrderBook) -> None:
        """Add an orderbook programmatically (for testing).

        Args:
            orderbook: OrderBook to add.
        """
        self._orderbooks[orderbook.market_id] = orderbook
