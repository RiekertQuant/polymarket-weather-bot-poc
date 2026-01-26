"""Real Polymarket client for live market data."""

import logging
from datetime import date
from typing import Optional

import requests

from src.polymarket.client_base import Market, OrderBook, PolymarketClientBase
from src.polymarket.parsing import parse_market_title

logger = logging.getLogger(__name__)


class RealPolymarketClient(PolymarketClientBase):
    """Real client that fetches data from Polymarket API.

    Note: This is a best-effort implementation. Polymarket's API may require
    authentication or may have rate limits. If unavailable, use MockPolymarketClient.
    """

    def __init__(self, api_url: str = "https://clob.polymarket.com"):
        """Initialize real client.

        Args:
            api_url: Polymarket API base URL.
        """
        self.api_url = api_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "PolymarketWeatherBot/0.1",
        })

    def _make_request(self, endpoint: str, params: Optional[dict] = None) -> Optional[dict]:
        """Make API request with error handling.

        Args:
            endpoint: API endpoint path.
            params: Query parameters.

        Returns:
            JSON response if successful, None otherwise.
        """
        try:
            url = f"{self.api_url}/{endpoint.lstrip('/')}"
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.warning(f"API request failed: {e}")
            logger.info(
                "Polymarket API unavailable. Consider using MockPolymarketClient "
                "with fixtures for testing."
            )
            return None

    def get_weather_markets(self) -> list[Market]:
        """Fetch weather-related markets from Polymarket.

        Returns:
            List of weather temperature markets.

        Note:
            This searches for markets with weather-related keywords.
            The actual API structure may differ from this implementation.
        """
        # TODO: Implement actual Polymarket API integration
        # The CLOB API structure needs to be verified
        # For now, attempt to search markets

        data = self._make_request("/markets", params={"tag": "weather"})
        if data is None:
            logger.warning(
                "Could not fetch markets from Polymarket API. "
                "Please use MockPolymarketClient with fixtures instead."
            )
            return []

        markets = []
        items = data if isinstance(data, list) else data.get("markets", [])

        for item in items:
            try:
                market = Market(
                    id=str(item.get("id", item.get("condition_id", ""))),
                    title=item.get("question", item.get("title", "")),
                    description=item.get("description", ""),
                    end_date=date.fromisoformat(
                        item.get("end_date_iso", item.get("end_date", "2099-12-31"))[:10]
                    ),
                    yes_price=float(item.get("yes_price", item.get("outcomePrices", [0.5])[0])),
                    volume=float(item.get("volume", 0)),
                    active=item.get("active", True),
                )

                # Parse to extract weather details
                parsed = parse_market_title(market.title, market.description)
                market.city = parsed.city
                market.threshold_celsius = parsed.threshold_celsius
                market.target_date = parsed.target_date

                if market.city is not None:
                    markets.append(market)

            except (KeyError, ValueError, TypeError) as e:
                logger.debug(f"Skipping market due to parse error: {e}")
                continue

        return markets

    def get_orderbook(self, market_id: str) -> Optional[OrderBook]:
        """Fetch orderbook for a market.

        Args:
            market_id: Market identifier.

        Returns:
            OrderBook if available.
        """
        # TODO: Implement actual orderbook fetching
        # This is a placeholder - actual endpoint may differ
        data = self._make_request(f"/orderbook/{market_id}")
        if data is None:
            return None

        try:
            bids = data.get("bids", [])
            asks = data.get("asks", [])

            best_bid = float(bids[0]["price"]) if bids else 0.0
            best_ask = float(asks[0]["price"]) if asks else 1.0
            bid_size = float(bids[0].get("size", 100)) if bids else 0.0
            ask_size = float(asks[0].get("size", 100)) if asks else 0.0

            return OrderBook(
                market_id=market_id,
                best_bid=best_bid,
                best_ask=best_ask,
                bid_size=bid_size,
                ask_size=ask_size,
            )
        except (KeyError, IndexError, ValueError) as e:
            logger.debug(f"Could not parse orderbook: {e}")
            return None

    def get_market_by_id(self, market_id: str) -> Optional[Market]:
        """Fetch a specific market by ID.

        Args:
            market_id: Market identifier.

        Returns:
            Market if found.
        """
        # TODO: Implement actual market fetching by ID
        data = self._make_request(f"/markets/{market_id}")
        if data is None:
            return None

        try:
            market = Market(
                id=str(data.get("id", data.get("condition_id", market_id))),
                title=data.get("question", data.get("title", "")),
                description=data.get("description", ""),
                end_date=date.fromisoformat(
                    data.get("end_date_iso", data.get("end_date", "2099-12-31"))[:10]
                ),
                yes_price=float(data.get("yes_price", 0.5)),
                volume=float(data.get("volume", 0)),
                active=data.get("active", True),
            )

            parsed = parse_market_title(market.title, market.description)
            market.city = parsed.city
            market.threshold_celsius = parsed.threshold_celsius
            market.target_date = parsed.target_date

            return market
        except (KeyError, ValueError, TypeError) as e:
            logger.debug(f"Could not parse market: {e}")
            return None
