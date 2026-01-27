"""Real Polymarket client for live market data."""

import logging
import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

import requests

from src.polymarket.client_base import Market, OrderBook, PolymarketClientBase
from src.polymarket.parsing import parse_market_title

logger = logging.getLogger(__name__)


def fahrenheit_to_celsius(f: float) -> float:
    """Convert Fahrenheit to Celsius."""
    return (f - 32) * 5 / 9


def parse_temp_range_title(title: str) -> Optional[tuple[str, float, float]]:
    """Parse temperature range from market title.

    Args:
        title: Market title like "Highest temperature in NYC on January 27?"
               or outcome like "20-21°F"

    Returns:
        Tuple of (city, low_f, high_f) if parseable, None otherwise.
    """
    # Check for city in title
    city = None
    if "nyc" in title.lower() or "new york" in title.lower():
        city = "New York City"

    return city


@dataclass
class OutcomeTemp:
    """Parsed temperature outcome with bounds and comparison type."""
    lower_celsius: Optional[float]  # Lower bound (None for "or below")
    upper_celsius: Optional[float]  # Upper bound (None for "or higher")
    comparison: str  # "<=", ">=", or "range"


def parse_outcome_temp(outcome: str) -> Optional[OutcomeTemp]:
    """Parse temperature range from outcome string.

    Uses ±0.5°F continuity correction so that adjacent ranges
    share boundaries and probabilities sum to 1.

    Args:
        outcome: String like "20-21°F", "15°F or below", "26°F or higher"

    Returns:
        OutcomeTemp with bounds in Celsius and comparison type.
    """
    outcome_lower = outcome.lower()

    # Pattern: "X°F or below" / "X°F or lower"
    match = re.search(r"(\d+)\s*°?\s*f\s+or\s+(below|lower)", outcome_lower)
    if match:
        temp_f = float(match.group(1))
        # "15°F or below" → temp <= 15.5°F (continuity correction)
        return OutcomeTemp(
            lower_celsius=None,
            upper_celsius=fahrenheit_to_celsius(temp_f + 0.5),
            comparison="<=",
        )

    # Pattern: "X°F or higher" / "X°F or above"
    match = re.search(r"(\d+)\s*°?\s*f\s+or\s+(higher|above)", outcome_lower)
    if match:
        temp_f = float(match.group(1))
        # "26°F or higher" → temp >= 25.5°F (continuity correction)
        return OutcomeTemp(
            lower_celsius=fahrenheit_to_celsius(temp_f - 0.5),
            upper_celsius=None,
            comparison=">=",
        )

    # Pattern: "X-Y°F" (range)
    match = re.search(r"(\d+)\s*-\s*(\d+)\s*°?\s*f", outcome_lower)
    if match:
        low_f = float(match.group(1))
        high_f = float(match.group(2))
        # "20-21°F" → 19.5°F <= temp <= 21.5°F (continuity correction)
        return OutcomeTemp(
            lower_celsius=fahrenheit_to_celsius(low_f - 0.5),
            upper_celsius=fahrenheit_to_celsius(high_f + 0.5),
            comparison="range",
        )

    return None


def parse_event_date(title: str, end_date_str: str) -> Optional[date]:
    """Parse target date from event title or end date.

    Args:
        title: Event title like "Highest temperature in NYC on January 27?"
        end_date_str: ISO date string from API

    Returns:
        Target date if found.
    """
    # Try to extract date from title
    months = {
        "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
        "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
        "august": 8, "aug": 8, "september": 9, "sep": 9, "october": 10, "oct": 10,
        "november": 11, "nov": 11, "december": 12, "dec": 12
    }

    title_lower = title.lower()
    for month_name, month_num in months.items():
        match = re.search(rf"{month_name}\s+(\d{{1,2}})", title_lower)
        if match:
            day = int(match.group(1))
            # Use end_date year as reference since that's authoritative
            try:
                ref_date = date.fromisoformat(end_date_str[:10])
                return date(ref_date.year, month_num, day)
            except (ValueError, TypeError):
                pass
            # Fallback: use current year
            year = datetime.now().year
            try:
                return date(year, month_num, day)
            except ValueError:
                pass

    # Fall back to end_date
    try:
        return date.fromisoformat(end_date_str[:10])
    except (ValueError, TypeError):
        return None


class RealPolymarketClient(PolymarketClientBase):
    """Real client that fetches data from Polymarket API.

    Uses the Gamma API for event/market discovery and CLOB API for orderbooks.
    """

    GAMMA_API_URL = "https://gamma-api.polymarket.com"
    CLOB_API_URL = "https://clob.polymarket.com"

    def __init__(self, api_url: str = "https://gamma-api.polymarket.com"):
        """Initialize real client.

        Args:
            api_url: Polymarket Gamma API base URL.
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
        """Fetch weather-related markets from Polymarket Gamma API.

        Queries weather-tagged events and parses each outcome
        (temperature range) as a separate Market object.

        Returns:
            List of weather temperature markets.
        """
        # Fetch weather events from Gamma API
        data = self._make_request("/events", params={
            "tag_slug": "weather",
            "closed": "false",
            "limit": "100",
        })
        if data is None:
            logger.warning("Could not fetch events from Polymarket Gamma API.")
            return []

        events = data if isinstance(data, list) else []
        logger.info(f"Fetched {len(events)} weather events from Gamma API")

        markets = []
        for event in events:
            event_title = event.get("title", "")
            event_desc = event.get("description", "")
            end_date_str = event.get("endDate", "2099-12-31")

            # Check if this is a temperature event we can parse
            city_info = parse_temp_range_title(event_title)
            if city_info is None:
                continue

            city = city_info
            target_date = parse_event_date(event_title, end_date_str)
            if target_date is None:
                continue

            # Each event has multiple outcome markets (temperature ranges)
            event_markets = event.get("markets", [])
            for mkt in event_markets:
                question = mkt.get("question", mkt.get("groupItemTitle", ""))
                outcome = mkt.get("groupItemTitle", question)

                temp_info = parse_outcome_temp(outcome)
                if temp_info is None:
                    temp_info = parse_outcome_temp(question)
                if temp_info is None:
                    continue

                # Get YES price from outcomePrices or bestAsk
                yes_price = 0.5
                outcome_prices = mkt.get("outcomePrices")
                if outcome_prices and isinstance(outcome_prices, str):
                    import json as _json
                    try:
                        prices = _json.loads(outcome_prices)
                        yes_price = float(prices[0])
                    except (ValueError, IndexError):
                        pass
                elif outcome_prices and isinstance(outcome_prices, list):
                    yes_price = float(outcome_prices[0])
                elif mkt.get("bestAsk"):
                    yes_price = float(mkt["bestAsk"])

                try:
                    end_date = date.fromisoformat(end_date_str[:10])
                except (ValueError, TypeError):
                    end_date = date(2099, 12, 31)

                market = Market(
                    id=str(mkt.get("id", mkt.get("conditionId", ""))),
                    title=f"{event_title} - {outcome}",
                    description=event_desc,
                    end_date=end_date,
                    yes_price=yes_price,
                    volume=float(mkt.get("volume", 0)),
                    active=mkt.get("active", True),
                    city=city,
                    threshold_celsius=temp_info.lower_celsius,
                    threshold_celsius_upper=temp_info.upper_celsius,
                    target_date=target_date,
                    comparison=temp_info.comparison,
                )
                markets.append(market)

                logger.debug(
                    f"Parsed market: {market.title} | "
                    f"city={city}, comp={temp_info.comparison}, "
                    f"low={temp_info.lower_celsius}, high={temp_info.upper_celsius}, "
                    f"price={yes_price:.3f}"
                )

        logger.info(f"Found {len(markets)} weather temperature markets")
        return markets

    def get_orderbook(self, market_id: str) -> Optional[OrderBook]:
        """Fetch orderbook for a market from CLOB API.

        Args:
            market_id: Market condition_id identifier.

        Returns:
            OrderBook if available.
        """
        try:
            url = f"{self.CLOB_API_URL}/book"
            response = self.session.get(url, params={"token_id": market_id}, timeout=10)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            logger.debug(f"Orderbook request failed: {e}")
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
        """Fetch a specific market by ID from Gamma API.

        Args:
            market_id: Market identifier.

        Returns:
            Market if found.
        """
        data = self._make_request(f"/markets/{market_id}")
        if data is None:
            return None

        # Gamma API may return a list or single object
        item = data[0] if isinstance(data, list) and data else data
        if not item:
            return None

        try:
            market = Market(
                id=str(item.get("id", item.get("conditionId", market_id))),
                title=item.get("question", item.get("title", "")),
                description=item.get("description", ""),
                end_date=date.fromisoformat(
                    item.get("endDate", item.get("end_date", "2099-12-31"))[:10]
                ),
                yes_price=float(item.get("outcomePrices", "[0.5]").strip("[]").split(",")[0]) if isinstance(item.get("outcomePrices"), str) else 0.5,
                volume=float(item.get("volume", 0)),
                active=item.get("active", True),
            )

            parsed = parse_market_title(market.title, market.description)
            market.city = parsed.city
            market.threshold_celsius = parsed.threshold_celsius
            market.target_date = parsed.target_date

            return market
        except (KeyError, ValueError, TypeError) as e:
            logger.debug(f"Could not parse market: {e}")
            return None
