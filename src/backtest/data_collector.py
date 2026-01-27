"""Polymarket historical data collector for backtesting."""

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import requests

from src.polymarket.parsing import parse_market_title, is_valid_weather_market

logger = logging.getLogger(__name__)


@dataclass
class PricePoint:
    """Single price observation."""

    timestamp: datetime
    price: float


@dataclass
class HistoricalMarket:
    """Historical market data with price history."""

    # Market identifiers
    condition_id: str
    token_id: str  # CLOB token ID for YES outcome

    # Market metadata
    question: str
    description: str
    end_date: date
    outcome: Optional[bool] = None  # True=YES won, False=NO won, None=unresolved

    # Parsed weather info
    city: Optional[str] = None
    threshold_celsius: Optional[float] = None
    target_date: Optional[date] = None
    comparison: str = ">="

    # Price history
    price_history: list[PricePoint] = field(default_factory=list)

    # Volume/liquidity
    volume: float = 0.0

    def to_dict(self) -> dict:
        """Convert to serializable dictionary."""
        return {
            "condition_id": self.condition_id,
            "token_id": self.token_id,
            "question": self.question,
            "description": self.description,
            "end_date": self.end_date.isoformat(),
            "outcome": self.outcome,
            "city": self.city,
            "threshold_celsius": self.threshold_celsius,
            "target_date": self.target_date.isoformat() if self.target_date else None,
            "comparison": self.comparison,
            "price_history": [
                {"timestamp": p.timestamp.isoformat(), "price": p.price}
                for p in self.price_history
            ],
            "volume": self.volume,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "HistoricalMarket":
        """Create from dictionary."""
        price_history = [
            PricePoint(
                timestamp=datetime.fromisoformat(p["timestamp"]),
                price=p["price"],
            )
            for p in data.get("price_history", [])
        ]

        return cls(
            condition_id=data["condition_id"],
            token_id=data["token_id"],
            question=data["question"],
            description=data.get("description", ""),
            end_date=date.fromisoformat(data["end_date"]),
            outcome=data.get("outcome"),
            city=data.get("city"),
            threshold_celsius=data.get("threshold_celsius"),
            target_date=date.fromisoformat(data["target_date"]) if data.get("target_date") else None,
            comparison=data.get("comparison", ">="),
            price_history=price_history,
            volume=data.get("volume", 0.0),
        )


class PolymarketHistoricalCollector:
    """Collects historical market data from Polymarket APIs."""

    GAMMA_URL = "https://gamma-api.polymarket.com"
    CLOB_URL = "https://clob.polymarket.com"

    # Weather-related search terms
    WEATHER_KEYWORDS = ["temperature", "celsius", "fahrenheit", "degrees", "weather", "°C", "°F"]
    CITY_KEYWORDS = ["new york", "nyc", "london", "seoul", "tokyo", "paris", "chicago", "los angeles"]

    def __init__(self, cache_dir: Path = Path("data/backtest_cache")):
        """Initialize collector.

        Args:
            cache_dir: Directory to cache collected data.
        """
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "PolymarketWeatherBot-Backtest/0.1",
        })

    def _rate_limit(self, delay: float = 0.5) -> None:
        """Simple rate limiting."""
        time.sleep(delay)

    def search_weather_markets(self, limit: int = 100) -> list[dict]:
        """Search Gamma API for weather-related markets.

        Args:
            limit: Maximum markets to fetch.

        Returns:
            List of raw market data from Gamma API.
        """
        weather_markets = []

        # Search using different keywords
        for keyword in self.WEATHER_KEYWORDS + self.CITY_KEYWORDS:
            try:
                self._rate_limit(0.3)

                # Gamma API search endpoint
                response = self.session.get(
                    f"{self.GAMMA_URL}/markets",
                    params={
                        "closed": "true",  # Include resolved markets for backtesting
                        "limit": limit,
                        "order": "endDate",
                        "ascending": "false",
                    },
                    timeout=15,
                )
                response.raise_for_status()
                data = response.json()

                markets = data if isinstance(data, list) else data.get("markets", [])

                # Filter for weather/temperature related
                for market in markets:
                    question = market.get("question", "").lower()
                    description = market.get("description", "").lower()
                    combined = f"{question} {description}"

                    # Check if temperature-related
                    is_temp_market = any(
                        kw in combined
                        for kw in ["celsius", "°c", "temperature", "degrees c"]
                    )

                    # Check if city-related
                    has_city = any(
                        city in combined
                        for city in ["new york", "nyc", "london", "seoul"]
                    )

                    if is_temp_market and has_city:
                        # Avoid duplicates
                        if not any(m.get("conditionId") == market.get("conditionId") for m in weather_markets):
                            weather_markets.append(market)

                logger.info(f"Search '{keyword}': found {len(markets)} markets, {len(weather_markets)} weather total")

            except requests.exceptions.RequestException as e:
                logger.warning(f"Gamma API request failed for '{keyword}': {e}")
                continue

        return weather_markets

    def get_price_history(
        self,
        token_id: str,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
        fidelity: int = 60,  # 60 minute intervals
    ) -> list[PricePoint]:
        """Fetch price history from CLOB API.

        Args:
            token_id: CLOB token ID.
            start_ts: Unix timestamp start (optional).
            end_ts: Unix timestamp end (optional).
            fidelity: Data resolution in minutes.

        Returns:
            List of PricePoint objects.
        """
        try:
            self._rate_limit(0.5)

            params = {"market": token_id, "fidelity": fidelity}

            if start_ts and end_ts:
                params["startTs"] = start_ts
                params["endTs"] = end_ts
            else:
                params["interval"] = "max"

            response = self.session.get(
                f"{self.CLOB_URL}/prices-history",
                params=params,
                timeout=15,
            )
            response.raise_for_status()
            data = response.json()

            history = data.get("history", [])

            price_points = []
            for point in history:
                try:
                    ts = point.get("t", 0)
                    price = point.get("p", 0.0)

                    price_points.append(PricePoint(
                        timestamp=datetime.utcfromtimestamp(ts),
                        price=float(price),
                    ))
                except (ValueError, TypeError) as e:
                    logger.debug(f"Skipping invalid price point: {e}")
                    continue

            logger.debug(f"Token {token_id}: {len(price_points)} price points")
            return price_points

        except requests.exceptions.RequestException as e:
            logger.warning(f"CLOB price history request failed for {token_id}: {e}")
            return []

    def collect_market(self, raw_market: dict) -> Optional[HistoricalMarket]:
        """Process a raw market and collect its data.

        Args:
            raw_market: Raw market data from Gamma API.

        Returns:
            HistoricalMarket if valid weather market, None otherwise.
        """
        question = raw_market.get("question", "")
        description = raw_market.get("description", "")

        # Parse weather info
        parsed = parse_market_title(question, description)

        if not is_valid_weather_market(parsed):
            return None

        # Extract IDs
        condition_id = raw_market.get("conditionId", raw_market.get("condition_id", ""))

        # Get token IDs - need the YES token
        tokens = raw_market.get("tokens", [])
        yes_token_id = None

        for token in tokens:
            outcome = token.get("outcome", "").upper()
            if outcome == "YES":
                yes_token_id = token.get("token_id", token.get("tokenId", ""))
                break

        if not yes_token_id:
            # Try alternative structure
            clob_token_ids = raw_market.get("clobTokenIds", [])
            if clob_token_ids:
                yes_token_id = clob_token_ids[0]  # First is typically YES

        if not yes_token_id:
            logger.warning(f"No YES token found for market: {question[:50]}")
            return None

        # Parse end date
        end_date_str = raw_market.get("endDate", raw_market.get("end_date_iso", ""))
        try:
            end_date = date.fromisoformat(end_date_str[:10]) if end_date_str else date.today()
        except ValueError:
            end_date = date.today()

        # Determine outcome
        outcome = None
        resolution = raw_market.get("resolution", raw_market.get("outcome", ""))
        if resolution:
            resolution_lower = str(resolution).lower()
            if resolution_lower in ["yes", "1", "true"]:
                outcome = True
            elif resolution_lower in ["no", "0", "false"]:
                outcome = False

        # Get price history
        price_history = self.get_price_history(yes_token_id)

        return HistoricalMarket(
            condition_id=condition_id,
            token_id=yes_token_id,
            question=question,
            description=description,
            end_date=end_date,
            outcome=outcome,
            city=parsed.city,
            threshold_celsius=parsed.threshold_celsius,
            target_date=parsed.target_date,
            comparison=parsed.comparison,
            price_history=price_history,
            volume=float(raw_market.get("volume", raw_market.get("volumeNum", 0)) or 0),
        )

    def collect_all(self, max_markets: int = 50) -> list[HistoricalMarket]:
        """Collect all historical weather markets.

        Args:
            max_markets: Maximum number of markets to collect.

        Returns:
            List of HistoricalMarket objects with price history.
        """
        logger.info("Searching for historical weather markets...")

        raw_markets = self.search_weather_markets(limit=200)
        logger.info(f"Found {len(raw_markets)} potential weather markets")

        markets = []
        for i, raw in enumerate(raw_markets[:max_markets]):
            logger.info(f"Processing market {i+1}/{min(len(raw_markets), max_markets)}")

            market = self.collect_market(raw)
            if market:
                markets.append(market)
                logger.info(f"  -> {market.city}: {market.question[:50]}... ({len(market.price_history)} prices)")

        logger.info(f"Collected {len(markets)} valid weather markets")
        return markets

    def save_to_cache(self, markets: list[HistoricalMarket], filename: str = "markets.json") -> Path:
        """Save collected markets to cache file.

        Args:
            markets: List of markets to save.
            filename: Cache filename.

        Returns:
            Path to saved file.
        """
        cache_path = self.cache_dir / filename

        data = {
            "collected_at": datetime.now().isoformat(),
            "count": len(markets),
            "markets": [m.to_dict() for m in markets],
        }

        with open(cache_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(markets)} markets to {cache_path}")
        return cache_path

    def load_from_cache(self, filename: str = "markets.json") -> list[HistoricalMarket]:
        """Load markets from cache file.

        Args:
            filename: Cache filename.

        Returns:
            List of HistoricalMarket objects.
        """
        cache_path = self.cache_dir / filename

        if not cache_path.exists():
            logger.warning(f"Cache file not found: {cache_path}")
            return []

        with open(cache_path) as f:
            data = json.load(f)

        markets = [HistoricalMarket.from_dict(m) for m in data.get("markets", [])]
        logger.info(f"Loaded {len(markets)} markets from cache")
        return markets
