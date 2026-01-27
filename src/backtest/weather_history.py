"""Historical weather data collector for backtesting."""

import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import requests

from src.weather.open_meteo import CITY_COORDS, CityCoordinates

logger = logging.getLogger(__name__)


@dataclass
class HistoricalForecast:
    """What was forecasted at a specific time for a future date."""

    forecast_date: date  # When the forecast was made
    target_date: date  # Date being forecasted
    city: str
    forecast_max_temp: float
    forecast_min_temp: float
    lead_days: int  # Days between forecast and target


@dataclass
class ActualWeather:
    """Actual observed weather."""

    date: date
    city: str
    actual_max_temp: float
    actual_min_temp: float


class WeatherHistoryCollector:
    """Collects historical weather forecasts and actuals from Open-Meteo."""

    HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    HISTORICAL_WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"

    def __init__(self, cache_dir: Path = Path("data/backtest_cache")):
        """Initialize collector.

        Args:
            cache_dir: Directory to cache weather data.
        """
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()

        # In-memory cache
        self._forecast_cache: dict[str, HistoricalForecast] = {}
        self._actual_cache: dict[str, ActualWeather] = {}

    def _get_coords(self, city: str) -> Optional[CityCoordinates]:
        """Get coordinates for a city."""
        return CITY_COORDS.get(city)

    def get_historical_forecast(
        self,
        city: str,
        forecast_date: date,
        target_date: date,
    ) -> Optional[HistoricalForecast]:
        """Get what was forecasted on forecast_date for target_date.

        Uses Open-Meteo Historical Forecast API to retrieve past predictions.

        Args:
            city: City name.
            forecast_date: Date when forecast was made.
            target_date: Date being forecasted.

        Returns:
            HistoricalForecast if available.
        """
        cache_key = f"{city}|{forecast_date}|{target_date}"
        if cache_key in self._forecast_cache:
            return self._forecast_cache[cache_key]

        coords = self._get_coords(city)
        if coords is None:
            logger.warning(f"Unknown city: {city}")
            return None

        lead_days = (target_date - forecast_date).days
        if lead_days < 0:
            logger.warning("Forecast date must be before or equal to target date")
            return None

        try:
            # Historical Forecast API requires start_date and end_date
            # and optionally past_days to specify how far back
            params = {
                "latitude": coords.latitude,
                "longitude": coords.longitude,
                "start_date": forecast_date.isoformat(),
                "end_date": target_date.isoformat(),
                "daily": "temperature_2m_max,temperature_2m_min",
                "timezone": coords.timezone,
            }

            response = self.session.get(
                self.HISTORICAL_FORECAST_URL,
                params=params,
                timeout=15,
            )
            response.raise_for_status()
            data = response.json()

            daily = data.get("daily", {})
            dates = daily.get("time", [])
            max_temps = daily.get("temperature_2m_max", [])
            min_temps = daily.get("temperature_2m_min", [])

            # Find the target date in results
            target_str = target_date.isoformat()
            for i, d in enumerate(dates):
                if d == target_str and i < len(max_temps) and i < len(min_temps):
                    forecast = HistoricalForecast(
                        forecast_date=forecast_date,
                        target_date=target_date,
                        city=city,
                        forecast_max_temp=max_temps[i],
                        forecast_min_temp=min_temps[i],
                        lead_days=lead_days,
                    )
                    self._forecast_cache[cache_key] = forecast
                    return forecast

            logger.debug(f"No forecast data for {city} on {target_date}")
            return None

        except requests.exceptions.RequestException as e:
            logger.warning(f"Historical forecast request failed: {e}")
            return None

    def get_actual_weather(self, city: str, target_date: date) -> Optional[ActualWeather]:
        """Get actual observed weather for a date.

        Uses Open-Meteo Historical Weather API (ERA5 reanalysis).

        Args:
            city: City name.
            target_date: Date to get weather for.

        Returns:
            ActualWeather if available.
        """
        cache_key = f"{city}|{target_date}"
        if cache_key in self._actual_cache:
            return self._actual_cache[cache_key]

        coords = self._get_coords(city)
        if coords is None:
            logger.warning(f"Unknown city: {city}")
            return None

        # Don't query future dates
        if target_date >= date.today():
            logger.debug(f"Date {target_date} is not in the past")
            return None

        try:
            params = {
                "latitude": coords.latitude,
                "longitude": coords.longitude,
                "start_date": target_date.isoformat(),
                "end_date": target_date.isoformat(),
                "daily": "temperature_2m_max,temperature_2m_min",
                "timezone": coords.timezone,
            }

            response = self.session.get(
                self.HISTORICAL_WEATHER_URL,
                params=params,
                timeout=15,
            )
            response.raise_for_status()
            data = response.json()

            daily = data.get("daily", {})
            max_temps = daily.get("temperature_2m_max", [])
            min_temps = daily.get("temperature_2m_min", [])

            if max_temps and min_temps:
                actual = ActualWeather(
                    date=target_date,
                    city=city,
                    actual_max_temp=max_temps[0],
                    actual_min_temp=min_temps[0],
                )
                self._actual_cache[cache_key] = actual
                return actual

            return None

        except requests.exceptions.RequestException as e:
            logger.warning(f"Historical weather request failed: {e}")
            return None

    def get_forecast_at_decision_time(
        self,
        city: str,
        target_date: date,
        days_before: int = 1,
    ) -> Optional[HistoricalForecast]:
        """Get the forecast that would have been available at decision time.

        This simulates what the bot would have seen when deciding to trade.

        Args:
            city: City name.
            target_date: Market resolution date.
            days_before: How many days before to get forecast (default 1 = day before).

        Returns:
            HistoricalForecast representing what was known at decision time.
        """
        forecast_date = target_date - timedelta(days=days_before)
        return self.get_historical_forecast(city, forecast_date, target_date)

    def evaluate_forecast_accuracy(
        self,
        city: str,
        target_date: date,
        threshold_celsius: float,
        comparison: str = ">=",
        days_before: int = 1,
    ) -> Optional[dict]:
        """Evaluate how accurate a forecast would have been.

        Args:
            city: City name.
            target_date: Market resolution date.
            threshold_celsius: Temperature threshold.
            comparison: ">=" or "<".
            days_before: Days before to make decision.

        Returns:
            Dict with forecast, actual, and whether prediction was correct.
        """
        forecast = self.get_forecast_at_decision_time(city, target_date, days_before)
        actual = self.get_actual_weather(city, target_date)

        if forecast is None or actual is None:
            return None

        # Determine predicted outcome based on forecast
        if comparison == ">=":
            predicted_yes = forecast.forecast_max_temp >= threshold_celsius
            actual_yes = actual.actual_max_temp >= threshold_celsius
        else:  # "<"
            predicted_yes = forecast.forecast_max_temp < threshold_celsius
            actual_yes = actual.actual_max_temp < threshold_celsius

        return {
            "city": city,
            "target_date": target_date,
            "threshold": threshold_celsius,
            "comparison": comparison,
            "forecast_max": forecast.forecast_max_temp,
            "actual_max": actual.actual_max_temp,
            "predicted_yes": predicted_yes,
            "actual_yes": actual_yes,
            "correct": predicted_yes == actual_yes,
            "forecast_error": forecast.forecast_max_temp - actual.actual_max_temp,
        }

    def collect_date_range(
        self,
        city: str,
        start_date: date,
        end_date: date,
    ) -> tuple[list[HistoricalForecast], list[ActualWeather]]:
        """Collect forecasts and actuals for a date range.

        Args:
            city: City name.
            start_date: Start of range.
            end_date: End of range.

        Returns:
            Tuple of (forecasts, actuals) lists.
        """
        forecasts = []
        actuals = []

        current = start_date
        while current <= end_date:
            # Get actual
            actual = self.get_actual_weather(city, current)
            if actual:
                actuals.append(actual)

            # Get day-before forecast
            forecast = self.get_forecast_at_decision_time(city, current, days_before=1)
            if forecast:
                forecasts.append(forecast)

            current += timedelta(days=1)

        logger.info(f"Collected {len(forecasts)} forecasts and {len(actuals)} actuals for {city}")
        return forecasts, actuals

    def save_cache(self, filename: str = "weather_cache.json") -> Path:
        """Save weather cache to file.

        Args:
            filename: Cache filename.

        Returns:
            Path to saved file.
        """
        cache_path = self.cache_dir / filename

        data = {
            "saved_at": datetime.now().isoformat(),
            "forecasts": [
                {
                    "forecast_date": f.forecast_date.isoformat(),
                    "target_date": f.target_date.isoformat(),
                    "city": f.city,
                    "forecast_max_temp": f.forecast_max_temp,
                    "forecast_min_temp": f.forecast_min_temp,
                    "lead_days": f.lead_days,
                }
                for f in self._forecast_cache.values()
            ],
            "actuals": [
                {
                    "date": a.date.isoformat(),
                    "city": a.city,
                    "actual_max_temp": a.actual_max_temp,
                    "actual_min_temp": a.actual_min_temp,
                }
                for a in self._actual_cache.values()
            ],
        }

        with open(cache_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved weather cache to {cache_path}")
        return cache_path

    def load_cache(self, filename: str = "weather_cache.json") -> bool:
        """Load weather cache from file.

        Args:
            filename: Cache filename.

        Returns:
            True if cache loaded successfully.
        """
        cache_path = self.cache_dir / filename

        if not cache_path.exists():
            return False

        try:
            with open(cache_path) as f:
                data = json.load(f)

            for f_data in data.get("forecasts", []):
                forecast = HistoricalForecast(
                    forecast_date=date.fromisoformat(f_data["forecast_date"]),
                    target_date=date.fromisoformat(f_data["target_date"]),
                    city=f_data["city"],
                    forecast_max_temp=f_data["forecast_max_temp"],
                    forecast_min_temp=f_data["forecast_min_temp"],
                    lead_days=f_data["lead_days"],
                )
                cache_key = f"{forecast.city}|{forecast.forecast_date}|{forecast.target_date}"
                self._forecast_cache[cache_key] = forecast

            for a_data in data.get("actuals", []):
                actual = ActualWeather(
                    date=date.fromisoformat(a_data["date"]),
                    city=a_data["city"],
                    actual_max_temp=a_data["actual_max_temp"],
                    actual_min_temp=a_data["actual_min_temp"],
                )
                cache_key = f"{actual.city}|{actual.date}"
                self._actual_cache[cache_key] = actual

            logger.info(f"Loaded {len(self._forecast_cache)} forecasts and {len(self._actual_cache)} actuals from cache")
            return True

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to load weather cache: {e}")
            return False
