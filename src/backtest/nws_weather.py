"""NWS station weather collector for backtesting.

Provides actual weather observations from NWS station data (e.g. KLGA for NYC),
matching Polymarket's resolution source. Forecasts are delegated to the
standard WeatherHistoryCollector (Open-Meteo).
"""

import json
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import requests

from src.backtest.weather_history import (
    ActualWeather,
    HistoricalForecast,
    WeatherHistoryCollector,
)

logger = logging.getLogger(__name__)

# NWS station IDs used by Polymarket for each city
CITY_NWS_STATIONS: dict[str, str] = {
    "New York City": "KLGA",
}

# Timezone offsets (standard time) for local-day filtering
CITY_UTC_OFFSETS: dict[str, int] = {
    "New York City": -5,  # EST
}


class NWSWeatherCollector:
    """Collects actual weather from NWS station observations.

    Fetches actual temperatures from the NWS API (matching Polymarket's
    resolution source) and delegates forecast requests to the standard
    WeatherHistoryCollector (Open-Meteo historical forecasts).
    """

    def __init__(self, cache_dir: Path = Path("data/backtest_cache")):
        """Initialize collector.

        Args:
            cache_dir: Directory to cache weather data.
        """
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "polymarket-weather-bot",
            "Accept": "application/geo+json",
        })

        # Delegate forecasts to the standard Open-Meteo collector
        self._forecast_delegate = WeatherHistoryCollector(cache_dir=cache_dir)

        # In-memory cache for NWS actuals
        self._actual_cache: dict[str, ActualWeather] = {}

    def _get_nws_daily_max(
        self,
        station_id: str,
        target_date: date,
        utc_offset: int,
    ) -> Optional[float]:
        """Fetch daily max temperature from NWS station observations.

        Queries the NWS API for all observations during the local calendar day,
        and returns the highest temperature in Fahrenheit (rounded to nearest
        integer, matching Polymarket's resolution).

        Args:
            station_id: NWS station identifier (e.g. "KLGA").
            target_date: The local calendar date.
            utc_offset: UTC offset in hours (e.g. -5 for EST).

        Returns:
            Max temperature in Fahrenheit as integer, or None if unavailable.
        """
        offset = timedelta(hours=utc_offset)
        start_utc = datetime(
            target_date.year, target_date.month, target_date.day,
            tzinfo=timezone.utc,
        ) - offset
        end_utc = start_utc + timedelta(days=1)

        url = f"https://api.weather.gov/stations/{station_id}/observations"
        params = {
            "start": start_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end": end_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

        try:
            resp = self.session.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.RequestException as e:
            logger.warning(f"NWS API request failed for {station_id}: {e}")
            return None

        max_temp_f: Optional[float] = None
        for obs in data.get("features", []):
            props = obs.get("properties", {})
            temp = props.get("temperature", {})
            val = temp.get("value")
            if val is not None:
                temp_f = val * 9 / 5 + 32
                if max_temp_f is None or temp_f > max_temp_f:
                    max_temp_f = temp_f

        if max_temp_f is not None:
            return round(max_temp_f)
        return None

    def get_actual_weather(self, city: str, target_date: date) -> Optional[ActualWeather]:
        """Get actual observed weather from NWS station data.

        For supported cities (NYC), fetches from NWS station observations.
        Returns None for unsupported cities.

        Args:
            city: City name.
            target_date: Date to get weather for.

        Returns:
            ActualWeather if available.
        """
        cache_key = f"{city}|{target_date}"
        if cache_key in self._actual_cache:
            return self._actual_cache[cache_key]

        station = CITY_NWS_STATIONS.get(city)
        utc_offset = CITY_UTC_OFFSETS.get(city)

        if station is None or utc_offset is None:
            logger.debug(f"No NWS station configured for {city}, skipping")
            return None

        # Don't query future dates
        if target_date >= date.today():
            logger.debug(f"Date {target_date} is not in the past")
            return None

        max_temp_f = self._get_nws_daily_max(station, target_date, utc_offset)
        if max_temp_f is None:
            logger.warning(f"No NWS data for {city} on {target_date}")
            return None

        # Convert rounded integer Fahrenheit back to Celsius for engine compatibility
        max_temp_c = (max_temp_f - 32) * 5 / 9

        actual = ActualWeather(
            date=target_date,
            city=city,
            actual_max_temp=max_temp_c,
            # NWS daily min is not fetched; use 0 as placeholder
            actual_min_temp=0.0,
        )
        self._actual_cache[cache_key] = actual
        logger.info(
            f"NWS {station}: {city} {target_date} -> {max_temp_f}°F ({max_temp_c:.1f}°C)"
        )
        return actual

    def get_forecast_at_decision_time(
        self,
        city: str,
        target_date: date,
        days_before: int = 1,
    ) -> Optional[HistoricalForecast]:
        """Get the forecast that would have been available at decision time.

        Delegates to WeatherHistoryCollector (Open-Meteo historical forecasts).

        Args:
            city: City name.
            target_date: Market resolution date.
            days_before: How many days before to get forecast (default 1).

        Returns:
            HistoricalForecast representing what was known at decision time.
        """
        return self._forecast_delegate.get_forecast_at_decision_time(
            city, target_date, days_before
        )

    def get_historical_forecast(
        self,
        city: str,
        forecast_date: date,
        target_date: date,
    ) -> Optional[HistoricalForecast]:
        """Get what was forecasted on forecast_date for target_date.

        Delegates to WeatherHistoryCollector.

        Args:
            city: City name.
            forecast_date: Date when forecast was made.
            target_date: Date being forecasted.

        Returns:
            HistoricalForecast if available.
        """
        return self._forecast_delegate.get_historical_forecast(
            city, forecast_date, target_date
        )

    def save_cache(self, filename: str = "nws_cache.json") -> Path:
        """Save NWS actual weather cache to file.

        Args:
            filename: Cache filename.

        Returns:
            Path to saved file.
        """
        cache_path = self.cache_dir / filename

        data = {
            "saved_at": datetime.now().isoformat(),
            "source": "NWS station observations",
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

        logger.info(f"Saved NWS cache to {cache_path}")

        # Also save delegate's forecast cache
        self._forecast_delegate.save_cache()

        return cache_path

    def load_cache(self, filename: str = "nws_cache.json") -> bool:
        """Load NWS actual weather cache from file.

        Args:
            filename: Cache filename.

        Returns:
            True if cache loaded successfully.
        """
        cache_path = self.cache_dir / filename

        loaded_nws = False
        if cache_path.exists():
            try:
                with open(cache_path) as f:
                    data = json.load(f)

                for a_data in data.get("actuals", []):
                    actual = ActualWeather(
                        date=date.fromisoformat(a_data["date"]),
                        city=a_data["city"],
                        actual_max_temp=a_data["actual_max_temp"],
                        actual_min_temp=a_data["actual_min_temp"],
                    )
                    cache_key = f"{actual.city}|{actual.date}"
                    self._actual_cache[cache_key] = actual

                logger.info(f"Loaded {len(self._actual_cache)} NWS actuals from cache")
                loaded_nws = True

            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to load NWS cache: {e}")

        # Also load delegate's forecast cache
        loaded_forecasts = self._forecast_delegate.load_cache()

        return loaded_nws or loaded_forecasts
