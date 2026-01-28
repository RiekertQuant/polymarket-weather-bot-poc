"""NWS (National Weather Service) forecast client.

Uses the same data source that Polymarket uses for resolution,
which should provide better alignment between forecasts and outcomes.
"""

import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

import requests

from src.weather.open_meteo import DailyForecast, CITY_COORDS

logger = logging.getLogger(__name__)


# NWS grid points for supported cities
# Find yours at: https://api.weather.gov/points/{lat},{lon}
NWS_GRIDPOINTS = {
    "New York City": {
        "office": "OKX",  # Upton, NY
        "grid_x": 33,
        "grid_y": 37,
    },
    # Add more cities as needed
}


@dataclass
class NWSForecastPeriod:
    """A single forecast period from NWS."""

    name: str
    start_time: datetime
    end_time: datetime
    temperature: int  # Fahrenheit (what Polymarket uses)
    temperature_unit: str
    is_daytime: bool
    short_forecast: str


class NWSForecastClient:
    """Client for NWS forecast API.

    Uses the same data source as Polymarket for resolution,
    which should improve forecast-to-outcome alignment.
    """

    BASE_URL = "https://api.weather.gov"

    def __init__(self, timeout: int = 15):
        """Initialize client.

        Args:
            timeout: Request timeout in seconds.
        """
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "polymarket-weather-bot",
            "Accept": "application/geo+json",
        })
        self._cache: dict[str, list[NWSForecastPeriod]] = {}

    def _fetch_forecast_periods(self, city: str) -> list[NWSForecastPeriod]:
        """Fetch forecast periods from NWS API."""
        if city in self._cache:
            return self._cache[city]

        grid = NWS_GRIDPOINTS.get(city)
        if not grid:
            logger.warning(f"No NWS grid point configured for {city}")
            return []

        url = f"{self.BASE_URL}/gridpoints/{grid['office']}/{grid['grid_x']},{grid['grid_y']}/forecast"

        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"NWS API request failed: {e}")
            return []

        periods = []
        for p in data.get("properties", {}).get("periods", []):
            try:
                periods.append(NWSForecastPeriod(
                    name=p["name"],
                    start_time=datetime.fromisoformat(p["startTime"].replace("Z", "+00:00")),
                    end_time=datetime.fromisoformat(p["endTime"].replace("Z", "+00:00")),
                    temperature=p["temperature"],
                    temperature_unit=p["temperatureUnit"],
                    is_daytime=p["isDaytime"],
                    short_forecast=p["shortForecast"],
                ))
            except (KeyError, ValueError) as e:
                logger.debug(f"Could not parse period: {e}")
                continue

        self._cache[city] = periods
        return periods

    def get_forecast(self, city: str, days: int = 7) -> list[DailyForecast]:
        """Fetch daily forecast for a city.

        Args:
            city: City name.
            days: Number of forecast days.

        Returns:
            List of DailyForecast objects.
        """
        periods = self._fetch_forecast_periods(city)
        if not periods:
            return []

        # Group periods by date and extract high/low
        daily_temps: dict[date, dict] = {}

        for period in periods:
            period_date = period.start_time.date()

            if period_date not in daily_temps:
                daily_temps[period_date] = {"high": None, "low": None}

            # Daytime periods have high temps, nighttime have lows
            if period.is_daytime:
                daily_temps[period_date]["high"] = period.temperature
            else:
                daily_temps[period_date]["low"] = period.temperature

        # Convert to DailyForecast objects
        forecasts = []
        for forecast_date in sorted(daily_temps.keys())[:days]:
            temps = daily_temps[forecast_date]
            high_f = temps.get("high")
            low_f = temps.get("low")

            if high_f is None:
                continue

            # Convert F to C for compatibility with existing code
            high_c = (high_f - 32) * 5 / 9
            low_c = (low_f - 32) * 5 / 9 if low_f else high_c - 10

            forecasts.append(DailyForecast(
                date=forecast_date,
                temperature_max=high_c,
                temperature_min=low_c,
                temperature_mean=(high_c + low_c) / 2,
            ))

        return forecasts

    def get_forecast_for_date(
        self,
        city: str,
        target_date: date,
    ) -> Optional[DailyForecast]:
        """Get forecast for a specific date.

        Args:
            city: City name.
            target_date: Target date for forecast.

        Returns:
            DailyForecast if available.
        """
        forecasts = self.get_forecast(city, days=7)
        for forecast in forecasts:
            if forecast.date == target_date:
                return forecast
        return None

    def get_raw_forecast_f(self, city: str, target_date: date) -> Optional[int]:
        """Get raw forecast high temperature in Fahrenheit.

        This is the exact format Polymarket uses for resolution.

        Args:
            city: City name.
            target_date: Target date.

        Returns:
            High temperature in Fahrenheit, or None if not available.
        """
        periods = self._fetch_forecast_periods(city)

        for period in periods:
            if period.start_time.date() == target_date and period.is_daytime:
                return period.temperature

        return None

    def clear_cache(self) -> None:
        """Clear the forecast cache."""
        self._cache.clear()
