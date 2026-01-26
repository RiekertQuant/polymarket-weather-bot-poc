"""Open-Meteo API client for weather forecasts."""

import logging
from dataclasses import dataclass
from datetime import date
from typing import Optional

import requests

logger = logging.getLogger(__name__)


@dataclass
class CityCoordinates:
    """Geographic coordinates for a city."""

    name: str
    latitude: float
    longitude: float
    timezone: str


# Pre-defined coordinates for supported cities
CITY_COORDS = {
    "New York City": CityCoordinates(
        name="New York City",
        latitude=40.7128,
        longitude=-74.0060,
        timezone="America/New_York",
    ),
    "London": CityCoordinates(
        name="London",
        latitude=51.5074,
        longitude=-0.1278,
        timezone="Europe/London",
    ),
    "Seoul": CityCoordinates(
        name="Seoul",
        latitude=37.5665,
        longitude=126.9780,
        timezone="Asia/Seoul",
    ),
}


@dataclass
class DailyForecast:
    """Daily weather forecast data."""

    date: date
    temperature_max: float  # Celsius
    temperature_min: float  # Celsius
    temperature_mean: float  # Celsius
    precipitation_probability: Optional[float] = None


class OpenMeteoClient:
    """Client for Open-Meteo free weather API."""

    BASE_URL = "https://api.open-meteo.com/v1/forecast"

    def __init__(self, timeout: int = 10):
        """Initialize client.

        Args:
            timeout: Request timeout in seconds.
        """
        self.timeout = timeout
        self.session = requests.Session()

    def get_city_coords(self, city: str) -> Optional[CityCoordinates]:
        """Get coordinates for a city.

        Args:
            city: City name.

        Returns:
            CityCoordinates if found.
        """
        return CITY_COORDS.get(city)

    def get_forecast(
        self,
        city: str,
        days: int = 7,
    ) -> list[DailyForecast]:
        """Fetch daily forecast for a city.

        Args:
            city: City name.
            days: Number of forecast days (1-16).

        Returns:
            List of DailyForecast objects.
        """
        coords = self.get_city_coords(city)
        if coords is None:
            logger.warning(f"Unknown city: {city}")
            return []

        params = {
            "latitude": coords.latitude,
            "longitude": coords.longitude,
            "daily": "temperature_2m_max,temperature_2m_min",
            "timezone": coords.timezone,
            "forecast_days": min(days, 16),
        }

        try:
            response = self.session.get(
                self.BASE_URL,
                params=params,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Weather API request failed: {e}")
            return []

        forecasts = []
        daily = data.get("daily", {})
        dates = daily.get("time", [])
        max_temps = daily.get("temperature_2m_max", [])
        min_temps = daily.get("temperature_2m_min", [])

        for i, date_str in enumerate(dates):
            if i >= len(max_temps) or i >= len(min_temps):
                break

            temp_max = max_temps[i]
            temp_min = min_temps[i]

            forecasts.append(DailyForecast(
                date=date.fromisoformat(date_str),
                temperature_max=temp_max,
                temperature_min=temp_min,
                temperature_mean=(temp_max + temp_min) / 2,
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
        days_ahead = (target_date - date.today()).days + 1
        if days_ahead < 1 or days_ahead > 16:
            logger.warning(f"Date {target_date} outside forecast range")
            return None

        forecasts = self.get_forecast(city, days=days_ahead)
        for forecast in forecasts:
            if forecast.date == target_date:
                return forecast

        return None


class MockOpenMeteoClient(OpenMeteoClient):
    """Mock client for testing without network calls."""

    def __init__(self, mock_forecasts: Optional[dict[str, list[DailyForecast]]] = None):
        """Initialize mock client.

        Args:
            mock_forecasts: Dictionary mapping city names to forecast lists.
        """
        super().__init__()
        self._mock_forecasts = mock_forecasts or {}

    def set_forecast(self, city: str, forecasts: list[DailyForecast]) -> None:
        """Set mock forecast for a city."""
        self._mock_forecasts[city] = forecasts

    def get_forecast(self, city: str, days: int = 7) -> list[DailyForecast]:
        """Return mock forecast."""
        return self._mock_forecasts.get(city, [])[:days]
