#!/usr/bin/env python3
"""Generate bias-corrected forecasts for upcoming NYC weather markets.

Applies trained bias correction to improve forecast accuracy.

Usage:
    python experiments/weather_sources/corrected_forecast.py
"""

import json
from datetime import date, timedelta
from pathlib import Path

import requests

NYC_LAT = 40.7128
NYC_LON = -74.0060
NYC_TZ = "America/New_York"


def load_bias_model() -> dict:
    """Load trained bias correction model."""
    path = Path("experiments/weather_sources/bias_models.json")
    if path.exists():
        with open(path) as f:
            return json.load(f)
    # Default if no model trained
    return {"additive_bias": 1.15, "linear_slope": 1.0, "linear_intercept": 0.0}


def fetch_forecast(target_date: date) -> float | None:
    """Fetch Open-Meteo forecast for target date."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": NYC_LAT,
        "longitude": NYC_LON,
        "daily": "temperature_2m_max",
        "timezone": NYC_TZ,
        "start_date": target_date.isoformat(),
        "end_date": target_date.isoformat(),
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        temps = data.get("daily", {}).get("temperature_2m_max", [])
        if temps:
            return temps[0] * 9/5 + 32  # Convert to F
    except Exception as e:
        print(f"Error: {e}")
    return None


def main():
    model = load_bias_model()
    bias = model.get("additive_bias", 1.15)

    print("=" * 70)
    print("BIAS-CORRECTED NYC TEMPERATURE FORECASTS")
    print("=" * 70)
    print(f"Correction: +{bias:.1f}°F (trained on NWS KLGA actuals)")
    print()

    today = date.today()

    print(f"{'Date':<12} {'Raw Forecast':>14} {'Corrected':>12} {'Rounded':>10}")
    print("-" * 50)

    for days_ahead in range(0, 5):
        target = today + timedelta(days=days_ahead)
        forecast = fetch_forecast(target)

        if forecast:
            corrected = forecast + bias
            rounded = round(corrected)
            print(f"{target!s:<12} {forecast:>14.1f}°F {corrected:>12.1f}°F {rounded:>10}°F")

    print()
    print("=" * 70)
    print("POLYMARKET MARKET IMPLICATIONS")
    print("=" * 70)
    print()

    # Show specific analysis for tomorrow
    tomorrow = today + timedelta(days=1)
    forecast = fetch_forecast(tomorrow)

    if forecast:
        corrected = forecast + bias
        rounded = round(corrected)

        print(f"Tomorrow ({tomorrow}):")
        print(f"  Raw forecast:       {forecast:.1f}°F")
        print(f"  Bias-corrected:     {corrected:.1f}°F")
        print(f"  Expected actual:    ~{rounded}°F")
        print()

        # Show which bands are likely
        print("  Band probabilities (rough estimates):")
        bands = [
            (rounded - 4, rounded - 3, "cold"),
            (rounded - 2, rounded - 1, "slightly cold"),
            (rounded, rounded + 1, "expected"),
            (rounded + 2, rounded + 3, "slightly warm"),
            (rounded + 4, rounded + 5, "warm"),
        ]

        for low, high, desc in bands:
            # Simple probability based on ~2°F standard error
            if desc == "expected":
                prob = "~35%"
            elif "slightly" in desc:
                prob = "~25%"
            else:
                prob = "~7%"
            print(f"    {low}-{high}°F: {prob} ({desc})")


if __name__ == "__main__":
    main()
