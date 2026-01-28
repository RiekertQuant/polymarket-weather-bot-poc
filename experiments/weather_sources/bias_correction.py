#!/usr/bin/env python3
"""Build and test bias correction models for weather forecasts.

Collects historical forecast vs actual data and trains simple correction models.

Usage:
    python experiments/weather_sources/bias_correction.py
"""

import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
import statistics

import requests

# NYC coordinates
NYC_LAT = 40.7128
NYC_LON = -74.0060
NYC_TZ = "America/New_York"
NWS_STATION = "KLGA"
NYC_UTC_OFFSET = -5


def fetch_historical_forecast(target_date: date, days_before: int = 1) -> Optional[float]:
    """Fetch what Open-Meteo forecasted N days before target_date.

    Uses the historical forecast API to get past predictions.
    """
    forecast_date = target_date - timedelta(days=days_before)

    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    params = {
        "latitude": NYC_LAT,
        "longitude": NYC_LON,
        "start_date": forecast_date.isoformat(),
        "end_date": target_date.isoformat(),
        "daily": "temperature_2m_max",
        "timezone": NYC_TZ,
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        daily = data.get("daily", {})
        dates = daily.get("time", [])
        temps = daily.get("temperature_2m_max", [])

        # Find target date in results
        target_str = target_date.isoformat()
        for i, d in enumerate(dates):
            if d == target_str and i < len(temps):
                return temps[i]
    except Exception as e:
        print(f"  Historical forecast error: {e}")
    return None


def fetch_nws_actual(target_date: date) -> Optional[float]:
    """Fetch actual max temp from NWS KLGA (returns Fahrenheit rounded)."""
    offset = timedelta(hours=NYC_UTC_OFFSET)
    start_utc = datetime(
        target_date.year, target_date.month, target_date.day,
        tzinfo=timezone.utc,
    ) - offset
    end_utc = start_utc + timedelta(days=1)

    url = f"https://api.weather.gov/stations/{NWS_STATION}/observations"
    params = {
        "start": start_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end": end_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    headers = {
        "User-Agent": "polymarket-weather-bot-experiment",
        "Accept": "application/geo+json",
    }

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        temps_c = []
        for obs in data.get("features", []):
            props = obs.get("properties", {})
            temp = props.get("temperature", {})
            val = temp.get("value")
            if val is not None:
                temps_c.append(val)

        if temps_c:
            max_temp_c = max(temps_c)
            return round(max_temp_c * 9/5 + 32)  # Rounded Fahrenheit
    except Exception as e:
        print(f"  NWS error for {target_date}: {e}")
    return None


def c_to_f(temp_c: float) -> float:
    return temp_c * 9/5 + 32


def collect_data(n_days: int = 14) -> list[dict]:
    """Collect forecast vs actual pairs for the past N days."""
    print(f"Collecting {n_days} days of forecast vs actual data...")
    print()

    data = []
    today = date.today()

    for days_ago in range(1, n_days + 1):
        target_date = today - timedelta(days=days_ago)

        # Get what was forecasted 1 day before
        forecast_c = fetch_historical_forecast(target_date, days_before=1)
        actual_f = fetch_nws_actual(target_date)

        if forecast_c is not None and actual_f is not None:
            forecast_f = c_to_f(forecast_c)
            error = forecast_f - actual_f

            data.append({
                "date": target_date.isoformat(),
                "forecast_f": forecast_f,
                "actual_f": actual_f,
                "error_f": error,
            })
            print(f"  {target_date}: forecast={forecast_f:.1f}°F, actual={actual_f}°F, error={error:+.1f}°F")
        else:
            print(f"  {target_date}: missing data (forecast={forecast_c}, actual={actual_f})")

    return data


def train_bias_correction(data: list[dict]) -> dict:
    """Train simple bias correction models."""
    if len(data) < 3:
        return {"error": "Not enough data"}

    forecasts = [d["forecast_f"] for d in data]
    actuals = [d["actual_f"] for d in data]
    errors = [d["error_f"] for d in data]

    # Model 1: Simple additive bias (actual = forecast + bias)
    mean_error = statistics.mean(errors)
    additive_bias = -mean_error  # correction to add

    # Model 2: Linear regression (actual = a*forecast + b)
    n = len(data)
    sum_x = sum(forecasts)
    sum_y = sum(actuals)
    sum_xy = sum(f * a for f, a in zip(forecasts, actuals))
    sum_x2 = sum(f * f for f in forecasts)

    # Calculate slope (a) and intercept (b)
    denom = n * sum_x2 - sum_x * sum_x
    if denom != 0:
        a = (n * sum_xy - sum_x * sum_y) / denom
        b = (sum_y - a * sum_x) / n
    else:
        a, b = 1.0, 0.0

    # Calculate corrected predictions and errors
    additive_corrected = [f + additive_bias for f in forecasts]
    linear_corrected = [a * f + b for f in forecasts]

    additive_errors = [c - a for c, a in zip(additive_corrected, actuals)]
    linear_errors = [c - a for c, a in zip(linear_corrected, actuals)]

    original_mae = statistics.mean(abs(e) for e in errors)
    additive_mae = statistics.mean(abs(e) for e in additive_errors)
    linear_mae = statistics.mean(abs(e) for e in linear_errors)

    return {
        "n_samples": n,
        "original": {
            "mean_error": mean_error,
            "mae": original_mae,
            "std_error": statistics.stdev(errors) if len(errors) > 1 else 0,
        },
        "additive_correction": {
            "bias": additive_bias,
            "formula": f"corrected = forecast + {additive_bias:.2f}",
            "mae": additive_mae,
        },
        "linear_correction": {
            "slope": a,
            "intercept": b,
            "formula": f"corrected = {a:.3f} * forecast + {b:.2f}",
            "mae": linear_mae,
        },
    }


def main():
    print("=" * 80)
    print("BIAS CORRECTION MODEL TRAINING")
    print("=" * 80)
    print(f"Location: NYC (KLGA)")
    print(f"Forecast source: Open-Meteo (day-before forecast)")
    print(f"Ground truth: NWS KLGA observations")
    print()

    # Collect data
    data = collect_data(n_days=14)

    if len(data) < 3:
        print("\nNot enough data to train models. NWS may not have historical data this far back.")
        return

    print()
    print("=" * 80)
    print("BIAS CORRECTION MODELS")
    print("=" * 80)
    print()

    models = train_bias_correction(data)

    print(f"Training samples: {models['n_samples']}")
    print()

    print("ORIGINAL (no correction):")
    print(f"  Mean error:  {models['original']['mean_error']:+.1f}°F")
    print(f"  MAE:         {models['original']['mae']:.1f}°F")
    print(f"  Std error:   {models['original']['std_error']:.1f}°F")
    print()

    print("ADDITIVE CORRECTION:")
    print(f"  Formula:     {models['additive_correction']['formula']}")
    print(f"  MAE:         {models['additive_correction']['mae']:.1f}°F")
    print(f"  Improvement: {(1 - models['additive_correction']['mae']/models['original']['mae'])*100:.0f}%")
    print()

    print("LINEAR CORRECTION:")
    print(f"  Formula:     {models['linear_correction']['formula']}")
    print(f"  MAE:         {models['linear_correction']['mae']:.1f}°F")
    print(f"  Improvement: {(1 - models['linear_correction']['mae']/models['original']['mae'])*100:.0f}%")
    print()

    # Save models
    output_path = Path("experiments/weather_sources/bias_models.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump({
            "trained_at": datetime.now().isoformat(),
            "n_samples": models["n_samples"],
            "additive_bias": models["additive_correction"]["bias"],
            "linear_slope": models["linear_correction"]["slope"],
            "linear_intercept": models["linear_correction"]["intercept"],
        }, f, indent=2)

    print(f"Models saved to: {output_path}")
    print()

    # Test on recent data
    print("=" * 80)
    print("VALIDATION ON RECENT DATA")
    print("=" * 80)
    print()

    bias = models["additive_correction"]["bias"]
    a = models["linear_correction"]["slope"]
    b = models["linear_correction"]["intercept"]

    print(f"{'Date':<12} {'Forecast':>10} {'Actual':>8} {'+ Bias':>10} {'Linear':>10}")
    print("-" * 55)

    for d in data[-5:]:  # Last 5 days
        forecast = d["forecast_f"]
        actual = d["actual_f"]
        additive = forecast + bias
        linear = a * forecast + b
        print(f"{d['date']:<12} {forecast:>10.1f} {actual:>8.0f} {additive:>10.1f} {linear:>10.1f}")


if __name__ == "__main__":
    main()
