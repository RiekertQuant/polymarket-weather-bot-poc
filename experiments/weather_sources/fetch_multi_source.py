#!/usr/bin/env python3
"""Fetch weather forecasts from multiple sources for comparison.

Tests different weather data sources to find the most accurate for NYC temperature prediction.

Sources tested:
1. Open-Meteo (GFS) - current default
2. Open-Meteo (ECMWF) - European model
3. Open-Meteo (GFS + ECMWF blend via best_match)
4. NWS KLGA observations - ground truth for Polymarket

Usage:
    python experiments/weather_sources/fetch_multi_source.py
"""

import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import requests

# NYC coordinates
NYC_LAT = 40.7128
NYC_LON = -74.0060
NYC_TZ = "America/New_York"

# NWS station for NYC (LaGuardia)
NWS_STATION = "KLGA"
NYC_UTC_OFFSET = -5  # EST


def fetch_open_meteo_gfs(target_date: date) -> Optional[dict]:
    """Fetch GFS forecast from Open-Meteo."""
    url = "https://api.open-meteo.com/v1/gfs"
    params = {
        "latitude": NYC_LAT,
        "longitude": NYC_LON,
        "daily": "temperature_2m_max,temperature_2m_min",
        "timezone": NYC_TZ,
        "start_date": target_date.isoformat(),
        "end_date": target_date.isoformat(),
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        daily = data.get("daily", {})
        if daily.get("temperature_2m_max"):
            return {
                "source": "Open-Meteo GFS",
                "max_temp_c": daily["temperature_2m_max"][0],
                "min_temp_c": daily["temperature_2m_min"][0],
            }
    except Exception as e:
        print(f"  GFS error: {e}")
    return None


def fetch_open_meteo_ecmwf(target_date: date) -> Optional[dict]:
    """Fetch ECMWF forecast from Open-Meteo."""
    url = "https://api.open-meteo.com/v1/ecmwf"
    params = {
        "latitude": NYC_LAT,
        "longitude": NYC_LON,
        "daily": "temperature_2m_max,temperature_2m_min",
        "timezone": NYC_TZ,
        "start_date": target_date.isoformat(),
        "end_date": target_date.isoformat(),
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        daily = data.get("daily", {})
        if daily.get("temperature_2m_max"):
            return {
                "source": "Open-Meteo ECMWF",
                "max_temp_c": daily["temperature_2m_max"][0],
                "min_temp_c": daily["temperature_2m_min"][0],
            }
    except Exception as e:
        print(f"  ECMWF error: {e}")
    return None


def fetch_open_meteo_best_match(target_date: date) -> Optional[dict]:
    """Fetch best_match (blended) forecast from Open-Meteo."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": NYC_LAT,
        "longitude": NYC_LON,
        "daily": "temperature_2m_max,temperature_2m_min",
        "timezone": NYC_TZ,
        "start_date": target_date.isoformat(),
        "end_date": target_date.isoformat(),
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        daily = data.get("daily", {})
        if daily.get("temperature_2m_max"):
            return {
                "source": "Open-Meteo Best Match",
                "max_temp_c": daily["temperature_2m_max"][0],
                "min_temp_c": daily["temperature_2m_min"][0],
            }
    except Exception as e:
        print(f"  Best Match error: {e}")
    return None


def fetch_open_meteo_ensemble(target_date: date) -> Optional[dict]:
    """Fetch ensemble forecast from Open-Meteo (GFS ensemble)."""
    url = "https://ensemble-api.open-meteo.com/v1/ensemble"
    params = {
        "latitude": NYC_LAT,
        "longitude": NYC_LON,
        "daily": "temperature_2m_max",
        "timezone": NYC_TZ,
        "start_date": target_date.isoformat(),
        "end_date": target_date.isoformat(),
        "models": "gfs_seamless",
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        daily = data.get("daily", {})
        # Ensemble returns multiple members
        max_temps = daily.get("temperature_2m_max", [])
        if max_temps:
            # Calculate mean and spread
            mean_temp = sum(max_temps) / len(max_temps)
            min_member = min(max_temps)
            max_member = max(max_temps)
            return {
                "source": "Open-Meteo GFS Ensemble",
                "max_temp_c": mean_temp,
                "ensemble_min": min_member,
                "ensemble_max": max_member,
                "ensemble_spread": max_member - min_member,
                "n_members": len(max_temps),
            }
    except Exception as e:
        print(f"  Ensemble error: {e}")
    return None


def fetch_nws_actual(target_date: date) -> Optional[dict]:
    """Fetch actual observed temperature from NWS KLGA."""
    # Convert local day boundaries to UTC
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
            max_temp_f = round(max_temp_c * 9/5 + 32)  # Polymarket uses rounded F
            return {
                "source": "NWS KLGA (actual)",
                "max_temp_c": max_temp_c,
                "max_temp_f_rounded": max_temp_f,
                "n_observations": len(temps_c),
            }
    except Exception as e:
        print(f"  NWS error: {e}")
    return None


def c_to_f(temp_c: float) -> float:
    """Convert Celsius to Fahrenheit."""
    return temp_c * 9/5 + 32


def main():
    print("=" * 80)
    print("MULTI-SOURCE WEATHER FORECAST COMPARISON")
    print("=" * 80)
    print(f"Location: NYC (LaGuardia - KLGA)")
    print()

    # Test dates: recent past (where we have NWS actuals) + future (forecasts only)
    today = date.today()

    past_dates = [today - timedelta(days=i) for i in range(1, 4)]  # Last 3 days
    future_dates = [today + timedelta(days=i) for i in range(0, 3)]  # Next 3 days

    # Collect data for past dates (with actuals)
    print("=" * 80)
    print("HISTORICAL COMPARISON (with NWS actuals)")
    print("=" * 80)
    print()

    results = []

    for target_date in past_dates:
        print(f"Date: {target_date}")
        print("-" * 40)

        # Fetch from all sources
        nws = fetch_nws_actual(target_date)
        gfs = fetch_open_meteo_gfs(target_date)
        ecmwf = fetch_open_meteo_ecmwf(target_date)
        best = fetch_open_meteo_best_match(target_date)

        if nws:
            actual_f = nws["max_temp_f_rounded"]
            print(f"  NWS KLGA (actual):     {actual_f}°F ({nws['max_temp_c']:.1f}°C)")

            for src, data in [("GFS", gfs), ("ECMWF", ecmwf), ("Best Match", best)]:
                if data:
                    pred_f = c_to_f(data["max_temp_c"])
                    error = pred_f - actual_f
                    print(f"  {src:<20} {pred_f:>5.1f}°F  (error: {error:+.1f}°F)")
                    results.append({
                        "date": target_date.isoformat(),
                        "source": src,
                        "predicted_f": pred_f,
                        "actual_f": actual_f,
                        "error_f": error,
                    })
        else:
            print("  NWS data not available")
        print()

    # Summary statistics
    if results:
        print("=" * 80)
        print("ERROR SUMMARY BY SOURCE")
        print("=" * 80)
        print()

        sources = set(r["source"] for r in results)
        print(f"{'Source':<20} {'Mean Error':>12} {'Abs Error':>12} {'Bias':>10}")
        print("-" * 60)

        for src in sorted(sources):
            src_results = [r for r in results if r["source"] == src]
            errors = [r["error_f"] for r in src_results]
            mean_error = sum(errors) / len(errors)
            mean_abs_error = sum(abs(e) for e in errors) / len(errors)
            bias = "warm" if mean_error > 0 else "cold"
            print(f"{src:<20} {mean_error:>+12.1f}°F {mean_abs_error:>12.1f}°F {bias:>10}")

    # Future forecasts
    print()
    print("=" * 80)
    print("FUTURE FORECASTS (for comparison)")
    print("=" * 80)
    print()

    for target_date in future_dates:
        print(f"Date: {target_date}")
        print("-" * 40)

        gfs = fetch_open_meteo_gfs(target_date)
        ecmwf = fetch_open_meteo_ecmwf(target_date)
        best = fetch_open_meteo_best_match(target_date)
        ensemble = fetch_open_meteo_ensemble(target_date)

        for src, data in [("GFS", gfs), ("ECMWF", ecmwf), ("Best Match", best)]:
            if data:
                pred_f = c_to_f(data["max_temp_c"])
                print(f"  {src:<20} {pred_f:>5.1f}°F")

        if ensemble:
            mean_f = c_to_f(ensemble["max_temp_c"])
            min_f = c_to_f(ensemble["ensemble_min"])
            max_f = c_to_f(ensemble["ensemble_max"])
            print(f"  {'GFS Ensemble':<20} {mean_f:>5.1f}°F  (range: {min_f:.0f}-{max_f:.0f}°F, {ensemble['n_members']} members)")
        print()

    print("=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print()
    print("Run this script over multiple days to build a dataset for bias correction.")
    print("Then train a simple linear model: actual = a*forecast + b")
    print()


if __name__ == "__main__":
    main()
