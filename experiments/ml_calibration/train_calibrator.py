#!/usr/bin/env python3
"""Train ML calibration model for weather forecast probabilities.

This model learns to correct probability estimates based on:
- Forecast temperature
- Days ahead (forecast horizon)
- Recent forecast errors (bias)
- Temperature band (extreme vs moderate)
- Season/month

Usage:
    python experiments/ml_calibration/train_calibrator.py
"""

import json
import pickle
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import requests
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import brier_score_loss, log_loss
from scipy import stats

# Constants
NYC_LAT = 40.7128
NYC_LON = -74.0060
KLGA_STATION = "KLGA"


def fetch_historical_forecasts(days_back: int = 30) -> list[dict]:
    """Fetch historical Open-Meteo forecasts from archive.

    Note: Open-Meteo historical API provides what the forecast WAS
    on a given date, not reanalysis data.
    """
    # For now, we'll simulate by using current forecast errors
    # In production, you'd store daily forecasts and compare to actuals
    print("Note: Using simulated historical data based on recent patterns")
    return []


def fetch_nws_actuals(days_back: int = 30) -> dict[date, int]:
    """Fetch actual high temperatures from NWS KLGA observations."""
    actuals = {}

    url = f"https://api.weather.gov/stations/{KLGA_STATION}/observations"
    headers = {"User-Agent": "polymarket-weather-bot"}

    try:
        # Get recent observations
        resp = requests.get(url, params={"limit": 500}, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        # Group by date and find daily max
        daily_temps: dict[date, list[float]] = {}

        for obs in data.get("features", []):
            props = obs.get("properties", {})
            timestamp = props.get("timestamp")
            temp_c = props.get("temperature", {}).get("value")

            if timestamp and temp_c is not None:
                obs_date = date.fromisoformat(timestamp[:10])
                if obs_date not in daily_temps:
                    daily_temps[obs_date] = []
                daily_temps[obs_date].append(temp_c)

        # Convert to daily max in Fahrenheit (rounded, like Polymarket)
        for d, temps in daily_temps.items():
            max_c = max(temps)
            max_f = round(max_c * 9/5 + 32)
            actuals[d] = max_f

    except Exception as e:
        print(f"Error fetching NWS actuals: {e}")

    return actuals


def fetch_open_meteo_forecast(target_date: date) -> float | None:
    """Fetch Open-Meteo forecast for a specific date."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": NYC_LAT,
        "longitude": NYC_LON,
        "daily": "temperature_2m_max",
        "timezone": "America/New_York",
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
        print(f"Error fetching Open-Meteo: {e}")

    return None


def fetch_nws_forecast(target_date: date) -> float | None:
    """Fetch NWS forecast for a specific date."""
    url = "https://api.weather.gov/gridpoints/OKX/33,37/forecast"
    headers = {"User-Agent": "polymarket-weather-bot"}

    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        for period in data.get("properties", {}).get("periods", []):
            if period.get("isDaytime"):
                period_date = date.fromisoformat(period["startTime"][:10])
                if period_date == target_date:
                    return float(period["temperature"])
    except Exception as e:
        print(f"Error fetching NWS forecast: {e}")

    return None


def generate_training_data() -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Generate training data from historical forecasts and actuals.

    Returns:
        X: Feature matrix
        y: Binary outcomes (1 if actual >= threshold, 0 otherwise)
        metadata: List of dicts with details for each sample
    """
    print("Fetching NWS actual temperatures...")
    actuals = fetch_nws_actuals(days_back=14)

    # Need at least 5 days for meaningful training
    if len(actuals) < 5:
        print(f"Only {len(actuals)} days of actuals, supplementing with synthetic data")
        return generate_synthetic_data()

    print(f"Found {len(actuals)} days of actual data")

    # Generate samples for various thresholds
    X_list = []
    y_list = []
    metadata = []

    # For each day we have actuals, create training samples
    for actual_date, actual_temp in sorted(actuals.items()):
        # Skip today (incomplete)
        if actual_date >= date.today():
            continue

        # Get what the forecast would have been
        # Note: In production, you'd store daily forecasts
        # For now, assume forecast = actual + some error (cold bias)
        forecast_bias = np.random.normal(-1.5, 1.0)  # Cold bias
        forecast_error = np.random.normal(0, 2.0)
        forecast_temp = actual_temp + forecast_bias + forecast_error

        # Generate samples for different thresholds
        for threshold in range(10, 50, 2):  # 10°F to 48°F in 2°F steps
            # Calculate raw probability using normal CDF
            sigma_f = 3.6  # ~2°C in Fahrenheit
            z_score = (threshold - forecast_temp) / sigma_f
            p_raw = 1.0 - stats.norm.cdf(z_score)  # P(actual >= threshold)

            # Outcome: did actual meet or exceed threshold?
            outcome = 1 if actual_temp >= threshold else 0

            # Features
            features = [
                p_raw,                              # Raw probability estimate
                forecast_temp,                      # Forecast temperature
                threshold,                          # Threshold temperature
                forecast_temp - threshold,          # Distance from threshold
                abs(forecast_temp - threshold),    # Absolute distance
                1 if threshold < 20 else 0,        # Extreme cold flag
                1 if threshold > 40 else 0,        # Warm flag
                actual_date.month,                  # Month (seasonality)
                actual_date.weekday(),              # Day of week (minor)
                sigma_f,                            # Uncertainty used
            ]

            X_list.append(features)
            y_list.append(outcome)
            metadata.append({
                "date": actual_date.isoformat(),
                "forecast": forecast_temp,
                "actual": actual_temp,
                "threshold": threshold,
                "p_raw": p_raw,
                "outcome": outcome,
            })

    return np.array(X_list), np.array(y_list), metadata


def generate_synthetic_data() -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Generate synthetic training data based on observed bias patterns."""
    print("Generating synthetic training data...")

    # Based on our observations:
    # - Open-Meteo has ~1-3°F cold bias
    # - NWS has ~0-1°F cold bias
    # - Errors are roughly normally distributed with std ~2.5°F

    np.random.seed(42)
    n_days = 60

    X_list = []
    y_list = []
    metadata = []

    for day_offset in range(n_days):
        sim_date = date.today() - timedelta(days=day_offset + 1)

        # Simulate "true" temperature based on season
        # January average high in NYC ~39°F, with day-to-day variation
        base_temp = 25 + 15 * np.sin(2 * np.pi * (sim_date.timetuple().tm_yday - 30) / 365)
        actual_temp = base_temp + np.random.normal(0, 8)  # Day-to-day variation

        # Simulate forecast with bias
        forecast_bias = np.random.normal(-1.5, 1.0)  # Cold bias
        forecast_error = np.random.normal(0, 2.5)
        forecast_temp = actual_temp + forecast_bias + forecast_error

        # Generate samples for different thresholds
        for threshold in range(10, 50, 2):
            sigma_f = 3.6
            z_score = (threshold - forecast_temp) / sigma_f
            p_raw = 1.0 - stats.norm.cdf(z_score)

            outcome = 1 if actual_temp >= threshold else 0

            features = [
                p_raw,
                forecast_temp,
                threshold,
                forecast_temp - threshold,
                abs(forecast_temp - threshold),
                1 if threshold < 20 else 0,
                1 if threshold > 40 else 0,
                sim_date.month,
                sim_date.weekday(),
                sigma_f,
            ]

            X_list.append(features)
            y_list.append(outcome)
            metadata.append({
                "date": sim_date.isoformat(),
                "forecast": round(forecast_temp, 1),
                "actual": round(actual_temp, 1),
                "threshold": threshold,
                "p_raw": round(p_raw, 4),
                "outcome": outcome,
            })

    print(f"Generated {len(X_list)} training samples")
    return np.array(X_list), np.array(y_list), metadata


def train_models(X: np.ndarray, y: np.ndarray) -> dict:
    """Train and evaluate multiple calibration models."""

    feature_names = [
        "p_raw", "forecast_temp", "threshold", "temp_diff",
        "abs_temp_diff", "is_extreme_cold", "is_warm",
        "month", "weekday", "sigma"
    ]

    print(f"\nTraining on {len(X)} samples...")
    print(f"Positive rate: {y.mean():.1%}")

    # Models to try
    models = {
        "logistic": LogisticRegression(max_iter=1000),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        ),
    }

    results = {}

    # Use stratified k-fold to ensure both classes in each fold
    cv = StratifiedKFold(n_splits=min(5, len(np.unique(y)) * 2), shuffle=True, random_state=42)

    for name, model in models.items():
        print(f"\n{name.upper()}:")

        # Cross-validation
        try:
            scores = cross_val_score(model, X, y, cv=cv, scoring="neg_brier_score")
        except Exception as e:
            print(f"  CV failed: {e}")
            scores = np.array([np.nan])
        brier = -scores.mean()

        # Fit on all data for final model
        model.fit(X, y)

        # Get predictions
        y_pred_proba = model.predict_proba(X)[:, 1]

        # Calculate metrics
        train_brier = brier_score_loss(y, y_pred_proba)
        train_log_loss = log_loss(y, y_pred_proba)

        print(f"  CV Brier Score: {brier:.4f}")
        print(f"  Train Brier:    {train_brier:.4f}")
        print(f"  Train Log Loss: {train_log_loss:.4f}")

        # Feature importance for tree models
        if hasattr(model, "feature_importances_"):
            print("  Feature Importance:")
            importances = list(zip(feature_names, model.feature_importances_))
            for feat, imp in sorted(importances, key=lambda x: -x[1])[:5]:
                print(f"    {feat}: {imp:.3f}")

        results[name] = {
            "model": model,
            "cv_brier": brier,
            "train_brier": train_brier,
        }

    # Compare to baseline (using raw probability)
    baseline_brier = brier_score_loss(y, X[:, 0])  # p_raw is first feature
    print(f"\nBASELINE (raw probability):")
    print(f"  Brier Score: {baseline_brier:.4f}")

    return results


def save_best_model(results: dict, output_dir: Path) -> str:
    """Save the best performing model."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find best model by CV Brier score
    best_name = min(results.keys(), key=lambda k: results[k]["cv_brier"])
    best_model = results[best_name]["model"]

    print(f"\nBest model: {best_name}")

    # Save model
    model_path = output_dir / "calibrator.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)

    # Save metadata
    meta_path = output_dir / "calibrator_meta.json"
    meta = {
        "model_type": best_name,
        "cv_brier_score": results[best_name]["cv_brier"],
        "train_brier_score": results[best_name]["train_brier"],
        "feature_names": [
            "p_raw", "forecast_temp", "threshold", "temp_diff",
            "abs_temp_diff", "is_extreme_cold", "is_warm",
            "month", "weekday", "sigma"
        ],
        "trained_date": date.today().isoformat(),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved model to {model_path}")
    print(f"Saved metadata to {meta_path}")

    return best_name


def demonstrate_calibration(results: dict, X: np.ndarray, metadata: list[dict]):
    """Show how calibration changes probability estimates."""
    print("\n" + "=" * 70)
    print("CALIBRATION DEMONSTRATION")
    print("=" * 70)

    best_name = min(results.keys(), key=lambda k: results[k]["cv_brier"])
    model = results[best_name]["model"]

    # Show a few examples
    print("\nSample predictions:")
    print(f"{'Forecast':<10} {'Threshold':<12} {'P_raw':<10} {'P_calibrated':<14} {'Actual':<8} {'Outcome'}")
    print("-" * 70)

    indices = np.random.choice(len(X), min(10, len(X)), replace=False)
    for i in sorted(indices):
        p_raw = X[i, 0]
        try:
            proba = model.predict_proba(X[i:i+1])
            p_cal = proba[0, 1] if proba.shape[1] > 1 else proba[0, 0]
        except Exception:
            p_cal = p_raw  # Fallback to raw
        meta = metadata[i]

        outcome_str = "WIN" if meta['outcome'] else "LOSS"
        print(f"{meta['forecast']:>8.1f}F {meta['threshold']:>10}F {p_raw:>8.1%} {p_cal:>12.1%} {meta['actual']:>6.1f}F {outcome_str:>6}")


def main():
    print("=" * 70)
    print("ML CALIBRATION MODEL TRAINING")
    print("=" * 70)

    # Generate training data
    X, y, metadata = generate_training_data()

    if len(X) == 0:
        print("No training data available!")
        return

    # Train models
    results = train_models(X, y)

    # Save best model
    output_dir = Path("experiments/ml_calibration")
    best_name = save_best_model(results, output_dir)

    # Demonstrate calibration
    demonstrate_calibration(results, X, metadata)

    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("1. Integrate calibrator into WeatherProbabilityEngine")
    print("2. Collect more real forecast data over time")
    print("3. Retrain periodically as more data becomes available")
    print("=" * 70)


if __name__ == "__main__":
    main()
