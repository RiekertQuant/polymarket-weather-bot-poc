"""Forecast logger for collecting ML training data.

Saves daily forecasts to compare with actual outcomes later.
This data is used to improve the ML calibration model.
"""

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class ForecastLogger:
    """Logs forecasts for future accuracy tracking."""

    def __init__(self, log_dir: Path = Path("data/forecast_history")):
        """Initialize forecast logger.

        Args:
            log_dir: Directory to store forecast logs.
        """
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log_forecast(
        self,
        city: str,
        target_date: date,
        forecast_temp_c: float,
        source: str,
        bias_correction: float = 0.0,
        ensemble_spread: Optional[float] = None,
        weather_regime: str = "unknown",
        sigma_used: float = 2.0,
    ) -> None:
        """Log a forecast for future accuracy tracking.

        Args:
            city: City name.
            target_date: Date the forecast is for.
            forecast_temp_c: Forecasted max temperature (Celsius).
            source: Forecast source ("nws", "open_meteo", "ensemble").
            bias_correction: Bias correction applied (Celsius).
            ensemble_spread: Spread between sources if ensemble.
            weather_regime: Detected weather regime.
            sigma_used: Sigma value used for probability calculation.
        """
        today = date.today()
        days_ahead = (target_date - today).days

        # Convert to Fahrenheit for easier comparison with Polymarket
        forecast_temp_f = forecast_temp_c * 9/5 + 32
        corrected_temp_f = (forecast_temp_c + bias_correction) * 9/5 + 32

        entry = {
            "logged_at": datetime.utcnow().isoformat() + "Z",
            "logged_date": today.isoformat(),
            "city": city,
            "target_date": target_date.isoformat(),
            "days_ahead": days_ahead,
            "forecast_temp_c": round(forecast_temp_c, 2),
            "forecast_temp_f": round(forecast_temp_f, 1),
            "bias_correction_c": round(bias_correction, 2),
            "corrected_temp_f": round(corrected_temp_f, 1),
            "source": source,
            "ensemble_spread": round(ensemble_spread, 2) if ensemble_spread else None,
            "weather_regime": weather_regime,
            "sigma_used": round(sigma_used, 2),
            "actual_temp_f": None,  # Filled in later by resolver
            "error_f": None,  # Filled in later
        }

        # Append to daily log file
        log_file = self.log_dir / f"forecasts_{today.isoformat()}.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

        logger.debug(f"Logged forecast: {city} {target_date} -> {corrected_temp_f:.1f}°F")

    def update_with_actual(
        self,
        city: str,
        target_date: date,
        actual_temp_f: int,
    ) -> int:
        """Update logged forecasts with actual temperature.

        Args:
            city: City name.
            target_date: Date the actual is for.
            actual_temp_f: Actual max temperature (Fahrenheit).

        Returns:
            Number of entries updated.
        """
        updated = 0

        # Search all log files for matching entries
        for log_file in self.log_dir.glob("forecasts_*.jsonl"):
            entries = []
            modified = False

            with open(log_file) as f:
                for line in f:
                    entry = json.loads(line.strip())
                    if (
                        entry["city"] == city
                        and entry["target_date"] == target_date.isoformat()
                        and entry["actual_temp_f"] is None
                    ):
                        entry["actual_temp_f"] = actual_temp_f
                        entry["error_f"] = round(entry["corrected_temp_f"] - actual_temp_f, 1)
                        modified = True
                        updated += 1
                    entries.append(entry)

            if modified:
                with open(log_file, "w") as f:
                    for entry in entries:
                        f.write(json.dumps(entry) + "\n")

        if updated:
            logger.info(f"Updated {updated} forecast entries with actual: {city} {target_date} = {actual_temp_f}°F")

        return updated

    def get_recent_errors(self, city: str, days: int = 14) -> list[dict]:
        """Get recent forecast errors for a city.

        Args:
            city: City name.
            days: Number of days to look back.

        Returns:
            List of entries with actual values and errors.
        """
        errors = []
        cutoff = date.today().isoformat()

        for log_file in sorted(self.log_dir.glob("forecasts_*.jsonl"), reverse=True):
            # Only check recent files
            file_date = log_file.stem.replace("forecasts_", "")
            if file_date < cutoff:
                days -= 1
                if days < 0:
                    break

            try:
                with open(log_file) as f:
                    for line in f:
                        entry = json.loads(line.strip())
                        if (
                            entry["city"] == city
                            and entry["actual_temp_f"] is not None
                        ):
                            errors.append(entry)
            except Exception:
                continue

        return errors

    def calculate_bias(self, city: str, days: int = 14) -> Optional[float]:
        """Calculate recent forecast bias for a city.

        Args:
            city: City name.
            days: Number of days to analyze.

        Returns:
            Mean error (positive = forecast runs warm) or None if no data.
        """
        errors = self.get_recent_errors(city, days)
        if not errors:
            return None

        error_values = [e["error_f"] for e in errors if e["error_f"] is not None]
        if not error_values:
            return None

        return sum(error_values) / len(error_values)
