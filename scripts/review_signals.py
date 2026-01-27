#!/usr/bin/env python3
"""Review historical signals and track outcomes.

This script checks past signals against actual weather data
to see how the bot's recommendations would have performed.
"""

import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.weather.open_meteo import OpenMeteoClient


def load_signals(signals_dir: Path = Path("data/signals")) -> list[dict]:
    """Load all historical signals.

    Args:
        signals_dir: Directory containing signal files.

    Returns:
        List of all signals across all scan dates.
    """
    all_signals = []

    if not signals_dir.exists():
        print(f"No signals directory found at {signals_dir}")
        return []

    for filepath in sorted(signals_dir.glob("signals_*.json")):
        with open(filepath) as f:
            data = json.load(f)

        scan_date = data.get("scan_date")
        for signal in data.get("signals", []):
            signal["scan_date"] = scan_date
            all_signals.append(signal)

    return all_signals


def check_outcome(signal: dict, weather_client: OpenMeteoClient) -> dict:
    """Check actual outcome of a signal.

    Args:
        signal: Signal dictionary.
        weather_client: Weather client for fetching actuals.

    Returns:
        Signal with outcome data added.
    """
    target_date = date.fromisoformat(signal["target_date"])

    # Can't check future dates
    if target_date >= date.today():
        signal["outcome"] = "pending"
        signal["actual_temp"] = None
        signal["outcome_correct"] = None
        signal["pnl"] = None
        return signal

    # Get actual temperature
    try:
        from src.backtest.weather_history import WeatherHistoryCollector
        collector = WeatherHistoryCollector()
        actual = collector.get_actual_weather(signal["city"], target_date)

        if actual:
            signal["actual_temp"] = actual.actual_max_temp
            threshold = signal["threshold_celsius"]

            # Did YES win?
            yes_won = actual.actual_max_temp >= threshold
            signal["yes_won"] = yes_won

            # We always signal BUY YES
            signal["outcome_correct"] = yes_won
            signal["outcome"] = "win" if yes_won else "loss"

            # Calculate P&L (assuming we bought at market price)
            entry_price = signal["market_price"]
            cost = signal.get("recommended_size_usd", 5.0)
            shares = cost / entry_price

            if yes_won:
                signal["pnl"] = shares * (1.0 - entry_price)  # Win: get $1 per share
            else:
                signal["pnl"] = -cost  # Loss: lose cost basis

            signal["return_pct"] = (signal["pnl"] / cost) * 100
        else:
            signal["outcome"] = "no_data"
            signal["actual_temp"] = None

    except Exception as e:
        signal["outcome"] = "error"
        signal["error"] = str(e)

    return signal


def generate_report(signals: list[dict]) -> str:
    """Generate performance report.

    Args:
        signals: List of signals with outcomes.

    Returns:
        Formatted report string.
    """
    lines = [
        "=" * 70,
        "SIGNAL PERFORMANCE REVIEW",
        "=" * 70,
        "",
    ]

    # Separate by outcome status
    resolved = [s for s in signals if s.get("outcome") in ["win", "loss"]]
    pending = [s for s in signals if s.get("outcome") == "pending"]

    lines.append(f"Total Signals: {len(signals)}")
    lines.append(f"Resolved: {len(resolved)}")
    lines.append(f"Pending: {len(pending)}")
    lines.append("")

    if resolved:
        wins = [s for s in resolved if s["outcome"] == "win"]
        losses = [s for s in resolved if s["outcome"] == "loss"]

        win_rate = len(wins) / len(resolved) * 100 if resolved else 0
        total_pnl = sum(s.get("pnl", 0) for s in resolved)
        avg_pnl = total_pnl / len(resolved) if resolved else 0

        lines.extend([
            "-" * 70,
            "PERFORMANCE SUMMARY",
            "-" * 70,
            "",
            f"  Wins:      {len(wins)}",
            f"  Losses:    {len(losses)}",
            f"  Win Rate:  {win_rate:.1f}%",
            "",
            f"  Total P&L: ${total_pnl:+.2f}",
            f"  Avg P&L:   ${avg_pnl:+.2f}",
            "",
        ])

        # By city
        cities = set(s["city"] for s in resolved)
        if len(cities) > 1:
            lines.extend([
                "-" * 70,
                "BY CITY",
                "-" * 70,
                "",
            ])
            for city in sorted(cities):
                city_signals = [s for s in resolved if s["city"] == city]
                city_wins = len([s for s in city_signals if s["outcome"] == "win"])
                city_pnl = sum(s.get("pnl", 0) for s in city_signals)
                city_wr = city_wins / len(city_signals) * 100 if city_signals else 0
                lines.append(f"  {city:20s} {len(city_signals):3d} trades | {city_wr:5.1f}% WR | ${city_pnl:+8.2f}")
            lines.append("")

        # Trade log
        lines.extend([
            "-" * 70,
            "RESOLVED TRADES",
            "-" * 70,
            "",
            f"{'Date':<12} {'City':<15} {'Thresh':<8} {'Price':<8} {'P(mod)':<8} {'Actual':<8} {'Result':<8} {'P&L':<10}",
            "-" * 70,
        ])

        for s in sorted(resolved, key=lambda x: x.get("target_date", "")):
            lines.append(
                f"{s.get('target_date', 'N/A'):<12} "
                f"{s['city']:<15} "
                f"{s['threshold_celsius']:>5.1f}C  "
                f"${s['market_price']:<6.3f} "
                f"{s['p_model']:<8.2f} "
                f"{s.get('actual_temp', 0):>5.1f}C  "
                f"{s['outcome'].upper():<8} "
                f"${s.get('pnl', 0):+.2f}"
            )

        lines.append("")

    if pending:
        lines.extend([
            "-" * 70,
            "PENDING SIGNALS",
            "-" * 70,
            "",
        ])
        for s in sorted(pending, key=lambda x: x.get("target_date", "")):
            lines.append(
                f"  {s.get('target_date', 'N/A')} | {s['city']} | "
                f"{s['threshold_celsius']}C @ ${s['market_price']:.3f} | "
                f"p={s['p_model']:.2f}"
            )
        lines.append("")

    lines.append("=" * 70)

    return "\n".join(lines)


def save_outcomes(signals: list[dict], output_path: Path = Path("data/signals/outcomes.json")):
    """Save outcomes to file.

    Args:
        signals: Signals with outcomes.
        output_path: Output file path.
    """
    # Summary stats
    resolved = [s for s in signals if s.get("outcome") in ["win", "loss"]]
    wins = len([s for s in resolved if s["outcome"] == "win"])

    data = {
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "total_signals": len(signals),
        "resolved": len(resolved),
        "wins": wins,
        "losses": len(resolved) - wins,
        "win_rate": wins / len(resolved) if resolved else 0,
        "total_pnl": sum(s.get("pnl", 0) for s in resolved),
        "signals": signals,
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved outcomes to {output_path}")


def main():
    print("Loading historical signals...")
    signals = load_signals()

    if not signals:
        print("No signals found. Run the daily scanner first.")
        return 1

    print(f"Found {len(signals)} signals")
    print("Checking outcomes against actual weather data...")

    weather_client = OpenMeteoClient()

    for i, signal in enumerate(signals):
        signals[i] = check_outcome(signal, weather_client)

    report = generate_report(signals)
    print(report)

    save_outcomes(signals)

    return 0


if __name__ == "__main__":
    sys.exit(main())
