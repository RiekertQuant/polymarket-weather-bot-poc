#!/usr/bin/env python3
"""Daily signal scanner - logs what trades the bot WOULD make.

This script is designed to run on GitHub Actions daily to track
the bot's theoretical performance against real Polymarket data.

Outputs signals to data/signals/ directory as JSON files.
"""

import json
import logging
import os
import sys
from datetime import datetime, date
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Settings
from src.polymarket.mock_client import MockPolymarketClient
from src.polymarket.real_client import RealPolymarketClient
from src.weather.open_meteo import OpenMeteoClient
from src.weather.nws_forecast import NWSForecastClient
from src.weather.probability import WeatherProbabilityEngine
from src.strategy.filters import MarketFilter
from src.strategy.decision import DecisionEngine
from src.strategy.sizing import PositionSizer
from src.notifications.discord import DiscordNotifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def scan_markets() -> dict:
    """Scan markets and generate signals.

    Returns:
        Dictionary with scan results and signals.
    """
    settings = Settings()

    # Try real client first, fall back to mock
    try:
        logger.info("Attempting to fetch real Polymarket data...")
        client = RealPolymarketClient()
        markets = client.get_weather_markets()
        data_source = "polymarket_api"
    except Exception as e:
        logger.warning(f"Real client failed: {e}, using mock data")
        client = MockPolymarketClient()
        markets = client.get_weather_markets()
        data_source = "mock_fixtures"

    logger.info(f"Found {len(markets)} markets from {data_source}")

    # Initialize weather client based on settings
    if settings.weather_source == "nws":
        weather_client = NWSForecastClient()
        weather_source = "nws"
        logger.info("Using NWS forecast API (matches Polymarket resolution source)")
    else:
        weather_client = OpenMeteoClient()
        weather_source = "open_meteo"
        logger.info("Using Open-Meteo forecast API")

    prob_engine = WeatherProbabilityEngine(
        weather_client=weather_client,
        default_sigma=settings.forecast_sigma,
    )
    decision_engine = DecisionEngine(
        market_filter=MarketFilter(settings),
        position_sizer=PositionSizer(settings),
    )

    # Process markets
    signals = []
    all_markets = []

    for market in markets:
        if not market.city or not market.target_date:
            continue

        # Range markets use threshold_celsius_upper; threshold/boundary markets use threshold_celsius
        has_bounds = (
            market.threshold_celsius is not None
            or market.threshold_celsius_upper is not None
        )
        if not has_bounds:
            continue

        # Calculate probability using the market's comparison type
        prob_result = prob_engine.calculate_probability(
            city=market.city,
            threshold=market.threshold_celsius if market.threshold_celsius is not None else 0.0,
            target_date=market.target_date,
            comparison=market.comparison,
            threshold_upper=market.threshold_celsius_upper,
        )

        if prob_result is None:
            continue

        # Make decision
        decision = decision_engine.evaluate_market(
            market=market,
            prob_result=prob_result,
        )

        market_data = {
            "market_id": market.id,
            "title": market.title,
            "city": market.city,
            "threshold_celsius": market.threshold_celsius,
            "threshold_celsius_upper": market.threshold_celsius_upper,
            "comparison": market.comparison,
            "target_date": market.target_date.isoformat(),
            "end_date": market.end_date.isoformat(),
            "market_price": market.yes_price,
            "p_model": prob_result.p_calibrated,
            "forecast_temp": prob_result.forecast_temp,
            "edge": decision.edge,
            "should_trade": decision.should_trade,
            "filter_reason": decision.filter_result.reason.value if decision.filter_result else None,
        }

        all_markets.append(market_data)

        if decision.should_trade:
            signal = {
                **market_data,
                "signal": "BUY_YES",
                "recommended_size_usd": decision.bet_size.amount_usd if decision.bet_size else 0,
                "potential_profit": decision.bet_size.potential_profit if decision.bet_size else 0,
            }
            signals.append(signal)
            logger.info(
                f"SIGNAL: {market.title} @ ${market.yes_price:.3f} | "
                f"p={prob_result.p_calibrated:.2f} | edge={decision.edge:+.2f}"
            )

    return {
        "scan_time": datetime.utcnow().isoformat() + "Z",
        "scan_date": date.today().isoformat(),
        "data_source": data_source,
        "weather_source": weather_source,
        "markets_scanned": len(all_markets),
        "signals_generated": len(signals),
        "signals": signals,
        "all_markets": all_markets,
        "settings": {
            "price_min": settings.price_min,
            "price_max": settings.price_max,
            "min_edge_absolute": settings.min_edge_absolute,
            "min_edge_relative": settings.min_edge_relative,
            "forecast_sigma": settings.forecast_sigma,
            "weather_source": weather_source,
        },
    }


def save_signal(result: dict, output_dir: Path = Path("data/signals")) -> Path:
    """Save signal to file.

    Args:
        result: Scan result dictionary.
        output_dir: Output directory.

    Returns:
        Path to saved file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Daily file
    date_str = date.today().isoformat()
    filepath = output_dir / f"signals_{date_str}.json"

    with open(filepath, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"Saved signals to {filepath}")

    # Also update latest.json
    latest_path = output_dir / "latest.json"
    with open(latest_path, "w") as f:
        json.dump(result, f, indent=2)

    # Update history summary
    update_history(result, output_dir)

    return filepath


def update_history(result: dict, output_dir: Path) -> None:
    """Update running history of signals.

    Args:
        result: Today's scan result.
        output_dir: Output directory.
    """
    history_path = output_dir / "history.json"

    # Load existing history
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
    else:
        history = {"scans": [], "total_signals": 0, "signals_by_city": {}}

    # Add today's summary
    scan_summary = {
        "date": result["scan_date"],
        "scan_time": result["scan_time"],
        "data_source": result["data_source"],
        "markets_scanned": result["markets_scanned"],
        "signals_generated": result["signals_generated"],
        "signals": result["signals"],
    }

    # Remove duplicate for today if re-running
    history["scans"] = [s for s in history["scans"] if s["date"] != result["scan_date"]]
    history["scans"].append(scan_summary)
    history["scans"].sort(key=lambda x: x["date"], reverse=True)

    # Keep last 90 days
    history["scans"] = history["scans"][:90]

    # Update totals
    history["total_signals"] = sum(s["signals_generated"] for s in history["scans"])

    # Signals by city
    city_counts = {}
    for scan in history["scans"]:
        for signal in scan.get("signals", []):
            city = signal.get("city", "Unknown")
            city_counts[city] = city_counts.get(city, 0) + 1
    history["signals_by_city"] = city_counts

    # Save
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)


def print_summary(result: dict) -> None:
    """Print summary to console."""
    print("\n" + "=" * 60)
    print("DAILY SIGNAL SCAN SUMMARY")
    print("=" * 60)
    print(f"Date: {result['scan_date']}")
    print(f"Data Source: {result['data_source']}")
    print(f"Weather Source: {result.get('weather_source', 'open_meteo')}")
    print(f"Markets Scanned: {result['markets_scanned']}")
    print(f"Signals Generated: {result['signals_generated']}")

    if result["signals"]:
        print("\n" + "-" * 60)
        print("TRADE SIGNALS")
        print("-" * 60)
        for sig in result["signals"]:
            print(f"  {sig['title']}")
            print(f"    Market Price: ${sig['market_price']:.3f}")
            print(f"    Model Prob:   {sig['p_model']:.1%}")
            print(f"    Edge:         {sig['edge']:+.1%}")
            print(f"    Rec. Size:    ${sig['recommended_size_usd']:.2f}")
            print()
    else:
        print("\nNo trade signals today.")

    print("=" * 60)


def _load_outcomes_summary() -> dict | None:
    """Load the outcomes summary from outcomes.json if it exists."""
    outcomes_path = Path("data/signals/outcomes.json")
    if not outcomes_path.exists():
        return None
    try:
        with open(outcomes_path) as f:
            data = json.load(f)
        return data.get("summary")
    except (json.JSONDecodeError, OSError):
        return None


def main():
    logger.info("Starting signal scan...")

    settings = Settings()
    result = scan_markets()
    filepath = save_signal(result)
    print_summary(result)

    # Discord notification
    if settings.discord_webhook_url and result["signals_generated"] > 0:
        notifier = DiscordNotifier(settings.discord_webhook_url)
        outcomes_summary = _load_outcomes_summary()
        notifier.send_signal_alert(result, outcomes_summary=outcomes_summary)
        logger.info("Discord alert dispatched")
    else:
        logger.info("Discord alert skipped (no URL or no signals)")

    # Output for GitHub Actions (using environment file)
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"signals_count={result['signals_generated']}\n")
            f.write(f"markets_count={result['markets_scanned']}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
