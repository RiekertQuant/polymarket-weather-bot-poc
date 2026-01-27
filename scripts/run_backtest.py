#!/usr/bin/env python3
"""Run backtest on historical Polymarket weather markets.

Usage:
    python scripts/run_backtest.py [--collect] [--max-markets N] [--no-cache]

Options:
    --collect       Force fresh data collection from Polymarket API
    --max-markets   Maximum markets to collect (default: 50)
    --no-cache      Don't use cached weather data
"""

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtest.data_collector import PolymarketHistoricalCollector, HistoricalMarket
from src.backtest.weather_history import WeatherHistoryCollector
from src.backtest.engine import BacktestEngine, BacktestConfig
from src.backtest.report import BacktestReport, print_quick_summary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


def collect_market_data(
    collector: PolymarketHistoricalCollector,
    max_markets: int = 50,
    force_refresh: bool = False,
) -> list[HistoricalMarket]:
    """Collect or load market data.

    Args:
        collector: Market data collector.
        max_markets: Maximum markets to collect.
        force_refresh: Force fresh collection.

    Returns:
        List of historical markets.
    """
    cache_file = "markets.json"

    # Try to load from cache first
    if not force_refresh:
        markets = collector.load_from_cache(cache_file)
        if markets:
            logger.info(f"Loaded {len(markets)} markets from cache")
            return markets

    # Collect fresh data
    logger.info("Collecting market data from Polymarket...")
    markets = collector.collect_all(max_markets=max_markets)

    if markets:
        collector.save_to_cache(markets, cache_file)

    return markets


def collect_weather_data(
    weather_collector: WeatherHistoryCollector,
    markets: list[HistoricalMarket],
    use_cache: bool = True,
) -> None:
    """Collect weather data for markets.

    Args:
        weather_collector: Weather data collector.
        markets: Markets to get weather for.
        use_cache: Whether to use cached data.
    """
    if use_cache:
        weather_collector.load_cache()

    # Collect weather for each market
    cities_dates = set()
    for market in markets:
        if market.city and market.target_date:
            cities_dates.add((market.city, market.target_date))

    logger.info(f"Collecting weather data for {len(cities_dates)} city/date combinations...")

    for city, target_date in cities_dates:
        # Get forecast (what was predicted day before)
        weather_collector.get_forecast_at_decision_time(city, target_date, days_before=1)

        # Get actual
        weather_collector.get_actual_weather(city, target_date)

    # Save cache
    weather_collector.save_cache()


def create_synthetic_markets() -> list[HistoricalMarket]:
    """Create synthetic markets for testing when API data unavailable.

    This generates realistic test data based on historical weather patterns.

    Returns:
        List of synthetic historical markets.
    """
    from src.backtest.data_collector import HistoricalMarket, PricePoint
    from datetime import datetime
    import random

    logger.info("Creating synthetic markets for testing...")

    markets = []
    cities = ["New York City", "London", "Seoul"]

    # Generate markets for past 30 days
    base_date = date.today() - timedelta(days=60)

    # Typical temperature ranges by city (winter)
    temp_ranges = {
        "New York City": (-5, 10),
        "London": (2, 12),
        "Seoul": (-10, 5),
    }

    for day_offset in range(30):
        target_date = base_date + timedelta(days=day_offset)

        for city in cities:
            temp_min, temp_max = temp_ranges[city]
            mid_temp = (temp_min + temp_max) / 2

            # Create a few markets per city/day with different thresholds
            thresholds = [
                mid_temp - 5,  # Easy YES (likely to hit)
                mid_temp,  # 50/50
                mid_temp + 5,  # Hard YES (unlikely)
            ]

            for threshold in thresholds:
                # Simulate realistic pricing
                # Markets priced based on naive probability
                actual_temp = random.gauss(mid_temp, 3)  # Simulated actual
                outcome = actual_temp >= threshold

                # Price should roughly reflect probability but with noise
                base_prob = 0.5 + (mid_temp - threshold) / 20
                base_prob = max(0.05, min(0.95, base_prob))
                price = base_prob + random.gauss(0, 0.1)
                price = max(0.02, min(0.98, price))

                # Create price history (simplified)
                price_history = []
                for hour in range(24):
                    ts = datetime.combine(target_date - timedelta(days=1), datetime.min.time())
                    ts = ts.replace(hour=hour)
                    # Add some price movement
                    hour_price = price + random.gauss(0, 0.02)
                    hour_price = max(0.01, min(0.99, hour_price))
                    price_history.append(PricePoint(timestamp=ts, price=hour_price))

                market = HistoricalMarket(
                    condition_id=f"synthetic-{city}-{target_date}-{threshold}",
                    token_id=f"token-{city}-{target_date}-{threshold}",
                    question=f"Will {city} hit {threshold:.0f}C on {target_date}?",
                    description=f"Synthetic market for backtesting",
                    end_date=target_date,
                    outcome=outcome,
                    city=city,
                    threshold_celsius=threshold,
                    target_date=target_date,
                    comparison=">=",
                    price_history=price_history,
                    volume=random.uniform(100, 5000),
                )

                markets.append(market)

    logger.info(f"Created {len(markets)} synthetic markets")
    return markets


def run_backtest(
    markets: list[HistoricalMarket],
    weather_collector: WeatherHistoryCollector,
    config: BacktestConfig,
) -> BacktestReport:
    """Run the backtest.

    Args:
        markets: Historical markets.
        weather_collector: Weather data collector.
        config: Backtest configuration.

    Returns:
        Backtest report.
    """
    engine = BacktestEngine(config=config, weather_collector=weather_collector)

    # Determine date range from markets
    dates = [m.target_date for m in markets if m.target_date]
    if not dates:
        logger.error("No valid markets with target dates")
        sys.exit(1)

    start_date = min(dates) - timedelta(days=7)
    end_date = max(dates)

    logger.info(f"Running backtest: {start_date} to {end_date}")
    logger.info(f"Markets: {len(markets)}")

    result = engine.run(markets, start_date=start_date, end_date=end_date)

    return BacktestReport(result)


def main():
    parser = argparse.ArgumentParser(description="Run backtest on Polymarket weather markets")
    parser.add_argument("--collect", action="store_true", help="Force fresh data collection")
    parser.add_argument("--max-markets", type=int, default=50, help="Max markets to collect")
    parser.add_argument("--no-cache", action="store_true", help="Don't use weather cache")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic test data")
    parser.add_argument("--save", action="store_true", help="Save report to file")
    parser.add_argument(
        "--initial-balance", type=float, default=1000.0, help="Initial balance"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("POLYMARKET WEATHER BOT - BACKTESTER")
    print("=" * 70)
    print()

    # Initialize collectors
    market_collector = PolymarketHistoricalCollector()
    weather_collector = WeatherHistoryCollector()

    # Phase 1: Get market data
    if args.synthetic:
        markets = create_synthetic_markets()
    else:
        markets = collect_market_data(
            market_collector,
            max_markets=args.max_markets,
            force_refresh=args.collect,
        )

    if not markets:
        logger.warning("No markets collected from API. Using synthetic data for demo...")
        markets = create_synthetic_markets()

    # Phase 2: Get weather data
    collect_weather_data(
        weather_collector,
        markets,
        use_cache=not args.no_cache,
    )

    # Phase 3: Run backtest
    config = BacktestConfig(
        initial_balance=args.initial_balance,
        # Use default strategy settings
    )

    report = run_backtest(markets, weather_collector, config)

    # Print results
    print()
    print(report.full_report())

    # Quick summary
    print()
    print_quick_summary(report.result)

    # Save if requested
    if args.save:
        report_path = report.save_report()
        print(f"\nReport saved to: {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
