#!/usr/bin/env python3
"""Compare backtest outcomes using ERA5 vs NWS KLGA actual temperatures.

Runs the same backtest with two different weather "actual" sources and
compares outcomes side-by-side to quantify the impact of the ERA5 vs KLGA
data mismatch.

Usage:
    python scripts/compare_weather_sources.py [--no-cache] [--synthetic]
"""

import argparse
import logging
import math
import sys
from datetime import date, timedelta
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtest.data_collector import PolymarketHistoricalCollector, HistoricalMarket
from src.backtest.weather_history import WeatherHistoryCollector, ActualWeather
from src.backtest.nws_weather import NWSWeatherCollector
from src.backtest.engine import BacktestEngine, BacktestConfig, BacktestResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def c_to_f(temp_c: float) -> float:
    """Convert Celsius to Fahrenheit."""
    return temp_c * 9 / 5 + 32


def load_markets(use_synthetic: bool = False) -> list[HistoricalMarket]:
    """Load market data from cache or generate synthetic."""
    collector = PolymarketHistoricalCollector()

    if not use_synthetic:
        markets = collector.load_from_cache("markets.json")
        if markets:
            logger.info(f"Loaded {len(markets)} markets from cache")
            return markets

    # Fall back to synthetic
    logger.info("Generating synthetic markets...")
    from scripts.run_backtest import create_synthetic_markets
    return create_synthetic_markets()


def collect_weather_for_markets(
    collector,
    markets: list[HistoricalMarket],
    use_cache: bool = True,
) -> None:
    """Pre-collect weather data for all markets."""
    if use_cache:
        collector.load_cache()

    cities_dates = set()
    for market in markets:
        if market.city and market.target_date:
            cities_dates.add((market.city, market.target_date))

    logger.info(f"Collecting weather for {len(cities_dates)} city/date pairs...")

    for city, target_date in cities_dates:
        collector.get_forecast_at_decision_time(city, target_date, days_before=1)
        collector.get_actual_weather(city, target_date)

    collector.save_cache()


def run_single_backtest(
    markets: list[HistoricalMarket],
    weather_collector,
    config: BacktestConfig,
) -> BacktestResult:
    """Run backtest with a specific weather collector."""
    engine = BacktestEngine(config=config, weather_collector=weather_collector)

    dates = [m.target_date for m in markets if m.target_date]
    if not dates:
        logger.error("No valid markets with target dates")
        sys.exit(1)

    start_date = min(dates) - timedelta(days=7)
    end_date = max(dates)

    return engine.run(markets, start_date=start_date, end_date=end_date)


def print_temperature_comparison(
    era5_collector: WeatherHistoryCollector,
    nws_collector: NWSWeatherCollector,
    markets: list[HistoricalMarket],
) -> list[dict]:
    """Print temperature comparison table for NYC dates.

    Returns list of comparison records for summary stats.
    """
    # Collect unique NYC dates with thresholds
    nyc_entries: dict[date, list[float]] = {}
    for market in markets:
        if market.city == "New York City" and market.target_date and market.threshold_celsius:
            nyc_entries.setdefault(market.target_date, []).append(market.threshold_celsius)

    if not nyc_entries:
        print("No New York City markets found for temperature comparison.")
        return []

    print()
    print("=" * 80)
    print("TEMPERATURE COMPARISON: Open-Meteo ERA5 vs NWS KLGA")
    print("=" * 80)
    print()
    print(f"{'Date':<12} {'ERA5 (F)':>9} {'KLGA (F)':>9} {'Diff (F)':>9} {'Threshold Flips'}")
    print("-" * 80)

    comparisons = []

    for target_date in sorted(nyc_entries.keys()):
        era5_actual = era5_collector.get_actual_weather("New York City", target_date)
        nws_actual = nws_collector.get_actual_weather("New York City", target_date)

        if era5_actual is None or nws_actual is None:
            era5_str = f"{c_to_f(era5_actual.actual_max_temp):.0f}" if era5_actual else "N/A"
            nws_str = f"{c_to_f(nws_actual.actual_max_temp):.0f}" if nws_actual else "N/A"
            print(f"{target_date!s:<12} {era5_str:>9} {nws_str:>9} {'---':>9}")
            continue

        era5_f = c_to_f(era5_actual.actual_max_temp)
        nws_f = c_to_f(nws_actual.actual_max_temp)
        diff_f = nws_f - era5_f

        # Check threshold flips
        thresholds = nyc_entries[target_date]
        flips = []
        for threshold_c in thresholds:
            threshold_f = c_to_f(threshold_c)
            era5_yes = era5_actual.actual_max_temp >= threshold_c
            nws_yes = nws_actual.actual_max_temp >= threshold_c
            if era5_yes != nws_yes:
                flips.append(f"{threshold_f:.0f}F")

        flip_str = ", ".join(flips) if flips else ""

        print(
            f"{target_date!s:<12} {era5_f:>9.1f} {nws_f:>9.1f} {diff_f:>+9.1f} {flip_str}"
        )

        comparisons.append({
            "date": target_date,
            "era5_f": era5_f,
            "nws_f": nws_f,
            "diff_f": diff_f,
            "flips": len(flips),
        })

    print("-" * 80)
    return comparisons


def print_dual_backtest_results(
    era5_result: BacktestResult,
    nws_result: BacktestResult,
) -> None:
    """Print side-by-side backtest comparison."""
    print()
    print("=" * 80)
    print("DUAL BACKTEST: ERA5 vs NWS KLGA")
    print("=" * 80)
    print()

    # Side-by-side metrics
    print(f"{'Metric':<25} {'ERA5':>15} {'NWS KLGA':>15} {'Diff':>15}")
    print("-" * 70)

    rows = [
        ("Total Trades",
         f"{era5_result.total_trades}",
         f"{nws_result.total_trades}",
         f"{nws_result.total_trades - era5_result.total_trades:+d}"),
        ("Winning Trades",
         f"{era5_result.winning_trades}",
         f"{nws_result.winning_trades}",
         f"{nws_result.winning_trades - era5_result.winning_trades:+d}"),
        ("Losing Trades",
         f"{era5_result.losing_trades}",
         f"{nws_result.losing_trades}",
         f"{nws_result.losing_trades - era5_result.losing_trades:+d}"),
        ("Win Rate",
         f"{era5_result.win_rate * 100:.1f}%",
         f"{nws_result.win_rate * 100:.1f}%",
         f"{(nws_result.win_rate - era5_result.win_rate) * 100:+.1f}pp"),
        ("Total P&L",
         f"${era5_result.total_pnl:+.2f}",
         f"${nws_result.total_pnl:+.2f}",
         f"${nws_result.total_pnl - era5_result.total_pnl:+.2f}"),
        ("Return %",
         f"{era5_result.total_return_pct:+.2f}%",
         f"{nws_result.total_return_pct:+.2f}%",
         f"{nws_result.total_return_pct - era5_result.total_return_pct:+.2f}pp"),
        ("Avg Trade P&L",
         f"${era5_result.avg_trade_pnl:+.2f}",
         f"${nws_result.avg_trade_pnl:+.2f}",
         f"${nws_result.avg_trade_pnl - era5_result.avg_trade_pnl:+.2f}"),
        ("Max Drawdown",
         f"${era5_result.max_drawdown:.2f}",
         f"${nws_result.max_drawdown:.2f}",
         f"${nws_result.max_drawdown - era5_result.max_drawdown:+.2f}"),
        ("Sharpe Ratio",
         f"{era5_result.sharpe_ratio:.2f}" if era5_result.sharpe_ratio else "N/A",
         f"{nws_result.sharpe_ratio:.2f}" if nws_result.sharpe_ratio else "N/A",
         ""),
    ]

    for label, era5_val, nws_val, diff_val in rows:
        print(f"{label:<25} {era5_val:>15} {nws_val:>15} {diff_val:>15}")

    print()


def print_per_trade_diff(
    era5_result: BacktestResult,
    nws_result: BacktestResult,
) -> None:
    """Print per-trade outcome differences between ERA5 and NWS."""
    print()
    print("-" * 80)
    print("PER-TRADE OUTCOME DIFFERENCES")
    print("-" * 80)
    print()

    # Build lookup by market_id for each source
    era5_trades = {t.market_id: t for t in era5_result.trades if t.resolved}
    nws_trades = {t.market_id: t for t in nws_result.trades if t.resolved}

    # Find trades present in both
    common_ids = set(era5_trades.keys()) & set(nws_trades.keys())

    if not common_ids:
        print("No common resolved trades to compare.")
        return

    diff_count = 0
    agree_count = 0

    print(
        f"{'Market':<40} {'ERA5 Out':>9} {'NWS Out':>9} {'ERA5 PnL':>10} {'NWS PnL':>10} {'Match':>6}"
    )
    print("-" * 80)

    for market_id in sorted(common_ids):
        era5_t = era5_trades[market_id]
        nws_t = nws_trades[market_id]

        era5_outcome = "YES" if era5_t.outcome else "NO"
        nws_outcome = "YES" if nws_t.outcome else "NO"
        match = era5_t.outcome == nws_t.outcome

        if match:
            agree_count += 1
        else:
            diff_count += 1

        # Only print differing trades (or all if few)
        if not match or len(common_ids) <= 20:
            title = era5_t.market_title[:38]
            era5_pnl = f"${era5_t.pnl:+.2f}" if era5_t.pnl is not None else "N/A"
            nws_pnl = f"${nws_t.pnl:+.2f}" if nws_t.pnl is not None else "N/A"
            match_str = "YES" if match else "DIFF"
            print(
                f"{title:<40} {era5_outcome:>9} {nws_outcome:>9} "
                f"{era5_pnl:>10} {nws_pnl:>10} {match_str:>6}"
            )

    print("-" * 80)
    print(f"Common trades: {len(common_ids)} | Agree: {agree_count} | Disagree: {diff_count}")

    # Trades only in one source
    era5_only = set(era5_trades.keys()) - set(nws_trades.keys())
    nws_only = set(nws_trades.keys()) - set(era5_trades.keys())
    if era5_only:
        print(f"Trades only in ERA5: {len(era5_only)}")
    if nws_only:
        print(f"Trades only in NWS: {len(nws_only)}")
    print()


def print_summary_stats(comparisons: list[dict], era5_result: BacktestResult, nws_result: BacktestResult) -> None:
    """Print summary statistics."""
    print()
    print("=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print()

    if comparisons:
        diffs = [c["diff_f"] for c in comparisons]
        n = len(diffs)
        mean_diff = sum(diffs) / n
        variance = sum((d - mean_diff) ** 2 for d in diffs) / n if n > 1 else 0
        std_diff = math.sqrt(variance)
        total_flips = sum(c["flips"] for c in comparisons)

        print("Temperature Differences (NWS KLGA - ERA5):")
        print(f"  Dates compared:         {n}")
        print(f"  Mean difference:        {mean_diff:+.1f}°F")
        print(f"  Std deviation:          {std_diff:.1f}°F")
        print(f"  Min difference:         {min(diffs):+.1f}°F")
        print(f"  Max difference:         {max(diffs):+.1f}°F")
        print(f"  Threshold flips:        {total_flips}")
        print()

    # Outcome disagreement
    era5_trades = {t.market_id: t for t in era5_result.trades if t.resolved}
    nws_trades = {t.market_id: t for t in nws_result.trades if t.resolved}
    common = set(era5_trades.keys()) & set(nws_trades.keys())

    disagree = sum(1 for mid in common if era5_trades[mid].outcome != nws_trades[mid].outcome)

    print("Outcome Disagreement:")
    print(f"  Common resolved trades: {len(common)}")
    print(f"  Outcome disagreements:  {disagree}")
    if common:
        print(f"  Disagreement rate:      {disagree / len(common) * 100:.1f}%")

    # PnL impact of disagreements
    pnl_diff = nws_result.total_pnl - era5_result.total_pnl
    print()
    print("P&L Impact:")
    print(f"  ERA5 total P&L:         ${era5_result.total_pnl:+.2f}")
    print(f"  NWS total P&L:          ${nws_result.total_pnl:+.2f}")
    print(f"  Difference:             ${pnl_diff:+.2f}")

    print()
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Compare backtest outcomes: ERA5 vs NWS KLGA"
    )
    parser.add_argument("--no-cache", action="store_true", help="Don't use cached weather data")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic test data")

    args = parser.parse_args()
    use_cache = not args.no_cache

    print("=" * 80)
    print("WEATHER SOURCE COMPARISON: Open-Meteo ERA5 vs NWS KLGA")
    print("=" * 80)
    print()

    # 1. Load markets
    markets = load_markets(use_synthetic=args.synthetic)
    if not markets:
        print("No markets available.")
        return 1

    print(f"Loaded {len(markets)} markets")
    print()

    # 2. Collect weather data for both sources
    print("--- Collecting ERA5 weather data ---")
    era5_collector = WeatherHistoryCollector()
    collect_weather_for_markets(era5_collector, markets, use_cache=use_cache)

    print()
    print("--- Collecting NWS KLGA weather data ---")
    nws_collector = NWSWeatherCollector()
    collect_weather_for_markets(nws_collector, markets, use_cache=use_cache)

    # 3. Temperature comparison table
    comparisons = print_temperature_comparison(era5_collector, nws_collector, markets)

    # 4. Dual backtest
    config = BacktestConfig()

    print()
    print("--- Running ERA5 backtest ---")
    era5_result = run_single_backtest(markets, era5_collector, config)

    print()
    print("--- Running NWS backtest ---")
    nws_result = run_single_backtest(markets, nws_collector, config)

    # 5. Side-by-side results
    print_dual_backtest_results(era5_result, nws_result)

    # 6. Per-trade differences
    print_per_trade_diff(era5_result, nws_result)

    # 7. Summary statistics
    print_summary_stats(comparisons, era5_result, nws_result)

    return 0


if __name__ == "__main__":
    sys.exit(main())
