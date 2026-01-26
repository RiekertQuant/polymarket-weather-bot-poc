#!/usr/bin/env python3
"""Demo script showing the bot making trades with mock data.

This script demonstrates the full trading flow with controlled
mock weather data to show trades being made.
"""

import logging
import sys
from datetime import date, timedelta
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.execution.paper_broker import PaperBroker
from src.polymarket.mock_client import MockPolymarketClient
from src.polymarket.client_base import Market
from src.strategy.decision import DecisionEngine
from src.strategy.filters import MarketFilter
from src.strategy.sizing import PositionSizer
from src.weather.open_meteo import MockOpenMeteoClient, DailyForecast
from src.weather.probability import WeatherProbabilityEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)


def create_demo_markets() -> list[Market]:
    """Create demo markets with known characteristics."""
    tomorrow = date.today() + timedelta(days=1)

    return [
        # Good trade: cheap price, high probability (forecast 10°C, threshold 5°C)
        Market(
            id="demo-london-5c",
            title="Will London hit 5°C tomorrow?",
            description="Demo market - high probability event priced cheap",
            end_date=tomorrow,
            yes_price=0.05,  # Cheap!
            volume=1000.0,
            active=True,
            city="London",
            threshold_celsius=5.0,
            target_date=tomorrow,
        ),
        # Good trade: cheap price, high probability (forecast 5°C, threshold 0°C)
        Market(
            id="demo-nyc-0c",
            title="Will NYC hit 0°C tomorrow?",
            description="Demo market - high probability event priced cheap",
            end_date=tomorrow,
            yes_price=0.04,  # Very cheap!
            volume=1500.0,
            active=True,
            city="New York City",
            threshold_celsius=0.0,
            target_date=tomorrow,
        ),
        # Good trade: cheap price, high probability (forecast 2°C, threshold -5°C)
        Market(
            id="demo-seoul-minus5c",
            title="Will Seoul hit -5°C tomorrow?",
            description="Demo market - high probability event",
            end_date=tomorrow,
            yes_price=0.03,  # Cheap!
            volume=800.0,
            active=True,
            city="Seoul",
            threshold_celsius=-5.0,
            target_date=tomorrow,
        ),
        # Skip: 50/50 price
        Market(
            id="demo-5050",
            title="Will London hit 10°C tomorrow?",
            description="Should be skipped - 50/50 price",
            end_date=tomorrow,
            yes_price=0.50,
            volume=2000.0,
            active=True,
            city="London",
            threshold_celsius=10.0,
            target_date=tomorrow,
        ),
        # Skip: expensive price
        Market(
            id="demo-expensive",
            title="Will NYC hit -20°C tomorrow?",
            description="Should be skipped - expensive",
            end_date=tomorrow,
            yes_price=0.85,
            volume=3000.0,
            active=True,
            city="New York City",
            threshold_celsius=-20.0,
            target_date=tomorrow,
        ),
        # Skip: low probability (no edge)
        Market(
            id="demo-no-edge",
            title="Will London hit 20°C tomorrow?",
            description="Low probability - no edge despite cheap price",
            end_date=tomorrow,
            yes_price=0.05,
            volume=500.0,
            active=True,
            city="London",
            threshold_celsius=20.0,
            target_date=tomorrow,
        ),
    ]


def create_mock_weather() -> MockOpenMeteoClient:
    """Create mock weather client with known forecasts."""
    tomorrow = date.today() + timedelta(days=1)

    mock_forecasts = {
        "London": [
            DailyForecast(
                date=tomorrow,
                temperature_max=10.0,  # Will hit 5°C easily (P > 99%)
                temperature_min=4.0,
                temperature_mean=7.0,
            )
        ],
        "New York City": [
            DailyForecast(
                date=tomorrow,
                temperature_max=5.0,  # Will hit 0°C easily (P > 99%)
                temperature_min=-2.0,
                temperature_mean=1.5,
            )
        ],
        "Seoul": [
            DailyForecast(
                date=tomorrow,
                temperature_max=2.0,  # Will hit -5°C easily (P > 99%)
                temperature_min=-3.0,
                temperature_mean=-0.5,
            )
        ],
    }

    return MockOpenMeteoClient(mock_forecasts)


def main():
    """Run demo trading cycle."""
    logger.info("=" * 70)
    logger.info("POLYMARKET WEATHER BOT - DEMO RUN (Paper Trading)")
    logger.info("=" * 70)
    logger.info("")
    logger.info("This demo uses mock weather data to show trades being made.")
    logger.info("")

    # Create components with mock data
    mock_weather = create_mock_weather()
    prob_engine = WeatherProbabilityEngine(
        weather_client=mock_weather,
        default_sigma=2.0,
    )

    market_filter = MarketFilter()
    position_sizer = PositionSizer()
    decision_engine = DecisionEngine(market_filter, position_sizer)

    broker = PaperBroker(initial_balance=30.0)

    logger.info(f"Initial balance: ${broker.get_balance():.2f}")
    logger.info("-" * 70)

    # Get demo markets
    markets = create_demo_markets()
    logger.info(f"Processing {len(markets)} demo markets...")
    logger.info("")

    # Calculate probabilities
    prob_results = {}
    for market in markets:
        prob = prob_engine.calculate_probability(
            city=market.city,
            threshold=market.threshold_celsius,
            target_date=market.target_date,
        )
        if prob:
            prob_results[market.id] = prob
            logger.info(
                f"Market: {market.title}\n"
                f"  Forecast: {prob.forecast_temp:.1f}°C | Threshold: {prob.threshold:.1f}°C\n"
                f"  P(model): {prob.p_raw:.3f} | Price: {market.yes_price:.3f} | "
                f"Edge: {prob.p_raw - market.yes_price:+.3f}"
            )
            logger.info("")

    # Make trading decisions
    logger.info("-" * 70)
    logger.info("TRADING DECISIONS")
    logger.info("-" * 70)

    decisions = decision_engine.evaluate_markets(
        markets=markets,
        prob_results=prob_results,
    )

    for decision in decisions:
        status = "✓ TRADE" if decision.should_trade else "✗ SKIP"
        reason = ""
        if not decision.should_trade and decision.filter_result:
            reason = f" ({decision.filter_result.reason.value})"

        logger.info(f"{status}: {decision.market_title}{reason}")

    # Execute trades
    logger.info("")
    logger.info("-" * 70)
    logger.info("EXECUTING TRADES")
    logger.info("-" * 70)

    trades_made = 0
    for decision in decisions:
        if decision.should_trade and decision.bet_size:
            result = broker.buy_yes(
                market_id=decision.market_id,
                market_title=decision.market_title,
                city=decision.city,
                amount_usd=decision.bet_size.amount_usd,
                price=decision.market_price,
            )

            if result.success:
                trades_made += 1
                logger.info(
                    f"Bought {result.shares_filled:.1f} YES shares of '{decision.city}' "
                    f"@ ${decision.market_price:.3f} for ${decision.bet_size.amount_usd:.2f}"
                )
                logger.info(f"  Potential profit if YES: ${decision.bet_size.potential_profit:.2f}")

    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("DEMO SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Markets analyzed: {len(markets)}")
    logger.info(f"Trades made: {trades_made}")
    logger.info(f"Remaining balance: ${broker.get_balance():.2f}")
    logger.info("")

    logger.info("Open Positions:")
    for pos in broker.get_positions():
        logger.info(
            f"  {pos.city}: {pos.shares:.1f} shares @ ${pos.entry_price:.3f} "
            f"(cost: ${pos.cost_basis:.2f})"
        )

    # Simulate settlement
    logger.info("")
    logger.info("-" * 70)
    logger.info("SIMULATING SETTLEMENT (all markets resolve YES)")
    logger.info("-" * 70)

    for pos in broker.get_positions():
        broker.settle_position(pos.id, outcome=True)

    logger.info("")
    logger.info(f"Final balance: ${broker.get_balance():.2f}")
    logger.info(f"Total P&L: ${broker.get_total_pnl():+.2f}")
    logger.info("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
