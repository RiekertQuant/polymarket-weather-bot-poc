"""Main runner for the trading bot."""

import logging
import sys
from datetime import date
from pathlib import Path
from typing import Optional

from src.config import settings
from src.execution.paper_broker import PaperBroker
from src.ml.calibrator import IsotonicCalibrator
from src.polymarket.mock_client import MockPolymarketClient
from src.polymarket.real_client import RealPolymarketClient
from src.storage.db import Database, TradeRepository, PredictionRepository
from src.strategy.decision import DecisionEngine
from src.strategy.filters import MarketFilter
from src.strategy.sizing import PositionSizer
from src.weather.open_meteo import OpenMeteoClient, MockOpenMeteoClient, DailyForecast
from src.weather.probability import WeatherProbabilityEngine, ProbabilityResult

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class TradingBot:
    """Main trading bot orchestrator."""

    def __init__(
        self,
        use_mock_weather: bool = False,
        mock_forecasts: Optional[dict[str, list[DailyForecast]]] = None,
    ):
        """Initialize trading bot.

        Args:
            use_mock_weather: If True, use mock weather client.
            mock_forecasts: Mock forecast data for testing.
        """
        logger.info("=" * 60)
        logger.info("POLYMARKET WEATHER BOT - PAPER TRADING MODE")
        logger.info("=" * 60)

        # Ensure we're in paper mode
        if settings.trading_mode != "paper":
            logger.error("SAFETY CHECK: Only paper trading is supported!")
            raise ValueError("Live trading is not implemented. Use paper mode.")

        # Initialize database
        self.db = Database(settings.db_path)
        self.db.initialize()
        self.trade_repo = TradeRepository(self.db)
        self.prediction_repo = PredictionRepository(self.db)

        # Initialize Polymarket client
        if settings.polymarket_client == "mock":
            logger.info("Using MockPolymarketClient (fixtures-based)")
            self.polymarket = MockPolymarketClient()
        else:
            logger.info("Using RealPolymarketClient (best-effort)")
            self.polymarket = RealPolymarketClient(settings.polymarket_api_url)

        # Initialize weather client
        if use_mock_weather:
            logger.info("Using MockOpenMeteoClient")
            self.weather_client = MockOpenMeteoClient(mock_forecasts)
        else:
            logger.info("Using OpenMeteoClient (live forecasts)")
            self.weather_client = OpenMeteoClient()

        # Load calibrator if available
        calibrator = self._load_calibrator()

        # Initialize probability engine
        self.prob_engine = WeatherProbabilityEngine(
            weather_client=self.weather_client,
            default_sigma=settings.forecast_sigma,
            calibrator=calibrator,
        )

        # Initialize strategy components
        self.market_filter = MarketFilter(settings)
        self.position_sizer = PositionSizer(settings)
        self.decision_engine = DecisionEngine(self.market_filter, self.position_sizer)

        # Initialize broker (paper trading)
        self.broker = PaperBroker(initial_balance=30.0)

        logger.info(f"Initial balance: ${self.broker.get_balance():.2f}")
        logger.info(f"Cities tracked: {settings.cities}")

    def _load_calibrator(self) -> Optional[IsotonicCalibrator]:
        """Load calibrator model if exists."""
        model_path = Path("models/calibrator.pkl")
        if model_path.exists():
            try:
                calibrator = IsotonicCalibrator.load(model_path)
                logger.info("Loaded calibration model")
                return calibrator
            except Exception as e:
                logger.warning(f"Could not load calibrator: {e}")
        return None

    def run(self) -> dict:
        """Run single trading cycle.

        Returns:
            Summary of run results.
        """
        logger.info("-" * 60)
        logger.info("Starting trading run...")

        # Track run stats
        stats = {
            "markets_scanned": 0,
            "markets_filtered": 0,
            "trades_made": 0,
            "total_risk": 0.0,
            "decisions": [],
        }

        # Get current risk levels
        city_risk = {}
        for city in settings.cities:
            city_risk[city] = self.trade_repo.get_total_city_risk(city)

        # Scan markets
        logger.info("Scanning for weather markets...")
        markets = self.polymarket.get_weather_markets()
        stats["markets_scanned"] = len(markets)
        logger.info(f"Found {len(markets)} weather markets")

        if not markets:
            logger.warning("No markets found. Check fixtures or API connection.")
            return stats

        # Calculate probabilities for each market
        prob_results: dict[str, ProbabilityResult] = {}
        for market in markets:
            if market.city and market.threshold_celsius and market.target_date:
                prob = self.prob_engine.calculate_probability(
                    city=market.city,
                    threshold=market.threshold_celsius,
                    target_date=market.target_date,
                )
                if prob:
                    prob_results[market.id] = prob
                    # Store prediction
                    self.prediction_repo.save_prediction(
                        market_id=market.id,
                        city=market.city,
                        target_date=market.target_date.isoformat(),
                        prob_result=prob,
                        market_price=market.yes_price,
                    )

        logger.info(f"Calculated probabilities for {len(prob_results)} markets")

        # Evaluate markets and make decisions
        decisions = self.decision_engine.evaluate_markets(
            markets=markets,
            prob_results=prob_results,
            city_risk=city_risk,
            daily_risk=sum(city_risk.values()),
            trades_today=len(self.broker.get_open_positions()),
        )

        # Log decisions
        for decision in decisions:
            logger.info(
                f"Market: {decision.market_title[:50]}... | "
                f"City: {decision.city} | "
                f"Price: {decision.market_price:.3f} | "
                f"P(model): {decision.p_model:.3f} | "
                f"Edge: {decision.edge:+.3f} | "
                f"Trade: {decision.should_trade}"
            )

            if decision.filter_result and not decision.filter_result.passed:
                logger.debug(f"  Filtered: {decision.filter_result.reason.value}")

        # Execute trades
        tradeable = [d for d in decisions if d.should_trade]
        stats["markets_filtered"] = len(markets) - len(tradeable)

        for decision in tradeable:
            if decision.bet_size is None:
                continue

            logger.info(
                f"EXECUTING TRADE: {decision.side} on {decision.city} @ "
                f"${decision.market_price:.3f} for ${decision.bet_size.amount_usd:.2f}"
            )

            result = self.broker.buy_yes(
                market_id=decision.market_id,
                market_title=decision.market_title,
                city=decision.city,
                amount_usd=decision.bet_size.amount_usd,
                price=decision.market_price,
            )

            if result.success:
                stats["trades_made"] += 1
                stats["total_risk"] += decision.bet_size.amount_usd
                logger.info(
                    f"Trade successful: {result.shares_filled:.2f} shares | "
                    f"Position ID: {result.position_id}"
                )

                # Save position to database
                position = self.broker.get_position(result.position_id)
                if position:
                    self.trade_repo.save_position(position)
            else:
                logger.error(f"Trade failed: {result.error_message}")

        stats["decisions"] = decisions

        # Summary
        logger.info("-" * 60)
        logger.info("RUN SUMMARY")
        logger.info(f"  Markets scanned: {stats['markets_scanned']}")
        logger.info(f"  Markets filtered: {stats['markets_filtered']}")
        logger.info(f"  Trades made: {stats['trades_made']}")
        logger.info(f"  Total risk: ${stats['total_risk']:.2f}")
        logger.info(f"  Balance: ${self.broker.get_balance():.2f}")
        logger.info(f"  Open positions: {len(self.broker.get_open_positions())}")
        logger.info("-" * 60)

        return stats

    def show_positions(self) -> None:
        """Display current positions."""
        positions = self.broker.get_positions()
        if not positions:
            logger.info("No positions")
            return

        logger.info("CURRENT POSITIONS:")
        for pos in positions:
            logger.info(
                f"  {pos.id} | {pos.city} | {pos.side} | "
                f"{pos.shares:.2f} shares @ ${pos.entry_price:.3f} | "
                f"Status: {pos.status.value}"
            )

    def cleanup(self) -> None:
        """Cleanup resources."""
        self.db.close()


def main():
    """Main entry point."""
    logger.info("Starting Polymarket Weather Bot (Paper Trading)")

    try:
        bot = TradingBot()
        stats = bot.run()
        bot.show_positions()
        bot.cleanup()

        logger.info("Bot run completed successfully")
        return 0

    except Exception as e:
        logger.exception(f"Bot run failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
