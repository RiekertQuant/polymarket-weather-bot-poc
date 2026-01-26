"""Trading decision logic."""

from dataclasses import dataclass
from datetime import date
from typing import Optional

from src.polymarket.client_base import Market
from src.strategy.filters import MarketFilter, FilterResult
from src.strategy.sizing import PositionSizer, BetSize
from src.weather.probability import ProbabilityResult


@dataclass
class TradingDecision:
    """A trading decision for a market."""

    market_id: str
    market_title: str
    city: str
    threshold_celsius: float
    target_date: date

    # Prices
    market_price: float  # Current YES price
    p_model: float  # Model probability
    edge: float  # p_model - market_price

    # Decision
    should_trade: bool
    side: str  # "YES" or "NO" or "NONE"
    bet_size: Optional[BetSize] = None

    # Filter info
    filter_result: Optional[FilterResult] = None

    # Probability details
    probability_result: Optional[ProbabilityResult] = None


class DecisionEngine:
    """Engine for making trading decisions."""

    def __init__(
        self,
        market_filter: Optional[MarketFilter] = None,
        position_sizer: Optional[PositionSizer] = None,
    ):
        """Initialize decision engine.

        Args:
            market_filter: Market filter for screening.
            position_sizer: Position sizer for bet sizing.
        """
        self.market_filter = market_filter or MarketFilter()
        self.position_sizer = position_sizer or PositionSizer()

    def evaluate_market(
        self,
        market: Market,
        prob_result: ProbabilityResult,
        current_city_risk: float = 0.0,
        current_daily_risk: float = 0.0,
        trades_today: int = 0,
    ) -> TradingDecision:
        """Evaluate a market and make trading decision.

        Args:
            market: Market to evaluate.
            prob_result: Probability calculation result.
            current_city_risk: Current risk for this city.
            current_daily_risk: Current daily risk total.
            trades_today: Number of trades already made today.

        Returns:
            TradingDecision with all details.
        """
        price = market.yes_price
        p_model = prob_result.p_calibrated

        # Apply filters
        filter_result = self.market_filter.apply_all_filters(
            price=price,
            p_model=p_model,
            city=market.city,
            active=market.active,
        )

        # Base decision
        decision = TradingDecision(
            market_id=market.id,
            market_title=market.title,
            city=market.city or "Unknown",
            threshold_celsius=market.threshold_celsius or 0.0,
            target_date=market.target_date or date.today(),
            market_price=price,
            p_model=p_model,
            edge=p_model - price,
            should_trade=False,
            side="NONE",
            filter_result=filter_result,
            probability_result=prob_result,
        )

        if not filter_result.passed:
            return decision

        # Calculate position size
        bet_size = self.position_sizer.calculate_bet_size(
            edge=decision.edge,
            price=price,
            current_city_risk=current_city_risk,
            current_daily_risk=current_daily_risk,
            trades_today=trades_today,
        )

        if bet_size is None or bet_size.amount_usd <= 0:
            return decision

        # We have a valid trade
        decision.should_trade = True
        decision.side = "YES"  # We're buying YES at cheap prices
        decision.bet_size = bet_size

        return decision

    def evaluate_markets(
        self,
        markets: list[Market],
        prob_results: dict[str, ProbabilityResult],
        city_risk: Optional[dict[str, float]] = None,
        daily_risk: float = 0.0,
        trades_today: int = 0,
    ) -> list[TradingDecision]:
        """Evaluate multiple markets.

        Args:
            markets: List of markets to evaluate.
            prob_results: Probability results keyed by market ID.
            city_risk: Current risk per city.
            daily_risk: Current daily risk total.
            trades_today: Number of trades already made today.

        Returns:
            List of TradingDecisions, sorted by edge descending.
        """
        city_risk = city_risk or {}
        decisions = []

        for market in markets:
            prob_result = prob_results.get(market.id)
            if prob_result is None:
                continue

            city = market.city or "Unknown"
            decision = self.evaluate_market(
                market=market,
                prob_result=prob_result,
                current_city_risk=city_risk.get(city, 0.0),
                current_daily_risk=daily_risk,
                trades_today=trades_today,
            )
            decisions.append(decision)

            # Update running totals if trade will be made
            if decision.should_trade and decision.bet_size:
                city_risk[city] = city_risk.get(city, 0.0) + decision.bet_size.amount_usd
                daily_risk += decision.bet_size.amount_usd
                trades_today += 1

        # Sort by edge (best opportunities first)
        decisions.sort(key=lambda d: d.edge, reverse=True)
        return decisions
